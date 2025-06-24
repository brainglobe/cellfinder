"""
Given an input folder containing a list of tiff, it loads it using dask
and does full 2d and 3d filtering, cell detection and cluster splitting.
For each filtering step it outputs the filtered image so you can see what
the step produces and tune parameters.

It outputs the following folders in the provided output directory (comment
out `save_tiffs` or filtering steps if you don't want wish to run / save them):

- input: The original input tiffs
- clipped: The first step of 2d filtering where the data is clipped to a max
      value (typically should be unchanged).
- enhanced: After it was run through the 2d filters.
- inside_brain: Tiled planes, with each tile indicating whether the tile is
      inside or outside the brain.
- filtered_2d: After the enhanced images was thresholded to a binary foreground
      / background.
- filtered_3d: After the 3d ball filtering.
- struct_id: The output from the cell detector, where each voxel has an ID with
      its cell number or zero if it's background.
- struct_type: Voxel values are 0 for background, 1 if it's a potential cell,
      2 if it's a structure to be split, and 3 if it's too big to be split.
- struct_type_split: Same as struct_type except we put a sphere with value 1
      centered on the structures that were split into cells.

There's also the following files:

- `structures.csv`. CSV that lists all the detected structures, their volumes,
  and type.
- `cells.xml`. All the detected cells before splitting structures.
- `structs_needs_split.xml`. All the detected structures that were too big
  to be cells that will be split.
- `structs_too_big.xml`. All the detected structures that were too big to even
  be split.
- `cells_split.xml`. All the additional cells that were generated during the
  split step.

To analyze, in Fiji open each directory as an image sequence (use virtual
option), check that each image sequence is 32-bit (the max of all of them)
or change the type to 32-bit. Then merge them as color channels with the
composite option and using the Image -> Color -> Channels tool switch them
from composite to grayscale.

This will load all the images into memory! So select only the directories
you wish to inspect. They should now be overlaid and you can inspect how
the algorithms processed cells.
"""

import csv
import dataclasses
import math
from pathlib import Path

import numpy as np
import tifffile
import torch
import tqdm
from brainglobe_utils.cells.cells import Cell
from brainglobe_utils.IO.cells import save_cells
from brainglobe_utils.IO.image.load import read_with_dask

from cellfinder.core.detect.filters.plane import TileProcessor
from cellfinder.core.detect.filters.setup_filters import DetectionSettings
from cellfinder.core.detect.filters.volume.ball_filter import BallFilter
from cellfinder.core.detect.filters.volume.structure_detection import (
    CellDetector,
    get_structure_centre,
)
from cellfinder.core.detect.filters.volume.structure_splitting import (
    split_cells,
)


def setup_filter(
    signal_path: Path,  # expect to load z, y, x
    batch_size: int = 1,
    torch_device="cpu",
    dtype=np.uint16,
    use_scipy=True,
    voxel_sizes: tuple[float, float, float] = (5, 2, 2),
    soma_diameter: float = 16,
    max_cluster_size: float = 100_000,
    ball_xy_size: float = 6,
    ball_z_size: float = 15,
    ball_overlap_fraction: float = 0.6,
    soma_spread_factor: float = 1.4,
    n_free_cpus: int = 2,
    log_sigma_size: float = 0.2,
    n_sds_above_mean_thresh: float = 10,
    split_ball_xy_size: int = 3,
    split_ball_z_size: int = 3,
    split_ball_overlap_fraction: float = 0.8,
    n_splitting_iter: int = 10,
    start_plane: int = 0,
    end_plane: int = 0,
):
    signal_array = read_with_dask(str(signal_path))
    if end_plane <= 0:
        end_plane = len(signal_array)
    signal_array = signal_array[start_plane:end_plane, :, :]

    signal_array = np.asarray(signal_array).astype(dtype)
    shape = signal_array.shape

    settings = DetectionSettings(
        plane_original_np_dtype=dtype,
        plane_shape=shape[1:],
        voxel_sizes=voxel_sizes,
        soma_spread_factor=soma_spread_factor,
        soma_diameter_um=soma_diameter,
        max_cluster_size_um3=max_cluster_size,
        ball_xy_size_um=ball_xy_size,
        ball_z_size_um=ball_z_size,
        start_plane=0,
        end_plane=len(signal_array),
        n_free_cpus=n_free_cpus,
        ball_overlap_fraction=ball_overlap_fraction,
        log_sigma_size=log_sigma_size,
        n_sds_above_mean_thresh=n_sds_above_mean_thresh,
        outlier_keep=False,
        artifact_keep=False,
        save_planes=False,
        batch_size=batch_size,
        torch_device=torch_device,
        n_splitting_iter=n_splitting_iter,
    )

    kwargs = dataclasses.asdict(settings)
    kwargs["ball_z_size_um"] = split_ball_z_size
    kwargs["ball_xy_size_um"] = split_ball_xy_size
    kwargs["ball_overlap_fraction"] = split_ball_overlap_fraction
    kwargs["torch_device"] = "cpu"
    kwargs["plane_original_np_dtype"] = np.float32
    splitting_settings = DetectionSettings(**kwargs)

    signal_array = settings.filter_data_converter_func(signal_array)
    signal_array = torch.from_numpy(signal_array).to(torch_device)

    tile_processor = TileProcessor(
        plane_shape=shape[1:],
        clipping_value=settings.clipping_value,
        threshold_value=settings.threshold_value,
        soma_diameter=settings.soma_diameter,
        log_sigma_size=settings.log_sigma_size,
        n_sds_above_mean_thresh=settings.n_sds_above_mean_thresh,
        torch_device=torch_device,
        dtype=settings.filtering_dtype.__name__,
        use_scipy=use_scipy,
    )

    ball_filter = BallFilter(
        plane_height=settings.plane_height,
        plane_width=settings.plane_width,
        ball_xy_size=settings.ball_xy_size,
        ball_z_size=settings.ball_z_size,
        overlap_fraction=settings.ball_overlap_fraction,
        threshold_value=settings.threshold_value,
        soma_centre_value=settings.soma_centre_value,
        tile_height=settings.tile_height,
        tile_width=settings.tile_width,
        dtype=settings.filtering_dtype.__name__,
        batch_size=batch_size,
        torch_device=torch_device,
        use_mask=True,
    )

    cell_detector = CellDetector(
        settings.plane_height,
        settings.plane_width,
        start_z=ball_filter.first_valid_plane,
        soma_centre_value=settings.detection_soma_centre_value,
    )

    return (
        settings,
        splitting_settings,
        tile_processor,
        ball_filter,
        cell_detector,
        signal_array,
        batch_size,
    )


def save_tiffs(
    root: Path,
    prefix: str,
    start_index: int,
    buffer: torch.Tensor | np.ndarray,
    total: int,
):
    root = root / prefix
    root.mkdir(parents=True, exist_ok=True)

    if isinstance(buffer, np.ndarray):
        arr = buffer
    else:
        arr = buffer.cpu().numpy()
    digits = int(math.ceil(math.log10(total)))
    for i, plane in enumerate(arr, start_index):
        tifffile.imwrite(
            root / f"{prefix}_{i:0{digits}}.tif",
            plane,
            compression="LZW",
        )


def dump_structures(
    output_root: Path,
    settings: DetectionSettings,
    splitting_settings: DetectionSettings,
    cell_detector: CellDetector,
    signal_array,
):
    max_vol = settings.max_cell_volume
    max_cluster = settings.max_cluster_size
    shape = signal_array.shape
    struct_type = np.zeros(shape, dtype=np.uint8)
    struct_type_split = np.zeros(shape, dtype=np.uint8)

    dia = splitting_settings.soma_diameter
    r1 = int(dia / 2)
    r2 = int(dia - r1)
    sphere = np.indices((dia,) * 3)
    position = np.array(
        np.array(
            [
                dia / 2,
            ]
            * 3
        )
    ).reshape((-1, 1, 1, 1))
    arr = np.linalg.norm(sphere - position, axis=0)
    sphere_mask = arr <= dia / 2

    cells = {
        "cells": [],
        "structs_needs_split": [],
        "structs_too_big": [],
        "cells_split": [],
    }

    with open(output_root / "structures.csv", "w", newline="") as fh:
        writer = csv.writer(fh, delimiter=",")
        writer.writerow(["id", "x", "y", "z", "volume", "volume_type"])

        for cell_id, cell_points in cell_detector.get_structures().items():
            vol = len(cell_points)
            x, y, z = get_structure_centre(cell_points)

            if vol < max_vol:
                tp = "maybe_cell"
                color = 1
                cells["cells"].append(Cell((x, y, z), Cell.CELL))
            elif vol < max_cluster:
                tp = "needs_split"
                color = 2
                cells["structs_needs_split"].append(
                    Cell((x, y, z), Cell.UNKNOWN)
                )
            else:
                tp = "too_big"
                color = 3
                cells["structs_too_big"].append(Cell((x, y, z), Cell.UNKNOWN))

            writer.writerow(list(map(str, [cell_id, x, y, z, vol, tp])))
            for p in cell_points:
                struct_type[p[2], p[1], p[0]] = color
                if tp != "needs_split":
                    struct_type_split[p[2], p[1], p[0]] = color

            if tp == "needs_split":
                centers = split_cells(cell_points, settings=splitting_settings)
                for x, y, z in centers:
                    x, y, z = map(int, [x, y, z])
                    if any(v < r1 for v in [x, y, z]):
                        continue
                    if any(v + r2 > d for d, v in zip(shape, [z, y, x])):
                        continue
                    struct_type_split[
                        z - r1 : z + r2, y - r1 : y + r2, x - r1 : x + r2
                    ][sphere_mask] = 1

                    cells["cells_split"].append(Cell((x, y, z), Cell.CELL))

    save_tiffs(output_root, "struct_type", 0, struct_type, len(struct_type))
    save_tiffs(
        output_root,
        "struct_type_split",
        0,
        struct_type_split,
        len(struct_type_split),
    )

    for name, item_cells in cells.items():
        save_cells(item_cells, str(output_root / f"{name}.xml"))


def pad_3d_filtered_images(
    output_root: Path,
    ball_filter: BallFilter,
    signal_array: torch.Tensor | np.ndarray,
    sample_plane: torch.Tensor | np.ndarray,
    n_saved_planes: int,
):
    """
    3d filters skip the first and last planes. This pads them by creating those
    planes as blank planes so that all outputs have same number of planes.
    """
    n = len(signal_array)

    if ball_filter.first_valid_plane:
        # 3d filters skip first few planes
        buff = np.zeros(
            (ball_filter.first_valid_plane, *sample_plane.shape),
            dtype=sample_plane.dtype,
        )
        save_tiffs(output_root, "filtered_3d", 0, buff, n)
        save_tiffs(
            output_root,
            "struct_id",
            0,
            buff.astype(np.uint32),
            n,
        )

        n_saved_planes += ball_filter.first_valid_plane

    if n_saved_planes < n:
        buff = np.zeros(
            (n - n_saved_planes, *sample_plane.shape), dtype=sample_plane.dtype
        )
        save_tiffs(output_root, "filtered_3d", n_saved_planes, buff, n)
        save_tiffs(
            output_root,
            "struct_id",
            n_saved_planes,
            buff.astype(np.uint32),
            n,
        )


def run_filter(
    output_root: Path,
    settings: DetectionSettings,
    splitting_settings: DetectionSettings,
    tile_processor: TileProcessor,
    ball_filter: BallFilter,
    cell_detector: CellDetector,
    signal_array,
    batch_size,
):
    detection_converter = settings.detection_data_converter_func
    previous_plane = None
    n = len(signal_array)
    n_3d_planes = 0
    middle_planes = None

    for i in tqdm.tqdm(range(0, len(signal_array), batch_size)):
        batch = signal_array[i : i + batch_size]
        save_tiffs(output_root, "input", i, batch, n)

        batch_clipped = torch.clone(batch)
        torch.clip_(batch_clipped, 0, tile_processor.clipping_value)
        save_tiffs(output_root, "clipped", i, batch_clipped, n)

        enhanced_planes = tile_processor.peak_enhancer.enhance_peaks(
            batch_clipped
        )
        save_tiffs(output_root, "enhanced", i, enhanced_planes, n)

        filtered_2d, inside_brain_tiles = tile_processor.get_tile_mask(batch)
        save_tiffs(output_root, "inside_brain", i, inside_brain_tiles, n)
        save_tiffs(output_root, "filtered_2d", i, filtered_2d, n)

        ball_filter.append(filtered_2d, inside_brain_tiles)
        if ball_filter.ready:
            ball_filter.walk()
            middle_planes = ball_filter.get_processed_planes()
            buff = middle_planes.copy()
            buff[buff != settings.soma_centre_value] = 0
            save_tiffs(
                output_root,
                "filtered_3d",
                n_3d_planes + ball_filter.first_valid_plane,
                buff,
                n,
            )

            detection_middle_planes = detection_converter(middle_planes)

            for k, (plane, detection_plane) in enumerate(
                zip(middle_planes, detection_middle_planes)
            ):
                previous_plane = cell_detector.process(
                    detection_plane, previous_plane
                )
                save_tiffs(
                    output_root,
                    "struct_id",
                    i + k + ball_filter.first_valid_plane,
                    previous_plane[None, :, :].astype(np.uint32),
                    n,
                )

            n_3d_planes += len(middle_planes)

    pad_3d_filtered_images(
        output_root,
        ball_filter,
        signal_array,
        middle_planes[0, :, :],
        n_3d_planes,
    )

    dump_structures(
        output_root, settings, splitting_settings, cell_detector, signal_array
    )


if __name__ == "__main__":
    with torch.inference_mode(True):
        filter_args = setup_filter(
            Path(r"D:\tiffs\MF1_158F_W\debug\input"),
            soma_diameter=8,
            ball_xy_size=8,
            ball_z_size=8,
            end_plane=0,
            ball_overlap_fraction=0.8,
            log_sigma_size=0.35,
            n_sds_above_mean_thresh=1,
            soma_spread_factor=4,
            max_cluster_size=10000,
            voxel_sizes=(4, 2.03, 2.03),
            torch_device="cuda",
            split_ball_xy_size=6,
            split_ball_z_size=15,
            split_ball_overlap_fraction=0.8,
            n_splitting_iter=10,
            batch_size=1,
        )
        run_filter(
            Path(r"D:\tiffs\MF1_158F_W\debug\output_sig_thresh"), *filter_args
        )
