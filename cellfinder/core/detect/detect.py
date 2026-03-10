"""
Detection is run in three steps:

1. 2D filtering
2. 3D filtering
3. Structure detection

In steps 1. and 2. filters are applied, and any bright points detected
post-filter are marked. To avoid using a separate mask array to mark the
bright points, the input data is clipped to [0, (max_val - 2)]
(max_val is the maximum value that the image data type can store), and:
- (max_val - 1) is used to mark bright points during 2D filtering
- (max_val) is used to mark bright points during 3D filtering
"""

import dataclasses
from datetime import datetime
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
from brainglobe_utils.cells.cells import Cell

from cellfinder.core import logger, types
from cellfinder.core.detect.filters.plane import TileProcessor
from cellfinder.core.detect.filters.setup_filters import DetectionSettings
from cellfinder.core.detect.filters.volume.volume_filter import VolumeFilter
from cellfinder.core.tools.tools import inference_wrapper


# -------------------------------------------------------------------------
# Validation helper for the detection pipeline
# -------------------------------------------------------------------------
def _validate_detection_inputs(
    signal_array,
    voxel_sizes,
    soma_diameter,
    ball_xy_size,
    ball_z_size,
    ball_overlap_fraction,
    batch_size,
):
    """
    Validate detection parameters before running the pipeline.
    """

    # Ensure numeric data
    if not np.issubdtype(signal_array.dtype, np.number):
        raise TypeError(
            f"signal_array must contain numeric values, got dtype {signal_array.dtype}"
        )

    # Ensure 3D volume
    if signal_array.ndim != 3:
        raise ValueError("Input data must be 3D"
        )

    # Voxel sizes must be positive
    if any(v <= 0 for v in voxel_sizes):
        raise ValueError("voxel_sizes must contain positive values")

    # Soma diameter must be positive
    if soma_diameter <= 0:
        raise ValueError("soma_diameter must be positive")

    # Ball filter dimensions must be positive
    if ball_xy_size <= 0 or ball_z_size <= 0:
        raise ValueError("ball filter sizes must be positive")

    # Overlap fraction must be valid
    if not (0 < ball_overlap_fraction <= 1):
        raise ValueError("ball_overlap_fraction must be between 0 and 1")

    # Batch size must be valid if provided
    if batch_size is not None and batch_size < 1:
        raise ValueError("batch_size must be >= 1")


@inference_wrapper
def main(
    signal_array: types.array,
    start_plane: int = 0,
    end_plane: int = -1,
    voxel_sizes: Tuple[float, float, float] = (5, 2, 2),
    soma_diameter: float = 16,
    max_cluster_size: float = 100_000,
    ball_xy_size: float = 6,
    ball_z_size: float = 15,
    ball_overlap_fraction: float = 0.6,
    soma_spread_factor: float = 1.4,
    n_free_cpus: int = 2,
    log_sigma_size: float = 0.2,
    n_sds_above_mean_thresh: float = 10,
    outlier_keep: bool = False,
    artifact_keep: bool = False,
    save_planes: bool = False,
    plane_directory: Optional[str] = None,
    batch_size: Optional[int] = None,
    torch_device: Optional[str] = None,
    pin_memory: bool = False,
    split_ball_xy_size: float = 6,
    split_ball_z_size: float = 15,
    split_ball_overlap_fraction: float = 0.8,
    n_splitting_iter: int = 10,
    n_sds_above_mean_tiled_thresh: float = 10,
    tiled_thresh_tile_size: float | None = None,
    *,
    callback: Optional[Callable[[int], None]] = None,
) -> List[Cell]:
    """
    Perform cell candidate detection on a 3D signal array.
    """

    start_time = datetime.now()

    if torch_device is None:
        torch_device = "cuda" if torch.cuda.is_available() else "cpu"

    if batch_size is None:
        if torch_device == "cpu":
            batch_size = 4
        else:
            batch_size = 1

    # Validate detection parameters early
    _validate_detection_inputs(
        signal_array,
        voxel_sizes,
        soma_diameter,
        ball_xy_size,
        ball_z_size,
        ball_overlap_fraction,
        batch_size,
    )

    if end_plane < 0:
        end_plane = len(signal_array)
    end_plane = min(len(signal_array), end_plane)

    torch_device = torch_device.lower()

    # Use SciPy filtering on CPU (better performance); use PyTorch on GPU
    if torch_device != "cuda":
        use_scipy = True
    else:
        use_scipy = False

    batch_size = max(batch_size, 1)

    # brainmapper can pass them in as str
    voxel_sizes = list(map(float, voxel_sizes))

    settings = DetectionSettings(
        plane_shape=signal_array.shape[1:],
        plane_original_np_dtype=signal_array.dtype,
        voxel_sizes=voxel_sizes,
        soma_spread_factor=soma_spread_factor,
        soma_diameter_um=soma_diameter,
        max_cluster_size_um3=max_cluster_size,
        ball_xy_size_um=ball_xy_size,
        ball_z_size_um=ball_z_size,
        start_plane=start_plane,
        end_plane=end_plane,
        n_free_cpus=n_free_cpus,
        ball_overlap_fraction=ball_overlap_fraction,
        log_sigma_size=log_sigma_size,
        n_sds_above_mean_thresh=n_sds_above_mean_thresh,
        n_sds_above_mean_tiled_thresh=n_sds_above_mean_tiled_thresh,
        tiled_thresh_tile_size=tiled_thresh_tile_size,
        outlier_keep=outlier_keep,
        artifact_keep=artifact_keep,
        save_planes=save_planes,
        plane_directory=plane_directory,
        batch_size=batch_size,
        torch_device=torch_device,
        pin_memory=pin_memory,
        n_splitting_iter=n_splitting_iter,
    )

    # replicate the settings specific to splitting
    kwargs = dataclasses.asdict(settings)
    kwargs["ball_z_size_um"] = split_ball_z_size
    kwargs["ball_xy_size_um"] = split_ball_xy_size
    kwargs["ball_overlap_fraction"] = split_ball_overlap_fraction
    kwargs["torch_device"] = "cpu"
    kwargs["plane_original_np_dtype"] = np.float32
    splitting_settings = DetectionSettings(**kwargs)

    # Create 3D analysis filter
    mp_3d_filter = VolumeFilter(settings=settings)

    # Create 2D analysis filter
    mp_tile_processor = TileProcessor(
        plane_shape=settings.plane_shape,
        clipping_value=settings.clipping_value,
        threshold_value=settings.threshold_value,
        n_sds_above_mean_thresh=settings.n_sds_above_mean_thresh,
        n_sds_above_mean_tiled_thresh=settings.n_sds_above_mean_tiled_thresh,
        tiled_thresh_tile_size=settings.tiled_thresh_tile_size,
        log_sigma_size=log_sigma_size,
        soma_diameter=settings.soma_diameter,
        torch_device=torch_device,
        dtype=settings.filtering_dtype.__name__,
        use_scipy=use_scipy,
    )

    orig_n_threads = torch.get_num_threads()
    torch.set_num_threads(settings.n_torch_comp_threads)

    # process the data
    mp_3d_filter.process(mp_tile_processor, signal_array, callback=callback)
    cells = mp_3d_filter.get_results(splitting_settings)

    torch.set_num_threads(orig_n_threads)

    time_elapsed = datetime.now() - start_time
    s = f"Detection complete. Found {len(cells)} cells in {time_elapsed}"
    logger.info(s)

    return cells
