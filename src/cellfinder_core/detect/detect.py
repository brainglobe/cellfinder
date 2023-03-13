import multiprocessing
from datetime import datetime
from queue import Queue
from typing import Callable, List, Optional, Tuple, Union

import dask.array as da
import numpy as np
from imlib.cells.cells import Cell
from imlib.general.system import get_num_processes

from cellfinder_core.detect.filters.plane import TileProcessor
from cellfinder_core.detect.filters.setup_filters import setup_tile_filtering
from cellfinder_core.detect.filters.volume.volume_filter import VolumeFilter


def calculate_parameters_in_pixels(
    voxel_sizes: Tuple[float, float, float],
    soma_diameter_um: float,
    max_cluster_size_um3: float,
    ball_xy_size_um: float,
    ball_z_size_um: float,
) -> Tuple[int, int, int, int]:
    """
    Convert the command-line arguments from real (um) units to pixels
    """

    mean_in_plane_pixel_size = 0.5 * (
        float(voxel_sizes[2]) + float(voxel_sizes[1])
    )
    voxel_volume = (
        float(voxel_sizes[2]) * float(voxel_sizes[1]) * float(voxel_sizes[0])
    )
    soma_diameter = int(round(soma_diameter_um / mean_in_plane_pixel_size))
    max_cluster_size = int(round(max_cluster_size_um3 / voxel_volume))
    ball_xy_size = int(round(ball_xy_size_um / mean_in_plane_pixel_size))
    ball_z_size = int(round(ball_z_size_um / float(voxel_sizes[0])))

    return soma_diameter, max_cluster_size, ball_xy_size, ball_z_size


def main(
    signal_array: Union[np.ndarray, da.Array],
    start_plane: int,
    end_plane: int,
    voxel_sizes: Tuple[float, float, float],
    soma_diameter: float,
    max_cluster_size: float,
    ball_xy_size: float,
    ball_z_size: float,
    ball_overlap_fraction: float,
    soma_spread_factor: float,
    n_free_cpus: int,
    log_sigma_size: float,
    n_sds_above_mean_thresh: float,
    outlier_keep: bool = False,
    artifact_keep: bool = False,
    save_planes: bool = False,
    plane_directory: Optional[str] = None,
    *,
    callback: Optional[Callable[[int], None]] = None,
) -> List[Cell]:
    """
    Parameters
    ----------
    callback : Callable[int], optional
        A callback function that is called every time a plane has finished
        being processed. Called with the plane number that has finished.
    """
    if not np.issubdtype(signal_array.dtype, np.integer):
        raise ValueError(
            "signal_array must be integer datatype, but has datatype "
            f"{signal_array.dtype}"
        )
    n_processes = get_num_processes(min_free_cpu_cores=n_free_cpus)
    start_time = datetime.now()

    (
        soma_diameter,
        max_cluster_size,
        ball_xy_size,
        ball_z_size,
    ) = calculate_parameters_in_pixels(
        voxel_sizes,
        soma_diameter,
        max_cluster_size,
        ball_xy_size,
        ball_z_size,
    )

    if end_plane == -1:
        end_plane = len(signal_array)
    signal_array = signal_array[start_plane:end_plane]

    callback = callback or (lambda *args, **kwargs: None)

    if signal_array.ndim != 3:
        raise IOError("Input data must be 3D")

    setup_params = (
        signal_array[0, :, :],
        soma_diameter,
        ball_xy_size,
        ball_z_size,
        ball_overlap_fraction,
        start_plane,
    )

    # Create 3D analysis filter
    mp_3d_filter = VolumeFilter(
        soma_diameter=soma_diameter,
        setup_params=setup_params,
        soma_size_spread_factor=soma_spread_factor,
        planes_paths_range=signal_array,
        save_planes=save_planes,
        plane_directory=plane_directory,
        start_plane=start_plane,
        max_cluster_size=max_cluster_size,
        outlier_keep=outlier_keep,
        artifact_keep=artifact_keep,
    )

    clipping_val, threshold_value = setup_tile_filtering(signal_array[0, :, :])
    # Create 2D analysis filter
    mp_tile_processor = TileProcessor(
        clipping_val,
        threshold_value,
        soma_diameter,
        log_sigma_size,
        n_sds_above_mean_thresh,
    )

    mp_ctx = multiprocessing.get_context("spawn")
    with mp_ctx.Pool(n_processes) as worker_pool:
        # Start 2D filter
        # Submits each plane to the worker pool, and sets up a queue of
        # asyncronous results
        async_results: Queue = Queue()

        # NOTE: Need to make sure every plane isn't read into memory at this
        # stage, as all of these jobs are submitted immediately to the pool.
        # *plane* is a dask array, so as long as it isn't forced into memory
        # (e.g. using np.array(plane)) here then there shouldn't be an issue
        for plane in signal_array:
            res = worker_pool.apply_async(
                mp_tile_processor.get_tile_mask, args=(plane,)
            )
            async_results.put(res)

        # Start 3D filter
        # This runs in the main thread, and blocks until the all the 2D and
        # then 3D filtering has finished
        cells = mp_3d_filter.process(async_results, callback=callback)

    print(
        "Detection complete - all planes done in : {}".format(
            datetime.now() - start_time
        )
    )
    return cells
