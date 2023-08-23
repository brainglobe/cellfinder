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

import multiprocessing
from datetime import datetime
from queue import Queue
from threading import Lock
from typing import Callable, List, Optional, Sequence, Tuple, TypeVar

import numpy as np
from brainglobe_utils.cells.cells import Cell
from brainglobe_utils.general.system import get_num_processes

from cellfinder_core import logger, types
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
    signal_array: types.array,
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
    n_ball_procs = max(n_processes - 1, 1)
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
        raise ValueError("Input data must be 3D")

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
        n_planes=len(signal_array),
        n_locks_release=n_ball_procs,
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

    # Force spawn context
    mp_ctx = multiprocessing.get_context("spawn")
    with mp_ctx.Pool(n_ball_procs) as worker_pool:
        async_results, locks = _map_with_locks(
            mp_tile_processor.get_tile_mask,
            signal_array,  # type: ignore
            worker_pool,
        )

        # Release the first set of locks for the 2D filtering
        for i in range(min(n_ball_procs + ball_z_size, len(locks))):
            logger.debug(f"🔓 Releasing lock for plane {i}")
            locks[i].release()

        # Start 3D filter
        #
        # This runs in the main thread, and blocks until the all the 2D and
        # then 3D filtering has finished. As batches of planes are filtered
        # by the 3D filter, it releases the locks of subsequent 2D filter
        # processes.
        cells = mp_3d_filter.process(async_results, locks, callback=callback)

    print(
        "Detection complete - all planes done in : {}".format(
            datetime.now() - start_time
        )
    )
    return cells


Tin = TypeVar("Tin")
Tout = TypeVar("Tout")


def _run_func_with_lock(
    func: Callable[[Tin], Tout], arg: Tin, lock: Lock
) -> Tout:
    """
    Run a function after acquiring a lock.
    """
    lock.acquire(blocking=True)
    return func(arg)


def _map_with_locks(
    func: Callable[[Tin], Tout],
    iterable: Sequence[Tin],
    worker_pool: multiprocessing.pool.Pool,
) -> Tuple[Queue, List[Lock]]:
    """
    Map a function to arguments, blocking execution.

    Maps *func* to args in *iterable*, but blocks all execution and
    return a queue of asyncronous results and locks for each of the
    results. Execution can be enabled by releasing the returned
    locks in order.
    """
    # Setup a manager to handle the locks
    m = multiprocessing.Manager()
    # Setup one lock per argument to be mapped
    locks = [m.Lock() for _ in range(len(iterable))]
    [lock.acquire(blocking=False) for lock in locks]

    async_results: Queue = Queue()

    for arg, lock in zip(iterable, locks):
        async_result = worker_pool.apply_async(
            _run_func_with_lock, args=(func, arg, lock)
        )
        async_results.put(async_result)

    return async_results, locks
