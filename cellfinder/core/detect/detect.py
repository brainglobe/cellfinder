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
from numba import set_num_threads

from cellfinder.core import logger, types
from cellfinder.core.detect.filters.plane import TileProcessor
from cellfinder.core.detect.filters.setup_filters import setup_tile_filtering
from cellfinder.core.detect.filters.volume.volume_filter import VolumeFilter


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

    if ball_z_size == 0:
        raise ValueError(
            "Ball z size has been calculated to be 0 voxels."
            " This may be due to large axial spacing of your data or the "
            "ball_z_size_um parameter being too small. "
            "Please check input parameters are correct. "
            "Note that cellfinder requires high resolution data in all "
            "dimensions, so that cells can be detected in multiple "
            "image planes."
        )
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
    Perform cell candidate detection on a 3D signal array.

    Parameters
    ----------
    signal_array : numpy.ndarray
        3D array representing the signal data.

    start_plane : int
        Index of the starting plane for detection.

    end_plane : int
        Index of the ending plane for detection.

    voxel_sizes : Tuple[float, float, float]
        Tuple of voxel sizes in each dimension (x, y, z).

    soma_diameter : float
        Diameter of the soma in physical units.

    max_cluster_size : float
        Maximum size of a cluster in physical units.

    ball_xy_size : float
        Size of the XY ball used for filtering in physical units.

    ball_z_size : float
        Size of the Z ball used for filtering in physical units.

    ball_overlap_fraction : float
        Fraction of overlap allowed between balls.

    soma_spread_factor : float
        Spread factor for soma size.

    n_free_cpus : int
        Number of free CPU cores available for parallel processing.

    log_sigma_size : float
        Size of the sigma for the log filter.

    n_sds_above_mean_thresh : float
        Number of standard deviations above the mean threshold.

    outlier_keep : bool, optional
        Whether to keep outliers during detection. Defaults to False.

    artifact_keep : bool, optional
        Whether to keep artifacts during detection. Defaults to False.

    save_planes : bool, optional
        Whether to save the planes during detection. Defaults to False.

    plane_directory : str, optional
        Directory path to save the planes. Defaults to None.

    callback : Callable[int], optional
        A callback function that is called every time a plane has finished
        being processed. Called with the plane number that has finished.

    Returns
    -------
    List[Cell]
        List of detected cells.
    """
    if not np.issubdtype(signal_array.dtype, np.integer):
        raise ValueError(
            "signal_array must be integer datatype, but has datatype "
            f"{signal_array.dtype}"
        )
    n_processes = get_num_processes(min_free_cpu_cores=n_free_cpus)
    n_ball_procs = max(n_processes - 1, 1)

    # we parallelize 2d filtering, which typically lags behind the 3d
    # processing so for n_ball_procs 2d filtering threads, ball_z_size will
    # typically be in use while the others stall waiting for 3d processing
    # so we can use those for other things, such as numba threading
    set_num_threads(max(n_ball_procs - int(ball_z_size), 1))

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
    signal_array = signal_array.astype(np.uint32)

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
        mp_3d_filter.process(async_results, locks, callback=callback)

        # it's now done filtering, get results with pool
        cells = mp_3d_filter.get_results(worker_pool)

    time_elapsed = datetime.now() - start_time
    logger.debug(
        f"All Planes done. Found {len(cells)} cells in {format(time_elapsed)}"
    )
    print("Detection complete - all planes done in : {}".format(time_elapsed))
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
