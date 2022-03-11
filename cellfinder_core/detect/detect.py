import multiprocessing
from datetime import datetime
from multiprocessing import Lock
from multiprocessing import Queue as MultiprocessingQueue
from typing import Callable

import numpy as np
from imlib.general.system import get_num_processes

from cellfinder_core.detect.filters.plane.multiprocessing import (
    MpTileProcessor,
)
from cellfinder_core.detect.filters.setup_filters import setup_tile_filtering
from cellfinder_core.detect.filters.volume.multiprocessing import Mp3DFilter


def calculate_parameters_in_pixels(
    voxel_sizes,
    soma_diameter_um,
    max_cluster_size_um3,
    ball_xy_size_um,
    ball_z_size_um,
):
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
    signal_array,
    start_plane,
    end_plane,
    voxel_sizes,
    soma_diameter,
    max_cluster_size,
    ball_xy_size,
    ball_z_size,
    ball_overlap_fraction,
    soma_spread_factor,
    n_free_cpus,
    log_sigma_size,
    n_sds_above_mean_thresh,
    outlier_keep=False,
    artifact_keep=False,
    save_planes=False,
    plane_directory=None,
    *,
    callback: Callable[[int], None] = None,
):
    """
    Parameters
    ----------
    callback : Callable[int], optional
        A callback function that is called every time a plane has finished
        being processed. Called with the plane number that has finished.
    """
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

    workers_queue: MultiprocessingQueue = MultiprocessingQueue(
        maxsize=n_processes
    )
    # WARNING: needs to be AT LEAST ball_z_size
    mp_3d_filter_queue: MultiprocessingQueue = MultiprocessingQueue(
        maxsize=ball_z_size
    )
    for _ in range(n_processes):
        # place holder for the queue to have the right size on first run
        workers_queue.put(None)

    if signal_array.ndim != 3:
        raise IOError("Input data must be 3D")

    setup_params = [
        signal_array[0, :, :],
        soma_diameter,
        ball_xy_size,
        ball_z_size,
        ball_overlap_fraction,
        start_plane,
    ]
    output_queue: MultiprocessingQueue = MultiprocessingQueue()
    planes_done_queue: MultiprocessingQueue = MultiprocessingQueue()

    # Create 3D analysis filter
    mp_3d_filter = Mp3DFilter(
        mp_3d_filter_queue,
        output_queue,
        planes_done_queue,
        soma_diameter,
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

    # start 3D analysis (waits for planes in queue)
    bf_process = multiprocessing.Process(target=mp_3d_filter.process, args=())
    bf_process.start()  # needs to be started before the loop
    clipping_val, threshold_value = setup_tile_filtering(signal_array[0, :, :])

    # Create 2D analysis filter
    mp_tile_processor = MpTileProcessor(workers_queue, mp_3d_filter_queue)
    prev_lock = Lock()

    # start 2D tile filter (output goes into queue for 3D analysis)
    # Creates a list of (running) processes for each 2D plane
    processes = []
    for plane_id, plane in enumerate(signal_array):
        workers_queue.get()
        lock = Lock()
        lock.acquire()
        p = multiprocessing.Process(
            target=mp_tile_processor.process,
            args=(
                plane_id,
                np.array(plane),
                prev_lock,
                lock,
                clipping_val,
                threshold_value,
                soma_diameter,
                log_sigma_size,
                n_sds_above_mean_thresh,
            ),
        )
        prev_lock = lock
        processes.append(p)
        p.start()

    # Trigger callback when 3D filtering is done on a plane
    nplanes_done = 0
    while nplanes_done < len(signal_array):
        callback(planes_done_queue.get(block=True))
        nplanes_done += 1

    # Wait for all the 2D filters to process
    for p in processes:
        p.join()
    # Tell 3D filter that there are no more planes left
    mp_3d_filter_queue.put((None, None, None))
    cells = output_queue.get()
    # Wait for 3D filter to finish
    bf_process.join()

    print(
        "Detection complete - all planes done in : {}".format(
            datetime.now() - start_time
        )
    )
    return cells
