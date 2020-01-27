from datetime import datetime
import logging
import multiprocessing
from multiprocessing import Queue as MultiprocessingQueue
from multiprocessing import Lock

from tqdm import tqdm
from imlib.general.system import get_sorted_file_paths, get_num_processes

from cellfinder.detect.filters.plane_filters.multiprocessing import (
    MpTileProcessor,
)
from cellfinder.detect.filters.setup_filters import setup
from cellfinder.detect.filters.volume_filters.multiprocessing import Mp3DFilter


def calculate_parameters_in_pixels(
    x_pixel_um,
    y_pixel_um,
    z_pixel_um,
    soma_diameter_um,
    max_cluster_size_um3,
    ball_xy_size_um,
    ball_z_size_um,
):
    """
    Convert the command-line arguments from real (um) units to pixels
    :param x_pixel_um:
    :param y_pixel_um:
    :param z_pixel_um:
    :param soma_diameter_um:
    :param max_cluster_size_um3:
    :param ball_xy_size_um:
    :param ball_z_size_um:
    :return:
    """

    mean_in_plane_pixel_size = 0.5 * (x_pixel_um + y_pixel_um)
    voxel_volume = x_pixel_um * y_pixel_um * z_pixel_um
    soma_diameter = int(round(soma_diameter_um / mean_in_plane_pixel_size))
    max_cluster_size = int(round(max_cluster_size_um3 / voxel_volume))
    ball_xy_size = int(round(ball_xy_size_um / mean_in_plane_pixel_size))
    ball_z_size = int(round(ball_z_size_um / z_pixel_um))

    return soma_diameter, max_cluster_size, ball_xy_size, ball_z_size


def main(args):
    n_processes = get_num_processes(min_free_cpu_cores=args.n_free_cpus)
    start_time = datetime.now()

    (
        soma_diameter,
        max_cluster_size,
        ball_xy_size,
        ball_z_size,
    ) = calculate_parameters_in_pixels(
        args.x_pixel_um,
        args.y_pixel_um,
        args.z_pixel_um,
        args.soma_diameter,
        args.max_cluster_size,
        args.ball_xy_size,
        args.ball_z_size,
    )

    # file extension only used if a directory is passed
    img_paths = get_sorted_file_paths(
        args.signal_planes_paths[0], file_extension="tif"
    )

    if args.end_plane == -1:
        args.end_plane = len(img_paths)
    planes_paths_range = img_paths[args.start_plane : args.end_plane]

    workers_queue = MultiprocessingQueue(maxsize=n_processes)
    # WARNING: needs to be AT LEAST ball_z_size
    mp_3d_filter_queue = MultiprocessingQueue(maxsize=ball_z_size)
    for plane_id in range(n_processes):
        # place holder for the queue to have the right size on first run
        workers_queue.put(None)

    clipping_val, threshold_value, ball_filter, cell_detector = setup(
        img_paths[0],
        soma_diameter,
        ball_xy_size,
        ball_z_size,
        ball_overlap_fraction=args.ball_overlap_fraction,
        z_offset=args.start_plane,
    )

    progress_bar = tqdm(
        total=len(planes_paths_range), desc="Processing planes"
    )
    mp_3d_filter = Mp3DFilter(
        mp_3d_filter_queue,
        ball_filter,
        cell_detector,
        soma_diameter,
        args.output_dir,
        soma_size_spread_factor=args.soma_spread_factor,
        progress_bar=progress_bar,
        save_planes=args.save_planes,
        plane_directory=args.plane_directory,
        start_plane=args.start_plane,
        max_cluster_size=max_cluster_size,
        outlier_keep=args.outlier_keep,
        artifact_keep=args.artifact_keep,
        save_csv=args.save_csv,
    )

    # start 3D analysis (waits for planes in queue)
    bf_process = multiprocessing.Process(target=mp_3d_filter.process, args=())
    bf_process.start()  # needs to be started before the loop

    mp_tile_processor = MpTileProcessor(workers_queue, mp_3d_filter_queue)
    prev_lock = Lock()
    processes = []

    # start 2D tile filter (output goes into queue for 3D analysis)
    for plane_id, path in enumerate(planes_paths_range):
        workers_queue.get()
        lock = Lock()
        lock.acquire()
        p = multiprocessing.Process(
            target=mp_tile_processor.process,
            args=(
                plane_id,
                path,
                prev_lock,
                lock,
                clipping_val,
                threshold_value,
                soma_diameter,
                args.log_sigma_size,
                args.n_sds_above_mean_thresh,
            ),
        )
        prev_lock = lock
        processes.append(p)
        p.start()

    processes[-1].join()
    mp_3d_filter_queue.put((None, None, None))  # Signal the end
    bf_process.join()

    logging.info(
        "Detection complete - all planes done in : {}".format(
            datetime.now() - start_time
        )
    )
