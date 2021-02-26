"""
N.B imports are within functions to prevent tensorflow being imported before
it's warnings are silenced
"""

import os
import logging
from imlib.general.logging import suppress_specific_logs

tf_suppress_log_messages = [
    "multiprocessing can interact badly with TensorFlow"
]


def main(
    signal_array,
    background_array,
    voxel_sizes,
    start_plane=0,
    end_plane=-1,
    trained_model=None,
    model_weights=None,
    model="resnet50_tv",
    batch_size=32,
    n_free_cpus=2,
    network_voxel_sizes=[5, 1, 1],
    soma_diameter=16,
    ball_xy_size=6,
    ball_z_size=15,
    ball_overlap_fraction=0.6,
    log_sigma_size=0.2,
    n_sds_above_mean_thresh=10,
    soma_spread_factor=1.4,
    max_cluster_size=100000,
    cube_width=50,
    cube_height=50,
    cube_depth=20,
    network_depth="50",
):
    suppress_tf_logging(tf_suppress_log_messages)

    from cellfinder_core.detect import detect
    from cellfinder_core.classify import classify
    from cellfinder_core.tools import prep
    from pathlib import Path

    home = Path.home()
    install_path = home / ".cellfinder"
    logging.info("Detecting cell candidates")

    points = detect.main(
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
    )

    model_weights = prep.prep_classification(
        trained_model, model_weights, install_path, model, n_free_cpus
    )
    if len(points) > 0:
        logging.info("Running classification")
        points = classify.main(
            points,
            signal_array,
            background_array,
            n_free_cpus,
            voxel_sizes,
            network_voxel_sizes,
            batch_size,
            cube_height,
            cube_width,
            cube_depth,
            trained_model,
            model_weights,
            network_depth,
        )
    else:
        logging.info("No candidates, skipping classification")
    return points
    # logging.info("Saving classified cells")
    # save_cells(points, classified_points_path)


def suppress_tf_logging(tf_suppress_log_messages):
    """
    Prevents many lines of logs such as:
    "2019-10-24 16:54:41.363978: I tensorflow/stream_executor/platform/default
    /dso_loader.cc:44] Successfully opened dynamic library libcuda.so.1"
    """
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    for message in tf_suppress_log_messages:
        suppress_specific_logs("tensorflow", message)
