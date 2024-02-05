"""
N.B imports are within functions to prevent tensorflow being imported before
it's warnings are silenced
"""

import os
from typing import Callable, List, Optional, Tuple

import numpy as np
from brainglobe_utils.general.logging import suppress_specific_logs

from cellfinder.core import logger
from cellfinder.core.download.models import model_type
from cellfinder.core.train.train_yml import depth_type

tf_suppress_log_messages = [
    "multiprocessing can interact badly with TensorFlow"
]


def main(
    signal_array: np.ndarray,
    background_array: np.ndarray,
    voxel_sizes: Tuple[int, int, int],
    start_plane: int = 0,
    end_plane: int = -1,
    trained_model: Optional[os.PathLike] = None,
    model_weights: Optional[os.PathLike] = None,
    model: model_type = "resnet50_tv",
    batch_size: int = 32,
    n_free_cpus: int = 2,
    network_voxel_sizes: Tuple[int, int, int] = (5, 1, 1),
    soma_diameter: int = 16,
    ball_xy_size: int = 6,
    ball_z_size: int = 15,
    ball_overlap_fraction: float = 0.6,
    log_sigma_size: float = 0.2,
    n_sds_above_mean_thresh: int = 10,
    soma_spread_factor: float = 1.4,
    max_cluster_size: int = 100000,
    cube_width: int = 50,
    cube_height: int = 50,
    cube_depth: int = 20,
    network_depth: depth_type = "50",
    *,
    detect_callback: Optional[Callable[[int], None]] = None,
    classify_callback: Optional[Callable[[int], None]] = None,
    detect_finished_callback: Optional[Callable[[list], None]] = None,
) -> List:
    """
    Parameters
    ----------
    detect_callback : Callable[int], optional
        Called every time a plane has finished being processed during the
        detection stage. Called with the plane number that has finished.
    classify_callback : Callable[int], optional
        Called every time tensorflow has finished classifying a point.
        Called with the batch number that has just finished.
    detect_finished_callback : Callable[list], optional
        Called after detection is finished with the list of detected points.
    """
    suppress_tf_logging(tf_suppress_log_messages)

    from cellfinder.core.classify import classify
    from cellfinder.core.detect import detect
    from cellfinder.core.tools import prep

    logger.info("Detecting cell candidates")

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
        callback=detect_callback,
    )

    if detect_finished_callback is not None:
        detect_finished_callback(points)

    install_path = None
    model_weights = prep.prep_model_weights(
        model_weights, install_path, model, n_free_cpus
    )
    if len(points) > 0:
        logger.info("Running classification")
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
            callback=classify_callback,
        )
    else:
        logger.info("No candidates, skipping classification")
    return points


def suppress_tf_logging(tf_suppress_log_messages: List[str]) -> None:
    """
    Prevents many lines of logs such as:
    "2019-10-24 16:54:41.363978: I tensorflow/stream_executor/platform/default
    /dso_loader.cc:44] Successfully opened dynamic library libcuda.so.1"
    """
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    for message in tf_suppress_log_messages:
        suppress_specific_logs("tensorflow", message)
