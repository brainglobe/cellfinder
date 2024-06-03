import os
from typing import Callable, List, Optional, Tuple

import numpy as np
from brainglobe_utils.cells.cells import Cell

from cellfinder.core import logger
from cellfinder.core.download.download import model_type
from cellfinder.core.train.train_yml import depth_type


def main(
    signal_array: np.ndarray,
    background_array: np.ndarray,
    voxel_sizes: Tuple[int, int, int],
    start_plane: int = 0,
    end_plane: int = -1,
    trained_model: Optional[os.PathLike] = None,
    model_weights: Optional[os.PathLike] = None,
    model: model_type = "resnet50_tv",
    batch_size: int = 64,
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
    skip_detection: bool = False,
    skip_classification: bool = False,
    detected_cells: List[Cell] = None,
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
        Called every time a point has finished being classified.
        Called with the batch number that has just finished.
    detect_finished_callback : Callable[list], optional
        Called after detection is finished with the list of detected points.
    """
    from cellfinder.core.classify import classify
    from cellfinder.core.detect import detect
    from cellfinder.core.tools import prep

    if not skip_detection:
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
    else:
        points = detected_cells or []  # if None
        detect_finished_callback(points)

    if not skip_classification:
        install_path = None
        model_weights = prep.prep_model_weights(
            model_weights, install_path, model
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
