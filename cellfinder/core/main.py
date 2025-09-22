import os
from typing import Callable, List, Optional, Tuple

from brainglobe_utils.cells.cells import Cell

from cellfinder.core import logger, types
from cellfinder.core.download.download import model_type
from cellfinder.core.train.train_yaml import depth_type


def main(
    signal_array: types.array,
    background_array: types.array,
    voxel_sizes: Tuple[float, float, float],
    start_plane: int = 0,
    end_plane: int = -1,
    trained_model: Optional[os.PathLike] = None,
    model_weights: Optional[os.PathLike] = None,
    model: model_type = "resnet50_tv",
    classification_batch_size: int = 64,
    n_free_cpus: int = 2,
    network_voxel_sizes: Tuple[float, float, float] = (5, 1, 1),
    soma_diameter: float = 16,
    ball_xy_size: float = 6,
    ball_z_size: float = 15,
    ball_overlap_fraction: float = 0.6,
    log_sigma_size: float = 0.2,
    n_sds_above_mean_thresh: float = 10,
    soma_spread_factor: float = 1.4,
    max_cluster_size: float = 100000,
    cube_width: int = 50,
    cube_height: int = 50,
    cube_depth: int = 20,
    network_depth: depth_type = "50",
    skip_detection: bool = False,
    skip_classification: bool = False,
    detected_cells: List[Cell] = None,
    detection_batch_size: Optional[int] = None,
    torch_device: Optional[str] = None,
    split_ball_xy_size: float = 6,
    split_ball_z_size: float = 15,
    split_ball_overlap_fraction: float = 0.8,
    n_splitting_iter: int = 10,
    *,
    detect_callback: Optional[Callable[[int], None]] = None,
    classify_callback: Optional[Callable[[int], None]] = None,
    detect_finished_callback: Optional[Callable[[list], None]] = None,
    normalize_channels: bool = False,
    normalization_down_sampling: int = 32,
) -> List[Cell]:
    """
    Parameters
    ----------
    signal_array : numpy.ndarray or dask array
        3D array representing the signal data in z, y, x order.
    background_array : numpy.ndarray or dask array
        3D array representing the signal data in z, y, x order.
    voxel_sizes : 3-tuple of floats
        Size of your voxels in the z, y, and x dimensions (microns).
    start_plane : int
        First plane index to process (inclusive, to process a subset of the
        data).
    end_plane : int
        Last plane index to process (exclusive, to process a subset of the
        data).
    trained_model : Optional[Path]
        Trained model file path (home directory (default) -> pretrained
        weights).
    model_weights : Optional[Path]
        Model weights path (home directory (default) -> pretrained
        weights).
    model: str
        Type of model to use. Defaults to `"resnet50_tv"`.
    classification_batch_size : int
        How many potential cells to classify at one time. The GPU/CPU
        memory must be able to contain at once this many data cubes for
        the models. For performance-critical applications, tune to maximize
        memory usage without running out. Check your GPU/CPU memory to verify
        it's not full.
    n_free_cpus : int
        How many CPU cores to leave free.
    network_voxel_sizes : 3-tuple of floats
        Size of the pre-trained network's voxels (microns) in the z, y, and x
        dimensions.
    soma_diameter : float
        The expected in-plane (xy) soma diameter (microns).
    ball_xy_size : float
        3d filter's in-plane (xy) filter ball size (microns).
    ball_z_size : float
        3d filter's axial (z) filter ball size (microns).
    ball_overlap_fraction : float
        3d filter's fraction of the ball filter needed to be filled by
        foreground voxels, centered on a voxel, to retain the voxel.
    log_sigma_size : float
        Gaussian filter width (as a fraction of soma diameter) used during
        2d in-plane Laplacian of Gaussian filtering.
    n_sds_above_mean_thresh : float
        Intensity threshold (the number of standard deviations above
        the mean) of the filtered 2d planes used to mark pixels as
        foreground or background.
    soma_spread_factor : float
        Cell spread factor for determining the largest cell volume before
        splitting up cell clusters. Structures with spherical volume of
        diameter `soma_spread_factor * soma_diameter` or less will not be
        split.
    max_cluster_size : float
        Largest detected cell cluster (in cubic um) where splitting
        should be attempted. Clusters above this size will be labeled
        as artifacts.
    cube_width: int
        The width of the data cube centered on the cell used for
        classification. Defaults to `50`.
    cube_height: int
        The height of the data cube centered on the cell used for
        classification. Defaults to `50`.
    cube_depth: int
        The depth of the data cube centered on the cell used for
        classification. Defaults to `20`.
    network_depth: str
        The network depth to use during classification. Defaults to `"50"`.
    skip_detection : bool
        If selected, the detection step is skipped and instead we get the
        detected cells from the cell layer below (from a previous
        detection run or import).
    skip_classification : bool
        If selected, the classification step is skipped and all cells from
        the detection stage are added.
    detected_cells: Optional list of Cell objects.
        If specified, the cells to use during classification.
    detection_batch_size: int
        The number of planes of the original data volume to process at
        once. The GPU/CPU memory must be able to contain this many planes
        for all the filters. For performance-critical applications, tune
        to maximize memory usage without running out. Check your GPU/CPU
        memory to verify it's not full.
    torch_device : str, optional
        The device on which to run the computation. If not specified (None),
        "cuda" will be used if a GPU is available, otherwise "cpu".
        You can also manually specify "cuda" or "cpu".
    split_ball_xy_size: float
        Similar to `ball_xy_size`, except the value to use for the 3d
        filter during cluster splitting.
    split_ball_z_size: float
        Similar to `ball_z_size`, except the value to use for the 3d filter
        during cluster splitting.
    split_ball_overlap_fraction: float
        Similar to `ball_overlap_fraction`, except the value to use for the
        3d filter during cluster splitting.
    n_splitting_iter: int
        The number of iterations to run the 3d filtering on a cluster. Each
        iteration reduces the cluster size by the voxels not retained in
        the previous iteration.
    detect_callback : Callable[int], optional
        Called every time a plane has finished being processed during the
        detection stage. Called with the plane number that has finished.
    classify_callback : Callable[int], optional
        Called every time a point has finished being classified.
        Called with the batch number that has just finished.
    detect_finished_callback : Callable[list], optional
        Called after detection is finished with the list of detected points.
    normalize_channels : bool
        If True, the signal and background data will be each normalized
        to a mean of zero and standard deviation of 1 before classification.
        Defaults to False.
    normalization_down_sampling : int
        If `normalize_channels` is True, the data arrays will be down-sampled
        in the first axis by this value before calculating their statistics
        before classification. E.g. a value of 2 means every second plane will
        be used. Defaults to 32.
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
            batch_size=detection_batch_size,
            torch_device=torch_device,
            callback=detect_callback,
            split_ball_z_size=split_ball_z_size,
            split_ball_xy_size=split_ball_xy_size,
            split_ball_overlap_fraction=split_ball_overlap_fraction,
            n_splitting_iter=n_splitting_iter,
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
                classification_batch_size,
                cube_height,
                cube_width,
                cube_depth,
                trained_model,
                model_weights,
                network_depth,
                callback=classify_callback,
                normalize_channels=normalize_channels,
                normalization_down_sampling=normalization_down_sampling,
            )
        else:
            logger.info("No candidates, skipping classification")
    return points
