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
    n_sds_above_mean_local_thresh: float = 10,
    local_thresh_tile_size: float | None = None,
    outlier_keep: bool = False,
    artifact_keep: bool = False,
    save_planes: bool = False,
    plane_directory: Optional[str] = None,
    batch_size: Optional[int] = None,
    torch_device: Optional[str] = None,
    split_ball_xy_size: int = 6,
    split_ball_z_size: int = 15,
    split_ball_overlap_fraction: float = 0.8,
    n_splitting_iter: int = 10,
    *,
    callback: Optional[Callable[[int], None]] = None,
) -> List[Cell]:
    """
    Perform cell candidate detection on a 3D signal array.

    Parameters
    ----------
    signal_array : numpy.ndarray or dask array
        3D array representing the signal data in z, y, x order.
    start_plane : int
        First plane index to process (inclusive, to process a subset of the
        data).
    end_plane : int
        Last plane index to process (exclusive, to process a subset of the
        data).
    voxel_sizes : 3-tuple of floats
        Size of your voxels in the z, y, and x dimensions (microns).
    soma_diameter : float
        The expected in-plane (xy) soma diameter (microns).
    max_cluster_size : int
        Largest detected cell cluster (in cubic um) where splitting
        should be attempted. Clusters above this size will be labeled
        as artifacts.
    ball_xy_size : float
        3d filter's in-plane (xy) filter ball size (microns).
    ball_z_size : float
        3d filter's axial (z) filter ball size (microns).
    ball_overlap_fraction : float
        3d filter's fraction of the ball filter needed to be filled by
        foreground voxels, centered on a voxel, to retain the voxel.
    soma_spread_factor : float
        Cell spread factor for determining the largest cell volume before
        splitting up cell clusters. Structures with spherical volume of
        diameter `soma_spread_factor * soma_diameter` or less will not be
        split.
    n_free_cpus : int
        How many CPU cores to leave free.
    log_sigma_size : float
        Gaussian filter width (as a fraction of soma diameter) used during
        2d in-plane filtering.
    n_sds_above_mean_thresh : float
        Intensity threshold (the number of standard deviations above
        the mean) of the filtered 2d planes used to mark pixels as
        foreground or background.
    outlier_keep : bool, optional
        Whether to keep outliers during detection. Defaults to False.
    artifact_keep : bool, optional
        Whether to keep artifacts during detection. Defaults to False.
    save_planes : bool, optional
        Whether to save the planes during detection. Defaults to False.
    plane_directory : str, optional
        Directory path to save the planes. Defaults to None.
    batch_size: int
        The number of planes of the original data volume to process at
        once. The GPU/CPU memory must be able to contain this many planes
        for all the filters. Tune to maximize memory usage without running
        out. Check your GPU/CPU memory to verify it's not full.
    torch_device : str, optional
        The device on which to run the computation. If not specified (None),
        "cuda" will be used if a GPU is available, otherwise "cpu".
        You can also manually specify "cuda" or "cpu".
    split_ball_xy_size: int
        Similar to `ball_xy_size`, except the value to use for the 3d
        filter during cluster splitting.
    split_ball_z_size: int
        Similar to `ball_z_size`, except the value to use for the 3d filter
        during cluster splitting.
    split_ball_overlap_fraction: float
        Similar to `ball_overlap_fraction`, except the value to use for the
        3d filter during cluster splitting.
    n_splitting_iter: int
        The number of iterations to run the 3d filtering on a cluster. Each
        iteration reduces the cluster size by the voxels not retained in
        the previous iteration.
    callback : Callable[int], optional
        A callback function that is called every time a plane has finished
        being processed. Called with the plane number that has finished.

    Returns
    -------
    List[Cell]
        List of detected potential cells and artifacts.
    """
    start_time = datetime.now()
    if torch_device is None:
        torch_device = "cuda" if torch.cuda.is_available() else "cpu"
    if batch_size is None:
        if torch_device == "cpu":
            batch_size = 4
        else:
            batch_size = 1

    if not np.issubdtype(signal_array.dtype, np.number):
        raise TypeError(
            "signal_array must be a numpy datatype, but has datatype "
            f"{signal_array.dtype}"
        )

    if signal_array.ndim != 3:
        raise ValueError("Input data must be 3D")

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
        n_sds_above_mean_local_thresh=n_sds_above_mean_local_thresh,
        local_thresh_tile_size=local_thresh_tile_size,
        outlier_keep=outlier_keep,
        artifact_keep=artifact_keep,
        save_planes=save_planes,
        plane_directory=plane_directory,
        batch_size=batch_size,
        torch_device=torch_device,
        n_splitting_iter=n_splitting_iter,
    )

    # replicate the settings specific to splitting, before we access anything
    # of the original settings, causing cached properties
    kwargs = dataclasses.asdict(settings)
    kwargs["ball_z_size_um"] = split_ball_z_size
    kwargs["ball_xy_size_um"] = split_ball_xy_size
    kwargs["ball_overlap_fraction"] = split_ball_overlap_fraction
    # always run on cpu because copying to gpu overhead is likely slower than
    # any benefit for detection on smallish volumes
    kwargs["torch_device"] = "cpu"
    # for splitting, we only do 3d filtering. Its input is a zero volume
    # with cell voxels marked with threshold_value. So just use float32
    # for input because the filters will also use float(32). So there will
    # not be need to convert the input a different dtype before passing to
    # the filters.
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
        n_sds_above_mean_local_thresh=settings.n_sds_above_mean_local_thresh,
        local_thresh_tile_size=settings.local_thresh_tile_size,
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
