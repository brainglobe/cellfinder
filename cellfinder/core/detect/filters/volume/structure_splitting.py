from copy import copy
from typing import List, Optional, Tuple, Type

import numpy as np
import torch

from cellfinder.core.detect.filters.setup_filters import DetectionSettings
from cellfinder.core.detect.filters.volume.ball_filter import (
    BallFilter,
    InvalidVolume,
)
from cellfinder.core.detect.filters.volume.structure_detection import (
    CellDetector,
    get_structure_centre,
)


class StructureSplitException(Exception):
    pass


def get_shape(
    xs: np.ndarray, ys: np.ndarray, zs: np.ndarray
) -> Tuple[int, int, int]:
    """
    Takes a list of x, y, z coordinates and returns a volume size such that
    all the points will fit into it (once the min of each dim is subtracted -
    i.e. the smallest point in each dim falls on zero).

    Axis order is x, y, z.
    """
    # +1 because difference. TEST:
    shape = tuple(int((dim.max() - dim.min()) + 1) for dim in (xs, ys, zs))
    return shape


def coords_to_volume(
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    intensity: Optional[np.ndarray],
    volume_shape: Tuple[int, int, int],
    ball_xy_padding: tuple[int, int],
    ball_z_padding: tuple[int, int],
    dtype: Type[np.number],
    threshold_value: int,
) -> tuple[torch.Tensor, Optional[np.ndarray]]:
    """
    Takes the series of x, y, z points along with the shape of the volume
    (also x, y, z order) that fully enclose them, relative to the minimum
    point in each dim of the data. It than expands the volume in each dim to
    account for the filtering ball diameter by adding padding
    before / after the dim. So each point in the expanded volume is shifted
    by the start padding internally and is then set to the threshold value.

    The result is that after ball filtering, all the points we get will be
    fully contained in the original volume and not in the padding. Of course
    the start padding will need to be subtracted from the point indices in the
    expanded volume.

    The volume is then transposed and returned in the Z, Y, X order.

    If `intensity` is not None, it also returns a volume, with the intensity
    of every point set to the corresponding intensity value.
    """
    # it's faster doing the work in numpy and then returning as torch array,
    # than doing the work in torch
    # Expanded to ensure the ball fits at all borders of input cuboid
    xy_add = sum(ball_xy_padding)
    z_add = sum(ball_z_padding)
    expanded_shape = (
        volume_shape[0] + xy_add,
        volume_shape[1] + xy_add,
        volume_shape[2] + z_add,
    )

    # volume is now x, y, z order
    volume = np.zeros(expanded_shape, dtype=dtype)
    # use largest type. These are small volumes and not processed much except
    # to find center, so it's not a large memory/cpu cost
    raw_volume = None
    if intensity is not None:
        raw_volume = np.zeros(volume.shape, dtype=np.float64)

    x_min, y_min, z_min = xs.min(), ys.min(), zs.min()

    # shift the points so any sphere centered on it would not have its
    # radius expand beyond the volume and so center of sphere would be in
    # original volume
    relative_xs = np.array((xs - x_min + ball_xy_padding[0]), dtype=np.int64)
    relative_ys = np.array((ys - y_min + ball_xy_padding[0]), dtype=np.int64)
    relative_zs = np.array((zs - z_min + ball_z_padding[0]), dtype=np.int64)

    # set each point as the center with a value of threshold
    volume[relative_xs, relative_ys, relative_zs] = threshold_value
    if intensity is not None:
        raw_volume[relative_xs, relative_ys, relative_zs] = intensity

    volume = volume.swapaxes(0, 2)
    if intensity is not None:
        raw_volume = raw_volume.swapaxes(0, 2)
    return torch.from_numpy(volume), raw_volume


def ball_filter_imgs(
    volume: torch.Tensor,
    settings: DetectionSettings,
    raw_volume: Optional[np.ndarray],
) -> CellDetector:
    """
    Apply ball filtering to a 3D volume and detect new structures from the
    original single structure.

    Uses the `BallFilter` class to perform ball filtering on the volume
    and the `CellDetector` class to detect the new cells.

    Parameters
    ----------
    volume : torch.Tensor
        The 3D volume to be 3D filtered (Z, Y, X order). Edited in place.
    settings : DetectionSettings
        The settings to use.
    raw_volume : np.ndarray or None
        The original input data of the same shape as `volume`, if provided.

    Returns
    -------
    cell_detector : CellDetector
        The `CellDetector` that tracks the newly detected structures.

    """
    detection_convert = settings.detection_data_converter_func
    batch_size = settings.batch_size

    # make sure volume is not less than kernel etc
    try:
        bf = BallFilter(
            plane_height=settings.plane_height,
            plane_width=settings.plane_width,
            ball_xy_size=settings.ball_xy_size,
            ball_z_size=settings.ball_z_size,
            overlap_fraction=settings.ball_overlap_fraction,
            threshold_value=settings.threshold_value,
            soma_centre_value=settings.soma_centre_value,
            tile_height=settings.tile_height,
            tile_width=settings.tile_width,
            dtype=settings.filtering_dtype.__name__,
            batch_size=batch_size,
            torch_device=settings.torch_device,
            use_mask=False,  # we don't need a mask here
        )
    except InvalidVolume:
        return CellDetector(
            settings.plane_height,
            settings.plane_width,
            start_z=0,
            soma_centre_value=settings.detection_soma_centre_value,
        )

    start_z = bf.first_valid_plane
    cell_detector = CellDetector(
        settings.plane_height,
        settings.plane_width,
        start_z=start_z,
        soma_centre_value=settings.detection_soma_centre_value,
    )

    previous_plane = None
    for z in range(0, volume.shape[0], batch_size):
        raw_planes_in = None
        if raw_volume is not None:
            raw_planes_in = raw_volume[z : z + batch_size, :, :]
        bf.append(volume[z : z + batch_size, :, :], raw_planes=raw_planes_in)

        if bf.ready:
            bf.walk()

            middle_planes = bf.get_processed_planes()
            raw_planes = None if raw_volume is None else bf.get_raw_planes()
            n = middle_planes.shape[0]

            # we edit volume, but only for planes already processed that won't
            # be passed to the filter in this run
            volume[start_z : start_z + n, :, :] = torch.from_numpy(
                middle_planes
            )
            start_z += n

            # convert to type needed for detection
            middle_planes = detection_convert(middle_planes)
            for i, plane in enumerate(middle_planes):
                raw_plane = None if raw_volume is None else raw_planes[i]
                previous_plane = cell_detector.process(
                    plane, previous_plane, raw_plane
                )

    return cell_detector


def iterative_ball_filter(
    volume: torch.Tensor,
    settings: DetectionSettings,
    raw_volume: Optional[np.ndarray],
) -> List[CellDetector]:
    """
    Apply iterative ball filtering to the given volume.
    The volume is eroded at each iteration, by subtracting 1 from the volume.

    Parameters
    ----------
    volume : torch.Tensor
        The input volume. It is edited inplace. Of shape Z, Y, X.
    settings : DetectionSettings
        The settings to use.
    raw_volume : np.ndarray or None
        The original input data of the same shape as `volume`, if provided.

    Returns
    -------
    cell_detectors: List of CellDetector.
        A list of CellDetector instances, each corresponding to the result
        of one iteration, in that order.
    """
    cell_detectors = []

    for i in range(settings.n_splitting_iter):
        cell_detector = ball_filter_imgs(volume, settings, raw_volume)
        volume.sub_(1)

        cell_detectors.append(cell_detector)
        if not cell_detector.n_structures:
            break

    return cell_detectors


def split_cells(
    cell_points: np.ndarray,
    settings: DetectionSettings,
    intensity: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, tuple[CellDetector, np.ndarray] | None]:
    """
    Split the given structure built from the given cell coordinates into
    smaller structures with their own cell centres.

    Parameters
    ----------
    cell_points : np.ndarray
        Array of cell points with shape (N, 3),
        where N is the number of cell points and each point is represented
        by its x, y, and z coordinates.
    settings : DetectionSettings
        The settings to use for splitting.
    intensity : np.ndarray or None
        An array of size N with the intensity of each point. Needed for
        computing the cell centre using the center of mass method, if selected.

    Returns
    -------
    centres, (cell_detector, offset): A 2-tuple of,
        np.ndarray: Array of absolute cell centres with shape (M, 3),
            where M is the number of individual cells and each centre is
            represented by its x, y, and z coordinates.
        (CellDetector, np.ndarray) or None: If None, then we didn't find any
        better cell candidates during splitting than the original single cell.
        Otherwise, it's the `CellDetector` with the structs from the best
        iteration and a size 3 `np.ndarray` with the offset of the structs in
        the cell detector. I.e. all coordinates in the cell detector is
        relative to the size of the cuboid containing only the `cell_points`.
        So the offset must be added to convert any voxel indices accessed via
        the cell detector (e.g. `cell_detector.get_structures`) to absolute
        voxel indices.
    """
    settings = copy(settings)
    if settings.detect_centre_of_intensity and intensity is None:
        raise ValueError(
            "Using center of intensity, but intensity no provided"
        )

    # these points are in x, y, z order columnwise, in absolute pixels
    # get center to start from in case we find no split points
    orig_centre = get_structure_centre(cell_points, intensity=intensity)

    xs = cell_points[:, 0]
    ys = cell_points[:, 1]
    zs = cell_points[:, 2]

    # total volume enclosing all points from the input
    original_bounding_cuboid_shape = get_shape(xs, ys, zs)

    # they should be the same dtype so as to not need a conversion before
    # passing the input data with marked cells to the filters (we currently
    # set both to float32)
    assert settings.filtering_dtype == settings.plane_original_np_dtype
    # Volume will be padded so not lose points on the edges. It's z, y, x order
    ball_xy_padding = BallFilter.min_xy_padding(settings.ball_xy_size)
    ball_z_padding = BallFilter.min_z_padding(settings.ball_z_size)
    vol, raw_vol = coords_to_volume(
        xs,
        ys,
        zs,
        intensity,
        volume_shape=original_bounding_cuboid_shape,
        ball_xy_padding=ball_xy_padding,
        ball_z_padding=ball_z_padding,
        dtype=settings.filtering_dtype,
        threshold_value=settings.threshold_value,
    )

    # get an estimate of how much memory processing a single batch of original
    # input planes takes. For this much smaller volume, our batch will be such
    # that it uses at most that much memory
    total_vol_size = (
        settings.plane_height * settings.plane_width * settings.batch_size
    )
    batch_size = total_vol_size // (vol.shape[1] * vol.shape[2])
    batch_size = min(batch_size, vol.shape[0])

    # update settings with our volume data
    settings.plane_shape = vol.shape[1:]
    settings.start_plane = 0
    settings.end_plane = vol.shape[0]
    settings.batch_size = batch_size

    # centres is a list of arrays of centres (1 array of centres per ball run)
    # in x, y, z order
    cell_detectors = iterative_ball_filter(vol, settings, raw_vol)
    struct_counts = [d.n_structures for d in cell_detectors]

    # if best split only resulted in one (or no) struct, stick with original
    if not struct_counts or max(struct_counts) <= 1:
        return orig_centre[None, :], None

    best_iteration = struct_counts.index(max(struct_counts))
    cell_detector = cell_detectors[best_iteration]
    # centers come in where zero is relative to expanded vol corner
    expanded_relative_centres = cell_detector.get_cell_centres(
        settings.detect_centre_of_intensity
    )

    # corner coordinates of original unexpended vol in absolute voxels
    orig_corner = np.array([xs.min(), ys.min(), zs.min()])
    # shape of the original unexpanded volume
    orig_cuboid_shape = np.array(original_bounding_cuboid_shape)
    # start padding added to start of original vol to gain expanded volume
    start_padding = np.array(
        [ball_xy_padding[0], ball_xy_padding[0], ball_z_padding[0]],
        dtype=np.int_,
    )

    # remove padding to get indices relative to original vol corner
    relative_centres = expanded_relative_centres - start_padding
    for x in relative_centres:
        # we allow to be sticking out by one on each side due to rounding, but
        # more than one should be impossible
        assert (x <= orig_cuboid_shape).all()
        assert (x >= -1).all()
    # but if they do stick out by one, clip so it's in the valid original vol
    relative_centres = np.clip(relative_centres, 0, orig_cuboid_shape - 1)

    # convert relative centers to absolute voxels in original vol
    absolute_centres = relative_centres + orig_corner[None, :]

    # any indices stored in cell detector is relative to the expanded vol so a
    # zero there (i.e. offset) should be relative to where the padding starts
    offset = orig_corner - start_padding
    return absolute_centres, (cell_detector, offset)
