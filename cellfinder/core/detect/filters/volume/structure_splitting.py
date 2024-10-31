from typing import List, Tuple, Type

import numpy as np
import torch

from cellfinder.core import logger
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
    all the points will fit into it. With axis order = x, y, z.
    """
    # +1 because difference. TEST:
    shape = tuple(int((dim.max() - dim.min()) + 1) for dim in (xs, ys, zs))
    return shape


def coords_to_volume(
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    volume_shape: Tuple[int, int, int],
    ball_radius: int,
    dtype: Type[np.number],
    threshold_value: int,
) -> torch.Tensor:
    """
    Takes the series of x, y, z points along with the shape of the volume
    that fully enclose them (also x, y, z order). It than expands the
    shape by the ball diameter in each axis. Then, each point, shifted
    by the radius internally is set to the threshold value.

    The volume is then transposed and returned in the Z, Y, X order.
    """
    # it's faster doing the work in numpy and then returning as torch array,
    # than doing the work in torch
    ball_diameter = ball_radius * 2
    # Expanded to ensure the ball fits even at the border
    expanded_shape = [dim_size + ball_diameter for dim_size in volume_shape]
    # volume is now x, y, z order
    volume = np.zeros(expanded_shape, dtype=dtype)

    x_min, y_min, z_min = xs.min(), ys.min(), zs.min()

    # shift the points so any sphere centered on it would not have its
    # radius expand beyond the volume
    relative_xs = np.array((xs - x_min + ball_radius), dtype=np.int64)
    relative_ys = np.array((ys - y_min + ball_radius), dtype=np.int64)
    relative_zs = np.array((zs - z_min + ball_radius), dtype=np.int64)

    # set each point as the center with a value of threshold
    for rel_x, rel_y, rel_z in zip(relative_xs, relative_ys, relative_zs):
        volume[rel_x, rel_y, rel_z] = threshold_value

    volume = volume.swapaxes(0, 2)
    return torch.from_numpy(volume)


def ball_filter_imgs(
    volume: torch.Tensor, settings: DetectionSettings
) -> np.ndarray:
    """
    Apply ball filtering to a 3D volume and detect cell centres.

    Uses the `BallFilter` class to perform ball filtering on the volume
    and the `CellDetector` class to detect cell centres.

    Args:
        volume (torch.Tensor): The 3D volume to be filtered (Z, Y, X order).
        settings (DetectionSettings):
            The settings to use.

    Returns:
        The 2D array of cell centres (N, 3) - X, Y, Z order.

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
        return np.empty((0, 3))

    start_z = bf.first_valid_plane
    cell_detector = CellDetector(
        settings.plane_height,
        settings.plane_width,
        start_z=start_z,
        soma_centre_value=settings.detection_soma_centre_value,
    )

    previous_plane = None
    for z in range(0, volume.shape[0], batch_size):
        bf.append(volume[z : z + batch_size, :, :])

        if bf.ready:
            bf.walk()

            middle_planes = bf.get_processed_planes()
            n = middle_planes.shape[0]

            # we edit volume, but only for planes already processed that won't
            # be passed to the filter in this run
            volume[start_z : start_z + n, :, :] = torch.from_numpy(
                middle_planes
            )
            start_z += n

            # convert to type needed for detection
            middle_planes = detection_convert(middle_planes)
            for plane in middle_planes:
                previous_plane = cell_detector.process(plane, previous_plane)

    return cell_detector.get_cell_centres()


def iterative_ball_filter(
    volume: torch.Tensor, settings: DetectionSettings
) -> Tuple[List[int], List[np.ndarray]]:
    """
    Apply iterative ball filtering to the given volume.
    The volume is eroded at each iteration, by subtracting 1 from the volume.

    Parameters:
        volume (torch.Tensor): The input volume. It is edited inplace.
            Of shape Z, Y, X.
        settings (DetectionSettings): The settings to use.

    Returns:
        tuple: A tuple containing two lists:
            The number of structures found in each iteration.
            The cell centres found in each iteration.
    """
    ns = []
    centres = []

    for i in range(settings.n_splitting_iter):
        cell_centres = ball_filter_imgs(volume, settings)
        volume.sub_(1)

        n_structures = len(cell_centres)
        ns.append(n_structures)
        centres.append(cell_centres)
        if n_structures == 0:
            break

    return ns, centres


def check_centre_in_cuboid(centre: np.ndarray, max_coords: np.ndarray) -> bool:
    """
    Checks whether a coordinate is in a cuboid.

    Parameters
    ----------

    centre : np.ndarray
        x, y, z coordinate.
    max_coords : np.ndarray
        Far corner of cuboid.

    Returns
    -------
    True if within cuboid, otherwise False.
    """
    relative_coords = centre
    if (relative_coords > max_coords).all():
        logger.info(
            'Relative coordinates "{}" exceed maximum volume '
            'dimension of "{}"'.format(relative_coords, max_coords)
        )
        return False
    else:
        return True


def split_cells(
    cell_points: np.ndarray, settings: DetectionSettings
) -> np.ndarray:
    """
    Split the given cell points into individual cell centres.

    Args:
        cell_points (np.ndarray): Array of cell points with shape (N, 3),
            where N is the number of cell points and each point is represented
            by its x, y, and z coordinates.
        settings (DetectionSettings) : The settings to use for splitting. It is
            modified inplace.

    Returns:
        np.ndarray: Array of absolute cell centres with shape (M, 3),
            where M is the number of individual cells and each centre is
            represented by its x, y, and z coordinates.
    """
    # these points are in x, y, z order columnwise, in absolute pixels
    orig_centre = get_structure_centre(cell_points)

    xs = cell_points[:, 0]
    ys = cell_points[:, 1]
    zs = cell_points[:, 2]

    # corner coordinates in absolute pixels
    orig_corner = np.array([xs.min(), ys.min(), zs.min()])
    # volume center relative to corner
    relative_orig_centre = np.array(
        [
            orig_centre[0] - orig_corner[0],
            orig_centre[1] - orig_corner[1],
            orig_centre[2] - orig_corner[2],
        ]
    )

    # total volume enclosing all points
    original_bounding_cuboid_shape = get_shape(xs, ys, zs)

    ball_radius = settings.ball_xy_size // 2
    # they should be the same dtype so as to not need a conversion before
    # passing the input data with marked cells to the filters (we currently
    # set both to float32)
    assert settings.filtering_dtype == settings.plane_original_np_dtype
    # volume will now be z, y, x order
    vol = coords_to_volume(
        xs,
        ys,
        zs,
        volume_shape=original_bounding_cuboid_shape,
        ball_radius=ball_radius,
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
    ns, centres = iterative_ball_filter(vol, settings)
    ns.insert(0, 1)
    centres.insert(0, np.array([relative_orig_centre]))

    best_iteration = ns.index(max(ns))
    # TODO: put constraint on minimum centres distance ?
    relative_centres = centres[best_iteration]

    if not settings.outlier_keep:
        # TODO: change to checking whether in original cluster shape
        original_max_coords = np.array(original_bounding_cuboid_shape)
        relative_centres = np.array(
            [
                x
                for x in relative_centres
                if check_centre_in_cuboid(x, original_max_coords)
            ]
        )

    absolute_centres = np.empty((len(relative_centres), 3))
    # convert centers to absolute pixels
    absolute_centres[:, 0] = orig_corner[0] + relative_centres[:, 0]
    absolute_centres[:, 1] = orig_corner[1] + relative_centres[:, 1]
    absolute_centres[:, 2] = orig_corner[2] + relative_centres[:, 2]

    return absolute_centres
