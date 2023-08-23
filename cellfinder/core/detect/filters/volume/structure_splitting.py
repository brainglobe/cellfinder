from typing import List, Tuple

import numpy as np

from cellfinder_core import logger
from cellfinder_core.detect.filters.volume.ball_filter import BallFilter
from cellfinder_core.detect.filters.volume.structure_detection import (
    CellDetector,
    get_structure_centre,
)


class StructureSplitException(Exception):
    pass


def get_shape(xs: np.ndarray, ys: np.ndarray, zs: np.ndarray) -> List[int]:
    # +1 because difference. TEST:
    shape = [int((dim.max() - dim.min()) + 1) for dim in (xs, ys, zs)]
    return shape


def coords_to_volume(
    xs: np.ndarray, ys: np.ndarray, zs: np.ndarray, ball_radius: int = 1
) -> np.ndarray:
    ball_diameter = ball_radius * 2
    # Expanded to ensure the ball fits even at the border
    expanded_shape = [
        dim_size + ball_diameter for dim_size in get_shape(xs, ys, zs)
    ]
    volume = np.zeros(expanded_shape, dtype=np.uint16)

    x_min, y_min, z_min = xs.min(), ys.min(), zs.min()

    relative_xs = np.array((xs - x_min + ball_radius), dtype=np.int64)
    relative_ys = np.array((ys - y_min + ball_radius), dtype=np.int64)
    relative_zs = np.array((zs - z_min + ball_radius), dtype=np.int64)

    # OPTIMISE: vectorize
    for rel_x, rel_y, rel_z in zip(relative_xs, relative_ys, relative_zs):
        volume[rel_x, rel_y, rel_z] = 65534
    return volume


def ball_filter_imgs(
    volume: np.ndarray,
    threshold_value: int,
    soma_centre_value: int,
    ball_xy_size: int = 3,
    ball_z_size: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    # OPTIMISE: reuse ball filter instance

    good_tiles_mask = np.ones((1, 1, volume.shape[2]), dtype=bool)

    plane_width, plane_height = volume.shape[:2]

    bf = BallFilter(
        plane_width,
        plane_height,
        ball_xy_size,
        ball_z_size,
        overlap_fraction=0.8,
        tile_step_width=plane_width,
        tile_step_height=plane_height,
        threshold_value=threshold_value,
        soma_centre_value=soma_centre_value,
    )
    cell_detector = CellDetector(
        plane_width, plane_height, start_z=ball_z_size // 2
    )

    # FIXME: hard coded type
    ball_filtered_volume = np.zeros(volume.shape, dtype=np.uint16)
    previous_plane = None
    for z in range(volume.shape[2]):
        bf.append(volume[:, :, z].astype(np.uint16), good_tiles_mask[:, :, z])
        if bf.ready:
            bf.walk()
            middle_plane = bf.get_middle_plane()
            ball_filtered_volume[:, :, z] = middle_plane[:]
            # DEBUG: TEST: transpose
            previous_plane = cell_detector.process(
                middle_plane.copy(), previous_plane
            )
    return ball_filtered_volume, cell_detector.get_cell_centres()


def iterative_ball_filter(
    volume: np.ndarray, n_iter: int = 10
) -> Tuple[List[int], List[np.ndarray]]:
    ns = []
    centres = []

    threshold_value = 65534
    soma_centre_value = 65535

    vol = volume.copy()  # TODO: check if required

    for i in range(n_iter):
        vol, cell_centres = ball_filter_imgs(
            vol, threshold_value, soma_centre_value
        )
        vol -= 1
        n_structures = len(cell_centres)
        ns.append(n_structures)
        centres.append(cell_centres)
        if n_structures == 0:
            break
    return ns, centres


def check_centre_in_cuboid(centre: np.ndarray, max_coords: np.ndarray) -> bool:
    """
    Checks whether a coordinate is in a cuboid
    :param centre: x,y,z coordinate
    :param max_coords: far corner of cuboid
    :return: True if within cuboid, otherwise False
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
    cell_points: np.ndarray, outlier_keep: bool = False
) -> np.ndarray:
    orig_centre = get_structure_centre(cell_points)

    xs = cell_points[:, 0]
    ys = cell_points[:, 1]
    zs = cell_points[:, 2]

    orig_corner = np.array(
        [
            orig_centre[0] - (orig_centre[0] - xs.min()),
            orig_centre[1] - (orig_centre[1] - ys.min()),
            orig_centre[2] - (orig_centre[2] - zs.min()),
        ]
    )

    relative_orig_centre = np.array(
        [
            orig_centre[0] - orig_corner[0],
            orig_centre[1] - orig_corner[1],
            orig_centre[2] - orig_corner[2],
        ]
    )

    original_bounding_cuboid_shape = get_shape(xs, ys, zs)

    ball_radius = 1
    vol = coords_to_volume(xs, ys, zs, ball_radius=ball_radius)

    # centres is a list of arrays of centres (1 array of centres per ball run)
    ns, centres = iterative_ball_filter(vol)
    ns.insert(0, 1)
    centres.insert(0, np.array([relative_orig_centre]))

    best_iteration = ns.index(max(ns))

    # TODO: put constraint on minimum centres distance ?
    relative_centres = centres[best_iteration]

    if not outlier_keep:
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
    # FIXME: extract functionality
    absolute_centres[:, 0] = orig_corner[0] + relative_centres[:, 0]
    absolute_centres[:, 1] = orig_corner[1] + relative_centres[:, 1]
    absolute_centres[:, 2] = orig_corner[2] + relative_centres[:, 2]

    return absolute_centres
