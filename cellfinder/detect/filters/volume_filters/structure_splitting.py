import numpy as np
import logging

from cellfinder.detect.filters.volume_filters.ball_filter import BallFilter
from cellfinder.detect.filters.volume_filters.structure_detection import (
    CellDetector,
    get_structure_centre_wrapper,
)


class StructureSplitException(Exception):
    pass


# def points_to_coords(cell_points):
#     xs = cell_points[:, 0]
#     ys = cell_points[:, 1]
#     zs = cell_points[:, 2]
#     return xs, ys, zs
#
#
# def file_to_coords(file_path):
#     cell_points = np.load(file_path)
#     return points_to_coords(cell_points)


def get_shape(xs, ys, zs):
    # +1 because difference. TEST:
    shape = [int((dim.max() - dim.min()) + 1) for dim in (xs, ys, zs)]
    return shape


def coords_to_volume(xs, ys, zs, ball_radius=1):
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
    volume, threshold_value, soma_centre_value, ball_xy_size=3, ball_z_size=3
):
    # OPTIMISE: reuse ball filter instance

    good_tiles_mask = np.ones((1, 1, volume.shape[2]), dtype=np.uint8)

    layer_width, layer_height = volume.shape[:2]

    bf = BallFilter(
        layer_width,
        layer_height,
        ball_xy_size,
        ball_z_size,
        overlap_fraction=0.8,
        tile_step_width=layer_width,
        tile_step_height=layer_height,
        threshold_value=threshold_value,
        soma_centre_value=soma_centre_value,
    )
    cell_detector = CellDetector(
        layer_width, layer_height, start_z=ball_z_size // 2
    )

    # FIXME: hard coded type
    ball_filtered_volume = np.zeros(volume.shape, dtype=np.uint16)
    for z in range(volume.shape[2]):
        bf.append(volume[:, :, z].astype(np.uint16), good_tiles_mask[:, :, z])
        if bf.ready:
            bf.walk()
            middle_plane = bf.get_middle_plane()
            ball_filtered_volume[:, :, z] = middle_plane[:]
            # DEBUG: TEST: transpose
            cell_detector.process(middle_plane.copy())
    return ball_filtered_volume, cell_detector.get_cell_centres()


def iterative_ball_filter(volume, n_iter=10):
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


def check_centre_in_cuboid(centre, max_coords):
    """
    Checks whether a coordinate is in a cuboid
    :param centre: x,y,z coordinate
    :param max_coords: far corner of cuboid
    :return: True if within cuboid, otherwise False
    """
    relative_coords = np.array([centre[k] for k in ("x", "y", "z")])
    if (relative_coords > max_coords).all():
        logging.info(
            'Relative coordinates "{}" exceed maximum volume '
            'dimension of "{}"'.format(relative_coords, max_coords)
        )
        return False
    else:
        return True


def split_cells(cell_points, relative=False, outlier_keep=False):
    orig_centre = get_structure_centre_wrapper(cell_points)

    xs = np.array([p["x"] for p in cell_points])  # TODO: use dataframe
    ys = np.array([p["y"] for p in cell_points])
    zs = np.array([p["z"] for p in cell_points])

    orig_corner = {
        "x": orig_centre["x"] - (orig_centre["x"] - xs.min()),
        "y": orig_centre["y"] - (orig_centre["y"] - ys.min()),
        "z": orig_centre["z"] - (orig_centre["z"] - zs.min()),
    }
    relative_orig_centre = {
        k: orig_centre[k] - orig_corner[k] for k in ("x", "y", "z")
    }

    original_bounding_cuboid_shape = get_shape(xs, ys, zs)

    ball_radius = 1
    vol = coords_to_volume(xs, ys, zs, ball_radius=ball_radius)

    # centres is a list of lists of centres (1 list of centres per ball run)
    ns, centres = iterative_ball_filter(vol)
    ns.insert(0, 1)
    centres.insert(0, [relative_orig_centre])

    best_iteration = ns.index(max(ns))

    # TODO: put constraint on minimum centres distance ?
    relative_centres = centres[best_iteration]

    if not outlier_keep:
        # TODO: change to checking whether in original cluster shape
        original_max_coords = np.array(original_bounding_cuboid_shape)
        relative_centres = [
            x
            for x in relative_centres
            if check_centre_in_cuboid(x, original_max_coords)
        ]

    if relative:
        return relative_centres
    else:
        absolute_centres = []
        # FIXME: extract functionality
        for relative_centre in relative_centres:
            absolute_centre = {
                "x": orig_corner["x"] + relative_centre["x"],
                "y": orig_corner["y"] + relative_centre["y"],
                "z": orig_corner["z"] + relative_centre["z"],
            }
            absolute_centres.append(absolute_centre)

        return absolute_centres
