import math
from typing import Tuple

import numpy as np

from cellfinder.core.detect.filters.volume.ball_filter import BallFilter
from cellfinder.core.detect.filters.volume.structure_detection import (
    CellDetector,
)
from cellfinder.core.tools.tools import get_max_possible_value


def get_ball_filter(
    *,
    plane: np.ndarray,
    soma_diameter: int,
    ball_xy_size: int,
    ball_z_size: int,
    ball_overlap_fraction: float = 0.6,
) -> BallFilter:
    # thrsh_val is used to clip the data in plane to make sure
    # a number is available to mark cells. soma_centre_val is the
    # number used to mark cells.
    max_value = get_max_possible_value(plane)
    thrsh_val = max_value - 1
    soma_centre_val = max_value

    tile_width = soma_diameter * 2
    plane_height, plane_width = plane.shape

    ball_filter = BallFilter(
        plane_width,
        plane_height,
        ball_xy_size,
        ball_z_size,
        overlap_fraction=ball_overlap_fraction,
        tile_step_width=tile_width,
        tile_step_height=tile_width,
        threshold_value=thrsh_val,
        soma_centre_value=soma_centre_val,
    )
    return ball_filter


def get_cell_detector(
    *, plane_shape: Tuple[int, int], ball_z_size: int, z_offset: int = 0
) -> CellDetector:
    plane_height, plane_width = plane_shape
    start_z = z_offset + int(math.floor(ball_z_size / 2))
    return CellDetector(plane_width, plane_height, start_z=start_z)


def setup_tile_filtering(plane: np.ndarray) -> Tuple[int, int]:
    """
    Setup values that are used to threshold the plane during 2D filtering.

    Returns
    -------
    clipping_value :
        Upper value used to clip planes before 2D filtering. This is chosen
        to leave two numbers left that can later be used to mark bright points
        during the 2D and 3D filtering stages.
    threshold_value :
        Value used to mark bright pixels after 2D filtering.
    """
    max_value = get_max_possible_value(plane)
    clipping_value = max_value - 2
    thrsh_val = max_value - 1

    return clipping_value, thrsh_val
