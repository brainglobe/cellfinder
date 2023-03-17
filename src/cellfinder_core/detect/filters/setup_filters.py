import math
from typing import Tuple

import numpy as np

from cellfinder_core.detect.filters.volume.ball_filter import BallFilter
from cellfinder_core.detect.filters.volume.structure_detection import (
    CellDetector,
)
from cellfinder_core.tools.tools import get_max_value


def get_ball_filter(
    *,
    plane: np.ndarray,
    soma_diameter: int,
    ball_xy_size: int,
    ball_z_size: int,
    ball_overlap_fraction: float = 0.6,
) -> BallFilter:
    max_value = get_max_value(plane)
    thrsh_val = max_value - 1
    soma_centre_val = max_value

    tile_width = soma_diameter * 2
    layer_height, layer_width = plane.shape

    ball_filter = BallFilter(
        layer_width,
        layer_height,
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
    layer_height, layer_width = plane_shape
    start_z = z_offset + int(math.floor(ball_z_size / 2))
    return CellDetector(layer_width, layer_height, start_z=start_z)


def setup_tile_filtering(plane: np.ndarray) -> Tuple[int, int]:
    max_value = get_max_value(plane)
    clipping_value = max_value - 2
    thrsh_val = max_value - 1

    return clipping_value, thrsh_val
