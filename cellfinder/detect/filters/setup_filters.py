import math

from tifffile import tifffile

from cellfinder.detect.filters.volume_filters.ball_filter import BallFilter
from cellfinder.detect.filters.volume_filters.structure_detection import (
    CellDetector,
)
from cellfinder.tools.tools import get_max_value


def setup(
    first_img_path,
    soma_diameter,
    ball_xy_size,
    ball_z_size,
    ball_overlap_fraction=0.6,
    z_offset=0,
):
    plane = tifffile.imread(first_img_path)
    plane = plane.T

    max_value = get_max_value(plane)
    clipping_value = max_value - 2
    thrsh_val = max_value - 1
    soma_centre_val = max_value

    tile_width = soma_diameter * 2
    layer_width, layer_height = plane.shape

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
    start_z = z_offset + int(math.floor(ball_z_size / 2))
    cell_detector = CellDetector(layer_width, layer_height, start_z=start_z)

    return clipping_value, thrsh_val, ball_filter, cell_detector
