from typing import Tuple

import numpy as np


def make_sphere(
    ball_shape: Tuple[int, int, int],
    radius: float,
    position: Tuple[int, int, int],
) -> np.ndarray:
    """
    Return a boolean array, with array elements inside a sphere set
    to 1 and those outside set to 0.

    Parameters
    ----------
    ball_shape :
        Shape of the output array.
    radius :
        Radius of the sphere.
    position :
        Centre of the sphere.
    """

    half_sizes = (radius,) * 3

    # generate the grid for the support points
    # centered at the position indicated by position
    grid = [slice(-x0, dim - x0) for x0, dim in zip(position, ball_shape)]
    meshedgrid = np.ogrid[grid]
    # calculate the distance of all points from `position` center
    # scaled by the radius
    arr = np.zeros(ball_shape, dtype=float)
    for x_i, half_size in zip(meshedgrid, half_sizes):
        arr += np.abs(x_i / half_size) ** 2
    # the inner part of the sphere will have distance below 1
    return arr <= 1.0


def four_connected_kernel():
    return np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
