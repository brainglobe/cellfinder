from numbers import Number

import numpy as np


def make_sphere(
    ball_shape: tuple[int, int, int],
    radius: tuple[float, float, float] | float,
    position: tuple[float, float, float],
) -> np.ndarray:
    """
    Return a boolean array, with array elements inside a sphere set
    to 1 and those outside set to 0.

    Parameters
    ----------
    ball_shape :
        Shape of the output array.
    radius :
        Radius of the sphere, either single radius for sphere or 3d radius for
        spheroid.
    position :
        Centre of the sphere (can be between voxels).
    """
    if isinstance(radius, Number):
        radius = (radius,) * 3

    # generate the grid for the support points
    grid = [slice(dim) for dim in ball_shape]
    meshedgrid = np.ogrid[grid]
    # calculate the distance of all points from `position` center
    # proportional of the radius so 1 would mean at the radius
    arr = np.zeros(ball_shape, dtype=float)
    for x_i, centre_i, radius_i in zip(meshedgrid, position, radius):
        arr += ((x_i - centre_i) / radius_i) ** 2

    # the inner part of the sphere will have distance below 1 b/c pythagoras
    return arr <= 1.0


def four_connected_kernel():
    return np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
