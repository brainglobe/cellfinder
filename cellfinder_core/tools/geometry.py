import numpy as np


def make_sphere(ball_shape, radius, position):
    """
    Assumes shape and position are both a 3-tuple of int or float
    the units are pixels / voxels (px for short)
    radius is a int or float in px

    :param tuple(int) ball_shape:
    :param float radius:
    :param tuple(int) position:
    :return:
    :rtype: np.ndarray
    """

    half_sizes = (radius,) * 3

    # generate the grid for the support points
    # centered at the position indicated by position
    grid = [slice(-x0, dim - x0) for x0, dim in zip(position, ball_shape)]
    position = np.ogrid[grid]
    # calculate the distance of all points from `position` center
    # scaled by the radius
    arr = np.zeros(ball_shape, dtype=float)
    for x_i, half_size in zip(position, half_sizes):
        arr += np.abs(x_i / half_size) ** 2
    # the inner part of the sphere will have distance below 1
    return arr <= 1.0


def four_connected_kernel():
    return np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.bool)
