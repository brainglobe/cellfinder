import pytest

import numpy as np

from cellfinder.detect.filters.volume_filters import structure_detection


def test_get_non_zero_ull_min():
    assert (
        structure_detection.get_non_zero_ull_min_wrapper(list(range(10))) == 1
    )
    assert (
        structure_detection.get_non_zero_ull_min_wrapper([0] * 10)
        == (2 ** 64) - 1
    )


class Point:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __str__(self):
        return "x: {}, y: {}, z: {}".format(self.x, self.y, self.z)


@pytest.fixture()
def three_d_cross():
    return np.array(
        [
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
            [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
        ]
    )


@pytest.fixture()
def structure(three_d_cross):
    coords = np.where(three_d_cross == 1)[::-1]
    s = [Point(*c) for c in zip(coords[0], coords[1], coords[2])]
    return s


def test_get_structure_centre(structure):
    result_point = structure_detection.get_structure_centre_wrapper(structure)
    assert (result_point["x"], result_point["y"], result_point["z"]) == (
        1,
        1,
        1,
    )
