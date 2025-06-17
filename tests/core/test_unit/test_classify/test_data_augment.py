import math

import pytest
import torch

from cellfinder.core.classify.augment import DataAugmentation


@pytest.fixture
def cube_with_side_dot() -> torch.Tensor:
    # DataAugmentation.DIM_ORDER of (c, y, x, z)
    data = torch.zeros((2, 11, 11, 7))
    # put dot at pixel in plane z = 5, on the x/y diagonal - both x and y are 8
    data[0, 8, 8, 5] = 1
    # put dot at pixel in plane z = 5, but x is 2 and y is 8
    data[1, 8, 2, 5] = 1
    return data


@pytest.fixture
def cube_with_center_dot() -> torch.Tensor:
    # DataAugmentation.DIM_ORDER of (c, y, x, z)
    data = torch.zeros((2, 11, 11, 7))
    data[:, 5, 5, 3] = 1
    return data


def test_augment_translate(cube_with_side_dot):
    c, y, x, z = cube_with_side_dot.shape
    translate_range = [(1 / 11, 1 / 11), (2 / 11, 2 / 11), (1 / 7, 1 / 7)]
    augmenter = DataAugmentation(
        volume_size={"x": x, "y": y, "z": z},
        augment_likelihood=1,
        translate_range=translate_range,
        data_dim_order=("c", "y", "x", "z"),
    )
    assert augmenter.update_parameters(), "Parameters should be randomized"
    augmented = augmenter(cube_with_side_dot)

    assert augmented.shape == cube_with_side_dot.shape
    assert augmented[0, 8, 8, 5] < 0.1
    assert augmented[1, 8, 2, 5] < 0.1
    assert augmented[0, 8 + 1, 8 + 2, 5 + 1] > 0.1
    assert augmented[1, 8 + 1, 2 + 2, 5 + 1] > 0.1


def test_augment_rotate(cube_with_side_dot):
    c, y, x, z = cube_with_side_dot.shape
    augmenter = DataAugmentation(
        volume_size={"x": x, "y": y, "z": z},
        augment_likelihood=1,
        rotate_range=[(0, 0), (0, 0), (math.pi / 2, math.pi / 2)],
        data_dim_order=("c", "y", "x", "z"),
    )
    assert augmenter.update_parameters(), "Parameters should be randomized"
    augmented = augmenter(cube_with_side_dot)

    assert augmented.shape == cube_with_side_dot.shape
    assert augmented[0, 8, 8, 5] < 0.1
    assert augmented[1, 8, 2, 5] < 0.1
    # we rotated around z axis by 45 degree. So x,y point at 8, just reflects
    # around x axis so y is still 8 (3 from end), but x becomes 10 - 8
    # (len = 11)
    assert augmented[0, 8, 10 - 8, 5] > 0.1
    # for center rotation of 45 degree from x = 2, y = 8 we end up with
    # x = -8 (10 - 8 = 2) and y is 2
    assert augmented[1, 2, 10 - 8, 5] > 0.1


def test_augment_scale(cube_with_center_dot):
    c, y, x, z = cube_with_center_dot.shape
    augmenter = DataAugmentation(
        volume_size={"x": x, "y": y, "z": z},
        augment_likelihood=1,
        scale_range=((3, 3),) * 3,
        data_dim_order=("c", "y", "x", "z"),
    )
    assert augmenter.update_parameters(), "Parameters should be randomized"
    augmented = augmenter(cube_with_center_dot)

    assert augmented.shape == cube_with_center_dot.shape
    assert augmented[0, 5, 5, 3] > 0.1
    assert augmented[0, 5 + 1, 5, 3] > 0.1
    assert augmented[0, 5, 5 + 1, 3] > 0.1
    assert augmented[0, 5, 5, 3 + 1] > 0.1

    assert augmented[1, 5, 5, 3] > 0.1
    assert augmented[1, 5 + 1, 5, 3] > 0.1
    assert augmented[1, 5, 5 + 1, 3] > 0.1
    assert augmented[1, 5, 5, 3 + 1] > 0.1


def test_augment_axis_flip(cube_with_side_dot):
    c, y, x, z = cube_with_side_dot.shape
    augmenter = DataAugmentation(
        volume_size={"x": x, "y": y, "z": z},
        augment_likelihood=1,
        flippable_axis=(0, 1),
        data_dim_order=("c", "y", "x", "z"),
    )
    assert augmenter.update_parameters(), "Parameters should be randomized"
    augmented = augmenter(cube_with_side_dot)

    assert augmented.shape == cube_with_side_dot.shape
    assert augmented[0, 8, 8, 5] < 0.1
    assert augmented[1, 8, 2, 5] < 0.1
    assert augmented[0, 2, 2, 5] > 0.1
    assert augmented[1, 2, 8, 5] > 0.1
