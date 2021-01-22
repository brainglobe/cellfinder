import pytest
import random

import numpy as np

import cellfinder_core.tools.tools as tools


a = [1, "a", 10, 30]
b = [30, 10, "c", "d"]

test_2d_img = np.array([[1, 2, 10, 100], [5, 25, 300, 1000], [1, 0, 0, 125]])
validate_2d_img = np.array(
    [
        [65.535, 131.07, 655.35, 6553.5],
        [327.675, 1638.375, 19660.5, 65535],
        [65.535, 0, 0, 8191.875],
    ]
)


def test_get_max_value():
    num = random.randint(0, 100)
    assert 255 == tools.get_max_value(np.array(num, dtype=np.uint8))
    assert 65535 == tools.get_max_value(np.array(num, dtype=np.uint16))


def test_union():
    ab = [1, "a", 10, 30, "c", "d"]
    assert set(ab) == set(tools.union(a, b))


def test_check_unique_list():
    assert (True, []) == tools.check_unique_list(a)
    repeating_list = [1, 2, 3, 3, "dog", "cat", "dog"]
    assert (False, [3, "dog"]) == tools.check_unique_list(repeating_list)


def test_common_member():
    assert (True, [10, 30]) == tools.common_member(a, b)


def test_get_number_of_bins_nd():
    array_tuple = (100, 1000, 5000)
    assert tools.get_number_of_bins_nd(array_tuple, 12) == (8, 83, 416)

    shape_dict = {"x": 20, "y": 200}
    assert tools.get_number_of_bins_nd(shape_dict, 8) == (2, 25)

    with pytest.raises(NotImplementedError):
        tools.get_number_of_bins_nd([200, 300, 400], 10)


def test_interchange_np_fiji_coordinates():
    tuple_in = (100, 200, 300)
    assert tools.interchange_np_fiji_coordinates(tuple_in) == (200, 100, 300)


def test_swap_elements_list():
    list_in = [1, 2, 3, 4]
    assert tools.swap_elements_list(list_in, 1, 2) == [1, 3, 2, 4]


def test_is_any_list_overlap():
    assert tools.is_any_list_overlap(a, b)
    assert not tools.is_any_list_overlap(a, [2, "b", (1, 2, 3)])
