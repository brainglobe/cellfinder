import pytest
import random

import numpy as np
from pathlib import Path

import cellfinder.tools.tools as tools

data_dir = Path("tests", "data")
jabberwocky = data_dir / "general" / "jabberwocky.txt"


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


def test_is_even():
    even_number = random.randrange(2, 1000, 2)
    odd_number = random.randrange(1, 1001, 2)
    with pytest.raises(NotImplementedError):
        assert tools.is_even(0)
    assert tools.is_even(even_number)
    assert not tools.is_even(odd_number)


def test_scale_to_16_bits():
    assert (validate_2d_img == tools.scale_to_16_bits(test_2d_img)).all()


def test_scale_to_16_bits():
    validate_2d_img_uint16 = validate_2d_img.astype(np.uint16, copy=False)
    assert (
        validate_2d_img_uint16
        == tools.scale_and_convert_to_16_bits(test_2d_img)
    ).all()


def test_unique_elements_list():
    list_in = [1, 2, 2, "a", "b", 1, "a", "dog"]
    unique_list = [1, 2, "a", "b", "dog"]
    assert tools.unique_elements_lists(list_in) == unique_list


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


def test_convert_shape_dict_to_array_shape():
    shape_dict = {"x": "100", "y": "30"}
    assert tools.convert_shape_dict_to_array_shape(shape_dict) == (30, 100)
    assert tools.convert_shape_dict_to_array_shape(
        shape_dict, type="fiji"
    ) == (100, 30)

    shape_dict["z"] = 10
    assert tools.convert_shape_dict_to_array_shape(shape_dict) == (30, 100, 10)

    with pytest.raises(NotImplementedError):
        tools.convert_shape_dict_to_array_shape(shape_dict, type="new type")


class Paths:
    def __init__(self, directory):
        self.one = directory / "one.aaa"
        self.two = directory / "two.bbb"
        self.tmp__three = directory / "three.ccc"
        self.tmp__four = directory / "four.ddd"


def test_delete_tmp(tmpdir):
    tmpdir = Path(tmpdir)
    paths = Paths(tmpdir)
    for attr, path in paths.__dict__.items():
        path.touch()
        print(path)
    assert len([child for child in tmpdir.iterdir()]) == 4
    tools.delete_temp(tmpdir, paths)
    assert len([child for child in tmpdir.iterdir()]) == 2

    tools.delete_temp(tmpdir, paths)


def test_is_any_list_overlap():
    assert tools.is_any_list_overlap(a, b)
    assert not tools.is_any_list_overlap(a, [2, "b", (1, 2, 3)])


def test_get_text_lines():
    line_5 = "The jaws that bite, the claws that catch!"
    first_line_alphabetically = "All mimsy were the borogoves,"
    assert tools.get_text_lines(jabberwocky, return_lines=5) == line_5
    assert (
        tools.get_text_lines(
            jabberwocky, return_lines=6, remove_empty_lines=False
        )
        == line_5
    )
    assert (
        tools.get_text_lines(jabberwocky, sort=True)[0]
        == first_line_alphabetically
    )
