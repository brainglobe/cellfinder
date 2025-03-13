from unittest.mock import patch

import numpy as np
import pytest

import cellfinder.core.tools.tools as tools

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


def test_inference_wrapper():
    did_run = False

    @tools.inference_wrapper
    def my_func(val, other=None):
        import torch

        nonlocal did_run

        assert torch.is_inference_mode_enabled()
        assert val == 1
        assert other == 5
        did_run = True

    my_func(1, other=5)
    assert did_run


# for float, the values come from the mantissa and it's the largest int value
# representable without losing significant digits
@pytest.mark.parametrize(
    "dtype,value",
    [
        (np.uint8, 2**8 - 1),
        (np.uint16, 2**16 - 1),
        (np.uint32, 2**32 - 1),
        (np.uint64, 2**64 - 1),
        (np.int8, 2**7 - 1),
        (np.int16, 2**15 - 1),
        (np.int32, 2**31 - 1),
        (np.int64, 2**63 - 1),
        (np.float32, 2**23),
        (np.float64, 2**52),
    ],
)
def test_get_max_possible_int_value(dtype, value):
    assert tools.get_max_possible_int_value(dtype) == value


def test_get_max_possible_int_value_bad_dtype():
    with pytest.raises(ValueError):
        tools.get_max_possible_int_value(np.str_)


# for float, the values come from the mantissa and it's the largest int value
# representable without losing significant digits
@pytest.mark.parametrize(
    "dtype,value",
    [
        (np.uint8, 0),
        (np.uint16, 0),
        (np.uint32, 0),
        (np.uint64, 0),
        (np.int8, -(2**7)),
        (np.int16, -(2**15)),
        (np.int32, -(2**31)),
        (np.int64, -(2**63)),
        (np.float32, -(2**23)),
        (np.float64, -(2**52)),
    ],
)
def test_get_min_possible_int_value(dtype, value):
    assert tools.get_min_possible_int_value(dtype) == value


def test_get_min_possible_int_value_bad_dtype():
    with pytest.raises(ValueError):
        tools.get_min_possible_int_value(np.str_)


@pytest.mark.parametrize(
    "src_dtype",
    [
        np.uint8,
        np.int8,
        np.uint16,
        np.int16,
        np.uint32,
        np.int32,
        np.uint64,
        np.int64,
        np.float32,
        np.float64,
    ],
)
@pytest.mark.parametrize(
    "dest_dtype", [np.uint8, np.uint16, np.uint32, np.uint64]
)
def test_get_data_converter_bad_dtype_target(src_dtype, dest_dtype):
    with pytest.raises(ValueError):
        tools.get_data_converter(src_dtype, dest_dtype)


@pytest.mark.parametrize(
    "src_dtype,dest_dtype",
    [
        (np.uint8, np.float32),
        (np.int8, np.float32),
        (np.uint16, np.float32),
        (np.int16, np.float32),
        (np.float32, np.float32),
        (np.uint8, np.float64),
        (np.int8, np.float64),
        (np.uint16, np.float64),
        (np.int16, np.float64),
        (np.uint32, np.float64),
        (np.int32, np.float64),
        (np.float32, np.float64),
        (np.float64, np.float64),
    ],
)
def test_get_data_converter_no_scaling(src_dtype, dest_dtype):
    # for these, the source is smaller than dest so no need to scale because
    # it'll fit directly into the dest dtype
    converter = tools.get_data_converter(src_dtype, dest_dtype)

    # min value
    if np.issubdtype(src_dtype, np.integer):
        src_min_val = np.iinfo(src_dtype).min
    else:
        assert np.issubdtype(src_dtype, np.floating)
        src_min_val = np.finfo(src_dtype).min
    src = np.full(5, src_min_val, dtype=src_dtype)
    dest = converter(src)
    assert np.array_equal(src, dest)
    assert dest.dtype == dest_dtype

    # other value
    src = np.full(5, 10, dtype=src_dtype)
    dest = converter(src)
    assert np.array_equal(src, dest)
    assert dest.dtype == dest_dtype

    # max value
    src_max_val = tools.get_max_possible_int_value(src_dtype)
    src = np.full(3, src_max_val, dtype=src_dtype)
    dest = converter(src)
    assert np.array_equal(src, dest)
    assert dest.dtype == dest_dtype


@pytest.mark.parametrize(
    "src_dtype,dest_dtype,divisor",
    [
        (np.uint32, np.float32, (2**32 - 1) / 2**23),
        (np.int32, np.float32, (2**31) / 2**23),
        (np.float64, np.float32, (2**52) / 2**23),
    ],
)
def test_get_data_converter_with_scaling(src_dtype, dest_dtype, divisor):
    # for these, the source is larger than dest type so we need to scale by max
    # value of each type, so it'll fit into the dest dtype
    converter = tools.get_data_converter(src_dtype, dest_dtype)

    # min value
    src_min_val = tools.get_min_possible_int_value(src_dtype)
    src = np.full(5, src_min_val, dtype=src_dtype)
    dest = converter(src)
    assert np.allclose(src / divisor, dest)
    assert dest.dtype == dest_dtype

    # other value
    src = np.full(5, 10, dtype=src_dtype)
    dest = converter(src)
    assert np.allclose(src / divisor, dest)
    assert dest.dtype == dest_dtype

    # max value
    src_max_val = tools.get_max_possible_int_value(src_dtype)
    src = np.full(3, src_max_val, dtype=src_dtype)
    dest = converter(src)
    assert np.allclose(src / divisor, dest)
    assert dest.dtype == dest_dtype


@pytest.mark.parametrize(
    "src_dtype,dest_dtype",
    [
        (np.uint64, np.float32),
        (np.int64, np.float32),
        (np.uint64, np.float64),
        (np.int64, np.float64),
    ],
)
def test_get_data_converter_with_bad_scaling(src_dtype, dest_dtype):
    # for these, to scale we'd need to have a type that is at least 64 bits,
    # so we can scale down. But float64 is too small
    with pytest.raises(ValueError):
        tools.get_data_converter(src_dtype, dest_dtype)


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

