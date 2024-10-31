from functools import wraps
from random import getrandbits, uniform
from typing import Callable, Optional, Type

import numpy as np
import torch
from natsort import natsorted


def inference_wrapper(func):
    """
    Decorator that makes the decorated function/method run with
    `torch.inference_mode` set to True.
    """

    @wraps(func)
    def inner_function(*args, **kwargs):
        with torch.inference_mode(True):
            return func(*args, **kwargs)

    return inner_function


def get_max_possible_int_value(dtype: Type[np.number]) -> int:
    """
    Returns the maximum allowed integer for a numpy array of given type.

    If dtype is of integer type, it's the maximum value. If it's a floating
    type, it's the maximum integer that can be accurately represented.
    E.g. for float32, only integers up to 2**24 can be represented (due to
    the number of bits representing the mantissa (significand).
    """
    if np.issubdtype(dtype, np.integer):
        return np.iinfo(dtype).max
    if np.issubdtype(dtype, np.floating):
        mant = np.finfo(dtype).nmant
        return 2**mant
    raise ValueError("datatype must be of integer or floating data type")


def get_min_possible_int_value(dtype: Type[np.number]) -> int:
    """
    Returns the minimum allowed integer for a numpy array of given type.

    If dtype is of integer type, it's the minimum value. If it's a floating
    type, it's the minimum integer that can be accurately represented.
    E.g. for float32, only integers up to -2**24 can be represented (due to
    the number of bits representing the mantissa (significand).
    """
    if np.issubdtype(dtype, np.integer):
        return np.iinfo(dtype).min
    if np.issubdtype(dtype, np.floating):
        mant = np.finfo(dtype).nmant
        # the sign bit is separate so we have the full mantissa for value
        return -(2**mant)
    raise ValueError("datatype must be of integer or floating data type")


def get_data_converter(
    src_dtype: Type[np.number], dest_dtype: Type[np.floating]
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Returns a function that can be called to convert one data-type to another,
    scaling the data down as needed.

    If the maximum value supported by the input data-type is smaller than that
    supported by destination data-type, the data will be scaled by the ratio
    of maximum integer representable by the `output / input` data-types.
    If the max is equal or less, it's simply converted to the target type.

    Parameters
    ----------
    src_dtype : np.dtype
        The data-type of the input data.
    dest_dtype : np.dtype
        The data-type of the returned data. Currently, it must be a floating
        type and `np.float32` or `np.float64`.

    Returns
    -------
    callable: function
        A function that takes a single input data parameter and returns
        the converted data.
    """
    if not np.issubdtype(dest_dtype, np.float32) and not np.issubdtype(
        dest_dtype, np.float64
    ):
        raise ValueError(
            f"Destination dtype must be a float32 or float64, "
            f"but it is {dest_dtype}"
        )

    in_min = get_min_possible_int_value(src_dtype)
    in_max = get_max_possible_int_value(src_dtype)
    out_min = get_min_possible_int_value(dest_dtype)
    out_max = get_max_possible_int_value(dest_dtype)
    in_abs_max = max(in_max, abs(in_min))
    out_abs_max = max(out_max, abs(out_min))

    def unchanged(data: np.ndarray) -> np.ndarray:
        return np.asarray(data)

    def float_to_float_scale_down(data: np.ndarray) -> np.ndarray:
        return ((np.asarray(data) / in_abs_max) * out_abs_max).astype(
            dest_dtype
        )

    def int_to_float_scale_down(data: np.ndarray) -> np.ndarray:
        # data must fit in float64
        data = np.asarray(data).astype(np.float64)
        return ((data / in_abs_max) * out_abs_max).astype(dest_dtype)

    def to_float_unscaled(data: np.ndarray) -> np.ndarray:
        return np.asarray(data).astype(dest_dtype)

    if src_dtype == dest_dtype:
        return unchanged

    # out can hold the largest in values - just convert to float
    if out_min <= in_min < in_max <= out_max:
        return to_float_unscaled

    # need to scale down before converting to float
    if np.issubdtype(src_dtype, np.integer):
        # if going to float32 and it didn't fit input must fit in 64-bit float
        # so we can temp store it there to scale. If going to float64, same.
        if in_max > get_max_possible_int_value(
            np.float64
        ) or in_min < get_min_possible_int_value(np.float64):
            raise ValueError(
                f"The input datatype {src_dtype} cannot fit in a "
                f"64-bit float"
            )
        return int_to_float_scale_down

    # for float input, however big it is, we can always scale it down in the
    # input data type before changing type
    return float_to_float_scale_down


def union(a, b):
    """
    Return the union (elements in both) of two lists
    :param list a:
    :param list b:
    :return: Union of the two lists
    """
    return list(set(a) | set(b))


def check_unique_list(in_list, natural_sort=True):
    """
    Checks if all the items in a list are unique or not
    :param list in_list: Input list
    :param bool natural_sort: Sort the resulting items naturally
    (default: True)
    :return: True/False and a list of any repeated values
    """
    unique = set(in_list)
    repeated_items = []

    for item in unique:
        count = in_list.count(item)
        if count > 1:
            repeated_items.append(item)

    if repeated_items:
        if natural_sort:
            repeated_items = natsorted(repeated_items)
        return False, repeated_items
    else:
        return True, []


def common_member(a, b, natural_sort=True):
    """
    Checks if two lists (or sets) have a common member, and if so, returns
    the common members.
    :param a: First list (or set)
    :param b: Second list (or set)
    :param bool natural_sort: Sort the resulting items naturally
    (default: True)
    :return: True/False and the list of values
    """
    a_set = set(a)
    b_set = set(b)
    intersection = list(a_set.intersection(b_set))
    if len(intersection) > 0:
        result = True
    else:
        result = False

    if natural_sort:
        intersection = natsorted(intersection)

    return result, intersection


def get_number_of_bins_nd(array_size, binning):
    """
    Generate the number of bins needed in three dimensions, based on the size
    of the array, and the binning.
    :param array_size: Size of the image array (tuple or dict)
    :param binning: How many pixels in each edge of nD equal sized bins
    :return:
    """
    if isinstance(array_size, tuple):
        bins = [int(size / binning) for size in array_size]
    elif isinstance(array_size, dict):
        bins = [int(size / binning) for size in array_size.values()]
    else:
        raise NotImplementedError(
            "Variable type: {} is not supported. Please"
            "use tuple or dict".format(type(array_size))
        )
    return tuple(bins)


def interchange_np_fiji_coordinates(tuple_in):
    """
    Swaps the first and second element of a tuple to swap between the numpy
    convention (vertical is first) and FIJI convention (horizontal is first)
    :param tuple_in: A 2+ element tuple
    :return: Same tuple with element 0 and 1 interchanged
    """
    tmp_list = list(tuple_in)
    tmp_list = swap_elements_list(tmp_list, 0, 1)
    return tuple(tmp_list)


def swap_elements_list(list_in, swap_a, swap_b):
    """
    Swap two elements in a list
    :param list_in: A list
    :param swap_a: Index of first element to swap
    :param swap_b: Index of second element to swap
    :return: List with swap_a and swap_b exchanged
    """
    list_in[swap_a], list_in[swap_b] = list_in[swap_b], list_in[swap_a]
    return list_in


def is_any_list_overlap(list_a, list_b):
    """
    Is there any overlap between two lists
    :param list_a: Any list
    :param list_b: Any other list
    :return: True if lists have shared elements
    """
    return any({*list_a} & {*list_b})


def random_bool(likelihood: Optional[float] = None) -> bool:
    """
    Return a random boolean (True/False). If "likelihood" is not None, this
    is biased.
    :param likelihood: Only return True if a random number in the range (0,1)
    is greater than this. Default: None.
    :return: Random (or biased) boolean
    """
    if likelihood is None:
        return bool(getrandbits(1))
    else:
        if uniform(0, 1) > likelihood:
            return True
        else:
            return False


def random_sign() -> int:
    """
    Returns a random sign (-1 or 1) with a 50/50 chance of each.
    :return: Random sign (-1 or 1)
    """
    if random_bool():
        return 1
    else:
        return -1


def random_probability() -> float:
    """
    Return a random probability in the range (0, 1)
    :return: Random probability in the range (0, 1)
    """
    return uniform(0, 1)


def all_elements_equal(x) -> bool:
    """
    Return True is all the elements in a series are equal
    :param x: Series of values (e.g. list or numpy array)
    :return: True if all elements are equal, False otherwise.
    """
    return len(set(x)) <= 1
