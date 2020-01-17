import logging
import os
import numpy as np

from natsort import natsorted
from random import getrandbits, uniform

from cellfinder.tools import system


def get_max_value(obj_in):
    """
    Returns the maximum allowed value for a specific object type.
    :param obj_in: Object
    :return int: Maximum value of object type
    """
    # TODO: Generalise, and not rely on parsing a string
    if obj_in.dtype == "uint8":
        return 255
    else:
        return 2 ** int(str(obj_in.dtype)[-2:]) - 1


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


def is_even(num):
    """
    Returns True if a number is even
    :param num:
    :return:
    """
    if num == 0:
        raise NotImplementedError(
            "Input number is 0. Evenness of 0 is not defined by this "
            "function."
        )
    if num % 2:
        return False
    else:
        return True


def scale_to_16_bits(img):
    """
    Normalise the input image to the full 0-2^16 bit depth.

    :param np.array img: The input image
    :return: The normalised image
    :rtype: np.array
    """
    normalised = img / img.max()
    return normalised * (2 ** 16 - 1)


def scale_and_convert_to_16_bits(img):
    """
    Normalise the input image to the full 0-2^16 bit depth, and return as
    type: "np.uint16".


    :param np.array img: The input image
    :return: The normalised, 16 bit image
    :rtype: np.array
    """
    img = scale_to_16_bits(img)
    return img.astype(np.uint16, copy=False)


def remove_empty_string_list(str_list):
    """
    Removes any empty strings from a list of strings
    :param str_list: List of strings
    :return: List of strings without the empty strings
    """
    return list(filter(None, str_list))


def unique_elements_lists(list_in):
    """ return the unique elements in a list"""
    return list(dict.fromkeys(list_in))


def get_number_of_bins_nd(array_size, binning):
    """
    Generate the number of bins needed in three dimensions, based on the size
    of the array, and the binning.
    :param array_size: Size of the image array (tuple or dict)
    :param binning: How many pixels in each edge of nD equal sized bins
    :return:
    """
    if type(array_size) is tuple:
        bins = [int(size / binning) for size in array_size]
    elif type(array_size) is dict:
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


def convert_shape_dict_to_array_shape(shape_dict, type="numpy"):
    """
    Converts a dict with "x", "y" (and optionally "z") attributes into
    a tuple that can be used to e.g. initialise a numpy array
    :param shape_dict: Dict with  "x", "y" (and optionally "z") attributes
    :param type: One of "numpy" or "fiji", to determine whether the "x" or the
    "y" attribute is the first dimension.
    :return: Tuple array shape
    """

    shape = []
    if type is "numpy":
        shape.append(int(shape_dict["y"]))
        shape.append(int(shape_dict["x"]))

    elif type is "fiji":
        shape.append(int(shape_dict["x"]))
        shape.append(int(shape_dict["y"]))
    else:
        raise NotImplementedError(
            "Type: {} not recognise, please specify "
            "'numpy' or 'fiji'".format(type)
        )
    if "z" in shape_dict:
        shape.append(int(shape_dict["z"]))

    return tuple(shape)


def delete_temp(directory, paths, prefix="tmp__"):
    """
    Removes all temp files (properties of an object starting with "tmp__")
    :param directory: Directory to delete tmp files from
    :param paths: cellfinder.prep.Paths object with temp paths.
    :param prefix: String that temporary files (to be deleted) begin with.
    """
    for path_name, path in paths.__dict__.items():
        if path_name.startswith(prefix):
            if system.check_path_in_dir(path, directory):
                try:
                    os.remove(path)
                except FileNotFoundError:
                    logging.warning(
                        f"File: {path} not found, not deleting. "
                        f"Proceeding anyway."
                    )


def is_any_list_overlap(list_a, list_b):
    """
    Is there any overlap between two lists
    :param list_a: Any list
    :param list_b: Any other list
    :return: True if lists have shared elements
    """
    return any({*list_a} & {*list_b})


def get_text_lines(
    file,
    return_lines=None,
    rstrip=True,
    sort=False,
    remove_empty_lines=True,
    encoding=None,
):
    """
    Return only the nth line of a text file
    :param file: Any text file
    :param return_lines: Which specific line/lines to read
    :param rstrip: Remove trailing characters
    :param sort: If true, naturally sort the data
    :param remove_empty_lines: If True, ignore empty lines
    :param encoding: What encoding the text file has.
    Default: None (platform dependent)
    :return: The nth line
    """
    with open(file, encoding=encoding) as f:
        lines = f.readlines()
    if rstrip:
        lines = [line.strip() for line in lines]
    if remove_empty_lines:
        lines = remove_empty_string_list(lines)
    if sort:
        lines = natsorted(lines)
    if return_lines is not None:
        lines = lines[return_lines]
    return lines


def random_bool(likelihood=None):
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


def random_sign():
    """
    Returns a random sign (-1 or 1) with a 50/50 chance of each.
    :return: Random sign (-1 or 1)
    """
    if random_bool():
        return 1
    else:
        return -1


def random_probability():
    """
    Return a random probability in the range (0, 1)
    :return: Random probability in the range (0, 1)
    """
    return uniform(0, 1)


def all_elements_equal(x):
    """
    Return True is all the elements in a series are equal
    :param x: Series of values (e.g. list or numpy array)
    :return: True if all elements are equal, False otherwise.
    """
    return len(set(x)) <= 1


def suppress_specific_logs(logger, message):
    logger = logging.getLogger(logger)

    class NoParsingFilter(logging.Filter):
        def filter(self, record):
            return not record.getMessage().startswith(message)

    logger.addFilter(NoParsingFilter())
