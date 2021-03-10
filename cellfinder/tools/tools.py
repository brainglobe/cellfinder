from natsort import natsorted


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


def is_any_list_overlap(list_a, list_b):
    """
    Is there any overlap between two lists
    :param list_a: Any list
    :param list_b: Any other list
    :return: True if lists have shared elements
    """
    return any({*list_a} & {*list_b})
