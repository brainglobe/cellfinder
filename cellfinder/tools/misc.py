import argparse


def check_positive_float(value, none_allowed=True):
    """
    Used in argparse to enforce positive floats
    FromL https://stackoverflow.com/questions/14117415
    :param value: Input value
    :param none_allowed: If false, throw an error for None values
    :return: Input value, if it's positive
    """
    ivalue = value
    if ivalue is not None:
        ivalue = float(ivalue)
        if ivalue < 0:
            raise argparse.ArgumentTypeError(
                "%s is an invalid positive value" % value
            )
    else:
        if not none_allowed:
            raise argparse.ArgumentTypeError("%s is an invalid value." % value)

    return ivalue


def check_positive_int(value, none_allowed=True):
    """
    Used in argparse to enforce positive ints
    FromL https://stackoverflow.com/questions/14117415
    :param value: Input value
    :param none_allowed: If false, throw an error for None values
    :return: Input value, if it's positive
    """
    ivalue = value
    if ivalue is not None:
        ivalue = int(ivalue)
        if ivalue < 0:
            raise argparse.ArgumentTypeError(
                "%s is an invalid positive value" % value
            )
    else:
        if not none_allowed:
            raise argparse.ArgumentTypeError("%s is an invalid value." % value)

    return ivalue
