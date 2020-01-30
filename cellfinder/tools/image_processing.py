import numpy as np

from imlib.general.numerical import is_even


def crop_center_2d(img, crop_x=None, crop_y=None):
    """
    Crops the centre of a 2D image, and returns a smaller array. If the desired
    dimension is larger than the original dimension, nothing is changed.
    :param img: 2D input image
    :param crop_x: New length in x (default: None, which does nothing)
    :param crop_y: New length in y (default: None, which does nothing)
    :return: New, smaller array
    """

    y, x = img.shape

    # TODO: simplify
    if crop_x is not None:
        if crop_x >= x:
            start_x = 0
            crop_x = x
        else:
            start_x = x // 2 - (crop_x // 2)
    else:
        start_x = 0
        crop_x = x

    if crop_y is not None:
        if crop_y >= y:
            start_y = 0
            crop_y = y
        else:
            start_y = y // 2 - (crop_y // 2)
    else:
        start_y = 0
        crop_y = y

    return img[start_y : start_y + crop_y, start_x : start_x + crop_x]


def pad_center_2d(img, x_size=None, y_size=None, pad_mode="edge"):
    """
    Pads the edges of a 2D image, and returns a larger image. If the desired
    dimension is smaller than the original dimension, nothing is changed.
    :param img: 2D input image
    :param x_size: New length in x (default: None, which does nothing)
    :param y_size: New length in y (default: None, which does nothing)
    :return: New, larger array
    """

    y, x = img.shape

    #  TODO: simplify

    if x_size is None:
        x_pad = 0
    elif x_size <= x:
        x_pad = 0
    else:
        x_pad = x_size - x

    if y_size is None:
        y_pad = 0
    elif y_size <= y:
        y_pad = 0
    else:
        y_pad = y_size - y

    if x_pad > 0:
        if is_even(x_pad):
            x_front = x_back = int(x_pad / 2)
        else:
            x_front = int(x_pad // 2)
            x_back = int(x_front + 1)
    else:
        x_front = x_back = 0

    if y_pad > 0:
        if is_even(y_pad):
            y_front = y_back = int(y_pad / 2)
        else:
            y_front = int(y_pad // 2)
            y_back = int(y_front + 1)
    else:
        y_front = y_back = 0

    return np.pad(img, ((y_front, y_back), (x_front, x_back)), pad_mode)
