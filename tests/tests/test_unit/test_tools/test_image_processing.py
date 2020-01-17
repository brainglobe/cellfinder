import random

import numpy as np

from cellfinder.tools import image_processing as img_tools


def test_crop_center_2d():
    x_shape = random.randint(2, 100)
    y_shape = random.randint(2, 100)
    img = np.random.rand(y_shape, x_shape)
    assert (
        img == img_tools.crop_center_2d(img, crop_x=x_shape, crop_y=y_shape)
    ).all()

    new_x_shape = random.randint(1, x_shape)
    new_y_shape = random.randint(1, y_shape)
    pad_img = img_tools.crop_center_2d(
        img, crop_x=new_x_shape, crop_y=new_y_shape
    )
    assert (new_y_shape, new_x_shape) == pad_img.shape


def test_pad_centre_2d():
    x_shape = random.randint(2, 100)
    y_shape = random.randint(2, 100)
    img = np.random.rand(y_shape, x_shape)
    assert (
        img == img_tools.pad_center_2d(img, x_size=x_shape, y_size=y_shape)
    ).all()

    new_x_shape = random.randint(x_shape, x_shape * 10)
    new_y_shape = random.randint(y_shape, y_shape * 10)
    pad_img = img_tools.pad_center_2d(
        img, x_size=new_x_shape, y_size=new_y_shape
    )
    assert (new_y_shape, new_x_shape) == pad_img.shape
