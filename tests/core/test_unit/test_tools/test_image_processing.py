import numpy as np
import pytest

from cellfinder.core.tools import image_processing as img_tools


def test_crop_center_2d():
    x_shape = 7
    y_shape = 8
    img = np.arange(y_shape * x_shape).reshape(y_shape, x_shape)
    assert (
        img == img_tools.crop_center_2d(img, crop_x=x_shape, crop_y=y_shape)
    ).all()

    new_x_shape = 4
    new_y_shape = 5
    pad_img = img_tools.crop_center_2d(
        img, crop_x=new_x_shape, crop_y=new_y_shape
    )
    assert (new_y_shape, new_x_shape) == pad_img.shape


def test_pad_centre_2d():
    x_shape = 7
    y_shape = 8
    img = np.arange(y_shape * x_shape).reshape(y_shape, x_shape)
    assert (
        img == img_tools.pad_center_2d(img, x_size=x_shape, y_size=y_shape)
    ).all()

    # Exercise both even and odd padding widths consistently across runs.
    new_x_shape = x_shape + 3
    new_y_shape = y_shape + 2
    pad_img = img_tools.pad_center_2d(
        img, x_size=new_x_shape, y_size=new_y_shape
    )
    assert (new_y_shape, new_x_shape) == pad_img.shape


@pytest.mark.parametrize("progress", [True, False])
def test_dataset_mean_std(progress):
    # checks that dataset_mean_std correctly computes the std/mean
    data = np.random.normal(100, 10, (10, 10, 10))

    mean, std = img_tools.dataset_mean_std(
        data, n_sampling_planes=5, show_progress=progress
    )
    # give it enough room for estimation error
    assert 90 < mean < 110
    assert 8 < std < 12
