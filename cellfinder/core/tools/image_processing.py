import numpy as np
import tqdm
from brainglobe_utils.general.numerical import is_even

from cellfinder.core import types
from cellfinder.core.tools.tools import get_data_converter


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


def dataset_mean_std(
    dataset: types.array,
    sampling_factor: int,
    show_progress: bool = True,
    progress_desc="Estimating channel mean/std",
) -> tuple[float, float]:
    """
    Calculates the mean and sample standard deviation of a 3d dataset using
    Welford's online algorithm, sampling it along its first dimension.

    :param dataset: A 3d dataset, such as a numpy or dask array.
    :param sampling_factor: The sampling factor to sample along the first
        dimension. E.g. if the dataset is 10 x 100 x 100 and `sampling_factor`
        is 3, then we'll use planes 0, 3, 6, 9 for the calculation (40_000
        data points).
    :param show_progress: Whether to show a progress bar during the
        calculation.
    :param progress_desc: If showing a progress bar, the description to use in
        it.
    :return: A 2-tuple of `(mean, std)` estimate of the dataset.
    """
    # based on https://en.wikipedia.org/wiki/
    # Algorithms_for_calculating_variance#Welford's_online_algorithm
    # and https://stackoverflow.com/q/56402955
    plane_n = dataset.shape[1] * dataset.shape[2]
    # get data converter from dataset to float64
    converter = get_data_converter(dataset.dtype, np.float64)

    count = 0
    mean = np.array(0, dtype=np.float64)
    sq_dist = np.array(0, dtype=np.float64)

    # make it a list so tqdm will know its full size
    samples = list(range(0, len(dataset), sampling_factor))
    if show_progress:
        it = tqdm.tqdm(samples, desc=progress_desc, unit="planes")
    else:
        it = samples

    for i in it:
        plane = converter(dataset[i, ...])
        # flatten it
        new_value = plane.reshape((plane_n,))

        count += plane_n
        delta = new_value - mean
        mean += np.sum(delta) / count
        delta2 = new_value - mean
        sq_dist += np.sum(np.multiply(delta, delta2))

    if count <= 1:
        raise ValueError("Not enough data to compute the variance")

    var_sample = sq_dist / (count - 1)
    std = np.sqrt(var_sample)

    return mean.item(), std.item()
