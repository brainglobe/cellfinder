import math
import numpy as np


# WARNING: skewed if does not fall exactly (integer multiple)
def bin_array(arr, bin_width=15, bin_height=15):
    bin_width = int(bin_width)
    bin_height = int(bin_height)

    n_bins_width = int(math.ceil(arr.shape[0] / bin_width))
    width = n_bins_width * bin_width
    n_bins_height = int(math.ceil(arr.shape[1] / bin_height))
    height = n_bins_height * bin_height

    # zero padding array to integer multiples of bin_width/bin_height
    padded_array = np.zeros((width, height), dtype=arr.dtype)
    padded_array[0 : arr.shape[0], 0 : arr.shape[1]] = arr

    binned_array = padded_array.reshape(
        padded_array.shape[0] // bin_width,
        bin_width,
        padded_array.shape[1] // bin_height,
        bin_height,
    )
    return binned_array


def pad_and_bin_max(arr, bin_width=15, bin_height=15):
    binned_array = bin_array(arr, bin_width, bin_height)
    return binned_array.max(axis=(1, 3))


def pad_and_bin_mean(arr, bin_width=15, bin_height=15):
    binned_array = bin_array(arr, bin_width, bin_height)
    return binned_array.mean(axis=(1, 3))


def bin_mean_3d(arr, bin_width, bin_height, bin_depth):
    if (arr.shape[0] % bin_width) != 0:
        raise ValueError(
            "Bin width must divide arr exactly, got {} and {}".format(
                arr.shape[0], bin_width
            )
        )
    if (arr.shape[1] % bin_height) != 0:
        raise ValueError(
            "Bin height must divide arr exactly, got {} and {}".format(
                arr.shape[1], bin_height
            )
        )
    if (arr.shape[2] % bin_depth) != 0:
        raise ValueError(
            "Bin depth must divide arr exactly, got {} and {}".format(
                arr.shape[2], bin_depth
            )
        )

    binned_arr = []
    for i in range(0, arr.shape[2] - 1, bin_depth):
        sub_stack = [
            pad_and_bin_mean(arr[:, :, j], bin_width, bin_height)
            for j in range(i, i + bin_depth)
        ]
        binned_arr.append(np.dstack(sub_stack).mean(axis=2))
    binned_arr = np.dstack(binned_arr)
    return binned_arr
