import math

import numpy as np


def get_2d_bins(
    arr: np.ndarray, bin_width: int, bin_height: int
) -> np.ndarray:
    """
    Bin a 2D array.

    Parameters
    ----------
    arr :
        Input array.
    bin_width, bin_height :
        Width/height of the bins.

    Returns
    -------
    binned_arr
        A 4D array. The dimensions correspond to:
            1. Bin number along first dimension of input array.
            2. Element index along first dimension within bin.
            3. Bin number along second dimension of input array.
            4. Element index along second dimension within bin.

    Notes
    -----
    The original array is zero padded so that it is tiled exactly
    by tiles of shape (bin_width, bin_height).
    """
    bin_width = int(bin_width)
    bin_height = int(bin_height)

    # Get number of required bins to tile arr
    n_bins_width = int(math.ceil(arr.shape[0] / bin_width))
    n_bins_height = int(math.ceil(arr.shape[1] / bin_height))
    # Get width/height to have integer number of bins that fully
    # covers arr
    width = n_bins_width * bin_width
    height = n_bins_height * bin_height

    # Zero padding array to integer multiples of bin_width/bin_height
    padded_array = np.zeros((width, height), dtype=arr.dtype)
    padded_array[0 : arr.shape[0], 0 : arr.shape[1]] = arr

    # Reshape into 4D array.
    binned_array = padded_array.reshape(
        n_bins_width,
        bin_width,
        n_bins_height,
        bin_height,
    )
    return binned_array


def binned_mean_2d(
    arr: np.ndarray, bin_width: int, bin_height: int
) -> np.ndarray:
    """
    Get mean value of *arr* in bins of shape (*bin_width*, *bin_height*)

    Notes
    -----
    The original array is zero padded so that it is tiled exactly
    by tiles of shape (*bin_width*, *bin_height*). If the tile shape
    does not exactly divide the array shape, values on the outer edge
    of the returned array (arr[-1, :] and arr[:, -1]) will have mean values
    calculated including this zero padding.
    """
    binned_array = get_2d_bins(arr, bin_width, bin_height)
    return binned_array.mean(axis=(1, 3))


def bin_mean_3d(
    arr: np.ndarray, bin_width: int, bin_height: int, bin_depth: int
) -> np.ndarray:
    """
    Bin a 3D array, and return an array of the mean value in each bin.

    Notes
    -----
    If the tile shape does not exactly divide the array shape an error is
    raised.
    """
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
            binned_mean_2d(arr[:, :, j], bin_width, bin_height)
            for j in range(i, i + bin_depth)
        ]
        binned_arr.append(np.dstack(sub_stack).mean(axis=2))
    return np.dstack(binned_arr)
