import logging
import tifffile
import imio

import numpy as np
import pandas as pd

from pathlib import Path
from skimage.filters import gaussian
from imlib.image.scale import scale_and_convert_to_16_bits
from imlib.image.masking import mask_image_threshold
from imlib.general.system import ensure_directory_exists


def get_bins(image_size, bin_sizes):
    """
    Given an image size, and bin size, return a list of the bin boundaries
    :param image_size: Size of the final image (tuple/list)
    :param bin_sizes: Bin sizes corresponding to the dimensions of
    "image_size" (tuple/list)
    :return: List of arrays of bin boundaries
    """
    bins = []
    for dim in range(0, len(image_size)):
        bins.append(np.arange(0, image_size[dim] + 1, bin_sizes[dim]))
    return bins


def run(
    downsampled_points,
    atlas,
    downsampled_shape,
    registered_atlas_path,
    output_filename,
    smoothing=None,
    mask=True,
):

    points = pd.read_hdf(downsampled_points).values

    bins = get_bins(downsampled_shape, (1, 1, 1))
    heatmap_array, _ = np.histogramdd(points, bins=bins)
    heatmap_array = heatmap_array.astype(np.uint16)

    if smoothing is not None:
        logging.debug("Smoothing heatmap")
        # assume isotropic atlas
        smoothing = int(round(smoothing / atlas.resolution[0]))
        heatmap_array = gaussian(heatmap_array, sigma=smoothing)

    if mask:
        logging.debug("Masking image based on registered atlas")
        atlas = tifffile.imread(registered_atlas_path)
        heatmap_array = mask_image_threshold(heatmap_array, atlas)
        del atlas

    logging.debug("Saving heatmap")
    heatmap_array = scale_and_convert_to_16_bits(heatmap_array)

    logging.debug("Ensuring output directory exists")
    ensure_directory_exists(Path(output_filename).parent)

    logging.debug("Saving heatmap image")
    imio.to_tiff(heatmap_array, output_filename)
