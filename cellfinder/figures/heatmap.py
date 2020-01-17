import logging

import numpy as np
from skimage.filters import gaussian
from skimage.transform import resize


from brainio import brainio

from cellfinder.tools import tools
from cellfinder.tools import image_processing as img_tools
import cellfinder.tools.figures as fig_tools


def heatmap(
    args,
    target_size,
    raw_image_shape,
    raw_image_bin_sizes,
    smoothing=10,
    mask=True,
    atlas=None,
    cells_only=True,
    convert_16bit=True,
    atlas_scale=None,
    transformation_matrix=None,
):
    """

    :param args:
    :param target_size: Size of the final heatmap
    :param raw_image_shape: Size of the raw data (coordinate space of the
    cells)
    :param raw_image_bin_sizes: List/tuple of the sizes of the bins in the
    raw data space
    :param smoothing: Smoothing kernel size, in the target image space
    :param atlas:
    :param mask:
    :param cells_only:
    :param convert_16bit:
    :param atlas_scale: Image scaling so that the resulting nifti can be
    processed using other tools.
    :param transformation_matrix: Transformation matrix so that the resulting
    nifti can be processed using other tools.
    """

    # TODO: compare the smoothing effects of gaussian filtering, and upsampling
    target_size = tools.convert_shape_dict_to_array_shape(
        target_size, type="fiji"
    )
    raw_image_shape = tools.convert_shape_dict_to_array_shape(
        raw_image_shape, type="fiji"
    )
    cells_array = fig_tools.get_cell_location_array(
        args.paths.classification_out_file, cells_only=cells_only
    )
    bins = fig_tools.get_bins(raw_image_shape, raw_image_bin_sizes)

    logging.debug("Generating heatmap (3D histogram)")
    heatmap_array, _ = np.histogramdd(cells_array, bins=bins)

    logging.debug("Resizing heatmap to the size of the target image")
    heatmap_array = resize(heatmap_array, target_size, order=0)
    if smoothing is not None:
        logging.debug(
            "Applying Gaussian smoothing with a kernel sigma of: "
            "{}".format(smoothing)
        )
        heatmap_array = gaussian(heatmap_array, sigma=smoothing)

    if mask:
        logging.debug("Masking image based on registered atlas")
        # copy, otherwise it's modified, which affects later figure generation
        atlas_for_mask = np.copy(atlas)
        heatmap_array = img_tools.mask_image_threshold(
            heatmap_array, atlas_for_mask
        )

    if convert_16bit:
        logging.debug("Converting to 16 bit")
        heatmap_array = tools.scale_and_convert_to_16_bits(heatmap_array)

    logging.debug("Saving heatmap image")
    brainio.to_nii(
        heatmap_array,
        args.paths.heatmap,
        scale=atlas_scale,
        affine_transform=transformation_matrix,
    )
