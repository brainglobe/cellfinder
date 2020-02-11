"""
Transforms ROI positions from ImageJ/FIJI into standard space
"""

# TODO: move into aMAP

import argparse
import numpy as np

from os import remove
from brainio import brainio

from pathlib import Path
from datetime import datetime
from skimage import morphology
from tqdm import tqdm
from read_roi import read_roi_zip
from scipy.ndimage import maximum_filter1d
from imlib.general.numerical import check_positive_int
from imlib.general.system import (
    safe_execute_command,
    SafeExecuteCommandError,
)
from imlib.general.exceptions import RegistrationError
from cellfinder.tools.source_files import source_custom_config
from amap.config.atlas import Atlas
from amap.config.config import get_binary
from amap.tools.source_files import get_niftyreg_binaries
import cellfinder.summarise.count_summary as cells_regions


# TODO: get this from amap
SOURCE_IMAGE_NAME = "downsampled.nii"
DEFAULT_CONTROL_POINT_FILE = "inverse_control_point_file.nii"
DEFAULT_OUTPUT_FILE_NAME = "roi_transformed.nii"
DEFAULT_TEMP_FILE_NAME = "ROI_TMP.nii"

PROGRAM_NAME = "reg_resample"


def transform_rois(
    roi_file,
    source_image_filename,
    destination_image_filename,
    control_point_file,
    output_filename,
    temp_output_filename,
    log_file_path,
    error_file_path,
    roi_reference_image=None,
    selem_size=15,
    nii_scale=None,
    transformation_matrix=None,
    debug=False,
    print_value_round_decimals=2,
    z_filter_padding=2,
):
    """
    Using a source image (e.g. downsampled stack), transform an ImageJ
    zipped collection of ROIs into the coordinate space of a
    destination image (e.g. an atlas), using the inverse control point file
    from an existing niftyreg registration

    :param roi_file: .zip collection of ImageJ ROIs
    :param source_image_filename: Image that the ROIs are defined in
    :param destination_image_filename: Image in the destination coordinate space
    :param control_point_file: Transformation from source to destination
    :param output_filename: output filename for the resulting nifti file
    :param temp_output_filename: Temporary file for registration
    :param log_file_path: Path to save niftyreg logs
    :param error_file_path: Path to save niftyreg errors
    :param roi_reference_image: Image on which the ROIs are defined (if not the
    downsampled image in the registration directory)
    :param selem_size: Structure element size for closing
    :param nii_scale: Scaling to correctly save the temporary nifti image
    :param transformation_matrix: Affine transform for the temporary nifti
    image
    :param print_value_round_decimals: How many decimal places to round
    values printed to console.
    :param z_filter_padding: Size of the filter in z when correcting for
    unlabled slices.
    :param debug: If True, don't delete temporary files
    """

    print("Loading ROIs")
    rois = read_roi_zip(roi_file)
    number_rois = len(rois)
    print(f"{number_rois} rois found")
    x = []
    y = []
    z = []
    for key in rois:
        for position in range(0, len(rois[key]["x"])):
            x.append(rois[key]["x"][position])
            y.append(rois[key]["y"][position])
            z.append(rois[key]["position"])

    print("Loading downsampled image image")
    downsampled_source_image = brainio.load_any(str(source_image_filename))

    print(
        f"Source image size: "
        f"x:{downsampled_source_image.shape[0]}, "
        f"y:{downsampled_source_image.shape[1]}, "
        f"y:{downsampled_source_image.shape[2]}"
    )

    downsampled_source_image[:] = 0

    if roi_reference_image is not None:
        print("Reference image flag used. Loading reference image")
        reference_image_shape = brainio.get_size_image_from_file_paths(
            roi_reference_image
        )

        print(
            f"Reference image shape is "
            f"x:{reference_image_shape['x']}, "
            f"y:{reference_image_shape['y']}, "
            f"z:{reference_image_shape['z']}"
        )

        x_downsample_factor = (
            reference_image_shape["x"] / downsampled_source_image.shape[0]
        )
        y_downsample_factor = (
            reference_image_shape["y"] / downsampled_source_image.shape[1]
        )
        z_downsample_factor = (
            reference_image_shape["z"] / downsampled_source_image.shape[2]
        )

        print(
            f"ROIs will be downsampled by a factor of "
            f"x:{round(x_downsample_factor, print_value_round_decimals)}, "
            f"y:{round(y_downsample_factor, print_value_round_decimals)}, "
            f"z:{round(z_downsample_factor, print_value_round_decimals)}"
        )

    # TODO: optimise this
    print("Creating temporary ROI image")
    for position in range(0, len(x)):
        if roi_reference_image is None:
            downsampled_source_image[x[position], y[position], z[position]] = 1
        else:
            x_scale = int(round(x[position] / x_downsample_factor))
            y_scale = int(round(y[position] / y_downsample_factor))
            z_scale = int(round(z[position] / z_downsample_factor))
            downsampled_source_image[x_scale, y_scale, z_scale] = 1

    print("Cleaning up ROI image")
    # TODO speed this up - parallelise?
    selem = morphology.selem.square(selem_size)
    for plane in tqdm(range(0, downsampled_source_image.shape[2])):
        tmp = morphology.binary.binary_closing(
            downsampled_source_image[:, :, plane], selem=selem
        )
        tmp = morphology.convex_hull_object(tmp)
        downsampled_source_image[
            :, :, plane
        ] = morphology.binary.binary_closing(tmp, selem=selem)

    if roi_reference_image is not None:
        if z_downsample_factor < 1:
            print(
                "ROI was defined at a lower z-resolution than the atlas. "
                "Correcting with a maximum filter"
            )
            z_filter_size = int(round(1 / z_scale)) + z_filter_padding
            downsampled_source_image = maximum_filter1d(
                downsampled_source_image, z_filter_size, axis=2
            )

    print(f"Saving temporary ROI image at: {temp_output_filename}")
    brainio.to_nii(
        downsampled_source_image,
        str(temp_output_filename),
        scale=nii_scale,
        affine_transform=transformation_matrix,
    )

    print("Preparing ROI registration")
    nifty_reg_binaries_folder = get_niftyreg_binaries()
    program_path = get_binary(nifty_reg_binaries_folder, PROGRAM_NAME)

    reg_cmd = prepare_segmentation_cmd(
        program_path,
        temp_output_filename,
        output_filename,
        destination_image_filename,
        control_point_file,
    )
    print("Running ROI registration")
    try:
        safe_execute_command(reg_cmd, log_file_path, error_file_path)
    except SafeExecuteCommandError as err:
        raise RegistrationError("ROI registration failed; {}".format(err))

    print(f"Registered ROI image can be found at {output_filename}")

    if not debug:
        print("Deleting temporary files")
        remove(temp_output_filename)
        remove(log_file_path)
        remove(error_file_path)


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        dest="rois", type=Path, help="ImageJ/FIJI ROI .zip collection."
    )
    parser.add_argument(
        dest="reg_dir",
        type=Path,
        help="Cellfinder registration directory containing the downsampled "
        "image that the ROIs were drawn on",
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output_filename",
        type=Path,
        default=None,
        help="Output filename. If not provided, will default to saving in the "
        "same directory as the ROIs",
    )
    parser.add_argument(
        "-r",
        "--reference",
        dest="reference_image",
        type=Path,
        default=None,
        help="If the ROIs are not defined on the downsampled image in the "
        "registration directory (i.e. on the raw, non-downsampled data),"
        "provide the path to this image here (e.g. the directory with a"
        " series of tiff files.)",
    )
    parser.add_argument(
        "--registration-config",
        dest="registration_config",
        type=str,
        help="To supply your own, custom registration configuration file.",
    )
    parser.add_argument(
        "--selem",
        dest="selem_size",
        type=check_positive_int,
        default=10,
        help="Square structure element size for closing ROIs",
    )
    parser.add_argument(
        "--zfill",
        dest="z_filter_padding",
        type=check_positive_int,
        default=2,
        help="Increase this number to fill in gaps in your ROIs in Z (in case"
        "you drew your ROIs on downsampled data).",
    )
    parser.add_argument(
        "--debug",
        dest="debug",
        action="store_true",
        help="Debug mode. Save all intermediate files for diagnosis of "
        "software issues.",
    )
    return parser


def main():
    start_time = datetime.now()
    print("Starting ROI transformation")
    args = get_parser().parse_args()

    rois = args.rois
    print(f"ROI file is: {rois}")

    if args.registration_config is None:
        args.registration_config = source_custom_config()
    atlas = Atlas(args.registration_config)
    source_image = args.reg_dir / SOURCE_IMAGE_NAME
    print(f"Source image is: {source_image}")

    destination_image = args.reg_dir / atlas.atlas_conf["default_brain_name"]
    print(f"Destination image is: {destination_image}")

    control_point_file = args.reg_dir / DEFAULT_CONTROL_POINT_FILE
    print(f"Transformation file is: {control_point_file}")

    if args.output_filename is None:
        output_filename = rois.parent / DEFAULT_OUTPUT_FILE_NAME
        temp_output_filename = rois.parent / DEFAULT_TEMP_FILE_NAME
        log_file_path = rois.parent / "roi_transform_log.txt"
        error_file_path = rois.parent / "roi_transform_error.txt"

    else:
        output_filename = args.output_filename
        temp_output_filename = (
            args.output_filename.parent / DEFAULT_TEMP_FILE_NAME
        )
        log_file_path = args.output_filename.parent / "roi_transform_log.txt"
        error_file_path = (
            args.output_filename.parent / "roi_transform_error.txt"
        )
    if not output_filename.parent.exists():
        output_filename.parent.mkdir()
    print(f"Output file is: {output_filename}")

    atlas = brainio.load_nii(str(destination_image), as_array=False)
    atlas_scale = atlas.header.get_zooms()

    atlas_pixel_sizes = cells_regions.get_atlas_pixel_sizes(
        args.registration_config
    )

    transformation_matrix = np.eye(4)
    for i, axis in enumerate(("x", "y", "z")):
        transformation_matrix[i, i] = atlas_pixel_sizes[axis]

    transform_rois(
        rois,
        source_image,
        destination_image,
        control_point_file,
        output_filename,
        temp_output_filename,
        log_file_path,
        error_file_path,
        roi_reference_image=args.reference_image,
        selem_size=args.selem_size,
        nii_scale=atlas_scale,
        transformation_matrix=transformation_matrix,
        debug=args.debug,
        z_filter_padding=args.z_filter_padding,
    )
    print("Finished. Total time taken: {}".format(datetime.now() - start_time))


def prepare_segmentation_cmd(
    program_path,
    floating_image_path,
    output_file_name,
    destination_image_filename,
    control_point_file,
):
    # TODO combine with amap.brain_registration
    cmd = "{} -cpp {} -flo {} -ref {} -res {}".format(
        program_path,
        control_point_file,
        floating_image_path,
        destination_image_filename,
        output_file_name,
    )
    return cmd


if __name__ == "__main__":
    main()
