import logging

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from datetime import datetime
from os.path import join

from imlib.general.system import (
    ensure_directory_exists,
    safe_execute_command,
    SafeExecuteCommandError,
)
from imlib.IO.cells import get_cells, save_cells
from imlib.cells.cells import transform_cell_positions
from imlib.general.system import delete_temp
from imlib.general.exceptions import TransformationError
from imlib.image.metadata import define_pixel_sizes
from brainio.brainio import load_any as load_any_image
from amap.register.registration_params import RegistrationParams

from cellfinder.tools import tools, prep
import cellfinder.tools.parser as cellfinder_parse
from cellfinder.tools.source_files import source_custom_config


def transform_cells_to_standard_space(args):
    if args.registration_config is None:
        args.registration_config = source_custom_config()

    reg_params = RegistrationParams(
        args.registration_config,
        affine_n_steps=args.affine_n_steps,
        affine_use_n_steps=args.affine_use_n_steps,
        freeform_n_steps=args.freeform_n_steps,
        freeform_use_n_steps=args.freeform_use_n_steps,
        bending_energy_weight=args.bending_energy_weight,
        grid_spacing_x=args.grid_spacing_x,
        smoothing_sigma_reference=args.smoothing_sigma_reference,
        smoothing_sigma_floating=args.smoothing_sigma_floating,
        histogram_n_bins_floating=args.histogram_n_bins_floating,
        histogram_n_bins_reference=args.histogram_n_bins_reference,
    )

    generate_deformation_field(args, reg_params)
    cells_only = not args.transform_all
    cells = get_cells(
        args.paths.classification_out_file, cells_only=cells_only
    )

    logging.info("Loading deformation field")
    deformation_field = load_any_image(args.paths.tmp__deformation_field)
    scales = get_scales(args, reg_params)
    field_scales = get_deformation_field_scales(reg_params)

    logging.info("Transforming cell positions")
    transformed_cells = transform_cell_positions(
        cells, deformation_field, field_scales, scales
    )

    logging.info("Saving transformed cell positions")

    save_cells(
        transformed_cells,
        args.paths.cells_in_standard_space,
        save_csv=args.save_csv,
    )

    if not args.debug:
        logging.info("Removing standard space transformation temp files")
        delete_temp(args.paths.standard_space_output_folder, args.paths)


def get_deformation_field_scales(reg_params):
    """
    Calculates the scaling of the deformation field from real space
    to voxels
    :param reg_params:
    :return:
    """
    x_scale = 1000 / float(reg_params.atlas_x_pix_size)
    y_scale = 1000 / float(reg_params.atlas_y_pix_size)
    z_scale = 1000 / float(reg_params.atlas_z_pix_size)

    return x_scale, y_scale, z_scale


def get_scales(args, reg_params):
    """
    Calculate the scaling factors from the coordinate space of the raw data
    (which the cell positions are defined in) to the (usually downsampled)
    reference space that the deformation field is defined in.
    :param args:
    :param reg_params:
    :return:
    """
    x_scale = args.x_pixel_um / float(reg_params.atlas_x_pix_size)
    y_scale = args.y_pixel_um / float(reg_params.atlas_y_pix_size)
    z_scale = args.z_pixel_um / float(reg_params.atlas_z_pix_size)

    return x_scale, y_scale, z_scale


def prepare_deformation_field_cmd(args, reg_params):
    """
    Prepares the niftyreg command to generate the deformation field from the
    control point file (allready generated in registration).
    :param args:
    :param reg_params:
    :return:
    """
    cmd = "{} -def {} {} -ref {}".format(
        reg_params.transform_program_path,
        args.paths.control_point_file_path,
        args.paths.tmp__deformation_field,
        args.paths.downsampled_background,
    )
    return cmd


def generate_deformation_field(args, reg_params):
    logging.info("Generating deformation field")
    try:
        safe_execute_command(
            prepare_deformation_field_cmd(args, reg_params),
            args.paths.tmp__deformation_log_file_path,
            args.paths.tmp__deformation_error_path,
        )
    except SafeExecuteCommandError as err:
        raise TransformationError(
            "Generation of deformation field failed ; {}".format(err)
        )


def cells_standard_space_cli_parser():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser = cli_parse(parser)
    parser = cellfinder_parse.pixel_parser(parser)
    parser = cellfinder_parse.standard_space_parse(parser)
    parser = cellfinder_parse.misc_parse(parser)

    return parser


def cli_parse(parser):
    cli_parser = parser.add_argument_group("Command line options")
    cli_parser.add_argument(
        "--cells",
        dest="cells_file_path",
        type=str,
        required=True,
        help="Path of the xml file containing cells to be transformed",
    )

    cli_parser.add_argument(
        "--transformation",
        dest="transformation",
        type=str,
        required=True,
        help="Path to the control point file (from niftyreg or cellfinder "
        "registration).",
    )

    cli_parser.add_argument(
        "-o",
        "--output-dir",
        dest="output_dir",
        type=str,
        required=True,
        help="Directory to save the cubes into",
    )

    cli_parser.add_argument(
        "--ref",
        dest="reference",
        type=str,
        required=True,
        help="Reference nii image, in the same space as the downsampled raw "
        "data.",
    )

    cli_parser.add_argument(
        "--registration-config",
        dest="registration_config",
        type=str,
        help="To supply your own, custom registration configuration file.",
    )
    return parser


def cli_path_update(paths, args):
    paths.control_point_file_path = args.transformation
    paths.downsampled_background = args.reference
    paths.classification_out_file = args.cells_file_path
    paths.output_dir = args.output_dir


def main():
    start_time = datetime.now()
    args = cells_standard_space_cli_parser().parse_args()
    args.paths = prep.Paths(args.output_dir)
    args.paths.standard_space_output_folder = args.output_dir
    args.paths.cells_in_standard_space = join(
        args.paths.output_dir, "cells_in_standard_space.xml"
    )
    cli_path_update(args.paths, args)
    args.paths.make_invert_cell_position_paths()
    args = define_pixel_sizes(args)

    # TODO: implement a recursive function to remove the need to do this
    # (probably using pathlib)
    ensure_directory_exists(args.paths.output_dir)
    ensure_directory_exists(args.paths.standard_space_output_folder)
    tools.start_logging(
        args.paths.output_dir,
        args=args,
        verbose=args.debug,
        filename="cells_to_standard_space",
        log_header="CELL TRANSFORMATION TO STANDARD SPACE LOG",
    )
    logging.info("Starting transformation of cell positions")
    transform_cells_to_standard_space(args)
    logging.info("Finished. Total time taken: %s", datetime.now() - start_time)


if __name__ == "__main__":
    main()
