"""
prep
==================
Functions to prepare files and directories needed for other functions
"""


import os
import logging
from fancylog import fancylog
from pathlib import Path

from imlib.general.system import ensure_directory_exists, get_num_processes
from imlib.image.metadata import define_pixel_sizes
from imlib.general.exceptions import CommandLineInputError
from imlib.general.config import get_config_obj
import amap.download.atlas as atlas_download

import cellfinder.tools.tf as tf_tools
import cellfinder as program_for_log
import cellfinder.tools.parser as parser
from cellfinder.download import models as model_download
from cellfinder.download.download import amend_cfg
from cellfinder.download.cli import temp_dir_path
from cellfinder.tools import tools, source_files, system


def check_input_arg_existance(args):
    """
    Does a simple check to ensure that input files/paths exist. Prevents a typo
    from causing cellfinder to only run partway. Doesn't check for validity
    etc, just existance.
    :param args: Cellfinder input arguments
    """
    check_list = [args.background_planes_path[0]]
    check_list = check_list + args.signal_planes_paths

    for path in check_list:
        if path is not None:
            system.catch_input_file_error(path)


class Paths:
    """
    A single class to hold all file paths that cellfinder may need. Any paths
    prefixed with "tmp__" refer to internal intermediate steps, and will be
    deleted if "--debug" is not used.
    """

    # TODO: Maybe transfer to a config file.
    def __init__(self, output_dir):
        self.output_dir = output_dir

    def make_reg_paths(self):
        self.registration_output_folder = os.path.join(
            self.output_dir, "registration"
        )
        self.downsampled_background = self.make_reg_path("downsampled.nii")
        self.tmp__downsampled_filtered = self.make_reg_path(
            "downsampled_filtered.nii"
        )
        self.registered_atlas_path = self.make_reg_path("registered_atlas.nii")
        self.hemispheres_atlas_path = self.make_reg_path(
            "registered_hemispheres.nii"
        )
        self.volume_csv_path = self.make_reg_path("volumes.csv")

        self.tmp__affine_registered_atlas_brain_path = self.make_reg_path(
            "affine_registered_atlas_brain.nii"
        )
        self.tmp__freeform_registered_atlas_brain_path = self.make_reg_path(
            "freeform_registered_atlas_brain.nii"
        )
        self.tmp__inverse_freeform_registered_atlas_brain_path = self.make_reg_path(
            "inverse_freeform_registered_brain.nii"
        )

        self.registered_atlas_img_path = self.make_reg_path(
            "registered_atlas.nii"
        )
        self.registered_hemispheres_img_path = self.make_reg_path(
            "registered_hemispheres.nii"
        )

        self.affine_matrix_path = self.make_reg_path("affine_matrix.txt")
        self.invert_affine_matrix_path = self.make_reg_path(
            "invert_affine_matrix.txt"
        )

        self.control_point_file_path = self.make_reg_path(
            "control_point_file.nii"
        )
        self.inverse_control_point_file_path = self.make_reg_path(
            "inverse_control_point_file.nii"
        )

        (
            self.tmp__affine_log_file_path,
            self.tmp__affine_error_path,
        ) = self.compute_reg_log_file_paths("affine")
        (
            self.tmp__freeform_log_file_path,
            self.tmp__freeform_error_file_path,
        ) = self.compute_reg_log_file_paths("freeform")
        (
            self.tmp__inverse_freeform_log_file_path,
            self.tmp__inverse_freeform_error_file_path,
        ) = self.compute_reg_log_file_paths("inverse_freeform")
        (
            self.tmp__segmentation_log_file,
            self.tmp__segmentation_error_file,
        ) = self.compute_reg_log_file_paths("segment")
        (
            self.tmp__invert_affine_log_file,
            self.tmp__invert_affine_error_file,
        ) = self.compute_reg_log_file_paths("invert_affine")

    def make_channel_specific_paths(self):
        self.cells_file_path = os.path.join(self.output_dir, "cells.xml")
        self.tmp__cubes_output_dir = os.path.join(self.output_dir, "cubes")
        self.classification_out_file = os.path.join(
            self.output_dir, "cell_classification.xml"
        )
        self.standard_space_output_folder = os.path.join(
            self.output_dir, "standard_space"
        )
        self.cells_in_standard_space = os.path.join(
            self.standard_space_output_folder, "cells_in_standard_space.xml"
        )
        self.figures_dir = os.path.join(self.output_dir, "figures")

    def make_figures_paths(self):
        self.heatmap = os.path.join(self.figures_dir, "heatmap.nii")

    def make_invert_cell_position_paths(self):
        self.tmp__deformation_field = os.path.join(
            self.standard_space_output_folder, "deformation_field.nii"
        )
        self.tmp__deformation_log_file_path = os.path.join(
            self.standard_space_output_folder, "deformation.log"
        )
        self.tmp__deformation_error_path = os.path.join(
            self.standard_space_output_folder, "deformation.err"
        )

    def make_reg_path(self, basename):
        """
        Compute the absolute path of the destination file to
        self.registration_output_folder.

        :param str basename:
        :return: The path
        :rtype: str
        """
        return os.path.join(self.registration_output_folder, basename)

    def compute_reg_log_file_paths(self, basename):
        """
        Compute the path of the log and err file for the step corresponding
        to basename

        :param str basename:
        :return: log_file_path, error_file_path
        """

        log_file_template = os.path.join(
            self.registration_output_folder, "{}.log"
        )
        error_file_template = os.path.join(
            self.registration_output_folder, "{}.err"
        )
        log_file_path = log_file_template.format(basename)
        error_file_path = error_file_template.format(basename)
        return log_file_path, error_file_path


def prep_cellfinder_general():
    args = parser.cellfinder_parser().parse_args()
    args = define_pixel_sizes(args)
    check_input_arg_existance(args)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    args.paths = Paths(args.output_dir)
    args.paths.make_reg_paths()

    fancylog.start_logging(
        args.output_dir,
        program_for_log,
        variables=[args, args.paths],
        verbose=args.debug,
        log_header="CELLFINDER LOG",
    )

    what_to_run = CalcWhatToRun(args)
    args.signal_ch_ids, args.background_ch_id = check_and_return_ch_ids(
        args.signal_ch_ids, args.background_ch_id, args.signal_planes_paths
    )
    return args, what_to_run


def check_and_return_ch_ids(signal_ids, background_id, signal_channel_list):
    """
    If channel IDs are provided (via CLI), then they are checked for
    suitability (i.e. they are unique).
    If channel IDs are not given, then unique IDs will be generated

    Assumes only 1 background channel (for registration)

    :param list signal_ids: List of ID ints, or None.
    :param int background_id: ID int, or None.
    :param list signal_channel_list: Input files
    :return: list of signal channel IDs, and background channel ID
    """

    # TODO: make this more general, so it gives informative error messages
    #  when called from cellfinder.extract.cli

    num_signal_channels = len(signal_channel_list)

    if signal_ids is None:
        if background_id is None:
            signal_ids = list(range(num_signal_channels))
            background_id = num_signal_channels  # max(signal_ids) + 1
            pass
        else:
            signal_start = background_id + 1
            signal_end = signal_start + num_signal_channels
            signal_ids = list(range(signal_start, signal_end))
            pass
    else:
        check_correct_number_signal_channels(signal_ids, num_signal_channels)
        check_signal_overlap(signal_ids)
        if background_id is None:
            background_id = max(signal_ids) + 1
            pass
        else:
            # N.B. check_id_overlap needs bg_ch to be iterable (e.g. a list)
            check_sig_bg_id_overlap(signal_ids, [background_id])
    return signal_ids, background_id


def check_correct_number_signal_channels(signal_ids, num_signal_channels):
    num_signal_ids = len(signal_ids)
    if num_signal_channels != num_signal_ids:
        logging.error(
            "Number of signal channel IDs ({}) does not match the number of "
            "signal channels ({})".format(num_signal_ids, num_signal_channels)
        )

        raise CommandLineInputError(
            "Number of signal channel IDs ({}) does not match the number of "
            "signal channels ({})".format(num_signal_ids, num_signal_channels)
        )


def check_signal_overlap(signal_ids):
    signal_unique, repeated_signal_ch = tools.check_unique_list(signal_ids)
    if not signal_unique:
        logging.error(
            "Non-unique values for '--signal-channel-ids' ({}) given. ID:"
            " {} given more than once.".format(signal_ids, repeated_signal_ch)
        )
        raise CommandLineInputError(
            "Non-unique values for '--signal-channel-ids' ({}) given. ID:"
            " {} given more than once.".format(signal_ids, repeated_signal_ch)
        )


def check_sig_bg_id_overlap(signal_ids, background_id):
    common_id_result, intersection = tools.common_member(
        signal_ids, background_id
    )

    if common_id_result:
        logging.error(
            "Non-unique values for '--signal-channel-ids' ({}) and for "
            "'--background-channel-id' ({}) given. ID: {} was given in"
            " both.".format(signal_ids, background_id, intersection)
        )
        raise CommandLineInputError(
            "Non-unique values for '--signal-channel-ids' ({}) and for "
            "'--background-channel-id' ({}) given. ID: {} was given in"
            " both.".format(signal_ids, background_id, intersection)
        )


class CalcWhatToRun:
    """
    Class to (hopefully) simplify what should and shouldn't be run
    """

    def __init__(self, args):
        self.detect = True
        self.classify = True
        self.register = True
        self.summarise = True
        self.figures = True
        self.standard_space = True

        # order is important
        self.cli_options(args)
        self.hierarchy()
        self.existence(args)

    def update(self, args):
        self.cli_options(args)
        self.hierarchy()
        self.existence(args)
        self.channel_specific_update(args)

    def cli_options(self, args):
        self.detect = not args.no_detection
        self.classify = not args.no_classification

        self.register = args.register
        self.summarise = args.summarise
        self.figures = args.figures

        self.standard_space = not args.no_standard_space

    def hierarchy(self):
        if self.summarise or self.figures:
            self.classify = True
            self.register = True

        if not self.detect or not self.classify:
            self.standard_space = False

    def existence(self, args):
        if os.path.exists(
            os.path.join(
                args.paths.registration_output_folder, "registered_atlas.nii"
            )
        ):
            logging.warning(
                "Registered atlas exists, assuming " "already run. Skipping."
            )
            self.register = False
        else:
            # If registration hasn't happened (and it's not going to), then
            # don't carry out standard space analysis
            if not self.register and not self.figures:
                self.standard_space = False

    def channel_specific_update(self, args):
        if os.path.exists(args.paths.cells_file_path):
            logging.warning(
                "Initial detection file exists (cells.xml), "
                "assuming already run. Skipping."
            )
            self.detect = False

        if os.path.exists(args.paths.classification_out_file):
            logging.warning(
                "Cell classification file "
                "(cell_classification.xml)"
                " exists, assuming already run. Skipping cell detection, "
                "and classification"
            )
            self.classify = False
            self.detect = False

        if os.path.exists(
            os.path.join(args.output_dir, "summary_cell_counts.csv")
        ):
            logging.warning(
                "Summary file exists, assuming already run. "
                "skipping everything."
            )
            self.summarise = False
            self.detect = False
            self.classify = False

        if os.path.exists(args.paths.figures_dir) and (
            not os.listdir(args.paths.figures_dir) == []
        ):

            logging.warning(
                "Figures directory not empty, assuming already run. "
                "Skipping detection, and classification."
            )
            self.figures = False
            self.detect = False
            self.classify = False

        if os.path.exists(args.paths.cells_in_standard_space):
            logging.warning(
                "Cells in standard space xml file exists, assuming already "
                "run. Skipping detection and classification."
            )
            self.detect = False
            self.classify = False
            self.standard_space = False


def prep_registration(args, sample_name="amap"):
    logging.info("Checking whether the atlas exists")
    _, atlas_files_exist = check_atlas_install()
    atlas_dir = os.path.join(args.install_path, "atlas")
    if not atlas_files_exist:
        if args.download_path is None:
            atlas_download_path = os.path.join(temp_dir_path, "atlas.tar.gz")
        else:
            atlas_download_path = os.path.join(
                args.download_path, "atlas.tar.gz"
            )
        if not args.no_atlas:
            logging.warning("Atlas does not exist, downloading.")
            atlas_download.main(args.atlas, atlas_dir, atlas_download_path)
        amend_cfg(new_atlas_folder=atlas_dir, atlas=args.atlas)

    if args.registration_config is None:
        args.registration_config = source_files.source_custom_config()
    args.target_brain_path = args.background_planes_path[0]
    args.sample_name = sample_name
    logging.debug("Making registration directory")
    ensure_directory_exists(args.paths.registration_output_folder)

    additional_images_downsample = {}
    for idx, images in enumerate(args.signal_planes_paths):
        channel = args.signal_ch_ids[idx]
        additional_images_downsample[f"channel_{channel}"] = images

    return args, additional_images_downsample


def check_atlas_install():
    """
    Checks whether the atlas directory exists, and whether it's empty or not.
    :return: Whether the directory exists, and whether the files also exist
    """
    # TODO: make more sophisticated, check for all files that might be needed
    dir_exists = False
    files_exist = False
    cfg_file_path = source_files.source_custom_config()
    if os.path.exists(cfg_file_path):
        config_obj = get_config_obj(cfg_file_path)
        atlas_conf = config_obj["atlas"]
        atlas_directory = atlas_conf["base_folder"]
        if os.path.exists(atlas_directory):
            dir_exists = True
            if not os.listdir(atlas_directory) == []:
                files_exist = True

    return dir_exists, files_exist


def prep_atlas_conf(args):
    """
    Used by cellfinder_run to source the correct registration configuration
    and also by cellfinder_count_summary to get atlas pixel sizes.

    :param args: Args possibly including "registration_config"
    :return: args: Args with atlas configuration for registration
    """
    if args.atlas_config is None:
        if (
            hasattr(args, "registration_config")
            and args.registration_config is not None
        ):
            args.atlas_config = args.registration_config
        else:
            # can get atlas pixel sizes from registration config
            args.atlas_config = source_files.source_custom_config()
    return args


def prep_classification(args):
    n_processes = get_num_processes(min_free_cpu_cores=args.n_free_cpus)
    prep_tensorflow(n_processes)
    args = prep_models(args)
    return args


def prep_training(args):
    n_processes = get_num_processes(min_free_cpu_cores=args.n_free_cpus)
    prep_tensorflow(n_processes)
    args = prep_models(args)
    return args


def prep_tensorflow(max_threads):
    tf_tools.set_tf_threads(max_threads)
    tf_tools.allow_gpu_memory_growth()


def prep_models(args):
    ## if no model or weights, set default weights
    if args.trained_model is None and args.model_weights is None:
        logging.debug("No model or weights supplied, so using the default")

        config_file = source_files.source_custom_config()
        if not Path(config_file).exists():
            logging.debug("Custom config does not exist, downloading models")
            model_path = model_download.main(args.model, args.install_path)
            amend_cfg(new_model_path=model_path)

        model_weights = get_model_weights(config_file)
        if model_weights is not "" and Path(model_weights).exists():
            args.model_weights = model_weights
        else:
            logging.debug("Model weights do not exist, downloading")
            model_path = model_download.main(args.model, args.install_path)
            amend_cfg(new_model_path=model_path)
            model_weights = get_model_weights(config_file)
            args.model_weights = model_weights
    return args


def get_model_weights(config_file):
    logging.debug(f"Reading config file: {config_file}")
    config_obj = get_config_obj(config_file)
    model_conf = config_obj["model"]
    model_weights = model_conf["model_path"]
    return model_weights


def prep_channel_specific_general(args, what_to_run):
    args.paths.output_dir = args.output_dir
    args.paths.make_channel_specific_paths()
    what_to_run.update(args)
    return args, what_to_run


def prep_candidate_detection(args):
    if args.save_planes:
        args.plane_directory = os.path.join(
            args.output_dir, "processed_planes"
        )
        ensure_directory_exists(args.plane_directory)
    else:
        args.plane_directory = None  # FIXME: remove this fudge

    return args


def standard_space_prep(args):
    ensure_directory_exists(args.paths.standard_space_output_folder)
    args.paths.make_invert_cell_position_paths()
    return args


def figures_prep(args):
    ensure_directory_exists(args.paths.figures_dir)
    args.paths.make_figures_paths()
    return args
