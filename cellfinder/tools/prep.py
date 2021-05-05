"""
prep
==================
Functions to prepare files and directories needed for other functions
"""


import os
import logging
import json

from fancylog import fancylog
from pathlib import PurePath

from imlib.general.system import ensure_directory_exists

from imlib.general.exceptions import CommandLineInputError


from cellfinder.tools.parser import cellfinder_parser
import cellfinder as program_for_log
import cellfinder.tools.parser as parser
from cellfinder.tools import tools, system
from argparse import Namespace
from brainreg.paths import Paths as BrainRegPaths
from bg_atlasapi import BrainGlobeAtlas


def get_arg_groups(args, parser):
    arg_groups = {}
    for group in parser._action_groups:
        group_dict = {
            a.dest: getattr(args, a.dest, None) for a in group._group_actions
        }
        arg_groups[group.title] = Namespace(**group_dict)

    return arg_groups


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

    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.registration_output_folder = os.path.join(
            self.output_dir, "registration"
        )
        self.metadata_path = os.path.join(self.output_dir, "cellfinder.json")
        self.registration_metadata_path = os.path.join(
            self.registration_output_folder, "brainreg.json"
        )

    def make_channel_specific_paths(self):
        self.points_directory = os.path.join(self.output_dir, "points")
        self.detected_points = os.path.join(self.points_directory, "cells.xml")
        self.classified_points = os.path.join(
            self.points_directory, "cell_classification.xml"
        )
        self.downsampled_points = os.path.join(
            self.points_directory, "downsampled.points"
        )
        self.atlas_points = os.path.join(self.points_directory, "atlas.points")
        self.brainrender_points = os.path.join(
            self.points_directory, "points.npy"
        )
        self.abc4d_points = os.path.join(self.points_directory, "abc4d.npy")

        self.tmp__cubes_output_dir = os.path.join(self.output_dir, "cubes")

        self.figures_directory = os.path.join(self.output_dir, "figures")
        self.heatmap = os.path.join(self.figures_directory, "heatmap.tiff")

        self.analysis_directory = os.path.join(self.output_dir, "analysis")
        self.summary_csv = os.path.join(self.analysis_directory, "summary.csv")
        self.all_points_csv = os.path.join(
            self.analysis_directory, "all_points.csv"
        )


def serialise(obj):
    if isinstance(obj, PurePath):
        return str(obj)
    else:
        return obj.__dict__


def log_metadata(file_path, args):
    with open(file_path, "w") as f:
        json.dump(args, f, default=serialise)


def prep_cellfinder_general():
    args = parser.cellfinder_parser().parse_args()
    arg_groups = get_arg_groups(args, cellfinder_parser())

    check_input_arg_existance(args)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    args.paths = Paths(args.output_dir)

    fancylog.start_logging(
        args.output_dir,
        program_for_log,
        variables=[args, args.paths],
        verbose=args.debug,
        log_header="CELLFINDER LOG",
    )

    log_metadata(args.paths.metadata_path, args)

    what_to_run = CalcWhatToRun(args)
    args.signal_ch_ids, args.background_ch_id = check_and_return_ch_ids(
        args.signal_ch_ids, args.background_ch_id, args.signal_planes_paths
    )
    args.brainreg_paths = BrainRegPaths(args.paths.registration_output_folder)
    atlas = BrainGlobeAtlas(args.atlas)
    return args, arg_groups, what_to_run, atlas


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
        self.analyse = True
        self.figures = True
        self.candidates_exist = True
        self.cells_exist = True

        self.atlas_image = os.path.join(
            args.paths.registration_output_folder, "registered_atlas.tiff"
        )
        # order is important
        self.cli_options(args)
        self.existence()

    def update(self, args):
        self.cli_options(args)
        self.existence()
        self.channel_specific_update(args)

    def cli_options(self, args):
        self.detect = not args.no_detection
        self.classify = not args.no_classification
        self.register = not args.no_register
        self.analyse = not args.no_analyse
        self.figures = not args.no_figures

    def existence(self):
        if os.path.exists(self.atlas_image):
            logging.warning(
                "Registered atlas exists, assuming " "already run. Skipping."
            )
            self.register = False

    def channel_specific_update(self, args):
        if os.path.exists(args.paths.detected_points):
            logging.warning(
                "Initial detection file exists (cells.xml), "
                "assuming already run. Skipping."
            )
            self.detect = False

        if os.path.exists(args.paths.classified_points):
            logging.warning(
                "Cell classification file "
                "(cell_classification.xml)"
                " exists, assuming already run. Skipping cell detection, "
                "and classification"
            )
            self.classify = False
            self.detect = False

        if not os.path.exists(self.atlas_image):
            self.analyse = False
            self.figures = False

        if os.path.exists(args.paths.summary_csv):
            self.analyse = False

        if os.path.exists(args.paths.heatmap):
            self.figures = False

    def update_if_candidates_required(self):
        if not self.candidates_exist:
            self.classify = False

    def update_if_cells_required(self):
        if not self.cells_exist:
            self.analyse = False
            self.figures = False


def prep_registration(args):
    args.target_brain_path = args.background_planes_path[0]
    logging.debug("Making registration directory")
    ensure_directory_exists(args.paths.registration_output_folder)
    log_metadata(args.paths.registration_metadata_path, args)
    additional_images_downsample = {}
    for idx, images in enumerate(args.signal_planes_paths):
        channel = args.signal_ch_ids[idx]
        additional_images_downsample[f"channel_{channel}"] = images
    return args, additional_images_downsample


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
