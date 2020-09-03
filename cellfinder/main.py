"""
main
===============

Runs each part of the cellfinder pipeline in turn.

N.B imports are within functions to prevent tensorflow being imported before
it's warnings are silenced
"""

import os
import logging
import json
from argparse import Namespace
from datetime import datetime
from imlib.general.logging import suppress_specific_logs

tf_suppress_log_messages = [
    "multiprocessing can interact badly with TensorFlow"
]


def get_arg_groups(args, parser):
    arg_groups = {}
    for group in parser._action_groups:
        group_dict = {
            a.dest: getattr(args, a.dest, None) for a in group._group_actions
        }
        arg_groups[group.title] = Namespace(**group_dict)

    return arg_groups


def log_metadata(file_path, args):
    with open(file_path, "w") as f:
        json.dump(args, f, default=lambda x: x.__dict__)


def main():
    suppress_tf_logging(tf_suppress_log_messages)
    from brainreg.main import main as register

    from cellfinder.tools import prep

    start_time = datetime.now()
    args, arg_groups, what_to_run = prep.prep_cellfinder_general()

    if what_to_run.register:
        # TODO: add register_part_brain option
        logging.info("Registering to atlas")
        args, additional_images_downsample = prep.prep_registration(args)
        register(
            args.atlas,
            args.orientation,
            args.target_brain_path,
            args.brainreg_paths,
            arg_groups["NiftyReg registration backend options"],
            x_pixel_um=args.x_pixel_um,
            y_pixel_um=args.y_pixel_um,
            z_pixel_um=args.z_pixel_um,
            sort_input_file=args.sort_input_file,
            n_free_cpus=args.n_free_cpus,
            additional_images_downsample=additional_images_downsample,
            backend=args.backend,
            debug=args.debug,
        )

    else:
        logging.info("Skipping registration")

    if len(args.signal_planes_paths) > 1:
        base_directory = args.output_dir

        for idx, signal_paths in enumerate(args.signal_planes_paths):
            channel = args.signal_ch_ids[idx]
            logging.info("Processing channel: " + str(channel))
            channel_directory = os.path.join(
                base_directory, "channel_" + str(channel)
            )
            if not os.path.exists(channel_directory):
                os.makedirs(channel_directory)

            # prep signal channel specific args
            args.signal_planes_paths[0] = signal_paths
            # TODO: don't overwrite args.output_dir - use Paths instead
            args.output_dir = channel_directory
            args.signal_channel = channel
            # Run for each channel
            run_all(args, what_to_run)

    else:
        args.signal_channel = args.signal_ch_ids[0]
        run_all(args, what_to_run)
    logging.info(
        "Finished. Total time taken: {}".format(datetime.now() - start_time)
    )


def run_all(args, what_to_run):
    from cellfinder.detect import detect
    from cellfinder.classify import classify

    # from cellfinder.figures import figures
    from cellfinder.tools import prep
    from pathlib import Path

    args, what_to_run = prep.prep_channel_specific_general(args, what_to_run)

    if what_to_run.detect:
        logging.info("Detecting cell candidates")
        args = prep.prep_candidate_detection(args)
        detect.main(args)
    else:
        logging.info("Skipping cell detection")

    if what_to_run.classify:
        logging.info("Running cell classification")
        args = prep.prep_classification(args)
        classify.main(args)
    else:
        logging.info("Skipping cell classification")

    from bg_atlasapi import BrainGlobeAtlas
    from cellfinder.bg_summarise import get_brain_structures

    # TODO: bring into whattorun
    if what_to_run.classify and os.path.exists(
        args.brainreg_paths.deformation_field_0
    ):
        atlas = BrainGlobeAtlas(args.atlas)

        deformation_field_paths = [
            args.brainreg_paths.deformation_field_0,
            args.brainreg_paths.deformation_field_1,
            args.brainreg_paths.deformation_field_2,
        ]

        get_brain_structures(
            atlas,
            args.orientation,
            args.x_pixel_um,
            args.y_pixel_um,
            args.z_pixel_um,
            args.paths.classification_out_file,
            args.signal_planes_paths[0],
            deformation_field_paths,
            args.brainreg_paths.volume_csv_path,
            Path(args.output_dir),
        )

    # if what_to_run.figures:
    #     logging.info("Generating figures")
    #     args = prep.figures_prep(args)
    #     figures.figures(args)
    # else:
    #     logging.info("Skipping figure generation")


def suppress_tf_logging(tf_suppress_log_messages):
    """
    Prevents many lines of logs such as:
    "2019-10-24 16:54:41.363978: I tensorflow/stream_executor/platform/default
    /dso_loader.cc:44] Successfully opened dynamic library libcuda.so.1"
    """
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    for message in tf_suppress_log_messages:
        suppress_specific_logs("tensorflow", message)


if __name__ == "__main__":
    main()
