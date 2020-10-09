"""
main
===============

Runs each part of the cellfinder pipeline in turn.

N.B imports are within functions to prevent tensorflow being imported before
it's warnings are silenced
"""

import os
import logging
import tifffile
from datetime import datetime
import bg_space as bgs
from imlib.general.logging import suppress_specific_logs

tf_suppress_log_messages = [
    "multiprocessing can interact badly with TensorFlow"
]


def get_downsampled_space(atlas, downsampled_image_path):
    target_shape = tifffile.imread(downsampled_image_path).shape
    downsampled_space = bgs.AnatomicalSpace(
        atlas.metadata["orientation"],
        shape=target_shape,
        resolution=atlas.resolution,
    )
    return downsampled_space


def main():
    suppress_tf_logging(tf_suppress_log_messages)
    from brainreg.main import main as register

    from cellfinder.tools import prep

    start_time = datetime.now()
    args, arg_groups, what_to_run, atlas = prep.prep_cellfinder_general()

    if what_to_run.register:
        # TODO: add register_part_brain option
        logging.info("Registering to atlas")
        args, additional_images_downsample = prep.prep_registration(args)
        register(
            args.atlas,
            args.orientation,
            args.target_brain_path,
            args.brainreg_paths,
            args.voxel_sizes,
            arg_groups["NiftyReg registration backend options"],
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
            run_all(args, what_to_run, atlas)

    else:
        args.signal_channel = args.signal_ch_ids[0]
        run_all(args, what_to_run, atlas)
    logging.info(
        "Finished. Total time taken: {}".format(datetime.now() - start_time)
    )


def run_all(args, what_to_run, atlas):
    from cellfinder.detect import detect
    from cellfinder.classify import classify
    from cellfinder.analyse import analyse
    from cellfinder.figures import figures

    from cellfinder.tools import prep

    args, what_to_run = prep.prep_channel_specific_general(args, what_to_run)

    if what_to_run.detect:
        logging.info("Detecting cell candidates")
        args = prep.prep_candidate_detection(args)
        detect.main(args)
    else:
        logging.info("Skipping cell detection")

    if what_to_run.classify:
        args = prep.prep_classification(args, what_to_run)
        if what_to_run.classify:
            logging.info("Running cell classification")
            what_to_run.cells_exist = classify.main(args)

        else:
            logging.info("No cells were detected, skipping classification.")

    else:
        logging.info("Skipping cell classification")

    what_to_run.update_if_cells_required()

    if what_to_run.analyse or what_to_run.figures:
        downsampled_space = get_downsampled_space(
            atlas, args.brainreg_paths.boundaries_file_path
        )

    if what_to_run.analyse:
        logging.info("Analysing cell positions")
        analyse.run(args, atlas, downsampled_space)
    else:
        logging.info("Skipping cell position analysis")

    if what_to_run.figures:
        logging.info("Generating figures")
        figures.run(args, atlas, downsampled_space.shape)
    else:
        logging.info("Skipping figure generation")


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
