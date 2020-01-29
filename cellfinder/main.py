"""
main
===============

Runs each part of the cellfinder pipeline in turn.

N.B imports are within functions to prevent tensorflow being imported before
it's warnings are silenced
"""

import os
import logging
from datetime import datetime
from imlib.general.logging import suppress_specific_logs

tf_suppress_log_messages = [
    "multiprocessing can interact badly with TensorFlow"
]


def main():
    suppress_tf_logging(tf_suppress_log_messages)
    import amap.main as register

    from cellfinder.tools import prep

    start_time = datetime.now()
    args, what_to_run = prep.prep_cellfinder_general()

    if what_to_run.register:
        # TODO: add register_part_brain option
        logging.info("Registering to atlas")
        args, additional_images_downsample = prep.prep_registration(args)

        register.main(
            args.registration_config,
            args.target_brain_path,
            args.paths.registration_output_folder,
            x_pixel_um=args.x_pixel_um,
            y_pixel_um=args.y_pixel_um,
            z_pixel_um=args.z_pixel_um,
            orientation=args.orientation,
            flip_x=args.flip_x,
            flip_y=args.flip_y,
            flip_z=args.flip_z,
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
            sort_input_file=args.sort_input_file,
            n_free_cpus=args.n_free_cpus,
            save_downsampled=not (args.no_save_downsampled),
            additional_images_downsample=additional_images_downsample,
            boundaries=not (args.no_boundaries),
            debug=args.debug,
        )
    else:
        logging.info("Skipping registration")

    if what_to_run.summarise:
        args = prep.prep_atlas_conf(args)

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
    import cellfinder.summarise.count_summary as cell_count_summary
    from cellfinder.figures import figures
    from cellfinder.tools import prep
    from cellfinder.standard_space.cells_to_standard_space import (
        transform_cells_to_standard_space,
    )

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

    if what_to_run.summarise:
        logging.info("Summarising cell counts")
        cell_count_summary.analysis_run(args)

    else:
        logging.info("Skipping cell count summary")

    if what_to_run.standard_space:
        logging.info("Converting cells to standard space")
        args = prep.standard_space_prep(args)
        transform_cells_to_standard_space(args)
    else:
        logging.info("Skipping converting cells to standard space")

    if what_to_run.figures:
        logging.info("Generating figures")
        args = prep.figures_prep(args)
        figures.figures(args)
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
