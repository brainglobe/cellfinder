## INACTIVE

import logging

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from datetime import datetime
from pathlib import Path
from natsort import natsorted

import cellfinder.tools.tools as tools
import cellfinder.tools.parser as cellfinder_parse
import cellfinder.extract.extract_cubes as extract_cubes
import cellfinder.tools.prep as prep
from cellfinder.tools.misc import check_positive_int
from cellfinder.tools.system import get_subdirectories as subdirs
from cellfinder.tools import system
from cellfinder.tools.metadata import define_pixel_sizes

# For compatiblity with ROI sorter
CELLS_DIR_NAME = "Cells"
NO_CELLS_DIR_NAME = "NoCells"
train_data_gen_dir_names = [CELLS_DIR_NAME, NO_CELLS_DIR_NAME]

SUPPORTED_CELL_FILE_FORMATS = [".xml", ".yml"]

# TODO: write tests


def cube_extract_cli_parser():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser = cli_parse(parser)
    parser = cellfinder_parse.cube_extract_parse(parser)
    parser = cellfinder_parse.pixel_parser(parser)
    parser = cellfinder_parse.misc_parse(parser)

    return parser


def cli_parse(parser):
    cli_parser = parser.add_argument_group("cellfinder_gen_cubes options")
    cli_parser.add_argument(
        "--cells",
        dest="cells_file_path",
        type=str,
        required=True,
        help="Text file or directory containing paths to the cell locations. "
        "If these paths contain subdirectories (e.g. cells and no_cells),"
        "they will be processed separately",
    )

    cli_parser.add_argument(
        "-i",
        "--img-paths",
        dest="raw_data_paths",
        type=str,
        nargs="+",
        required=True,
        help="Text file or directory containing paths to the image files. Can "
        "also be a text file of directories, or a directory of text files "
        "etc. ",
    )

    cli_parser.add_argument(
        "-o",
        "--output-dir",
        dest="output_dir",
        type=str,
        required=True,
        help="Top-level directory to save the cubes into",
    )

    cli_parser.add_argument(
        "--channel-ids",
        dest="channel_ids",
        type=check_positive_int,
        nargs="+",
        help="Channel ID numbers, in the same order as '--cells'."
        " Will default to '0, 1, 2' etc, but maybe useful to specify.",
    )

    return parser


def prep_channel_ids(args):
    args.signal_ch_ids, _ = prep.check_and_return_ch_ids(
        args.channel_ids, None, args.raw_data_paths
    )
    args.background_ch_id = []
    return args


def extract_loop(args):
    system.ensure_directory_exists(args.paths.tmp__cubes_output_dir)
    extract_cubes.main(args)


def main():
    # TODO: remove converting to strings when cellfinder is python >3.6 only
    start_time = datetime.now()
    args = cube_extract_cli_parser().parse_args()
    args = define_pixel_sizes(args)
    args = prep_channel_ids(args)
    # TODO: can these be set in argparse?
    args.output_dir = Path(args.output_dir)
    args.cells_file_path = Path(args.cells_file_path)
    args.cube_extract_cli = True
    args.paths = prep.Paths(args.output_dir)
    system.ensure_directory_exists(args.output_dir)
    tools.start_logging(
        str(args.output_dir),
        args=args,
        verbose=args.debug,
        filename="batch_cube_extraction",
        log_header="CELLFINDER BATCH CUBE EXTRACTION LOG",
    )
    logging.info("Starting cube extraction")
    num_channels = len(args.raw_data_paths)

    if args.cells_file_path.is_dir():
        indv_cells_paths = natsorted(subdirs(args.cells_file_path))
    elif args.cells_file_path.is_file():
        tmp_paths = tools.get_text_lines(args.cells_file_path)
        indv_cells_paths = natsorted([Path(path) for path in tmp_paths])

    if indv_cells_paths[0].is_dir():
        logging.info("Assuming cube definition data is output from ROI Sorter")
        cells_dir_names = [path.name for path in indv_cells_paths]
        if tools.is_any_list_overlap(
            cells_dir_names, train_data_gen_dir_names
        ):
            # If there is only training data from one sample, make a list of 1
            indv_cells_paths = [args.cells_file_path]
        for idx, sample_dir in enumerate(indv_cells_paths):
            logging.info("Extracting cubes from: {}".format(sample_dir.name))
            args.all_planes_paths = []
            system.ensure_directory_exists(
                args.output_dir.joinpath(sample_dir.name)
            )
            for channel in range(num_channels):
                args.all_planes_paths.append(
                    tools.get_text_lines(
                        args.raw_data_paths[channel],
                        return_lines=idx,
                        sort=True,
                    )
                )

            for sub_dir_name in train_data_gen_dir_names:
                if sub_dir_name in subdirs(sample_dir, names_only=True):
                    logging.info("Extracting: {}".format(sub_dir_name))
                    args.paths.tmp__cubes_output_dir = str(
                        args.output_dir.joinpath(sample_dir.name, sub_dir_name)
                    )
                    args.paths.cells_file_path = str(
                        args.cells_file_path.joinpath(
                            sample_dir.name, sub_dir_name
                        )
                    )
                    extract_loop(args)

    else:
        if indv_cells_paths[0].suffix in SUPPORTED_CELL_FILE_FORMATS:
            logging.info(
                "Cube definition file is: '{}', proceeding."
                "".format(indv_cells_paths[0].suffix)
            )
            raise NotImplementedError(
                "Cube definition file is: '{}', NOT YET SUPPORTED."
                "".format(indv_cells_paths[0].suffix)
            )
        else:
            raise NotImplementedError(
                "File format: '{}' is not yet supported. Please use the "
                "output of ROI Sorter, or provide one of: {}".format(
                    indv_cells_paths[0].suffix, SUPPORTED_CELL_FILE_FORMATS
                )
            )

    logging.info("Finished. Total time taken: %s", datetime.now() - start_time)


if __name__ == "__main__":
    main()
