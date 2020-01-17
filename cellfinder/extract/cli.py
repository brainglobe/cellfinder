import logging

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from datetime import datetime

from cellfinder.tools import tools, system, prep
import cellfinder.tools.parser as cellfinder_parse
import cellfinder.extract.extract_cubes as extract_cubes
from cellfinder.tools.misc import check_positive_int
from cellfinder.tools.metadata import define_pixel_sizes


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
        help="Path of the xml file (or Roi sorter output directory)"
        "containing the ROIs to be extracted",
    )

    cli_parser.add_argument(
        "-i",
        "--img-paths",
        dest="all_planes_paths",
        type=str,
        nargs="+",
        required=True,
        help="Path to the directory of the image files. Can also be a text"
        "file pointing to the files.",
    )

    cli_parser.add_argument(
        "-o",
        "--output-dir",
        dest="cubes_output_dir",
        type=str,
        required=True,
        help="Directory to save the cubes into",
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
        args.channel_ids, None, args.all_planes_paths
    )
    args.background_ch_id = []
    return args


def main():
    start_time = datetime.now()
    args = cube_extract_cli_parser().parse_args()
    args = define_pixel_sizes(args)
    args = prep_channel_ids(args)
    args.cube_extract_cli = True
    args.paths = prep.Paths(args.cubes_output_dir)
    args.paths.tmp__cubes_output_dir = args.cubes_output_dir
    args.paths.cells_file_path = args.cells_file_path
    system.ensure_directory_exists(args.paths.tmp__cubes_output_dir)

    tools.start_logging(
        args.paths.tmp__cubes_output_dir,
        args=args,
        verbose=args.debug,
        filename="cube_extraction",
        log_header="CELLFINDER CUBE EXTRACTION LOG",
    )
    logging.info("Starting cube extraction")

    extract_cubes.main(args)
    logging.info("Finished. Total time taken: %s", datetime.now() - start_time)


if __name__ == "__main__":
    main()
