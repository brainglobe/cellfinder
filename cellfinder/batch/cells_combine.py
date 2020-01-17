"""
cells_combine
==================
Script to combine cell files into one
"""
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

SUPPORTED_FORMATS = [".xml", ".csv", ".yml"]


def main():
    args = cell_combine_cli_parser().parse_args()
    pass


def cell_combine_cli_parser():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser = cli_parse(parser)

    return parser


def cli_parse(parser):
    cli_parser = parser.add_argument_group("Cell combine options")
    cli_parser.add_argument(
        "--cells",
        dest="cells_file_path",
        type=str,
        nargs="+",
        required=True,
        help="Directory or list of cell files to combine. Supported formats "
        "are: {} ".format(SUPPORTED_FORMATS),
    )

    cli_parser.add_argument(
        "--transform-all",
        dest="transform_all",
        action="store_true",
        help="Combine all cell positions (including artifacts). By default, "
        "only positions classified as cells will be included. ",
    )

    cli_parser.add_argument(
        "-o",
        "--output-dir",
        dest="output_dir",
        type=str,
        required=True,
        help="Top-level directory to save the cubes into",
    )
    return parser


if __name__ == "__main__":
    main()
