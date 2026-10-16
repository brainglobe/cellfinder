from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path

from cellfinder.core.download.download import (
    DEFAULT_DOWNLOAD_DIRECTORY,
    download_models,
)


def download_parser(parser: ArgumentParser) -> ArgumentParser:
    """
    Configure the argument parser for downloading files.

    Parameters
    ----------
    parser : ArgumentParser
        The argument parser to configure.

    Returns
    -------
    ArgumentParser
        The configured argument parser.

    """

    parser.add_argument(
        "--install-path",
        dest="install_path",
        type=Path,
        default=DEFAULT_DOWNLOAD_DIRECTORY,
        help="The path to install files to.",
    )
    parser.add_argument(
        "--model",
        dest="model",
        type=str,
        default="resnet50_tv",
        help="The model to use",
    )
    return parser


def get_parser() -> ArgumentParser:
    """
    Create an argument parser for downloading files.

    Returns
    -------
    ArgumentParser
        The configured argument parser.

    """
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser = download_parser(parser)
    return parser


def main() -> None:
    """
    Run the main download function.

    """
    args = get_parser().parse_args()
    download_models(args.model, args.install_path)


if __name__ == "__main__":
    main()
