from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path

from cellfinder.core.download.download import (
    DEFAULT_DOWNLOAD_DIRECTORY,
    amend_user_configuration,
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
        "--no-amend-config",
        dest="no_amend_config",
        action="store_true",
        help="Don't amend the config file",
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
    Run the main download function, and optionally amend the user
    configuration.

    """
    args = get_parser().parse_args()
    model_path = download_models(args.model, args.install_path)

    if not args.no_amend_config:
        amend_user_configuration(new_model_path=model_path)


if __name__ == "__main__":
    main()
