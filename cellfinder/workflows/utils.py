import argparse
import logging
import sys
from pathlib import Path
from typing import List

DEFAULT_JSON_CONFIGS_PATH = Path(__file__).resolve().parent / "configs"

DEFAULT_JSON_CONFIG_PATH_CELLFINDER = (
    DEFAULT_JSON_CONFIGS_PATH / "cellfinder.json"
)


def setup_logger() -> logging.Logger:
    """Setup a logger for workflow runs

    The logger's level is set to DEBUG, and it
    is linked to a handler that writes to the
    console. This utility function helps run
    workflows, and test their logs, in a
    consistent way.

    Returns
    -------
    logging.Logger
        a logger object configured for workflow runs
    """
    # define handler that writes to stdout
    console_handler = logging.StreamHandler(sys.stdout)
    console_format = logging.Formatter("%(name)s %(levelname)s: %(message)s")
    console_handler.setFormatter(console_format)
    console_handler.set_name("console_handler")

    # define logger and link to handler
    logger = logging.getLogger(
        __name__
    )  # if imported as a module, the logger is named after the module
    logger.setLevel(logging.DEBUG)
    logger.addHandler(console_handler)
    return logger


def config_parser(
    argv: List[str],
    default_config: str,
) -> argparse.Namespace:
    """Define argument parser for a workflow script.

    The only CLI argument defined in the parser is
    the input config file. The list of input arguments
    `argv` can be an empty list.

    Both the list of input arguments and the default config to use if
    no config is specified must be passed as an input to this
    function.

    Parameters
    ----------
    argv_ : List[str]
        _description_
    default_config : str
        _description_

    Returns
    -------
    args : argparse.Namespace
        command line input arguments parsed
    """

    # initialise argument parser
    parser = argparse.ArgumentParser(
        description=(
            """
            To launch the cellfinder workflow with default parameters, run:
                `cellfinder-workflow`.
            The default parameters are those specifed in brainglobe_workflows/
            cellfinder/configs/cellfinder.json.


            To launch the cellfinder workflow with a specific set of input
            parameters, run:
                `cellfinder-workflow --config path/to/config.json`,
            where `path/to/input/config.json` is the json file with the
            desired parameters.
            """
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # add arguments
    parser.add_argument(
        "-c",
        "--config",
        default=default_config,
        type=str,
        metavar="CONFIG",  # a name for usage messages
        help="",
    )

    # build parser object
    args = parser.parse_args(argv)

    return args
