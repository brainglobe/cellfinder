"""This script reproduces the most common cellfinder workflow

It receives as an (optional) command line input the path to a configuration
json file, that holds the values of the required parameters for the workflow.

    python brainglobe_workflows/cellfinder.py --config path/to/config.json

If no input json file is passed as a configuration, the default
configuration defined at brainglobe_workflows/cellfinder/default_config.json
is used.

    python brainglobe_workflows/cellfinder.py

"""

import datetime
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import pooch
from brainglobe_utils.IO.cells import save_cells
from utils import (
    DEFAULT_JSON_CONFIG_PATH_CELLFINDER,
    config_parser,
    setup_logger,
)
from utils import __name__ as LOGGER_NAME

from cellfinder.core.main import main as cellfinder_run
from cellfinder.core.tools.IO import read_z_stack
from cellfinder.core.train.train_yml import depth_type

Pathlike = Union[str, os.PathLike]


@dataclass
class CellfinderConfig:
    """Define the parameters for the cellfinder workflow.

    There are three types of fields:
    - required attributes: must be provided, they do not have a default value;
    - optional attributes: they have a default value if not specified;
    - internal attributes: their names start with _, indicating these are
      private. Any functionality to update them should be a class method.
    """

    # Required parameters
    voxel_sizes: tuple[float, float, float]
    start_plane: int
    end_plane: int
    trained_model: Optional[os.PathLike]
    model_weights: Optional[os.PathLike]
    model: str
    batch_size: int
    n_free_cpus: int
    network_voxel_sizes: tuple[int, int, int]
    soma_diameter: int
    ball_xy_size: int
    ball_z_size: int
    ball_overlap_fraction: float
    log_sigma_size: float
    n_sds_above_mean_thresh: int
    soma_spread_factor: float
    max_cluster_size: int
    cube_width: int
    cube_height: int
    cube_depth: int
    network_depth: depth_type

    # Optional parameters

    # install path: default path for downloaded and output data
    _install_path: Pathlike = (
        Path.home() / ".brainglobe" / "workflows" / "cellfinder"
    )

    # input data paths
    # Note: if not specified, the signal and background data
    # are assumed to be under "signal" and "background"
    # dirs under _install_path/cellfinder_test_data/
    # (see __post_init__ method)
    input_data_dir: Optional[Pathlike] = None
    signal_subdir: Pathlike = "signal"
    background_subdir: Pathlike = "background"

    # output data paths
    # Note: if output_parent_dir is not specified,
    # it is assumed to be under _install_path
    # (see __post_init__ method)
    output_dir_basename: str = "cellfinder_output_"
    detected_cells_filename: str = "detected_cells.xml"
    output_parent_dir: Optional[Pathlike] = None

    # source of data to download
    # if not specified in JSON, it is set to None
    data_url: Optional[str] = None
    data_hash: Optional[str] = None

    # Internal parameters
    # even though these are optional we don't expect users to
    # change them
    _signal_dir_path: Optional[Pathlike] = None
    _background_dir_path: Optional[Pathlike] = None
    _list_signal_files: Optional[list] = None
    _list_background_files: Optional[list] = None
    _detected_cells_path: Pathlike = ""
    _output_path: Pathlike = ""

    def __post_init__(self: "CellfinderConfig"):
        """Executed after __init__ function.

        We use this method to define attributes of the data class
        as a function of other attributes.
        See https://peps.python.org/pep-0557/#post-init-processing

        The attributes added are input and output data paths

        Parameters
        ----------
        self : CellfinderConfig
            a CellfinderConfig instance
        """

        # Add input data paths to config
        self.add_input_paths()

        # Add output paths to config
        self.add_output_paths()

    def add_output_paths(self):
        """Adds output paths to the config

        Specifically, it adds:
        - output_parent_dir: set to a a timestamped output directory if not
          set in __init__();
        - _detected_cells_path: path to the output file

        Parameters
        ----------
        config : CellfinderConfig
            a cellfinder config
        """

        # Fill in output directory if not specified
        if self.output_parent_dir is None:
            self.output_parent_dir = Path(self._install_path)

        # Add to config the path to timestamped output directory
        timestamp = datetime.datetime.now()
        timestamp_formatted = timestamp.strftime("%Y%m%d_%H%M%S")
        self._output_path = Path(self.output_parent_dir) / (
            str(self.output_dir_basename) + timestamp_formatted
        )
        self._output_path.mkdir(
            parents=True,  # create any missing parents
            exist_ok=True,  # ignore FileExistsError exceptions
        )

        # Add to config the path to the output file
        self._detected_cells_path = (
            self._output_path / self.detected_cells_filename
        )

    def add_input_paths(self):
        """Adds input data paths to the config.

        Specifically, it adds:
        - input_data_dir: set to a default value if not set in __init__();
        - _signal_dir_path: full path to the directory with the signal files
        - _background_dir_path: full path to the directory with the
          background files.
        - _list_signal_files: list of signal files
        - _list_background_files: list of background files

        Parameters
        ----------
        config : CellfinderConfig
            a cellfinder config with input data files to be validated

        Notes
        -----
        The signal and background files are first searched locally at the
        given location. If not found, we attempt to download them from GIN
        and place them at the specified location (input_data_dir).

        - If both parent data directories (signal and background) exist
        locally, the lists of signal and background files are added to
        the config.
        - If exactly one of the parent data directories is missing, an error
        message is logged.
        - If neither of them exist, the data is retrieved from the provided GIN
        repository. If no URL or hash to GIN is provided, an error is thrown.

        """
        # Fetch logger
        logger = logging.getLogger(LOGGER_NAME)

        # Fill in input data directory if not specified
        if self.input_data_dir is None:
            self.input_data_dir = (
                Path(self._install_path) / "cellfinder_test_data"
            )

        # Fill in signal and background paths derived from 'input_data_dir'
        self._signal_dir_path = self.input_data_dir / Path(self.signal_subdir)
        self._background_dir_path = self.input_data_dir / Path(
            self.background_subdir
        )

        # Check if input data directories (signal and background) exist
        # locally.
        # If both directories exist, get list of signal and background files
        if (
            Path(self._signal_dir_path).exists()
            and Path(self._background_dir_path).exists()
        ):
            logger.info("Fetching input data from the local directories")

            self._list_signal_files = [
                f
                for f in Path(self._signal_dir_path).resolve().iterdir()
                if f.is_file()
            ]
            self._list_background_files = [
                f
                for f in Path(self._background_dir_path).resolve().iterdir()
                if f.is_file()
            ]

        # If exactly one of the input data directories is missing, print error
        elif (
            Path(self._signal_dir_path).resolve().exists()
            or Path(self._background_dir_path).resolve().exists()
        ):
            if not Path(self._signal_dir_path).resolve().exists():
                logger.error(
                    f"The directory {self._signal_dir_path} does not exist",
                )
            else:
                logger.error(
                    f"The directory {self._background_dir_path} "
                    "does not exist",
                )

        # If neither of the input data directories exist,
        # retrieve data from GIN repository and add list of files to config
        else:
            # Check if GIN URL and hash are defined (log error otherwise)
            if self.data_url and self.data_hash:
                # get list of files in GIN archive with pooch.retrieve
                list_files_archive = pooch.retrieve(
                    url=self.data_url,
                    known_hash=self.data_hash,
                    path=Path(
                        self.input_data_dir
                    ).parent,  # zip will be downloaded here
                    progressbar=True,
                    processor=pooch.Unzip(
                        extract_dir=Path(self.input_data_dir).stem,
                        # files are unpacked here, a dir
                        # *relative* to the path set in 'path'
                    ),
                )
                logger.info(
                    "Fetching input data from the provided GIN repository"
                )

                # Check signal and background parent directories exist now
                assert Path(self._signal_dir_path).resolve().exists()
                assert Path(self._background_dir_path).resolve().exists()

                # Add signal files to config
                self._list_signal_files = [
                    f
                    for f in list_files_archive
                    if f.startswith(
                        str(Path(self._signal_dir_path).resolve()),
                    )
                ]

                # Add background files to config
                self._list_background_files = [
                    f
                    for f in list_files_archive
                    if f.startswith(
                        str(Path(self._background_dir_path).resolve()),
                    )
                ]
            # If one of URL/hash to GIN repo not defined, throw an error
            else:
                logger.error(
                    "Input data not found locally, and URL/hash to "
                    "GIN repository not provided",
                )


def read_cellfinder_config(
    input_config_path: str, log_on: bool = False
) -> CellfinderConfig:
    """Instantiate a CellfinderConfig from the input json file.

    Assumes config is json serializable.

    Parameters
    ----------
    input_config_path : Path
        Absolute path to a cellfinder config file
    log_on : bool, optional
        whether to log the info messages from reading the config
        to the logger, by default False

    Returns
    -------
    CellfinderConfig:
        The cellfinder config object, populated with data from the input
    """

    # read input config
    with open(input_config_path) as cfg:
        config_dict = json.load(cfg)
    config = CellfinderConfig(**config_dict)

    # print config's origin to log if required
    if log_on:
        logger = logging.getLogger(LOGGER_NAME)
        logger.info(f"Input config read from {input_config_path}")
        if input_config_path == DEFAULT_JSON_CONFIG_PATH_CELLFINDER:
            logger.info("Using default config file")

    return config


def setup(input_config_path: str) -> CellfinderConfig:
    # setup logger
    _ = setup_logger()

    # read config
    cfg = read_cellfinder_config(input_config_path)

    return cfg


def run_workflow_from_cellfinder_run(cfg: CellfinderConfig):
    """Run workflow based on the cellfinder.core.main.main()
    function.

    The steps are:
    1. Read the input signal and background data as two separate
       Dask arrays (or in-memory numpy arrays if single file tiff stack).
    2. Run the main cellfinder pipeline on the input arrays,
       with the parameters defined in the input configuration (cfg).
    3. Save the detected cells as an xml file to the location specified in
       the input configuration (cfg).

    Parameters
    ----------
    cfg : CellfinderConfig
        a class with the required setup methods and parameters for
        the cellfinder workflow
    """
    # Read input data as Dask or numpy arrays
    signal_array = read_z_stack(str(cfg._signal_dir_path))
    background_array = read_z_stack(str(cfg._background_dir_path))

    # Run main analysis using `cellfinder_run`
    detected_cells = cellfinder_run(
        signal_array,
        background_array,
        cfg.voxel_sizes,
        cfg.start_plane,
        cfg.end_plane,
        cfg.trained_model,
        cfg.model_weights,
        cfg.model,
        cfg.batch_size,
        cfg.n_free_cpus,
        cfg.network_voxel_sizes,
        cfg.soma_diameter,
        cfg.ball_xy_size,
        cfg.ball_z_size,
        cfg.ball_overlap_fraction,
        cfg.log_sigma_size,
        cfg.n_sds_above_mean_thresh,
        cfg.soma_spread_factor,
        cfg.max_cluster_size,
        cfg.cube_width,
        cfg.cube_height,
        cfg.cube_depth,
        cfg.network_depth,
    )

    # Save results to xml file
    save_cells(
        detected_cells,
        cfg._detected_cells_path,
    )


def main(
    input_config: str = str(DEFAULT_JSON_CONFIG_PATH_CELLFINDER),
) -> CellfinderConfig:
    """Setup and run cellfinder workflow.

    This function runs the setup steps required
    to run the cellfinder workflow, and the
    workflow itself. Note that only the workflow
    will be benchmarked.

    Parameters
    ----------
    input_config : str, optional
        Absolute path to input config file,
        by default str(DEFAULT_JSON_CONFIG_PATH_CELLFINDER)

    Returns
    -------
    cfg : CellfinderConfig
        a class with the required setup methods and parameters for
        the cellfinder workflow
    """
    # run setup
    cfg = setup(input_config)

    # run workflow
    run_workflow_from_cellfinder_run(cfg)  # only this will be benchmarked

    return cfg


if __name__ == "__main__":
    # parse CLI arguments
    args = config_parser(
        sys.argv[1:],
        str(DEFAULT_JSON_CONFIG_PATH_CELLFINDER),
    )

    # run setup and workflow
    _ = main(args.config)
