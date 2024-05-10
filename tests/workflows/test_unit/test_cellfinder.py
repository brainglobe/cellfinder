import json
import logging
import re
from pathlib import Path

import pytest

from cellfinder.workflows.utils import setup_logger


@pytest.fixture()
def config_force_GIN_dict(
    config_GIN_dict: dict,
    tmp_path: Path,
    GIN_default_location: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> dict:
    """
    Fixture returning a config as a dictionary, which has a
    Pytest-generated temporary directory as input data location,
    and that monkeypatches pooch.retrieve()

    Since there is no data at the input_data_dir location, the GIN download
    will be triggered, but the monkeypatched pooch.retrieve() will copy the
    files rather than download them.

    Parameters
    ----------
    config_GIN_dict : dict
        dictionary with the config for a workflow that uses the downloaded
        GIN data
    tmp_path : Path
        path to pytest-generated temporary directory
    GIN_default_location : Path
        path to the default location where to download GIN data
    monkeypatch : pytest.MonkeyPatch
        a monkeypatch fixture

    Returns
    -------
    dict
        dictionary with the config for a workflow that triggers the downloaded
        GIN data
    """

    import shutil

    import pooch

    # read GIN config as dict
    config_dict = config_GIN_dict.copy()

    # point to a temporary directory in input_data_dir
    config_dict["input_data_dir"] = str(tmp_path)

    # monkeypatch pooch.retrieve()
    # when called copy GIN downloaded data, instead of downloading it
    def mock_pooch_download(
        url="", known_hash="", path="", progressbar="", processor=""
    ):
        # Copy destination
        GIN_copy_destination = tmp_path

        # copy only relevant subdirectories
        for subdir in ["signal", "background"]:
            shutil.copytree(
                GIN_default_location / subdir,  # src
                GIN_copy_destination / subdir,  # dest
                dirs_exist_ok=True,
            )

        # List of files in destination
        list_of_files = [
            str(f) for f in GIN_copy_destination.glob("**/*") if f.is_file()
        ]
        list_of_files.sort()

        return list_of_files

    # monkeypatch pooch.retreive with mock_pooch_download()
    monkeypatch.setattr(pooch, "retrieve", mock_pooch_download)

    return config_dict


@pytest.fixture()
def config_missing_signal_dict(config_local_dict: dict) -> dict:
    """
    Fixture that returns a config as a dictionary, pointing to a local dataset,
    whose signal directory does not exist

    Parameters
    ----------
    config_local_dict : _type_
        dictionary with the config for a workflow that uses local data

    Returns
    -------
    dict
        dictionary with the config for a workflow that uses local data, but
        whose signal directory does not exist.
    """
    config_dict = config_local_dict.copy()
    config_dict["signal_subdir"] = "_"

    return config_dict


@pytest.fixture()
def config_missing_background_dict(config_local_dict: dict) -> dict:
    """
    Fixture that returns a config as a dictionary, pointing to a local dataset,
    whose background directory does not exist

    Parameters
    ----------
    config_local_dict : dict
        dictionary with the config for a workflow that uses local data

    Returns
    -------
    dict
        dictionary with the config for a workflow that uses local data, but
        whose background directory does not exist.
    """
    config_dict = config_local_dict.copy()
    config_dict["background_subdir"] = "_"

    return config_dict


@pytest.fixture()
def config_not_GIN_nor_local_dict(config_local_dict: dict) -> dict:
    """
    Fixture that returns a config as a dictionary, whose input_data_dir
    directory does not exist and with no references to a GIN dataset.

    Parameters
    ----------
    config_local_dict : dict
        dictionary with the config for a workflow that uses local data

    Returns
    -------
    dict
        dictionary with the config for a workflow that uses local data, but
        whose input_data_dir directory does not exist and with no references
        to a GIN dataset.
    """
    config_dict = config_local_dict.copy()
    config_dict["input_data_dir"] = "_"

    config_dict["data_url"] = None
    config_dict["data_hash"] = None

    return config_dict


@pytest.mark.parametrize(
    "input_config_dict, message_pattern",
    [
        (
            "config_force_GIN_dict",
            "Fetching input data from the provided GIN repository",
        ),
        (
            "config_local_dict",
            "Fetching input data from the local directories",
        ),
        (
            "config_missing_signal_dict",
            "The directory .+ does not exist$",
        ),
        ("config_missing_background_dict", "The directory .+ does not exist$"),
        (
            "config_not_GIN_nor_local_dict",
            "Input data not found locally, and URL/hash to "
            "GIN repository not provided",
        ),
    ],
)
def test_add_input_paths(
    caplog: pytest.LogCaptureFixture,
    input_config_dict: dict,
    message_pattern: str,
    request: pytest.FixtureRequest,
):
    """
    Test the addition of signal and background files to the cellfinder config

    Parameters
    ----------
    caplog : pytest.LogCaptureFixture
        Pytest fixture to capture the logs during testing
    input_config_dict : dict
        input config as a dict
    message_pattern : str
        Expected pattern in the log
    request : pytest.FixtureRequest
        Pytest fixture to enable requesting fixtures by name
    """

    from cellfinder.workflows.cellfinder import CellfinderConfig

    # instantiate custom logger
    _ = setup_logger()

    # instantiate config object
    _ = CellfinderConfig(**request.getfixturevalue(input_config_dict))

    # check log messages
    assert len(caplog.messages) > 0
    out = re.fullmatch(message_pattern, caplog.messages[-1])
    assert out is not None
    assert out.group() is not None


@pytest.mark.parametrize(
    "input_config_path, message",
    [
        ("default_input_config_cellfinder", "Using default config file"),
        ("config_local_json", "Input config read from"),
    ],
)
def test_read_cellfinder_config(
    input_config_path: str,
    message: str,
    caplog: pytest.LogCaptureFixture,
    request: pytest.FixtureRequest,
):
    """
    Test reading a cellfinder config

    Parameters
    ----------
    input_config_path : str
        path to input config file
    message : str
        Expected message in the log
    caplog : pytest.LogCaptureFixture
        Pytest fixture to capture the logs during testing
    request : pytest.FixtureRequest
        Pytest fixture to enable requesting fixtures by name

    """
    from cellfinder.workflows.cellfinder import read_cellfinder_config

    # instantiate custom logger
    _ = setup_logger()

    # read Cellfinder config
    config = read_cellfinder_config(
        request.getfixturevalue(input_config_path), log_on=True
    )

    # read json as dict
    with open(request.getfixturevalue(input_config_path)) as cfg:
        config_dict = json.load(cfg)

    # check keys of dictionary are a subset of Cellfinder config attributes
    assert all(
        [ky in config.__dataclass_fields__.keys() for ky in config_dict.keys()]
    )

    # check logs
    assert message in caplog.text

    # check all signal files exist
    assert config._list_signal_files
    assert all([Path(f).is_file() for f in config._list_signal_files])

    # check all background files exist
    assert config._list_background_files
    assert all([Path(f).is_file() for f in config._list_background_files])

    # check output directory exists
    assert Path(config._output_path).resolve().is_dir()

    # check output directory name has correct format
    out = re.fullmatch(
        str(config.output_dir_basename) + "\\d{8}_\\d{6}$",
        Path(config._output_path).stem,
    )
    assert out is not None
    assert out.group() is not None

    # check output file path is as expected
    assert (
        Path(config._detected_cells_path)
        == Path(config._output_path) / config.detected_cells_filename
    )


@pytest.mark.parametrize(
    "input_config",
    [
        "default_input_config_cellfinder",
        "config_local_json",
        "config_GIN_json",
    ],
)
def test_setup(
    input_config: str, custom_logger_name: str, request: pytest.FixtureRequest
):
    """
    Test the full setup for the cellfinder workflow.

    Parameters
    ----------
    input_config : str
        Path to input config file
    custom_logger_name : str
        Name of custom logger
    request : pytest.FixtureRequest
        Pytest fixture to enable requesting fixtures by name
    """
    from cellfinder.workflows.cellfinder import CellfinderConfig
    from cellfinder.workflows.cellfinder import setup as setup_workflow

    # run setup on default configuration
    cfg = setup_workflow(str(request.getfixturevalue(input_config)))

    # check logger exists
    logger = logging.getLogger(custom_logger_name)
    assert logger.level == logging.DEBUG
    assert logger.hasHandlers()

    # check config is CellfinderConfig
    assert isinstance(cfg, CellfinderConfig)


@pytest.mark.parametrize(
    "input_config",
    [
        "default_input_config_cellfinder",
        "config_local_json",
        "config_GIN_json",
    ],
)
def test_run_workflow_from_cellfinder_run(
    input_config: str, request: pytest.FixtureRequest
):
    """
    Test running cellfinder workflow

    Parameters
    ----------
    input_config : str
        Path to input config json file
    request : pytest.FixtureRequest
        Pytest fixture to enable requesting fixtures by name
    """
    from cellfinder.workflows.cellfinder import (
        run_workflow_from_cellfinder_run,
    )
    from cellfinder.workflows.cellfinder import setup as setup_workflow

    # run setup
    cfg = setup_workflow(str(request.getfixturevalue(input_config)))

    # run workflow
    run_workflow_from_cellfinder_run(cfg)

    # check output files exist
    assert Path(cfg._detected_cells_path).is_file()
