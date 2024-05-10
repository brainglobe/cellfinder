"""Pytest fixtures shared across unit and integration tests"""

import json
from pathlib import Path

import pooch
import pytest


@pytest.fixture(autouse=True)
def mock_home_directory(monkeypatch: pytest.MonkeyPatch):
    """
    Monkeypatch pathlib.Path.home()

    Instead of returning the usual home path, the
    monkeypatched version returns the path to
    Path.home() / ".brainglobe-tests"

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
       a monkeypatch fixture
    """
    # define mock home path
    home_path = Path.home()  # actual home path
    mock_home_path = home_path / ".brainglobe-tests"

    # create mock home directory if it doesn't exist
    if not mock_home_path.exists():
        mock_home_path.mkdir()

    # monkeypatch Path.home() to point to the mock home
    def mock_home():
        return mock_home_path

    monkeypatch.setattr(Path, "home", mock_home)


@pytest.fixture()
def default_input_config_cellfinder() -> Path:
    """
    Fixture for the path to the default input
    configuration file for the cellfinder workflow

    Returns
    -------
    Path
        Path to default input config

    """
    from cellfinder.workflows.utils import DEFAULT_JSON_CONFIG_PATH_CELLFINDER

    return DEFAULT_JSON_CONFIG_PATH_CELLFINDER


@pytest.fixture(scope="session")
def cellfinder_GIN_data() -> dict:
    """
    Fixture for the location of the test data in the GIN repository

    Returns
    -------
    dict
        URL and hash of the GIN repository with the cellfinder test data
    """
    return {
        "url": "https://gin.g-node.org/BrainGlobe/test-data/raw/master/cellfinder/cellfinder-test-data.zip",
        "hash": "b0ef53b1530e4fa3128fcc0a752d0751909eab129d701f384fc0ea5f138c5914",  # noqa
    }


@pytest.fixture()
def GIN_default_location() -> Path:
    """A fixture returning a path to the default location
    where GIN data is downloaded.

    Returns
    -------
    Path
        path to the default location where to download GIN data
    """

    return (
        Path.home()
        / ".brainglobe"
        / "workflows"
        / "cellfinder"
        / "cellfinder_test_data"
    )


@pytest.fixture()
def config_GIN_dict(
    cellfinder_GIN_data: dict,
    default_input_config_cellfinder: Path,
    GIN_default_location: Path,
) -> dict:
    """
    Fixture that returns a config as a dictionary, pointing to the location
    where the GIN data would be by default, and download the data there.

    If the file exists in the given path and the hash matches, pooch.retrieve()
    will not download the file and instead its path is returned.

    Parameters
    ----------
    cellfinder_GIN_data : dict
        dictionary with the location of the test data in the GIN repository
    default_input_config_cellfinder : Path
        path to the default input configuration file for the cellfinder
        workflow
    GIN_default_location : Path
        path to the default location where to download GIN data

    Returns
    -------
    dict
        dictionary with the config for a workflow that uses the downloaded
        GIN data
    """

    # read default config as a dictionary
    with open(default_input_config_cellfinder) as cfg:
        config_dict = json.load(cfg)

    # modify the default config:
    # - add url
    # - add data hash
    # - remove input_data_dir if present
    config_dict["data_url"] = cellfinder_GIN_data["url"]
    config_dict["data_hash"] = cellfinder_GIN_data["hash"]
    if "input_data_dir" in config_dict.keys():
        del config_dict["input_data_dir"]

    # download GIN data to default location for GIN
    # if the file exists in the given path and the hash matches,
    # it will not be downloaded and the absolute path to the file is returned.
    pooch.retrieve(
        url=cellfinder_GIN_data["url"],
        known_hash=cellfinder_GIN_data["hash"],
        path=GIN_default_location.parent,  # path to download zip to
        progressbar=True,
        processor=pooch.Unzip(extract_dir=GIN_default_location.stem),
    )

    return config_dict


@pytest.fixture()
def config_local_dict(
    config_GIN_dict: dict,
    GIN_default_location: Path,
) -> dict:
    """
    Fixture that returns a config as a dictionary, pointing to a local dataset.

    The data is copied to the local directory from the
    default location used in the config_GIN_dict fixture.

    Parameters
    ----------
    config_GIN_dict : dict
        dictionary with the config for a workflow that uses the downloaded
        GIN data
    GIN_default_location : Path
        path to the default location where to download GIN data\

    Returns
    -------
    dict
        dictionary with the config for a workflow that uses local data
    """
    import shutil

    # copy GIN config as a dictionary
    config_dict = config_GIN_dict.copy()

    # modify the GIN config:
    # - remove url
    # - remove data hash
    # - set input data directory to a local directory under home
    config_dict["data_url"] = None
    config_dict["data_hash"] = None
    config_dict["input_data_dir"] = str(Path.home() / "local_cellfinder_data")

    # copy data from default GIN location to the local location
    shutil.copytree(
        GIN_default_location,
        config_dict["input_data_dir"],
        dirs_exist_ok=True,
    )

    return config_dict


@pytest.fixture()
def config_GIN_json(config_GIN_dict: dict, tmp_path: Path) -> Path:
    """
    Fixture that returns a config as a JSON file path, that points to GIN
    downloaded data as input

    Parameters
    ----------
    config_GIN_dict : dict
        dictionary with the config for a workflow that uses the downloaded
        GIN data
    tmp_path : Path
        Pytest fixture providing a temporary path

    Returns
    -------
    Path
        path to a cellfinder config JSON file
    """
    # define location of input config file
    config_file_path = tmp_path / "input_config.json"

    # write config dict to that location
    with open(config_file_path, "w") as js:
        json.dump(config_GIN_dict, js)

    return config_file_path


@pytest.fixture()
def config_local_json(config_local_dict: dict, tmp_path: Path) -> Path:
    """
    Fixture that returns config as a JSON file path, that points to local data
    as input

    Parameters
    ----------
    config_local_dict : dict
        _description_
    tmp_path : Path
        Pytest fixture providing a temporary path

    Returns
    -------
    Path
        path to a cellfinder config JSON file
    """
    # define location of input config file
    config_file_path = tmp_path / "input_config.json"

    # write config dict to that location
    with open(config_file_path, "w") as js:
        json.dump(config_local_dict, js)

    return config_file_path
