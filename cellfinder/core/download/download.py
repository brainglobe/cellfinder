import os
from pathlib import Path
from typing import Literal

import pooch
from brainglobe_utils.general.config import get_config_obj

from cellfinder import DEFAULT_CELLFINDER_DIRECTORY
from cellfinder.core.tools.source_files import (
    default_configuration_path,
    user_specific_configuration_path,
)

DEFAULT_DOWNLOAD_DIRECTORY = DEFAULT_CELLFINDER_DIRECTORY / "models"


MODEL_URL = "https://gin.g-node.org/cellfinder/models/raw/master"

model_filenames = {
    "resnet50_tv": "resnet50_tv.h5",
    "resnet50_all": "resnet50_weights.h5",
}

model_hashes = {
    "resnet50_tv": "63d36af456640590ba6c896dc519f9f29861015084f4c40777a54c18c1fc4edd",  # noqa: E501
    "resnet50_all": None,
}


model_type = Literal["resnet50_tv", "resnet50_all"]


def download_models(
    model_name: model_type, download_path: os.PathLike
) -> Path:
    """
    For a given model name and download path, download the model file
    and return the path to the downloaded file.

    Parameters
    ----------
    model_name : model_type
        The name of the model to be downloaded.
    download_path : os.PathLike
        The path where the model file will be downloaded.

    Returns
    -------
    Path
        The path to the downloaded model file.

    """

    download_path = Path(download_path)
    filename = model_filenames[model_name]
    model_path = pooch.retrieve(
        url=f"{MODEL_URL}/{filename}",
        known_hash=model_hashes[model_name],
        path=download_path,
        fname=filename,
        progressbar=True,
    )

    return Path(model_path)


def amend_user_configuration(new_model_path=None) -> None:
    """
    Amends the user configuration to contain the configuration
    in new_model_path, if specified.

    Parameters
    ----------
    new_model_path : Path, optional
        The path to the new model configuration.
    """
    print("(Over-)writing custom user configuration")

    original_config = default_configuration_path()
    new_config = user_specific_configuration_path()
    if new_model_path is not None:
        write_model_to_config(new_model_path, original_config, new_config)


def write_model_to_config(new_model_path, orig_config, custom_config):
    """
    Update the model path in the custom configuration file, by
    reading the lines in the original configuration file, replacing
    the line starting with "model_path =" and writing these
    lines to the custom file.

    Parameters
    ----------
    new_model_path : str
        The new path to the model.
    orig_config : str
        The path to the original configuration file.
    custom_config : str
        The path to the custom configuration file to be created.

    Returns
    -------
    None

    """
    config_obj = get_config_obj(orig_config)
    model_conf = config_obj["model"]
    orig_path = model_conf["model_path"]

    with open(orig_config, "r") as in_conf:
        data = in_conf.readlines()
    for i, line in enumerate(data):
        data[i] = line.replace(
            f"model_path = '{orig_path}", f"model_path = '{new_model_path}"
        )

    custom_config_path = Path(custom_config)
    custom_config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(custom_config, "w") as out_conf:
        out_conf.writelines(data)
