import os
from pathlib import Path
from typing import Literal

import pooch

from cellfinder import DEFAULT_CELLFINDER_DIRECTORY

DEFAULT_DOWNLOAD_DIRECTORY = DEFAULT_CELLFINDER_DIRECTORY / "models"


MODEL_URL = "https://gin.g-node.org/cellfinder/models/raw/master"
HF_1CH_URL = "https://huggingface.co/brainglobe/cellfinder_single_channel_default/resolve/main"  # noqa: E501

model_filenames = {
    "resnet50_tv": "resnet50_tv.h5",
    "resnet50_all": "resnet50_weights.h5",
    "resnet50_1ch": "resnet50_single_channel.keras",
}

model_urls = {
    "resnet50_tv": f"{MODEL_URL}/resnet50_tv.h5",
    "resnet50_all": f"{MODEL_URL}/resnet50_weights.h5",
    "resnet50_1ch": f"{HF_1CH_URL}/resnet50_single_channel.keras",
}

model_hashes = {
    "resnet50_tv": "63d36af456640590ba6c896dc519f9f29861015084f4c40777a54c18c1fc4edd",  # noqa: E501
    "resnet50_all": None,
    "resnet50_1ch": "4c0af5e916195603266fc18686a84e7156683cbd6e91b27385e9d6e0b5ef5a55",  # noqa: E501
}


model_type = Literal["resnet50_tv", "resnet50_all", "resnet50_1ch"]


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
        url=model_urls[model_name],
        known_hash=model_hashes[model_name],
        path=download_path,
        fname=filename,
        progressbar=True,
    )

    return Path(model_path)
