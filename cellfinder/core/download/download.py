import os
from pathlib import Path
from typing import Literal

import pooch

from cellfinder import DEFAULT_CELLFINDER_DIRECTORY

DEFAULT_DOWNLOAD_DIRECTORY = DEFAULT_CELLFINDER_DIRECTORY / "models"


MODEL_URL = "https://gin.g-node.org/cellfinder/models/raw/master"
HF_1CH_URL = "https://huggingface.co/brainglobe/cellfinder_single_channel_default/resolve/main"  # noqa: E501
HF_2D_URL = "https://huggingface.co/brainglobe/cellfinder_2d_default/resolve/main"  # noqa: E501

model_filenames = {
    "resnet50_tv": "resnet50_tv.h5",
    "resnet50_all": "resnet50_weights.h5",
    "resnet50_1ch": "resnet50_single_channel.keras",
    "resnet50_2d": "resnet50_2d.keras",
    "resnet50_2d_1ch": "resnet50_2d_single_channel.keras",
}

model_urls = {
    "resnet50_tv": f"{MODEL_URL}/resnet50_tv.h5",
    "resnet50_all": f"{MODEL_URL}/resnet50_weights.h5",
    "resnet50_1ch": f"{HF_1CH_URL}/resnet50_single_channel.keras",
    "resnet50_2d": f"{HF_2D_URL}/resnet50_2d.keras",
    "resnet50_2d_1ch": f"{HF_2D_URL}/resnet50_2d_single_channel.keras",
}

model_hashes = {
    "resnet50_tv": "63d36af456640590ba6c896dc519f9f29861015084f4c40777a54c18c1fc4edd",  # noqa: E501
    "resnet50_all": None,
    "resnet50_1ch": "4c0af5e916195603266fc18686a84e7156683cbd6e91b27385e9d6e0b5ef5a55",  # noqa: E501
    "resnet50_2d": "6e40bf25452e99ce0334d85c413ac5029b113adef1f81f2fa46d0d546bb880f6",  # noqa: E501
    "resnet50_2d_1ch": "ae71226de27d93ec9238a09eda4c47e77b2435a840cbbf512eefdb957590e8b0",  # noqa: E501
}

model_dimensions = {
    "resnet50_tv": 3,
    "resnet50_all": 3,
    "resnet50_1ch": 3,
    "resnet50_2d": 2,
    "resnet50_2d_1ch": 2,
}

model_channels = {
    "resnet50_tv": 2,
    "resnet50_all": 2,
    "resnet50_1ch": 1,
    "resnet50_2d": 2,
    "resnet50_2d_1ch": 1,
}


model_type = Literal[
    "resnet50_tv",
    "resnet50_all",
    "resnet50_1ch",
    "resnet50_2d",
    "resnet50_2d_1ch",
]

DEFAULT_MODEL = "resnet50_tv"


def models_for_dimensions(dimensions: int) -> list:
    """
    The registered model names that build a `dimensions`D network.
    """
    return [
        name for name, dims in model_dimensions.items() if dims == dimensions
    ]


def default_model(dimensions: int, has_background: bool) -> str:
    """
    The pretrained model matching the processing mode and channel count.
    """
    for name in models_for_dimensions(dimensions):
        if model_channels[name] == (2 if has_background else 1):
            return name
    raise ValueError(
        f"No pretrained model for dimensions={dimensions} with "
        f"{'two' if has_background else 'one'} channel(s)."
    )


def validate_model_dimensions(model_name: str, dimensions: int) -> None:
    """
    Raise `ValueError` if `model_name` does not build a `dimensions`D network.
    """
    if model_name not in model_dimensions:
        return
    if model_dimensions[model_name] != dimensions:
        raise ValueError(
            f"Model {model_name!r} is "
            f"{model_dimensions[model_name]}D, but dimensions="
            f"{dimensions} was requested. Available models: "
            f"{models_for_dimensions(dimensions)}."
        )


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
