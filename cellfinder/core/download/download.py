import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Literal, Optional

import pooch
from huggingface_hub import hf_hub_download

from cellfinder import DEFAULT_CELLFINDER_DIRECTORY

DEFAULT_DOWNLOAD_DIRECTORY = DEFAULT_CELLFINDER_DIRECTORY / "models"


MODEL_URL = "https://gin.swc.ucl.ac.uk/brainglobe/cellfinder/raw/main/models"
HF_1CH_URL = "https://huggingface.co/brainglobe/cellfinder_single_channel_default/resolve/main"  # noqa: E501
HF_2D_URL = "https://huggingface.co/brainglobe/cellfinder_2d_default/resolve/main"  # noqa: E501


@dataclass(frozen=True)
class PretrainedModelInfo:
    """Metadata describing a pre-trained model."""

    filename: str
    hf_repo_id: Optional[str] = None
    gin_url: Optional[str] = None
    known_hash: Optional[str] = None
    dimensions: int = 3
    channels: int = 2


PRETRAINED_MODELS: Dict[str, PretrainedModelInfo] = {
    "resnet50_tv": PretrainedModelInfo(
        filename="resnet50_tv.h5",
        gin_url=f"{MODEL_URL}/resnet50_tv.h5",
        known_hash="63d36af456640590ba6c896dc519f9f29861015084f4c40777a54c18c1fc4edd",  # noqa: E501
        dimensions=3,
        channels=2,
    ),
    "resnet50_all": PretrainedModelInfo(
        filename="resnet50_weights.h5",
        gin_url=f"{MODEL_URL}/resnet50_weights.h5",
        known_hash=None,
        dimensions=3,
        channels=2,
    ),
    "resnet50_1ch": PretrainedModelInfo(
        filename="resnet50_single_channel.keras",
        hf_repo_id="brainglobe/cellfinder_single_channel_default",
        known_hash="4c0af5e916195603266fc18686a84e7156683cbd6e91b27385e9d6e0b5ef5a55",  # noqa: E501
        dimensions=3,
        channels=1,
    ),
    "resnet50_2d": PretrainedModelInfo(
        filename="resnet50_2d.keras",
        hf_repo_id="brainglobe/cellfinder_2d_default",
        known_hash="6e40bf25452e99ce0334d85c413ac5029b113adef1f81f2fa46d0d546bb880f6",  # noqa: E501
        dimensions=2,
        channels=2,
    ),
    "resnet50_2d_1ch": PretrainedModelInfo(
        filename="resnet50_2d_single_channel.keras",
        hf_repo_id="brainglobe/cellfinder_2d_default",
        known_hash="ae71226de27d93ec9238a09eda4c47e77b2435a840cbbf512eefdb957590e8b0",  # noqa: E501
        dimensions=2,
        channels=1,
    ),
}

model_filenames = {k: v.filename for k, v in PRETRAINED_MODELS.items()}
model_urls = {
    "resnet50_tv": f"{MODEL_URL}/resnet50_tv.h5",
    "resnet50_all": f"{MODEL_URL}/resnet50_weights.h5",
    "resnet50_1ch": f"{HF_1CH_URL}/resnet50_single_channel.keras",
    "resnet50_2d": f"{HF_2D_URL}/resnet50_2d.keras",
    "resnet50_2d_1ch": f"{HF_2D_URL}/resnet50_2d_single_channel.keras",
}
model_hashes = {k: v.known_hash for k, v in PRETRAINED_MODELS.items()}
model_dimensions = {k: v.dimensions for k, v in PRETRAINED_MODELS.items()}
model_channels = {k: v.channels for k, v in PRETRAINED_MODELS.items()}


model_type = Literal[
    "resnet50_tv",
    "resnet50_all",
    "resnet50_1ch",
    "resnet50_2d",
    "resnet50_2d_1ch",
]


def download_models(
    model_name: model_type,
    download_path: os.PathLike,
    local_files_only: bool = False,
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
    local_files_only : bool, optional
        If True, only use locally cached files and do not attempt to
        reach remote servers. Defaults to False.

    Returns
    -------
    Path
        The path to the downloaded model file.

    """
    if model_name not in PRETRAINED_MODELS:
        raise ValueError(
            f"Model {model_name!r} not recognized. "
            f"Available models: {sorted(PRETRAINED_MODELS.keys())}"
        )

    info = PRETRAINED_MODELS[model_name]
    download_path = Path(download_path)

    if info.hf_repo_id is not None:
        model_path = hf_hub_download(
            repo_id=info.hf_repo_id,
            filename=info.filename,
            cache_dir=download_path,
            local_files_only=local_files_only,
        )
        return Path(model_path)

    url = info.gin_url or model_urls[model_name]
    filename = info.filename
    retrieve_kwargs = dict(
        known_hash=info.known_hash,
        path=download_path,
        fname=filename,
        progressbar=True,
    )
    try:
        # Try the secure endpoint first; fall back to http for internal
        # networks that can only reach GIN over plain http.
        model_path = pooch.retrieve(url=url, **retrieve_kwargs)
    except OSError as e:
        if "gin" in url:
            model_path = pooch.retrieve(
                url=url.replace("https://", "http://"), **retrieve_kwargs
            )
        else:
            raise e

    return Path(model_path)
