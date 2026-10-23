import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Literal, Optional

import pooch

from cellfinder import DEFAULT_CELLFINDER_DIRECTORY

DEFAULT_DOWNLOAD_DIRECTORY = DEFAULT_CELLFINDER_DIRECTORY / "models"

MODEL_URL = "https://gin.swc.ucl.ac.uk/brainglobe/cellfinder/raw/main/models"


@dataclass(frozen=True)
class PretrainedModelInfo:
    """Metadata describing a pre-trained model."""

    filename: str
    hf_repo_id: Optional[str] = None
    hf_revision: Optional[str] = None
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
        hf_revision="aa326ad87a43f5db6754897d9a8f491f98d862d7",
        known_hash="4c0af5e916195603266fc18686a84e7156683cbd6e91b27385e9d6e0b5ef5a55",  # noqa: E501
        dimensions=3,
        channels=1,
    ),
    "resnet50_2d": PretrainedModelInfo(
        filename="resnet50_2d.keras",
        hf_repo_id="brainglobe/cellfinder_2d_default",
        hf_revision="1bb0a3ead0e88209ffa9887690a2bdb51d04d904",
        known_hash="6e40bf25452e99ce0334d85c413ac5029b113adef1f81f2fa46d0d546bb880f6",  # noqa: E501
        dimensions=2,
        channels=2,
    ),
    "resnet50_2d_1ch": PretrainedModelInfo(
        filename="resnet50_2d_single_channel.keras",
        hf_repo_id="brainglobe/cellfinder_2d_default",
        hf_revision="1bb0a3ead0e88209ffa9887690a2bdb51d04d904",
        known_hash="ae71226de27d93ec9238a09eda4c47e77b2435a840cbbf512eefdb957590e8b0",  # noqa: E501
        dimensions=2,
        channels=1,
    ),
}

model_filenames = {k: v.filename for k, v in PRETRAINED_MODELS.items()}
model_urls = {
    k: v.gin_url
    or (
        f"https://huggingface.co/{v.hf_repo_id}/resolve/"
        f"{v.hf_revision}/{v.filename}"
    )
    for k, v in PRETRAINED_MODELS.items()
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
    model_path = download_path / info.filename

    if local_files_only:
        if not model_path.exists():
            raise FileNotFoundError(
                f"{model_path} not found and local_files_only=True"
            )
        return model_path

    url = model_urls[model_name]
    headers = {}
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get(
        "HUGGING_FACE_HUB_TOKEN"
    )
    if hf_token and info.hf_repo_id:
        headers["Authorization"] = f"Bearer {hf_token}"

    downloader = pooch.HTTPDownloader(headers=headers, progressbar=True)
    retrieve_kwargs = dict(
        known_hash=info.known_hash,
        path=download_path,
        fname=info.filename,
        downloader=downloader,
    )
    try:
        # Try the secure endpoint first; fall back to http for internal
        # networks that can only reach GIN over plain http.
        retrieved_path = pooch.retrieve(url=url, **retrieve_kwargs)
    except OSError as e:
        if "gin" in url:
            retrieved_path = pooch.retrieve(
                url=url.replace("https://", "http://"), **retrieve_kwargs
            )
        else:
            raise e

    return Path(retrieved_path)
