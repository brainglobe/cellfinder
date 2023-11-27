import os
from pathlib import Path
from typing import Literal

from cellfinder.core import logger
from cellfinder.core.download.download import download

model_weight_urls = {
    "resnet50_tv": "https://gin.g-node.org/cellfinder/models/raw/"
    "master/resnet50_tv.h5",
    "resnet50_all": "https://gin.g-node.org/cellfinder/models/raw/"
    "master/resnet50_weights.h5",
}

download_requirements_gb = {
    "resnet50_tv": 0.18,
    "resnet50_all": 0.18,
}

model_type = Literal["resnet50_tv", "resnet50_all"]


def main(model_name: model_type, download_path: os.PathLike) -> Path:
    """
    For a given model name and download path, download the model file
    and return the path to the downloaded file.
    """
    download_path = Path(download_path)

    model_weight_dir = download_path / "model_weights"
    model_path = model_weight_dir / f"{model_name}.h5"
    if not model_path.exists():
        model_weight_dir.mkdir(parents=True)

        logger.info(
            f"Downloading '{model_name}' model. This may take a little while."
        )

        download(
            model_path,
            model_weight_urls[model_name],
            model_name,
            download_requires=download_requirements_gb[model_name],
        )

    else:
        logger.info(f"Model already exists at {model_path}. Skipping download")

    return model_path
