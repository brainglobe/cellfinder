"""
prep
==================
Functions to prepare files and directories needed for other functions
"""

import os
from pathlib import Path
from typing import Optional

from cellfinder.core import logger
from cellfinder.core.download.download import (
    DEFAULT_DOWNLOAD_DIRECTORY,
    download_models,
    model_type,
)


def prep_model_weights(
    model_weights: Optional[os.PathLike],
    install_path: Optional[os.PathLike],
    model_name: model_type,
) -> Path:
    # prepare models (get default weights or provided ones)
    model_weights = prep_models(model_weights, install_path, model_name)

    return model_weights


def prep_models(
    model_weights_path: Optional[os.PathLike],
    install_path: Optional[os.PathLike],
    model_name: model_type,
) -> Path:
    if model_weights_path is None:
        logger.debug(f"No model supplied, downloading {model_name}")
        install_path = install_path or DEFAULT_DOWNLOAD_DIRECTORY
        return download_models(model_name, install_path)

    model_weights = Path(model_weights_path)
    if not model_weights.exists():
        raise FileNotFoundError(f"Model weights not found: {model_weights}")
    return model_weights
