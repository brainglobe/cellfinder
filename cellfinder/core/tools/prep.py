"""
prep
==================
Functions to prepare files and directories needed for other functions
"""

import os
from pathlib import Path
from typing import Optional

from brainglobe_utils.general.config import get_config_obj

from cellfinder.core import logger
from cellfinder.core.download.download import (
    DEFAULT_DOWNLOAD_DIRECTORY,
    amend_user_configuration,
    download_models,
    model_type,
)
from cellfinder.core.tools.source_files import user_specific_configuration_path


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
    install_path = install_path or DEFAULT_DOWNLOAD_DIRECTORY
    # if no model or weights, set default weights
    if model_weights_path is None:
        logger.debug("No model supplied, so using the default")

        config_file = user_specific_configuration_path()

        if not Path(config_file).exists():
            logger.debug("Custom config does not exist, downloading models")
            model_path = download_models(model_name, install_path)
            amend_user_configuration(new_model_path=model_path)

        model_weights = get_model_weights(config_file)
        if not model_weights.exists():
            logger.debug("Model weights do not exist, downloading")
            model_path = download_models(model_name, install_path)
            amend_user_configuration(new_model_path=model_path)
            model_weights = get_model_weights(config_file)
    else:
        model_weights = Path(model_weights_path)
    return model_weights


def get_model_weights(config_file: os.PathLike) -> Path:
    logger.debug(f"Reading config file: {config_file}")
    config_obj = get_config_obj(config_file)
    model_conf = config_obj["model"]
    model_weights = model_conf["model_path"]
    return Path(model_weights)
