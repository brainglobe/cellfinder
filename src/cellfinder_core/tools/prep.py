"""
prep
==================
Functions to prepare files and directories needed for other functions
"""
import os
from pathlib import Path
from typing import Optional

from imlib.general.config import get_config_obj
from imlib.general.system import get_num_processes

import cellfinder_core.tools.tf as tf_tools
from cellfinder_core import logger
from cellfinder_core.download import models as model_download
from cellfinder_core.download.download import amend_cfg
from cellfinder_core.tools.source_files import source_custom_config_cellfinder

home = Path.home()
DEFAULT_INSTALL_PATH = home / ".cellfinder"


def prep_classification(
    trained_model: Optional[os.PathLike],
    model_weights: Optional[os.PathLike],
    install_path: Optional[os.PathLike],
    model_name: str,
    n_free_cpus: int,
) -> Path:
    n_processes = get_num_processes(min_free_cpu_cores=n_free_cpus)
    prep_tensorflow(n_processes)
    model_weights = prep_models(
        trained_model, model_weights, install_path, model_name
    )

    return model_weights


def prep_training(
    n_free_cpus: int,
    trained_model: Optional[os.PathLike],
    model_weights: Optional[os.PathLike],
    install_path: Optional[os.PathLike],
    model_name: str,
) -> Path:
    n_processes = get_num_processes(min_free_cpu_cores=n_free_cpus)
    prep_tensorflow(n_processes)
    model_weights = prep_models(
        trained_model, model_weights, install_path, model_name
    )
    return model_weights


def prep_tensorflow(max_threads: int) -> None:
    tf_tools.set_tf_threads(max_threads)
    tf_tools.allow_gpu_memory_growth()


def prep_models(
    trained_model_path: Optional[os.PathLike],
    model_weights_path: Optional[os.PathLike],
    install_path: Optional[os.PathLike],
    model_name: str,
) -> Path:
    install_path = install_path or DEFAULT_INSTALL_PATH
    # if no model or weights, set default weights
    if model_weights_path is None:
        logger.debug("No model supplied, so using the default")

        config_file = source_custom_config_cellfinder()

        if not Path(config_file).exists():
            logger.debug("Custom config does not exist, downloading models")
            model_path = model_download.main(model_name, install_path)
            amend_cfg(new_model_path=model_path)

        model_weights = get_model_weights(config_file)
        if not model_weights.exists():
            logger.debug("Model weights do not exist, downloading")
            model_path = model_download.main(model_name, install_path)
            amend_cfg(new_model_path=model_path)
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
