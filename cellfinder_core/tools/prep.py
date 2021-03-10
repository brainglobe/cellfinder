"""
prep
==================
Functions to prepare files and directories needed for other functions
"""


import logging

from pathlib import Path

from imlib.general.system import get_num_processes

from imlib.general.config import get_config_obj

import cellfinder_core.tools.tf as tf_tools
from cellfinder_core.download import models as model_download
from cellfinder_core.download.download import amend_cfg
from cellfinder_core.tools.source_files import source_custom_config_cellfinder


def prep_classification(
    trained_model, model_weights, install_path, model, n_free_cpus
):
    n_processes = get_num_processes(min_free_cpu_cores=n_free_cpus)
    prep_tensorflow(n_processes)
    model_weights = prep_models(
        trained_model, model_weights, install_path, model
    )

    return model_weights


def prep_training(
    n_free_cpus, trained_model, model_weights, install_path, model
):
    n_processes = get_num_processes(min_free_cpu_cores=n_free_cpus)
    prep_tensorflow(n_processes)
    model_weights = prep_models(
        trained_model, model_weights, install_path, model
    )
    return model_weights


def prep_tensorflow(max_threads):
    tf_tools.set_tf_threads(max_threads)
    tf_tools.allow_gpu_memory_growth()


def prep_models(trained_model, model_weights, install_path, model):
    # if no model or weights, set default weights
    if trained_model is None and model_weights is None:
        logging.debug("No model or weights supplied, so using the default")

        config_file = source_custom_config_cellfinder()

        if not Path(config_file).exists():
            logging.debug("Custom config does not exist, downloading models")
            model_path = model_download.main(model, install_path)
            amend_cfg(new_model_path=model_path)

        model_weights = get_model_weights(config_file)
        if model_weights != "" and Path(model_weights).exists():
            model_weights = model_weights
        else:
            logging.debug("Model weights do not exist, downloading")
            model_path = model_download.main(model, install_path)
            amend_cfg(new_model_path=model_path)
            model_weights = get_model_weights(config_file)
    return model_weights


def get_model_weights(config_file):
    logging.debug(f"Reading config file: {config_file}")
    config_obj = get_config_obj(config_file)
    model_conf = config_obj["model"]
    model_weights = model_conf["model_path"]
    return model_weights
