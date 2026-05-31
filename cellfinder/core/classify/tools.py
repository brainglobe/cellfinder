import os
from typing import Optional

import keras
from keras import Model

from cellfinder.core import logger
from cellfinder.core.classify.resnet import build_model, layer_type


def get_model(
    existing_model: Optional[os.PathLike] = None,
    model_weights: Optional[os.PathLike] = None,
    network_depth: Optional[layer_type] = None,
    learning_rate: float = 0.0001,
    inference: bool = False,
    continue_training: bool = False,
    num_channels: int = 2,
) -> Model:
    """Returns the correct model based on the arguments passed
    :param existing_model: An existing, trained model. This is returned if it
    exists
    :param model_weights: This file is used to set the model weights if it
    exists
    :param network_depth: This defines the type of model to be created if
    necessary
    :param learning_rate: For creating a new model
    :param inference: If True, will ensure that a trained model exists. E.g.
    by using the default one
    :param continue_training: If True, will ensure that a trained model
    exists. E.g. by using the default one
    :param num_channels: Number of input channels for a freshly built model.
    ``2`` for the standard signal+background model, ``1`` for a single-channel
    (signal-only) model. Ignored when ``existing_model`` is loaded.
    :return: A keras model

    """
    if existing_model is not None or network_depth is None:
        logger.debug(f"Loading model: {existing_model}")
        model = keras.models.load_model(existing_model)
    else:
        logger.debug(f"Creating a new instance of model: {network_depth}")
        model = build_model(
            shape=(50, 50, 20, num_channels),
            network_depth=network_depth,
            learning_rate=learning_rate,
        )
        if inference or continue_training:
            logger.debug(
                f"Setting model weights according to: {model_weights}",
            )
            if model_weights is None:
                raise OSError(
                    "`model_weights` must be provided for inference "
                    "or continued training."
                )
            try:
                model.load_weights(model_weights)
            except (OSError, ValueError) as e:
                raise ValueError(
                    f"Error loading weights: {model_weights}.\n"
                    "Provided weights don't match the model architecture.\n"
                ) from e

    if inference:
        model.trainable = False

    return model
