import os
from typing import Optional, Tuple

import keras
from keras import Model

from cellfinder.core import logger
from cellfinder.core.classify.resnet import build_model, layer_type


def model_input_channels(model: Model) -> int:
    """The number of input channels the model expects."""
    return tuple(model.inputs[0].shape)[-1]


def get_model(
    existing_model: Optional[os.PathLike] = None,
    model_weights: Optional[os.PathLike] = None,
    network_depth: Optional[layer_type] = None,
    learning_rate: float = 0.0001,
    inference: bool = False,
    continue_training: bool = False,
    num_channels: int = 2,
    dimensions: int = 3,
    shape: Optional[Tuple[int, ...]] = None,
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
    (signal-only) model. Ignored when ``existing_model`` is loaded or when
    ``shape`` is given.
    :param dimensions: Whether to build a 3D or 2D network when creating a new
    model.
    :param shape: The input shape (excluding the batch dimension) to use when
    creating a new model. If None, a default for `dimensions` with
    `num_channels` channels is used.
    :return: A keras model

    """
    if existing_model is not None or network_depth is None:
        logger.debug(f"Loading model: {existing_model}")
        model = keras.models.load_model(existing_model)
        expected_rank = dimensions + 2
        if len(model.input_shape) != expected_rank:
            raise ValueError(
                f"Loaded model expects {len(model.input_shape) - 2}D input, "
                f"but dimensions={dimensions} was requested."
            )
    else:
        logger.debug(f"Creating a new instance of model: {network_depth}")
        if shape is None:
            shape = (
                (50, 50, num_channels)
                if dimensions == 2
                else (50, 50, 20, num_channels)
            )
        model = build_model(
            network_depth=network_depth,
            learning_rate=learning_rate,
            dimensions=dimensions,
            shape=shape,
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
                model.optimizer.build(model.trainable_variables)
                model.load_weights(model_weights)
            except (OSError, ValueError) as e:
                raise ValueError(
                    f"Error loading weights: {model_weights}.\n"
                    "Provided weights don't match the model architecture.\n"
                ) from e

    if inference:
        model.trainable = False

    return model
