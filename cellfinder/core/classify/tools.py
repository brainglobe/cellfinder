import os
from pathlib import Path
from typing import Optional, Tuple

import keras
from keras import Model

from cellfinder.core import logger
from cellfinder.core.classify.resnet import build_model, layer_type


def model_input_channels(model: Model) -> int:
    """The number of input channels the model expects."""
    return tuple(model.inputs[0].shape)[-1]


def _load_existing_model(
    existing_model: os.PathLike, dimensions: int
) -> Model:
    """
    Load a saved model, checking its input rank matches `dimensions`.
    """
    logger.debug(f"Loading model: {existing_model}")
    model = keras.models.load_model(existing_model)
    if len(model.input_shape) != dimensions + 2:
        raise ValueError(
            f"Loaded model expects {len(model.input_shape) - 2}D input, "
            f"but dimensions={dimensions} was requested."
        )
    return model


def _default_shape(dimensions: int, num_channels: int) -> Tuple[int, ...]:
    """
    The input shape used for a new model when none is given.
    """
    spatial = (50, 50) if dimensions == 2 else (50, 50, 20)
    return (*spatial, num_channels)


def _set_model_weights(
    model: Model,
    model_weights: Optional[os.PathLike],
    continue_training: bool = False,
) -> None:
    """
    Set a new model's weights from `model_weights`, which must be provided.
    """
    logger.debug(f"Setting model weights according to: {model_weights}")
    if model_weights is None:
        raise OSError(
            "`model_weights` must be provided for inference "
            "or continued training."
        )
    try:
        if (
            continue_training
            and hasattr(model, "optimizer")
            and model.optimizer is not None
        ):
            model.optimizer.build(model.trainable_variables)
        model.load_weights(model_weights)
    except (OSError, ValueError) as e:
        raise ValueError(
            f"Error loading weights: {model_weights}.\n"
            "Provided weights don't match the model architecture.\n"
        ) from e


def export_weights_only(
    source_model: os.PathLike,
    target_weights: os.PathLike,
) -> Path:
    """
    Export weights-only from a trained model file to a `.weights.h5` file.

    Parameters
    ----------
    source_model : os.PathLike
        Path to the saved source model (e.g. `.keras` file).
    target_weights : os.PathLike
        Path where the weights-only `.weights.h5` file will be saved.

    Returns
    -------
    Path
        Path to the saved weights file.
    """
    source_path = Path(source_model)
    target_path = Path(target_weights)

    if not source_path.exists():
        raise FileNotFoundError(f"Source model not found: {source_path}")

    logger.debug(f"Loading full model from {source_path} to extract weights")
    model = keras.models.load_model(source_path)

    target_path.parent.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Saving weights-only to {target_path}")
    model.save_weights(target_path)

    return target_path


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

    One of `existing_model` or `network_depth` must be given; `existing_model`
    takes precedence when both are.

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
    :param dimensions: Whether a 3D or 2D network is expected. Checked against
    the rank of a loaded ``existing_model``, and used to pick the default shape
    when ``shape`` is None. A new model's rank otherwise follows ``shape``.
    :param shape: The input shape (excluding the batch dimension) to use when
    creating a new model. If None, a default for `dimensions` with
    `num_channels` channels is used.
    :return: A keras model

    """
    if existing_model is not None:
        model = _load_existing_model(existing_model, dimensions)
    elif network_depth is None:
        raise ValueError(
            "Either `existing_model` or `network_depth` must be provided."
        )
    else:
        logger.debug(f"Creating a new instance of model: {network_depth}")
        model = build_model(
            network_depth=network_depth,
            learning_rate=learning_rate,
            shape=(
                _default_shape(dimensions, num_channels)
                if shape is None
                else shape
            ),
        )
        if inference or continue_training:
            _set_model_weights(
                model, model_weights, continue_training=continue_training
            )

    if inference:
        model.trainable = False

    return model
