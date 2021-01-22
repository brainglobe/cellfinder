import logging

import numpy as np
import tensorflow as tf

from cellfinder_core.classify.resnet import build_model


def get_model(
    existing_model=None,
    model_weights=None,
    network_depth=None,
    learning_rate=0.0001,
    inference=False,
    continue_training=False,
):
    """
    Returns the correct model based on the arguments passed
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
    :return: A tf.keras model

    """
    if existing_model is not None:
        logging.debug(f"Loading model: {existing_model}")
        return tf.keras.models.load_model(existing_model)
    else:
        logging.debug(f"Creating a new instance of model: {network_depth}")
        model = build_model(
            network_depth=network_depth, learning_rate=learning_rate
        )
        if inference or continue_training:
            logging.debug(
                f"Setting model weights according to: {model_weights}"
            )
            model.load_weights(model_weights)
        return model


def make_lists(tiff_files, train=True):

    signal_list = []
    background_list = []
    if train:
        labels = []

    for group in tiff_files:
        for image in group:
            signal_list.append(image.img_files[0])
            background_list.append(image.img_files[1])
            if train:
                if image.label == "no_cell":
                    labels.append(0)
                elif image.label == "cell":
                    labels.append(1)

    if train:
        labels = np.array(labels)
        return signal_list, background_list, labels
    else:
        return signal_list, background_list
