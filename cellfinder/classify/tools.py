import logging

from skimage.io import imread
import pandas as pd
import numpy as np
from tqdm import tqdm
import tensorflow as tf

from cellfinder.classify.resnet import build_model


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


def populate_array_with_cubes(images, idx, signal_im, background_im):
    # if paths are pathlib objs, skimage only reads one plane
    images[idx, :, :, :, 0] = np.moveaxis(imread(signal_im), 0, 2)
    images[idx, :, :, :, 1] = np.moveaxis(imread(background_im), 0, 2)
    return images


def load_cubes_from_dir(
    directory, shape, signal_extension, background_extension
):

    cubes_list = list(directory.glob("*" + signal_extension))
    number_images = len(cubes_list)
    images = np.empty(((number_images,) + shape))

    for idx, signal_im in enumerate(cubes_list):
        signal_im = str(signal_im)
        background_im = signal_im.replace(
            signal_extension, background_extension
        )
        images = populate_array_with_cubes(
            images, idx, signal_im, background_im
        )

    return images.astype(np.float16)


def load_cubes_from_csv(
    signal_csv, background_csv, labels_csv, shape, n_test_samples, skip_amt
):
    signal = pd.read_csv(signal_csv, header=None)
    signal_list = list(signal[1])
    if n_test_samples is not None:
        signal_list = signal_list[0:n_test_samples:skip_amt]

    background = pd.read_csv(background_csv, header=None)
    background_list = list(background[1])
    if n_test_samples is not None:
        background_list = background_list[0:n_test_samples:skip_amt]

    number_images = len(signal_list)
    images = np.empty(((number_images,) + shape))

    print(f"Loading {number_images} images")
    for idx, signal_im in enumerate(tqdm(signal_list)):
        background_im = background_list[idx]
        images = populate_array_with_cubes(
            images, idx, signal_im, background_im
        )

    # required editing yaml handling
    labels = pd.read_csv(labels_csv, header=None)[1]
    labels = np.array(labels).astype(np.int8)
    if n_test_samples is not None:
        labels = labels[0:n_test_samples:skip_amt]
    labels = tf.keras.utils.to_categorical(labels).astype(np.uint8)
    return images, labels


def load_data(
    shape, cells_dir, no_cells_dir, signal_extension, background_extension
):
    cell_images = load_cubes_from_dir(
        cells_dir, shape, signal_extension, background_extension
    )
    non_cell_images = load_cubes_from_dir(
        no_cells_dir, shape, signal_extension, background_extension
    )
    images = np.append(cell_images, non_cell_images, axis=0)

    cell_labels = np.ones((len(cell_images), 1))
    non_cell_labels = np.zeros((len(non_cell_images), 1))
    labels = np.append(cell_labels, non_cell_labels)
    labels = tf.keras.utils.to_categorical(labels).astype(np.uint8)

    return images, labels


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


def batch_viewer(
    image_batch,
    opacity=0.5,
    signal_name="Signal channnel",
    background_name="Background channel",
    signal_colormap="magenta",
    background_colormap="cyan",
):
    """
    Tool to visualise cube generator batches
    :param image_batch:
    :param opacity:
    :param signal_name:
    :param background_name:
    :param signal_colormap:
    :param background_colormap:
    :return:
    """
    import napari

    image = np.moveaxis(image_batch, 3, 1)
    scale = calculate_scale(image[0, :, :, :, 0])

    with napari.gui_qt():
        v = napari.Viewer()
        v.add_image(
            image[..., 0],
            name=signal_name,
            opacity=opacity,
            colormap=signal_colormap,
            scale=scale,
        )
        v.add_image(
            image[..., 1],
            name=background_name,
            opacity=opacity,
            colormap=background_colormap,
            scale=scale,
        )


def calculate_scale(image):
    scale = []
    for dim in image.shape:
        scale.append(100 / dim)
    return tuple(scale)
