from typing import Callable, Optional

import numpy as np
from imlib.general.system import get_num_processes
from tensorflow import keras

from cellfinder_core import logger
from cellfinder_core.classify.cube_generator import CubeGeneratorFromFile
from cellfinder_core.classify.tools import get_model
from cellfinder_core.train.train_yml import models


def main(
    points,
    signal_array,
    background_array,
    n_free_cpus,
    voxel_sizes,
    network_voxel_sizes,
    batch_size,
    cube_height,
    cube_width,
    cube_depth,
    trained_model,
    model_weights,
    network_depth,
    max_workers=3,
    *,
    callback: Optional[Callable[[int], None]] = None,
):
    """
    Parameters
    ----------
    callback : Callable[int], optional
        A callback function that is called during classification. Called with
        the batch number once that batch has been classified.
    """
    if signal_array.ndim != 3:
        raise IOError("Signal data must be 3D")
    if background_array.ndim != 3:
        raise IOError("Background data must be 3D")

    if callback is not None:
        callbacks = [BatchEndCallback(callback)]
    else:
        callbacks = None

    # Too many workers doesn't increase speed, and uses huge amounts of RAM
    workers = get_num_processes(
        min_free_cpu_cores=n_free_cpus, n_max_processes=max_workers
    )

    logger.debug("Initialising cube generator")
    inference_generator = CubeGeneratorFromFile(
        points,
        signal_array,
        background_array,
        voxel_sizes,
        network_voxel_sizes,
        batch_size=batch_size,
        cube_width=cube_width,
        cube_height=cube_height,
        cube_depth=cube_depth,
    )

    model = get_model(
        existing_model=trained_model,
        model_weights=model_weights,
        network_depth=models[network_depth],
        inference=True,
    )

    logger.info("Running inference")
    predictions = model.predict(
        inference_generator,
        use_multiprocessing=True,
        workers=workers,
        verbose=True,
        callbacks=callbacks,
    )
    predictions = predictions.round()
    predictions = predictions.astype("uint16")

    predictions = np.argmax(predictions, axis=1)
    points_list = []

    # only go through the "extractable" points
    for idx, cell in enumerate(inference_generator.ordered_points):
        cell.type = predictions[idx] + 1
        points_list.append(cell)

    return points_list


class BatchEndCallback(keras.callbacks.Callback):
    def __init__(self, callback):
        self._callback = callback

    def on_predict_batch_end(self, batch, logs=None):
        self._callback(batch)
