import os
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import keras
import numpy as np
from brainglobe_utils.cells.cells import Cell
from brainglobe_utils.general.system import get_num_processes

from cellfinder.core import logger, types
from cellfinder.core.classify.cube_generator import CubeGeneratorFromFile
from cellfinder.core.classify.tools import get_model
from cellfinder.core.train.train_yml import depth_type, models


def main(
    points: List[Cell],
    signal_array: types.array,
    background_array: types.array,
    n_free_cpus: int,
    voxel_sizes: Tuple[int, int, int],
    network_voxel_sizes: Tuple[int, int, int],
    batch_size: int,
    cube_height: int,
    cube_width: int,
    cube_depth: int,
    trained_model: Optional[os.PathLike],
    model_weights: Optional[os.PathLike],
    network_depth: depth_type,
    max_workers: int = 3,
    *,
    callback: Optional[Callable[[int], None]] = None,
) -> List:
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
    workers = get_num_processes(min_free_cpu_cores=n_free_cpus)

    start_time = datetime.now()

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
        use_multiprocessing=False,
        workers=workers,
    )

    model = get_model(
        existing_model=trained_model,
        model_weights=model_weights,
        network_depth=models[network_depth],
        inference=True,
    )

    logger.info("Running inference")
    # in Keras 3.0 multiprocessing params are specified in the generator
    predictions = model.predict(
        inference_generator,
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

    time_elapsed = datetime.now() - start_time
    print(
        "Classfication complete - all points done in : {}".format(time_elapsed)
    )

    return points_list


class BatchEndCallback(keras.callbacks.Callback):
    def __init__(self, callback: Callable[[int], None]):
        self._callback = callback

    def on_predict_batch_end(
        self, batch: int, logs: Optional[Dict[str, Any]] = None
    ) -> None:
        self._callback(batch)
