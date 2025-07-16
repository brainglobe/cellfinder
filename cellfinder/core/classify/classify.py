import os
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import keras
import numpy as np
from brainglobe_utils.cells.cells import Cell
from brainglobe_utils.general.system import get_num_processes

from cellfinder.core import logger, types
from cellfinder.core.classify.cube_generator import CubeGeneratorFromFile
from cellfinder.core.classify.tools import get_model
from cellfinder.core.train.train_yaml import depth_type, models


def main(
    points: List[Cell],
    signal_array: types.array,
    background_array: types.array,
    n_free_cpus: int,
    voxel_sizes: Tuple[float, float, float],
    network_voxel_sizes: Tuple[float, float, float],
    batch_size: int,
    cube_height: int,
    cube_width: int,
    cube_depth: int,
    trained_model: Optional[os.PathLike],
    model_weights: Optional[os.PathLike],
    network_depth: depth_type,
    max_workers: int = 3,
    pin_memory: bool = False,
    *,
    callback: Optional[Callable[[int], None]] = None,
) -> List[Cell]:
    """
    Parameters
    ----------

    points: List of Cell objects
        The potential cells to classify.
    signal_array : numpy.ndarray or dask array
        3D array representing the signal data in z, y, x order.
    background_array : numpy.ndarray or dask array
        3D array representing the signal data in z, y, x order.
    n_free_cpus : int
        How many CPU cores to leave free.
    voxel_sizes : 3-tuple of floats
        Size of your voxels in the z, y, and x dimensions.
    network_voxel_sizes : 3-tuple of floats
        Size of the pre-trained network's voxels in the z, y, and x dimensions.
    batch_size : int
        How many potential cells to classify at one time. The GPU/CPU
        memory must be able to contain at once this many data cubes for
        the models. For performance-critical applications, tune to maximize
        memory usage without running out. Check your GPU/CPU memory to verify
        it's not full.
    cube_height: int
        The height of the data cube centered on the cell used for
        classification. Defaults to `50`.
    cube_width: int
        The width of the data cube centered on the cell used for
        classification. Defaults to `50`.
    cube_depth: int
        The depth of the data cube centered on the cell used for
        classification. Defaults to `20`.
    trained_model : Optional[Path]
        Trained model file path (home directory (default) -> pretrained
        weights).
    model_weights : Optional[Path]
        Model weights path (home directory (default) -> pretrained
        weights).
    network_depth: str
        The network depth to use during classification. Defaults to `"50"`.
    max_workers: int
        The number of sub-processes to use for data loading / processing.
        Defaults to 8.
    pin_memory: bool
        Pins data to be sent to the GPU to the CPU memory. This allows faster
        GPU data speeds, but can only be used if the data used by the GPU can
        stay in the CPU RAM while the GPU uses it. I.e. there's enough RAM.
        Otherwise, if there's a risk of the RAM being paged, it shouldn't be
        used. Defaults to False.
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

    if trained_model and Path(trained_model).suffix == ".h5":
        logger.warning(
            "Weights provided in place of the model, "
            "loading weights into default model."
        )
        model_weights = trained_model
        trained_model = None

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
    logger.info(
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
