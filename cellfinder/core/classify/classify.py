import os
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import keras
import torch
from brainglobe_utils.cells.cells import Cell
from brainglobe_utils.general.system import get_num_processes
from torch.utils.data import DataLoader

from cellfinder.core import logger, types
from cellfinder.core.classify.cube_generator import (
    CuboidBatchSampler,
    CuboidStackDataset,
)
from cellfinder.core.classify.tools import get_model
from cellfinder.core.tools.image_processing import dataset_mean_std
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
    max_workers: int = 6,
    pin_memory: bool = True,
    *,
    callback: Optional[Callable[[int], None]] = None,
    normalize_channels: bool = False,
    normalization_down_sampling: int = 32,
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
    workers = min(workers, max_workers)

    start_time = datetime.now()

    voxel_sizes = list(map(float, voxel_sizes))

    signal_normalization = background_normalization = None
    if normalize_channels:
        logger.debug("Calculating channels norms")
        signal_normalization = dataset_mean_std(
            signal_array, normalization_down_sampling
        )
        background_normalization = dataset_mean_std(
            background_array, normalization_down_sampling
        )
        logger.debug(
            f"Signal channel norm is: {signal_normalization}. "
            f"Background channel norm is: {background_normalization}"
        )

    logger.debug("Initialising cube generator")
    dataset = CuboidStackDataset(
        signal_array=signal_array,
        background_array=background_array,
        signal_normalization=signal_normalization,
        background_normalization=background_normalization,
        points=points,
        data_voxel_sizes=voxel_sizes,
        network_voxel_sizes=network_voxel_sizes,
        network_cuboid_voxels=(cube_depth, cube_height, cube_width),
        axis_order=("z", "y", "x"),
        max_axis_0_cuboids_buffered=1,
    )
    # we use our own sampler so we can control the ordering
    sampler = CuboidBatchSampler(
        dataset=dataset,
        batch_size=batch_size,
        sort_by_axis="z",
        auto_shuffle=False,
    )
    data_loader = DataLoader(
        dataset=dataset,
        sampler=sampler,
        batch_size=None,
        num_workers=workers,
        pin_memory=pin_memory,
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
    if workers:
        dataset.start_dataset_thread(workers)
    try:
        predictions = model.predict(
            data_loader,
            verbose=True,
            callbacks=callbacks,
        )
    finally:
        dataset.stop_dataset_thread()

    predictions = torch.argmax(torch.from_numpy(predictions), dim=1)
    points_list = []

    # only go through the "extractable" points
    k = 0
    # the sampler doesn't auto shuffle, so the classification order (i.e. order
    # in `predictions`) is the same order as the sampler returns the batches.
    # Use that to get the corresponding row in points_arr, which gives us the
    # `index` of the row in the original point in the input points list
    for arr in sampler:
        for i in arr:
            p_idx = int(dataset.points_arr[i, 4].item())
            # don't use the original cell, use a copy
            cell = deepcopy(points[p_idx])
            cell.type = int((predictions[k] + 1).item())
            points_list.append(cell)
            k += 1

    time_elapsed = datetime.now() - start_time
    logger.info(
        f"Classification complete - {len(points_list)} points "
        f"done in : {time_elapsed}"
    )

    return points_list


class BatchEndCallback(keras.callbacks.Callback):
    def __init__(self, callback: Callable[[int], None]):
        self._callback = callback

    def on_predict_batch_end(
        self, batch: int, logs: Optional[Dict[str, Any]] = None
    ) -> None:
        self._callback(batch)
