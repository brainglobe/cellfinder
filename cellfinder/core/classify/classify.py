import os
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import keras
import numpy as np
from brainglobe_utils.cells.cells import Cell
from brainglobe_utils.general.system import get_num_processes
from torch.utils.data import DataLoader

from cellfinder.core import logger, types
from cellfinder.core.classify.cube_generator import (
    CuboidBatchSampler,
    CuboidStackDataset,
)
from cellfinder.core.classify.tools import get_model
from cellfinder.core.train.train_yaml import depth_type, models


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
    max_workers: int = 6,
    pin_memory: bool = True,
    *,
    callback: Optional[Callable[[int], None]] = None,
) -> List[Cell]:
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
    workers = min(workers, max_workers)

    start_time = datetime.now()

    voxel_sizes = list(map(float, voxel_sizes))
    logger.debug("Initialising cube generator")
    dataset = CuboidStackDataset(
        signal_array=signal_array,
        background_array=background_array,
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
        num_workers=workers,
        drop_last=False,
        pin_memory=pin_memory,
        collate_fn=CuboidBatchSampler.loader_collate_identity,
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

    predictions = predictions.round()
    predictions = predictions.astype("uint16")

    predictions = np.argmax(predictions, axis=1)
    points_list = []

    # only go through the "extractable" points
    for idx, arr in enumerate(dataset.points_arr):
        cell = Cell(
            (arr["x"], arr["y"], arr["z"]), cell_type=predictions[idx] + 1
        )
        points_list.append(cell)

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
