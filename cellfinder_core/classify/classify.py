import logging
import numpy as np
from imlib.general.system import get_num_processes


from cellfinder_core.classify.tools import get_model
from cellfinder_core.classify.cube_generator import CubeGeneratorFromFile
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
):

    # Too many workers doesn't increase speed, and uses huge amounts of RAM
    workers = get_num_processes(
        min_free_cpu_cores=n_free_cpus, n_max_processes=max_workers
    )

    logging.debug("Initialising cube generator")
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

    logging.info("Running inference")
    predictions = model.predict(
        inference_generator,
        use_multiprocessing=True,
        workers=workers,
        verbose=True,
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
