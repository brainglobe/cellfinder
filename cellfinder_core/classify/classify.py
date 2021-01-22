import logging
import numpy as np
from imlib.general.system import get_sorted_file_paths, get_num_processes


from imlib.IO.cells import save_cells
from cellfinder_core.classify.tools import get_model
from cellfinder_core.classify.cube_generator import CubeGeneratorFromFile
from cellfinder_core.train.train_yml import models


def main(
    signal_paths,
    background_planes_path,
    n_free_cpus,
    detected_points,
    voxel_sizes,
    network_voxel_sizes,
    batch_size,
    cube_height,
    cube_width,
    cube_depth,
    trained_model,
    model_weights,
    network_depth,
    classified_points_path,
    save_csv=False,
    max_workers=3,
):
    signal_images = get_sorted_file_paths(signal_paths, file_extension="tif")
    background_images = get_sorted_file_paths(
        background_planes_path, file_extension="tif"
    )

    # Too many workers doesn't increase speed, and uses huge amounts of RAM
    workers = get_num_processes(
        min_free_cpu_cores=n_free_cpus, n_max_processes=max_workers
    )

    logging.debug("Initialising cube generator")
    inference_generator = CubeGeneratorFromFile(
        detected_points,
        signal_images,
        background_images,
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
    cells_list = []

    # only go through the "extractable" cells
    for idx, cell in enumerate(inference_generator.ordered_cells):
        cell.type = predictions[idx] + 1
        cells_list.append(cell)

    logging.info("Saving classified cells")
    save_cells(cells_list, classified_points_path, save_csv=save_csv)
