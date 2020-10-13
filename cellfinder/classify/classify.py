import logging
import numpy as np
from imlib.general.system import get_sorted_file_paths, get_num_processes


from imlib.IO.cells import save_cells, get_cells
from imlib.cells.cells import MissingCellsError
from cellfinder.classify.tools import get_model
from cellfinder.classify.cube_generator import CubeGeneratorFromFile
from cellfinder.train.train_yml import models


def main(args, max_workers=3):
    signal_paths = args.signal_planes_paths[args.signal_channel]
    background_paths = args.background_planes_path[0]
    signal_images = get_sorted_file_paths(signal_paths, file_extension="tif")
    background_images = get_sorted_file_paths(
        background_paths, file_extension="tif"
    )

    # Too many workers doesn't increase speed, and uses huge amounts of RAM
    workers = get_num_processes(
        min_free_cpu_cores=args.n_free_cpus, n_max_processes=max_workers
    )

    logging.debug("Initialising cube generator")
    inference_generator = CubeGeneratorFromFile(
        args.paths.detected_points,
        signal_images,
        background_images,
        args.voxel_sizes,
        args.network_voxel_sizes,
        batch_size=args.batch_size,
        cube_width=args.cube_width,
        cube_height=args.cube_height,
        cube_depth=args.cube_depth,
    )

    model = get_model(
        existing_model=args.trained_model,
        model_weights=args.model_weights,
        network_depth=models[args.network_depth],
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
    save_cells(
        cells_list, args.paths.classified_points, save_csv=args.save_csv
    )
    try:
        get_cells(args.paths.classified_points, cells_only=True)
        return True
    except MissingCellsError:
        return False
