"""
main
===============

Trains a network based on a yaml file specifying cubes of cells/non cells.
"""

import os
from argparse import (
    ArgumentDefaultsHelpFormatter,
    ArgumentParser,
    ArgumentTypeError,
)
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Dict, Literal, Sequence

from brainglobe_utils.cells.cells import Cell
from brainglobe_utils.general.numerical import (
    check_positive_float,
    check_positive_int,
)
from brainglobe_utils.general.system import (
    ensure_directory_exists,
    get_num_processes,
)
from brainglobe_utils.IO.cells import find_relevant_tiffs
from brainglobe_utils.IO.yaml import read_yaml_section
from fancylog import fancylog
from keras.callbacks import (
    CSVLogger,
    LearningRateScheduler,
    ModelCheckpoint,
    TensorBoard,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

import cellfinder.core as program_for_log
from cellfinder.core import logger
from cellfinder.core.classify.cube_generator import (
    CuboidBatchSampler,
    CuboidTiffDataset,
)
from cellfinder.core.classify.resnet import layer_type
from cellfinder.core.classify.tools import get_model
from cellfinder.core.download.download import DEFAULT_DOWNLOAD_DIRECTORY
from cellfinder.core.tools.prep import prep_model_weights
from cellfinder.core.tools.tiff import TiffDir, TiffFile, TiffList

depth_type = Literal["18", "34", "50", "101", "152"]

models: Dict[depth_type, layer_type] = {
    "18": "18-layer",
    "34": "34-layer",
    "50": "50-layer",
    "101": "101-layer",
    "152": "152-layer",
}


CUBE_WIDTH = 50
CUBE_HEIGHT = 50
CUBE_DEPTH = 20


def lr_scheduler(
    epoch: int,
    lr: float,
    multiplier: float,
    epoch_list: Sequence[int],
) -> float:
    if epoch in epoch_list:
        return lr * multiplier
    return lr


def valid_model_depth(depth):
    """
    Ensures a correct existing_model is chosen
    :param value: Input value
    :param models: Dict of allowed models
    :return: Input value, if it corresponds to a valid existing_model
    """

    if depth in models.keys():
        return depth
    else:
        raise ArgumentTypeError(
            f"Model depth: {depth} is not valid. Please "
            f"choose one of: {list(models.keys())}"
        )


def misc_parse(parser):
    misc_parser = parser.add_argument_group("Misc options")
    misc_parser.add_argument(
        "--n-free-cpus",
        dest="n_free_cpus",
        type=check_positive_int,
        default=2,
        help="The number of CPU cores on the machine to leave "
        "unused by the program to spare resources.",
    )
    misc_parser.add_argument(
        "--max-ram",
        dest="max_ram",
        type=check_positive_float,
        default=None,
        help="Maximum amount of RAM to use (in GB) - not currently fully "
        "implemented for all parts of cellfinder",
    )
    misc_parser.add_argument(
        "--save-csv",
        dest="save_csv",
        action="store_true",
        help="Save .csv files of cell locations (in addition to xml)."
        "Useful for importing into other software.",
    )
    misc_parser.add_argument(
        "--debug",
        dest="debug",
        action="store_true",
        help="Debug mode. Will increase verbosity of logging and save all "
        "intermediate files for diagnosis of software issues.",
    )
    misc_parser.add_argument(
        "--sort-input-file",
        dest="sort_input_file",
        action="store_true",
        help="If set to true, the input text file will be sorted using "
        "natural sorting. This means that the file paths will be "
        "sorted as would be expected by a human and "
        "not purely alphabetically",
    )
    return parser


def training_parse():
    from cellfinder.core.download.cli import (
        download_parser,
    )

    training_parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter
    )
    training_parser.add_argument(
        "-y",
        "--yaml",
        dest="yaml_file",
        nargs="+",
        required=True,
        type=str,
        help="The path to the yaml run file.",
    )
    training_parser.add_argument(
        "-o",
        "--output-dir",
        dest="output_dir",
        required=True,
        type=str,
        help="Output directory for the final model.",
    )
    training_parser.add_argument(
        "--continue-training",
        dest="continue_training",
        action="store_true",
        help="Continue training from an existing trained model. If no model "
        "or model weights are specified, this will continue from the "
        "included model.",
    )
    training_parser.add_argument(
        "--trained-model",
        dest="trained_model",
        type=str,
        help="Path to the trained model",
    )
    training_parser.add_argument(
        "--model-weights",
        dest="model_weights",
        type=str,
        help="Path to existing model weights",
    )
    training_parser.add_argument(
        "--network-depth",
        dest="network_depth",
        type=valid_model_depth,
        default="50",
        help="Resnet depth (based on He et al. (2015)",
    )
    training_parser.add_argument(
        "--batch-size",
        dest="batch_size",
        type=check_positive_int,
        default=16,
        help="Training batch size",
    )
    training_parser.add_argument(
        "--epochs",
        dest="epochs",
        type=check_positive_int,
        default=100,
        help="Number of training epochs",
    )
    training_parser.add_argument(
        "--max-workers",
        dest="max_workers",
        type=check_positive_int,
        default=3,
        help="Maximum number of worker processes to use to load data",
    )
    training_parser.add_argument(
        "--test-fraction",
        dest="test_fraction",
        type=float,
        default=0.1,
        help="Fraction of training data to use for validation",
    )
    training_parser.add_argument(
        "--learning-rate",
        dest="learning_rate",
        type=check_positive_float,
        default=0.0001,
        help="Learning rate for training the model",
    )
    training_parser.add_argument(
        "--no-augment",
        dest="no_augment",
        action="store_true",
        help="Don't apply data augmentation",
    )
    training_parser.add_argument(
        "--augment-likelihood",
        dest="augment_likelihood",
        type=check_positive_float,
        default=0.9,
        help="Value `[0, 1]` with the probability of a data item being "
        "augmented. I.e. `0.9` means 90% of the data will have been "
        "augmented.",
    )
    training_parser.add_argument(
        "--save-weights",
        dest="save_weights",
        action="store_true",
        help="Only store the model weights, and not the full model. Useful to "
        "save storage space.",
    )
    training_parser.add_argument(
        "--no-save-checkpoints",
        dest="no_save_checkpoints",
        action="store_true",
        help="Store the model at intermediate points during training",
    )
    training_parser.add_argument(
        "--tensorboard",
        action="store_true",
        help="Log to output_directory/tensorboard",
    )
    training_parser.add_argument(
        "--save-progress",
        dest="save_progress",
        action="store_true",
        help="Save training progress to a .csv file",
    )
    training_parser.add_argument(
        "--normalize-channels",
        dest="normalize_channels",
        action="store_true",
        help="Normalize the training data to the mean/std of the datasets "
        "from which the cubes came from",
    )
    training_parser.add_argument(
        "--lr-schedule",
        dest="lr_schedule",
        nargs="*",
        type=partial(check_positive_int, none_allowed=False),
        default=(),
        help="If not empty, the list of epochs when to multiply the current "
        "learning rate by the lr_multiplier. E.g. if it's [10, 25], we "
        "start with a learning rate of 0.001, and lr_multiplier is "
        "0.1, then the LR will be 0.001 for epochs 0-9, 0.0001 for 10-24,"
        " and 00001 for epoch 25 and beyond.",
    )
    training_parser.add_argument(
        "--lr-multiplier",
        dest="lr_multiplier",
        type=partial(check_positive_float, none_allowed=False),
        default=0.1,
        help="The multiplier by which to multiply the previous learning rate "
        "at the epochs listed in lr_schedule.",
    )

    training_parser = misc_parse(training_parser)
    training_parser = download_parser(training_parser)
    args = training_parser.parse_args()

    return args


def parse_yaml(yaml_files, section="data"):
    data = []
    for yaml_file in yaml_files:
        data.extend(read_yaml_section(yaml_file, section))
    return data


def get_tiff_files(yaml_contents: list[dict]) -> list[list[TiffFile]]:
    """
    Takes a yaml file representing multiple folders each containing many
    extracted cube tiff files. It returns a corresponding list of lists of
    `TiffFile`, where in the sub-list each `TiffFile` represents a tiff in
    the given directory.
    """
    tiff_lists = []
    for d in yaml_contents:
        if d["bg_channel"] < 0:
            channels = [d["signal_channel"]]
            channels_metadata = [
                {},
            ]
        else:
            channels = [d["signal_channel"], d["bg_channel"]]
            channels_metadata = [{}, {}]

        if "signal_mean" in d:
            channels_metadata[0] = {
                "mean": float(d["signal_mean"]),
                "std": float(d["signal_std"]),
            }
        # if we have norm for signal we must have for background
        if "signal_mean" in d and d["bg_channel"] >= 0:
            channels_metadata[1] = {
                "mean": float(d["bg_mean"]),
                "std": float(d["bg_std"]),
            }

        if "cell_def" in d and d["cell_def"]:
            ch1_tiffs = [
                os.path.join(d["cube_dir"], f)
                for f in os.listdir(d["cube_dir"])
                if f.endswith("Ch" + str(channels[0]) + ".tif")
            ]
            tiff_lists.append(
                TiffList(
                    find_relevant_tiffs(ch1_tiffs, d["cell_def"]),
                    channels,
                    channels_metadata,
                    d["type"],
                )
            )
        else:
            tiff_lists.append(
                TiffDir(d["cube_dir"], channels, channels_metadata, d["type"])
            )

    tiff_files = [tiff_dir.make_tifffile_list() for tiff_dir in tiff_lists]
    return tiff_files


def make_tiff_lists(
    tiff_files: list[list[TiffFile]],
) -> tuple[list[tuple[list[str], list[dict]]], list[Cell]]:

    cells = []
    filenames = []

    for group in tiff_files:
        for image in group:
            filenames.append((image.img_files, image.channels_metadata))
            cells.append(image.as_cell())

    return filenames, cells


def cli():
    args = training_parse()
    ensure_directory_exists(args.output_dir)

    fancylog.start_logging(
        args.output_dir,
        program_for_log,
        variables=[args],
        log_header="CELLFINDER TRAINING LOG",
    )

    output_dir = Path(args.output_dir)
    run(
        output_dir,
        args.yaml_file,
        n_free_cpus=args.n_free_cpus,
        trained_model=args.trained_model,
        model_weights=args.model_weights,
        install_path=args.install_path,
        model=args.model,
        network_depth=args.network_depth,
        learning_rate=args.learning_rate,
        continue_training=args.continue_training,
        test_fraction=args.test_fraction,
        batch_size=args.batch_size,
        max_workers=args.max_workers,
        no_augment=args.no_augment,
        augment_likelihood=args.augment_likelihood,
        tensorboard=args.tensorboard,
        save_weights=args.save_weights,
        no_save_checkpoints=args.no_save_checkpoints,
        save_progress=args.save_progress,
        epochs=args.epochs,
        normalize_channels=args.normalize_channels,
        lr_schedule=args.lr_schedule,
        lr_multiplier=args.lr_multiplier,
    )


def get_dataloader(
    cells: list[Cell],
    filenames: list[tuple[list[str], list[dict]]],
    batch_size: int,
    n_processes: int,
    pin_memory: bool,
    auto_shuffle: bool,
    augment: bool,
    augment_likelihood: float,
    normalize_channels: bool,
) -> tuple[DataLoader, CuboidTiffDataset]:
    points_filenames = [f[0] for f in filenames]

    points_norm = None
    if normalize_channels:
        points_norm = []
        for names, channels_norm in filenames:
            # check the first channel for metadata. We expect all or none
            # of the channels to have metadata
            if not channels_norm[0]:
                raise ValueError(f"Data mean and std not found for {names}")

            norms = [(ch["mean"], ch["std"]) for ch in channels_norm]
            points_norm.append(norms)

    dataset = CuboidTiffDataset(
        points=cells,
        points_filenames=points_filenames,
        points_normalization=points_norm,
        data_voxel_sizes=(1, 1, 1),
        network_voxel_sizes=(1, 1, 1),
        network_cuboid_voxels=(CUBE_DEPTH, CUBE_HEIGHT, CUBE_WIDTH),
        axis_order=("z", "y", "x"),
        target_output="label",
        augment=augment,
        augment_likelihood=augment_likelihood,
    )
    # we use our own sampler so we can control the ordering
    sampler = CuboidBatchSampler(
        dataset=dataset,
        batch_size=batch_size,
        auto_shuffle=auto_shuffle,
    )
    data_loader = DataLoader(
        dataset=dataset,
        sampler=sampler,
        batch_size=None,
        num_workers=n_processes,
        pin_memory=pin_memory,
    )
    return data_loader, dataset


def run(
    output_dir,
    yaml_file,
    n_free_cpus=2,
    trained_model=None,
    model_weights=None,
    install_path=DEFAULT_DOWNLOAD_DIRECTORY,
    model="resnet50_tv",
    network_depth="50",
    learning_rate=0.0001,
    continue_training=False,
    test_fraction=0.1,
    batch_size=16,
    no_augment=False,
    tensorboard=False,
    save_weights=False,
    no_save_checkpoints=False,
    save_progress=False,
    epochs=100,
    max_workers: int = 3,
    pin_memory: bool = True,
    normalize_channels: bool = False,
    lr_schedule: Sequence[int] = (),
    lr_multiplier: float = 0.1,
    augment_likelihood: float = 0.9,
):
    start_time = datetime.now()

    ensure_directory_exists(output_dir)
    model_weights = prep_model_weights(
        model_weights=model_weights,
        install_path=install_path,
        model_name=model,
    )

    yaml_contents = parse_yaml(yaml_file)

    tiff_files = get_tiff_files(yaml_contents)
    logger.info(
        f"Found {sum(len(imlist) for imlist in tiff_files)} images "
        f"from {len(yaml_contents)} datasets "
        f"in {len(yaml_file)} yaml files"
    )

    model = get_model(
        existing_model=trained_model,
        model_weights=model_weights,
        network_depth=models[network_depth],
        learning_rate=learning_rate,
        continue_training=continue_training,
    )

    filenames_train, cells_train = make_tiff_lists(tiff_files)

    n_processes = get_num_processes(min_free_cpu_cores=n_free_cpus)
    n_processes = min(n_processes, max_workers)
    if test_fraction > 0:
        logger.info("Splitting data into training and validation datasets")
        (
            filenames_train,
            filenames_test,
            cells_train,
            cells_test,
        ) = train_test_split(
            filenames_train,
            cells_train,
            test_size=test_fraction,
        )

        logger.info(
            f"Using {len(filenames_train)} images for training and "
            f"{len(filenames_test)} images for validation"
        )
        validation_data_loader, validation_dataset = get_dataloader(
            cells_test,
            filenames_test,
            batch_size,
            n_processes,
            pin_memory,
            auto_shuffle=False,
            augment=False,
            augment_likelihood=augment_likelihood,
            normalize_channels=normalize_channels,
        )

        # for saving checkpoints
        base_checkpoint_file_name = "-epoch.{epoch:02d}-loss-{val_loss:.3f}"

    else:
        logger.info("No validation data selected.")
        validation_data_loader = None
        validation_dataset = None
        base_checkpoint_file_name = "-epoch.{epoch:02d}"

    training_data_loader, training_dataset = get_dataloader(
        cells_train,
        filenames_train,
        batch_size,
        n_processes,
        pin_memory,
        auto_shuffle=True,
        augment=not no_augment,
        augment_likelihood=augment_likelihood,
        normalize_channels=normalize_channels,
    )
    callbacks = []

    if tensorboard:
        logdir = output_dir / "tensorboard"
        ensure_directory_exists(logdir)
        tensorboard = TensorBoard(
            log_dir=logdir,
            histogram_freq=0,
            write_graph=True,
            update_freq="epoch",
        )
        callbacks.append(tensorboard)

    if not no_save_checkpoints:
        if save_weights:
            filepath = str(
                output_dir
                / ("weight" + base_checkpoint_file_name + ".weights.h5")
            )
        else:
            filepath = str(
                output_dir / ("model" + base_checkpoint_file_name + ".keras")
            )

        checkpoints = ModelCheckpoint(
            filepath,
            save_weights_only=save_weights,
        )
        callbacks.append(checkpoints)

    if save_progress:
        csv_filepath = str(output_dir / "training.csv")
        csv_logger = CSVLogger(csv_filepath)
        callbacks.append(csv_logger)

    if lr_schedule:
        # we need to drop the lr by a given schedule. This is called at the
        # start of each epoch and is zero based. E.g. if epoch 10 is listed,
        # it'll drop at the start of the 11th epoch.
        lr_callback = partial(
            lr_scheduler, multiplier=lr_multiplier, epoch_list=lr_schedule
        )
        callbacks.append(LearningRateScheduler(lr_callback))

    logger.info("Beginning training.")
    if n_processes:
        training_dataset.start_dataset_thread(n_processes)
        if validation_dataset is not None:
            validation_dataset.start_dataset_thread(n_processes)
    try:
        model.fit(
            x=training_data_loader,
            validation_data=validation_data_loader,
            epochs=epochs,
            callbacks=callbacks,
        )
    finally:
        try:
            training_dataset.stop_dataset_thread()
        finally:
            if validation_dataset is not None:
                validation_dataset.stop_dataset_thread()

    if save_weights:
        logger.info("Saving model weights")
        model.save_weights(output_dir / "model.weights.h5")
    else:
        logger.info("Saving model")
        model.save(output_dir / "model.keras")

    logger.info(
        "Finished training, " "Total time taken: %s",
        datetime.now() - start_time,
    )


if __name__ == "__main__":
    cli()
