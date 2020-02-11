"""
main
===============

Trains a network based on a yaml file specifying cubes of cells/non cells.

N.B imports are within functions to prevent tensorflow being imported before
it's warnings are silenced
"""


import os

from datetime import datetime
from pathlib import Path
from argparse import (
    ArgumentParser,
    ArgumentDefaultsHelpFormatter,
    ArgumentTypeError,
)
from sklearn.model_selection import train_test_split
from imlib.general.numerical import check_positive_float, check_positive_int
from imlib.general.system import ensure_directory_exists, get_num_processes
from imlib.IO.cells import find_relevant_tiffs
from imlib.IO.yaml import read_yaml_section

tf_suppress_log_messages = [
    "sample_weight modes were coerced from",
    "multiprocessing can interact badly with TensorFlow",
]

models = {
    "18": "18-layer",
    "34": "34-layer",
    "50": "50-layer",
    "101": "101-layer",
    "152": "152-layer",
}


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


def training_parse():
    from cellfinder.tools.parser import misc_parse
    from cellfinder.download.cli import model_parser, download_directory_parser

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
        "--save-weights",
        dest="save_weights",
        action="store_true",
        help="Only store the model weights, and not the full model. Useful to "
        "save storage space.",
    )
    training_parser.add_argument(
        "--save-checkpoints",
        dest="save_checkpoints",
        action="store_true",
        help="Store the model after each epoch",
    )
    training_parser.add_argument(
        "--tensorboard",
        action="store_true",
        help="Log to output_directory/tensorboard",
    )

    training_parser = misc_parse(training_parser)
    training_parser = model_parser(training_parser)
    training_parser = download_directory_parser(training_parser)
    args = training_parser.parse_args()

    return args


def parse_yaml(yaml_files, section="data"):
    data = []
    for yaml_file in yaml_files:
        data.extend(read_yaml_section(yaml_file, section))
    return data


def get_tiff_files(yaml_contents):
    from cellfinder.tools.tiff import TiffList, TiffDir

    tiff_lists = []
    for d in yaml_contents:
        if d["bg_channel"] < 0:
            channels = [d["signal_channel"]]
        else:
            channels = [d["signal_channel"], d["bg_channel"]]
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
                    d["type"],
                )
            )
        else:
            tiff_lists.append(TiffDir(d["cube_dir"], channels, d["type"]))

    tiff_files = [tiff_dir.make_tifffile_list() for tiff_dir in tiff_lists]
    return tiff_files


def main(max_workers=3):
    from cellfinder.main import suppress_tf_logging

    suppress_tf_logging(tf_suppress_log_messages)

    from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint

    from cellfinder.tools.prep import prep_training
    from cellfinder.classify.tools import make_lists, get_model
    from cellfinder.classify.cube_generator import CubeGeneratorFromDisk

    start_time = datetime.now()
    args = training_parse()
    output_dir = Path(args.output_dir)
    ensure_directory_exists(output_dir)
    args = prep_training(args)
    yaml_contents = parse_yaml(args.yaml_file)
    tiff_files = get_tiff_files(yaml_contents)

    # Too many workers doesn't increase speed, and uses huge amounts of RAM
    workers = get_num_processes(
        min_free_cpu_cores=args.n_free_cpus, n_max_processes=max_workers
    )

    model = get_model(
        existing_model=args.trained_model,
        model_weights=args.model_weights,
        network_depth=models[args.network_depth],
        learning_rate=args.learning_rate,
        continue_training=args.continue_training,
    )

    signal_train, background_train, labels_train = make_lists(tiff_files)

    if args.test_fraction > 0:
        (
            signal_train,
            signal_test,
            background_train,
            background_test,
            labels_train,
            labels_test,
        ) = train_test_split(
            signal_train,
            background_train,
            labels_train,
            test_size=args.test_fraction,
        )
        validation_generator = CubeGeneratorFromDisk(
            signal_test,
            background_test,
            labels=labels_test,
            batch_size=args.batch_size,
            train=True,
        )
    else:
        validation_generator = None

    training_generator = CubeGeneratorFromDisk(
        signal_train,
        background_train,
        labels=labels_train,
        batch_size=args.batch_size,
        shuffle=True,
        train=True,
        augment=not args.no_augment,
    )
    callbacks = []

    if args.tensorboard:
        logdir = output_dir / "tensorboard"
        ensure_directory_exists(logdir)
        tensorboard = TensorBoard(
            log_dir=logdir,
            histogram_freq=0,
            write_graph=True,
            update_freq="epoch",
        )
        callbacks.append(tensorboard)

    if args.save_checkpoints:
        if args.save_weights:
            filepath = str(
                output_dir / "weights.{epoch:02d}-{val_loss:.3f}.h5"
            )
        else:
            filepath = str(output_dir / "model.{epoch:02d}-{val_loss:.3f}.h5")

        checkpoints = ModelCheckpoint(
            filepath, save_weights_only=args.save_weights
        )
        callbacks.append(checkpoints)

    model.fit(
        training_generator,
        validation_data=validation_generator,
        use_multiprocessing=True,
        workers=workers,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    if args.save_weights:
        print("Saving model weights")
        model.save_weights(str(output_dir / "model_weights.h5"))
    else:
        print("Saving model")
        model.save(output_dir / "model.h5")

    print(
        "Finished training, " "Total time taken: %s",
        datetime.now() - start_time,
    )


if __name__ == "__main__":
    main()
