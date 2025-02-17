from datetime import datetime
from pathlib import Path

import dask.array as da
import zarr
from brainglobe_utils.general.system import (
    ensure_directory_exists,
    get_num_processes,
)
from keras.src.callbacks import CSVLogger, ModelCheckpoint
from sklearn.model_selection import train_test_split

from cellfinder.core import logger
from cellfinder.core.classify.cube_generator import CubeGeneratorFromDask
from cellfinder.core.classify.tools import get_model

output_dir = Path("/home/igor/ADL4IA-2025/2-channel")

start_time = datetime.now()

ensure_directory_exists(output_dir)

data_path = Path("/home/igor/ADL4IA-2025/data.zarr")

data_group = zarr.open(data_path)

data_train = da.from_array(data_group["raw_data"]).compute()
labels_train = da.from_array(data_group["labels"]).compute()

learning_rate = 0.0001
test_fraction = 0.1
val_fraction = 0.1
batch_size = 64
epochs = 2

model = get_model(
    network_depth="50-layer",
    learning_rate=learning_rate,
    continue_training=False,
)

n_processes = get_num_processes(min_free_cpu_cores=2)

logger.info("Reserving test set")
data_train, data_test, labels_train, labels_test = train_test_split(
    data_train, labels_train, test_size=test_fraction, stratify=labels_train
)

logger.info("Splitting data into training and validation datasets")
(
    data_train,
    data_val,
    labels_train,
    labels_val,
) = train_test_split(
    data_train,
    labels_train,
    test_size=test_fraction,
    stratify=labels_train,
)

logger.info(
    f"Using {len(data_train)} images for training and "
    f"{len(data_val)} images for validation"
)
validation_generator = CubeGeneratorFromDask(
    data_val,
    labels=labels_val,
    batch_size=batch_size,
    train=True,
    use_multiprocessing=False,
    workers=n_processes,
)

# for saving checkpoints
base_checkpoint_file_name = "-epoch.{epoch:02d}-loss-{val_loss:.3f}"

training_generator = CubeGeneratorFromDask(
    data_train,
    labels=labels_train,
    batch_size=batch_size,
    shuffle=True,
    train=True,
    augment=True,
    use_multiprocessing=False,
    workers=n_processes,
)

callbacks = []

filepath = str(output_dir / ("model" + base_checkpoint_file_name + ".keras"))

checkpoints = ModelCheckpoint(
    filepath,
    save_weights_only=False,
)
callbacks.append(checkpoints)

csv_filepath = str(output_dir / "training.csv")
csv_logger = CSVLogger(csv_filepath)
callbacks.append(csv_logger)

logger.info("Beginning training.")
# Keras 3.0: `use_multiprocessing` input is set in the
# `training_generator` (False by default)
model.fit(
    training_generator,
    validation_data=validation_generator,
    epochs=epochs,
    callbacks=callbacks,
)

logger.info("Saving model")
model.save(output_dir / "model.keras")

logger.info(
    "Finished training, " "Total time taken: %s",
    datetime.now() - start_time,
)
