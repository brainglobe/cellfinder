from datetime import datetime
from pathlib import Path

import dask.array as da
import keras
import numpy as np
import zarr
from sklearn.model_selection import train_test_split

from cellfinder.core import logger

# output_dir = Path("/home/igor/ADL4IA-2025/2-channel")

start_time = datetime.now()

# ensure_directory_exists(output_dir)

data_path = Path("/Volumes/AT_T7_1TB_A/ADL4IA-2025/data.zarr")
model2_chan_path = Path(
    "/Volumes/AT_T7_1TB_A/ADL4IA-2025/2-channel/2-channel/best/model-epoch.31-loss-0.084.keras"
)

data_group = zarr.open(data_path)

background = False
data_train = da.from_array(data_group["raw_data"])
labels_train = da.from_array(data_group["labels"])

learning_rate = 0.0001
test_fraction = 0.1
val_fraction = 0.1
batch_size = 64
epochs = 100
random_seed = 42

model = keras.saving.load_model(model2_chan_path)

logger.info("Get test set")
data_train, data_test, labels_train, labels_test = train_test_split(
    data_train,
    labels_train,
    test_size=test_fraction,
    stratify=labels_train,
    random_state=random_seed,
)

data_test = data_test.compute()

predictions = model.predict(data_test, batch_size=batch_size, verbose=1)
predictions = predictions.round()
predictions = predictions.astype("uint16")

predictions = np.argmax(predictions, axis=1)

accuracy = np.sum(predictions == labels_test) / len(labels_test)

print(f"Accuracy on test set using 2 channels: {accuracy.compute()}")

model1_chan_path = Path(
    "/Volumes/AT_T7_1TB_A/ADL4IA-2025/1-channel/model.keras"
)

model = keras.saving.load_model(model1_chan_path)

predictions = model.predict(
    data_test[..., 0], batch_size=batch_size, verbose=1
)
predictions = predictions.round()
predictions = predictions.astype("uint16")

predictions = np.argmax(predictions, axis=1)

accuracy = np.sum(predictions == labels_test) / len(labels_test)

print(f"Acurracy on test set using 1 channel: {accuracy.compute()}")
