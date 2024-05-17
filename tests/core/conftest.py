import os
from typing import Tuple

import keras.src.backend.common.global_state
import numpy as np
import pytest
import torch.backends.mps
from skimage.filters import gaussian

from cellfinder.core.download.download import (
    DEFAULT_DOWNLOAD_DIRECTORY,
    download_models,
)


@pytest.fixture(scope="session", autouse=True)
def set_device_macos_ci_testing():
    device = os.environ["CELLFINDER_TEST_DEVICE"]

    if device == "cpu" and torch.backends.mps.is_available():
        keras.src.backend.common.global_state.set_global_attribute(
            "torch_device", device
        )


@pytest.fixture(scope="session")
def download_default_model():
    """
    Check that the classification model is already downloaded
    at the beginning of a pytest session.
    """
    download_models("resnet50_tv", DEFAULT_DOWNLOAD_DIRECTORY)


@pytest.fixture(scope="session")
def synthetic_bright_spots() -> Tuple[np.ndarray, np.ndarray]:
    """
    Creates a synthetic signal array with grid of bright spots
    in a 3d numpy array to be used for cell detection testing.
    """
    shape = (100, 100, 100)

    signal_array = np.zeros(shape)
    signal_array[25, 25, 25] = 1
    signal_array[75, 25, 25] = 1
    signal_array[25, 75, 25] = 1
    signal_array[25, 25, 75] = 1
    signal_array[75, 75, 25] = 1
    signal_array[75, 25, 75] = 1
    signal_array[25, 75, 75] = 1
    signal_array[75, 75, 75] = 1

    # convert to 16-bit integer
    signal_array = (signal_array * 65535).astype(np.uint16)

    # blur a bit to roughly match the size of the cells in the sample data
    signal_array = gaussian(signal_array, sigma=2, preserve_range=True).astype(
        np.uint16
    )

    background_array = np.zeros_like(signal_array)

    return signal_array, background_array
