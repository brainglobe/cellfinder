import os
import platform
from typing import Tuple

import numpy as np
import pytest
from skimage.filters import gaussian

from cellfinder.core.download import models
from cellfinder.core.tools.prep import DEFAULT_INSTALL_PATH


@pytest.fixture(scope="session", autouse=True)
def macos_use_cpu_only():
    """
    Ensure torch only uses the CPU when running on arm based macOS.
    """
    if platform.system() == "Darwin" and platform.processor() == "arm":
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"


@pytest.fixture(scope="session")
def download_default_model():
    """
    Check that the classification model is already downloaded
    at the beginning of a pytest session.
    """
    models.main("resnet50_tv", DEFAULT_INSTALL_PATH)


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
