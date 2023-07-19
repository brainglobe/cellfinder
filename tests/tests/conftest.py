import os
from typing import Tuple

import numpy as np
import pytest
from skimage.filters import gaussian

from cellfinder_core.download import models
from cellfinder_core.tools.prep import DEFAULT_INSTALL_PATH


@pytest.fixture(scope="session")
def no_free_cpus() -> int:
    """
    Set number of free CPUs so all available CPUs are used by the tests.
    """
    return 0


@pytest.fixture(scope="session")
def run_on_one_cpu_only() -> int:
    """
    Set number of free CPUs so tests can use exactly one CPU.
    """
    cpus = os.cpu_count()
    if cpus is not None:
        return cpus - 1
    else:
        raise ValueError("No CPUs available.")


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
