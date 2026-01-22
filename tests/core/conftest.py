import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pooch
import pytest
import torch.backends.mps
from skimage.filters import gaussian

from cellfinder.core.download.download import (
    DEFAULT_DOWNLOAD_DIRECTORY,
    download_models,
)
from cellfinder.core.tools.system import force_cpu


@pytest.fixture(scope="session", autouse=True)
def set_device_arm_macos_ci():
    """
    Ensure that the device is set to CPU when running on arm based macOS
    GitHub runners. This is to avoid the following error:
    https://discuss.pytorch.org/t/mps-back-end-out-of-memory-on-github-action/189773/5
    """
    if (
        os.getenv("GITHUB_ACTIONS") == "true"
        and torch.backends.mps.is_available()
    ):
        force_cpu()


def pytest_collection_modifyitems(session, config, items: List[pytest.Item]):
    # this hook is called by pytest after test collection. Move the
    # test_detection test to the end because if it's run in the middle we run
    # into numba issue #9576 and the tests fail
    # end_files are moved to the end, in the given order
    end_files = [
        "test_connected_components_labelling.py",
        "test_structure_detection.py",
    ]

    items_new = [t for t in items if t.path.name not in end_files]
    for name in end_files:
        items_new.extend([t for t in items if t.path.name == name])

    items[:] = items_new


@pytest.fixture
def test_data_registry():
    """
    Create a test data registry for BrainGlobe.

    Returns:
        pooch.Pooch: The test data registry object.

    """
    registry = pooch.create(
        path=pooch.os_cache("brainglobe_test_data"),
        base_url="https://gin.g-node.org/BrainGlobe/test-data/raw/master/cellfinder/",
        env="BRAINGLOBE_TEST_DATA_DIR",
    )

    registry.load_registry(
        Path(__file__).parent.parent / "data" / "pooch_registry.txt"
    )
    return registry


def mark_sphere(
    data_zyx: np.ndarray, center_xyz, radius: int, fill_value: int
) -> None:
    shape_zyx = data_zyx.shape

    z, y, x = np.mgrid[
        0 : shape_zyx[0] : 1, 0 : shape_zyx[1] : 1, 0 : shape_zyx[2] : 1
    ]
    dist = np.sqrt(
        (x - center_xyz[0]) ** 2
        + (y - center_xyz[1]) ** 2
        + (z - center_xyz[2]) ** 2
    )
    # 100 seems to be the right size so std is not too small for filters
    data_zyx[dist <= radius] = fill_value


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


@pytest.fixture(scope="session")
def synthetic_single_spot() -> (
    Tuple[np.ndarray, np.ndarray, Tuple[int, int, int]]
):
    """
    Creates a synthetic signal array with a single spherical spot
    in a 3d numpy array to be used for cell detection testing.

    The max value is 100 and min is zero. The array is a floating type.
    You must convert it to the right data type for your tests.
    Also, `n_sds_above_mean_thresh` must be 1 or larger.
    """
    shape_zyx = 20, 50, 50
    c_xyz = 25, 25, 10

    signal_array = np.zeros(shape_zyx)
    background_array = np.zeros_like(signal_array)
    mark_sphere(signal_array, center_xyz=c_xyz, radius=2, fill_value=100)

    # 1 std should be larger, so it can be considered bright
    assert np.mean(signal_array) + np.std(signal_array) > 1

    return signal_array, background_array, c_xyz


@pytest.fixture(scope="session")
def synthetic_spot_clusters() -> (
    Tuple[np.ndarray, np.ndarray, List[Tuple[int, int, int]]]
):
    """
    Creates a synthetic signal array with a 4 overlapping spherical spots
    in a 3d numpy array to be used for cell cluster splitting testing.

    The max value is 100 and min is zero. The array is a floating type.
    You must convert it to the right data type for your tests.
    Also, `n_sds_above_mean_thresh` must be 1 or larger.
    """
    shape_zyx = 20, 100, 100
    radius = 5
    s = 50 - radius * 4
    centers_xyz = [
        (s, 50, 10),
        (s + 2 * radius - 1, 50, 10),
        (s + 4 * radius - 2, 50, 10),
        (s + 6 * radius - 3, 50, 10),
    ]

    signal_array = np.zeros(shape_zyx)
    background_array = np.zeros_like(signal_array)

    for center in centers_xyz:
        mark_sphere(
            signal_array, center_xyz=center, radius=radius, fill_value=100
        )

    return signal_array, background_array, centers_xyz


@pytest.fixture(scope="session")
def repo_data_path() -> Path:
    """
    The root path where the data used during test is stored
    """
    # todo: use mod relative paths to find data instead of depending on cwd
    return Path(__file__).parent.parent / "data"
