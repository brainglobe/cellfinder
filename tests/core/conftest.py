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
    data_zyx: np.ndarray,
    center_xyz,
    radius: int,
    center_fill_value: int,
    outside_fill_value: int,
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
    inside = dist <= radius

    if center_fill_value == outside_fill_value:
        data_zyx[inside] = center_fill_value
    else:
        diff = center_fill_value - outside_fill_value
        assert diff > 0
        max_val = np.max(dist[inside])
        dist /= max_val
        dist = (1 - dist) * diff + outside_fill_value
        data_zyx[inside] = dist[inside]


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
    mark_sphere(signal_array, c_xyz, 2, 100, 100)

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
        mark_sphere(signal_array, center, radius, 100, 100)

    return signal_array, background_array, centers_xyz


@pytest.fixture(scope="session")
def synthetic_intensity_dropoff_spot() -> tuple[
    np.ndarray,
    np.ndarray,
    tuple[int, int, int],
    tuple[int, int, int],
    tuple[int, int, int],
]:
    """
    Creates a synthetic signal array with a single spherical spot
    that slowly drops off in intensity in the x direction
    as a 3d numpy array to be used for cell detection testing.

    The center value is bright, less bright at the edges of the
    sphere, and darker further at some distance in the x direction.
    The array is a floating type and you must convert it to the right
    data type for your tests. Also, `n_sds_above_mean_thresh` must be 0
    to exclude any non-zero areas.

    It returns the signal and background np arrays and 3 x, y, z position
    tuples `center`, `mid`, `end`. `center` is the center of the sphere.
    `mid` is the mid-point of all non-zero voxels. `end` is the of the non-zero
    voxels in the x-direction. I.e. centered in y, z. But in x it's where the
    voxels are the least bright at the end of the falloff.
    """
    # overall shape and center of sphere
    shape_zyx = 20, 50, 50
    c_zyx = 10, 25, 15
    # radius of the sphere
    r = 7
    # the dist from the center, along which the brightness will dropoff *after*
    # the end of the sphere is reached in x
    x_r = 22
    # brightness of sphere center
    center_val = 1000
    # brightness of the sphere at its radius
    bright_fill = 100
    # brightness of the end of falloff at x_r.
    mute_fill = 10
    # center of all the non-zero voxels
    c_overall_zyx = 10, 25, 15 - r + (r + x_r - 1) // 2
    # pos of the end of the falloff
    end_zyx = 10, 25, 15 + x_r

    signal_array = np.zeros(shape_zyx)
    background_array = np.zeros_like(signal_array)
    # mark the sphere
    mark_sphere(signal_array, c_zyx[::-1], r, center_val, bright_fill)

    # add the brightness dropoff. Start with overall grid
    z, y, x = np.mgrid[
        0 : shape_zyx[0] : 1, 0 : shape_zyx[1] : 1, 0 : shape_zyx[2] : 1
    ]

    # locate voxels only within the z and y radius
    within_z_rad = np.abs(z - c_zyx[0]) <= r
    within_y_rad = np.abs(y - c_zyx[1]) <= r
    # and the voxels that is on the right half of the sphere, but not past the
    # end of the dropoff area
    within_x_pos_rad = np.logical_and(x - c_zyx[2] >= 0, x - c_zyx[2] <= x_r)
    within_yz_rad = np.logical_and(within_y_rad, within_z_rad)
    # voxels that are in z, and y radius, are positive x, but below dropoff end
    within_xyz_rad = np.logical_and(within_yz_rad, within_x_pos_rad)
    # only get the voxels in the above volume, but *outside* the marked sphere.
    # These voxels will be updated with gradient dropoff
    valid_and_outside_x_r = np.logical_and(
        within_xyz_rad, np.logical_not(signal_array)
    )

    # mark the dropoff voxels in the range between bright_fill and mute_fill,
    # with lower intensity further from the sphere center (outside the sphere)
    dist = np.sqrt(
        (x - c_zyx[2]) ** 2 + (y - c_zyx[1]) ** 2 + (z - c_zyx[0]) ** 2
    )
    r_dist = dist[c_zyx[0], c_zyx[1], c_zyx[2] + r]
    # get the distance relative to dist at the sphere radius
    dist -= r_dist
    # the max distance at the dropoff
    x_r_dist = dist[c_zyx[0], c_zyx[1], c_zyx[2] + x_r]

    # normalize ratio to 0-1 so we can subtract and go from bright_fill to
    # mute_fill at the furthest dist. We use sqrt to make the brightness drop
    # off faster
    ratio = np.sqrt(np.abs(dist / x_r_dist))
    dist += bright_fill - ratio * (bright_fill - mute_fill)
    signal_array[valid_and_outside_x_r] = dist[valid_and_outside_x_r]

    return (
        signal_array,
        background_array,
        c_zyx[::-1],
        c_overall_zyx[::-1],
        end_zyx[::-1],
    )


@pytest.fixture(scope="session")
def repo_data_path() -> Path:
    """
    The root path where the data used during test is stored
    """
    # todo: use mod relative paths to find data instead of depending on cwd
    return Path(__file__).parent.parent / "data"
