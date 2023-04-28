import os
from math import isclose

import imlib.IO.cells as cell_io
import numpy as np
import pytest
from imlib.general.system import get_num_processes

from cellfinder_core.main import main
from cellfinder_core.tools.IO import read_with_dask

data_dir = os.path.join(
    os.getcwd(), "tests", "data", "integration", "detection"
)
signal_data_path = os.path.join(data_dir, "crop_planes", "ch0")
background_data_path = os.path.join(data_dir, "crop_planes", "ch1")
cells_validation_xml = os.path.join(data_dir, "cell_classification.xml")

voxel_sizes = [5, 2, 2]
DETECTION_TOLERANCE = 2


class UnixFS:
    @staticmethod
    def rm(filename):
        os.remove(filename)


def test_unix_fs(mocker):
    mocker.patch("os.remove")
    UnixFS.rm("file")
    os.remove.assert_called_once_with("file")


@pytest.fixture
def signal_array():
    return read_with_dask(signal_data_path)


@pytest.fixture
def background_array():
    return read_with_dask(background_data_path)


# FIXME: This isn't a very good example
@pytest.mark.slow
def test_detection_full(signal_array, background_array, n_free_cpus):
    cells_test = main(
        signal_array,
        background_array,
        voxel_sizes,
        n_free_cpus=n_free_cpus,
    )
    cells_validation = cell_io.get_cells(cells_validation_xml)

    num_non_cells_validation = sum(
        [cell.type == 1 for cell in cells_validation]
    )
    num_cells_validation = sum([cell.type == 2 for cell in cells_validation])

    num_non_cells_test = sum([cell.type == 1 for cell in cells_test])
    num_cells_test = sum([cell.type == 2 for cell in cells_test])

    assert isclose(
        num_non_cells_validation,
        num_non_cells_test,
        abs_tol=DETECTION_TOLERANCE,
    )
    assert isclose(
        num_cells_validation, num_cells_test, abs_tol=DETECTION_TOLERANCE
    )


def test_detection_small_planes(
    signal_array, background_array, n_free_cpus, mocker
):
    # Check that processing works when number of planes < number of processes
    nproc = get_num_processes(n_free_cpus)
    n_planes = 2

    # Don't want to bother classifying in this test, so mock classifcation
    mocker.patch("cellfinder_core.classify.classify.main")

    pytest.mark.skipif(
        nproc < n_planes,
        f"Number of available processes is {nproc}. "
        f"Test only effective if number of available processes >= {n_planes}.",
    )

    main(
        signal_array[0:n_planes],
        background_array[0:n_planes],
        voxel_sizes,
        ball_z_size=5,
        n_free_cpus=n_free_cpus,
    )


def test_callbacks(signal_array, background_array, n_free_cpus):
    # 20 is minimum number of planes needed to find > 0 cells
    signal_array = signal_array[0:20]
    background_array = background_array[0:20]

    planes_done = []
    batches_classified = []
    points_found = []

    def detect_callback(plane):
        planes_done.append(plane)

    def classify_callback(batch):
        batches_classified.append(batch)

    def detect_finished_callback(points):
        points_found.append(points)

    main(
        signal_array,
        background_array,
        voxel_sizes,
        detect_callback=detect_callback,
        classify_callback=classify_callback,
        detect_finished_callback=detect_finished_callback,
        n_free_cpus=n_free_cpus,
    )

    np.testing.assert_equal(planes_done, np.arange(len(signal_array)))
    np.testing.assert_equal(batches_classified, [0])

    ncalls = len(points_found)
    assert ncalls == 1, f"Expected 1 call to callback, got {ncalls}"
    npoints = len(points_found[0])
    assert npoints == 120, f"Expected 120 points, found {npoints}"


def test_floating_point_error(signal_array, background_array):
    signal_array = signal_array.astype(float)
    with pytest.raises(ValueError, match="signal_array must be integer"):
        main(signal_array, background_array, voxel_sizes)


def test_synthetic_data(synthetic_bright_spots, n_free_cpus):
    signal_array, background_array = synthetic_bright_spots
    detected = main(
        signal_array,
        background_array,
        voxel_sizes,
        n_free_cpus=n_free_cpus,
    )
    assert len(detected) == 8


@pytest.mark.parametrize("ndim", [1, 2, 4])
def test_data_dimension_error(ndim):
    # Check for an error when non-3D data input
    shape = (2, 3, 4, 5)[:ndim]
    signal_array = np.random.randint(
        low=0, high=2**16, size=shape, dtype=np.uint16
    )
    background_array = np.random.randint(
        low=0, high=2**16, size=shape, dtype=np.uint16
    )

    with pytest.raises(ValueError, match="Input data must be 3D"):
        main(signal_array, background_array, voxel_sizes)
