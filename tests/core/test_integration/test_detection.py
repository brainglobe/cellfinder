import os
from math import isclose

import brainglobe_utils.IO.cells as cell_io
import numpy as np
import pytest
from brainglobe_utils.general.system import get_num_processes

from cellfinder.core.main import main
from cellfinder.core.tools.IO import read_with_dask

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
@pytest.mark.parametrize(
    "cpus_to_leave_available",
    [
        pytest.param(0, id="Leave no CPUS free"),
        pytest.param(-1, id="Only use one CPU"),
    ],
)
def test_detection_full(
    signal_array, background_array, cpus_to_leave_available: int
):
    """
    cpus_to_leave_available is interpreted as follows:

    - For values >=0, this is the number of CPUs to leave available
    to the system when running this test.
    - For values <0, this is HOW MANY CPUS to request be used to
    run the test.

    In each case, we check that we will be running on at least one CPU,
    and not requesting more CPUs than the system can provide.
    """
    # Determine the number of CPUs to leave available
    system_cpus = os.cpu_count()
    # How many CPUs do we want to leave free?
    if cpus_to_leave_available >= 0:
        n_free_cpus = cpus_to_leave_available
    else:
        # Number of CPUs to keep free is <0, interpret as
        # number of CPUs _to use_. Thus;
        # n_free_cpus = system_cpus - |cpus_to_leave_available|
        n_free_cpus = system_cpus - abs(cpus_to_leave_available)
    # Check that there are enough CPUs
    if not 0 <= n_free_cpus < system_cpus:
        RuntimeError(
            f"Not enough CPUS available (you want to leave {n_free_cpus} "
            f"available, but there are only {system_cpus} on the system)."
        )

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
    signal_array,
    background_array,
    mocker,
    cpus_to_leave_free: int = 0,
):
    # Check that processing works when number of planes < number of processes
    nproc = get_num_processes(cpus_to_leave_free)
    n_planes = 2

    # Don't want to bother classifying in this test, so mock classifcation
    mocker.patch("cellfinder.core.classify.classify.main")

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
        n_free_cpus=cpus_to_leave_free,
    )


def test_callbacks(
    signal_array, background_array, cpus_to_leave_free: int = 0
):
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
        n_free_cpus=cpus_to_leave_free,
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


def test_synthetic_data(synthetic_bright_spots, cpus_to_leave_free: int = 0):
    signal_array, background_array = synthetic_bright_spots
    detected = main(
        signal_array,
        background_array,
        voxel_sizes,
        n_free_cpus=cpus_to_leave_free,
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
