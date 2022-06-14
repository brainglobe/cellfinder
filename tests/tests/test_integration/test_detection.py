import os
from math import isclose

import imlib.IO.cells as cell_io
import numpy as np
import pytest

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


@pytest.fixture
def signal_array():
    return read_with_dask(signal_data_path)


@pytest.fixture
def background_array():
    return read_with_dask(background_data_path)


# FIXME: This isn't a very good example


@pytest.mark.slow
def test_detection_full():

    signal_array = read_with_dask(signal_data_path)
    background_array = read_with_dask(background_data_path)

    cells_test = main(
        signal_array,
        background_array,
        voxel_sizes,
        n_free_cpus=0,
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


def test_callbacks(signal_array, background_array):
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
        n_free_cpus=0,
    )

    np.testing.assert_equal(planes_done, np.arange(len(signal_array)))
    np.testing.assert_equal(batches_classified, [0])

    ncalls = len(points_found)
    assert ncalls == 1, f"Expected 1 call to callback, got {ncalls}"
    npoints = len(points_found[0])
    assert npoints == 120, f"Expected 120 points, found {npoints}"
