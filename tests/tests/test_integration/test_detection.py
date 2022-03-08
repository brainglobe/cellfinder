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

    signal_array = signal_array[0:1]
    background_array = background_array[0:1]
    planes_done = []

    def detect_callback(plane):
        planes_done.append(plane)

    main(
        signal_array,
        background_array,
        voxel_sizes,
        detect_callback=detect_callback,
    )

    np.testing.assert_equal(planes_done, [0])
