import os
import sys
from math import isclose

import brainglobe_utils.IO.cells as cell_io
import pytest

from cellfinder.main import main as cellfinder_run

data_dir = os.path.join(
    os.getcwd(), "tests", "data", "integration", "detection"
)
signal_data = os.path.join(data_dir, "crop_planes", "ch0")
background_data = os.path.join(data_dir, "crop_planes", "ch1")
cells_validation_xml = os.path.join(data_dir, "cell_classification.xml")

x_pix = "2"
y_pix = "2"
z_pix = "5"

DETECTION_TOLERANCE = 2


# FIXME: This isn't a very good example


@pytest.mark.slow
def test_detection_full(tmpdir):
    cellfinder_args = [
        "cellfinder",
        "-s",
        signal_data,
        "-b",
        background_data,
        "-o",
        str(tmpdir),
        "-v",
        z_pix,
        y_pix,
        x_pix,
        "--orientation",
        "psl",
        "--n-free-cpus",
        "0",
        "--no-register",
        "--save-planes",
    ]
    sys.argv = cellfinder_args
    cellfinder_run()

    cells_test_xml = tmpdir / "points" / "cell_classification.xml"

    cells_validation = cell_io.get_cells(cells_validation_xml)
    cells_test = cell_io.get_cells(str(cells_test_xml))

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
    # Check that planes are saved
    for i in range(2, 30):
        assert (
            tmpdir / "processed_planes" / f"plane_{str(i).zfill(4)}.tif"
        ).exists()
