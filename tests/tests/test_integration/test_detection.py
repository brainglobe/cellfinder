import os
import sys
import pytest

from math import isclose

from cellfinder.main import main as cellfinder_run
import imlib.IO.cells as cell_io

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
    tmpdir = str(tmpdir)

    cellfinder_args = [
        "cellfinder_run",
        "-s",
        signal_data,
        "-b",
        background_data,
        "-o",
        tmpdir,
        "-x",
        x_pix,
        "-y",
        y_pix,
        "-z",
        z_pix,
        "--n-free-cpus",
        "0",
        "--no-standard-space",
        "--save-planes",
    ]
    sys.argv = cellfinder_args
    cellfinder_run()

    cells_test_xml = os.path.join(tmpdir, "cell_classification.xml")

    cells_validation = cell_io.get_cells(cells_validation_xml)
    cells_test = cell_io.get_cells(cells_test_xml)

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
