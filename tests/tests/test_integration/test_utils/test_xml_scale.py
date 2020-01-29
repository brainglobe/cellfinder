import os
import sys

from imlib.IO.cells import get_cells
from cellfinder.utils.xml_scale import main as cellfinder_xml_scale_run

cells_dir = os.path.join("tests", "data", "cells")
orig_xml_path = os.path.join(cells_dir, "cells.xml")
half_scale_scaled_xml_path = os.path.join(
    cells_dir, "cells_rescaled_05_05_1.xml"
)
order_magnitude_scaled_xml_path = os.path.join(
    cells_dir, "cells_rescaled_10_100_1000.xml"
)

SCALED_XML_FILE_NAME = "cells_rescaled.xml"


def run_xml_scale(xml_file, x_scale, y_scale, z_scale, output_dir):
    cellfinder_xml_scale_args = [
        "cellfinder_xml_scale",
        xml_file,
        "-x",
        str(x_scale),
        "-y",
        str(y_scale),
        "-z",
        str(z_scale),
        "-o",
        str(output_dir),
    ]

    sys.argv = cellfinder_xml_scale_args
    cellfinder_xml_scale_run()

    scaled_cells = get_cells(os.path.join(output_dir, SCALED_XML_FILE_NAME))
    return scaled_cells


def test_xml_scale_cli(tmpdir):
    scaled_cells = run_xml_scale(orig_xml_path, 0.5, 0.5, 1, tmpdir)
    assert scaled_cells == get_cells(half_scale_scaled_xml_path)

    scaled_cells = run_xml_scale(orig_xml_path, 10, 100, 1000, tmpdir)
    assert scaled_cells == get_cells(order_magnitude_scaled_xml_path)
