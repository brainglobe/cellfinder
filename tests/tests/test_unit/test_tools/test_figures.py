import os
import numpy as np

import cellfinder.tools.figures as fig_tools

cells_dir = os.path.join("tests", "data", "cells")
orig_xml_path = os.path.join(cells_dir, "cells.xml")
half_scale_scaled_xml_path = os.path.join(
    cells_dir, "cells_rescaled_05_05_1.xml"
)
order_magnitude_scaled_xml_path = os.path.join(
    cells_dir, "cells_rescaled_10_100_1000.xml"
)


def test_get_bins():
    image_size = (1000, 20, 125, 725)
    bin_sizes = (500, 2, 25, 100)

    dim0_bins = np.array((0, 500))
    dim1_bins = np.array((0, 2, 4, 6, 8, 10, 12, 14, 16, 18))
    dim2_bins = np.array((0, 25, 50, 75, 100))
    dim3_bins = np.array((0, 100, 200, 300, 400, 500, 600, 700))
    bins = fig_tools.get_bins(image_size, bin_sizes)

    assert (dim0_bins == bins[0]).all()
    assert (dim1_bins == bins[1]).all()
    assert (dim2_bins == bins[2]).all()
    assert (dim3_bins == bins[3]).all()


# def test_get_cell_location_array():
#     cell_array = fig_tools.get_cell_location_array(
#         orig_xml_path, cells_only=False)
#     xml_scaled_cell_array = fig_tools.get_cell_location_array(
#         half_scale_scaled_xml_path, cells_only=False
#     )
#
#     assert False
