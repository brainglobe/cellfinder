import os

import cellfinder.cells.tools as cell_tools
from cellfinder.IO.cells import get_cells

data_dir = os.path.join("tests", "data")

xml_path = os.path.join(data_dir, "cells", "cells.xml")


def test_group_cells_by_z():
    z_planes_validate = [
        1272,
        1273,
        1274,
        1275,
        1276,
        1277,
        1278,
        1279,
        1280,
        1281,
        1282,
        1283,
        1284,
        1285,
        1286,
        1287,
        1288,
        1289,
        1290,
        1291,
        1292,
        1294,
        1295,
        1296,
        1297,
        1298,
    ]

    cell_numbers_in_groups_validate = [
        1,
        3,
        7,
        8,
        3,
        1,
        4,
        3,
        1,
        2,
        2,
        1,
        1,
        2,
        5,
        2,
        2,
        2,
        3,
        1,
        1,
        6,
        1,
        1,
        1,
        1,
    ]

    cells = get_cells(xml_path)
    cells_groups = cell_tools.group_cells_by_z(cells)
    z_planes_test = list(cells_groups.keys())
    z_planes_test.sort()

    assert z_planes_validate == z_planes_test

    cell_numbers_in_groups_test = [
        len(cells_groups[plane]) for plane in z_planes_test
    ]
    assert cell_numbers_in_groups_validate == cell_numbers_in_groups_test
