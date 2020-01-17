import os
from natsort import natsorted

from cellfinder.cells import cells

data_dir = os.path.join("tests", "data")
cubes_dir = os.path.join(data_dir, "cube_extract", "cubes")


def test_pos_from_file_name():
    positions_validate = [
        [392, 522, 10],
        [340, 1004, 15],
        [340, 1004, 15],
        [392, 522, 10],
    ]
    cube_files = os.listdir(cubes_dir)
    positions = []
    for file in cube_files:
        positions.append(cells.pos_from_file_name(file))
    assert natsorted(positions) == natsorted(positions_validate)
