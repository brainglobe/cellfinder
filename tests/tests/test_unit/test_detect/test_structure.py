import numpy as np

from cellfinder_core.detect.filters.volume.structure_detection import (
    CellDetector,
)


def test_detection():
    width, height = 3, 2
    depth = 2

    data = np.zeros((depth, width, height)).astype(np.uint64)

    detector = CellDetector(width, height, start_z=0)

    data[0, 0, 0] = detector.SOMA_CENTRE_VALUE
    data[0, 2, 0] = detector.SOMA_CENTRE_VALUE
    for plane in data:
        detector.process(plane)

    coords = detector.get_coords_list()
    expected = {1: [{"x": 0, "y": 0, "z": 0}], 2: [{"x": 2, "y": 0, "z": 0}]}
    assert coords == expected
