import numpy as np
import pytest

from cellfinder_core.detect.filters.volume.structure_detection import (
    CellDetector,
)

# Image volume dimensions
width, height = 3, 2
depth = 2

# Each item in the test data contains:
#
# - A list of indices to mark as structure pixels
# - A dict of expected structure coordinates
test_data = [
    (
        # No pixels
        [],
        {},
    ),
    (
        # Two pixels connected in a single structure along x
        [(0, 0, 0), (0, 1, 0)],
        {1: [{"x": 0, "y": 0, "z": 0}, {"x": 1, "y": 0, "z": 0}]},
    ),
    (
        # Two pixels connected in a single structure along y
        [(0, 0, 0), (0, 0, 1)],
        {1: [{"x": 0, "y": 0, "z": 0}, {"x": 0, "y": 1, "z": 0}]},
    ),
    (
        # Two pixels connected in a single structure along z
        [(0, 0, 0), (1, 0, 0)],
        {1: [{"x": 0, "y": 0, "z": 0}, {"x": 0, "y": 0, "z": 1}]},
    ),
    (
        # Four pixels all connected and spread across x-y-z
        [(0, 0, 0), (1, 0, 0), (1, 1, 0), (1, 0, 1)],
        {
            1: [
                {"x": 0, "y": 0, "z": 0},
                {"x": 0, "y": 0, "z": 1},
                {"x": 1, "y": 0, "z": 1},
                {"x": 0, "y": 1, "z": 1},
            ]
        },
    ),
    (
        # Two disconnected single-pixel structures
        [(0, 0, 0), (0, 2, 0)],
        {1: [{"x": 0, "y": 0, "z": 0}], 2: [{"x": 2, "y": 0, "z": 0}]},
    ),
]


@pytest.mark.parametrize("pixels,expected_coords", test_data)
def test_detection(pixels, expected_coords):
    data = np.zeros((depth, width, height)).astype(np.uint64)
    detector = CellDetector(width, height, start_z=0)

    for pix in pixels:
        data[pix] = detector.SOMA_CENTRE_VALUE

    for plane in data:
        detector.process(plane)

    coords = detector.get_coords_list()
    assert coords == expected_coords
