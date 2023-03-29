import numpy as np
import pytest

from cellfinder_core.detect.filters.volume.structure_detection import (
    CellDetector,
    Point,
    get_non_zero_ull_min_wrapper,
    get_structure_centre_wrapper,
)


def test_get_non_zero_ull_min():
    assert get_non_zero_ull_min_wrapper(list(range(10))) == 1
    assert get_non_zero_ull_min_wrapper([0] * 10) == (2**64) - 1


@pytest.fixture()
def three_d_cross():
    return np.array(
        [
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
            [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
        ]
    )


@pytest.fixture()
def structure(three_d_cross):
    coords = np.where(three_d_cross == 1)[::-1]
    s = [Point(*c) for c in zip(coords[0], coords[1], coords[2])]
    return s


def test_get_structure_centre(structure):
    result_point = get_structure_centre_wrapper(structure)
    assert (result_point.x, result_point.y, result_point.z) == (
        1,
        1,
        1,
    )


# Image volume dimensions
width, height = 3, 2
depth = 2

# Each item in the test data contains:
#
# - A list of indices to mark as structure pixels (ordering: [x, z, y])
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
        {1: [Point(0, 0, 0), Point(1, 0, 0)]},
    ),
    (
        # Two pixels connected in a single structure along y
        [(0, 0, 0), (0, 0, 1)],
        {1: [Point(0, 0, 0), Point(0, 1, 0)]},
    ),
    (
        # Two pixels connected in a single structure along z
        [(0, 0, 0), (1, 0, 0)],
        {1: [Point(0, 0, 0), Point(0, 0, 1)]},
    ),
    (
        # Four pixels all connected and spread across x-y-z
        [(0, 0, 0), (1, 0, 0), (1, 1, 0), (1, 0, 1)],
        {1: [Point(0, 0, 0), Point(0, 0, 1), Point(1, 0, 1), Point(0, 1, 1)]},
    ),
    (
        # three initially disconnected pixels that then get merged
        # by a fourth pixel
        [(1, 1, 0), (0, 1, 1), (1, 0, 1), (1, 1, 1)],
        {1: [Point(1, 1, 0), Point(0, 1, 1), Point(1, 0, 1), Point(1, 1, 1)]},
    ),
    (
        # Three pixels in x-y plane that require structure merging
        [(1, 0, 0), (0, 1, 0), (1, 1, 0)],
        {
            1: [
                Point(1, 0, 0),
                Point(0, 0, 1),
                Point(1, 0, 1),
            ]
        },
    ),
    (
        # Two disconnected single-pixel structures
        [(0, 0, 0), (0, 2, 0)],
        {1: [Point(0, 0, 0)], 2: [Point(2, 0, 0)]},
    ),
    (
        # Two disconnected single-pixel structures along a diagonal
        [(0, 0, 0), (1, 1, 1)],
        {1: [Point(0, 0, 0)], 2: [Point(1, 1, 1)]},
    ),
]


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.uint32, np.uint64])
@pytest.mark.parametrize("pixels,expected_coords", test_data)
def test_detection(dtype, pixels, expected_coords):
    data = np.zeros((depth, width, height)).astype(dtype)
    detector = CellDetector(width, height, start_z=0)

    # This is the value used by BallFilter to mark pixels
    max_poss_value = np.iinfo(dtype).max
    for pix in pixels:
        data[pix] = max_poss_value

    for plane in data:
        detector.process(plane)

    coords = detector.get_coords_list()
    assert coords == expected_coords
