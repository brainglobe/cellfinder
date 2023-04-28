import numpy as np
import pytest

from cellfinder_core.detect.filters.volume.structure_detection import (
    CellDetector,
    Point,
    get_non_zero_dtype_min,
    get_structure_centre_wrapper,
)


def coords_to_points(coords_arrays):
    # Convert from arrays to dicts
    coords = {}
    for sid in coords_arrays:
        coords[sid] = []
        for row in coords_arrays[sid]:
            coords[sid].append(Point(row[0], row[1], row[2]))
    return coords


@pytest.mark.parametrize(
    ("dtype", "expected"),
    [
        (np.uint64, 2**64 - 1),
        (np.uint32, 2**32 - 1),
        (np.uint16, 2**16 - 1),
        (np.uint8, 2**8 - 1),
    ],
)
def test_get_non_zero_dtype_min(dtype, expected):
    assert get_non_zero_dtype_min(np.arange(10, dtype=dtype)) == 1
    assert get_non_zero_dtype_min(np.zeros(10, dtype=dtype)) == expected


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
        {
            1: [
                Point(1, 1, 0),
                Point(1, 0, 1),
                Point(0, 1, 1),
                Point(1, 1, 1),
            ]
        },
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

    previous_plane = None
    for plane in data:
        previous_plane = detector.process(plane, previous_plane)

    coords = detector.get_coords_dict()
    assert coords_to_points(coords) == expected_coords
