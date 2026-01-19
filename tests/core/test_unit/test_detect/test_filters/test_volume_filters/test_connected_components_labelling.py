from typing import Type

import numpy as np
import pytest

from cellfinder.core.detect.filters.volume.structure_detection import (
    CellDetector,
)
from cellfinder.core.tools.tools import get_max_possible_int_value


@pytest.mark.parametrize(
    "linear_size, datatype",
    [
        (254, np.uint16),  # results in < 2**16 structures, so not a problem
        (254, np.uint32),  # none of these are a problem with uint32
        (256, np.uint32),
        (258, np.uint32),
    ],
)
def test_connect_four_limits(
    linear_size: int, datatype: Type[np.number]
) -> None:
    """
    Test for `connect_four` with a rectangular plane (2-to-1 length ratio)
    containing a checkerboard of pixels marked as cells ("structures").
    The choice of the checkerboard pattern ensure the minimal number of pixels
    are used to execute this test (because all cell-pixels are disconnected).
    The parameters are designed to bridge the max unsigned 16-bit integer.
    When the maximum is reached, tests are expected to fail with uint16 input.

    We check that
    * the checkerboard has the expected number of structures
    * there is exactly one structure with id 1
    * there is exactly one structure with the maximum id...
    * ...and that structure is in the expected place (top-right pixel)
    """
    height = linear_size * 2
    width = linear_size
    # use a very large value - similar to how it is normally used
    soma_centre_value = get_max_possible_int_value(datatype)

    checkerboard = np.zeros((height, width), dtype=datatype)
    i = np.arange(height)[:, np.newaxis]  # rows
    j = np.arange(width)[np.newaxis, :]  # cols
    checkerboard[(i + j) % 2 == 0] = soma_centre_value

    actual_nonzeros = np.count_nonzero(checkerboard)
    expected_nonzeros = linear_size**2
    assert (
        actual_nonzeros == expected_nonzeros
    ), "Checkerboard didn't have the expected number of non-zeros"

    cell_detector = CellDetector(height, width, 0, soma_centre_value)
    labelled_plane = cell_detector.connect_four(checkerboard, None)
    one_count = np.count_nonzero(labelled_plane == 1)

    assert one_count == 1, "There was not exactly one pixel with label 1."
    assert (
        labelled_plane[height - 1, width - 1] == actual_nonzeros
    ), "The last labelled pixel did not have the maximum struct id."
    assert np.all(
        (labelled_plane != 0) == (checkerboard != 0)
    ), "Structures should be exactly where centers were marked"


@pytest.mark.parametrize("linear_size", [256, 258])
def test_connect_four_uint16_overflow(linear_size: int) -> None:
    datatype = np.uint16

    height = linear_size * 2
    width = linear_size
    soma_centre_value = get_max_possible_int_value(datatype)

    checkerboard = np.zeros((height, width), dtype=datatype)
    i = np.arange(height)[:, np.newaxis]
    j = np.arange(width)[np.newaxis, :]
    checkerboard[(i + j) % 2 == 0] = soma_centre_value

    cell_detector = CellDetector(height, width, 0, soma_centre_value)

    with pytest.raises(ValueError, match="overflow|label"):
        cell_detector.connect_four(checkerboard, None)
