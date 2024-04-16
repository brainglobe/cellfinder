import numpy as np
import pytest

from cellfinder.core.detect.filters.volume.structure_detection import (
    CellDetector,
)


@pytest.mark.parametrize(
    "linear_size, datatype",
    [
        (254, np.uint16),  # results in < 2**16 structures, so not a problem
        pytest.param(
            256, np.uint16, marks=pytest.mark.xfail
        ),  # results in 2**16 structures, so last structure gets labelled 0
        pytest.param(
            258, np.uint16, marks=pytest.mark.xfail
        ),  # results in > 2**16 structures, so two structures with label 1
        (254, np.uint32),  # none of these are a problem with uint32
        (256, np.uint32),
        (258, np.uint32),
    ],
)
def test_connect_four_limits(linear_size, datatype):
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
    SOMA_CENTRE_VALUE = np.iinfo(datatype).max
    checkerboard = np.zeros((linear_size * 2, linear_size), dtype=datatype)
    for i in range(linear_size * 2):
        for j in range(linear_size):
            if (i + j) % 2 == 0:
                checkerboard[i, j] = SOMA_CENTRE_VALUE

    actual_nonzeros = np.count_nonzero(checkerboard)
    expected_nonzeros = linear_size**2
    assert (
        actual_nonzeros == expected_nonzeros
    ), "Checkerboard didn't have the expected number of non-zeros"

    cell_detector = CellDetector(linear_size * 2, linear_size, 0)
    labelled_plane = cell_detector.connect_four(checkerboard, None)
    one_count = np.count_nonzero(labelled_plane == 1)

    assert one_count == 1, "There was not exactly one pixel with label 1."
    assert (
        labelled_plane[linear_size * 2 - 1, linear_size - 1] == actual_nonzeros
    ), "The last labelled pixel did not have the maximum struct id."
