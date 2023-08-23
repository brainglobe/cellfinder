import numpy as np

import cellfinder.core.tools.geometry as geometry


def test_make_sphere():
    pass


def test_four_connected_kernel():
    np.testing.assert_array_equal(
        geometry.four_connected_kernel(),
        np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    )
