import numpy as np

import cellfinder_core.tools.geometry as geometry


def test_make_sphere():
    pass


def test_four_connected_kernel():
    assert (
        (np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.bool))
        == geometry.four_connected_kernel()
    ).all()
