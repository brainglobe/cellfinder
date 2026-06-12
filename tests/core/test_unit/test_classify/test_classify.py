from unittest.mock import MagicMock

import numpy as np
import pytest
from brainglobe_utils.cells.cells import Cell

from cellfinder.core.classify import classify


def test_classify_channel_mismatch_raises(synthetic_single_spot, mocker):
    """A model whose channel count differs from the data raises clearly."""
    signal_array, _background, c_xyz = synthetic_single_spot
    signal_array = signal_array.astype(np.uint16)
    points = [Cell(tuple(int(c) for c in c_xyz), Cell.UNKNOWN)]

    # a two-channel model fed single-channel (background=None) data
    fake_model = MagicMock()
    fake_model.inputs = [MagicMock(shape=(None, 50, 50, 20, 2))]
    mocker.patch(
        "cellfinder.core.classify.classify.get_model",
        return_value=fake_model,
    )

    with pytest.raises(ValueError, match="expects 2-channel input but 1"):
        classify.main(
            points,
            signal_array,
            None,
            0,
            (5, 1, 1),
            (5, 1, 1),
            1,
            50,
            50,
            20,
            None,
            None,
            "50",
        )
