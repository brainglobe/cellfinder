from unittest.mock import MagicMock

import numpy as np
import pytest
from brainglobe_utils.cells.cells import Cell

from cellfinder.core.classify import classify
from cellfinder.core.classify.resnet import build_model


def test_classify_single_channel_runs(synthetic_single_spot, tmp_path):
    """Single-channel data classifies end to end with a 1-channel model."""
    signal_array, _background, c_xyz = synthetic_single_spot
    signal_array = signal_array.astype(np.uint16)
    points = [Cell(tuple(int(c) for c in c_xyz), Cell.UNKNOWN)]

    model = build_model(shape=(50, 50, 20, 1), network_depth="18-layer")
    model_path = tmp_path / "model_1ch.keras"
    model.save(model_path)

    result = classify.main(
        points=points,
        signal_array=signal_array,
        background_array=None,
        n_free_cpus=0,
        voxel_sizes=(5, 1, 1),
        network_voxel_sizes=(5, 1, 1),
        batch_size=1,
        cube_height=50,
        cube_width=50,
        cube_depth=20,
        trained_model=model_path,
        model_weights=None,
        network_depth="18",
    )

    assert len(result) == len(points)
    assert all(cell.type in (Cell.CELL, Cell.NO_CELL) for cell in result)


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
