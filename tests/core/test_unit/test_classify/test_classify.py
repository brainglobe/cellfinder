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


def test_promote_2d_arrays_adds_z_axis():
    """2D signal and background arrays gain a singleton z axis, and
    2-element voxel sizes gain a z entry."""
    signal, background, voxel_sizes, network_voxel_sizes = (
        classify._promote_2d_arrays(
            np.zeros((40, 40), dtype=np.uint16),
            np.zeros((40, 40), dtype=np.uint16),
            (1, 1),
            (1, 1),
        )
    )

    assert signal.shape == (1, 40, 40)
    assert background.shape == (1, 40, 40)
    assert voxel_sizes == (1.0, 1, 1)
    assert network_voxel_sizes == (1.0, 1, 1)


def test_promote_2d_arrays_matches_network_z_to_data():
    """3D inputs pass through unchanged, but the network z voxel size is
    matched to the data so the depth-1 cube is not rescaled in z."""
    signal, background, voxel_sizes, network_voxel_sizes = (
        classify._promote_2d_arrays(
            np.zeros((10, 40, 40), dtype=np.uint16),
            None,
            (5, 1, 1),
            (2, 1, 1),
        )
    )

    assert signal.shape == (10, 40, 40)
    assert background is None
    assert voxel_sizes == (5, 1, 1)
    assert network_voxel_sizes == (5, 1, 1)


@pytest.mark.parametrize(
    "signal_ndim,background_ndim,match",
    [
        (4, 2, "signal data"),
        (2, 4, "background data"),
    ],
)
def test_promote_2d_arrays_bad_ndim_raises(
    signal_ndim, background_ndim, match
):
    """Arrays that are neither 2D nor 3D are rejected."""
    with pytest.raises(IOError, match=match):
        classify._promote_2d_arrays(
            np.zeros((4,) * signal_ndim, dtype=np.uint16),
            np.zeros((4,) * background_ndim, dtype=np.uint16),
            (1, 1),
            (1, 1),
        )


def test_squeeze_depth1_batch():
    """A depth-1 batch loses its z axis; a deeper one is rejected."""
    assert classify._squeeze_depth1_batch(np.zeros((2, 1, 8, 8)), 1).shape == (
        2,
        8,
        8,
    )

    with pytest.raises(ValueError, match="depth-1 cube, but got depth 3"):
        classify._squeeze_depth1_batch(np.zeros((2, 3, 8, 8)), 1)


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
