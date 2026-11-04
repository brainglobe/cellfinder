from unittest.mock import patch

import numpy as np
import pytest

from cellfinder.core.classify import tools
from cellfinder.core.classify.resnet import build_model


def test_missing_weights():
    with pytest.raises(
        OSError,
        match="`model_weights` must be provided for inference "
        "or continued training.",
    ):
        tools.get_model(
            network_depth="101-layer",
            inference=True,
            model_weights=None,
        )


@patch("cellfinder.core.classify.tools.build_model")
def test_incorrect_weights(mock_build_model):
    mock_model = mock_build_model.return_value
    mock_model.load_weights.side_effect = ValueError(
        "Input 0 of layer 'resunit2_block0_bn_a' is incompatible"
        "with the layer: "
        "expected axis 3 of input shape to have value 6, but received"
        "input with shape "
        "(64, 12, 13, 7, 64)"
    )

    with pytest.raises(
        ValueError,
        match="Provided weights don't match the model architecture.",
    ):
        tools.get_model(
            network_depth="101-layer",
            inference=True,
            model_weights="incorrect_weights.h5",
        )


@pytest.mark.parametrize("num_channels", [1, 2])
def test_get_model_num_channels(num_channels):
    model = tools.get_model(
        network_depth="18-layer", num_channels=num_channels
    )
    assert tools.model_input_channels(model) == num_channels


@pytest.mark.parametrize("num_channels", [1, 2])
def test_get_model_loads_keras_weights(tmp_path, num_channels):
    """A ``.keras`` file loads into a freshly built model of matching
    channels and the weights actually transfer."""
    source = build_model(
        shape=(50, 50, 20, num_channels), network_depth="18-layer"
    )
    weights_path = tmp_path / "model.keras"
    source.save(weights_path)

    model = tools.get_model(
        network_depth="18-layer",
        inference=True,
        model_weights=weights_path,
        num_channels=num_channels,
    )

    assert tools.model_input_channels(model) == num_channels
    x = np.random.RandomState(0).rand(2, 50, 50, 20, num_channels)
    x = x.astype("float32")
    assert np.allclose(
        source.predict(x, verbose=0), model.predict(x, verbose=0)
    )


@pytest.mark.parametrize("num_channels", [1, 2])
def test_get_model_loads_weights_only(tmp_path, num_channels):
    """A `.weights.h5` file loads into a freshly built model of matching
    channels and the weights transfer without optimizer warnings."""
    source = build_model(
        shape=(50, 50, 20, num_channels), network_depth="18-layer"
    )
    weights_path = tmp_path / "model.weights.h5"
    source.save_weights(weights_path)

    model = tools.get_model(
        network_depth="18-layer",
        inference=True,
        model_weights=weights_path,
        num_channels=num_channels,
    )

    assert tools.model_input_channels(model) == num_channels
    x = np.random.RandomState(0).rand(2, 50, 50, 20, num_channels)
    x = x.astype("float32")
    assert np.allclose(
        source.predict(x, verbose=0), model.predict(x, verbose=0)
    )


def test_export_weights_only(tmp_path):
    """export_weights_only extracts weights from a saved model into a
    standalone .weights.h5 file."""
    source = build_model(shape=(50, 50, 20, 1), network_depth="18-layer")
    full_model_path = tmp_path / "full_model.keras"
    source.save(full_model_path)

    exported_weights_path = tmp_path / "exported.weights.h5"
    res = tools.export_weights_only(full_model_path, exported_weights_path)

    assert res == exported_weights_path
    assert exported_weights_path.exists()
    assert exported_weights_path.stat().st_size > 0

    # Verify that the extracted weights load cleanly into a new model
    target = tools.get_model(
        network_depth="18-layer",
        inference=True,
        model_weights=exported_weights_path,
        num_channels=1,
    )
    x = np.random.RandomState(42).rand(2, 50, 50, 20, 1).astype("float32")
    assert np.allclose(
        source.predict(x, verbose=0), target.predict(x, verbose=0)
    )


def test_export_weights_only_missing_source_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="Source model not found"):
        tools.export_weights_only(
            tmp_path / "non_existent.keras",
            tmp_path / "target.weights.h5",
        )


def test_set_model_weights_inference_skips_optimizer_build(mocker):
    """When inference=True, optimizer.build is skipped to avoid optimizer
    variable mismatch warnings."""
    mock_model = mocker.MagicMock()
    mock_model.optimizer = mocker.MagicMock()

    tools._set_model_weights(
        mock_model, "dummy.weights.h5", continue_training=False
    )
    mock_model.optimizer.build.assert_not_called()
    mock_model.load_weights.assert_called_once_with("dummy.weights.h5")

    tools._set_model_weights(
        mock_model, "dummy.weights.h5", continue_training=True
    )
    mock_model.optimizer.build.assert_called_once()


@pytest.mark.parametrize("shape", [None, (50, 50, 2)])
def test_get_model_builds_2d(shape):
    model = tools.get_model(
        network_depth="18-layer",
        dimensions=2,
        shape=shape,
    )
    # batch + y + x + channels
    assert len(model.input_shape) == 4
    assert model.input_shape == (None, 50, 50, 2)


def test_get_model_without_model_or_depth():
    with pytest.raises(
        ValueError,
        match="Either `existing_model` or `network_depth` must be provided.",
    ):
        tools.get_model(existing_model=None, network_depth=None)


def test_get_model_dimension_mismatch_on_load(tmp_path):
    model_path = tmp_path / "model_3d.keras"
    build_model(shape=(50, 50, 20, 2), network_depth="18-layer").save(
        model_path
    )

    with pytest.raises(ValueError, match="expects 3D input"):
        tools.get_model(existing_model=model_path, dimensions=2)
