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
