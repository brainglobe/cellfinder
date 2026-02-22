from unittest.mock import patch

import pytest

from cellfinder.core.classify import tools


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


@patch("cellfinder.core.classify.tools.build_model")
@patch("cellfinder.core.classify.tools.keras.models.load_model")
def test_get_model_existing_takes_precedence(
    mock_load_model, mock_build_model
):
    """Test that existing_model takes precedence over network_depth."""
    tools.get_model(existing_model="/some/path", network_depth="18-layer")
    mock_load_model.assert_called_once_with("/some/path")
    mock_build_model.assert_not_called()


@patch("cellfinder.core.classify.tools.build_model")
def test_get_model_builds_with_depth_only(mock_build_model):
    """network_depth alone should build and return a new model."""
    tools.get_model(network_depth="18-layer")
    mock_build_model.assert_called_once_with(
        network_depth="18-layer",
        learning_rate=0.0001,
    )


def test_get_model_no_model_no_depth():
    """Both existing_model and network_depth as None should raise."""
    with pytest.raises(
        ValueError,
        match="Either `existing_model` or `network_depth` must be provided.",
    ):
        tools.get_model(existing_model=None, network_depth=None)
