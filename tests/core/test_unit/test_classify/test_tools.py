from unittest.mock import patch

import pytest

from cellfinder.core.classify import tools
from cellfinder.core.classify.tools import build_model


def test_model_weights_none_raises_oserror():
    with pytest.raises(
        OSError,
        match="`model_weights` must be provided for inference or"
        "continued training.",
    ):
        build_model(
            input_shape=(64, 64, 64, 1),
            learning_rate=0.001,
            model_weights=None,
            inference=True,
        )


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
