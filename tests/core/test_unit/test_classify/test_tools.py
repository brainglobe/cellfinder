from unittest.mock import patch

import pytest
import tools


@pytest.mark.parametrize(
    "model_weights, inference, expected_exception, expected_message",
    [
        (
            None,
            True,
            OSError,
            "`model_weights` must be provided for inference",
        ),
    ],
)
def test_missing_weights(
    model_weights, inference, expected_exception, expected_message
):
    with pytest.raises(expected_exception, match=expected_message):
        tools.get_model(
            network_depth="shallow",
            inference=inference,
            model_weights=model_weights,
        )


@patch("tools.build_model")
def test_incorrect_weights(mock_build_model):
    mock_model = mock_build_model.return_value
    mock_model.load_weights.side_effect = ValueError(
        "Provided weights don't match the model architecture."
    )

    with pytest.raises(
        ValueError,
        match="Provided weights don't match the model architecture.",
    ):
        tools.get_model(
            network_depth="shallow",
            inference=True,
            model_weights="incorrect_weights.h5",
        )
