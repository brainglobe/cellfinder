from unittest.mock import patch
import pytest
from cellfinder.core.classify import tools  # << Change this import

@pytest.mark.parametrize(
    "model_weights, inference, expected_exception, expected_message",
    [
        (
            None,
            True,
            OSError,
            "`model_weights` must be provided for inference or continued training.",
        ),
    ],
)
def test_missing_weights(
    model_weights, inference, expected_exception, expected_message
):
    with pytest.raises(expected_exception, match=expected_message):
        tools.get_model(
            network_depth="101-layer",
            inference=inference,
            model_weights=model_weights,
        )


@patch("cellfinder.core.classify.tools.build_model")   # << Change here too
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
            network_depth="101-layer",
            inference=True,
            model_weights="incorrect_weights.h5",
        )
