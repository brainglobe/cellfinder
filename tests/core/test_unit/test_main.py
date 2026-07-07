from unittest.mock import patch

import numpy as np
import pytest

from cellfinder.core.main import main


@pytest.fixture
def signal_array():
    return np.empty((5, 50, 50), dtype=np.uint16)


@pytest.fixture
def background_array():
    return np.empty((5, 50, 50), dtype=np.uint16)


@patch("cellfinder.core.detect.detect.main")
def test_invalid_trained_model_fails_before_detection(
    mock_detect, signal_array, background_array
):
    with pytest.raises(FileNotFoundError, match="Trained model not found"):
        main(
            signal_array=signal_array,
            background_array=background_array,
            voxel_sizes=(5, 2, 2),
            trained_model="/nonexistent/model.keras",
        )

    mock_detect.assert_not_called()


@patch("cellfinder.core.detect.detect.main")
def test_invalid_model_weights_fails_before_detection(
    mock_detect, signal_array, background_array, tmp_path
):
    bad_weights = tmp_path / "nonexistent_weights.h5"

    with pytest.raises(FileNotFoundError, match="Model weights not found"):
        main(
            signal_array=signal_array,
            background_array=background_array,
            voxel_sizes=(5, 2, 2),
            model_weights=bad_weights,
        )

    mock_detect.assert_not_called()


@patch("cellfinder.core.detect.detect.main", return_value=[])
@patch("cellfinder.core.tools.prep.prep_model_weights")
def test_valid_weights_allows_detection(
    mock_prep_weights, mock_detect, signal_array, background_array
):
    mock_prep_weights.return_value = "/some/weights.h5"

    main(
        signal_array=signal_array,
        background_array=background_array,
        voxel_sizes=(5, 2, 2),
    )

    mock_prep_weights.assert_called_once()
    mock_detect.assert_called_once()


@patch("cellfinder.core.detect.detect.main", return_value=[])
@patch("cellfinder.core.tools.prep.prep_model_weights")
def test_optional_background_allows_detection(
    mock_prep_weights, mock_detect, signal_array
):
    mock_prep_weights.return_value = "/some/weights.h5"

    main(
        signal_array=signal_array,
        background_array=None,
        voxel_sizes=(5, 2, 2),
    )

    mock_detect.assert_called_once()


@patch("cellfinder.core.detect.detect.main", return_value=[])
@patch("cellfinder.core.tools.prep.prep_model_weights")
def test_missing_background_selects_single_channel_model(
    mock_prep_weights, mock_detect, signal_array
):
    mock_prep_weights.return_value = "/some/weights.keras"

    main(
        signal_array=signal_array,
        background_array=None,
        voxel_sizes=(5, 2, 2),
    )

    assert mock_prep_weights.call_args.args[2] == "resnet50_1ch"


@patch("cellfinder.core.detect.detect.main", return_value=[])
@patch("cellfinder.core.tools.prep.prep_model_weights")
def test_explicit_model_is_kept_without_background(
    mock_prep_weights, mock_detect, signal_array
):
    mock_prep_weights.return_value = "/some/weights.h5"

    main(
        signal_array=signal_array,
        background_array=None,
        voxel_sizes=(5, 2, 2),
        model="resnet50_all",
    )

    assert mock_prep_weights.call_args.args[2] == "resnet50_all"


@patch("cellfinder.core.classify.classify.main", return_value=[])
@patch("cellfinder.core.detect.detect.main", return_value=[1])
@patch("cellfinder.core.tools.prep.prep_model_weights")
def test_dimensions_threaded_to_detect_and_classify(
    mock_prep_weights,
    mock_detect,
    mock_classify,
    signal_array,
    background_array,
):
    mock_prep_weights.return_value = "/some/weights.h5"

    main(
        signal_array=signal_array,
        background_array=background_array,
        voxel_sizes=(5, 2, 2),
        dimensions=2,
    )

    assert mock_detect.call_args.kwargs["dimensions"] == 2
    assert mock_classify.call_args.kwargs["dimensions"] == 2


def test_main_bad_dimensions(signal_array, background_array):
    with pytest.raises(ValueError, match="dimensions must be 2 or 3"):
        main(
            signal_array=signal_array,
            background_array=background_array,
            voxel_sizes=(5, 2, 2),
            dimensions=4,
            skip_classification=True,
        )
