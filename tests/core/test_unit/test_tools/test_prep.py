import pytest

from cellfinder.core.download import download as dl
from cellfinder.core.tools import prep


def test_prep_models_downloads_requested_model(tmp_path, mocker):
    """Each requested model name resolves to its own download."""
    downloaded = []

    def fake_download(model_name, install_path):
        downloaded.append(model_name)
        path = tmp_path / dl.model_filenames[model_name]
        path.write_text("weights")
        return path

    mocker.patch.object(prep, "download_models", side_effect=fake_download)

    first = prep.prep_model_weights(None, tmp_path, "resnet50_tv")
    second = prep.prep_model_weights(None, tmp_path, "resnet50_1ch")

    assert first.name == dl.model_filenames["resnet50_tv"]
    assert second.name == dl.model_filenames["resnet50_1ch"]
    assert downloaded == ["resnet50_tv", "resnet50_1ch"]


def test_prep_models_uses_provided_path(tmp_path, mocker):
    """An explicit weights path is used as-is, without downloading."""
    download = mocker.patch.object(prep, "download_models")
    weights = tmp_path / "weights.h5"
    weights.write_text("weights")

    result = prep.prep_model_weights(weights, tmp_path, "resnet50_tv")

    assert result == weights
    download.assert_not_called()


def test_prep_models_missing_provided_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="Model weights not found"):
        prep.prep_model_weights(
            tmp_path / "missing.h5", tmp_path, "resnet50_tv"
        )
