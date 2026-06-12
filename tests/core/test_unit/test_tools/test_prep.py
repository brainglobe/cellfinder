from cellfinder.core.download import download as dl
from cellfinder.core.tools import prep


def test_prep_models_redownloads_on_model_change(tmp_path, mocker):
    """Switching ``model_name`` re-resolves weights instead of reusing the
    cached config from a previous, different model."""
    config_file = tmp_path / "cellfinder.conf.custom"

    downloaded = []

    def fake_download(model_name, install_path):
        downloaded.append(model_name)
        path = tmp_path / dl.model_filenames[model_name]
        path.write_text("weights")
        return path

    mocker.patch.object(
        prep, "user_specific_configuration_path", return_value=config_file
    )
    mocker.patch(
        "cellfinder.core.download.download."
        "user_specific_configuration_path",
        return_value=config_file,
    )
    mocker.patch.object(prep, "download_models", side_effect=fake_download)

    first = prep.prep_model_weights(None, tmp_path, "resnet50_tv")
    assert first.name == dl.model_filenames["resnet50_tv"]

    second = prep.prep_model_weights(None, tmp_path, "resnet50_1ch")
    assert second.name == dl.model_filenames["resnet50_1ch"]

    # the second request must trigger its own download, not reuse the cache
    third = prep.prep_model_weights(None, tmp_path, "resnet50_1ch")
    assert third.name == dl.model_filenames["resnet50_1ch"]

    assert downloaded == ["resnet50_tv", "resnet50_1ch"]
