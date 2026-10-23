import pooch
import pytest

from cellfinder.core.download import cli, download


def test_model_registry_consistent():
    assert set(download.model_filenames) == set(download.model_hashes)
    assert set(download.model_filenames) == set(download.model_urls)
    assert set(download.model_filenames) == set(download.PRETRAINED_MODELS)
    assert set(download.model_filenames) == set(download.model_dimensions)
    assert set(download.model_filenames) == set(download.model_channels)


def test_single_channel_model_registered():
    assert "resnet50_1ch" in download.model_filenames
    assert "resnet50_1ch" in download.model_hashes
    assert download.model_filenames["resnet50_1ch"].endswith(".keras")
    assert "huggingface.co" in download.model_urls["resnet50_1ch"]
    assert "aa326ad87a43f5db6754897d9a8f491f98d862d7" in (
        download.model_urls["resnet50_1ch"]
    )
    assert download.PRETRAINED_MODELS["resnet50_1ch"].hf_repo_id == (
        "brainglobe/cellfinder_single_channel_default"
    )
    assert download.PRETRAINED_MODELS["resnet50_1ch"].hf_revision == (
        "aa326ad87a43f5db6754897d9a8f491f98d862d7"
    )


@pytest.mark.parametrize("model_name", ["resnet50_2d", "resnet50_2d_1ch"])
def test_2d_models_registered(model_name):
    assert download.model_filenames[model_name].endswith(".keras")
    assert "huggingface.co" in download.model_urls[model_name]
    assert download.model_hashes[model_name] is not None
    assert download.model_dimensions[model_name] == 2
    assert (
        download.PRETRAINED_MODELS[model_name].hf_repo_id
        == "brainglobe/cellfinder_2d_default"
    )
    assert (
        download.PRETRAINED_MODELS[model_name].hf_revision
        == "1bb0a3ead0e88209ffa9887690a2bdb51d04d904"
    )


def test_download_models_huggingface(tmp_path, mocker):
    mock_retrieve = mocker.patch(
        "pooch.retrieve",
        return_value=str(tmp_path / "resnet50_single_channel.keras"),
    )

    result = download.download_models("resnet50_1ch", tmp_path)

    mock_retrieve.assert_called_once()
    assert (
        mock_retrieve.call_args.kwargs["fname"]
        == "resnet50_single_channel.keras"
    )
    assert (
        "aa326ad87a43f5db6754897d9a8f491f98d862d7"
        in mock_retrieve.call_args.kwargs["url"]
    )
    downloader = mock_retrieve.call_args.kwargs["downloader"]
    assert isinstance(downloader, pooch.HTTPDownloader)
    assert result == tmp_path / "resnet50_single_channel.keras"


def test_download_models_huggingface_with_token(tmp_path, mocker, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_test_secret_token")
    mock_retrieve = mocker.patch(
        "pooch.retrieve",
        return_value=str(tmp_path / "resnet50_single_channel.keras"),
    )

    download.download_models("resnet50_1ch", tmp_path)

    mock_retrieve.assert_called_once()
    downloader = mock_retrieve.call_args.kwargs["downloader"]
    assert downloader.kwargs.get("headers", {}).get("Authorization") == (
        "Bearer hf_test_secret_token"
    )


def test_download_models_local_files_only_exists(tmp_path, mocker):
    mock_retrieve = mocker.patch("pooch.retrieve")
    local_file = tmp_path / "resnet50_single_channel.keras"
    local_file.write_text("dummy")

    result = download.download_models(
        "resnet50_1ch", tmp_path, local_files_only=True
    )

    mock_retrieve.assert_not_called()
    assert result == local_file


def test_download_models_local_files_only_missing_raises(tmp_path):
    with pytest.raises(
        FileNotFoundError, match="not found and local_files_only=True"
    ):
        download.download_models(
            "resnet50_1ch", tmp_path, local_files_only=True
        )


def test_download_models_gin(tmp_path, mocker):
    mock_retrieve = mocker.patch(
        "pooch.retrieve",
        return_value=str(tmp_path / "resnet50_tv.h5"),
    )

    result = download.download_models("resnet50_tv", tmp_path)

    mock_retrieve.assert_called_once()
    assert (
        mock_retrieve.call_args.kwargs["fname"]
        == download.model_filenames["resnet50_tv"]
    )
    assert result == tmp_path / "resnet50_tv.h5"


def test_download_models_unknown_raises(tmp_path):
    with pytest.raises(
        ValueError, match="Model 'non_existent_model' not recognized"
    ):
        download.download_models("non_existent_model", tmp_path)


def test_cli_main_downloads_requested_model(mocker):
    download_models = mocker.patch.object(cli, "download_models")
    mocker.patch(
        "sys.argv", ["cellfinder_download", "--model", "resnet50_1ch"]
    )

    cli.main()

    download_models.assert_called_once()
    assert download_models.call_args.args[0] == "resnet50_1ch"
