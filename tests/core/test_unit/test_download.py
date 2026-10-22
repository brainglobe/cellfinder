from pathlib import Path

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
    assert download.PRETRAINED_MODELS["resnet50_1ch"].hf_repo_id == (
        "brainglobe/cellfinder_single_channel_default"
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


def test_download_models_huggingface(tmp_path, mocker):
    mock_hf_download = mocker.patch(
        "cellfinder.core.download.download.hf_hub_download",
        return_value=str(tmp_path / "resnet50_single_channel.keras"),
    )

    result = download.download_models("resnet50_1ch", tmp_path)

    mock_hf_download.assert_called_once_with(
        repo_id="brainglobe/cellfinder_single_channel_default",
        filename="resnet50_single_channel.keras",
        cache_dir=Path(tmp_path),
        local_files_only=False,
    )
    assert result == tmp_path / "resnet50_single_channel.keras"


def test_download_models_huggingface_local_files_only(tmp_path, mocker):
    mock_hf_download = mocker.patch(
        "cellfinder.core.download.download.hf_hub_download",
        return_value=str(tmp_path / "resnet50_single_channel.keras"),
    )

    download.download_models("resnet50_1ch", tmp_path, local_files_only=True)

    mock_hf_download.assert_called_once_with(
        repo_id="brainglobe/cellfinder_single_channel_default",
        filename="resnet50_single_channel.keras",
        cache_dir=Path(tmp_path),
        local_files_only=True,
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
