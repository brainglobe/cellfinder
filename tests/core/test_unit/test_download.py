from cellfinder.core.download import cli, download


def test_model_registry_consistent():
    assert set(download.model_filenames) == set(download.model_hashes)
    assert set(download.model_filenames) == set(download.model_urls)


def test_single_channel_model_registered():
    assert "resnet50_1ch" in download.model_filenames
    assert "resnet50_1ch" in download.model_hashes
    assert download.model_filenames["resnet50_1ch"].endswith(".keras")
    assert "huggingface.co" in download.model_urls["resnet50_1ch"]


def test_cli_main_downloads_requested_model(mocker):
    download_models = mocker.patch.object(cli, "download_models")
    mocker.patch(
        "sys.argv", ["cellfinder_download", "--model", "resnet50_1ch"]
    )

    cli.main()

    download_models.assert_called_once()
    assert download_models.call_args.args[0] == "resnet50_1ch"
