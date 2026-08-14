import pytest

from cellfinder.core.download import cli, download


def test_model_registry_consistent():
    assert set(download.model_filenames) == set(download.model_hashes)
    assert set(download.model_filenames) == set(download.model_urls)
    assert set(download.model_filenames) == set(download.model_dimensions)
    assert set(download.model_filenames) == set(download.model_channels)


def test_single_channel_model_registered():
    assert "resnet50_1ch" in download.model_filenames
    assert "resnet50_1ch" in download.model_hashes
    assert download.model_filenames["resnet50_1ch"].endswith(".keras")
    assert "huggingface.co" in download.model_urls["resnet50_1ch"]


@pytest.mark.parametrize("model_name", ["resnet50_2d", "resnet50_2d_1ch"])
def test_2d_models_registered(model_name):
    assert download.model_filenames[model_name].endswith(".keras")
    assert "huggingface.co" in download.model_urls[model_name]
    assert download.model_hashes[model_name] is not None
    assert download.model_dimensions[model_name] == 2


def test_default_model_is_3d_two_channel():
    assert download.DEFAULT_MODEL in download.model_filenames
    assert download.model_dimensions[download.DEFAULT_MODEL] == 3
    assert download.model_channels[download.DEFAULT_MODEL] == 2


def test_models_for_dimensions_splits_registry():
    assert set(download.models_for_dimensions(2)) == {
        "resnet50_2d",
        "resnet50_2d_1ch",
    }
    assert download.DEFAULT_MODEL in download.models_for_dimensions(3)
    assert "resnet50_2d" not in download.models_for_dimensions(3)


@pytest.mark.parametrize(
    "dimensions,has_background,expected",
    [
        (3, True, "resnet50_tv"),
        (3, False, "resnet50_1ch"),
        (2, True, "resnet50_2d"),
        (2, False, "resnet50_2d_1ch"),
    ],
)
def test_default_model_matches_mode(dimensions, has_background, expected):
    assert download.default_model(dimensions, has_background) == expected


def test_default_model_without_a_registered_match():
    with pytest.raises(ValueError, match="No pretrained model"):
        download.default_model(4, True)


def test_validate_model_dimensions_rejects_mismatch():
    with pytest.raises(ValueError, match="is 2D, but dimensions=3"):
        download.validate_model_dimensions("resnet50_2d", 3)
    with pytest.raises(ValueError, match="is 3D, but dimensions=2"):
        download.validate_model_dimensions("resnet50_tv", 2)


def test_validate_model_dimensions_accepts_match():
    download.validate_model_dimensions("resnet50_2d", 2)
    download.validate_model_dimensions("resnet50_tv", 3)


def test_validate_model_dimensions_ignores_unknown_model():
    download.validate_model_dimensions("/path/to/custom.keras", 2)


def test_cli_main_downloads_requested_model(mocker):
    download_models = mocker.patch.object(cli, "download_models")
    mocker.patch(
        "sys.argv", ["cellfinder_download", "--model", "resnet50_1ch"]
    )

    cli.main()

    download_models.assert_called_once()
    assert download_models.call_args.args[0] == "resnet50_1ch"
