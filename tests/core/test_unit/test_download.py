from cellfinder.core.download import download


def test_model_registry_consistent():
    assert set(download.model_filenames) == set(download.model_hashes)
    assert set(download.model_filenames) == set(download.model_urls)


def test_single_channel_model_registered():
    assert "resnet50_1ch" in download.model_filenames
    assert "resnet50_1ch" in download.model_hashes
    assert download.model_filenames["resnet50_1ch"].endswith(".keras")
    assert "huggingface.co" in download.model_urls["resnet50_1ch"]
