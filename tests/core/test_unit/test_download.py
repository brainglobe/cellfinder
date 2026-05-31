from cellfinder.core.download import download


def test_model_registry_consistent():
    assert set(download.model_filenames) == set(download.model_hashes)


def test_single_channel_model_registered():
    assert "resnet50_1ch" in download.model_filenames
    assert "resnet50_1ch" in download.model_hashes
