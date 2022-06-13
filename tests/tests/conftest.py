import pytest

from cellfinder_core.download import models
from cellfinder_core.tools.prep import DEFAULT_INSTALL_PATH


# Check that the classification model is already downloaded
# at the beginning of a pytest session
@pytest.fixture(scope="session")
def download_default_model():
    models.main("resnet50_tv", DEFAULT_INSTALL_PATH)
