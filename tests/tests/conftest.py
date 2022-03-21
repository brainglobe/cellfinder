import pathlib
import sys

import pytest
from cellfinder_core.download.cli import main as cellfinder_download
from imlib.general.config import get_config_obj

test_data_dir = pathlib.Path(__file__) / ".." / ".." / "data"
data_dir = test_data_dir / "brain"
test_output_dir = test_data_dir / "registration_output"

TEST_ATLAS = "allen_2017_100um"


def download_atlas(directory):
    download_args = [
        "cellfinder_download",
        "--atlas",
        TEST_ATLAS,
        "--install-path",
        directory,
        "--no-amend-config",
        "--no-models",
    ]
    sys.argv = download_args
    cellfinder_download()
    return directory


def generate_test_config(atlas_dir):
    config = test_data_dir / "config" / "test.conf"
    config_obj = get_config_obj(config)
    atlas_conf = config_obj["atlas"]
    orig_base_directory = atlas_conf["base_folder"]

    with open(config, "r") as in_conf:
        data = in_conf.readlines()
    for i, line in enumerate(data):
        data[i] = line.replace(
            f"base_folder = '{orig_base_directory}",
            f"base_folder = '{atlas_dir / 'atlas' / TEST_ATLAS}",
        )
    test_config = atlas_dir / "config.conf"
    with open(test_config, "w") as out_conf:
        out_conf.writelines(data)

    return test_config


@pytest.fixture()
def test_config_path(tmpdir):
    print("fixture")
    atlas_directory = str(tmpdir)
    download_atlas(atlas_directory)
    test_config = generate_test_config(atlas_directory)
    return test_config
