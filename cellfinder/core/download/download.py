import os
import shutil
import tarfile
import urllib.request
from pathlib import Path

from brainglobe_utils.general.config import get_config_obj
from brainglobe_utils.general.system import disk_free_gb

from cellfinder.core.tools.source_files import (
    default_configuration_path,
    user_specific_configuration_path,
)


class DownloadError(Exception):
    pass


def download_file(destination_path, file_url, filename):
    direct_download = True
    file_url = file_url.format(int(direct_download))
    print(f"Downloading file: {filename}")
    with urllib.request.urlopen(file_url) as response:
        with open(destination_path, "wb") as outfile:
            shutil.copyfileobj(response, outfile)


def extract_file(tar_file_path, destination_path):
    tar = tarfile.open(tar_file_path)
    tar.extractall(path=destination_path)
    tar.close()


# TODO: check that intermediate folders exist
def download(
    download_path,
    url,
    file_name,
    install_path=None,
    download_requires=None,
    extract_requires=None,
):
    if not os.path.exists(os.path.dirname(download_path)):
        raise DownloadError(
            f"Could not find directory '{os.path.dirname(download_path)}' "
            f"to download file: {file_name}"
        )

    if (download_requires is not None) and (
        disk_free_gb(os.path.dirname(download_path)) < download_requires
    ):
        raise DownloadError(
            f"Insufficient disk space in {os.path.dirname(download_path)} to"
            f"download file: {file_name}"
        )

    if install_path is not None:
        if not os.path.exists(install_path):
            raise DownloadError(
                f"Could not find directory '{install_path}' "
                f"to extract file: {file_name}"
            )

        if (extract_requires is not None) and (
            disk_free_gb(install_path) < extract_requires
        ):
            raise DownloadError(
                f"Insufficient disk space in {install_path} to"
                f"extract file: {file_name}"
            )

    download_file(download_path, url, file_name)
    if install_path is not None:
        extract_file(download_path, install_path)
        os.remove(download_path)


def amend_user_configuration(new_model_path=None) -> None:
    """
    Amends the user configuration to contain the configuration
    in new_model_path, if specified.

    Parameters
    ----------
    new_model_path : str, optional
        The path to the new model configuration.
    """
    print("(Over-)writing custom user configuration")

    original_config = default_configuration_path()
    new_config = user_specific_configuration_path()
    if new_model_path is not None:
        write_model_to_config(new_model_path, original_config, new_config)


def write_model_to_config(new_model_path, orig_config, custom_config):
    """
    Update the model path in the custom configuration file, by
    reading the lines in the original configuration file, replacing
    the line starting with "model_path =" and writing these
    lines to the custom file.

    Parameters
    ----------
    new_model_path : str
        The new path to the model.
    orig_config : str
        The path to the original configuration file.
    custom_config : str
        The path to the custom configuration file to be created.

    Returns
    -------
    None

    """
    config_obj = get_config_obj(orig_config)
    model_conf = config_obj["model"]
    orig_path = model_conf["model_path"]

    with open(orig_config, "r") as in_conf:
        data = in_conf.readlines()
    for i, line in enumerate(data):
        data[i] = line.replace(
            f"model_path = '{orig_path}", f"model_path = '{new_model_path}"
        )

    custom_config_path = Path(custom_config)
    custom_config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(custom_config, "w") as out_conf:
        out_conf.writelines(data)
