from pathlib import Path

from cellfinder import DEFAULT_CELLFINDER_DIRECTORY


def default_configuration_path():
    """
    Returns the default configuration path for cellfinder.

    Returns:
        Path: The default configuration path.
    """
    return Path(__file__).parent.parent / "config" / "cellfinder.conf"


def user_specific_configuration_path():
    """
    Returns the path to the user-specific configuration file for cellfinder.

    This function returns the path to the user-specific configuration file
    for cellfinder. The user-specific configuration file is located in the
    user's home directory under the ".brainglobe/cellfinder" folder
    and is named "cellfinder.conf.custom".

    Returns:
        Path: The path to the custom configuration file.

    """
    return DEFAULT_CELLFINDER_DIRECTORY / "cellfinder.conf.custom"
