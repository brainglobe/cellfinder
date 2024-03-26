from pathlib import Path


def default_configuration_path():
    """
    Returns the default configuration path for cellfinder.

    Returns:
        Path: The default configuration path.
    """
    return Path(__file__).parent.parent / "config" / "cellfinder.conf"


def custom_configuration_path():
    """
    Returns the path to the custom configuration file for cellfinder.

    This function returns the path to the custom configuration file
    for cellfinder. The custom configuration file is located in the
    user's home directory under the ".cellfinder" folder and is named
    "cellfinder.conf.custom".

    Returns:
        Path: The path to the custom configuration file.

    """
    return Path.home() / ".cellfinder" / "cellfinder.conf.custom"
