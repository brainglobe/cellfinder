from pathlib import Path


def source_config_cellfinder():
    return Path(__file__).parent.parent / "config" / "cellfinder.conf"


def source_custom_config_cellfinder():
    return Path(__file__).parent.parent / "config" / "cellfinder.conf.custom"
