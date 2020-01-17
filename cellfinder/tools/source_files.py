from pkg_resources import resource_filename
from amap.config.atlas import Atlas


def source_config():
    return resource_filename("cellfinder", "config/cellfinder.conf")


def source_custom_config():
    return resource_filename("cellfinder", "config/cellfinder.conf.custom")


def get_structures_path(config=None):
    if config is None:
        config = source_custom_config()
    atlas = Atlas(config)
    return atlas.get_structures_path()
