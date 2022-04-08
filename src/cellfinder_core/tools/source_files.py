from pkg_resources import resource_filename


def source_config_cellfinder():
    return resource_filename("cellfinder_core", "config/cellfinder.conf")


def source_custom_config_cellfinder():
    return resource_filename(
        "cellfinder_core", "config/cellfinder.conf.custom"
    )
