import pytest


@pytest.fixture()
def custom_logger_name() -> str:
    """Return name of custom logger created in workflow utils

    Returns
    -------
    str
        Name of custom logger
    """
    from cellfinder.workflows.utils import __name__ as logger_name

    return logger_name
