from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("cellfinder-core")
except PackageNotFoundError as e:
    raise PackageNotFoundError("cellfinder-core package not installed") from e

__author__ = "Adam Tyson, Christian Niedworok, Charly Rousseau"
__license__ = "BSD-3-Clause"

import logging

logger = logging.getLogger("cellfinder_core")
