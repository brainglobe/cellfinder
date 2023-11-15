import warnings

warnings.warn(
    "cellfinder (CLI) has migrated. Please use brainglobe-workflows instead: https://github.com/brainglobe/brainglobe-workflows",
    DeprecationWarning,
)

from importlib.metadata import metadata

__version__ = metadata("cellfinder")["version"]
__author__ = metadata("cellfinder")["author-email"]
__license__ = metadata("cellfinder")["license"]

del metadata
