from importlib.metadata import metadata

__version__ = metadata("cellfinder")["version"]
__author__ = metadata("cellfinder")["author-email"]
__license__ = metadata("cellfinder")["license"]

del metadata
