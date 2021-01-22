try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

# replace the asterisk with named imports
from .cellfinder_napari import napari_get_reader


__all__ = ["napari_get_reader"]
