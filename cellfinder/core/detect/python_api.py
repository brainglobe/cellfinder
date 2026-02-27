from typing import Callable, List, Optional

from brainglobe_utils.cells.cells import Cell

from cellfinder.core import types
from cellfinder.core.detect.detect import main as _detect_main
from cellfinder.core.tools.tools import inference_wrapper


@inference_wrapper
def main(
    signal_array: types.array,
    *args,
    callback: Optional[Callable[[int], None]] = None,
    **kwargs,
) -> List[Cell]:
    """
    Public Python API entry point for cell detection.

    This is a thin wrapper around `cellfinder.core.detect.detect.main`
    to provide a stable import path for external users and napari.
    """

    if signal_array is None:
        raise ValueError("signal_array cannot be None")

    return _detect_main(
        signal_array=signal_array,
        *args,
        callback=callback,
        **kwargs,
    )
