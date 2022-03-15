import glob
import pathlib

import napari
import numpy as np
import pytest
from PIL import Image

from cellfinder_napari.detect import detect

test_data_dir = pathlib.Path(__file__) / ".." / ".." / ".." / "test_data"


def load_stack(directory: pathlib.Path) -> np.ndarray:
    """
    Load a stack of .tif image files in the given directory.
    """
    tif_glob = str((directory / "*.tif").resolve())
    ims = []
    for f in sorted(glob.glob(tif_glob)):
        ims.append(np.array(Image.open(f)))
    return np.stack(ims, axis=0)


# @pytest.fixture
# def signal_image():
#    return load_stack(test_data_dir / "crop_planes" / "ch0")


# @pytest.fixture
# def background_image():
#    return load_stack(test_data_dir / "crop_planes" / "ch1")


def test_detect_function(
    make_napari_viewer,
):  # , signal_image, background_image):
    # Smoke test that running cell detection from the widget works

    # Create viewer and add images
    viewer = make_napari_viewer()
    # signal_layer = viewer.add_image(signal_image, name='signal')
    # background_layer = viewer.add_image(background_image, name='background')

    # Create widget
    widget = detect()
    viewer.window.add_dock_widget(widget)

    # Select images, and run detection
    widget.reset_choices()
    # widget.signal_image.value = signal_layer
    # widget.background_image.value = background_layer

    # TODO: work out how to run detection here whilst blocking execution.
    # Currently calling widget() will start detection, but pytest will
    # pass immediately because new processes are spawned to do the detection.
    # Perhaps use https://pytest-qt.readthedocs.io/en/latest/signals.html ?
