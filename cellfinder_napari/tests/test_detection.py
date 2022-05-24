import glob
import pathlib

import napari
import numpy as np
import pytest
from PIL import Image

from cellfinder_napari.detect import detect
from cellfinder_napari.detect.detect_containers import (
    ClassificationInputs,
    DataInputs,
    DetectionInputs,
    MiscInputs,
)
from cellfinder_napari.detect.thread_worker import Worker
from cellfinder_napari.sample_data import load_sample


def test_add_detect_widget(make_napari_viewer):
    """
    Smoke test to check that adding detection widget works
    """
    viewer = make_napari_viewer()
    widget = detect()
    viewer.window.add_dock_widget(widget)


def test_detect_worker():
    """
    Smoke test to check that the detection worker runs
    """
    data = load_sample()
    signal = data[0][0]
    background = data[1][0]

    worker = Worker(
        DataInputs(signal_array=signal, background_array=background),
        DetectionInputs(),
        ClassificationInputs(trained_model=None),
        MiscInputs(start_plane=0, end_plane=1),
    )
    worker.work()
