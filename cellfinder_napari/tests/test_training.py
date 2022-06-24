from pathlib import Path

import pytest

from cellfinder_napari.train import train


@pytest.fixture
def training_widget(make_napari_viewer):
    viewer = make_napari_viewer()
    widget = train()
    viewer.window.add_dock_widget(widget)
    return widget


def test_add_training_widget(training_widget):
    """
    Smoke test to check that adding training widget works
    """
    widget = training_widget
    assert widget is not None


def test_reset_to_defaults(training_widget):
    # change a few widgets to non-default values
    training_widget.yaml_files.value = ["file_1.yml", "file_2.yml"]
    training_widget.continue_training.value = True
    training_widget.epochs.value = 50
    training_widget.test_fraction.value = 0.20

    # click reset button
    training_widget.reset_button.clicked()

    # check values have been reset
    assert len(training_widget.yaml_files.value) == 1
    assert training_widget.yaml_files.value[0] == Path.home()
    assert not training_widget.continue_training.value
    assert training_widget.epochs.value == 100
    assert training_widget.test_fraction.value == 0.10
