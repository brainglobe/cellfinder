from pathlib import Path
from unittest.mock import patch

import pytest

from cellfinder_napari.train.train import training_widget
from cellfinder_napari.train.train_containers import (
    MiscTrainingInputs,
    OptionalNetworkInputs,
    OptionalTrainingInputs,
    TrainingDataInputs,
)


@pytest.fixture
def get_training_widget(make_napari_viewer):
    viewer = make_napari_viewer()
    widget = training_widget()
    _, widget = viewer.window.add_plugin_dock_widget(
        plugin_name="cellfinder-napari", widget_name="Train network"
    )
    viewer.window.add_dock_widget(widget)
    return widget


def test_reset_to_defaults(get_training_widget):
    """
    A simple test for the reset button.
    Checks widgets of a few different types are reset as expected.
    """
    # change a few widgets to non-default values
    get_training_widget.yaml_files.value = ["file_1.yml", "file_2.yml"]
    get_training_widget.continue_training.value = True
    get_training_widget.epochs.value = 50
    get_training_widget.test_fraction.value = 0.20

    # click reset button
    get_training_widget.reset_button.clicked()

    # check values have been reset
    assert len(get_training_widget.yaml_files.value) == 1
    assert get_training_widget.yaml_files.value[0] == Path.home()
    assert not get_training_widget.continue_training.value
    assert get_training_widget.epochs.value == 100
    assert get_training_widget.test_fraction.value == 0.10


def test_run_with_no_yaml_files(get_training_widget):
    """
    Checks whether expected info message will be shown to user if they don't specify YAML file(s).
    """
    with patch("cellfinder_napari.train.train.show_info") as show_info:
        get_training_widget.call_button.clicked()
        show_info.assert_called_once_with(
            "Please select a YAML file for training"
        )


def test_run_with_virtual_yaml_files(get_training_widget):
    """
    Checks that training is run with expected set of parameters.
    """
    with patch("cellfinder_napari.train.train.run_training") as run_training:
        # make default input valid - need yml files (they don't technically have to exist)
        virtual_yaml_files = (
            Path.home() / "file_1.yml",
            Path.home() / "file_2.yml",
        )
        get_training_widget.yaml_files.value = virtual_yaml_files
        get_training_widget.call_button.clicked()

        # create expected arguments for run
        expected_training_args = TrainingDataInputs()
        expected_network_args = OptionalNetworkInputs()
        expected_optional_training_args = OptionalTrainingInputs()
        expected_misc_args = MiscTrainingInputs()

        # we expect the widget to make some changes to the defaults
        # displayed before calling the training backend
        expected_training_args.yaml_files = virtual_yaml_files
        expected_network_args.trained_model = None
        expected_network_args.model_weights = None

        run_training.assert_called_once_with(
            expected_training_args,
            expected_network_args,
            expected_optional_training_args,
            expected_misc_args,
        )
