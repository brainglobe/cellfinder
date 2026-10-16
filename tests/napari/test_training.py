from pathlib import Path
from unittest.mock import patch

import pytest

from cellfinder.napari.train.train import training_widget
from cellfinder.napari.train.train_containers import (
    MiscTrainingInputs,
    OptionalNetworkInputs,
    OptionalTrainingInputs,
    TrainingDataInputs,
)


@pytest.fixture
def make_dummy_yaml(tmp_path):
    """Returns a factory for empty YAML files in a temporary directory."""

    def _make_dummy_yaml(index: int) -> Path:
        yaml_file = tmp_path / f"file_{index}.yaml"
        yaml_file.write_text("[]\n")
        return yaml_file

    return _make_dummy_yaml


@pytest.fixture
def get_training_widget(make_napari_viewer):
    viewer = make_napari_viewer()
    widget = training_widget()
    _, widget = viewer.window.add_plugin_dock_widget(
        plugin_name="cellfinder", widget_name="Train network"
    )
    viewer.window.add_dock_widget(widget)
    return widget


def test_reset_to_defaults(get_training_widget):
    """
    A simple test for the reset button.
    Checks widgets of a few different types are reset as expected.
    """
    # change a few widgets to non-default values
    get_training_widget.yaml_files.value = ["file_1.yaml", "file_2.yaml"]
    get_training_widget.continue_training.value = True
    get_training_widget.epochs.value = 50
    get_training_widget.test_fraction.value = 0.20

    # click reset button
    get_training_widget.reset_button.clicked()

    # check values have been reset
    assert get_training_widget.yaml_files.value is None
    assert get_training_widget.output_directory.value is None
    assert get_training_widget.trained_model.value is None
    assert get_training_widget.model_weights.value is None
    assert not get_training_widget.continue_training.value
    assert get_training_widget.epochs.value == 100
    assert get_training_widget.test_fraction.value == 0.10


def test_run_with_no_yaml_files(get_training_widget):
    """
    Checks whether expected info message will be shown to user if they don't
    specify YAML file(s).
    """
    with patch("cellfinder.napari.train.train.show_info") as show_info:
        get_training_widget.call_button.clicked()
        show_info.assert_called_once_with(
            "Please select a YAML file for training"
        )


def test_run_with_no_output_directory(get_training_widget, make_dummy_yaml):
    """
    Checks the user is told to pick an output directory rather than the
    training silently writing to whatever the default happens to be.
    """
    get_training_widget.yaml_files.value = (make_dummy_yaml(1),)

    with (
        patch("cellfinder.napari.train.train.show_info") as show_info,
        patch("cellfinder.napari.train.train.run_training") as run_training,
    ):
        get_training_widget.call_button.clicked()

    show_info.assert_called_once_with("Please select an output directory")
    run_training.assert_not_called()


@pytest.mark.parametrize(
    "filename, expected_message",
    [
        ("not_yaml.txt", "Not a YAML file: not_yaml.txt"),
        ("missing.yaml", "YAML file does not exist: {path}"),
    ],
)
def test_run_with_invalid_yaml_files(
    get_training_widget, tmp_path, filename, expected_message
):
    """
    Checks YAML files are rejected if they are the wrong format or absent.
    """
    yaml_file = tmp_path / filename
    get_training_widget.yaml_files.value = (yaml_file,)
    get_training_widget.output_directory.value = tmp_path

    with (
        patch("cellfinder.napari.train.train.show_info") as show_info,
        patch("cellfinder.napari.train.train.run_training") as run_training,
    ):
        get_training_widget.call_button.clicked()

    show_info.assert_called_once_with(expected_message.format(path=yaml_file))
    run_training.assert_not_called()


def test_run_with_yaml_files(get_training_widget, tmp_path, make_dummy_yaml):
    """
    Checks that training is run with expected set of parameters.
    """
    output_directory = tmp_path / "output"
    output_directory.mkdir()
    yaml_files = tuple(make_dummy_yaml(index) for index in (1, 2))

    with patch("cellfinder.napari.train.train.run_training") as run_training:
        get_training_widget.yaml_files.value = yaml_files
        get_training_widget.output_directory.value = output_directory
        get_training_widget.call_button.clicked()

        # create expected arguments for run
        expected_training_args = TrainingDataInputs()
        expected_network_args = OptionalNetworkInputs()
        expected_optional_training_args = OptionalTrainingInputs()
        expected_misc_args = MiscTrainingInputs()
        # run_training calls lr_schedule with empty list instead of tuple,
        # so to do equality comparison, we need to set default to list also
        expected_optional_training_args.lr_schedule = []

        expected_training_args.yaml_files = yaml_files
        expected_training_args.output_directory = output_directory

        run_training.assert_called_once_with(
            expected_training_args,
            expected_network_args,
            expected_optional_training_args,
            expected_misc_args,
        )
