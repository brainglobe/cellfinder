import sys
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
    assert len(get_training_widget.yaml_files.value) == 1
    assert get_training_widget.yaml_files.value[0] == Path.home()
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


def test_run_with_virtual_yaml_files(get_training_widget):
    """
    Checks that training is run with expected set of parameters.
    """
    with patch("cellfinder.napari.train.train.run_training") as run_training:
        # make default input valid - need yaml files (they don't technically
        # have to exist)
        virtual_yaml_files = (
            Path.home() / "file_1.yaml",
            Path.home() / "file_2.yaml",
        )
        get_training_widget.yaml_files.value = virtual_yaml_files
        get_training_widget.call_button.clicked()

        # create expected arguments for run
        expected_training_args = TrainingDataInputs()
        expected_network_args = OptionalNetworkInputs()
        expected_optional_training_args = OptionalTrainingInputs()
        expected_misc_args = MiscTrainingInputs()
        # run_training calls lr_schedule with empty list instead of tuple,
        # so to do equality comparison, we need to set default to list also
        expected_optional_training_args.lr_schedule = []

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


@pytest.mark.skipif(
    sys.version_info < (3, 13),
    reason="requires ThreadingMock, only available in Python 3.13+",
)
def test_args_properly_set(get_training_widget):
    """
    Checks that training is run with parameters from GUI.
    """
    from unittest.mock import ThreadingMock

    with patch(
        "cellfinder.napari.train.train.train_yaml", new_callable=ThreadingMock
    ) as train_yaml:
        # make default input valid - need yaml files (they don't technically
        # have to exist)
        virtual_yaml_files = (
            Path.home() / "file_1.yaml",
            Path.home() / "file_2.yaml",
        )
        get_training_widget.yaml_files.value = virtual_yaml_files
        get_training_widget.epochs.value = 77
        get_training_widget.batch_size.value = 27
        get_training_widget.test_fraction.value = 0.21
        get_training_widget.learning_rate.value = 0.0023
        get_training_widget.lr_schedule.value = [12, 37, 57]
        get_training_widget.lr_multiplier.value = 0.61
        get_training_widget.augment_likelihood.value = 0.83
        get_training_widget.flippable_axis.value = [0, 2]
        get_training_widget.rotate_range.value = 33, 145
        get_training_widget.translate_range.value = -0.33, 0.47
        get_training_widget.scale_range.value = 0.67, 1.37
        get_training_widget.intensity_range.value = 0.49, 1.83
        get_training_widget.number_of_free_cpus.value = 7
        get_training_widget.call_button.clicked()

        train_yaml.wait_until_called(timeout=60)
        train_yaml.assert_called_once()
        called_kwargs = train_yaml.call_args.kwargs

        assert list(called_kwargs["yaml_file"]) == list(virtual_yaml_files)
        assert called_kwargs["epochs"] == 77
        assert called_kwargs["batch_size"] == 27
        assert called_kwargs["test_fraction"] == 0.21
        assert called_kwargs["learning_rate"] == 0.0023
        assert list(called_kwargs["lr_schedule"]) == [12, 37, 57]
        assert called_kwargs["lr_multiplier"] == 0.61
        assert called_kwargs["augment_likelihood"] == 0.83
        assert set(called_kwargs["flippable_axis"]) == {0, 2}
        assert (
            list(called_kwargs["rotate_range"])
            == [
                (33, 145),
            ]
            * 3
        )
        assert (
            list(called_kwargs["translate_range"])
            == [
                (-0.33, 0.47),
            ]
            * 3
        )
        assert (
            list(called_kwargs["scale_range"])
            == [
                (0.67, 1.37),
            ]
            * 3
        )
        assert list(called_kwargs["intensity_range"]) == [0.49, 1.83]
        assert called_kwargs["n_free_cpus"] == 7
