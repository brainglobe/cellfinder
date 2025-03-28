from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from magicgui.widgets import ProgressBar

from cellfinder.napari.train.train import TrainingWorker, training_widget
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


@pytest.mark.xfail(reason="See discussion in #443", raises=AssertionError)
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


@pytest.mark.xfail(reason="See discussion in #443", raises=AssertionError)
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


@pytest.mark.xfail(reason="See discussion in #443", raises=AssertionError)
def test_run_with_virtual_yaml_files(get_training_widget):
    """
    Checks that training is run with expected set of parameters.
    """
    with patch(
        "cellfinder.napari.train.train.TrainingWorker"
    ) as mock_training_worker:
        mock_worker_instance = mock_training_worker.return_value
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

        # we expect the widget to make some changes to the defaults
        # displayed before calling the training backend
        expected_training_args.yaml_files = virtual_yaml_files
        expected_network_args.trained_model = None
        expected_network_args.model_weights = None

        mock_training_worker.assert_called_once_with(
            expected_training_args,
            expected_network_args,
            expected_optional_training_args,
            expected_misc_args,
        )

        mock_worker_instance.start.assert_called_once()


@pytest.fixture
def training_worker_inputs():
    """Fixture to provide standard inputs for TrainingWorker tests."""
    return {
        "training_inputs": TrainingDataInputs(),
        "network_inputs": OptionalNetworkInputs(),
        "training_options": OptionalTrainingInputs(),
        "misc_inputs": MiscTrainingInputs(),
    }


def test_training_worker(training_worker_inputs):
    """Test TrainingWorker initialization and progress bar callback."""

    worker = TrainingWorker(
        training_worker_inputs["training_inputs"],
        training_worker_inputs["network_inputs"],
        training_worker_inputs["training_options"],
        training_worker_inputs["misc_inputs"],
    )

    assert (
        worker.training_data_inputs
        == training_worker_inputs["training_inputs"]
    )
    assert (
        worker.optional_network_inputs
        == training_worker_inputs["network_inputs"]
    )
    assert (
        worker.optional_training_inputs
        == training_worker_inputs["training_options"]
    )
    assert worker.misc_training_inputs == training_worker_inputs["misc_inputs"]

    progress_bar = ProgressBar()
    worker.connect_progress_bar_callback(progress_bar)

    worker.signals.update_progress.emit("Test", 10, 5)
    assert progress_bar.label == "Test"
    assert progress_bar.max == 10
    assert progress_bar.value == 5


@patch("cellfinder.napari.train.train.train_yaml")
def test_training_worker_execution(mock_train_yaml, training_worker_inputs):
    """Test the training worker execution and callbacks."""

    inputs = training_worker_inputs
    inputs["training_inputs"].as_core_arguments = MagicMock(
        return_value={"yaml_file": ["test.yaml"]}
    )
    inputs["network_inputs"].as_core_arguments = MagicMock(
        return_value={"model": "test_model"}
    )
    inputs["training_options"].as_core_arguments = MagicMock(
        return_value={"epochs": 5}
    )
    inputs["misc_inputs"].as_core_arguments = MagicMock(
        return_value={"n_free_cpus": 1}
    )

    worker = TrainingWorker(
        inputs["training_inputs"],
        inputs["network_inputs"],
        inputs["training_options"],
        inputs["misc_inputs"],
    )
    worker.signals.update_progress = MagicMock()

    worker.work()

    worker.signals.update_progress.emit.assert_any_call(
        "Starting training...", 1, 0
    )
    worker.signals.update_progress.emit.assert_any_call(
        "Training complete", 1, 1
    )

    mock_train_yaml.assert_called_once()
    assert "epoch_callback" in mock_train_yaml.call_args.kwargs

    callback = mock_train_yaml.call_args.kwargs["epoch_callback"]
    callback(3, 5)
    # epoch 3 means 2 completed epochs
    worker.signals.update_progress.emit.assert_any_call(
        "Training epoch 3/5", 5, 2
    )
