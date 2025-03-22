from pathlib import Path
from typing import Optional

from magicgui import magicgui
from magicgui.widgets import FunctionGui, ProgressBar, PushButton
from napari.qt.threading import WorkerBase, WorkerBaseSignals
from napari.utils.notifications import show_info
from qtpy.QtCore import Signal
from qtpy.QtWidgets import QScrollArea

from cellfinder.core.train.train_yaml import run as train_yaml
from cellfinder.napari.utils import cellfinder_header, html_label_widget

from .train_containers import (
    MiscTrainingInputs,
    OptionalNetworkInputs,
    OptionalTrainingInputs,
    TrainingDataInputs,
)


class MyWorkerSignals(WorkerBaseSignals):
    """
    Signals used by the Worker class (label, max, value).
    """

    update_progress = Signal(str, int, int)


class TrainingWorker(WorkerBase):
    """
    Worker that runs training in a separate thread and updates progress bar.
    """

    def __init__(
        self,
        training_data_inputs: TrainingDataInputs,
        optional_network_inputs: OptionalNetworkInputs,
        optional_training_inputs: OptionalTrainingInputs,
        misc_training_inputs: MiscTrainingInputs,
    ):
        super().__init__(SignalsClass=MyWorkerSignals)
        self.training_data_inputs = training_data_inputs
        self.optional_network_inputs = optional_network_inputs
        self.optional_training_inputs = optional_training_inputs
        self.misc_training_inputs = misc_training_inputs

    def connect_progress_bar_callback(self, progress_bar: ProgressBar):
        """Connects the progress bar to the worker."""

        def update_progress_bar(label: str, max: int, value: int):
            progress_bar.label = label
            progress_bar.max = max
            progress_bar.value = value

        self.signals.update_progress.connect(update_progress_bar)

    def work(self) -> None:
        """Execute the training with progress updates."""
        self.signals.update_progress.emit("Starting training...", 1, 0)

        # callbacks for training progress
        def epoch_callback(epoch: int, total_epochs: int):
            self.signals.update_progress.emit(
                f"Training epoch {epoch}/{total_epochs}", total_epochs, epoch
            )

        # Run the training with the callback
        train_yaml(
            **self.training_data_inputs.as_core_arguments(),
            **self.optional_network_inputs.as_core_arguments(),
            **self.optional_training_inputs.as_core_arguments(),
            **self.misc_training_inputs.as_core_arguments(),
            epoch_callback=epoch_callback,
        )

        self.signals.update_progress.emit("Training complete", 1, 1)


def training_widget() -> FunctionGui:
    progress_bar = ProgressBar()

    @magicgui(
        training_label=html_label_widget("Network training", tag="h3"),
        **TrainingDataInputs.widget_representation(),
        **OptionalNetworkInputs.widget_representation(),
        **OptionalTrainingInputs.widget_representation(),
        **MiscTrainingInputs.widget_representation(),
        call_button=True,
        reset_button=dict(widget_type="PushButton", text="Reset defaults"),
        scrollable=True,
    )
    def widget(
        training_label: dict,
        data_options: dict,
        yaml_files: Path,
        output_directory: Path,
        network_options: dict,
        trained_model: Optional[Path],
        model_weights: Optional[Path],
        model_depth: str,
        pretrained_model: str,
        training_options: dict,
        continue_training: bool,
        augment: bool,
        tensorboard: bool,
        save_checkpoints: bool,
        save_progress: bool,
        epochs: int,
        learning_rate: float,
        batch_size: int,
        test_fraction: float,
        misc_options: dict,
        number_of_free_cpus: int,
        reset_button: PushButton,
    ):
        """
        Parameters
        ----------
        yaml_files : Path
            YAML files containing paths to training data
        output_directory : Path
            Directory to save the output trained model
        trained_model : Optional[Path]
            Existing pre-trained model
        model_weights : Optional[Path]
            Existing pre-trained model weights
            Should be set along with "Model depth"
        model_depth : str
            ResNet model depth (as per He et al. (2015))
        pretrained_model : str
            Which pre-trained model to use
            (Supplied with cellfinder)
        continue_training : bool
            Continue training from an existing trained model
            If no trained model or model weights are specified,
            this will continue from the pretrained model
        augment : bool
            Augment the training data to improve generalisation
        tensorboard : bool
            Log to output_directory/tensorboard
        save_checkpoints : bool
            Store the model at intermediate points during training
        save_progress : bool
            Save training progress to a .csv file
        epochs : int
            Number of training epochs
            (How many times to use each training data point)
        learning_rate : float
            Learning rate for training the model
        batch_size : int
            Training batch size
        test_fraction : float
            Fraction of training data to use for validation
        number_of_free_cpus : int
            How many CPU cores to leave free
        reset_button : PushButton
            Reset parameters to default
        """
        trained_model = None if trained_model == Path.home() else trained_model
        model_weights = None if model_weights == Path.home() else model_weights

        training_data_inputs = TrainingDataInputs(yaml_files, output_directory)

        optional_network_inputs = OptionalNetworkInputs(
            trained_model,
            model_weights,
            model_depth,
            pretrained_model,
        )

        optional_training_inputs = OptionalTrainingInputs(
            continue_training,
            augment,
            tensorboard,
            save_checkpoints,
            save_progress,
            epochs,
            learning_rate,
            batch_size,
            test_fraction,
        )

        misc_training_inputs = MiscTrainingInputs(number_of_free_cpus)

        if yaml_files[0] == Path.home():  # type: ignore
            show_info("Please select a YAML file for training")
        else:
            worker = TrainingWorker(
                training_data_inputs,
                optional_network_inputs,
                optional_training_inputs,
                misc_training_inputs,
            )
            worker.connect_progress_bar_callback(progress_bar)
            worker.errored.connect(
                lambda e: show_info(f"Error during training: {str(e)}")
            )
            worker.start()

    widget.native.layout().insertWidget(0, cellfinder_header())

    @widget.reset_button.changed.connect
    def restore_defaults():
        defaults = {
            **TrainingDataInputs.defaults(),
            **OptionalNetworkInputs.defaults(),
            **OptionalTrainingInputs.defaults(),
            **MiscTrainingInputs.defaults(),
        }
        for name, value in defaults.items():
            # ignore fields with no default
            if value is not None:
                getattr(widget, name).value = value

    scroll = QScrollArea()
    scroll.setWidget(widget._widget._qwidget)
    widget._widget._qwidget = scroll
    widget.insert(widget.index("number_of_free_cpus") + 1, progress_bar)

    return widget
