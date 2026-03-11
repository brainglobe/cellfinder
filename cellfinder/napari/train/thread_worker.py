from magicgui.widgets import ProgressBar
from napari.qt.threading import WorkerBase, WorkerBaseSignals
from qtpy.QtCore import Signal

from cellfinder.core.train.train_yaml import run as train_yaml_run

from .train_containers import (
    MiscTrainingInputs,
    OptionalNetworkInputs,
    OptionalTrainingInputs,
    TrainingDataInputs,
)


class MyTrainingWorkerSignals(WorkerBaseSignals):
    """
    Signals used by the TrainingWorker class below.
    """

    # Emits (label, max, value) for the progress bar
    update_progress_bar = Signal(str, int, int)


class TrainingWorker(WorkerBase):
    """
    Runs cellfinder training in a separate thread, to prevent GUI blocking.

    Also handles callbacks between the worker thread and main napari GUI
    thread to update a progress bar.
    """

    def __init__(
        self,
        training_data_inputs: TrainingDataInputs,
        optional_network_inputs: OptionalNetworkInputs,
        optional_training_inputs: OptionalTrainingInputs,
        misc_training_inputs: MiscTrainingInputs,
    ):
        super().__init__(SignalsClass=MyTrainingWorkerSignals)
        self.training_data_inputs = training_data_inputs
        self.optional_network_inputs = optional_network_inputs
        self.optional_training_inputs = optional_training_inputs
        self.misc_training_inputs = misc_training_inputs

    def connect_progress_bar_callback(self, progress_bar: ProgressBar):
        """
        Connects the progress bar to the worker so that updates are shown
        on the bar.
        """

        def update_progress_bar(label: str, max: int, value: int):
            progress_bar.label = label
            progress_bar.max = max
            progress_bar.value = value

        self.update_progress_bar.connect(update_progress_bar)

    def work(self) -> None:
        self.update_progress_bar.emit("Preparing training...", 1, 0)

        def progress_callback(label: str, value: int, max_val: int) -> None:
            self.update_progress_bar.emit(label, max_val, value)

        train_yaml_run(
            **self.training_data_inputs.as_core_arguments(),
            **self.optional_network_inputs.as_core_arguments(),
            **self.optional_training_inputs.as_core_arguments(),
            **self.misc_training_inputs.as_core_arguments(),
            progress_callback=progress_callback,
        )
        self.update_progress_bar.emit("Training finished!", 1, 1)
