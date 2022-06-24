from typing import Iterable

from cellfinder_core.main import main as cellfinder_run
from napari.qt.threading import WorkerBase, WorkerBaseSignals, thread_worker
from qtpy.QtCore import QObject, Signal

from .detect_containers import (
    ClassificationInputs,
    DataInputs,
    DetectionInputs,
    MiscInputs,
)


class MyWorkerSignals(WorkerBaseSignals):
    """
    Signals used by the Worker class below.
    """

    # Emits (label, max, value) for the progress bar
    update_progress_bar = Signal(str, int, int)


class Worker(WorkerBase):
    """
    Runs cellfinder in a separate thread, to prevent GUI blocking.

    Also handles callbacks between the worker thread and main napari GUI thread
    to update a progress bar.
    """

    def __init__(
        self,
        data_inputs: DataInputs,
        detection_inputs: DetectionInputs,
        classification_inputs: ClassificationInputs,
        misc_inputs: MiscInputs,
    ):
        super().__init__(SignalsClass=MyWorkerSignals)
        self.data_inputs = data_inputs
        self.detection_inputs = detection_inputs
        self.classification_inputs = classification_inputs
        self.misc_inputs = misc_inputs

    def work(self) -> list:
        self.update_progress_bar.emit("Setting up detection...", 1, 0)

        def detect_callback(plane: int) -> None:
            self.update_progress_bar.emit(
                "Detecting cells",
                self.data_inputs.nplanes,
                plane + 1,
            )

        def detect_finished_callback(points: list) -> None:
            self.npoints_detected = len(points)
            self.update_progress_bar.emit("Setting up classification...", 1, 0)

        def classify_callback(batch: int) -> None:
            self.update_progress_bar.emit(
                "Classifying cells",
                # Default cellfinder-core batch size is 32. This seems to give
                # a slight underestimate of the number of batches though, so
                # allow for batch number to go over this
                max(self.npoints_detected // 32 + 1, batch + 1),
                batch + 1,
            )

        result = cellfinder_run(
            **self.data_inputs.as_core_arguments(),
            **self.detection_inputs.as_core_arguments(),
            **self.classification_inputs.as_core_arguments(),
            **self.misc_inputs.as_core_arguments(),
            detect_callback=detect_callback,
            classify_callback=classify_callback,
            detect_finished_callback=detect_finished_callback,
        )
        self.update_progress_bar.emit("Finished classification", 1, 1)
        return result
