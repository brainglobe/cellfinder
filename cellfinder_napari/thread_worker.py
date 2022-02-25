from cellfinder_core.main import main as cellfinder_run

from napari.qt.threading import thread_worker
from cellfinder_napari.input_containers import (
    DataInputs,
    DetectionInputs,
    ClassificationInputs,
    MiscInputs,
)


@thread_worker
def run(
    data_inputs: DataInputs,
    detection_inputs: DetectionInputs,
    classification_inputs: ClassificationInputs,
    misc_inputs: MiscInputs,
):
    """Runs cellfinder in a separate thread, to prevent GUI blocking."""
    points = cellfinder_run(
        **data_inputs.as_core_arguments(),
        **detection_inputs.as_core_arguments(),
        **classification_inputs.as_core_arguments(),
        **misc_inputs.as_core_arguments(),
    )
    return points
