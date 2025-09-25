from pytestqt.qtbot import QtBot

from cellfinder.napari.detect.detect_containers import (
    ClassificationInputs,
    DataInputs,
    DetectionInputs,
    MiscInputs,
)
from cellfinder.napari.detect.thread_worker import Worker
from cellfinder.napari.sample_data import load_sample


def run_worker_test(
    qtbot: QtBot,
    skip_detection: bool,
    skip_classification: bool,
    expected_labels: list,
):
    data = load_sample()
    signal = data[1][0]
    background = data[0][0]

    worker = Worker(
        DataInputs(signal_array=signal, background_array=background),
        DetectionInputs(skip_detection=skip_detection),
        ClassificationInputs(
            skip_classification=skip_classification, trained_model=None
        ),
        MiscInputs(start_plane=0, end_plane=-1),
    )

    emitted = []
    worker.update_progress_bar.connect(
        lambda *args: (print("Signal:", args), emitted.append(args))
    )

    with qtbot.waitSignal(worker.finished, timeout=300000):  # 5 minutes
        worker.start()

    emitted_labels = [e[0] for e in emitted]

    for label in expected_labels:
        assert label in emitted_labels, f"{label} not emitted"


def test_signals_detection_and_classification(qtbot: QtBot):
    """Test with both detection and classification enabled."""
    expected = [
        "Setting up detection...",
        "Detecting cells",
        "Setting up classification...",
        # "Classifying cells", this is
        # commented because in this sample example zero cells are detected
        "Finished classification",
    ]
    run_worker_test(
        qtbot,
        skip_detection=False,
        skip_classification=False,
        expected_labels=expected,
    )


def test_signals_detection_only(qtbot: QtBot):
    """Test with classification skipped, detection enabled."""
    expected = [
        "Setting up detection...",
        "Detecting cells",
        "Finished detection",
    ]
    run_worker_test(
        qtbot,
        skip_detection=False,
        skip_classification=True,
        expected_labels=expected,
    )
