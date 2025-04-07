import pytest

from cellfinder.napari.detect.detect_containers import (
    ClassificationInputs,
    DataInputs,
    DetectionInputs,
    MiscInputs,
)
from cellfinder.napari.detect.thread_worker import Worker
from cellfinder.napari.sample_data import load_sample


def prepare_test(skip_classification=False, skip_detection=False):
    '''prepare worker instance to test'''
    data = load_sample()
    signal = data[0][0]
    background = data[1][0]

    worker = Worker(
        DataInputs(signal_array=signal, background_array=background),
        DetectionInputs(skip_detection=skip_detection),
        ClassificationInputs(
            skip_classification=skip_classification, trained_model=None
        ),
        MiscInputs(start_plane=0, end_plane=1),
    )
    return worker


def record_signal(emitted_signals, *args):
    '''record all emited signals after triggering work'''
    emitted_signals.append(args)


def check_emitted_signals(
    emitted_signals, expected_signals, non_expected_signals
):
    '''check for correctnes of the collected signals'''
    emitted_strings = [signal[0] for signal in emitted_signals]

    # Assert that emitted signals match what you expect
    assert (
        emitted_strings == expected_signals
    ), f"Expected {expected_signals}, but got {emitted_strings}"

    # Assert that none of the non-expected signals were emitted
    for non_signal in non_expected_signals:
        assert (
            non_signal not in emitted_strings
        ), f"Unexpected signal emitted: {non_signal}"


expected_signals_classification = [
    "Setting up classification...",
    "Finished classification",
]
expected_signals_detetction = ["Setting up detection...", "Finished detection"]


@pytest.mark.parametrize(
    "skip_classification,skip_detection, "
    "expected_signals, non_expected_signals",
    [
        (
            True,
            False,
            expected_signals_detetction,
            expected_signals_classification,
        ),
        (
            False,
            True,
            expected_signals_classification,
            expected_signals_detetction,
        ),
    ],
)
def test_signal_emission(
    skip_classification, skip_detection, expected_signals, non_expected_signals
):
    """Test the signals emitted when skipping classification or detection."""
    worker = prepare_test(
        skip_classification=skip_classification, skip_detection=skip_detection
    )

    emitted_signals = []
    worker.update_progress_bar.connect(
        lambda *args: record_signal(emitted_signals, *args)
    )
    worker.work()

    check_emitted_signals(
        emitted_signals, expected_signals, non_expected_signals
    )
