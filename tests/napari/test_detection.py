from unittest.mock import patch

import napari
import pytest

from cellfinder.napari.detect import detect_widget
from cellfinder.napari.detect.detect_containers import (
    ClassificationInputs,
    DataInputs,
    DetectionInputs,
    MiscInputs,
)
from cellfinder.napari.detect.thread_worker import Worker
from cellfinder.napari.sample_data import load_sample


@pytest.fixture
def get_detect_widget(make_napari_viewer):
    viewer = make_napari_viewer()
    widget = detect_widget()
    for layer in load_sample():
        viewer.add_layer(napari.layers.Image(layer[0], **layer[1]))
    _, widget = viewer.window.add_plugin_dock_widget(
        plugin_name="cellfinder", widget_name="Cell detection"
    )
    return widget


def test_detect_worker():
    """
    Smoke test to check that the detection worker runs
    """
    data = load_sample()
    signal = data[0][0]
    background = data[1][0]

    worker = Worker(
        DataInputs(signal_array=signal, background_array=background),
        DetectionInputs(),
        ClassificationInputs(trained_model=None),
        MiscInputs(start_plane=0, end_plane=1),
    )
    worker.work()


def test_data_inputs_without_background():
    """
    A missing background array is passed through to core as None.
    """
    signal = load_sample()[0][0]
    core_arguments = DataInputs(
        signal_array=signal, background_array=None
    ).as_core_arguments()
    assert core_arguments["background_array"] is None
    assert core_arguments["signal_array"] is signal


def test_detect_worker_without_background():
    """
    Smoke test to check that the detection worker runs single-channel.
    """
    signal = load_sample()[0][0]

    worker = Worker(
        DataInputs(signal_array=signal, background_array=None),
        DetectionInputs(),
        ClassificationInputs(trained_model=None),
        MiscInputs(start_plane=0, end_plane=1),
    )
    worker.work()


@pytest.mark.parametrize(
    argnames="analyse_local",
    argvalues=[True, False],  # increase test coverage by covering both cases
)
def test_run_detect(get_detect_widget, analyse_local):
    """
    Test backend is called
    """
    with patch("cellfinder.napari.detect.detect.Worker") as worker:
        get_detect_widget.analyse_local.value = analyse_local
        get_detect_widget.skip_classification.value = True
        get_detect_widget.call_button.clicked()
        assert worker.called


def test_run_detect_without_inputs():
    """ """
    with patch("cellfinder.napari.detect.detect.show_info") as show_info:
        widget = (
            detect_widget()
        )  # won't have image layers, so should notice and show info
        widget.call_button.clicked()
        assert show_info.called


def test_run_detect_without_background_needs_single_channel_model(
    get_detect_widget,
):
    """
    Classifying without a background and with the default pretrained
    (two-channel) weights is rejected with a helpful message, and the
    worker never starts.
    """
    widget = get_detect_widget
    with (
        patch("cellfinder.napari.detect.detect.show_info") as show_info,
        patch("cellfinder.napari.detect.detect.Worker") as worker,
    ):
        widget.background_image_opt.background_image.value = None
        widget.skip_classification.value = False
        widget.use_pre_trained_weights.value = True
        widget.call_button.clicked()

        show_info.assert_called_once()
        assert "single-channel model" in show_info.call_args.args[0]
        assert not worker.called


def test_reset_defaults(get_detect_widget):
    """Smoke test that restore defaults doesn't error."""
    get_detect_widget.reset_button.clicked()
