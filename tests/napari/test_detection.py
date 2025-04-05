from unittest.mock import patch
from unittest.mock import MagicMock
from cellfinder.napari.detect.detect import get_results_callback
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


@pytest.mark.xfail(reason="See discussion in #443", raises=AssertionError)
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


@pytest.mark.xfail(reason="See discussion in #443", raises=AssertionError)
def test_reset_defaults(get_detect_widget):
    """Smoke test that restore defaults doesn't error."""
    get_detect_widget.reset_button.clicked()

# NEW TESTS FOR EMPTY POINTS HANDLING
@pytest.fixture
def mock_viewer(make_napari_viewer):
    """Fixture to mock a napari viewer with empty layers."""
    viewer = make_napari_viewer()
    viewer.layers = MagicMock()  # Ensures no real layers interfere
    return viewer

def test_handle_empty_results_notification(mock_viewer):
    """Test that empty results trigger a user notification and log."""
    with patch("napari.utils.notifications.show_info") as mock_show_info, \
         patch("logging.Logger.info") as mock_logger:
        
        # Trigger empty points case
        callback = get_results_callback(skip_classification=False, viewer=mock_viewer)
        callback(points=[])
        
        # Verify notification content
        mock_show_info.assert_called_once_with(
            "No cells detected. Please try:\n"
            "- Adjusting detection thresholds\n"
            "- Changing soma diameter parameter\n"
            "- Verifying image quality"
        )
        # Verify log
        mock_logger.assert_called_once_with("Cell detection completed with no results")

def test_done_func_skips_layer_add_on_empty_points(mock_viewer):
    """Test that no layer is added when points are empty."""
    with patch("cellfinder.napari.detect.detect.add_single_layer") as mock_add_single, \
         patch("cellfinder.napari.detect.detect.add_classified_layers") as mock_add_classified:
        
        # Tests both skip_classification cases
        for skip in [True, False]:
            callback = get_results_callback(skip_classification=skip, viewer=mock_viewer)
            callback(points=[])
            
            # Ensures no layer was added
            mock_add_single.assert_not_called()
            mock_add_classified.assert_not_called()