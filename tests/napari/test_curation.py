from pathlib import Path
from unittest.mock import patch

import napari
import numpy as np
import pytest
import yaml
from napari.layers import Image, Points

from cellfinder.napari import sample_data
from cellfinder.napari.curation import CurationWidget
from cellfinder.napari.sample_data import load_sample


@pytest.fixture
def curation_widget(make_napari_viewer) -> CurationWidget:
    """
    Create a viewer, add the curation widget, and return the widget.
    The viewer can be accessed using ``widget.viewer``.
    """
    viewer = make_napari_viewer()
    _, widget = viewer.window.add_plugin_dock_widget(
        plugin_name="cellfinder", widget_name="Curation"
    )
    return widget


@pytest.mark.xfail(reason="See discussion in #443", raises=AssertionError)
def test_add_new_training_layers(curation_widget):
    viewer = curation_widget.viewer
    layers = viewer.layers
    # Check that layers list starts off empty
    assert len(layers) == 0
    curation_widget.add_training_data()
    assert len(layers) == 2

    assert all(isinstance(layer, Points) for layer in layers)

    assert layers[0].name == "Training data (cells)"
    assert layers[1].name == "Training data (non cells)"


def test_update_voxel_size(curation_widget: CurationWidget):
    assert curation_widget.voxel_sizes == [5, 2, 2]
    curation_widget.voxel_sizes_boxes[0].setValue(3)
    curation_widget.voxel_sizes_boxes[1].setValue(4)
    curation_widget.voxel_sizes_boxes[2].setValue(5)
    assert curation_widget.voxel_sizes == [3, 4, 5]


def test_update_normalization_down_sampling(curation_widget: CurationWidget):
    assert curation_widget.normalization_down_sampling == 32
    curation_widget.norm_sampling_box.setValue(8)
    assert curation_widget.normalization_down_sampling == 8


@pytest.mark.xfail(reason="See discussion in #443", raises=AssertionError)
def test_cell_marking(curation_widget, tmp_path):
    """
    Check that marking cells and non-cells works as expected.
    """
    widget = curation_widget
    widget.add_training_data()
    viewer = widget.viewer

    cell_layer = widget.training_data_cell_layer
    non_cell_layer = widget.training_data_non_cell_layer

    # Check that no cells have been marked yet
    assert all(
        layer.data.shape == (0, 3) for layer in [cell_layer, non_cell_layer]
    )

    # Add a points layer to select points from
    points = Points(
        np.array([[16, 17, 18], [13, 14, 15]]), name="selection_points"
    )
    # Adding the layer automatically selects it in the layer list
    viewer.add_layer(points)

    # Select the first point, and add as a cell
    points.selected_data = [0]
    curation_widget.mark_as_cell()
    assert np.array_equal(cell_layer.data, np.array([[16, 17, 18]]))
    assert non_cell_layer.data.shape[0] == 0

    # Select the second point, and add as a non-cell
    points.selected_data = [1]
    curation_widget.mark_as_non_cell()
    assert np.array_equal(cell_layer.data, np.array([[16, 17, 18]]))
    assert np.array_equal(non_cell_layer.data, np.array([[13, 14, 15]]))

    # Add signal/background images to the viewer and widget
    layer_data = sample_data.load_sample()
    signal = Image(layer_data[0][0], **layer_data[0][1])
    background = Image(layer_data[1][0], **layer_data[1][1])
    widget.signal_layer = signal
    widget.background_layer = background

    widget.output_directory = tmp_path
    widget.save_training_data(prompt_for_directory=False, block=True)

    assert (tmp_path / "training.yaml").exists()
    # Check that two .tif files are saved for both cells and non_cells
    assert len(list((tmp_path / "non_cells").glob("*.tif"))) == 2
    assert len(list((tmp_path / "cells").glob("*.tif"))) == 2

    with open(tmp_path / "training.yaml", "r") as fh:
        yaml_data = yaml.safe_load(fh)

    for item in yaml_data["data"]:
        assert "cube_dir" in item
        assert "signal_channel" in item
        assert "bg_channel" in item
        assert "type" in item
        assert "signal_mean" in item
        assert "signal_std" in item
        assert "bg_mean" in item
        assert "bg_std" in item


@pytest.fixture
def valid_curation_widget(make_napari_viewer) -> CurationWidget:
    """
    Setup up a valid curation widget,
    complete with training data layers and points,
    and signal+background images.
    """
    viewer = make_napari_viewer()
    image_layers = load_sample()
    for layer in image_layers:
        viewer.add_layer(napari.layers.Image(layer[0], **layer[1]))

    num_dw = len(viewer.window._dock_widgets)
    _, curation_widget = viewer.window.add_plugin_dock_widget(
        plugin_name="cellfinder", widget_name="Curation"
    )
    assert len(viewer.window._dock_widgets) == num_dw + 1

    curation_widget.add_training_data()
    # Add a points layer to select points from
    points = Points(
        np.array([[16, 17, 18], [13, 14, 15]]), name="selection_points"
    )
    # Adding the layer automatically selects it in the layer list
    viewer.add_layer(points)

    # Select the first point, and add as a cell
    points.selected_data = [0]
    curation_widget.mark_as_cell()

    # Select the second point, and add as a non-cell
    points.selected_data = [1]
    curation_widget.mark_as_non_cell()

    curation_widget.signal_image_choice.setCurrentText("Signal")
    curation_widget.background_image_choice.setCurrentText("Background")
    curation_widget.set_signal_image()
    curation_widget.set_background_image()
    return curation_widget


@pytest.mark.xfail(reason="See discussion in #443", raises=AssertionError)
def test_check_image_data_for_extraction(valid_curation_widget):
    """
    Check valid curation widget has extractable data.
    """
    assert valid_curation_widget.check_image_data_for_extraction()


@pytest.mark.xfail(reason="See discussion in #443", raises=AssertionError)
def test_check_image_data_wrong_shape(valid_curation_widget):
    """
    Check curation widget shows expected user message if images don't have
    identical shape.
    """
    with patch("cellfinder.napari.curation.show_info") as show_info:
        signal_layer_with_wrong_shape = napari.layers.Image(
            np.zeros(shape=(1, 1)), name="Wrong shape"
        )
        valid_curation_widget.viewer.add_layer(signal_layer_with_wrong_shape)
        valid_curation_widget.signal_image_choice.setCurrentText("Wrong shape")
        valid_curation_widget.set_signal_image()
        valid_curation_widget.check_image_data_for_extraction()
        show_info.assert_called_once_with(
            "Please ensure both signal and background images are the "
            "same size and shape."
        )


@pytest.mark.xfail(reason="See discussion in #443", raises=AssertionError)
def test_check_image_data_missing_signal(valid_curation_widget):
    """
    Check curation widget shows expected user message if signal image is
    missing.
    """
    with patch("cellfinder.napari.curation.show_info") as show_info:
        valid_curation_widget.signal_layer = None
        valid_curation_widget.check_image_data_for_extraction()
        show_info.assert_called_once_with(
            "Please ensure both signal and background images are loaded "
            "into napari, and selected in the sidebar. "
        )


@pytest.mark.xfail(reason="See discussion in #443", raises=AssertionError)
def test_is_data_extractable(curation_widget, valid_curation_widget):
    """Check is_data_extractable works as expected."""
    assert not curation_widget.is_data_extractable()
    assert valid_curation_widget.is_data_extractable()


@pytest.mark.xfail(reason="See discussion in #443", raises=AssertionError)
def test_get_output_directory(valid_curation_widget):
    """Check get_output_directory returns expected value."""
    with patch(
        "cellfinder.napari.curation.QFileDialog.getExistingDirectory"
    ) as get_directory:
        get_directory.return_value = ""
        valid_curation_widget.get_output_directory()
        assert valid_curation_widget.output_directory is None

        get_directory.return_value = Path.home()
        valid_curation_widget.get_output_directory()
        assert valid_curation_widget.output_directory == Path.home()


@pytest.mark.xfail(reason="See discussion in #443", raises=AssertionError)
def test_check_layer_removal_sync(valid_curation_widget):
    """
    Check that removing a layer from the viewer also removes it from the
    widget.
    """
    viewer = valid_curation_widget.viewer
    viewer.layers.select_all()
    viewer.layers.remove_selected()
    assert valid_curation_widget.signal_layer is None
    assert valid_curation_widget.background_layer is None
    assert valid_curation_widget.training_data_cell_layer is None
    assert valid_curation_widget.training_data_non_cell_layer is None
