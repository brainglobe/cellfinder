import numpy as np
import pytest
from napari.layers import Image, Points

from cellfinder_napari import sample_data
from cellfinder_napari.curation import CurationWidget


@pytest.fixture
def curation_widget(make_napari_viewer):
    """
    Create a viewer, add the curation widget, and return the widget.
    The viewer can be accessed using ``widget.viewer``.
    """
    viewer = make_napari_viewer()
    widget = CurationWidget(viewer)
    viewer.window.add_dock_widget(widget)
    return widget


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

    assert (tmp_path / "training.yml").exists()
    # Check that two .tif files are saved for both cells and non_cells
    assert len(list((tmp_path / "non_cells").glob("*.tif"))) == 2
    assert len(list((tmp_path / "cells").glob("*.tif"))) == 2
