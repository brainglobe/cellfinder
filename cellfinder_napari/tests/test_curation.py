import pytest
from napari.layers import Points

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
