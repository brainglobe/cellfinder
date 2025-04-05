import numpy as np
import pytest
from unittest.mock import MagicMock
from brainglobe_utils.cells.cells import Cell

from cellfinder.napari.utils import (
    add_classified_layers,
    cells_to_array,
    html_label_widget,
    napari_array_to_cells,
    napari_points_axis_order,
)


@pytest.mark.xfail(reason="See discussion in #443", raises=AssertionError)
def test_add_classified_layers(make_napari_viewer):
    """Smoke test for add_classified_layers utility"""
    cell_pos = [1, 2, 3]
    unknown_pos = [4, 5, 6]
    points = [
        Cell(pos=cell_pos, cell_type=Cell.CELL),
        Cell(pos=unknown_pos, cell_type=Cell.UNKNOWN),
    ]
    viewer = make_napari_viewer()
    n_layers = len(viewer.layers)
    # adds a "detected" and a "rejected layer"
    add_classified_layers(
        points, viewer, unknown_name="rejected", cell_name="accepted"
    )
    assert len(viewer.layers) == n_layers + 2

    # check names match
    rej_layer = cell_layer = None
    for layer in reversed(viewer.layers):
        if layer.name == "accepted" and cell_layer is None:
            cell_layer = layer
        if layer.name == "rejected" and rej_layer is None:
            rej_layer = layer
    assert cell_layer is not None
    assert rej_layer is not None
    assert cell_layer.data is not None
    assert rej_layer.data is not None

    # check data added in correct column order
    # CELL types
    cell_data = np.array([cell_pos])
    assert np.all(
        cells_to_array(points, Cell.CELL, napari_order=False) == cell_data
    )
    # convert to napari order and check it is in napari
    cell_data = cell_data[:, napari_points_axis_order]
    assert np.all(cell_layer.data == cell_data)

    # UNKNOWN type
    rej_data = np.array([unknown_pos])
    assert np.all(
        cells_to_array(points, Cell.UNKNOWN, napari_order=False) == rej_data
    )
    # convert to napari order and check it is in napari
    rej_data = rej_data[:, napari_points_axis_order]
    assert np.all(rej_layer.data == rej_data)

    # get cells back from napari points
    cells_again = napari_array_to_cells(cell_layer.data, cell_type=Cell.CELL)
    cells_again.extend(
        napari_array_to_cells(rej_layer.data, cell_type=Cell.UNKNOWN)
    )
    assert cells_again == points

def test_add_single_layer(make_napari_viewer):
    """Test adding a single layer with cells."""
    viewer = make_napari_viewer()
    points = [Cell((1, 2, 3), Cell.CELL)]
    
    # Test adding CELL type
    add_single_layer(points, viewer, "Test Layer", Cell.CELL)
    assert len(viewer.layers) == 1
    layer = viewer.layers[0]
    assert layer.name == "Test Layer"
    assert np.array_equal(layer.data, np.array([[3, 2, 1]]))  # napari order

def test_add_single_layer_empty(make_napari_viewer):
    """Test empty cell list doesn't create a layer."""
    viewer = make_napari_viewer()
    add_single_layer([], viewer, "Empty", Cell.CELL)
    assert len(viewer.layers) == 0  # No layer should be added

def test_cells_to_array_axis_orders():
    """Test axis order handling (napari vs. brainglobe)."""
    cells = [Cell((1, 2, 3), Cell.CELL)]
    
    # Tests napari order (should reverse axes)
    napari_result = cells_to_array(cells, Cell.CELL, napari_order=True)
    assert np.array_equal(napari_result, np.array([[3, 2, 1]]))
    
    # Tests brainglobe order (original XYZ)
    bg_result = cells_to_array(cells, Cell.CELL, napari_order=False)
    assert np.array_equal(bg_result, np.array([[1, 2, 3]]))

def test_napari_array_to_cells_empty():
    """Test empty Points layer returns empty list."""
    mock_points = MagicMock()
    mock_points.data = np.zeros((0, 3))
    result = napari_array_to_cells(mock_points, Cell.CELL)
    assert result == []

def test_napari_array_to_cells_axis_conversion():
    """Test axis order conversion from napari to brainglobe."""
    mock_points = MagicMock()
    mock_points.data = np.array([[3, 2, 1]])  # napari order (ZYX)
    
    # Should convert back to brainglobe order (XYZ)
    cells = napari_array_to_cells(mock_points, Cell.CELL)
    assert cells[0].x == 1 and cells[0].y == 2 and cells[0].z == 3


def test_html_label_widget():
    """Simple unit test for the HTML Label widget"""
    label_widget = html_label_widget("A nice label", tag="h1")
    assert label_widget["widget_type"] == "Label"
    assert label_widget["label"] == "<h1>A nice label</h1>"

def test_cellfinder_header():
    """Test header widget creation."""
    header = cellfinder_header()
    assert "cellfinder" in header.text
    assert "documentation" in header.toolTip()
