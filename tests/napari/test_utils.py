import numpy as np
import pytest
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


def test_html_label_widget():
    """Simple unit test for the HTML Label widget"""
    label_widget = html_label_widget("A nice label", tag="h1")
    assert label_widget["widget_type"] == "Label"
    assert label_widget["label"] == "<h1>A nice label</h1>"

def test_get_plane_size_in_memory_numpy():
    arr = np.zeros((10, 256, 256), dtype=np.uint16) 
    expected_size=((256*256*2)*1.1)/(1024**2)
    result = get_plane_size_in_memory(arr)
    assert abs(result - expected_size) < 0.01

def test_get_plane_size_in_memory_dask():
    darr = da.zeros((5, 128, 128), chunks=(1, 128, 128), dtype=np.uint8)
    expected_size = ((128 * 128 * 1) * 1.1) / (1024 ** 2)
    result = get_plane_size_in_memory(darr)
    assert abs(result - expected_size) < 0.01
