import numpy as np
from brainglobe_utils.cells.cells import Cell

from cellfinder.napari.utils import (
    add_classified_layers,
    add_single_layer,
    cells_to_array,
    html_label_widget,
    napari_array_to_cells,
    napari_points_axis_order,
)


def test_add_classified_layers(make_napari_viewer):
    """Smoke test for add_classified_layers utility"""
    cell_pos = [1, 2, 3]
    unknown_pos = [4, 5, 6]
    points = [
        Cell(pos=cell_pos, cell_type=Cell.CELL, metadata={"size": 12}),
        Cell(pos=unknown_pos, cell_type=Cell.UNKNOWN, metadata={"radius": 3}),
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
    assert cell_layer.features is not None
    assert rej_layer.features is not None

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
    cells_again = napari_array_to_cells(cell_layer, cell_type=Cell.CELL)
    cells_again.extend(
        napari_array_to_cells(rej_layer, cell_type=Cell.UNKNOWN)
    )
    assert cells_again == points


def test_add_classified_layers_propagates_scale(make_napari_viewer):
    """Ensure scale is propagated to created Points layers."""
    scale = (5.0, 1.46, 1.46)

    points = [
        Cell(pos=[1, 2, 3], cell_type=Cell.CELL),
        Cell(pos=[4, 5, 6], cell_type=Cell.UNKNOWN),
    ]

    viewer = make_napari_viewer()

    add_classified_layers(
        points,
        viewer,
        unknown_name="rejected",
        cell_name="accepted",
        scale=scale,
    )
    # scale is applied to both created Points layers
    accepted = viewer.layers["accepted"]
    rejected = viewer.layers["rejected"]

    assert np.allclose(accepted.scale, scale)
    assert np.allclose(rejected.scale, scale)


def test_add_single_layer(make_napari_viewer):
    """Smoke test for add_single_layer utility"""
    cell_pos = [1, 2, 3], [4, 5, 6]
    # make sure that even if the metadata is different across points, it all
    # comes back correct
    points = [
        Cell(pos=cell_pos[0], cell_type=Cell.CELL, metadata={"size": 12}),
        Cell(pos=cell_pos[1], cell_type=Cell.CELL, metadata={"radius": 3}),
    ]
    viewer = make_napari_viewer()
    n_layers = len(viewer.layers)
    # adds a "detected" and a "rejected layer"
    add_single_layer(
        points,
        viewer,
        name="my_points",
        cell_type=Cell.CELL,
    )
    assert len(viewer.layers) == n_layers + 1

    # check name match
    cell_layer = None
    for layer in reversed(viewer.layers):
        if layer.name == "my_points" and cell_layer is None:
            cell_layer = layer
    assert cell_layer is not None
    assert cell_layer.data is not None
    assert cell_layer.features is not None

    # check data added in correct column order
    cell_data = np.array(cell_pos)
    assert np.all(
        cells_to_array(points, Cell.CELL, napari_order=False) == cell_data
    )
    # convert to napari order and check it is in napari
    cell_data = cell_data[:, napari_points_axis_order]
    assert np.all(cell_layer.data == cell_data)

    # get cells back from napari points
    cells_again = napari_array_to_cells(cell_layer, cell_type=Cell.CELL)
    assert cells_again == points


def test_html_label_widget():
    """Simple unit test for the HTML Label widget"""
    label_widget = html_label_widget("A nice label", tag="h1")
    assert label_widget["widget_type"] == "Label"
    assert label_widget["label"] == "<h1>A nice label</h1>"
