from brainglobe_utils.cells.cells import Cell

from cellfinder.napari.utils import (
    add_layers,
    html_label_widget,
)


def test_add_layers(make_napari_viewer):
    """Smoke test for add_layers utility"""
    points = [
        Cell(pos=[1, 2, 3], cell_type=Cell.CELL),
        Cell(pos=[4, 5, 6], cell_type=Cell.UNKNOWN),
    ]
    viewer = make_napari_viewer()
    n_layers = len(viewer.layers)
    add_layers(points, viewer)  # adds a "detected" and a "rejected layer"
    assert len(viewer.layers) == n_layers + 2


def test_html_label_widget():
    """Simple unit test for the HTML Label widget"""
    label_widget = html_label_widget("A nice label", tag="h1")
    assert label_widget["widget_type"] == "Label"
    assert label_widget["label"] == "<h1>A nice label</h1>"
