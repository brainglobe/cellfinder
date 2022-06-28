import pytest
from imlib.cells.cells import Cell
from qtpy.QtWidgets import QGridLayout

from ..utils import add_button, add_combobox, add_layers, html_label_widget


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


@pytest.mark.parametrize("label_stack", [True, False])
@pytest.mark.parametrize("label", ["A label", None])
def test_add_combobox(label, label_stack):
    """
    Smoke test for add_combobox for all conditional branches
    """
    layout = QGridLayout()
    combobox = add_combobox(
        layout,
        row=0,
        label=label,
        items=["item 1", "item 2"],
        label_stack=label_stack,
    )
    assert combobox is not None


@pytest.mark.parametrize(
    argnames="alignment", argvalues=["center", "left", "right"]
)
def test_add_button(alignment):
    """
    Smoke tests for add_button for all conditional branches
    """
    layout = QGridLayout()
    button = add_button(
        layout=layout,
        connected_function=lambda: None,
        label="A button",
        row=0,
        alignment=alignment,
    )
    assert button is not None
