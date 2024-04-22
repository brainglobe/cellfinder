from typing import List, Tuple

import napari
import napari.layers
import numpy as np
from brainglobe_utils.cells.cells import Cell
from brainglobe_utils.qtpy.logo import header_widget


def html_label_widget(label: str, *, tag: str = "b") -> dict:
    """
    Create a HMTL label for use with magicgui.
    """
    return dict(
        widget_type="Label",
        label=f"<{tag}>{label}</{tag}>",
    )


def cellfinder_header():
    """
    Create the header containing the brainglobe logo and documentation links
    for all cellfinder widgets.
    """
    return header_widget(
        "cellfinder",
        "Efficient cell detection in large images.",
        documentation_path="cellfinder/user-guide/napari-plugin/index.html",
        citation_doi="https://doi.org/10.1371/journal.pcbi.1009074",
        help_text="For help, hover the cursor over each parameter.",
    )


# the xyz axis order in napari relative to ours. I.e. our zeroth axis is the
# napari last axis. Ours is XYZ.
napari_points_axis_order = 2, 1, 0
# the xyz axis order in brainglobe relative to napari. I.e. napari's zeroth
# axis is our last axis - it's just flipped
brainglobe_points_axis_order = napari_points_axis_order


def add_classified_layers(
    points: List[Cell],
    viewer: napari.Viewer,
    unknown_name: str = "Rejected",
    cell_name: str = "Detected",
) -> None:
    """
    Adds cell candidates as two separate point layers - unknowns and cells, to
    the napari viewer. Does not add any other cell types, only Cell.UNKNOWN
    and Cell.CELL from the list of cells.
    """
    viewer.add_points(
        cells_to_array(points, Cell.UNKNOWN, napari_order=True),
        name=unknown_name,
        size=15,
        n_dimensional=True,
        opacity=0.6,
        symbol="ring",
        face_color="lightskyblue",
        visible=False,
        metadata=dict(point_type=Cell.UNKNOWN),
    )
    viewer.add_points(
        cells_to_array(points, Cell.CELL, napari_order=True),
        name=cell_name,
        size=15,
        n_dimensional=True,
        opacity=0.6,
        symbol="ring",
        face_color="lightgoldenrodyellow",
        metadata=dict(point_type=Cell.CELL),
    )


def add_single_layer(
    points: List[Cell],
    viewer: napari.Viewer,
    name: str,
    cell_type: int,
) -> None:
    """
    Adds all cells of cell_type Cell.TYPE to a new point layer in the napari
    viewer, with given name.
    """
    viewer.add_points(
        cells_to_array(points, cell_type, napari_order=True),
        name=name,
        size=15,
        n_dimensional=True,
        opacity=0.6,
        symbol="ring",
        face_color="lightskyblue",
        visible=True,
        metadata=dict(point_type=cell_type),
    )


def cells_to_array(
    cells: List[Cell], cell_type: int, napari_order: bool = True
) -> np.ndarray:
    """
    Converts all the cells of the given type as a 2D pos array.
    The column order is either XYZ, otherwise it's the napari ordering
    of the 3 axes (napari_points_axis_order).
    """
    cells = [c for c in cells if c.type == cell_type]
    if not cells:
        # make sure we return 2d array if cells is empty
        return np.zeros((0, 3), dtype=np.int_)
    points = np.array([(c.x, c.y, c.z) for c in cells])

    if napari_order:
        return points[:, napari_points_axis_order]
    return points


def napari_array_to_cells(
    points: napari.layers.Points,
    cell_type: int,
    brainglobe_order: Tuple[int, int, int] = brainglobe_points_axis_order,
) -> List[Cell]:
    """
    Takes a napari Points layer and returns a list of cell objects, one for
    each point in the layer.
    """
    data = np.asarray(points.data)[:, brainglobe_order].tolist()

    cells = []
    for row in data:
        cells.append(Cell(pos=row, cell_type=cell_type))

    return cells
