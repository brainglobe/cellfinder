import logging
from typing import List, Tuple

import napari
import napari.layers
import numpy as np
from brainglobe_utils.cells.cells import Cell
from brainglobe_utils.qtpy.logo import header_widget

logger = logging.getLogger(__name__)


def html_label_widget(label: str, *, tag: str = "b") -> dict:
    """Create a HTML label for use with magicgui."""
    return {
        "widget_type": "Label",
        "label": f"<{tag}>{label}</{tag}>",
    }


def cellfinder_header():
    """Create the header containing the brainglobe logo and documentation links."""
    return header_widget(
        "cellfinder",
        "Efficient cell detection in large images.",
        documentation_path="cellfinder/user-guide/napari-plugin/index.html",
        citation_doi="https://doi.org/10.1371/journal.pcbi.1009074",
        help_text="For help, hover the cursor over each parameter.",
    )


# Axis order mapping
napari_points_axis_order = (2, 1, 0)
brainglobe_points_axis_order = napari_points_axis_order


def add_classified_layers(
    points: List[Cell],
    viewer: napari.Viewer,
    unknown_name: str = "Rejected",
    cell_name: str = "Detected",
) -> None:
    """
    Adds cell candidates as two separate point layers.
    Only adds layers if cells of that type exist.
    """
    unknown_cells = cells_to_array(points, Cell.UNKNOWN, napari_order=True)
    detected_cells = cells_to_array(points, Cell.CELL, napari_order=True)

    if len(unknown_cells) > 0:
        viewer.add_points(
            unknown_cells,
            name=unknown_name,
            size=15,
            n_dimensional=True,
            opacity=0.6,
            symbol="ring",
            face_color="lightskyblue",
            visible=False,
            metadata={"point_type": Cell.UNKNOWN},
        )

    if len(detected_cells) > 0:
        viewer.add_points(
            detected_cells,
            name=cell_name,
            size=15,
            n_dimensional=True,
            opacity=0.6,
            symbol="ring",
            face_color="lightgoldenrodyellow",
            metadata={"point_type": Cell.CELL},
        )


def add_single_layer(
    points: List[Cell],
    viewer: napari.Viewer,
    name: str,
    cell_type: int,
) -> None:
    """Adds all cells of specified type to a new point layer."""
    points_array = cells_to_array(points, cell_type, napari_order=True)
    if len(points_array) > 0:
        viewer.add_points(
            points_array,
            name=name,
            size=15,
            n_dimensional=True,
            opacity=0.6,
            symbol="ring",
            face_color="lightskyblue",
            visible=True,
            metadata={"point_type": cell_type},
        )


def cells_to_array(
    cells: List[Cell], cell_type: int, napari_order: bool = True
) -> np.ndarray:
    """
    Converts cells of given type to a 2D position array.
    Returns empty array if no cells found.
    """
    if not cells:
        logger.debug("Empty cell list received")
        return np.zeros((0, 3), dtype=np.float32)

    filtered_cells = [c for c in cells if c.type == cell_type]
    if not filtered_cells:
        return np.zeros((0, 3), dtype=np.float32)

    points = np.array([(c.x, c.y, c.z) for c in filtered_cells])
    return points[:, napari_points_axis_order] if napari_order else points


def napari_array_to_cells(
    points: napari.layers.Points,
    cell_type: int,
    brainglobe_order: Tuple[int, int, int] = brainglobe_points_axis_order,
) -> List[Cell]:
    """Converts napari Points layer to list of Cell objects."""
    if points is None or len(points.data) == 0:
        return []

    data = np.asarray(points.data)[:, brainglobe_order].tolist()
    return [Cell(pos=row, cell_type=cell_type) for row in data]
