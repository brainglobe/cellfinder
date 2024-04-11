from typing import List, Tuple

import napari
import numpy as np
import pandas as pd
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


def add_layers(points: List[Cell], viewer: napari.Viewer) -> None:
    """
    Adds classified cell candidates as two separate point layers to the napari
    viewer.
    """
    detected, rejected = cells_to_array(points)

    viewer.add_points(
        rejected,
        name="Rejected",
        size=15,
        n_dimensional=True,
        opacity=0.6,
        symbol="ring",
        face_color="lightskyblue",
        visible=False,
        metadata=dict(point_type=Cell.UNKNOWN),
    )
    viewer.add_points(
        detected,
        name="Detected",
        size=15,
        n_dimensional=True,
        opacity=0.6,
        symbol="ring",
        face_color="lightgoldenrodyellow",
        metadata=dict(point_type=Cell.CELL),
    )


def cells_df_as_np(
    cells_df: pd.DataFrame,
    new_order: List[int] = [2, 1, 0],
    type_column: str = "type",
) -> np.ndarray:
    """
    Convert a dataframe to an array, dropping *type_column* and re-ordering
    the columns with *new_order*.
    """
    cells_df = cells_df.drop(columns=[type_column])
    cells = cells_df[cells_df.columns[new_order]]
    cells = cells.to_numpy()
    return cells


def cells_to_array(cells: List[Cell]) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.DataFrame([c.to_dict() for c in cells])
    points = cells_df_as_np(df[df["type"] == Cell.CELL])
    rejected = cells_df_as_np(df[df["type"] == Cell.UNKNOWN])
    return points, rejected
