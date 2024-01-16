from typing import List, Tuple

import napari
import numpy as np
import pandas as pd
from brainglobe_utils.cells.cells import Cell
from pkg_resources import resource_filename

brainglobe_logo = resource_filename(
    "cellfinder", "napari/images/brainglobe.png"
)


widget_header = """
<p>Efficient cell detection in large images.</p>
<p><a href="https://brainglobe.info" style="color:gray;">Website</a></p>
<p><a href="https://brainglobe.info/documentation/cellfinder/user-guide/napari-plugin/index.html" style="color:gray;">Documentation</a></p>
<p><a href="https://github.com/brainglobe/cellfinder" style="color:gray;">Source</a></p>
<p><a href="https://doi.org/10.1371/journal.pcbi.1009074" style="color:gray;">Citation</a></p>
<p><small>For help, hover the cursor over each parameter.</small>
"""  # noqa: E501


def html_label_widget(label: str, *, tag: str = "b") -> dict:
    """
    Create a HMTL label for use with magicgui.
    """
    return dict(
        widget_type="Label",
        label=f"<{tag}>{label}</{tag}>",
    )


header_label_widget = html_label_widget(
    f"""
<img src="{brainglobe_logo}"width="100">
<p>cellfinder</p>
""",
    tag="h1",
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
