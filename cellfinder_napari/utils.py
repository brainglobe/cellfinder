from typing import Callable, List, Optional, Tuple

import napari
import numpy as np
import pandas as pd
from imlib.cells.cells import Cell
from pkg_resources import resource_filename
from qtpy.QtWidgets import (
    QComboBox,
    QLabel,
    QLayout,
    QMessageBox,
    QPushButton,
    QWidget,
)

brainglobe_logo = resource_filename(
    "cellfinder_napari", "images/brainglobe.png"
)


widget_header = """
<p>Efficient cell detection in large images.</p>
<p><a href="https://cellfinder.info" style="color:gray;">Website</a></p>
<p><a href="https://docs.brainglobe.info/cellfinder-napari/introduction" style="color:gray;">Documentation</a></p>
<p><a href="https://github.com/brainglobe/cellfinder-napari" style="color:gray;">Source</a></p>
<p><a href="https://www.biorxiv.org/content/10.1101/2020.10.21.348771v2" style="color:gray;">Citation</a></p>
<p><small>For help, hover the cursor over each parameter.</small>
"""


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
    Adds classified cell candidates as two separate point layers to the napari viewer.
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


def add_combobox(
    layout: QLayout,
    label: str,
    items: List[str],
    row: int,
    column: int = 0,
    label_stack: bool = False,
    callback=None,
    width: int = 150,
) -> Tuple[QComboBox, Optional[QLabel]]:
    """
    Add a selection box to *layout*.
    """
    if label_stack:
        combobox_row = row + 1
        combobox_column = column
    else:
        combobox_row = row
        combobox_column = column + 1
    combobox = QComboBox()
    combobox.addItems(items)
    if callback:
        combobox.currentIndexChanged.connect(callback)
    combobox.setMaximumWidth = width

    if label is not None:
        combobox_label = QLabel(label)
        combobox_label.setMaximumWidth = width
        layout.addWidget(combobox_label, row, column)
    else:
        combobox_label = None

    layout.addWidget(combobox, combobox_row, combobox_column)
    return combobox, combobox_label


def add_button(
    label: str,
    layout: QLayout,
    connected_function: Callable,
    row: int,
    column: int = 0,
    visibility: bool = True,
    minimum_width: int = 0,
    alignment: str = "center",
) -> QPushButton:
    """
    Add a button to *layout*.
    """
    button = QPushButton(label)
    if alignment == "center":
        pass
    elif alignment == "left":
        button.setStyleSheet("QPushButton { text-align: left; }")
    elif alignment == "right":
        button.setStyleSheet("QPushButton { text-align: right; }")

    button.setVisible(visibility)
    button.setMinimumWidth(minimum_width)
    layout.addWidget(button, row, column)
    button.clicked.connect(connected_function)
    return button


def display_question(widget: QWidget, title: str, message: str) -> bool:
    """
    Display a warning in a pop up that informs about overwriting files.
    """
    message_reply = QMessageBox.question(
        widget,
        title,
        message,
        QMessageBox.Yes | QMessageBox.Cancel,
    )
    if message_reply == QMessageBox.Yes:
        return True
    else:
        return False
