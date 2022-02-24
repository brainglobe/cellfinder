import pandas as pd
from imlib.cells.cells import Cell
from pkg_resources import resource_filename
from qtpy.QtWidgets import QComboBox, QLabel, QMessageBox, QPushButton
from imlib.cells.cells import Cell

brainglobe_logo = resource_filename(
    "cellfinder_napari", "images/brainglobe.png"
)

def html_label_widget(label : str, tag : str = 'b'):
    return dict(
            widget_type="Label",
            label=f"<{tag}>{label}</{tag}>",
        )


def add_layers(points, viewer):
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


def cells_df_as_np(cells_df, new_order=[2, 1, 0], type_column="type"):
    cells_df = cells_df.drop(columns=[type_column])
    cells = cells_df[cells_df.columns[new_order]]
    cells = cells.to_numpy()
    return cells


def cells_to_array(cells):
    df = pd.DataFrame([c.to_dict() for c in cells])
    points = cells_df_as_np(df[df["type"] == Cell.CELL])
    rejected = cells_df_as_np(df[df["type"] == Cell.UNKNOWN])
    return points, rejected


def add_combobox(
    layout,
    label,
    items,
    row,
    column=0,
    label_stack=False,
    callback=None,
    width=150,
):
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
    label,
    layout,
    connected_function,
    row,
    column=0,
    visibility=True,
    minimum_width=0,
    alignment="center",
):
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


def display_info(widget, title, message):
    """
    Display a warning in a pop up that informs
    about overwriting files
    """
    QMessageBox.information(widget, title, message, QMessageBox.Ok)


def display_question(widget, title, message):
    """
    Display a warning in a pop up that informs
    about overwriting files
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
