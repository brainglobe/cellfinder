import napari
from cellfinder_napari.utils import cells_to_array
from imlib.cells.cells import Cell

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
