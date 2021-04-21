import napari
from pathlib import Path

from magicgui import magic_factory, widgets
from typing import List
from math import ceil

# from napari.qt.threading import thread_worker
from copy import deepcopy

from cellfinder_core.main import main as cellfinder_run
from cellfinder_core.classify.cube_generator import get_cube_depth_min_max
from .utils import cells_to_array

# TODO:
# how to store & fetch pre-trained models?

# TODO: params to add
NETWORK_VOXEL_SIZES = [5, 1, 1]
CUBE_WIDTH = 50
CUBE_HEIGHT = 20
CUBE_DEPTH = 20


def init(widget):
    widget.insert(0, widgets.Label(value="<h2>cellfinder</h2>"))
    widget.insert(1, widgets.Label(value="<h3>Cell detection</h3>"))
    widget.insert(2, widgets.Label(value="<b>Data:</b>"))
    widget.insert(8, widgets.Label(value="<b>Detection:</b>"))
    widget.insert(17, widgets.Label(value="<b>Classification:</b>"))
    widget.insert(19, widgets.Label(value="<b>Misc:</b>"))


@magic_factory(
    voxel_size_z=dict(label="Voxel size (z)", step=0.1),
    voxel_size_y=dict(label="Voxel size (y)", step=0.1),
    voxel_size_x=dict(label="Voxel size (x)", step=0.1),
    ball_xy_size=dict(label="Ball filter (xy)"),
    ball_z_size=dict(label="Ball filter (z)"),
    Soma_diameter=dict(step=0.1),
    Ball_overlap=dict(step=0.1),
    Filter_width=dict(step=0.1),
    Cell_spread=dict(step=0.1),
    Start_plane=dict(min=0, max=100000),
    End_plane=dict(min=0, max=100000),
    # Classification_batch_size=dict(max=4096),
    call_button=True,
    widget_init=init
    # persist=True,
)
def detect(
    viewer: napari.Viewer,
    Signal_image: napari.layers.Image,
    Background_image: napari.layers.Image,
    voxel_size_z: float = 5,
    voxel_size_y: float = 2,
    voxel_size_x: float = 2,
    Soma_diameter: float = 16.0,
    ball_xy_size: float = 6,
    ball_z_size: float = 15,
    Ball_overlap: float = 0.6,
    Filter_width: float = 0.2,
    Threshold: int = 10,
    Cell_spread: float = 1.4,
    Max_cluster: int = 100000,
    Trained_model: Path = Path.home(),
    Start_plane: int = 0,
    End_plane: int = 0,
    Number_of_free_cpus: int = 2,
    Analyse_field_of_view: bool = False,
) -> List[napari.types.LayerDataTuple]:
    """

    Parameters
    ----------
    voxel_size_z : float
        Size of your voxels in the axial dimension
    voxel_size_y : float
        Size of your voxels in x (left to right)
    voxel_size_z : float
        Size of your voxels in the y (top to bottom)
    Soma_diameter : float
        The expected in-plane soma diameter (microns)
    ball_xy_size : float
        Elliptical morphological in-plane filter size (microns)
    ball_z_size : float
        Elliptical morphological axial filter size (microns)
    Start_plane : int
        First plane to process (to process a subset of the data)
    End_plane : int
        Last plane to process (to process a subset of the data)
    Ball_overlap : float
        Fraction of the morphological filter needed to be filled
        to retain a voxel
    Filter_width : float
        Laplacian of Gaussian filter width (as a fraction of soma diameter)
    Threshold : int
        Cell intensity threshold (as a multiple of noise above the mean)
    Cell_spread : float
        Cell spread factor (for splitting up cell clusters)
    Max_cluster : int
        Largest putative cell cluster (in cubic um) where splitting
        should be attempted
    Number_of_free_cpus : int
        How many CPU cores to leave free
    Analyse_field_of_view : Only analyse the visible part of the image,
        with the minimum amount of 3D information
    """
    if End_plane == 0:
        End_plane = len(Signal_image.data)

    voxel_sizes = (voxel_size_z, voxel_size_y, voxel_size_x)
    if Trained_model == Path.home():
        Trained_model = None

    if Analyse_field_of_view:
        index = list(
            slice(int(i[0]), int(i[1])) for i in Signal_image.corner_pixels.T
        )
        index[0] = slice(0, len(Signal_image.data))

        signal_data = Signal_image.data[tuple(index)]
        background_data = Signal_image.data[tuple(index)]

        current_plane = viewer.dims.current_step[0]

        # so a reasonable number of cells in the plane are detected
        planes_needed = 2 * int(
            ceil((CUBE_DEPTH * NETWORK_VOXEL_SIZES[0]) / voxel_size_z)
        )

        Start_plane, End_plane = get_cube_depth_min_max(
            current_plane, planes_needed
        )
        Start_plane = max(0, Start_plane)
        End_plane = min(len(Signal_image.data), End_plane)

    else:
        signal_data = Signal_image.data
        background_data = Background_image.data

    points = run(
        signal_data,
        background_data,
        voxel_sizes,
        Soma_diameter,
        ball_xy_size,
        ball_z_size,
        Start_plane,
        End_plane,
        Ball_overlap,
        Filter_width,
        Threshold,
        Cell_spread,
        Max_cluster,
        Trained_model,
        Number_of_free_cpus,
        # Classification_batch_size,
    )
    for point in points:
        point.x = point.x + Signal_image.corner_pixels[0][2]
        point.y = point.y + Signal_image.corner_pixels[0][1]

    points, rejected = cells_to_array(points)

    points_properties = {
        "name": "Detected",
        "size": 15,
        "n_dimensional": True,
        "opacity": 0.6,
        "symbol": "ring",
        "face_color": "lightgoldenrodyellow",
        "visible": True,
    }
    rejected_properties = deepcopy(points_properties)
    rejected_properties["face_color"] = "lightskyblue"
    rejected_properties["name"] = "Rejected"

    rejected_properties["visible"] = False

    return [
        (rejected, rejected_properties, "points"),
        (points, points_properties, "points"),
    ]


# @thread_worker
def run(
    signal,
    background,
    voxel_sizes,
    Soma_diameter,
    ball_xy_size,
    ball_z_size,
    Start_plane,
    End_plane,
    Ball_overlap,
    Filter_width,
    Threshold,
    Cell_spread,
    Max_cluster,
    Trained_model,
    Number_of_free_cpus,
    # Classification_batch_size,
):

    points = cellfinder_run(
        signal,
        background,
        voxel_sizes,
        soma_diameter=Soma_diameter,
        ball_xy_size=ball_xy_size,
        ball_z_size=ball_z_size,
        start_plane=Start_plane,
        end_plane=End_plane,
        ball_overlap_fraction=Ball_overlap,
        log_sigma_size=Filter_width,
        n_sds_above_mean_thresh=Threshold,
        soma_spread_factor=Cell_spread,
        max_cluster_size=Max_cluster,
        trained_model=Trained_model,
        n_free_cpus=Number_of_free_cpus,
        # batch_size=Classification_batch_size,
    )
    return points
