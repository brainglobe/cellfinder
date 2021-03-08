import napari
from pathlib import Path

from magicgui import magic_factory
from napari_plugin_engine import napari_hook_implementation
from typing import List

# from napari.qt.threading import thread_worker
from copy import deepcopy
from cellfinder_core.main import main as cellfinder_run
from .utils import cells_to_array

# TODO:
# how to store & fetch pre-trained models?


@magic_factory(
    header=dict(widget_type="Label", label="<h1>cellfinder</h1>"),
    data_section=dict(widget_type="Label", label="<h3>Data:</h3>"),
    voxel_size_z=dict(label="Voxel size (z)", step=0.1),
    voxel_size_y=dict(label="Voxel size (y)", step=0.1),
    voxel_size_x=dict(label="Voxel size (x)", step=0.1),
    performance_section=dict(widget_type="Label", label="<h3>Detection:</h3>"),
    ball_xy_size=dict(label="Ball filter (xy)"),
    ball_z_size=dict(label="Ball filter (z)"),
    Soma_diameter=dict(step=0.1),
    Ball_overlap=dict(step=0.1),
    Filter_width=dict(step=0.1),
    Cell_spread=dict(step=0.1),
    Start_plane=dict(min=0, max=100000),
    End_plane=dict(min=0, max=100000),
    classification_section=dict(widget_type="Label", label="<h3>Classification:</h3>"),
    # Classification_batch_size=dict(max=4096),
    misc_section=dict(widget_type="Label", label="<h3>Misc:</h3>"),
    call_button=True,
    # persist=True,
)
def cellfinder(
    header,
    data_section,
    Signal_image: napari.layers.Image,
    Background_image: napari.layers.Image,
    voxel_size_z: float = 5,
    voxel_size_y: float = 2,
    voxel_size_x: float = 2,
    performance_section=None,
    Soma_diameter: float = 16.0,
    ball_xy_size: float = 6,
    ball_z_size: float = 15,
    Ball_overlap: float = 0.6,
    Filter_width: float = 0.2,
    Threshold: int = 10,
    Cell_spread: float = 1.4,
    Max_cluster: int = 100000,
    classification_section=None,
    # Classification_batch_size: int = 2048,
    Trained_model: Path = Path.home(),
    misc_section=None,
    Start_plane: int = 0,
    End_plane: int = 0,
    Number_of_free_cpus: int = 2,
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
        Fraction of the morphological filter needed to be filled to retain a voxel
    Filter_width : float
        Laplacian of Gaussian filter width (as a fraction of soma diameter)
    Threshold : int
        Cell intensity threshold (as a multiple of noise above the mean)
    Cell_spread : float
        Cell spread factor (for splitting up cell clusters)
    Max_cluster : int
        Largest putative cell cluster (in cubic um) where splitting should be attempted
    Number_of_free_cpus : int
        How many CPU cores to leave free
    """

    if End_plane == 0:
        End_plane = len(Signal_image.data)

    voxel_sizes = (voxel_size_z, voxel_size_y, voxel_size_x)
    if Trained_model == Path.home():
        Trained_model = None

    points = run(
        Signal_image.data,
        Background_image.data,
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


@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    return cellfinder, {"name": "Cell detection"}
