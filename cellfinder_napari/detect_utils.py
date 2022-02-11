from dataclasses import dataclass, asdict, fields
from pathlib import Path

import napari
from cellfinder_core.main import main as cellfinder_run
from imlib.cells.cells import Cell
from napari.qt.threading import thread_worker

from cellfinder_napari.utils import cells_to_array

@dataclass
class DataInputs:
    signal : napari.layers.Image
    background: napari.layers.Image
    voxel_sizes: tuple

@dataclass
class DetectionInputs:
    start_plane: int
    end_plane: int
    soma_diameter: float
    ball_xy_size: float
    ball_z_size: float
    ball_overlap_fraction: float
    log_sigma_size: float
    n_sds_above_mean_thresh: int
    soma_spread_factor: float
    max_cluster_size: int

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

@thread_worker
def run(
    data_inputs : DataInputs,
    detection_inputs: DetectionInputs,
    Trained_model,
    Number_of_free_cpus : int,
    # Classification_batch_size,
):
    """Runs cellfinder in a separate thread, to prevent GUI blocking."""
    points = cellfinder_run(
        *asdict(data_inputs).values(),
        **asdict(detection_inputs),
        trained_model=Trained_model,
        n_free_cpus=Number_of_free_cpus,
        # batch_size=Classification_batch_size,
    )
    return points

def default_parameters():
    return dict(
    voxel_size_z=5,
    voxel_size_y=2,
    voxel_size_x=2,
    Soma_diameter=16.0,
    ball_xy_size=6,
    ball_z_size=15,
    Ball_overlap=0.6,
    Filter_width=0.2,
    Threshold=10,
    Cell_spread=1.4,
    Max_cluster=100000,
    Trained_model=Path.home(),
    Start_plane=0,
    End_plane=0,
    Number_of_free_cpus=2,
    Analyse_local=False,
    Debug=False,
    )
