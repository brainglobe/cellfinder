from pathlib import Path

from cellfinder_core.main import main as cellfinder_run
from imlib.cells.cells import Cell
from napari.qt.threading import thread_worker

from cellfinder_napari.utils import cells_to_array


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
    """Runs cellfinder in a separate thread, to prevent GUI blocking."""
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
