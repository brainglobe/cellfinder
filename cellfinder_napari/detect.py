from math import ceil
from pathlib import Path
from typing import List

import napari
from cellfinder_core.classify.cube_generator import get_cube_depth_min_max
from magicgui import magicgui

from cellfinder_napari.input_containers import (
    DataInputs,
    DetectionInputs,
    MiscInputs,
    ClassificationInputs,
)
from cellfinder_napari.utils import (
    brainglobe_logo,
    add_layers,
    html_label_widget,
)
from cellfinder_napari.thread_worker import run

NETWORK_VOXEL_SIZES = [5, 1, 1]
CUBE_WIDTH = 50
CUBE_HEIGHT = 20
CUBE_DEPTH = 20

# If using ROI, how many extra planes to analyse
MIN_PLANES_ANALYSE = 0


def detect():
    @magicgui(
        header=html_label_widget(
            f'<img src="{brainglobe_logo}"width="100">cellfinder', "h1"
        ),
        detection_label=html_label_widget("Cell detection", "h3"),
        **DataInputs.widget_representation(),
        **DetectionInputs.widget_representation(),
        **ClassificationInputs.widget_representation(),
        **MiscInputs.widget_representation(),
        call_button=True,
        persist=True,
        reset_button=dict(widget_type="PushButton", text="Reset defaults"),
    )
    def widget(
        header,
        detection_label,
        data_options,
        viewer: napari.Viewer,
        signal_image: napari.layers.Image,
        background_image: napari.layers.Image,
        voxel_size_z: float,
        voxel_size_y: float,
        voxel_size_x: float,
        detection_options,
        soma_diameter: float,
        ball_xy_size: float,
        ball_z_size: float,
        ball_overlap_fraction: float,
        log_sigma_size: float,
        n_sds_above_mean_thresh: int,
        soma_spread_factor: float,
        max_cluster_size: int,
        classification_options,
        trained_model: Path,
        misc_options,
        start_plane: int,
        end_plane: int,
        n_free_cpus: int,
        analyse_local: bool,
        debug: bool,
        reset_button,
    ) -> List[napari.types.LayerDataTuple]:
        """

        Parameters
        ----------
        signal_image : napari.layers.Image
             Image layer containing the labelled cells
        background_image : napari.layers.Image
             Image layer without labelled cells
        voxel_size_z : float
            Size of your voxels in the axial dimension
        voxel_size_y : float
            Size of your voxels in the y direction (top to bottom)
        voxel_size_x : float
            Size of your voxels in the x direction (left to right)
        soma_diameter : float
            The expected in-plane soma diameter (microns)
        ball_xy_size : float
            Elliptical morphological in-plane filter size (microns)
        ball_z_size : float
            Elliptical morphological axial filter size (microns)
        ball_overlap_fraction : float
            Fraction of the morphological filter needed to be filled
            to retain a voxel
        log_sigma_size : float
            Laplacian of Gaussian filter width (as a fraction of soma diameter)
        n_sds_above_mean_thresh : int
            Cell intensity threshold (as a multiple of noise above the mean)
        soma_spread_factor : float
            Cell spread factor (for splitting up cell clusters)
        max_cluster_size : int
            Largest putative cell cluster (in cubic um) where splitting
            should be attempted
        trained_model : Path
            Trained model file path (home directory (default) -> pretrained weights)
        start_plane : int
            First plane to process (to process a subset of the data)
        end_plane : int
            Last plane to process (to process a subset of the data)
        n_free_cpus : int
            How many CPU cores to leave free
        analyse_local : bool
            Only analyse planes around the current position
        debug : bool
            Increase logging
        reset_button :
            Reset parameters to default
        """
        data_inputs = DataInputs(
            signal_image.data,
            background_image.data,
            voxel_size_z,
            voxel_size_y,
            voxel_size_x,
        )

        detection_inputs = DetectionInputs(
            soma_diameter,
            ball_xy_size,
            ball_z_size,
            ball_overlap_fraction,
            log_sigma_size,
            n_sds_above_mean_thresh,
            soma_spread_factor,
            max_cluster_size,
        )

        if trained_model == Path.home():
            trained_model = None
        classification_inputs = ClassificationInputs(trained_model)

        if end_plane == 0:
            end_plane = len(signal_image.data)

        if analyse_local:
            current_plane = viewer.dims.current_step[0]

            # so a reasonable number of cells in the plane are detected
            planes_needed = MIN_PLANES_ANALYSE + int(
                ceil((CUBE_DEPTH * NETWORK_VOXEL_SIZES[0]) / voxel_size_z)
            )

            start_plane, end_plane = get_cube_depth_min_max(
                current_plane, planes_needed
            )
            start_plane = max(0, start_plane)
            end_plane = min(len(signal_image.data), end_plane)

        misc_inputs = MiscInputs(
            start_plane, end_plane, n_free_cpus, analyse_local, debug
        )

        worker = run(
            data_inputs,
            detection_inputs,
            classification_inputs,
            misc_inputs,
        )
        worker.returned.connect(
            lambda points: add_layers(points, viewer=viewer)
        )
        worker.start()

    widget.header.value = (
        "<p>Efficient cell detection in large images.</p>"
        '<p><a href="https://cellfinder.info" style="color:gray;">Website</a></p>'
        '<p><a href="https://docs.brainglobe.info/cellfinder/napari-plugin" style="color:gray;">Documentation</a></p>'
        '<p><a href="https://github.com/brainglobe/cellfinder-napari" style="color:gray;">Source</a></p>'
        '<p><a href="https://www.biorxiv.org/content/10.1101/2020.10.21.348771v2" style="color:gray;">Citation</a></p>'
        "<p><small>For help, hover the cursor over each parameter.</small>"
    )
    widget.header.native.setOpenExternalLinks(True)

    @widget.reset_button.changed.connect
    def restore_defaults():
        defaults = {
            **DataInputs.defaults(),
            **DetectionInputs.defaults(),
            **ClassificationInputs.defaults(),
            **MiscInputs.defaults(),
        }
        for name, value in defaults.items():
            if hasattr(widget, name):
                getattr(widget, name).value = value

    return widget
