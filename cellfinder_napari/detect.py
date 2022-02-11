from math import ceil
from pathlib import Path
from typing import List

import napari
from cellfinder_core.classify.cube_generator import get_cube_depth_min_max
from magicgui import magicgui

from cellfinder_napari.detect_utils import DataInputs, DetectionInputs, add_layers, default_parameters, run
from cellfinder_napari.utils import brainglobe_logo

NETWORK_VOXEL_SIZES = [5, 1, 1]
CUBE_WIDTH = 50
CUBE_HEIGHT = 20
CUBE_DEPTH = 20

# If using ROI, how many extra planes to analyse
MIN_PLANES_ANALYSE = 0

def detect():
    DEFAULT_PARAMETERS = default_parameters()
    @magicgui(
        header=dict(
            widget_type="Label",
            label=f'<h1><img src="{brainglobe_logo}"width="100">cellfinder</h1>',
        ),
        detection_label=dict(
            widget_type="Label",
            label="<h3>Cell detection</h3>",
        ),
        data_options=dict(
            widget_type="Label",
            label="<b>Data:</b>",
        ),
        detection_options=dict(
            widget_type="Label",
            label="<b>Detection:</b>",
        ),
        classification_options=dict(
            widget_type="Label",
            label="<b>Classification:</b>",
        ),
        misc_options=dict(
            widget_type="Label",
            label="<b>Misc:</b>",
        ),
        voxel_size_z=dict(
            value=DEFAULT_PARAMETERS["voxel_size_z"],
            label="Voxel size (z)",
            step=0.1,
        ),
        voxel_size_y=dict(
            value=DEFAULT_PARAMETERS["voxel_size_y"],
            label="Voxel size (y)",
            step=0.1,
        ),
        voxel_size_x=dict(
            value=DEFAULT_PARAMETERS["voxel_size_x"],
            label="Voxel size (x)",
            step=0.1,
        ),
        soma_diameter=dict(
            value=DEFAULT_PARAMETERS["soma_diameter"], step=0.1
        ),
        ball_xy_size=dict(
            value=DEFAULT_PARAMETERS["ball_xy_size"], label="Ball filter (xy)"
        ),
        ball_z_size=dict(
            value=DEFAULT_PARAMETERS["ball_z_size"], label="Ball filter (z)"
        ),
        ball_overlap=dict(value=DEFAULT_PARAMETERS["ball_overlap"], step=0.1),
        filter_width=dict(value=DEFAULT_PARAMETERS["filter_width"], step=0.1),
        threshold=dict(value=DEFAULT_PARAMETERS["threshold"], step=0.1),
        cell_spread=dict(value=DEFAULT_PARAMETERS["cell_spread"], step=0.1),
        max_cluster=dict(
            value=DEFAULT_PARAMETERS["max_cluster"], min=0, max=10000000
        ),
        trained_model=dict(value=DEFAULT_PARAMETERS["trained_model"]),
        start_plane=dict(
            value=DEFAULT_PARAMETERS["start_plane"], min=0, max=100000
        ),
        end_plane=dict(
            value=DEFAULT_PARAMETERS["end_plane"], min=0, max=100000
        ),
        number_of_free_cpus=dict(
            value=DEFAULT_PARAMETERS["number_of_free_cpus"]
        ),
        analyse_local=dict(
            value=DEFAULT_PARAMETERS["analyse_local"], label="Analyse local"
        ),
        debug=dict(value=DEFAULT_PARAMETERS["debug"]),
        # Classification_batch_size=dict(max=4096),
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
        ball_overlap: float,
        filter_width: float,
        threshold: int,
        cell_spread: float,
        max_cluster: int,
        classification_options,
        trained_model: Path,
        misc_options,
        start_plane: int,
        end_plane: int,
        number_of_free_cpus: int,
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
        ball_overlap : float
            Fraction of the morphological filter needed to be filled
            to retain a voxel
        filter_width : float
            Laplacian of Gaussian filter width (as a fraction of soma diameter)
        threshold : int
            Cell intensity threshold (as a multiple of noise above the mean)
        cell_spread : float
            Cell spread factor (for splitting up cell clusters)
        max_cluster : int
            Largest putative cell cluster (in cubic um) where splitting
            should be attempted
        trained_model : Path
            Trained model file path (home directory (default) -> pretrained weights)
        start_plane : int
            First plane to process (to process a subset of the data)
        end_plane : int
            Last plane to process (to process a subset of the data)
        number_of_free_cpus : int
            How many CPU cores to leave free
        analyse_local : bool
            Only analyse planes around the current position
        debug : bool
            Increase logging
        reset_button :
            Reset parameters to default
        """

        if end_plane == 0:
            end_plane = len(signal_image.data)

        if trained_model == Path.home():
            trained_model = None

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

        # initialise input
        voxel_sizes = (voxel_size_z, voxel_size_y, voxel_size_x)
        dataInputs = DataInputs(signal_image.data, \
            background_image.data,
            voxel_sizes)

        detectionInputs = DetectionInputs( \
            start_plane, 
            end_plane,
            soma_diameter,
            ball_xy_size,
            ball_z_size, 
            ball_overlap,
            filter_width,
            threshold,
            cell_spread,
            max_cluster)

        worker = run(
            dataInputs,
            detectionInputs,
            trained_model,
            number_of_free_cpus,
            # Classification_batch_size,
        )
        worker.returned.connect(lambda points : add_layers(points, viewer=viewer))
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
        for name, value in DEFAULT_PARAMETERS.items():
            getattr(widget, name).value = value

    return widget
