from math import ceil
from pathlib import Path
from typing import Optional

import napari
from magicgui import magicgui
from magicgui.widgets import FunctionGui, ProgressBar
from napari.utils.notifications import show_info
from qtpy.QtWidgets import QScrollArea

from cellfinder.core.classify.cube_generator import get_cube_depth_min_max
from cellfinder.napari.utils import (
    add_layers,
    cellfinder_header,
    html_label_widget,
)

from .detect_containers import (
    ClassificationInputs,
    DataInputs,
    DetectionInputs,
    MiscInputs,
)
from .thread_worker import Worker

NETWORK_VOXEL_SIZES = [5, 1, 1]
CUBE_WIDTH = 50
CUBE_HEIGHT = 20
CUBE_DEPTH = 20

# If using ROI, how many extra planes to analyse
MIN_PLANES_ANALYSE = 0


def detect_widget() -> FunctionGui:
    """
    Create a detection plugin GUI.
    """
    progress_bar = ProgressBar()

    # options that is filled in from the gui
    options = {"signal_image": None, "background_image": None, "viewer": None}

    # signal and background images are separated out from the main magicgui
    # parameter selections and are inserted as widget children in their own
    # sub-containers of the root. Because if these image parameters are
    # included in the root widget, every time *any* parameter updates, the gui
    # freezes for a bit likely because magicgui is processing something for
    # all the parameters when any parameter changes. And this processing takes
    # particularly long for image parameters. Placing them as sub-containers
    # alleviates this
    @magicgui(
        call_button=False,
        persist=False,
        scrollable=False,
        labels=False,
        auto_call=True,
    )
    def signal_image_opt(
        viewer: napari.Viewer,
        signal_image: napari.layers.Image,
    ):
        """
        magicgui widget for setting the signal_image parameter.

        Parameters
        ----------
        signal_image : napari.layers.Image
             Image layer containing the labelled cells
        """
        options["signal_image"] = signal_image
        options["viewer"] = viewer

    @magicgui(
        call_button=False,
        persist=False,
        scrollable=False,
        labels=False,
        auto_call=True,
    )
    def background_image_opt(
        background_image: napari.layers.Image,
    ):
        """
        magicgui widget for setting the background image parameter.

        Parameters
        ----------
        background_image : napari.layers.Image
             Image layer without labelled cells
        """
        options["background_image"] = background_image

    @magicgui(
        detection_label=html_label_widget("Cell detection", tag="h3"),
        **DataInputs.widget_representation(),
        **DetectionInputs.widget_representation(),
        **ClassificationInputs.widget_representation(),
        **MiscInputs.widget_representation(),
        call_button=True,
        persist=True,
        reset_button=dict(widget_type="PushButton", text="Reset defaults"),
        scrollable=True,
    )
    def widget(
        detection_label,
        data_options,
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
        trained_model: Optional[Path],
        use_pre_trained_weights: bool,
        misc_options,
        start_plane: int,
        end_plane: int,
        n_free_cpus: int,
        analyse_local: bool,
        debug: bool,
        reset_button,
    ) -> None:
        """
        Run detection and classification.

        Parameters
        ----------
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
        use_pre_trained_weights : bool
            Select to use pre-trained model weights
        trained_model : Optional[Path]
            Trained model file path (home directory (default) -> pretrained
            weights)
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
        # we must manually call so that the parameters of these functions are
        # initialized and updated. Because, if the images are open in napari
        # before we open cellfinder, then these functions may never be called,
        # even though the image filenames are shown properly in the parameters
        # in the gui. Likely auto_call doesn't make magicgui call the functions
        # in this circumstance, only if the parameters are updated once
        # cellfinder plugin is fully open and initialized
        signal_image_opt()
        background_image_opt()

        signal_image = options["signal_image"]
        background_image = options["background_image"]
        viewer = options["viewer"]

        if signal_image is None or background_image is None:
            show_info("Both signal and background images must be specified.")
            return

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

        if use_pre_trained_weights:
            trained_model = None
        classification_inputs = ClassificationInputs(
            use_pre_trained_weights, trained_model
        )

        end_plane = len(signal_image.data) if end_plane == 0 else end_plane

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

        worker = Worker(
            data_inputs,
            detection_inputs,
            classification_inputs,
            misc_inputs,
        )
        worker.returned.connect(
            lambda points: add_layers(points, viewer=viewer)
        )

        # Make sure if the worker emits an error, it is propagated to this
        # thread
        def reraise(e):
            raise Exception from e

        worker.errored.connect(reraise)

        def update_progress_bar(label: str, max: int, value: int):
            progress_bar.label = label
            progress_bar.max = max
            progress_bar.value = value

        worker.update_progress_bar.connect(update_progress_bar)
        worker.start()

    widget.native.layout().insertWidget(0, cellfinder_header())

    @widget.reset_button.changed.connect
    def restore_defaults():
        """
        Restore default widget values.
        """
        defaults = {
            **DataInputs.defaults(),
            **DetectionInputs.defaults(),
            **ClassificationInputs.defaults(),
            **MiscInputs.defaults(),
        }
        for name, value in defaults.items():
            if value is not None:  # ignore fields with no default
                getattr(widget, name).value = value

    # Insert progress bar before the run and reset buttons
    widget.insert(-3, progress_bar)

    # add the signal and background image parameters
    # make it look as if it's directly in the root container
    signal_image_opt.margins = 0, 0, 0, 0
    # the parameters are updated using `auto_call` only. If False, magicgui
    # passes these as args to widget(), which doesn't list them as args
    signal_image_opt.gui_only = True
    widget.insert(3, signal_image_opt)
    widget.signal_image_opt.label = "Signal image"

    background_image_opt.margins = 0, 0, 0, 0
    background_image_opt.gui_only = True
    widget.insert(4, background_image_opt)
    widget.background_image_opt.label = "Background image"

    scroll = QScrollArea()
    scroll.setWidget(widget._widget._qwidget)
    widget._widget._qwidget = scroll

    return widget
