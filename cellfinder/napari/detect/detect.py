from functools import partial
from math import ceil
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import napari
import napari.layers
from brainglobe_utils.cells.cells import Cell
from magicgui import magicgui
from magicgui.widgets import FunctionGui, ProgressBar
from napari.utils.notifications import show_info
from qtpy.QtWidgets import QScrollArea

from cellfinder.core.classify.cube_generator import get_cube_depth_min_max
from cellfinder.napari.utils import (
    add_classified_layers,
    add_single_layer,
    cellfinder_header,
    html_label_widget,
    napari_array_to_cells,
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


def get_heavy_widgets(
    options: Dict[str, Any],
) -> Tuple[Callable, Callable, Callable]:
    # signal and other input are separated out from the main magicgui
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
        call_button=False,
        persist=False,
        scrollable=False,
        labels=False,
        auto_call=True,
    )
    def cell_layer_opt(
        cell_layer: napari.layers.Points,
    ):
        """
        magicgui widget for setting the cell layer input when detection is
        skipped.

        Parameters
        ----------
        cell_layer : napari.layers.Points
            If detection is skipped, select the cell layer containing the
            detected cells to use for classification
        """
        options["cell_layer"] = cell_layer

    return signal_image_opt, background_image_opt, cell_layer_opt


def add_heavy_widgets(
    root: FunctionGui,
    widgets: Tuple[FunctionGui, ...],
    new_names: Tuple[str, ...],
    insertions: Tuple[str, ...],
) -> None:
    for widget, new_name, insertion in zip(widgets, new_names, insertions):
        # make it look as if it's directly in the root container
        widget.margins = 0, 0, 0, 0
        # the parameters of these widgets are updated using `auto_call` only.
        # If False, magicgui passes these as args to root() when the root's
        # function runs. But that doesn't list them as args of its function
        widget.gui_only = True
        root.insert(root.index(insertion) + 1, widget)
        getattr(root, widget.name).label = new_name


def restore_options_defaults(widget: FunctionGui) -> None:
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


def get_results_callback(
    skip_classification: bool, viewer: napari.Viewer
) -> Callable:
    """
    Returns the callback that is connected to output of the pipeline.
    It returns the detected points that we have to visualize.
    """
    if skip_classification:
        # after detection w/o classification, everything is unknown
        def done_func(points):
            add_single_layer(
                points,
                viewer=viewer,
                name="Cell candidates",
                cell_type=Cell.UNKNOWN,
            )

    else:
        # after classification we have either cell or unknown
        def done_func(points):
            add_classified_layers(
                points,
                viewer=viewer,
                unknown_name="Rejected",
                cell_name="Detected",
            )

    return done_func


def find_local_planes(
    viewer: napari.Viewer,
    voxel_size_z: float,
    signal_image: napari.layers.Image,
) -> Tuple[int, int]:
    """
    When detecting only locally, it returns the start and end planes to use.
    """
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

    return start_plane, end_plane


def reraise(e: Exception) -> None:
    """Re-raises the exception."""
    raise Exception from e


def detect_widget() -> FunctionGui:
    """
    Create a detection plugin GUI.
    """
    progress_bar = ProgressBar()

    # options that is filled in from the gui
    options = {
        "signal_image": None,
        "background_image": None,
        "viewer": None,
        "cell_layer": None,
    }

    signal_image_opt, background_image_opt, cell_layer_opt = get_heavy_widgets(
        options
    )

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
        skip_detection: bool,
        soma_diameter: float,
        log_sigma_size: float,
        n_sds_above_mean_thresh: float,
        n_sds_above_mean_tiled_thresh: float,
        tiled_thresh_tile_size: float,
        ball_xy_size: float,
        ball_z_size: float,
        ball_overlap_fraction: float,
        detection_batch_size: int,
        soma_spread_factor: float,
        max_cluster_size: float,
        classification_options,
        skip_classification: bool,
        use_pre_trained_weights: bool,
        trained_model: Optional[Path],
        classification_batch_size: int,
        misc_options,
        start_plane: int,
        end_plane: int,
        n_free_cpus: int,
        analyse_local: bool,
        use_gpu: bool,
        pin_memory: bool,
        debug: bool,
        reset_button,
    ) -> None:
        """
        Run detection and classification.

        Parameters
        ----------
        voxel_size_z : float
            Size of your voxels in the axial dimension (microns)
        voxel_size_y : float
            Size of your voxels in the y direction (top to bottom) (microns)
        voxel_size_x : float
            Size of your voxels in the x direction (left to right) (microns)
        skip_detection : bool
            If selected, the detection step is skipped and instead we get the
            detected cells from the cell layer below (from a previous
            detection run or import)
        soma_diameter : float
            The expected in-plane (xy) soma diameter (microns)
        log_sigma_size : float
            Gaussian filter width (as a fraction of soma diameter) used during
            2d in-plane Laplacian of Gaussian filtering
        n_sds_above_mean_thresh : float
            Per-plane intensity threshold (the number of standard deviations
            above the mean) of the filtered 2d planes used to mark pixels as
            foreground or background
        n_sds_above_mean_tiled_thresh : float
            Per-plane, per-tile intensity threshold (the number of standard
            deviations above the mean) for the filtered 2d planes used to mark
            pixels as foreground or background. When used, (tile size is not
            zero) a pixel is marked as foreground if its intensity is above
            both the per-plane and per-tile threshold. I.e. it's above the set
            number of standard deviations of the per-plane average and of the
            per-plane per-tile average for the tile that contains it.
        tiled_thresh_tile_size : float
            The tile size used to tile the x, y plane to calculate the local
            average intensity for the tiled threshold. The value is multiplied
            by soma diameter (i.e. 1 means one soma diameter). If zero, the
            tiled threshold is disabled and only the per-plane threshold is
            used. Tiling is done with 50% overlap when striding.
        ball_xy_size : float
            3d filter's in-plane (xy) filter ball size (microns)
        ball_z_size : float
            3d filter's axial (z) filter ball size (microns)
        ball_overlap_fraction : float
            3d filter's fraction of the ball filter needed to be filled by
            foreground voxels, centered on a voxel, to retain the voxel
        detection_batch_size: int
            The number of planes of the original data volume to process at
            once. The GPU/CPU memory must be able to contain this many planes
            for all the filters. For performance-critical applications, tune
            to maximize memory usage without
            running out. Check your GPU/CPU memory to verify it's not full
        soma_spread_factor : float
            Cell spread factor for determining the largest cell volume before
            splitting up cell clusters. Structures with spherical volume of
            diameter `soma_spread_factor * soma_diameter` or less will not be
            split
        max_cluster_size : float
            Largest detected cell cluster (in cubic um) where splitting
            should be attempted. Clusters above this size will be labeled
            as artifacts
        skip_classification : bool
            If selected, the classification step is skipped and all cells from
            the detection stage are added
        use_pre_trained_weights : bool
            Select to use pre-trained model weights
        trained_model : Optional[Path]
            Trained model file path (home directory (default) -> pretrained
            weights)
        classification_batch_size : int
            How many potential cells to classify at one time. The GPU/CPU
            memory must be able to contain at once this many data cubes for
            the models. For performance-critical applications, tune to
            maximize memory usage without running
            out. Check your GPU/CPU memory to verify it's not full
        start_plane : int
            First plane to process (to process a subset of the data)
        end_plane : int
            Last plane to process (to process a subset of the data)
        n_free_cpus : int
            How many CPU cores to leave free
        analyse_local : bool
            Only analyse planes around the current position
        use_gpu : bool
            If True, use GPU for processing (if available); otherwise, use CPU.
        pin_memory: bool
            Pins data to be sent to the GPU to the CPU memory. This allows
            faster GPU data speeds, but can only be used if the data used by
            the GPU can stay in the CPU RAM while the GPU uses it. I.e. there's
            enough RAM. Otherwise, if there's a risk of the RAM being paged, it
            shouldn't be used. Defaults to False.
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
        cell_layer_opt()

        signal_image = options["signal_image"]

        if signal_image is None or options["background_image"] is None:
            show_info("Both signal and background images must be specified.")
            return

        detected_cells = []
        if skip_detection:
            if options["cell_layer"] is None:
                show_info(
                    "Skip detection selected, but no existing cell layer "
                    "is selected."
                )
                return

            # set cells as unknown so that classification will process them
            detected_cells = napari_array_to_cells(
                options["cell_layer"], Cell.UNKNOWN
            )

        data_inputs = DataInputs(
            signal_image.data,
            options["background_image"].data,
            voxel_size_z,
            voxel_size_y,
            voxel_size_x,
        )

        detection_inputs = DetectionInputs(
            skip_detection,
            detected_cells,
            soma_diameter,
            ball_xy_size,
            ball_z_size,
            ball_overlap_fraction,
            log_sigma_size,
            n_sds_above_mean_thresh,
            n_sds_above_mean_tiled_thresh,
            tiled_thresh_tile_size,
            soma_spread_factor,
            max_cluster_size,
            detection_batch_size,
        )

        if use_pre_trained_weights:
            trained_model = None
        classification_inputs = ClassificationInputs(
            skip_classification,
            use_pre_trained_weights,
            trained_model,
            classification_batch_size,
        )

        if analyse_local:
            start_plane, end_plane = find_local_planes(
                options["viewer"], voxel_size_z, signal_image
            )
        elif not end_plane:
            end_plane = len(signal_image.data)

        misc_inputs = MiscInputs(
            start_plane,
            end_plane,
            n_free_cpus,
            analyse_local,
            use_gpu,
            pin_memory,
            debug,
        )

        worker = Worker(
            data_inputs,
            detection_inputs,
            classification_inputs,
            misc_inputs,
        )

        worker.returned.connect(
            get_results_callback(skip_classification, options["viewer"])
        )
        # Make sure if the worker emits an error, it is propagated to this
        # thread
        worker.errored.connect(reraise)
        worker.connect_progress_bar_callback(progress_bar)

        worker.start()

    widget.native.layout().insertWidget(0, cellfinder_header())

    # reset restores defaults
    widget.reset_button.changed.connect(
        partial(restore_options_defaults, widget)
    )

    # Insert progress bar before the run and reset buttons
    widget.insert(widget.index("debug") + 1, progress_bar)

    # add the signal and background image etc.
    add_heavy_widgets(
        widget,
        (background_image_opt, signal_image_opt, cell_layer_opt),
        ("Background image", "Signal image", "Candidate cell layer"),
        ("voxel_size_z", "voxel_size_z", "soma_diameter"),
    )

    scroll = QScrollArea()
    scroll.setWidget(widget._widget._qwidget)
    widget._widget._qwidget = scroll

    return widget
