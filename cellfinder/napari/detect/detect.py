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
import logging

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

logger = logging.getLogger(__name__)

NETWORK_VOXEL_SIZES = [5, 1, 1]
CUBE_WIDTH = 50
CUBE_HEIGHT = 20
CUBE_DEPTH = 20
MIN_PLANES_ANALYSE = 0

def get_heavy_widgets(options: Dict[str, Any]) -> Tuple[Callable, Callable, Callable]:
    @magicgui(call_button=False, persist=False, scrollable=False, labels=False, auto_call=True)
    def signal_image_opt(viewer: napari.Viewer, signal_image: napari.layers.Image):
        options["signal_image"] = signal_image
        options["viewer"] = viewer

    @magicgui(call_button=False, persist=False, scrollable=False, labels=False, auto_call=True)
    def background_image_opt(background_image: napari.layers.Image):
        options["background_image"] = background_image

    @magicgui(call_button=False, persist=False, scrollable=False, labels=False, auto_call=True)
    def cell_layer_opt(cell_layer: napari.layers.Points):
        options["cell_layer"] = cell_layer

    return signal_image_opt, background_image_opt, cell_layer_opt

def add_heavy_widgets(root: FunctionGui, widgets: Tuple[FunctionGui, ...], 
                     new_names: Tuple[str, ...], insertions: Tuple[str, ...]) -> None:
    for widget, new_name, insertion in zip(widgets, new_names, insertions):
        widget.margins = (0, 0, 0, 0)
        widget.gui_only = True
        root.insert(root.index(insertion) + 1, widget)
        getattr(root, widget.name).label = new_name

def restore_options_defaults(widget: FunctionGui) -> None:
    defaults = {
        **DataInputs.defaults(),
        **DetectionInputs.defaults(),
        **ClassificationInputs.defaults(),
        **MiscInputs.defaults(),
    }
    for name, value in defaults.items():
        if value is not None:
            getattr(widget, name).value = value

def get_results_callback(skip_classification: bool, viewer: napari.Viewer) -> Callable:
    def handle_empty_results():
        show_info("No cells detected. Try adjusting:\n"
                 "- Detection thresholds\n"
                 "- Soma diameter\n"
                 "- Image preprocessing")
    
    if skip_classification:
        def done_func(points):
            if not points:
                handle_empty_results()
                return
            add_single_layer(
                points,
                viewer=viewer,
                name="Cell candidates",
                cell_type=Cell.UNKNOWN,
            )
    else:
        def done_func(points):
            if not points:
                handle_empty_results()
                return
            add_classified_layers(
                points,
                viewer=viewer,
                unknown_name="Rejected",
                cell_name="Detected",
            )
    return done_func

def find_local_planes(viewer: napari.Viewer, voxel_size_z: float, 
                     signal_image: napari.layers.Image) -> Tuple[int, int]:
    current_plane = viewer.dims.current_step[0]
    planes_needed = MIN_PLANES_ANALYSE + int(
        ceil((CUBE_DEPTH * NETWORK_VOXEL_SIZES[0]) / voxel_size_z)
    )
    start_plane, end_plane = get_cube_depth_min_max(current_plane, planes_needed)
    return max(0, start_plane), min(len(signal_image.data), end_plane)

def reraise(e: Exception) -> None:
    raise e

def detect_widget() -> FunctionGui:
    progress_bar = ProgressBar()
    options = {
        "signal_image": None,
        "background_image": None,
        "viewer": None,
        "cell_layer": None,
    }

    signal_image_opt, background_image_opt, cell_layer_opt = get_heavy_widgets(options)

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
        ball_xy_size: float,
        ball_z_size: float,
        ball_overlap_fraction: float,
        log_sigma_size: float,
        n_sds_above_mean_thresh: int,
        soma_spread_factor: float,
        max_cluster_size: int,
        classification_options,
        skip_classification: bool,
        use_pre_trained_weights: bool,
        trained_model: Optional[Path],
        batch_size: int,
        misc_options,
        start_plane: int,
        end_plane: int,
        n_free_cpus: int,
        analyse_local: bool,
        debug: bool,
        reset_button,
    ) -> None:
        # Validate parameters
        if soma_diameter <= 0 or ball_xy_size <= 0 or ball_z_size <= 0:
            show_info("Soma diameter and filter sizes must be positive")
            return

        signal_image_opt()
        background_image_opt()
        cell_layer_opt()

        if options["signal_image"] is None or options["background_image"] is None:
            show_info("Both signal and background images must be specified.")
            return

        detected_cells = []
        if skip_detection:
            if options["cell_layer"] is None:
                show_info("Skip detection selected but no cell layer selected.")
                return
            detected_cells = napari_array_to_cells(options["cell_layer"], Cell.UNKNOWN)

        data_inputs = DataInputs(
            options["signal_image"].data,
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
            soma_spread_factor,
            max_cluster_size,
        )

        if use_pre_trained_weights:
            trained_model = None
        classification_inputs = ClassificationInputs(
            skip_classification,
            use_pre_trained_weights,
            trained_model,
            batch_size,
        )

        if analyse_local:
            start_plane, end_plane = find_local_planes(
                options["viewer"], voxel_size_z, options["signal_image"]
            )
        elif not end_plane:
            end_plane = len(options["signal_image"].data)

        misc_inputs = MiscInputs(
            start_plane, end_plane, n_free_cpus, analyse_local, debug
        )

        worker = Worker(
            data_inputs,
            detection_inputs,
            classification_inputs,
            misc_inputs,
        )

        def handle_result(points):
            if not points:
                logger.warning("Cell detection completed with no cells found")
                show_info("No cells detected. Try adjusting parameters.")
            else:
                logger.info(f"Found {len(points)} cells")
                get_results_callback(skip_classification, options["viewer"])(points)

        worker.returned.connect(handle_result)
        worker.errored.connect(reraise)
        worker.connect_progress_bar_callback(progress_bar)
        worker.start()

    widget.native.layout().insertWidget(0, cellfinder_header())
    widget.reset_button.changed.connect(partial(restore_options_defaults, widget))
    widget.insert(widget.index("debug") + 1, progress_bar)

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