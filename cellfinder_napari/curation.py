import tifffile
import napari
import numpy as np
from qtpy import QtCore
from pathlib import Path
from napari.qt.threading import thread_worker
from qtpy.QtWidgets import (
    QLabel,
    QWidget,
    QFileDialog,
    QGridLayout,
    QGroupBox,
)
from brainglobe_napari_io.cellfinder.utils import convert_layer_to_cells
from imlib.cells.cells import Cell
from imlib.general.system import ensure_directory_exists
from imlib.IO.yaml import save_yaml

from .utils import add_combobox, add_button, display_info


# Constants used throughout
WINDOW_HEIGHT = 750
WINDOW_WIDTH = 1500
COLUMN_WIDTH = 150


class CurationWidget(QWidget):
    def __init__(
        self,
        viewer: napari.viewer.Viewer,
        cube_depth=20,
        cube_width=50,
        cube_height=50,
        network_voxel_sizes=[5, 1, 1],
        n_free_cpus=2,
        save_empty_cubes=False,
        max_ram=None,
    ):

        super(CurationWidget, self).__init__()

        self.non_cells_to_extract = None
        self.cells_to_extract = None

        self.cube_depth = cube_depth
        self.cube_width = cube_width
        self.cube_height = cube_height
        self.network_voxel_sizes = network_voxel_sizes
        self.n_free_cpus = n_free_cpus
        self.save_empty_cubes = save_empty_cubes
        self.max_ram = max_ram
        self.voxel_sizes = [5, 2, 2]
        self.batch_size = 32
        self.viewer = viewer

        self.signal_layer = None
        self.background_layer = None
        self.training_data_cell_layer = None
        self.training_data_non_cell_layer = None

        self.image_layer_names = self._get_layer_names()
        self.point_layer_names = self._get_layer_names(
            layer_type=napari.layers.Points
        )

        self.output_directory = None

        self.setup_main_layout()

        @self.viewer.layers.events.connect
        def update_layer_list(v):
            self.image_layer_names = self._get_layer_names()
            self.point_layer_names = self._get_layer_names(
                layer_type=napari.layers.Points
            )
            self._update_combobox_options(
                self.signal_image_choice, self.image_layer_names
            )
            self._update_combobox_options(
                self.background_image_choice, self.image_layer_names
            )
            self._update_combobox_options(
                self.training_data_cell_choice, self.point_layer_names
            )
            self._update_combobox_options(
                self.training_data_non_cell_choice, self.point_layer_names
            )

    @staticmethod
    def _update_combobox_options(combobox, options_list):
        original_text = combobox.currentText()
        combobox.clear()
        combobox.addItems(options_list)
        combobox.setCurrentText(original_text)

    def _get_layer_names(self, layer_type=napari.layers.Image, default=""):
        layer_names = [
            layer.name
            for layer in self.viewer.layers
            if type(layer) == layer_type
        ]

        if layer_names:
            return [default] + layer_names
        else:
            return [default]

    def setup_main_layout(self):
        """
        Construct main layout of widget
        """
        self.layout = QGridLayout()
        self.layout.setContentsMargins(10, 10, 10, 10)
        self.layout.setAlignment(QtCore.Qt.AlignTop)
        self.layout.setSpacing(4)

        self.add_loading_panel(1)

        self.status_label = QLabel()
        self.status_label.setText("Ready")
        self.layout.addWidget(self.status_label, 7, 0)

        self.setLayout(self.layout)

    def add_loading_panel(self, row, column=0):

        self.load_data_panel = QGroupBox("Load data")
        self.load_data_layout = QGridLayout()
        self.load_data_layout.setSpacing(15)
        self.load_data_layout.setContentsMargins(10, 10, 10, 10)
        self.load_data_layout.setAlignment(QtCore.Qt.AlignBottom)

        self.signal_image_choice, _ = add_combobox(
            self.load_data_layout,
            "Signal image",
            self.image_layer_names,
            1,
            callback=self.set_signal_image,
        )
        self.background_image_choice, _ = add_combobox(
            self.load_data_layout,
            "Background image",
            self.image_layer_names,
            2,
            callback=self.set_background_image,
        )
        self.training_data_cell_choice, _ = add_combobox(
            self.load_data_layout,
            "Training data (cells)",
            self.point_layer_names,
            3,
            callback=self.set_training_data_cell,
        )
        self.training_data_non_cell_choice, _ = add_combobox(
            self.load_data_layout,
            "Training_data (non_cells)",
            self.point_layer_names,
            4,
            callback=self.set_training_data_non_cell,
        )
        self.mark_as_cell_button = add_button(
            "Mark as cell(s)",
            self.load_data_layout,
            self.mark_as_cell,
            5,
        )
        self.mark_as_non_cell_button = add_button(
            "Mark as non cell(s)",
            self.load_data_layout,
            self.mark_as_non_cell,
            5,
            column=1,
        )
        self.add_training_data_button = add_button(
            "Add training data layers",
            self.load_data_layout,
            self.add_training_data,
            6,
        )
        self.save_training_data_button = add_button(
            "Save training data",
            self.load_data_layout,
            self.save_training_data,
            6,
            column=1,
        )
        self.load_data_layout.setColumnMinimumWidth(0, COLUMN_WIDTH)
        self.load_data_panel.setLayout(self.load_data_layout)
        self.load_data_panel.setVisible(True)
        self.layout.addWidget(self.load_data_panel, row, column, 1, 1)

    def set_signal_image(self):
        if self.signal_image_choice.currentText() != "":
            self.signal_layer = self.viewer.layers[
                self.signal_image_choice.currentText()
            ]

    def set_background_image(self):
        if self.background_image_choice.currentText() != "":
            self.background_layer = self.viewer.layers[
                self.background_image_choice.currentText()
            ]

    def set_training_data_cell(self):
        if self.training_data_cell_choice.currentText() != "":
            self.training_data_cell_layer = self.viewer.layers[
                self.training_data_cell_choice.currentText()
            ]
            self.training_data_cell_layer.metadata["point_type"] = Cell.CELL
            self.training_data_cell_layer.metadata["training_data"] = True

    def set_training_data_non_cell(self):
        if self.training_data_non_cell_choice.currentText() != "":
            self.training_data_non_cell_layer = self.viewer.layers[
                self.training_data_non_cell_choice.currentText()
            ]
            self.training_data_non_cell_layer.metadata[
                "point_type"
            ] = Cell.UNKNOWN
            self.training_data_non_cell_layer.metadata["training_data"] = True

    def add_training_data(
        self,
    ):
        cell_name = "Training data (cells)"
        non_cell_name = "Training data (non cells)"
        if not (
            self.training_data_cell_layer and self.training_data_non_cell_layer
        ):
            if not self.training_data_cell_layer:
                self.training_data_cell_layer = self.viewer.add_points(
                    None,
                    ndim=3,
                    symbol="ring",
                    n_dimensional=True,
                    size=15,
                    opacity=0.6,
                    face_color="lightgoldenrodyellow",
                    name=cell_name,
                    metadata=dict(point_type=Cell.CELL, training_data=True),
                )
                self.training_data_cell_choice.setCurrentText(cell_name)

            if not self.training_data_non_cell_layer:
                self.training_data_non_cell_layer = self.viewer.add_points(
                    None,
                    ndim=3,
                    symbol="ring",
                    n_dimensional=True,
                    size=15,
                    opacity=0.6,
                    face_color="lightskyblue",
                    name=non_cell_name,
                    metadata=dict(point_type=Cell.UNKNOWN, training_data=True),
                )
                self.training_data_non_cell_choice.setCurrentText(
                    non_cell_name
                )

        else:
            display_info(
                self,
                "Training data layers exist",
                "Training data layers already exist,  "
                "no more layers will be added.",
            )

    def mark_as_cell(self):
        self.mark_point_as_type("cell")

    def mark_as_non_cell(self):
        self.mark_point_as_type("non-cell")

    def mark_point_as_type(self, point_type):
        if not (
            self.training_data_cell_layer and self.training_data_non_cell_layer
        ):
            display_info(
                self,
                "No training data layers",
                "No training data layers have been chosen. "
                "Please add training data layers. ",
            )
            return

        if len(self.viewer.layers.selected) == 1:
            layer = self.viewer.layers.selected[0]
            if type(layer) == napari.layers.Points:

                if len(layer.data) > 0:
                    if point_type == "cell":
                        destination_layer = self.training_data_cell_layer
                    else:
                        destination_layer = self.training_data_non_cell_layer
                    print(
                        f"Adding {len(layer.selected_data)} "
                        f"points to layer: {destination_layer.name}"
                    )

                    for selected_point in layer.selected_data:
                        destination_layer.data = np.vstack(
                            (
                                destination_layer.data,
                                layer.data[selected_point],
                            )
                        )

                else:
                    display_info(
                        self,
                        "Not points selected",
                        "No points are selected in the current layer. "
                        "Please select some points.",
                    )

            else:
                display_info(
                    self,
                    "Not a points layer",
                    "This is not a points layer. "
                    "Please choose a points layer, and select some points.",
                )
        elif len(self.viewer.layers.selected) == 0:
            display_info(
                self,
                "No layers selected",
                "No layers are selected. "
                "Please choose a single points layer, and select some points.",
            )
        else:
            display_info(
                self,
                "Too many layers selected",
                "More than one layer is selected. "
                "Please choose a single points layer, and select some points.",
            )

    def save_training_data(self):
        if self.is_data_extractable():
            self.get_output_directory()
            if self.output_directory != "":
                self.__extract_cubes()
                self.__save_yaml_file()
                print("Done")

            self.status_label.setText("Ready")

    def __extract_cubes(self):
        self.status_label.setText("Extracting cubes")
        self.convert_layers_to_cells()

        worker = extract_cubes(
            self.cells_to_extract,
            self.non_cells_to_extract,
            self.output_directory,
            self.signal_layer.data,
            self.background_layer.data,
            self.voxel_sizes,
            self.network_voxel_sizes,
            self.batch_size,
            self.cube_width,
            self.cube_height,
            self.cube_depth,
        )
        worker.start()
        self.status_label.setText("Ready")

    def is_data_extractable(self):
        if (
            self.check_training_data_exists()
            and self.check_image_data_for_extraction()
        ):
            return True
        else:
            return False

    def check_image_data_for_extraction(self):
        if self.signal_layer and self.background_layer:
            if (
                self.signal_layer.data.shape
                == self.background_layer.data.shape
            ):
                return True
            else:
                display_info(
                    self,
                    "Images not the same shape",
                    "Please ensure both signal and background images are the "
                    "same size and shape.",
                )
                return False

        else:
            display_info(
                self,
                "No image data for cube extraction",
                "Please ensure both signal and background images are loaded "
                "into napari, and selected in the sidebar. ",
            )
            return False

    def check_training_data_exists(self):
        if not (
            self.training_data_cell_layer or self.training_data_non_cell_layer
        ):
            display_info(
                self,
                "No training data",
                "No training data layers have been added. "
                "Please add a layer and annotate some points.",
            )
            return False
        else:
            if (
                len(self.training_data_cell_layer.data) > 0
                or len(self.training_data_non_cell_layer.data) > 0
            ):
                return True
            else:
                display_info(
                    self,
                    "No training data",
                    "No training data points have been added. "
                    "Please annotate some points.",
                )
                return False

    def get_output_directory(self):
        """
        Shows file dialog to choose output directory
        """
        self.status_label.setText("Setting output directory...")
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        self.output_directory = QFileDialog.getExistingDirectory(
            self,
            "Select output directory",
            options=options,
        )
        if self.output_directory != "":
            self.output_directory = Path(self.output_directory)

    def convert_layers_to_cells(self):

        self.cells_to_extract = convert_layer_to_cells(
            self.training_data_cell_layer.data
        )
        self.non_cells_to_extract = convert_layer_to_cells(
            self.training_data_non_cell_layer.data, cells=False
        )

        self.cells_to_extract = list(set(self.cells_to_extract))
        self.non_cells_to_extract = list(set(self.non_cells_to_extract))

    def __save_yaml_file(self):
        # TODO: implement this in a portable way
        yaml_filename = self.output_directory / "training.yml"
        yaml_section = [
            {
                "cube_dir": str(self.output_directory / "cells"),
                "cell_def": "",
                "type": "cell",
                "signal_channel": 0,
                "bg_channel": 1,
            },
            {
                "cube_dir": str(self.output_directory / "non_cells"),
                "cell_def": "",
                "type": "no_cell",
                "signal_channel": 0,
                "bg_channel": 1,
            },
        ]

        yaml_contents = {"data": yaml_section}
        save_yaml(yaml_contents, yaml_filename)


@thread_worker
def extract_cubes(
    cells_to_extract,
    non_cells_to_extract,
    output_directory,
    signal_array,
    background_array,
    voxel_sizes,
    network_voxel_sizes,
    batch_size,
    cube_width,
    cube_height,
    cube_depth,
):
    from cellfinder_core.classify.cube_generator import (
        CubeGeneratorFromFile,
    )

    to_extract = {
        "cells": cells_to_extract,
        "non_cells": non_cells_to_extract,
    }

    for cell_type, cell_list in to_extract.items():
        print(f"Extracting type: {cell_type}")
        cell_type_output_directory = output_directory / cell_type
        print(f"Saving to: {cell_type_output_directory}")
        ensure_directory_exists(str(cell_type_output_directory))

        cube_generator = CubeGeneratorFromFile(
            cell_list,
            signal_array,
            background_array,
            voxel_sizes,
            network_voxel_sizes,
            batch_size=batch_size,
            cube_width=cube_width,
            cube_height=cube_height,
            cube_depth=cube_depth,
            extract=True,
        )

        extract_batches(cube_generator, cell_type_output_directory)
        print("Done")


def extract_batches(cube_generator, output_directory):
    for batch_idx, (image_batch, batch_info) in enumerate(cube_generator):
        image_batch = image_batch.astype(np.int16)
        for point, point_info in zip(image_batch, batch_info):
            point = np.moveaxis(point, 2, 0)

            for channel in range(0, point.shape[-1]):
                save_cube(point, point_info, channel, output_directory)


def save_cube(array, point_info, channel, output_directory):
    filename = (
        f"pCellz{point_info['z']}y{point_info['y']}"
        f"x{point_info['x']}Ch{channel}.tif"
    )
    tifffile.imsave(output_directory / filename, array[:, :, :, channel])
