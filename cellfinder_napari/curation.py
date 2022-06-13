from pathlib import Path
from typing import List, Optional, Tuple, Union

import napari
import numpy as np
import tifffile
from brainglobe_napari_io.cellfinder.utils import convert_layer_to_cells
from imlib.cells.cells import Cell
from imlib.IO.yaml import save_yaml
from magicgui.widgets import ProgressBar
from napari.qt.threading import thread_worker
from napari.utils.notifications import show_info
from qtpy import QtCore
from qtpy.QtWidgets import (
    QComboBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QLabel,
    QWidget,
)

from .utils import add_button, add_combobox, display_question

# Constants used throughout
WINDOW_HEIGHT = 750
WINDOW_WIDTH = 1500
COLUMN_WIDTH = 150


class CurationWidget(QWidget):
    def __init__(
        self,
        viewer: napari.viewer.Viewer,
        cube_depth: int = 20,
        cube_width: int = 50,
        cube_height: int = 50,
        network_voxel_sizes: Tuple[int, int, int] = (5, 1, 1),
        n_free_cpus: int = 2,
        save_empty_cubes: bool = False,
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

        self.output_directory: Optional[Path] = None

        self.setup_main_layout()

        @self.viewer.layers.events.connect
        def update_layer_list(v: napari.viewer.Viewer):
            """
            Update internal list of layers whenever the napari layers list
            is updated.
            """
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
    def _update_combobox_options(combobox: QComboBox, options_list: List[str]):
        original_text = combobox.currentText()
        combobox.clear()
        combobox.addItems(options_list)
        combobox.setCurrentText(original_text)

    def _get_layer_names(
        self,
        layer_type: napari.layers.Layer = napari.layers.Image,
        default: str = "",
    ) -> List[str]:
        """
        Get list of layer names of a given layer type.
        """
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
        Construct main layout of widget.
        """
        self.layout = QGridLayout()
        self.layout.setContentsMargins(10, 10, 10, 10)
        self.layout.setAlignment(QtCore.Qt.AlignTop)
        self.layout.setSpacing(4)

        self.add_loading_panel(1)

        self.status_label = QLabel()
        row, col = 7, 0
        self.layout.addWidget(self.status_label, row, col)
        self.update_status_label("Ready")

        self.progress_bar = ProgressBar()
        row, col = 8, 0
        self.layout.addWidget(self.progress_bar.native, row, col)

        self.setLayout(self.layout)

    def add_loading_panel(self, row: int, column: int = 0):

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
        """
        Set signal layer from current signal text box selection.
        """
        if self.signal_image_choice.currentText() != "":
            self.signal_layer = self.viewer.layers[
                self.signal_image_choice.currentText()
            ]

    def set_background_image(self):
        """
        Set background layer from current background text box selection.
        """
        if self.background_image_choice.currentText() != "":
            self.background_layer = self.viewer.layers[
                self.background_image_choice.currentText()
            ]

    def set_training_data_cell(self):
        """
        Set cell training data from current training data text box selection.
        """
        if self.training_data_cell_choice.currentText() != "":
            self.training_data_cell_layer = self.viewer.layers[
                self.training_data_cell_choice.currentText()
            ]
            self.training_data_cell_layer.metadata["point_type"] = Cell.CELL
            self.training_data_cell_layer.metadata["training_data"] = True

    def set_training_data_non_cell(self):
        """
        Set non-cell training data from current training data text box selection.
        """
        if self.training_data_non_cell_choice.currentText() != "":
            self.training_data_non_cell_layer = self.viewer.layers[
                self.training_data_non_cell_choice.currentText()
            ]
            self.training_data_non_cell_layer.metadata[
                "point_type"
            ] = Cell.UNKNOWN
            self.training_data_non_cell_layer.metadata["training_data"] = True

    def add_training_data(self):
        cell_name = "Training data (cells)"
        non_cell_name = "Training data (non cells)"

        overwrite = False
        if self.training_data_cell_layer or self.training_data_non_cell_layer:
            overwrite = display_question(
                self,
                "Training data layers exist",
                "Training data layers already exist,  "
                "overwrite with empty layers?.",
            )
        else:
            if self.training_data_cell_layer:
                self.training_data_cell_layer.remove()
            self._add_training_data_layers(cell_name, non_cell_name)

        if overwrite:
            try:
                self.viewer.layers.remove(cell_name)
                self.viewer.layers.remove(non_cell_name)
            except ValueError:
                pass

            self._add_training_data_layers(cell_name, non_cell_name)

    def _add_training_data_layers(self, cell_name: str, non_cell_name: str):

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
        self.training_data_non_cell_choice.setCurrentText(non_cell_name)

    def mark_as_cell(self):
        self.mark_point_as_type("cell")

    def mark_as_non_cell(self):
        self.mark_point_as_type("non-cell")

    def mark_point_as_type(self, point_type: str):
        if not (
            self.training_data_cell_layer and self.training_data_non_cell_layer
        ):
            show_info(
                "No training data layers have been chosen. "
                "Please add training data layers. ",
            )
            return

        if len(self.viewer.layers.selection) == 1:
            layer = list(self.viewer.layers.selection)[0]
            if type(layer) == napari.layers.Points:

                if len(layer.data) > 0:
                    if point_type == "cell":
                        destination_layer = self.training_data_cell_layer
                    else:
                        destination_layer = self.training_data_non_cell_layer
                    show_info(
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
                    show_info(
                        "No points are selected in the current layer. "
                        "Please select some points.",
                    )

            else:
                show_info(
                    "This is not a points layer. "
                    "Please choose a points layer, and select some points.",
                )
        elif len(self.viewer.layers.selected) == 0:
            show_info(
                "No layers are selected. "
                "Please choose a single points layer, and select some points.",
            )
        else:
            show_info(
                "More than one layer is selected. "
                "Please choose a single points layer, and select some points.",
            )

    def save_training_data(
        self, *, block: bool = False, prompt_for_directory: bool = True
    ) -> None:
        """
        Parameters
        ----------
        block :
            If `True` block execution until all cubes are saved.
        prompt_for_directory :
            If `True` show a file dialog for the user to select a directory.
        """
        if self.is_data_extractable():
            if prompt_for_directory:
                self.get_output_directory()
            if self.output_directory is not None:
                self.__extract_cubes(block=block)
                self.__save_yaml_file()
                show_info("Done")

            self.update_status_label("Ready")

    def __extract_cubes(self, *, block=False):
        """
        Parameters
        ----------
        block :
            If `True` block execution until all cubes are saved.
        """
        self.update_status_label("Extracting cubes")
        self.convert_layers_to_cells()

        if block:
            cubes = self.extract_cubes()
            while True:
                try:
                    next(cubes)
                except StopIteration:
                    break
        else:

            @thread_worker(connect={"yielded": self.update_progress})
            def extract_cubes():
                yield from self.extract_cubes()

            extract_cubes()

    def is_data_extractable(self) -> bool:
        if (
            self.check_training_data_exists()
            and self.check_image_data_for_extraction()
        ):
            return True
        else:
            return False

    def check_image_data_for_extraction(self) -> bool:
        if self.signal_layer and self.background_layer:
            if (
                self.signal_layer.data.shape
                == self.background_layer.data.shape
            ):
                return True
            else:
                show_info(
                    "Please ensure both signal and background images are the "
                    "same size and shape.",
                )
                return False

        else:
            show_info(
                "Please ensure both signal and background images are loaded "
                "into napari, and selected in the sidebar. ",
            )
            return False

    def check_training_data_exists(self) -> bool:
        if not (
            self.training_data_cell_layer or self.training_data_non_cell_layer
        ):
            show_info(
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
                show_info(
                    "No training data points have been added. "
                    "Please annotate some points.",
                )
                return False

    def get_output_directory(self):
        """
        Shows file dialog to choose output directory
        """
        self.update_status_label("Setting output directory...")
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        self.output_directory = QFileDialog.getExistingDirectory(
            self,
            "Select output directory",
            options=options,
        )
        if self.output_directory != "":
            self.output_directory = Path(self.output_directory)
        else:
            self.output_directory = None

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

    def update_progress(self, attributes: dict):
        """
        Update progress bar with ``attributes``.
        """
        for attr in attributes:
            self.progress_bar.__setattr__(attr, attributes[attr])

    def update_status_label(self, label: str):
        self.status_label.setText(label)

    def extract_cubes(self):
        """
        Yields
        ------
        dict
            Attributes used to update a progress bar. The keys can be any of
            the properties of `magicgui.widgets.ProgressBar`.
        """
        from cellfinder_core.classify.cube_generator import (
            CubeGeneratorFromFile,
        )

        to_extract = {
            "cells": self.cells_to_extract,
            "non_cells": self.non_cells_to_extract,
        }

        for cell_type, cell_list in to_extract.items():
            cell_type_output_directory = self.output_directory / cell_type
            cell_type_output_directory.mkdir(exist_ok=True, parents=True)
            self.update_status_label(f"Saving {cell_type}...")

            cube_generator = CubeGeneratorFromFile(
                cell_list,
                self.signal_layer.data,
                self.background_layer.data,
                self.voxel_sizes,
                self.network_voxel_sizes,
                batch_size=self.batch_size,
                cube_width=self.cube_width,
                cube_height=self.cube_height,
                cube_depth=self.cube_depth,
                extract=True,
            )
            # Set up progress bar
            yield {
                "value": 0,
                "min": 0,
                "max": len(cube_generator),
            }

            for i, (image_batch, batch_info) in enumerate(cube_generator):
                image_batch = image_batch.astype(np.int16)

                for point, point_info in zip(image_batch, batch_info):
                    point = np.moveaxis(point, 2, 0)
                    for channel in range(point.shape[-1]):
                        save_cube(
                            point,
                            point_info,
                            channel,
                            cell_type_output_directory,
                        )

                # Update progress bar
                yield {"value": i + 1}

            self.update_status_label("Finished saving cubes")


def save_cube(
    array: np.ndarray, point_info: dict, channel: int, output_directory: Path
):
    filename = (
        f"pCellz{point_info['z']}y{point_info['y']}"
        f"x{point_info['x']}Ch{channel}.tif"
    )
    tifffile.imwrite(output_directory / filename, array[:, :, :, channel])
