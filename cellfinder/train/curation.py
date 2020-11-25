import json
import napari

import numpy as np

from pathlib import Path
from qtpy import QtCore


from bg_atlasapi import BrainGlobeAtlas
from imlib.IO.cells import save_cells
from imlib.cells.cells import Cell
from napari_cellfinder.cellfinder import get_cell_arrays
from brainreg.paths import Paths as BrainregPaths
from brainreg_segment.layout.gui_elements import (
    add_button,
)
from qtpy.QtWidgets import (
    QLabel,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QWidget,
)


from cellfinder.main import get_downsampled_space
from cellfinder.analyse.analyse import run_analysis

from imlib.general.system import get_sorted_file_paths
from napari.utils.io import magic_imread


# Constants used throughout
WINDOW_HEIGHT = 750
WINDOW_WIDTH = 1500
COLUMN_WIDTH = 150


class CurationWidget(QWidget):
    def __init__(
        self,
        viewer,
    ):
        super(CurationWidget, self).__init__()

        self.viewer = viewer

        self.background_layer = []
        self.background_path = ""

        self.signal_layer = []
        self.signal_path = ""

        self.directory = ""
        self.output_directory = None

        self.setup_main_layout()

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
        self.layout.addWidget(self.status_label, 5, 0)

        self.setLayout(self.layout)

    def add_loading_panel(self, row, column=0):
        """
        Loading panel:
            - Load project (sample space)
            - Load project (atlas space)
            - Atlas chooser
        """
        self.load_data_panel = QGroupBox("Load data")
        self.load_data_layout = QGridLayout()
        self.load_data_layout.setSpacing(15)
        self.load_data_layout.setContentsMargins(10, 10, 10, 10)
        self.load_data_layout.setAlignment(QtCore.Qt.AlignBottom)

        # self.load_cellfinder_dir_button = add_button(
        #     "Load cellfinder project",
        #     self.load_data_layout,
        #     self.get_cellfinder_directory,
        #     0,
        #     0,
        #     minimum_width=COLUMN_WIDTH,
        # )

        # self.load_background_button = add_button(
        #     "Load background",
        #     self.load_data_layout,
        #     self.get_background,
        #     1,
        #     0,
        #     minimum_width=COLUMN_WIDTH,
        # )

        self.load_signal_button = add_button(
            "Load signal",
            self.load_data_layout,
            self.get_signal,
            2,
            0,
            minimum_width=COLUMN_WIDTH,
        )

        self.add_cells_button = add_button(
            "Add cell count",
            self.load_data_layout,
            self.add_cell_count,
            3,
            0,
            minimum_width=COLUMN_WIDTH,
        )

        self.add_cells_button = add_button(
            "Load cell count",
            self.load_data_layout,
            self.load_cells,
            4,
            0,
            minimum_width=COLUMN_WIDTH,
        )

        self.save_cells_button = add_button(
            "Save cells",
            self.load_data_layout,
            self.save_cell_count,
            5,
            0,
            minimum_width=COLUMN_WIDTH,
        )
        self.analyse_cells_button = add_button(
            "Analyse cells",
            self.load_data_layout,
            self.analyse_cells,
            6,
            0,
            minimum_width=COLUMN_WIDTH,
        )

        self.load_data_layout.setColumnMinimumWidth(0, COLUMN_WIDTH)
        self.load_data_panel.setLayout(self.load_data_layout)
        self.load_data_panel.setVisible(True)
        self.layout.addWidget(self.load_data_panel, row, column, 1, 1)

    def get_cellfinder_directory(self):
        self.status_label.setText("Loading...")
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        cellfinder_directory = QFileDialog.getExistingDirectory(
            self,
            "Select cellfinder directory",
            options=options,
        )

        if not cellfinder_directory:
            return

        if self.directory != cellfinder_directory:
            self.directory = Path(cellfinder_directory)
        else:
            print(f"{str(cellfinder_directory)} already loaded.")
            return

        # Otherwise, proceed loading brainreg dir
        self.load_cellfinder_directory()
        self.status_label.setText("Ready...")

    def load_cellfinder_directory(self):
        try:
            self.viewer.open(str(self.directory), plugin="cellfinder")
        except ValueError:
            print(
                f"The directory ({self.directory}) does not appear to be "
                f"a cellfinder directory, please try again."
            )
            return

    def get_background(self):
        self.background_layer, self.background_path = self.get_data(
            name="background", visible=False
        )

    def get_signal(self):
        self.signal_layer, self.signal_path = self.get_data(name="signal")
        self.status_label.setText("Ready")

    def get_data(self, name="", visible=True):
        self.status_label.setText("Loading...")
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        directory = QFileDialog.getExistingDirectory(
            self,
            f"Select {name} channel",
            options=options,
        )

        if not directory:
            return

        if self.directory != directory:
            self.directory = Path(directory)
        else:
            print(f"{str(directory)} already loaded.")
            return

        return self.load_data(name=name, visible=visible), directory

    def load_data(self, name="", visible=True):
        try:
            img_paths = get_sorted_file_paths(
                self.directory, file_extension=".tif"
            )
            images = magic_imread(img_paths, use_dask=True, stack=True)
            return self.viewer.add_image(images)
            # return self.viewer.open(
            #     str(self.directory), name=name, visible=visible
            # )
        except ValueError:
            print(
                f"The directory ({self.directory}) cannot be "
                f"loaded, please try again."
            )
            return

    def add_cell_count(self):
        self.cell_layer = self.viewer.add_points(
            np.empty((0, 3)),
            symbol="ring",
            n_dimensional=True,
            size=10,
            opacity=0.6,
            face_color="lightgoldenrodyellow",
            name="cells",
        )

    def load_cells(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        filename = QFileDialog.getOpenFileName(
            self,
            "Select cell file...",
            filter="cellfinder xml files (*.xml)",
            options=options,
        )
        cells, _ = get_cell_arrays(filename[0])

        self.cell_layer = self.viewer.add_points(
            cells,
            symbol="ring",
            n_dimensional=True,
            size=10,
            opacity=0.6,
            face_color="lightgoldenrodyellow",
            name="cells",
        )

    def save_cell_count(self):
        self.status_label.setText("Saving cells")
        print("Saving cells")
        self.get_output_directory()
        filename = self.output_directory / "cells.xml"

        cells_to_save = []
        for idx, point in enumerate(self.cell_layer.data):
            cell = Cell([point[2], point[1], point[0]], Cell.CELL)
            cells_to_save.append(cell)

        save_cells(cells_to_save, str(filename))

        self.status_label.setText("Ready")
        print("Done!")

    def get_output_directory(self):
        if self.output_directory is None:
            options = QFileDialog.Options()
            options |= QFileDialog.DontUseNativeDialog
            self.output_directory = QFileDialog.getExistingDirectory(
                self,
                "Select output directory",
                options=options,
            )
            self.output_directory = Path(self.output_directory)

    def analyse_cells(self):
        self.status_label.setText("Analysing cells")
        print("Analysing cells")

        self.get_output_directory()
        self.get_brainreg_directory()

        analyse_cell_positions(
            self.cell_layer.data,
            self.brainreg_directory,
            self.signal_path,
            self.output_directory,
        )
        self.status_label.setText("Ready")

    def get_brainreg_directory(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        self.brainreg_directory = QFileDialog.getExistingDirectory(
            self,
            "Select brainreg directory",
            options=options,
        )


def analyse_cell_positions(
    points, brainreg_directory, signal_path, output_directory
):
    brainreg_paths = BrainregPaths(brainreg_directory)

    with open(brainreg_paths.metadata_path) as json_file:
        metadata = json.load(json_file)

    atlas = BrainGlobeAtlas(metadata["atlas"])

    downsampled_space = get_downsampled_space(
        atlas, brainreg_paths.boundaries_file_path
    )
    deformation_field_paths = [
        brainreg_paths.deformation_field_0,
        brainreg_paths.deformation_field_1,
        brainreg_paths.deformation_field_2,
    ]

    run_analysis(
        points,
        signal_path,
        metadata["orientation"],
        metadata["voxel_sizes"],
        atlas,
        deformation_field_paths,
        downsampled_space,
        output_directory / "downsampled.points",
        output_directory / "atlas.points",
        output_directory / "points.npy",
        brainreg_paths.volume_csv_path,
        output_directory / "all_points.csv",
        output_directory / "summary.csv",
    )

    print("Done")


def main():
    print("Loading curation GUI.\n ")
    with napari.gui_qt():
        viewer = napari.Viewer(title="Curation GUI")
        viewer.window.resize(WINDOW_WIDTH, WINDOW_HEIGHT)
        widget = CurationWidget(viewer)
        viewer.window.add_dock_widget(widget, name="Curation", area="right")


if __name__ == "__main__":
    main()
