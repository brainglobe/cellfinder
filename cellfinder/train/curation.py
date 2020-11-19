import napari
import numpy as np
from pathlib import Path
from napari.qt.threading import thread_worker
from qtpy import QtCore

from qtpy.QtWidgets import (
    QLabel,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QWidget,
    QMessageBox,
)

from bg_atlasapi import BrainGlobeAtlas

from brainreg_segment.layout.gui_elements import (
    add_button,
    add_combobox,
)

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

        # general variables
        self.viewer = viewer

        # # Main layers
        # self.base_layer = []  # Contains registered brain / reference brain
        # self.atlas_layer = []  # Contains annotations / region information
        #
        # # Track variables
        # self.track_layers = []
        #
        # # Region variables
        # self.label_layers = []
        #
        # # Atlas variables
        # self.current_atlas_name = ""
        # self.atlas = None
        #
        # self.boundaries_string = boundaries_string
        self.directory = ""
        # # Set up segmentation methods
        # self.region_seg = RegionSeg(self)
        # self.track_seg = TrackSeg(self)

        # Generate main layout
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

        self.load_button = add_button(
            "Load project",
            self.load_data_layout,
            self.get_cellfinder_directory,
            0,
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

    def load_cellfinder_directory(self):
        try:
            self.viewer.open(str(self.directory), plugin="cellfinder")
            # self.paths = Paths(
            #     self.directory,
            #     standard_space=self.standard_space,
            # )
            # self.initialise_loaded_data()
        except ValueError:
            print(
                f"The directory ({self.directory}) does not appear to be "
                f"a cellfinder directory, please try again."
            )
            return




def main():
    print("Loading curation GUI.\n ")
    with napari.gui_qt():
        viewer = napari.Viewer(title="Curation GUI")
        viewer.window.resize(WINDOW_WIDTH, WINDOW_HEIGHT)
        widget = CurationWidget(viewer)
        viewer.window.add_dock_widget(widget, name="Curation", area="right")


if __name__ == "__main__":
    main()
