import math
import os
import logging

from imlib.cells.cells import Cell
from imlib.IO.cells import save_cells
from tifffile import tifffile

from cellfinder.detect.filters.volume_filters.structure_detection import (
    get_structure_centre_wrapper,
)

from cellfinder.detect.filters.volume_filters.structure_splitting import (
    split_cells,
    StructureSplitException,
)


class Mp3DFilter(object):
    def __init__(
        self,
        data_queue,
        simple_ball_filter,
        cell_detector,
        soma_diameter,
        output_folder,
        soma_size_spread_factor=1.4,
        progress_bar=None,
        save_planes=False,
        plane_directory=None,
        start_plane=0,
        max_cluster_size=5000,
        outlier_keep=False,
        artifact_keep=True,
        save_csv=False,
    ):
        self.data_queue = data_queue
        self.simple_ball_filter = simple_ball_filter
        self.cell_detector = cell_detector
        self.soma_diameter = soma_diameter
        self.output_folder = output_folder
        self.soma_size_spread_factor = soma_size_spread_factor
        self.progress_bar = progress_bar
        self.z = start_plane
        self.save_planes = save_planes
        self.plane_directory = plane_directory
        self.max_cluster_size = max_cluster_size
        self.outlier_keep = outlier_keep

        self.artifact_keep = artifact_keep

        self.output_file = "cells"
        self.save_csv = save_csv

    def process(self):
        while True:
            plane_id, plane, mask = self.data_queue.get()
            logging.debug(f"Plane {plane_id} received for 3D filtering")

            if plane_id is None:
                self.progress_bar.close()
                logging.debug("3D filter done")
                self.get_results()
                break

            logging.debug(f"Adding plane {plane_id} for 3D filtering")
            self.simple_ball_filter.append(plane, mask)

            if self.simple_ball_filter.ready:
                logging.debug(f"Ball filtering plane {plane_id}")
                self.simple_ball_filter.walk()

                middle_plane = self.simple_ball_filter.get_middle_plane()
                if self.save_planes:
                    self.save_plane(middle_plane)

                logging.debug(f"Detecting structures for plane {plane_id}")
                self.cell_detector.process(middle_plane)

                logging.debug(f"Structures done for plane {plane_id}")

            else:
                logging.debug(
                    f"Skipping plane {plane_id} for 3D filter"
                    " (out of bounds)"
                )

            self.z += 1
            if self.progress_bar is not None:
                self.progress_bar.update()

    def save_plane(self, plane):
        plane_name = f"plane_{str(self.z).zfill(4)}.tif"
        f_path = os.path.join(self.plane_directory, plane_name)
        tifffile.imsave(f_path, plane.T)

    def get_results(self):
        logging.info("Splitting cell clusters and writing results")

        max_cell_volume = sphere_volume(
            self.soma_size_spread_factor * self.soma_diameter / 2
        )

        cells = []
        for (
            cell_id,
            cell_points,
        ) in self.cell_detector.get_coords_list().items():
            cell_volume = len(cell_points)

            if cell_volume < max_cell_volume:
                cell_centre = get_structure_centre_wrapper(cell_points)
                cells.append(
                    Cell(
                        (cell_centre["x"], cell_centre["y"], cell_centre["z"]),
                        Cell.UNKNOWN,
                    )
                )
            else:
                if cell_volume < self.max_cluster_size:
                    try:
                        cell_centres = split_cells(
                            cell_points, outlier_keep=self.outlier_keep
                        )
                    except (ValueError, AssertionError) as err:
                        raise StructureSplitException(
                            f"Cell {cell_id}, error; {err}"
                        )
                    for cell_centre in cell_centres:
                        cells.append(
                            Cell(
                                (
                                    cell_centre["x"],
                                    cell_centre["y"],
                                    cell_centre["z"],
                                ),
                                Cell.UNKNOWN,
                            )
                        )
                else:
                    cell_centre = get_structure_centre_wrapper(cell_points)
                    cells.append(
                        Cell(
                            (
                                cell_centre["x"],
                                cell_centre["y"],
                                cell_centre["z"],
                            ),
                            Cell.ARTIFACT,
                        )
                    )

        xml_file_path = os.path.join(
            self.output_folder, self.output_file + ".xml"
        )
        save_cells(
            cells,
            xml_file_path,
            save_csv=self.save_csv,
            artifact_keep=self.artifact_keep,
        )


def sphere_volume(radius):
    return (4 / 3) * math.pi * radius ** 3
