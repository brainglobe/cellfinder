import os
import math
import logging

from tqdm import tqdm

from imlib.cells.cells import Cell
from tifffile import tifffile

from cellfinder_core.detect.filters.volume.structure_detection import (
    get_structure_centre_wrapper,
)

from cellfinder_core.detect.filters.volume.structure_splitting import (
    split_cells,
    StructureSplitException,
)
from cellfinder_core.detect.filters.setup_filters import setup


class Mp3DFilter(object):
    def __init__(
        self,
        data_queue,
        output_queue,
        soma_diameter,
        soma_size_spread_factor=1.4,
        setup_params=None,
        planes_paths_range=None,
        save_planes=False,
        plane_directory=None,
        start_plane=0,
        max_cluster_size=5000,
        outlier_keep=False,
        artifact_keep=True,
    ):
        self.data_queue = data_queue
        self.output_queue = output_queue
        self.soma_diameter = soma_diameter
        self.soma_size_spread_factor = soma_size_spread_factor
        self.progress_bar = None
        self.planes_paths_range = planes_paths_range
        self.z = start_plane
        self.save_planes = save_planes
        self.plane_directory = plane_directory
        self.max_cluster_size = max_cluster_size
        self.outlier_keep = outlier_keep

        self.artifact_keep = artifact_keep

        self.clipping_val = None
        self.threshold_value = None
        self.ball_filter = None
        self.cell_detector = None
        self.setup_params = setup_params

    def process(self):
        self.progress_bar = tqdm(
            total=len(self.planes_paths_range), desc="Processing planes"
        )

        self.ball_filter, self.cell_detector = setup(
            self.setup_params[0],
            self.setup_params[1],
            self.setup_params[2],
            self.setup_params[3],
            ball_overlap_fraction=self.setup_params[4],
            z_offset=self.setup_params[5],
        )

        while True:
            plane_id, plane, mask = self.data_queue.get()
            logging.debug(f"Plane {plane_id} received for 3D filtering")
            print(f"Plane {plane_id} received for 3D filtering")

            if plane_id is None:
                self.progress_bar.close()
                logging.debug("3D filter done")
                cells = self.get_results()
                self.output_queue.put(cells)
                break

            logging.debug(f"Adding plane {plane_id} for 3D filtering")
            self.ball_filter.append(plane, mask)

            if self.ball_filter.ready:
                logging.debug(f"Ball filtering plane {plane_id}")
                self.ball_filter.walk()

                middle_plane = self.ball_filter.get_middle_plane()
                if self.save_planes:
                    self.save_plane(middle_plane)

                logging.debug(f"Detecting structures for plane {plane_id}")
                self.cell_detector.process(middle_plane)

                logging.debug(f"Structures done for plane {plane_id}")
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

        return cells


def sphere_volume(radius):
    return (4 / 3) * math.pi * radius ** 3
