import math
import os
from queue import Queue
from typing import Callable, Optional, Sequence

from imlib.cells.cells import Cell
from tifffile import tifffile
from tqdm import tqdm

from cellfinder_core import logger
from cellfinder_core.detect.filters.setup_filters import setup
from cellfinder_core.detect.filters.volume.structure_detection import (
    get_structure_centre_wrapper,
)
from cellfinder_core.detect.filters.volume.structure_splitting import (
    StructureSplitException,
    split_cells,
)


class VolumeFilter(object):
    def __init__(
        self,
        *,
        soma_diameter: float,
        soma_size_spread_factor: float = 1.4,
        setup_params: Sequence,
        planes_paths_range: Sequence,
        save_planes: bool = False,
        plane_directory: Optional[str] = None,
        start_plane: int = 0,
        max_cluster_size: int = 5000,
        outlier_keep: bool = False,
        artifact_keep: bool = True,
    ):
        self.soma_diameter = soma_diameter
        self.soma_size_spread_factor = soma_size_spread_factor
        self.planes_paths_range = planes_paths_range
        self.z = start_plane
        self.save_planes = save_planes
        self.plane_directory = plane_directory
        self.max_cluster_size = max_cluster_size
        self.outlier_keep = outlier_keep

        self.artifact_keep = artifact_keep

        self.clipping_val = None
        self.threshold_value = None
        self.setup_params = setup_params

        self.ball_filter, self.cell_detector = setup(
            self.setup_params[0],
            self.setup_params[1],
            self.setup_params[2],
            self.setup_params[3],
            ball_overlap_fraction=self.setup_params[4],
            z_offset=self.setup_params[5],
        )

    def process(
        self,
        async_result_queue: Queue,
        *,
        callback: Callable[[int], None],
    ):
        progress_bar = tqdm(
            total=len(self.planes_paths_range), desc="Processing planes"
        )
        while not async_result_queue.empty():
            # Get result from the queue.
            #
            # It is important to remove the result from the queue here
            # to free up memory once this plane has been processed by
            # the 3D filter here
            result = async_result_queue.get()
            # .get() blocks until the result is available
            plane, mask = result.get()

            logger.debug(f"Plane {self.z} received for 3D filtering")
            print(f"Plane {self.z} received for 3D filtering")

            logger.debug(f"Adding plane {self.z} for 3D filtering")
            self.ball_filter.append(plane, mask)

            if self.ball_filter.ready:
                logger.debug(f"Ball filtering plane {self.z}")
                self.ball_filter.walk()

                middle_plane = self.ball_filter.get_middle_plane()
                if self.save_planes:
                    self.save_plane(middle_plane)

                logger.debug(f"Detecting structures for plane {self.z}")
                self.cell_detector.process(middle_plane)

                logger.debug(f"Structures done for plane {self.z}")
                logger.debug(
                    f"Skipping plane {self.z} for 3D filter" " (out of bounds)"
                )

            callback(self.z)
            self.z += 1
            progress_bar.update()

        progress_bar.close()
        logger.debug("3D filter done")
        return self.get_results()

    def save_plane(self, plane):
        plane_name = f"plane_{str(self.z).zfill(4)}.tif"
        f_path = os.path.join(self.plane_directory, plane_name)
        tifffile.imsave(f_path, plane.T)

    def get_results(self):
        logger.info("Splitting cell clusters and writing results")

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
    return (4 / 3) * math.pi * radius**3
