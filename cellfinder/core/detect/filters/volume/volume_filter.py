import math
import os
from queue import Queue
from threading import Lock
from typing import Any, Callable, List, Optional, Tuple

import numpy as np
from brainglobe_utils.cells.cells import Cell
from tifffile import tifffile
from tqdm import tqdm

from cellfinder.core import logger
from cellfinder.core.detect.filters.setup_filters import (
    get_ball_filter,
    get_cell_detector,
)
from cellfinder.core.detect.filters.volume.structure_detection import (
    get_structure_centre,
)
from cellfinder.core.detect.filters.volume.structure_splitting import (
    StructureSplitException,
    split_cells,
)


class VolumeFilter(object):
    def __init__(
        self,
        *,
        soma_diameter: float,
        soma_size_spread_factor: float = 1.4,
        setup_params: Tuple[np.ndarray, Any, int, int, float, Any],
        n_planes: int,
        n_locks_release: int,
        save_planes: bool = False,
        plane_directory: Optional[str] = None,
        start_plane: int = 0,
        max_cluster_size: int = 5000,
        outlier_keep: bool = False,
        artifact_keep: bool = True,
    ):
        self.soma_diameter = soma_diameter
        self.soma_size_spread_factor = soma_size_spread_factor
        self.n_planes = n_planes
        self.z = start_plane
        self.save_planes = save_planes
        self.plane_directory = plane_directory
        self.max_cluster_size = max_cluster_size
        self.outlier_keep = outlier_keep
        self.n_locks_release = n_locks_release

        self.artifact_keep = artifact_keep

        self.clipping_val = None
        self.threshold_value = None
        self.setup_params = setup_params

        self.previous_plane: Optional[np.ndarray] = None

        self.ball_filter = get_ball_filter(
            plane=self.setup_params[0],
            soma_diameter=self.setup_params[1],
            ball_xy_size=self.setup_params[2],
            ball_z_size=self.setup_params[3],
            ball_overlap_fraction=self.setup_params[4],
        )

        self.cell_detector = get_cell_detector(
            plane_shape=self.setup_params[0].shape,  # type: ignore
            ball_z_size=self.setup_params[3],
            z_offset=self.setup_params[5],
        )

    def process(
        self,
        async_result_queue: Queue,
        locks: List[Lock],
        *,
        callback: Callable[[int], None],
    ) -> List[Cell]:
        progress_bar = tqdm(total=self.n_planes, desc="Processing planes")
        for z in range(self.n_planes):
            # Get result from the queue.
            #
            # It is important to remove the result from the queue here
            # to free up memory once this plane has been processed by
            # the 3D filter here
            logger.debug(f"🏐 Waiting for plane {z}")
            result = async_result_queue.get()
            # .get() blocks until the result is available
            plane, mask = result.get()
            logger.debug(f"🏐 Got plane {z}")

            self.ball_filter.append(plane, mask)

            if self.ball_filter.ready:
                # Let the next 2D filter run
                z_release = z + self.n_locks_release + 1
                if z_release < len(locks):
                    logger.debug(f"🔓 Releasing lock for plane {z_release}")
                    locks[z_release].release()

                self._run_filter()

            callback(self.z)
            self.z += 1
            progress_bar.update()

        progress_bar.close()
        logger.debug("3D filter done")
        return self.get_results()

    def _run_filter(self) -> None:
        logger.debug(f"🏐 Ball filtering plane {self.z}")
        self.ball_filter.walk()

        middle_plane = self.ball_filter.get_middle_plane()
        if self.save_planes:
            self.save_plane(middle_plane)

        logger.debug(f"🏫 Detecting structures for plane {self.z}")
        self.previous_plane = self.cell_detector.process(
            middle_plane, self.previous_plane
        )

        logger.debug(f"🏫 Structures done for plane {self.z}")

    def save_plane(self, plane: np.ndarray) -> None:
        if self.plane_directory is None:
            raise ValueError(
                "plane_directory must be set to save planes to file"
            )
        plane_name = f"plane_{str(self.z).zfill(4)}.tif"
        f_path = os.path.join(self.plane_directory, plane_name)
        tifffile.imsave(f_path, plane.T)

    def get_results(self) -> List[Cell]:
        logger.info("Splitting cell clusters and writing results")

        max_cell_volume = sphere_volume(
            self.soma_size_spread_factor * self.soma_diameter / 2
        )

        cells = []

        logger.debug(
            f"Processing {len(self.cell_detector.coords_maps.items())} cells"
        )
        for cell_id, cell_points in self.cell_detector.coords_maps.items():
            cell_volume = len(cell_points)

            if cell_volume < max_cell_volume:
                cell_centre = get_structure_centre(cell_points)
                cells.append(
                    Cell(
                        (
                            cell_centre[0],
                            cell_centre[1],
                            cell_centre[2],
                        ),
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
                                    cell_centre[0],
                                    cell_centre[1],
                                    cell_centre[2],
                                ),
                                Cell.UNKNOWN,
                            )
                        )
                else:
                    cell_centre = get_structure_centre(cell_points)
                    cells.append(
                        Cell(
                            (
                                cell_centre[0],
                                cell_centre[1],
                                cell_centre[2],
                            ),
                            Cell.ARTIFACT,
                        )
                    )

        logger.debug("Finished splitting cell clusters.")
        return cells


def sphere_volume(radius: float) -> float:
    return (4 / 3) * math.pi * radius**3
