import multiprocessing
from dataclasses import dataclass
from multiprocessing.synchronize import Lock as LockBase
from typing import Optional

import numpy as np

from cellfinder_core.detect.filters.plane.classical_filter import enhance_peaks
from cellfinder_core.detect.filters.plane.tile_walker import TileWalker


@dataclass
class MpTileProcessor:
    thread_q: multiprocessing.Queue
    ball_filter_q: multiprocessing.Queue
    clipping_value: float
    threshold_value: float
    soma_diameter: float
    log_sigma_size: float
    n_sds_above_mean_thresh: float

    def get_tile_mask(self, plane: np.ndarray):
        laplace_gaussian_sigma = self.log_sigma_size * self.soma_diameter
        plane = plane.T
        np.clip(plane, 0, self.clipping_value, out=plane)

        walker = TileWalker(plane, self.soma_diameter)

        walker.walk_out_of_brain_only()

        thresholded_img = enhance_peaks(
            walker.thresholded_img,
            self.clipping_value,
            gaussian_sigma=laplace_gaussian_sigma,
        )

        # threshold
        avg = thresholded_img.ravel().mean()
        sd = thresholded_img.ravel().std()

        plane[
            thresholded_img > avg + self.n_sds_above_mean_thresh * sd
        ] = self.threshold_value
        return walker.good_tiles_mask.astype(np.uint8)

    def process(
        self,
        plane_id: int,
        plane: np.ndarray,
        previous_lock: Optional[LockBase],
        self_lock: LockBase,
    ):
        """
        Parameters
        ----------
        previous_lock :
            Lock for the previous tile in the processing queue.
        self_lock :
            Lock for the current tile.
        """
        self_lock.acquire()
        tile_mask = self.get_tile_mask(plane)

        # Wait for previous plane to be done
        if previous_lock is not None:
            previous_lock.acquire()
            previous_lock.release()

        self.ball_filter_q.put((plane_id, plane, tile_mask))
        self.thread_q.put(plane_id)
        self_lock.release()
