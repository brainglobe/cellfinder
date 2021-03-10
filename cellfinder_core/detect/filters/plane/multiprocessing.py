import numpy as np

from cellfinder_core.detect.filters.plane.classical_filter import (
    enhance_peaks,
)
from cellfinder_core.detect.filters.plane.tile_walker import TileWalker


class MpTileProcessor(object):
    def __init__(self, thread_q, ball_filter_q):
        self.thread_q = thread_q
        self.ball_filter_q = ball_filter_q

    def process(
        self,
        plane_id,
        plane,
        previous_lock,
        self_lock,
        clipping_value,
        threshold_value,
        soma_diameter,
        log_sigma_size,
        n_sds_above_mean_thresh,
    ):
        laplace_gaussian_sigma = log_sigma_size * soma_diameter
        plane = plane.T
        np.clip(plane, 0, clipping_value, out=plane)

        walker = TileWalker(plane, soma_diameter)

        walker.walk_out_of_brain_only()

        thresholded_img = enhance_peaks(
            walker.thresholded_img,
            clipping_value,
            gaussian_sigma=laplace_gaussian_sigma,
        )

        # threshold
        avg = thresholded_img.ravel().mean()
        sd = thresholded_img.ravel().std()

        plane[
            thresholded_img > avg + n_sds_above_mean_thresh * sd
        ] = threshold_value
        tile_mask = walker.good_tiles_mask.astype(np.uint8)

        with previous_lock:
            pass
        self.ball_filter_q.put((plane_id, plane, tile_mask))
        self.thread_q.put(plane_id)
        self_lock.release()
