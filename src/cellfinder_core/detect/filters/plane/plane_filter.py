from dataclasses import dataclass
from typing import Tuple

import dask.array as da
import numpy as np

from cellfinder_core.detect.filters.plane.classical_filter import enhance_peaks
from cellfinder_core.detect.filters.plane.tile_walker import TileWalker


@dataclass
class TileProcessor:
    clipping_value: float
    threshold_value: float
    soma_diameter: float
    log_sigma_size: float
    n_sds_above_mean_thresh: float

    def get_tile_mask(self, plane: da.array) -> Tuple[np.ndarray, np.ndarray]:
        """
        This thresholds the input plane, and returns a mask indicating which
        tiles are inside the brain.

        The input plane is:

        1. Clipped to self.threshold value
        2. Run through a peak enhancement filter (see `classical_filter.py`)
        3. Thresholded. Any values that are larger than
           (mean + stddev * sself.n_sds_above_mean_thresh) are set to
           self.threshold_value in-place.

        Parameters
        ----------
        plane :
            Input plane.

        Returns
        -------
        plane :
            Thresholded plane.
        inside_brain_tiles :
            Boolean mask indicating which tiles are inside (1) or
            outside (0) the brain.
        """
        laplace_gaussian_sigma = self.log_sigma_size * self.soma_diameter
        plane = plane.T
        plane = np.clip(plane, 0, self.clipping_value)
        # Read plane from a dask array into memory as a numpy array
        plane = np.array(plane)

        # Get tiles that are within the brain
        walker = TileWalker(plane, self.soma_diameter)
        walker.walk_out_of_brain_only()
        inside_brain_tiles = walker.bright_tiles_mask.astype(np.uint8)

        # Threshold the image
        thresholded_img = enhance_peaks(
            plane.copy(),
            self.clipping_value,
            gaussian_sigma=laplace_gaussian_sigma,
        )
        avg = np.mean(thresholded_img)
        sd = np.std(thresholded_img)
        threshold = avg + self.n_sds_above_mean_thresh * sd
        plane[thresholded_img > threshold] = self.threshold_value

        return plane, inside_brain_tiles
