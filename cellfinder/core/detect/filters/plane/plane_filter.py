from dataclasses import dataclass
from threading import Lock
from typing import Optional, Tuple

import dask.array as da
import numpy as np

from cellfinder.core import types
from cellfinder.core.detect.filters.plane.classical_filter import enhance_peaks
from cellfinder.core.detect.filters.plane.tile_walker import TileWalker


@dataclass
class TileProcessor:
    """
    Attributes
    ----------
    clipping_value :
        Upper value that the input plane is clipped to.
    threshold_value :
        Value used to mark bright features in the input planes after they have
        been run through the 2D filter.
    """

    clipping_value: int
    threshold_value: int
    soma_diameter: int
    log_sigma_size: float
    n_sds_above_mean_thresh: float

    def get_tile_mask(
        self, plane: types.array, lock: Optional[Lock] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        This thresholds the input plane, and returns a mask indicating which
        tiles are inside the brain.

        The input plane is:

        1. Clipped to [0, self.clipping_value]
        2. Run through a peak enhancement filter (see `classical_filter.py`)
        3. Thresholded. Any values that are larger than
           (mean + stddev * self.n_sds_above_mean_thresh) are set to
           self.threshold_value in-place.

        Parameters
        ----------
        plane :
            Input plane.
        lock :
            If given, block reading the plane into memory until the lock
            can be acquired.

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
        np.clip(plane, 0, self.clipping_value, out=plane)
        if lock is not None:
            lock.acquire(blocking=True)
        # Read plane from a dask array into memory as a numpy array
        if isinstance(plane, da.Array):
            plane = np.array(plane)

        # Get tiles that are within the brain
        walker = TileWalker(plane, self.soma_diameter)
        walker.mark_bright_tiles()
        inside_brain_tiles = walker.bright_tiles_mask

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
