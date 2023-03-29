import math

import numpy as np

from cellfinder_core.detect.filters.plane.base_tile_filter import (
    OutOfBrainTileFilter,
    is_low_average,
)


class TileWalker:
    """
    A class to segment a 2D image into tiles, and mark each of the
    tiles as good or bad depending on whether the average image
    value in each tile is above a threshold.

    Attributes
    ----------
    good_tiles_mask :
        An array whose entries correspond to each tile.
        The values are set in self.walk_out_of_brain_only()
    """

    def __init__(self, img, soma_diameter):
        self.img = img
        self.img_width, self.img_height = img.shape
        self.soma_diameter = soma_diameter
        self.tile_width = self.soma_diameter * 2
        self.tile_height = self.soma_diameter * 2

        n_tiles_width = math.ceil(self.img_width / self.tile_width)
        n_tiles_height = math.ceil(self.img_height / self.tile_height)
        self.good_tiles_mask = np.zeros(
            (n_tiles_width, n_tiles_height), dtype=bool
        )

        self.x = 0
        self.y = 0
        self.tile_idx = 0

        corner_intensity = img[
            0 : self.tile_width, 0 : self.tile_height
        ].mean()
        corner_sd = img[0 : self.tile_width, 0 : self.tile_height].std()
        # add 1 to ensure not 0, as disables
        out_of_brain_threshold = (corner_intensity + (2 * corner_sd)) + 1

        self.ftf = OutOfBrainTileFilter(
            out_of_brain_intensity_threshold=out_of_brain_threshold
        )

    def _get_tiles(self):  # WARNING: crops to integer steps
        """
        Generator that yields tiles of the 2D image.
        """
        for y in range(
            0, self.img_height - self.tile_height, self.tile_height
        ):
            for x in range(
                0, self.img_width - self.tile_width, self.tile_width
            ):
                tile = self.img[
                    x : x + self.tile_width, y : y + self.tile_height
                ]
                yield x, y, tile

    def walk_out_of_brain_only(self):
        """
        Loop through tiles, and if the average value of a tile is
        greater than the intensity threshold mark the tile as good
        in self.good_tiles_mask.
        """
        threshold = self.ftf.out_of_brain_intensity_threshold
        if threshold == 0:
            return

        for x, y, tile in self._get_tiles():
            self.x = x
            self.y = y
            self.ftf.set_tile(tile)
            self.ftf.keep = not is_low_average(tile, threshold)
            if self.ftf.keep:
                mask_x = self.x // self.tile_width
                mask_y = self.y // self.tile_height
                self.good_tiles_mask[mask_x, mask_y] = True
