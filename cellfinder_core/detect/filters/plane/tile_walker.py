import math

import numpy as np

from cellfinder_core.detect.filters.plane.base_tile_filter import (
    OutOfBrainTileFilter,
)


class TileWalker(object):
    def __init__(self, img, soma_diameter):
        self.img = img
        self.thresholded_img = img.copy()
        self.img_width, self.img_height = img.shape
        self.soma_diameter = soma_diameter
        self.tile_width = self.soma_diameter * 2
        self.tile_height = self.soma_diameter * 2

        n_tiles_width = math.ceil(self.img_width / self.tile_width)
        n_tiles_height = math.ceil(self.img_height / self.tile_height)
        self.good_tiles_mask = np.zeros(
            (n_tiles_width, n_tiles_height), dtype=np.bool
        )

        self.x = 0
        self.y = 0
        self.tile_idx = 0

        corner_intensity = img[
            0 : self.tile_width, 0 : self.tile_height
        ].mean()
        corner_sd = img[0 : self.tile_width, 0 : self.tile_height].std()
        out_of_brain_threshold = (
            corner_intensity + (2 * corner_sd)
        ) + 1  # add 1 to ensure not 0, as disables

        self.ftf = OutOfBrainTileFilter(
            out_of_brain_intensity_threshold=out_of_brain_threshold
        )

    def _get_tiles(self):  # WARNING: crops to integer steps
        for y in range(
            0, self.img_height - self.tile_height, self.tile_height
        ):
            for x in range(
                0, self.img_width - self.tile_width, self.tile_width
            ):
                read_tile = self.img[
                    x : x + self.tile_width, y : y + self.tile_height
                ]
                write_tile = self.thresholded_img[
                    x : x + self.tile_width, y : y + self.tile_height
                ]
                yield x, y, read_tile, write_tile

    def walk_out_of_brain_only(self):
        for x, y, tile, write_tile in self._get_tiles():
            self.x = x
            self.y = y
            self.ftf.set_tile(tile)
            if self.ftf.out_of_brain_intensity_threshold:
                if not self.ftf.is_low_average():
                    mask_x = self.x // self.tile_width
                    mask_y = self.y // self.tile_height
                    self.good_tiles_mask[mask_x, mask_y] = True
