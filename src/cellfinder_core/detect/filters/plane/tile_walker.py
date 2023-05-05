import math
from typing import Generator, Tuple

import numpy as np
from numba import njit


class TileWalker:
    """
    A class to segment a 2D image into tiles, and mark each of the
    tiles as bright or dark depending on whether the average image
    value in each tile is above a threshold.

    The threshold is set using the tile of data containing the corner (0, 0).
    The mean and standard deviation of this tile is calculated, and
    the threshold set at 1 + mean + (2 * stddev).

    Attributes
    ----------
    bright_tiles_mask :
        An boolean array whose entries correspond to whether each tile is
        bright (1) or dark (0). The values are set in
        self.mark_bright_tiles().
    """

    def __init__(self, img: np.ndarray, soma_diameter: int) -> None:
        self.img = img
        self.img_width, self.img_height = img.shape
        self.tile_width = soma_diameter * 2
        self.tile_height = soma_diameter * 2

        n_tiles_width = math.ceil(self.img_width / self.tile_width)
        n_tiles_height = math.ceil(self.img_height / self.tile_height)
        self.bright_tiles_mask = np.zeros(
            (n_tiles_width, n_tiles_height), dtype=bool
        )

        corner_tile = img[0 : self.tile_width, 0 : self.tile_height]
        corner_intensity = np.mean(corner_tile)
        corner_sd = np.std(corner_tile)
        # add 1 to ensure not 0, as disables
        self.out_of_brain_threshold = (corner_intensity + (2 * corner_sd)) + 1

    def _get_tiles(self) -> Generator[Tuple[int, int, np.ndarray], None, None]:
        """
        Generator that yields tiles of the 2D image.

        Notes
        -----
        The final tile in each dimension can have a smaller size than the
        rest of the tiles if the tile shape does not exactly divide the
        image shape.
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

    def mark_bright_tiles(self) -> None:
        """
        Loop through tiles, and if the average value of a tile is
        greater than the intensity threshold mark the tile as bright
        in self.bright_tiles_mask.
        """
        threshold = self.out_of_brain_threshold
        if threshold == 0:
            return

        for x, y, tile in self._get_tiles():
            if not is_low_average(tile, threshold):
                mask_x = x // self.tile_width
                mask_y = y // self.tile_height
                self.bright_tiles_mask[mask_x, mask_y] = True


@njit
def is_low_average(tile: np.ndarray, threshold: float) -> bool:
    """
    Return `True` if the average value of *tile* is below *threshold*.
    """
    avg = np.mean(tile)
    return avg < threshold
