import math
from typing import Tuple

import torch
import torch.nn.functional as F


class TileWalker:
    """
    A class to segment a 2D image into tiles, and mark each of the
    tiles as bright or dark depending on whether the average image
    value in each tile is above a threshold.

    The threshold is set using the tile of data containing the corner (0, 0).
    The mean and standard deviation of this tile is calculated, and
    the threshold set at 1 + mean + (2 * stddev).

    Parameters
    ----------
    plane_shape : tuple(int, int)
        Height/width of the planes.
    soma_diameter : float
        Diameter of the soma in voxels.
    """

    def __init__(
        self, plane_shape: Tuple[int, int], soma_diameter: int
    ) -> None:

        self.img_height, self.img_width = plane_shape
        self.tile_height = soma_diameter * 2
        self.tile_width = soma_diameter * 2

        self.n_tiles_height = math.ceil(self.img_height / self.tile_height)
        self.n_tiles_width = math.ceil(self.img_width / self.tile_width)

    def get_bright_tiles(self, planes: torch.Tensor) -> torch.Tensor:
        """
        Takes a 3d z-stack. For each z it computes the mean/std of the corner
        tile and uses that to get a in/out of brain threshold for each z.

        Parameters
        ----------
        planes : torch.Tensor
            3d z-stack.

        Returns
        -------
        out_of_brain_thresholds : torch.Tensor
            3d z-stack whose planar shape is the number of tiles in a plane.
            The returned data will be on the same torch device as the input
            planes.
        """
        return _get_bright_tiles(
            planes,
            self.n_tiles_height,
            self.n_tiles_width,
            self.tile_height,
            self.tile_width,
        )

    def get_tiled_buffer(self, depth: int, device: str):
        return torch.zeros(
            (depth, self.n_tiles_height, self.n_tiles_width),
            dtype=torch.bool,
            device=device,
        )


@torch.jit.script
def _get_out_of_brain_threshold(
    planes: torch.Tensor, tile_height: int, tile_width: int
) -> torch.Tensor:
    """
    Takes a 3d z-stack. For each z it computes the mean/std of the corner tile
    and uses that to get a in/out of brain threshold for each z-stack.

    Parameters
    ----------
    planes :
        3d z-stack.
    tile_height :
        Height of each tile.
    tile_width :
        Width of each tile.

    Returns
    -------
    out_of_brain_thresholds :
        1d z-stack.
    """
    # get corner tile
    corner_tiles = planes[:, 0:tile_height, 0:tile_width]
    # convert from ZYX -> ZK, where K is the elements in the corner tile
    corner_tiles = corner_tiles.reshape((planes.shape[0], -1))

    # need to operate in float64, in case the values are large
    corner64 = corner_tiles.type(torch.float64)
    corner_intensity = torch.mean(corner64, dim=1).type(planes.dtype)
    # for parity with past when we used np.std, which defaults to ddof=0
    corner_sd = torch.std(corner64, dim=1, correction=0).type(planes.dtype)
    # add 1 to ensure not 0, as disables
    out_of_brain_thresholds = corner_intensity + 2 * corner_sd + 1

    return out_of_brain_thresholds


@torch.jit.script
def _get_bright_tiles(
    planes: torch.Tensor,
    n_tiles_height: int,
    n_tiles_width: int,
    tile_height: int,
    tile_width: int,
) -> torch.Tensor:
    """
    Loop through the tiles of the plane for each plane. And if the average
    value of a tile is greater than the intensity threshold of that plain,
    mark the tile as bright.
    """
    bright_tiles_mask = torch.zeros(
        (planes.shape[0], n_tiles_height, n_tiles_width),
        dtype=torch.bool,
        device=planes.device,
    )
    # if we don't have enough size for a single tile, it's all outside
    if planes.shape[1] < tile_height or planes.shape[2] < tile_width:
        return bright_tiles_mask

    # for each plane, the threshold
    out_of_brain_thresholds = _get_out_of_brain_threshold(
        planes, tile_height, tile_width
    )
    # thresholds Z -> ZYX shape
    thresholds = out_of_brain_thresholds.view(-1, 1, 1)

    # ZYX -> ZCYX required for function (C=1)
    planes = planes.unsqueeze(1)
    # get the average of each tile
    tile_avg = F.avg_pool2d(
        planes,
        (tile_height, tile_width),
        ceil_mode=False,  # default is False, but to make sure
    )
    # go back from ZCYX -> ZYX
    tile_avg = tile_avg[:, 0, :, :]

    bright = tile_avg >= thresholds
    # tile_avg and bright may be smaller than bright_tiles_mask because
    # avg_pool2d first subtracts the kernel size before computing # tiles.
    # So contain view to that size
    bright_tiles_mask[:, : bright.shape[1], : bright.shape[2]][bright] = True

    return bright_tiles_mask
