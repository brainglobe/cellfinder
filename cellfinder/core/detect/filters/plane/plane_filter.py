from typing import Tuple

import torch
import torch.nn.functional as F

from cellfinder.core.detect.filters.plane.classical_filter import PeakEnhancer
from cellfinder.core.detect.filters.plane.tile_walker import TileWalker


class TileProcessor:
    """
    Processor that filters each plane to highlight the peaks and also
    tiles and thresholds each plane returning a mask indicating which
    tiles are inside the brain.

    Each input plane is:

    1. Clipped to [0, clipping_value].
    2. Tiled and compared to the corner tile. Any tile that is "bright"
       according to `TileWalker` is marked as being in the brain.
    3. Filtered
       1. Run through the peak enhancement filter (see `PeakEnhancer`)
       2. Thresholded. Any values that are larger than
          (mean + stddev * n_sds_above_mean_thresh) are set to
          threshold_value.

    Parameters
    ----------
    plane_shape : tuple(int, int)
        Height/width of the planes.
    clipping_value : int
        Upper value that the input planes are clipped to. Result is scaled so
        max is this value.
    threshold_value : int
        Value used to mark bright features in the input planes after they have
        been run through the 2D filter.
    n_sds_above_mean_thresh : float
        Number of standard deviations above the mean threshold to use for
        determining whether a voxel is bright.
    log_sigma_size : float
        Size of the Gaussian sigma for the Laplacian of Gaussian filtering.
    soma_diameter : float
        Diameter of the soma in voxels.
    torch_device: str
        The device on which the data and processing occurs on. Can be e.g.
        "cpu", "cuda" etc. Any data passed to the filter must be on this
        device. Returned data will also be on this device.
    dtype : str
        The data-type of the input planes and the type to use internally.
        E.g. "float32".
    use_scipy : bool
        If running on the CPU whether to use the scipy filters or the same
        pytorch filters used on CUDA. Scipy filters can be faster.
    """

    # Upper value that the input plane is clipped to. Result is scaled so
    # max is this value
    clipping_value: int
    # Value used to mark bright features in the input planes after they have
    # been run through the 2D filter
    threshold_value: int
    # voxels who are this many std above mean or more are set to
    # threshold_value
    n_sds_above_mean_thresh: float
    # If used, voxels who are this many or more std above mean of the
    # containing tile as well as above n_sds_above_mean_thresh for the plane
    # average are set to threshold_value.
    n_sds_above_mean_tiled_thresh: float
    # the tile size, in pixels, that will be used to tile the x, y plane when
    # we calculate the per-tile mean / std for use with
    # n_sds_above_mean_tiled_thresh. We use 50% overlap when tiling.
    local_threshold_tile_size_px: int = 0
    # the torch device name
    torch_device: str = ""

    # filter that finds the peaks in the planes
    peak_enhancer: PeakEnhancer = None
    # generates tiles of the planes, with each tile marked as being inside
    # or outside the brain based on brightness
    tile_walker: TileWalker = None

    def __init__(
        self,
        plane_shape: Tuple[int, int],
        clipping_value: int,
        threshold_value: int,
        n_sds_above_mean_thresh: float,
        n_sds_above_mean_tiled_thresh: float,
        tiled_thresh_tile_size: float | None,
        log_sigma_size: float,
        soma_diameter: int,
        torch_device: str,
        dtype: str,
        use_scipy: bool,
    ):
        self.clipping_value = clipping_value
        self.threshold_value = threshold_value
        self.n_sds_above_mean_thresh = n_sds_above_mean_thresh
        self.n_sds_above_mean_tiled_thresh = n_sds_above_mean_tiled_thresh
        if tiled_thresh_tile_size:
            self.local_threshold_tile_size_px = int(
                round(soma_diameter * tiled_thresh_tile_size)
            )
        self.torch_device = torch_device

        laplace_gaussian_sigma = log_sigma_size * soma_diameter
        self.peak_enhancer = PeakEnhancer(
            torch_device=torch_device,
            dtype=getattr(torch, dtype),
            clipping_value=self.clipping_value,
            laplace_gaussian_sigma=laplace_gaussian_sigma,
            use_scipy=use_scipy,
        )

        self.tile_walker = TileWalker(
            plane_shape=plane_shape,
            soma_diameter=soma_diameter,
        )

    def get_tile_mask(
        self, planes: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Applies the filtering listed in the class description.

        Parameters
        ----------
        planes : torch.Tensor
            Input planes (z-stack). Note, the input data is modified.

        Returns
        -------
        planes : torch.Tensor
            Filtered and thresholded planes (z-stack).
        inside_brain_tiles : torch.Tensor
            Boolean mask indicating which tiles are inside (1) or
            outside (0) the brain.
            It's a z-stack whose planes are the shape of the number of tiles
            in each planar axis.
        """
        torch.clip_(planes, 0, self.clipping_value)
        # Get tiles that are within the brain
        inside_brain_tiles = self.tile_walker.get_bright_tiles(planes)
        # Threshold the image
        enhanced_planes = self.peak_enhancer.enhance_peaks(planes)

        _threshold_planes(
            planes,
            enhanced_planes,
            self.n_sds_above_mean_thresh,
            self.n_sds_above_mean_tiled_thresh,
            self.local_threshold_tile_size_px,
            self.threshold_value,
            self.torch_device,
        )

        return planes, inside_brain_tiles

    def get_tiled_buffer(self, depth: int, device: str):
        return self.tile_walker.get_tiled_buffer(depth, device)


@torch.jit.script
def _threshold_planes(
    planes: torch.Tensor,
    enhanced_planes: torch.Tensor,
    n_sds_above_mean_thresh: float,
    n_sds_above_mean_tiled_thresh: float,
    local_threshold_tile_size_px: int,
    threshold_value: int,
    torch_device: str,
) -> None:
    """
    Sets each plane (in-place) to threshold_value, where the corresponding
    enhanced_plane > mean + n_sds_above_mean_thresh*std. Each plane will be
    set to zero elsewhere.
    """
    z, y, x = enhanced_planes.shape

    # ---- get per-plane global threshold ----
    planes_1d = enhanced_planes.view(z, -1)
    # add back last dim
    std, mean = torch.std_mean(planes_1d, dim=1, keepdim=True)
    threshold = mean.unsqueeze(2) + n_sds_above_mean_thresh * std.unsqueeze(2)
    above_global = enhanced_planes > threshold

    # ---- calculate the local tiled threshold ----
    # we do 50% overlap so there's no jumps at boundaries
    stride = local_threshold_tile_size_px // 2
    # make tile even for ease of computation
    tile_size = stride * 2
    # Due to 50% overlap, to get tiles we move the tile by half tile (stride).
    # Total moves will be y // stride - 2 (we start already with mask on first
    # tile). So add back 1 for the first tile. Partial tiles are dropped
    n_y_tiles = max(y // stride - 1, 1) if stride else 1
    n_x_tiles = max(x // stride - 1, 1) if stride else 1
    do_tile_y = n_y_tiles >= 2
    do_tile_x = n_x_tiles >= 2
    # we want at least one axis to have at least two tiles
    if local_threshold_tile_size_px >= 2 and (do_tile_y or do_tile_x):
        # num edge pixels dropped b/c moving by stride would move tile off edge
        y_rem = y % stride
        x_rem = x % stride
        enhanced_planes_raw = enhanced_planes
        if do_tile_y:
            enhanced_planes = enhanced_planes[:, y_rem // 2 :, :]
        if do_tile_x:
            enhanced_planes = enhanced_planes[:, :, x_rem // 2 :]

        # add empty channel dim after z "batch" dim -> zcyx
        enhanced_planes = enhanced_planes.unsqueeze(1)
        # unfold makes it 3 dim, z, M, L. L is number of tiles, M is tile area
        unfolded = F.unfold(
            enhanced_planes,
            (tile_size if do_tile_y else y, tile_size if do_tile_x else x),
            stride=stride,
        )
        # average the tile areas, for each tile
        std, mean = torch.std_mean(unfolded, dim=1, keepdim=True)
        threshold = mean + n_sds_above_mean_tiled_thresh * std

        # reshape it back into Y by X tiles, instead of YX being one dim
        threshold = threshold.reshape((z, n_y_tiles, n_x_tiles))

        # we need total size of n_tiles * stride + stride + rem for the
        # original size. So we add 2 strides and then chop off the excess above
        # rem. We center it because of 50% overlap, the first tile is actually
        # centered in between the first two strides
        offsets = [(0, y), (0, x)]
        for dim, do_tile, n_tiles, n, rem in [
            (1, do_tile_y, n_y_tiles, y, y_rem),
            (2, do_tile_x, n_x_tiles, x, x_rem),
        ]:
            if do_tile:
                repeats = (
                    torch.ones(n_tiles, dtype=torch.int, device=torch_device)
                    * stride
                )
                # add total of 2 additional strides
                repeats[0] = 2 * stride
                repeats[-1] = 2 * stride
                output_size = (n_tiles + 2) * stride

                threshold = threshold.repeat_interleave(
                    repeats, dim=dim, output_size=output_size
                )
                # drop the excess we gained from padding rem to whole stride
                offset = (stride - rem) // 2
                offsets[dim - 1] = offset, n + offset

        # can't use slice(...) objects in jit code so use actual indices
        (a, b), (c, d) = offsets
        threshold = threshold[:, a:b, c:d]

        above_local = enhanced_planes_raw > threshold
        above = torch.logical_and(above_global, above_local)
    else:
        above = above_global

    planes[above] = threshold_value
    # subsequent steps only care about the values that are set to threshold or
    # above in planes. We set values in *planes* to threshold based on the
    # value in *enhanced_planes*. So, there could be values in planes that are
    # at threshold already, but in enhanced_planes they are not. So it's best
    # to zero all other values, so voxels previously at threshold don't count
    planes[torch.logical_not(above)] = 0
