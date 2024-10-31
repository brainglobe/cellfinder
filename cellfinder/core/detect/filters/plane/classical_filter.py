import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter, laplace
from scipy.signal import medfilt2d


@torch.jit.script
def normalize(
    filtered_planes: torch.Tensor,
    flip: bool,
    max_value: float = 1.0,
) -> None:
    """
    Normalizes the 3d tensor so each z-plane is independently scaled to be
    in the [0, max_value] range. If `flip` is `True`, the sign of the tensor
    values are flipped before any processing.

    It is done to filtered_planes inplace.
    """
    num_z = filtered_planes.shape[0]
    filtered_planes_1d = filtered_planes.view(num_z, -1)

    if flip:
        filtered_planes_1d.mul_(-1)

    planes_min = torch.min(filtered_planes_1d, dim=1, keepdim=True)[0]
    filtered_planes_1d.sub_(planes_min)
    # take max after subtraction
    planes_max = torch.max(filtered_planes_1d, dim=1, keepdim=True)[0]
    # if min = max = zero, divide by 1 - it'll stay zero
    planes_max[planes_max == 0] = 1
    filtered_planes_1d.div_(planes_max)

    if max_value != 1.0:
        # To leave room to label in the 3d detection.
        filtered_planes_1d.mul_(max_value)


@torch.jit.script
def filter_for_peaks(
    planes: torch.Tensor,
    med_kernel: torch.Tensor,
    gauss_kernel: torch.Tensor,
    gauss_kernel_size: int,
    lap_kernel: torch.Tensor,
    device: str,
    clipping_value: float,
) -> torch.Tensor:
    """
    Takes the 3d z-stack and returns a new z-stack where the peaks are
    highlighted.

    It applies a median filter -> gaussian filter -> laplacian filter.
    """
    filtered_planes = planes.unsqueeze(1)  # ZYX -> ZCYX input, C=channels

    # ------------------ median filter ------------------
    # extracts patches to compute median over for each pixel
    # We go from ZCYX -> ZCYX, C=1 to C=9 with C containing the elements around
    # each Z,X,Y voxel over which we compute the median
    # Zero padding is ok here
    filtered_planes = F.conv2d(filtered_planes, med_kernel, padding="same")
    # we're going back to ZCYX=Z1YX by taking median of patches in C dim
    filtered_planes = filtered_planes.median(dim=1, keepdim=True)[0]

    # ------------------ gaussian filter ------------------
    # normalize the input data to 0-1 range. Otherwise, if the values are
    # large, we'd need a float64 so conv result is accurate
    normalize(filtered_planes, flip=False)

    # we need to do reflection padding around the tensor for parity with scipy
    # gaussian filtering. Scipy does reflection in a manner typically called
    # symmetric: (dcba|abcd|dcba). Torch does it like this: (dcb|abcd|cba). So
    # we manually do symmetric padding below
    pad = gauss_kernel_size // 2
    padding_mode = "reflect"
    # if data is too small for reflect, just use constant border value
    if pad >= filtered_planes.shape[-1] or pad >= filtered_planes.shape[-2]:
        padding_mode = "replicate"
    filtered_planes = F.pad(filtered_planes, (pad,) * 4, padding_mode, 0.0)
    # We reflected torch style, so copy/shift everything by one to be symmetric
    filtered_planes[:, :, :pad, :] = filtered_planes[
        :, :, 1 : pad + 1, :
    ].clone()
    filtered_planes[:, :, -pad:, :] = filtered_planes[
        :, :, -pad - 1 : -1, :
    ].clone()
    filtered_planes[:, :, :, :pad] = filtered_planes[
        :, :, :, 1 : pad + 1
    ].clone()
    filtered_planes[:, :, :, -pad:] = filtered_planes[
        :, :, :, -pad - 1 : -1
    ].clone()

    # We apply the 1D gaussian filter twice, once for Y and once for X. The
    # filter shape passed in is 11K1 or 111K, depending on device. Where
    # K=filter size
    # see https://discuss.pytorch.org/t/performance-issue-for-conv2d-with-1d-
    # filter-along-a-dim/201734/2 for the reason for the moveaxis depending
    # on the device
    if device == "cpu":
        # kernel shape is 11K1. First do Y (second to last axis)
        filtered_planes = F.conv2d(
            filtered_planes, gauss_kernel, padding="valid"
        )
        # To do X, exchange X,Y axis, filter, change back. On CPU, Y (second
        # to last) axis is faster.
        filtered_planes = F.conv2d(
            filtered_planes.moveaxis(-1, -2), gauss_kernel, padding="valid"
        ).moveaxis(-1, -2)
    else:
        # kernel shape is 111K
        # First do Y (second to last axis). Exchange X,Y axis, filter, change
        # back. On CUDA, X (last) axis is faster.
        filtered_planes = F.conv2d(
            filtered_planes.moveaxis(-1, -2), gauss_kernel, padding="valid"
        ).moveaxis(-1, -2)
        # now do X, last axis
        filtered_planes = F.conv2d(
            filtered_planes, gauss_kernel, padding="valid"
        )

    # ------------------ laplacian filter ------------------
    # it's a 2d filter. Need to pad using symmetric for scipy parity. But,
    # torch doesn't have it, and we used a kernel of size 3, so for padding of
    # 1, replicate == symmetric. That's enough for parity with past scipy. If
    # we change kernel size in the future, we may have to do as above
    padding = lap_kernel.shape[-1] // 2
    filtered_planes = F.pad(filtered_planes, (padding,) * 4, "replicate")
    filtered_planes = F.conv2d(filtered_planes, lap_kernel, padding="valid")

    # we don't need the channel axis
    filtered_planes = filtered_planes[:, 0, :, :]

    # scale back to full scale, filtered values are negative so flip
    normalize(filtered_planes, flip=True, max_value=clipping_value)
    return filtered_planes


class PeakEnhancer:
    """
    A class that filters each plane in a z-stack such that peaks are
    visualized.

    It uses a series of 2D filters of median -> gaussian ->
    laplacian. Then normalizes each plane to be between [0, clipping_value].

    Parameters
    ----------
    torch_device: str
        The device on which the data and processing occurs on. Can be e.g.
        "cpu", "cuda" etc. Any data passed to the filter must be on this
        device. Returned data will also be on this device.
    dtype : torch.dtype
        The data-type of the input planes and the type to use internally.
        E.g. `torch.float32`.
    clipping_value : int
        The value such that after normalizing, the max value will be this
        clipping_value.
    laplace_gaussian_sigma : float
        Size of the sigma for the gaussian filter.
    use_scipy : bool
        If running on the CPU whether to use the scipy filters or the same
        pytorch filters used on CUDA. Scipy filters can be faster.
    """

    # binary kernel that generates square patches for each pixel so we can find
    # the median around the pixel
    med_kernel: torch.Tensor

    # gaussian 1D kernel with kernel/weight shape 11K1 or 111K, depending
    # on device. Where K=filter size
    gauss_kernel: torch.Tensor

    # 2D laplacian kernel with kernel/weight shape KxK. Where
    # K=filter size
    lap_kernel: torch.Tensor

    # the value such that after normalizing, the max value will be this
    # clipping_value
    clipping_value: float

    # sigma value for gaussian filter
    laplace_gaussian_sigma: float

    # the torch device to run on. E.g. cpu/cuda.
    torch_device: str

    # when running on CPU whether to use pytorch or scipy for filters
    use_scipy: bool

    median_filter_size: int = 3
    """
    The median filter size in x/y direction.

    **Must** be odd.
    """

    def __init__(
        self,
        torch_device: str,
        dtype: torch.dtype,
        clipping_value: float,
        laplace_gaussian_sigma: float,
        use_scipy: bool,
    ):
        super().__init__()
        self.torch_device = torch_device.lower()
        self.clipping_value = clipping_value
        self.laplace_gaussian_sigma = laplace_gaussian_sigma
        self.use_scipy = use_scipy

        # all these kernels are odd in size
        self.med_kernel = self._get_median_kernel(torch_device, dtype)
        self.gauss_kernel = self._get_gaussian_kernel(
            torch_device, dtype, laplace_gaussian_sigma
        )
        self.lap_kernel = self._get_laplacian_kernel(torch_device, dtype)

    @property
    def gaussian_filter_size(self) -> int:
        """
        The gaussian filter 1d size.

        It is odd.
        """
        return 2 * int(round(4 * self.laplace_gaussian_sigma)) + 1

    def _get_median_kernel(
        self, torch_device: str, dtype: torch.dtype
    ) -> torch.Tensor:
        """
        Gets a median patch generator kernel, already on the correct
        device.

        Based on how kornia does it for median filtering.
        """
        # must be odd kernel
        kernel_n = self.median_filter_size
        if not (kernel_n % 2):
            raise ValueError("The median filter size must be odd")

        # extract patches to compute median over for each pixel. When passing
        # input we go from ZCYX -> ZCYX, C=1 to C=9 and containing the elements
        # around each Z,X,Y over which we can then compute the median
        window_range = kernel_n * kernel_n  # e.g. 3x3
        kernel = torch.zeros(
            (window_range, window_range), device=torch_device, dtype=dtype
        )
        idx = torch.arange(window_range, device=torch_device)
        # diagonal of e.g. 9x9 is 1
        kernel[idx, idx] = 1.0
        # out channels, in channels, n*y, n*x. The kernel collects all the 3x3
        # elements around a pixel, using a binary mask for each element, as a
        # separate channel. So we go from 1 to 9 channels in the output
        kernel = kernel.view(window_range, 1, kernel_n, kernel_n)

        return kernel

    def _get_gaussian_kernel(
        self,
        torch_device: str,
        dtype: torch.dtype,
        laplace_gaussian_sigma: float,
    ) -> torch.Tensor:
        """Gets the 1D gaussian kernel used to filter the data."""
        # we do 2 1D filters, once on each y, x dim.
        # shape of kernel will be 11K1 with dims Z, C, Y, X. C=1, Z is expanded
        # to number of z during filtering.
        kernel_size = self.gaussian_filter_size

        # to get the values of a 1D gaussian kernel, we pass a single impulse
        # data through the filter, which recovers the filter values. We do this
        # because scipy doesn't make their kernel available in public API and
        # we want parity with scipy filtering
        impulse = np.zeros(kernel_size)
        # the impulse needs to be to the left of center
        impulse[kernel_size // 2] = 1
        kernel = gaussian_filter(
            impulse, laplace_gaussian_sigma, mode="constant", cval=0
        )
        # kernel should be fully symmetric
        assert kernel[0] == kernel[-1]
        gauss_kernel = torch.from_numpy(kernel).type(dtype).to(torch_device)

        # default shape is (y, x) with y axis filtered only - we transpose
        # input to filter on x
        gauss_kernel = gauss_kernel.view(1, 1, -1, 1)

        # see https://discuss.pytorch.org/t/performance-issue-for-conv2d-
        # with-1d-filter-along-a-dim/201734. Conv2d is faster on a specific dim
        # for 1D filters depending on CPU/CUDA. See also filter_for_peaks
        # on CPU, we only do conv2d on the (1st) dim
        if torch_device != "cpu":
            # on CUDA, we only filter on the x dim, flipping input to filter y
            gauss_kernel = gauss_kernel.view(1, 1, 1, -1)

        return gauss_kernel

    def _get_laplacian_kernel(
        self, torch_device: str, dtype: torch.dtype
    ) -> torch.Tensor:
        """Gets a 2d laplacian kernel, based on scipy's laplace."""
        # for parity with scipy, scipy computes the laplacian with default
        # parameters and kernel size 3 using filter coefficients [1, -2, 1].
        # Each filtered pixel is the sum of the filter around the pixel
        # vertically and horizontally. We can do it in 2d at once with
        # coefficients below (faster than 2x1D for such small filter)
        return torch.as_tensor(
            [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
            dtype=dtype,
            device=torch_device,
        ).view(1, 1, 3, 3)

    def enhance_peaks(self, planes: torch.Tensor) -> torch.Tensor:
        """
        Applies the filtering and normalization to the 3d z-stack (not inplace)
        and returns the filtered z-stack.
        """
        if self.torch_device == "cpu" and self.use_scipy:
            filtered_planes = planes.clone()
            for i in range(planes.shape[0]):
                img = planes[i, :, :].numpy()
                img = medfilt2d(img)
                img = gaussian_filter(img, self.laplace_gaussian_sigma)
                img = laplace(img)
                filtered_planes[i, :, :] = torch.from_numpy(img)

            # laplace makes values negative so flip
            normalize(
                filtered_planes,
                flip=True,
                max_value=self.clipping_value,
            )
            return filtered_planes

        filtered_planes = filter_for_peaks(
            planes,
            self.med_kernel,
            self.gauss_kernel,
            self.gaussian_filter_size,
            self.lap_kernel,
            self.torch_device,
            self.clipping_value,
        )
        return filtered_planes
