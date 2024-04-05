import numpy as np
from scipy.ndimage import gaussian_filter, laplace
from scipy.signal import medfilt2d


def enhance_peaks(
    img: np.ndarray, clipping_value: float, gaussian_sigma: float = 2.5
) -> np.ndarray:
    """
    Enhances the peaks (bright pixels) in an input image.

    Parameters:
    ----------
    img : np.ndarray
        Input image.
    clipping_value : float
        Maximum value for the enhanced image.
    gaussian_sigma : float, optional
        Standard deviation for the Gaussian filter. Default is 2.5.

    Returns:
    -------
    np.ndarray
        Enhanced image with peaks.

    Notes:
    ------
    The enhancement process includes the following steps:
    1. Applying a 2D median filter.
    2. Applying a Laplacian of Gaussian filter (LoG).
    3. Multiplying by -1 (bright spots respond negative in a LoG).
    4. Rescaling image values to range from 0 to clipping value.
    """
    type_in = img.dtype
    filtered_img = medfilt2d(img.astype(np.float64))
    filtered_img = gaussian_filter(filtered_img, gaussian_sigma)
    filtered_img = laplace(filtered_img)
    filtered_img *= -1

    filtered_img -= filtered_img.min()
    filtered_img /= filtered_img.max()

    # To leave room to label in the 3d detection.
    filtered_img *= clipping_value
    return filtered_img.astype(type_in)
