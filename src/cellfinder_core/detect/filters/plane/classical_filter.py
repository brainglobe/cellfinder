import numpy as np
from scipy.ndimage import gaussian_filter, laplace
from scipy.signal import medfilt2d


def enhance_peaks(
    img: np.ndarray, clipping_value: float, gaussian_sigma: float = 2.5
) -> np.ndarray:
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
