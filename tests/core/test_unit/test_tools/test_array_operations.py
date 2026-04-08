import numpy as np
import pytest

from cellfinder.core.tools import array_operations as ops


def _bin_mean_3d_reference(
    arr: np.ndarray, bin_width: int, bin_height: int, bin_depth: int
) -> np.ndarray:
    width_bins = arr.shape[0] // bin_width
    height_bins = arr.shape[1] // bin_height
    depth_bins = arr.shape[2] // bin_depth
    out = np.empty((width_bins, height_bins, depth_bins), dtype=np.float64)

    for i in range(width_bins):
        for j in range(height_bins):
            for k in range(depth_bins):
                x0, x1 = i * bin_width, (i + 1) * bin_width
                y0, y1 = j * bin_height, (j + 1) * bin_height
                z0, z1 = k * bin_depth, (k + 1) * bin_depth
                out[i, j, k] = arr[x0:x1, y0:y1, z0:z1].mean()

    return out


def test_bin_mean_3d_values_and_shape():
    arr = np.arange(4 * 6 * 8, dtype=np.float32).reshape(4, 6, 8)
    result = ops.bin_mean_3d(arr, bin_width=2, bin_height=3, bin_depth=4)
    expected = _bin_mean_3d_reference(arr, 2, 3, 4)

    assert result.shape == (2, 2, 2)
    assert np.allclose(result, expected)


@pytest.mark.parametrize(
    "shape,bin_sizes",
    [
        ((5, 4, 4), (2, 2, 2)),
        ((4, 5, 4), (2, 2, 2)),
        ((4, 4, 5), (2, 2, 2)),
    ],
)
def test_bin_mean_3d_requires_exact_division(shape, bin_sizes):
    arr = np.zeros(shape, dtype=np.float32)
    bw, bh, bd = bin_sizes
    with pytest.raises(ValueError):
        ops.bin_mean_3d(arr, bw, bh, bd)
