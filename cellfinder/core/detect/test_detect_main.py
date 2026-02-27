import numpy as np
import pytest
from brainglobe_utils.cells.cells import Cell

from cellfinder.core.detect.detect import main


def test_main_returns_list_on_zeros():
    """Test main() on a small 3D zero array returns an empty list"""
    arr = np.zeros((2, 4, 4), dtype=np.float32)
    result = main(arr, torch_device="cpu")
    assert isinstance(result, list)
    assert all(isinstance(c, Cell) for c in result) or len(result) == 0


def test_main_raises_on_none_input():
    """Test main() raises ValueError when given None"""
    with pytest.raises(ValueError):
        main(None, torch_device="cpu")


def test_main_raises_on_wrong_dimension():
    """Test main() raises ValueError when array is not 3D"""
    arr = np.zeros((4, 4), dtype=np.float32)  # 2D array
    with pytest.raises(ValueError):
        main(arr, torch_device="cpu")
