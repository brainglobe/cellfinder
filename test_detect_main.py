# test_detect_main.py

import numpy as np
import pytest
from cellfinder.core.detect.detect import main as detect_main

def test_detect_returns_list():
    # create a dummy 3D array
    arr = np.zeros((3, 5, 5), dtype=np.float32)
    result = detect_main(arr)
    assert isinstance(result, list)

def test_detect_raises_type_error_on_wrong_dtype():
    arr = np.zeros((3, 5, 5), dtype=object)
    with pytest.raises(TypeError):
        detect_main(arr)

def test_detect_raises_value_error_on_none_input():
    with pytest.raises(ValueError):
        detect_main(None)