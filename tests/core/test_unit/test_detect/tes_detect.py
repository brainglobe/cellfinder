import numpy as np

from cellfinder.core.detect.detect import main, parse_range


def test_parse_range_full():
    s = parse_range(None, 10)
    assert s.start == 0
    assert s.stop == 10


def test_detect_main_returns_cells():
    arr = np.zeros((3, 10, 10), dtype=np.float32)
    result = main(arr)
    assert isinstance(result, list)
