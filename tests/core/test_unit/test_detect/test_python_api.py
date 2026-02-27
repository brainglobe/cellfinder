from unittest.mock import Mock

import numpy as np
import pytest

from cellfinder.core.detect import python_api


def test_python_api_calls_detect_main(monkeypatch):
    signal = np.zeros((5, 10, 10), dtype=np.uint16)

    mock_detect = Mock(return_value=[])
    monkeypatch.setattr(python_api, "_detect_main", mock_detect)

    result = python_api.main(signal)

    mock_detect.assert_called_once()
    assert result == []


def test_python_api_raises_on_none_input():
    """
    Test that python_api.main raises TypeError when given None input.
    """
    with pytest.raises(ValueError):
        python_api.main(None)
