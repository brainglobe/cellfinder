"""
An integration test module for structure splitting

The associated test dataset (exposed here through fixtures) was derived from
"MS_cx_left". It was created by stitching four 370x370 x-y-plane patches from
slices 1000:1021. This was manually determined to be a minimal example of
covering the structure splitting code with real life data.
"""

import os

import pytest

from cellfinder_core.main import main
from cellfinder_core.tools.IO import read_with_dask

data_dir = os.path.join(
    os.getcwd(), "tests", "data", "integration", "detection"
)
signal_data_path = os.path.join(
    data_dir, "MS_cx_left_corners_stitched", "signal"
)
background_data_path = os.path.join(
    data_dir, "MS_cx_left_corners_stitched", "background"
)

voxel_sizes = [5, 2.31, 2.31]


@pytest.fixture
def signal_array():
    """A signal array that contains a structure that needs splitting."""
    return read_with_dask(signal_data_path)


@pytest.fixture
def background_array():
    """A background array that contains a structure that needs splitting."""
    return read_with_dask(background_data_path)


def test_structure_splitting(signal_array, background_array):
    """
    Smoke test to ensure structure splitting code doesn't break.
    """
    main(
        signal_array,
        background_array,
        voxel_sizes,
        n_free_cpus=1,
    )
