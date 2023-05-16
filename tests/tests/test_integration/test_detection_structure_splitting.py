"""
An integration test module for structure splitting

The associated test dataset (exposed here through fixtures) is a minimal
example created to cover the structure splitting code with (modified)
real life data.
"""

import os

import pytest

from cellfinder_core.main import main
from cellfinder_core.tools.IO import read_with_dask

data_dir = os.path.join(
    os.getcwd(), "tests", "data", "integration", "detection"
)
signal_data_path = os.path.join(data_dir, "structure_split_test", "signal")
background_data_path = os.path.join(
    data_dir, "structure_split_test", "background"
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
        n_free_cpus=0,
    )
