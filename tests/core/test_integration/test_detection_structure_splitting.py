"""
An integration test module for structure splitting

The associated test dataset (exposed here through fixtures) is a minimal
example created to cover the structure splitting code with (modified)
real life data.
"""

import numpy as np
import pytest
import torch
from brainglobe_utils.IO.image.load import read_with_dask

from cellfinder.core.detect.filters.setup_filters import DetectionSettings
from cellfinder.core.detect.filters.volume.structure_splitting import (
    ball_filter_imgs,
    split_cells,
)
from cellfinder.core.main import main

voxel_sizes = [5, 2.31, 2.31]


@pytest.fixture
def signal_array(repo_data_path):
    """A signal array that contains a structure that needs splitting."""
    return read_with_dask(
        str(
            repo_data_path
            / "integration"
            / "detection"
            / "structure_split_test"
            / "signal"
        )
    )


@pytest.fixture
def background_array(repo_data_path):
    """A background array that contains a structure that needs splitting."""
    return read_with_dask(
        str(
            repo_data_path
            / "integration"
            / "detection"
            / "structure_split_test"
            / "background"
        )
    )


def test_structure_splitting(signal_array, background_array, no_free_cpus):
    """
    Smoke test to ensure structure splitting code doesn't break.
    """
    main(
        signal_array,
        background_array,
        voxel_sizes,
        n_free_cpus=no_free_cpus,
    )


def test_underflow_issue_435():
    # two cells centered at (9, 10, 10), (19, 10, 10) with radius 5
    p1 = np.array([9, 10, 10])
    p2 = np.array([19, 10, 10])
    radius = 5

    bright_voxels = np.zeros((30, 20, 20), dtype=np.bool_)

    pos = np.empty((30, 20, 20, 3))
    pos[:, :, :, 0] = np.arange(30).reshape((-1, 1, 1))
    pos[:, :, :, 1] = np.arange(20).reshape((1, -1, 1))
    pos[:, :, :, 2] = np.arange(20).reshape((1, 1, -1))

    dist1 = pos - p1.reshape((1, 1, 1, 3))
    dist1 = np.sqrt(np.sum(np.square(dist1), axis=3))
    inside1 = dist1 <= radius
    dist2 = pos - p2.reshape((1, 1, 1, 3))
    dist2 = np.sqrt(np.sum(np.square(dist2), axis=3))
    inside2 = dist2 <= radius

    bright_voxels[np.logical_or(inside1, inside2)] = True
    bright_indices = np.argwhere(bright_voxels)

    settings = DetectionSettings(
        plane_shape=(100, 100),
        plane_original_np_dtype=np.float32,
        voxel_sizes=(1, 1, 1),
        ball_xy_size_um=3,
        ball_z_size_um=3,
        ball_overlap_fraction=0.8,
        soma_diameter_um=7,
    )
    centers, (detector, offset) = split_cells(bright_indices, settings)

    expected = set(map(tuple, [p1.tolist(), p2.tolist()]))
    got = set(map(tuple, centers.tolist()))
    assert expected == got

    assert detector.n_structures == 2


def test_ball_filter_imgs_invalid_volume():
    """Checks that an invalid volume returns empty array instead."""
    settings = DetectionSettings(
        plane_shape=(100, 30),
        plane_original_np_dtype=np.float32,
        voxel_sizes=(1, 1, 1),
        ball_xy_size_um=50,
    )

    cell_detector = ball_filter_imgs(torch.zeros((5, 100, 30)), settings, None)
    assert not cell_detector.n_structures


def test_using_coi_without_intensity():
    cell_points = np.zeros((30, 20, 20), dtype=np.bool_)
    settings = DetectionSettings(
        plane_shape=(100, 100),
        plane_original_np_dtype=np.float32,
        detect_centre_of_intensity=True,
    )

    with pytest.raises(ValueError):
        split_cells(cell_points, settings, None)
