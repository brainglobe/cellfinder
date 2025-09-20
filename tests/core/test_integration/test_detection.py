import os
from math import isclose

import brainglobe_utils.IO.cells as cell_io
import numpy as np
import pytest
import torch
from brainglobe_utils.cells.cells import Cell
from brainglobe_utils.general.system import get_num_processes
from brainglobe_utils.IO.image.load import read_with_dask

from cellfinder.core.detect.detect import main as detect_main
from cellfinder.core.detect.filters.volume.ball_filter import InvalidVolume
from cellfinder.core.main import main

data_dir = os.path.join(
    os.getcwd(), "tests", "data", "integration", "detection"
)
signal_data_path = os.path.join(data_dir, "crop_planes", "ch0")
background_data_path = os.path.join(data_dir, "crop_planes", "ch1")
cells_validation_xml = os.path.join(data_dir, "cell_classification.xml")

voxel_sizes = [5, 2, 2]
DETECTION_TOLERANCE = 2


class UnixFS:
    @staticmethod
    def rm(filename):
        os.remove(filename)


def test_unix_fs(mocker):
    mocker.patch("os.remove")
    UnixFS.rm("file")
    os.remove.assert_called_once_with("file")


@pytest.fixture
def signal_array():
    return read_with_dask(signal_data_path)


@pytest.fixture
def background_array():
    return read_with_dask(background_data_path)


def count_matched_cells(cell_test, cell_validation, tolerance=0):
    """
    This function is used to check whether the cell's location
    has matches in the validation dataset,
    and counts the number of matched cells.
    """
    matched = 0
    for cell in cell_test:
        for cell_v in cell_validation:
            if (
                abs(cell.x - cell_v.x) <= tolerance
                and abs(cell.y - cell_v.y) <= tolerance
                and abs(cell.z - cell_v.z) <= tolerance
            ):
                matched += 1
                break
    return matched


# FIXME: This isn't a very good example
@pytest.mark.slow
@pytest.mark.parametrize(
    "free_cpus",
    [
        pytest.param("no_free_cpus", id="No free CPUs"),
        pytest.param("run_on_one_cpu_only", id="One CPU"),
    ],
)
def test_detection_full(signal_array, background_array, free_cpus, request):
    n_free_cpus = request.getfixturevalue(free_cpus)
    cells_test = main(
        signal_array,
        background_array,
        voxel_sizes,
        n_free_cpus=n_free_cpus,
    )
    cells_validation = cell_io.get_cells(cells_validation_xml)

    num_non_cells_validation = sum(
        [cell.type == 1 for cell in cells_validation]
    )
    num_cells_validation = sum([cell.type == 2 for cell in cells_validation])

    num_non_cells_test = sum([cell.type == 1 for cell in cells_test])
    num_cells_test = sum([cell.type == 2 for cell in cells_test])
    num_of_matched_cells = count_matched_cells(cells_test, cells_validation)

    assert isclose(
        num_non_cells_validation,
        num_non_cells_test,
        abs_tol=DETECTION_TOLERANCE,
    )
    assert isclose(
        num_cells_validation, num_cells_test, abs_tol=DETECTION_TOLERANCE
    )
    assert num_of_matched_cells >= len(cells_validation) * 0.92, (
        f"Number of matched cells by location is"
        f"{num_of_matched_cells}"
        f"which is less than 92% of the validation cells"
    )


def test_detection_small_planes(
    signal_array,
    background_array,
    no_free_cpus,
    mocker,
):
    # Check that processing works when number of planes < number of processes
    nproc = get_num_processes(no_free_cpus)
    n_planes = 2

    # Don't want to bother classifying in this test, so mock classifcation
    mocker.patch("cellfinder.core.classify.classify.main")

    pytest.mark.skipif(
        nproc < n_planes,
        f"Number of available processes is {nproc}. "
        f"Test only effective if number of available processes >= {n_planes}.",
    )

    main(
        signal_array[0:n_planes],
        background_array[0:n_planes],
        voxel_sizes,
        ball_z_size=5,
        n_free_cpus=no_free_cpus,
    )


def test_callbacks(signal_array, background_array, no_free_cpus):
    # 20 is minimum number of planes needed to find > 0 cells
    signal_array = signal_array[0:20]
    background_array = background_array[0:20]

    planes_done = []
    batches_classified = []
    points_found = []

    def detect_callback(plane):
        planes_done.append(plane)

    def classify_callback(batch):
        batches_classified.append(batch)

    def detect_finished_callback(points):
        points_found.append(points)

    main(
        signal_array,
        background_array,
        voxel_sizes,
        detect_callback=detect_callback,
        classify_callback=classify_callback,
        detect_finished_callback=detect_finished_callback,
        n_free_cpus=no_free_cpus,
        ball_z_size=15,
    )

    skipped_planes = int(round(15 / voxel_sizes[0])) - 1
    skip_start = skipped_planes // 2
    skip_end = skipped_planes - skip_start
    n = len(signal_array) - skip_end
    np.testing.assert_equal(planes_done, np.arange(skip_start, n))
    np.testing.assert_equal(batches_classified, [0])

    ncalls = len(points_found)
    assert ncalls == 1, f"Expected 1 call to callback, got {ncalls}"
    npoints = len(points_found[0])
    assert npoints == 120, f"Expected 120 points, found {npoints}"


@pytest.mark.parametrize("normalize", [True, False])
def test_synthetic_data(synthetic_bright_spots, no_free_cpus, normalize):
    signal_array, background_array = synthetic_bright_spots
    detected = main(
        signal_array,
        background_array,
        voxel_sizes,
        n_free_cpus=no_free_cpus,
        classify_normalize_channels=normalize,
    )
    assert len(detected) == 8


@pytest.mark.parametrize("ndim", [1, 2, 4])
def test_data_dimension_error(ndim):
    # Check for an error when non-3D data input
    shape = (2, 3, 4, 5)[:ndim]
    signal_array = np.random.randint(
        low=0, high=2**16, size=shape, dtype=np.uint32
    )
    background_array = np.random.randint(
        low=0, high=2**16, size=shape, dtype=np.uint32
    )

    with pytest.raises(ValueError, match="Input data must be 3D"):
        main(signal_array, background_array, voxel_sizes)


@pytest.mark.parametrize("device", ["cuda", "cpu"])
@pytest.mark.parametrize(
    "dtype",
    [
        np.uint8,
        np.uint16,
        np.uint32,
        np.int8,
        np.int16,
        np.int32,
        np.float32,
        np.float64,
    ],
)
def test_signal_data_types(synthetic_single_spot, no_free_cpus, dtype, device):

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("Cuda is not available")

    signal_array, background_array, center = synthetic_single_spot
    signal_array = signal_array.astype(dtype)
    # for signed ints, make some data negative
    if np.issubdtype(dtype, np.signedinteger):
        # min of signal_array is zero
        assert np.isclose(0, np.min(signal_array))
        shift = (np.max(signal_array) - np.min(signal_array)) // 2
        signal_array = signal_array - shift

    background_array = background_array.astype(dtype)
    detected = main(
        signal_array,
        background_array,
        n_sds_above_mean_thresh=1.0,
        voxel_sizes=voxel_sizes,
        n_free_cpus=no_free_cpus,
        skip_classification=True,
        torch_device=device,
    )

    assert len(detected) == 1
    assert detected[0] == Cell(center, Cell.UNKNOWN)


@pytest.mark.parametrize("device", ["cuda", "cpu"])
def test_detection_scipy_torch(synthetic_single_spot, no_free_cpus, device):

    if device == "cuda" and not torch.cuda.is_available():
        pytest.xfail("Cuda is not available")

    signal_array, background_array, center = synthetic_single_spot
    signal_array = signal_array.astype(np.float32)

    detected = detect_main(
        signal_array,
        n_sds_above_mean_thresh=1.0,
        voxel_sizes=voxel_sizes,
        n_free_cpus=no_free_cpus,
        torch_device=device,
    )

    assert len(detected) == 1
    assert detected[0] == Cell(center, Cell.UNKNOWN)


@pytest.mark.parametrize("device", ["cuda", "cpu"])
def test_detection_cluster_splitting(
    synthetic_spot_clusters, no_free_cpus, device
):
    """
    Test cluster splitting for overlapping cells.

    Test filtering/detection on cpu and cuda. Because splitting is only on cpu
    so make sure if detection is on cuda, splitting still works.
    """

    if device == "cuda" and not torch.cuda.is_available():
        pytest.xfail("Cuda is not available")

    signal_array, background_array, centers_xyz = synthetic_spot_clusters
    signal_array = signal_array.astype(np.float32)

    detected = detect_main(
        signal_array,
        n_sds_above_mean_thresh=1.0,
        voxel_sizes=voxel_sizes,
        n_free_cpus=no_free_cpus,
        torch_device=device,
    )

    assert len(detected) == len(centers_xyz)
    for cell, center in zip(detected, centers_xyz):
        p = [cell.x, cell.y, cell.z]
        d = np.sqrt(np.sum(np.square(np.subtract(center, p))))
        assert d <= 3
        assert cell.type == Cell.UNKNOWN


def test_detection_cell_too_large(synthetic_spot_clusters, no_free_cpus):
    """
    Test we detect one big artifact if the signal has a too large foreground
    structure.
    """
    # max_cell_volume is volume of soma * spread sphere. For values below
    # radius is 7 pixels. So volume is ~1500 pixels
    signal_array = np.zeros((15, 100, 100), dtype=np.float32)
    # set volume larger than max volume to bright
    signal_array[6 : 6 + 6, 40 : 40 + 26, 40 : 40 + 26] = 1000

    detected = detect_main(
        signal_array,
        n_sds_above_mean_thresh=1.0,
        voxel_sizes=(5, 2, 2),
        soma_diameter=20,
        n_free_cpus=no_free_cpus,
        max_cluster_size=2000 * 5 * 2 * 2,
    )

    assert len(detected) == 1
    # not sure why it subtracts one to center, but probably rounding
    assert detected[0] == Cell([39 + 13, 39 + 13, 5 + 3], Cell.ARTIFACT)


@pytest.mark.parametrize("y,x", [(100, 30), (30, 100)])
def test_detection_plane_too_small(synthetic_spot_clusters, y, x):
    # plane smaller than ball filter kernel should cause error
    with pytest.raises(InvalidVolume):
        detect_main(
            np.zeros((5, y, x)),
            n_sds_above_mean_thresh=1.0,
            voxel_sizes=(1, 1, 1),
            ball_xy_size=50,
        )
