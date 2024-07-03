import numpy as np
import pytest
import torch
from brainglobe_utils.IO.cells import get_cells_xml
from brainglobe_utils.IO.image.load import read_with_dask
from pytest_mock.plugin import MockerFixture

try:
    from brainglobe_utils.cells.cells import match_cells
except ImportError:
    match_cells = None

from cellfinder.core.detect.detect import main
from cellfinder.core.tools.IO import fetch_pooch_directory
from cellfinder.core.tools.threading import ExecutionFailure
from cellfinder.core.tools.tools import get_max_possible_int_value

# even though we are testing volume filter as unit test, we are running through
# main and mocking VolumeFilter because that's the easiest way to instantiate
# it with proper args and run it


class ExceptionTest(Exception):
    pass


def load_pooch_dir(test_data_registry, path):
    data_path = fetch_pooch_directory(test_data_registry, path)
    return read_with_dask(data_path)


def raise_exception(*args, **kwargs):
    raise ExceptionTest("Bad times")


def run_main_assert_exception():
    # run volume filter - it should raise the ExceptionTest via
    # ExecutionFailure from thread/process
    try:
        # must be on cpu b/c only on cpu do we do 2d filtering in subprocess
        # lots of planes so it doesn't end naturally quickly
        main(
            signal_array=np.zeros((500, 500, 500), dtype=np.uint16),
            torch_device="cpu",
        )
        assert False, "should have raised exception"
    except ExecutionFailure as e:
        e2 = e.__cause__
        assert type(e2) is ExceptionTest and e2.args == ("Bad times",)


def test_2d_filter_process_exception(mocker: MockerFixture):
    # check sub-process that does 2d filter. That exception ends things clean
    mocker.patch(
        "cellfinder.core.detect.filters.volume.volume_filter._plane_filter",
        new=raise_exception,
    )
    run_main_assert_exception()


def test_2d_filter_feeder_thread_exception(mocker: MockerFixture):
    # check data feeder thread. That exception ends things clean
    from cellfinder.core.detect.filters.volume.volume_filter import (
        VolumeFilter,
    )

    mocker.patch.object(
        VolumeFilter, "_feed_signal_batches", new=raise_exception
    )
    run_main_assert_exception()


def test_2d_filter_cell_detection_thread_exception(mocker: MockerFixture):
    # check cell detection thread. That exception ends things clean
    from cellfinder.core.detect.filters.volume.volume_filter import (
        VolumeFilter,
    )

    mocker.patch.object(
        VolumeFilter, "_run_filter_thread", new=raise_exception
    )
    run_main_assert_exception()


def test_3d_filter_main_thread_exception(mocker: MockerFixture):
    # raises exception in the _process method in the main thread - after the
    # subprocess and secondary threads were spun up. This makes sure that those
    # subprocess and threads don't get stuck if main thread crashes
    from cellfinder.core.detect.filters.volume.volume_filter import (
        VolumeFilter,
    )

    mocker.patch.object(VolumeFilter, "_process", new=raise_exception)
    with pytest.raises(ExceptionTest):
        main(signal_array=np.zeros((500, 500, 500)), torch_device="cpu")


@pytest.mark.parametrize("batch_size", [1, 2, 3, 4])
def test_feeder_thread_batch(batch_size: int):
    # checks various batch sizes to see if there are issues
    # this also tests a batch size of 3 but 5 planes. So the feeder thread
    # will feed us a batch of 3 and a batch of 2. It tests that filters can
    # handle unequal batch sizes
    planes = []

    def callback(z):
        planes.append(z)

    main(
        signal_array=np.zeros((5, 50, 50)),
        torch_device="cpu",
        batch_size=batch_size,
        callback=callback,
    )

    assert planes == list(range(1, 4))


def test_not_enough_planes():
    # checks that even if there are not enough planes for volume filtering, it
    # doesn't raise errors or gets stuck
    planes = []

    def callback(z):
        planes.append(z)

    main(
        signal_array=np.zeros((2, 50, 50)),
        torch_device="cpu",
        callback=callback,
    )

    assert not planes


def test_filtered_plane_range(mocker: MockerFixture):
    # check that even if input data is negative, filtered data is non-negative
    detector = mocker.patch(
        "cellfinder.core.detect.filters.volume.volume_filter.CellDetector",
        autospec=True,
    )

    # input data in range (-500, 500)
    data = ((np.random.random((6, 50, 50)) - 0.5) * 1000).astype(np.float32)
    data[1:3, 25:30, 25:30] = 5000
    main(signal_array=data)

    calls = detector.return_value.process.call_args_list
    assert len(calls)
    for call in calls:
        plane, *_ = call.args
        # should have either zero or soma value or both
        assert len(np.unique(plane)) in (1, 2)
        assert np.min(plane) >= 0


def test_saving_filtered_planes(tmp_path):
    # check that we can save filtered planes
    path = tmp_path / "save_planes"
    path.mkdir()

    main(
        signal_array=np.zeros((6, 50, 50)),
        save_planes=True,
        plane_directory=str(path),
    )

    files = [p.name for p in path.iterdir() if p.is_file()]
    # we're skipping first and last plane that isn't filtered due to kernel
    assert len(files) == 4
    assert set(files) == {
        "plane_0002.tif",
        "plane_0003.tif",
        "plane_0004.tif",
        "plane_0005.tif",
    }


def test_saving_filtered_planes_no_dir():
    # asked to save but didn't provide directory
    with pytest.raises(ExecutionFailure) as exc_info:
        main(
            signal_array=np.zeros((6, 50, 50)),
            save_planes=True,
            plane_directory=None,
        )
    assert type(exc_info.value.__cause__) is ValueError


@pytest.mark.parametrize(
    "signal,filtered,cells,soma_diameter,voxel_sizes,cell_tol",
    [
        (
            "edge_cells_brain/signal",
            "edge_cells_brain/3d_filter",
            "edge_cells_brain/detected_cells.xml",
            16,
            (5, 2, 2),
            0,
        ),
        (
            "bright_brain/signal",
            "bright_brain/3d_filter",
            "bright_brain/detected_cells.xml",
            30,
            (5.06, 4.5, 4.5),
            1,
        ),
    ],
)
@pytest.mark.parametrize(
    "torch_device,use_scipy", [("cpu", False), ("cpu", True), ("cuda", False)]
)
def test_3d_filtering(
    signal,
    filtered,
    cells,
    soma_diameter,
    voxel_sizes,
    cell_tol,
    torch_device,
    use_scipy,
    no_free_cpus,
    tmp_path,
    test_data_registry,
):
    # test that the full 2d/3d matches the saved data
    if torch_device == "cuda" and not torch.cuda.is_available():
        pytest.skip("Cuda is not available")

    # check input data size/type is as expected
    data = np.asarray(load_pooch_dir(test_data_registry, signal))
    filtered = np.asarray(load_pooch_dir(test_data_registry, filtered))
    cells = get_cells_xml(test_data_registry.fetch(cells))
    assert data.dtype == np.uint16
    assert filtered.dtype == np.uint32
    assert data.shape == (filtered.shape[0] + 2, *filtered.shape[1:])

    path = tmp_path / "3d_filter"
    path.mkdir()
    cells_our = main(
        signal_array=data,
        voxel_sizes=voxel_sizes,
        soma_diameter=soma_diameter,
        max_cluster_size=100000,
        ball_xy_size=6,
        ball_z_size=15,
        ball_overlap_fraction=0.6,
        soma_spread_factor=1.4,
        n_free_cpus=no_free_cpus,
        log_sigma_size=0.2,
        n_sds_above_mean_thresh=10,
        save_planes=True,
        plane_directory=str(path),
        batch_size=1,
    )

    filtered_our = np.asarray(read_with_dask(str(path)))
    assert filtered_our.shape == filtered.shape
    assert filtered_our.dtype == np.uint16
    # we need to rescale our data because the original data saved to uint32
    # (even though it fit in uint16), so rescale the max value the soma was
    # saved as, to make comparison better
    filtered_our = filtered_our.astype(np.uint32)
    max16 = get_max_possible_int_value(np.uint16)
    max32 = get_max_possible_int_value(np.uint32)
    filtered_our[filtered_our == max16] = max32

    # we only care about the soma value as only that is used in the next step
    # in cell detection, so set everything else to zero
    filtered_our[filtered_our != max16] = 0
    filtered[filtered_our != max32] = 0

    # the number of pixels per plane that are different (marked bright/not)
    diff = np.sum(np.sum(filtered_our != filtered, axis=2), axis=1)
    # 100% same
    assert np.all(diff == 0)

    # check that the resulting cells are the same. We expect them to be the
    # same, cells at different pos count as different
    cells_our = set(cells_our)
    cells = set(cells)
    diff = len(cells_our - cells) + len(cells - cells_our)
    assert diff <= cell_tol
