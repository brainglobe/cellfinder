from typing import Sequence

import numpy as np
import pytest
import tifffile
import torch
from brainglobe_utils.cells.cells import Cell
from pytest_mock.plugin import MockerFixture
from torch.utils.data import DataLoader

from cellfinder.core.classify.cube_generator import (
    CachedArrayStackImageData,
    CachedCuboidImageDataBase,
    CachedStackImageDataBase,
    CachedTiffCuboidImageData,
    CuboidBatchSampler,
    CuboidDatasetBase,
    CuboidStackDataset,
    CuboidTiffDataset,
    ImageDataBase,
    get_data_cuboid_range,
)
from cellfinder.core.tools.threading import ExecutionFailure

try:
    from brainglobe_utils.cells.cells import file_name_from_cell
except ImportError:

    def file_name_from_cell(cell: Cell, channel: int) -> str:
        name = f"x{int(cell.x)}_y{int(cell.y)}_z{int(cell.z)}Ch{channel}.tif"
        return name


PT_TYPE = tuple[int, int, int]


unique_int_val = 0


@pytest.fixture(scope="function")
def unique_int() -> int:
    """Returns a unique int every time called."""
    global unique_int_val
    unique_int_val += 1
    return unique_int_val


def sample_volume(x: int, y: int, z: int, c: int, seed: int) -> np.ndarray:
    """
    Returns numpy volume with the given size using the given seed to make it
    unique.
    """
    x, y, z, c = (
        x,
        y,
        z,
        c,
    )
    data = np.arange(x * y * z * c) + seed
    data = data.reshape((x, y, z, c)).astype(np.uint16)
    return data


def point_to_slice(
    point: PT_TYPE,
    cube_size: PT_TYPE,
) -> list[slice]:
    """Returns slices to index volume to get cube around point."""
    slices = []
    for p, s, ax in zip(point, cube_size, ["x", "y", "z"]):
        slices.append(slice(*get_data_cuboid_range(p, s, ax)))
    return slices


def to_numpy_cubes(
    volume: np.ndarray, points: Sequence[PT_TYPE], cube_size: PT_TYPE
) -> tuple[list[np.ndarray], np.ndarray]:
    """Extracts numpy cubes around the points in the volume."""
    points_arr = np.empty(
        len(points), dtype=[("x", "<f8"), ("y", "<f8"), ("z", "<f8")]
    )
    cubes = []
    for i, point in enumerate(points):
        points_arr[i] = tuple(point)
        cube = volume[tuple(point_to_slice(point, cube_size))]
        cubes.append(cube)
    return cubes, points_arr


def to_tiff_cubes(
    volume: np.ndarray, cube_size: PT_TYPE, points: Sequence[PT_TYPE], tmp_path
) -> tuple[list[Sequence[str]], list[np.ndarray], np.ndarray]:
    """Creates tiff files for the cubes around the points in the volume."""
    cubes, points_arr = to_numpy_cubes(volume, points, cube_size)
    # create the tiff files, one per channel per point
    filenames = []
    for (x, y, z), cube in zip(points, cubes):
        # don't have to center cube on point, just get a unique cube
        sig = tmp_path / file_name_from_cell(
            Cell([x, y, z], Cell.UNKNOWN), channel=0
        )
        tifffile.imwrite(sig, cube[:, :, :, 0])

        back = tmp_path / file_name_from_cell(
            Cell([x, y, z], Cell.UNKNOWN), channel=1
        )
        tifffile.imwrite(back, cube[:, :, :, 1])

        filenames.append((str(sig), str(back)))

    return filenames, cubes, points_arr


def assert_loader_cubes_matches_cubes(
    cubes: Sequence[np.ndarray],
    data_loader: ImageDataBase,
    batches: Sequence[Sequence[int]],
) -> None:
    """
    Checks that the provided cubes from the input data are the same as the
    cubes we get from the data loader.
    """
    cubes = [torch.from_numpy(cube) for cube in cubes]
    for batch in batches:
        for i in batch:
            # compare each cube with cube from data loader
            assert torch.equal(cubes[i], data_loader.get_point_cuboid_data(i))
        batch_cubes = [cubes[i][None, ...] for i in batch]

        # compare batch of cubes with cubes from data loader
        cube_batch = torch.concatenate(batch_cubes, 0)
        loader_batch = torch.zeros_like(cube_batch)
        data_loader.get_point_batch_cuboid_data(loader_batch, batch)
        assert torch.equal(cube_batch, loader_batch)


def assert_loader_cubes_bad_indices(
    data_shape: Sequence[int],
    data_loader: ImageDataBase,
    individuals: Sequence[int],
    batches: Sequence[Sequence[int]],
) -> None:
    """Checks that trying to access cubes beyond valid cubes fails."""
    for i in individuals:
        # check individual cubes indices that are out of bounds
        with pytest.raises(IndexError):
            data_loader.get_point_cuboid_data(i)

    for batch in batches:
        # check batches that have indices that are out of bounds
        loader_batch = torch.empty(data_shape, dtype=torch.uint16)
        with pytest.raises(IndexError):
            data_loader.get_point_batch_cuboid_data(loader_batch, batch)


def assert_loader_cubes_bad_size(
    data_shape: Sequence[int],
    data_loader: ImageDataBase,
    batch: Sequence[int],
) -> None:
    """Checks that providing a wrong sized buffer for cube batches fails."""
    # for batches, we need to provide buffer. Check buffers with wrong size
    for i in range(1, len(data_shape)):
        shape = list(data_shape)
        # increase that dim size by one
        shape[i] += 1

        loader_batch = torch.empty(shape, dtype=torch.uint16)
        with pytest.raises(ValueError):
            data_loader.get_point_batch_cuboid_data(loader_batch, batch)


def test_array_image_data(unique_int):
    """
    Checks that the data returned by the CachedArrayStackImageData for given
    points matches the data it should return.
    """
    volume = sample_volume(20, 20, 30, 2, unique_int)
    points = [(5, 5, 10), (10, 10, 20)]
    cube_size = 3, 3, 5
    cubes, points_arr = to_numpy_cubes(volume, points, cube_size)

    stack = CachedArrayStackImageData(
        input_arrays=[volume[:, :, :, 0], volume[:, :, :, 1]],
        max_axis_0_cuboids_buffered=3,
        data_axis_order=("x", "y", "z"),
        cuboid_size=cube_size,
        points_arr=points_arr,
    )

    assert stack.cuboid_with_channels_size == (*cube_size, 2)
    assert stack.data_with_channels_axis_order == ("x", "y", "z", "c")

    assert_loader_cubes_matches_cubes(
        cubes, stack, [[0, 1], [0], [1], [0], [1, 0], [1, 1]]
    )
    assert_loader_cubes_bad_indices((2, *cube_size, 2), stack, [2], [[1, 2]])
    assert_loader_cubes_bad_size((2, *cube_size, 2), stack, [0, 1])


def test_tiff_image_data(unique_int, tmp_path):
    """
    Checks that the data returned by the CachedTiffCuboidImageData for given
    points matches the data it should return.
    """
    volume = sample_volume(20, 20, 30, 2, unique_int)
    points = [(5, 5, 10), (10, 10, 20)]
    cube_size = 3, 3, 5
    filenames, cubes, points_arr = to_tiff_cubes(
        volume, cube_size, points, tmp_path
    )

    tiffs = CachedTiffCuboidImageData(
        filenames_arr=np.array(filenames).astype(np.str_),
        max_cuboids_buffered=3,
        data_axis_order=("x", "y", "z"),
        cuboid_size=cube_size,
        points_arr=points_arr,
    )

    assert tiffs.cuboid_with_channels_size == (*cube_size, 2)
    assert tiffs.data_with_channels_axis_order == ("x", "y", "z", "c")

    assert_loader_cubes_matches_cubes(
        cubes, tiffs, [[0, 1], [0], [1], [0], [1, 0], [1, 1]]
    )
    assert_loader_cubes_bad_indices((2, *cube_size, 2), tiffs, [2], [[1, 2]])
    assert_loader_cubes_bad_size((2, *cube_size, 2), tiffs, [0, 1])


@pytest.mark.parametrize("cached", [0, 1])
def test_array_image_data_cache(unique_int, cached, mocker: MockerFixture):
    """Checks that CachedArrayStackImageData properly caches the first axis."""
    volume = sample_volume(20, 20, 30, 2, unique_int)
    points = [(5, 5, 10), (10, 10, 20)]
    cube_size = 3, 3, 5
    cubes, points_arr = to_numpy_cubes(volume, points, cube_size)

    stack = CachedArrayStackImageData(
        input_arrays=[volume[:, :, :, 0], volume[:, :, :, 1]],
        max_axis_0_cuboids_buffered=cached,
        data_axis_order=("x", "y", "z"),
        cuboid_size=cube_size,
        points_arr=points_arr,
    )

    spy = mocker.spy(stack, "read_plane")
    stack.get_point_cuboid_data(0)
    stack.get_point_cuboid_data(1)
    stack.get_point_cuboid_data(0)
    batch = torch.zeros((2, *cube_size, 2))
    stack.get_point_batch_cuboid_data(batch, [0, 1])

    # number of planes per cache dim cube
    n = cube_size[0]
    match cached:
        # cache is always in addition to one cube
        case 0:
            # only one cube is ever cached in memory. Plus 2 channels
            assert spy.call_count == (n + n + n + n) * 2
        case 1:
            # should only ever read each plane once as there's enough cache
            assert spy.call_count == n * 2 * 2


@pytest.mark.parametrize("cached", [0, 1])
def test_tiff_image_data_cache(
    unique_int, tmp_path, cached, mocker: MockerFixture
):
    """Checks that CachedTiffCuboidImageData properly caches the first axis."""
    volume = sample_volume(20, 20, 30, 2, unique_int)
    points = [(5, 5, 10), (10, 10, 20)]
    cube_size = 3, 3, 5
    filenames, cubes, points_arr = to_tiff_cubes(
        volume, cube_size, points, tmp_path
    )

    tiffs = CachedTiffCuboidImageData(
        filenames_arr=np.array(filenames).astype(np.str_),
        max_cuboids_buffered=cached,
        data_axis_order=("x", "y", "z"),
        cuboid_size=cube_size,
        points_arr=points_arr,
    )

    spy = mocker.spy(tiffs, "read_cuboid")
    tiffs.get_point_cuboid_data(0)
    tiffs.get_point_cuboid_data(1)
    tiffs.get_point_cuboid_data(0)
    batch = torch.zeros((2, *cube_size, 2))
    tiffs.get_point_batch_cuboid_data(batch, [0, 1])

    # number of cubes per
    match cached:
        # cache is always in addition to one cube
        case 0:
            # only one cube is ever cached in memory, per channel
            assert spy.call_count == (1 + 1 + 1 + 1) * 2
        case 1:
            # should only ever read each cube once as there's enough cache
            assert spy.call_count == 2 * 2


def assert_dataset_cubes_matches_cubes(
    cubes: Sequence[np.ndarray],
    data_loader: CuboidDatasetBase | dict,
    batches: Sequence[Sequence[int]],
) -> None:
    """
    Checks that the provided cubes from the data loader or dict are the same
    as the cubes we get from the data loader.
    """
    cubes = [torch.from_numpy(cube) for cube in cubes]
    for batch in batches:
        for i in batch:
            # compare each cube with cube from data loader
            assert torch.equal(cubes[i], data_loader[i])
        batch_cubes = [cubes[i][None, ...] for i in batch]

        # compare batch of cubes with cubes from data loader
        cube_batch = torch.concatenate(batch_cubes, 0)
        assert torch.equal(cube_batch, data_loader[batch])


def assert_dataset_cubes_bad_indices(
    data_loader: CuboidDatasetBase,
    individuals: Sequence[int],
    batches: Sequence[Sequence[int]],
) -> None:
    """Checks that trying to access cubes beyond valid cubes fails."""
    for i in individuals:
        # check individual cubes indices that are out of bounds
        with pytest.raises(IndexError):
            data_loader[i]

    for batch in batches:
        # check batches that have indices that are out of bounds
        with pytest.raises(IndexError):
            data_loader[batch]


def get_sample_dataset_12(
    seed=0,
    target_output=None,
    augment=False,
    augment_likelihood=0.9,
    flippable_axis=(0, 1, 2),
    output_axis_order=("x", "y", "z", "c"),
):
    """
    Returns a numpy volume, a set of 12 points in the volume, extracted cubes
    from that original volume centered on the points, and a CuboidStackDataset
    representing the volume.
    """
    volume = sample_volume(60, 60, 30, 2, seed)
    points = [
        (x, y, z) for x in (27, 29) for y in (28, 30) for z in (12, 13, 18)
    ]
    cube_size = 50, 50, 20
    cubes, _ = to_numpy_cubes(volume, points, cube_size)

    stack = CuboidStackDataset(
        points=[Cell(pos, Cell.UNKNOWN) for pos in points],
        data_voxel_sizes=(1, 1, 5),
        network_voxel_sizes=(1, 1, 5),
        network_cuboid_voxels=cube_size,
        axis_order=("x", "y", "z"),
        output_axis_order=output_axis_order,
        augment=augment,
        augment_likelihood=augment_likelihood,
        flippable_axis=flippable_axis,
        rotate_range=None,
        translate_range=None,
        scale_range=None,
        target_output=target_output,
        signal_array=volume[..., 0],
        background_array=volume[..., 1],
        max_axis_0_cuboids_buffered=3,
    )
    return stack, points, cubes


def test_array_dataset(unique_int):
    """
    Checks that the data returned by the CuboidStackDataset for given
    points matches the data it should return.
    """
    stack, points, cubes = get_sample_dataset_12(unique_int)
    cube_size = cubes[0].shape

    assert stack.cuboid_with_channels_size == cube_size
    assert stack.src_image_data.cuboid_with_channels_size == cube_size
    assert stack.src_image_data.data_with_channels_axis_order == (
        "x",
        "y",
        "z",
        "c",
    )

    # check various batches are correctly returned
    assert_dataset_cubes_matches_cubes(
        cubes, stack, [[0, 5], [0], [3], [0], [5, 0], [2, 2]]
    )
    assert_dataset_cubes_bad_indices(stack, [15], [[1, 14]])


def test_array_dataset_signal_only(unique_int):
    """
    Checks that when using only the signal channel, the data returned by the
    CuboidStackDataset for given points matches the data it should return.
    """
    volume = sample_volume(60, 60, 30, 1, unique_int)
    points = [(x, 28, 18) for x in (27, 29)]
    cube_size = 50, 50, 20
    cubes, _ = to_numpy_cubes(volume, points, cube_size)

    stack = CuboidStackDataset(
        points=[Cell(pos, Cell.UNKNOWN) for pos in points],
        data_voxel_sizes=(1, 1, 5),
        network_voxel_sizes=(1, 1, 5),
        network_cuboid_voxels=cube_size,
        axis_order=("x", "y", "z"),
        output_axis_order=("x", "y", "z", "c"),
        signal_array=volume[..., 0],
        background_array=None,
    )

    cube_size = cubes[0].shape

    assert stack.cuboid_with_channels_size == cube_size
    assert stack.src_image_data.cuboid_with_channels_size == cube_size

    # check various batches are correctly returned
    assert_dataset_cubes_matches_cubes(
        cubes,
        stack,
        [
            [0, 1],
        ],
    )


def test_tiff_image_dataset(unique_int, tmp_path):
    """
    Checks that the data returned by the CuboidTiffDataset for given points
    matches the data it should return.
    """
    volume = sample_volume(60, 60, 30, 2, unique_int)
    points = [
        (x, y, z) for x in (27, 29) for y in (28, 30) for z in (12, 13, 18)
    ]
    cube_size = 50, 50, 20
    filenames, cubes, _ = to_tiff_cubes(volume, cube_size, points, tmp_path)

    tiffs = CuboidTiffDataset(
        points=[Cell(pos, Cell.UNKNOWN) for pos in points],
        data_voxel_sizes=(1, 1, 5),
        network_voxel_sizes=(1, 1, 5),
        network_cuboid_voxels=cube_size,
        axis_order=("x", "y", "z"),
        output_axis_order=("x", "y", "z", "c"),
        augment=False,
        points_filenames=filenames,
        max_cuboids_buffered=3,
    )

    assert tiffs.cuboid_with_channels_size == (*cube_size, 2)
    assert tiffs.src_image_data.cuboid_with_channels_size == (*cube_size, 2)
    assert tiffs.src_image_data.data_with_channels_axis_order == (
        "x",
        "y",
        "z",
        "c",
    )

    # check various batches are correctly returned
    assert_dataset_cubes_matches_cubes(
        cubes, tiffs, [[0, 5], [0], [3], [0], [5, 0], [2, 2]]
    )
    assert_dataset_cubes_bad_indices(tiffs, [15], [[1, 14]])


@pytest.mark.parametrize(
    "batch_size,batches",
    [
        (1, [[i] for i in range(12)]),
        (4, [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]),
        (5, [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11]]),
    ],
)
def test_sampler_batch_size(batch_size: int, batches: list[list[int]]):
    """
    Checks that the dataset sampler with different batch sizes returns the
    correct batches.
    """
    dataset, points, _ = get_sample_dataset_12()

    sampler = CuboidBatchSampler(
        dataset=dataset,
        batch_size=batch_size,
        auto_shuffle=False,
        sort_by_axis=None,
    )

    assert len(sampler) == len(batches)
    samples = list(sampler)
    assert len(samples) == len(batches)

    for i, batch in enumerate(batches):
        assert np.array_equal(samples[i], batch)


@pytest.mark.parametrize(
    "axis_name,axis",
    [
        ("y", 1),
        ("z", 2),
    ],
)
def test_sampler_sort(axis_name: str, axis: int):
    """
    Checks that the dataset sampler with sorting for different axes returns the
    batches correctly sorted.
    """
    dataset, points, _ = get_sample_dataset_12()

    sampler = CuboidBatchSampler(
        dataset=dataset,
        batch_size=4,
        auto_shuffle=False,
        sort_by_axis=axis_name,
    )

    assert len(sampler) == 3
    samples = list(sampler)
    assert len(samples) == 3

    # for the given axis, points should only be increasing
    indices = np.concatenate(samples)
    last = -1
    for i in indices:
        assert points[i][axis] >= last
        last = points[i][axis]


def test_sampler_shuffle():
    """Checks that the dataset sampler with shuffling works."""
    dataset, points, _ = get_sample_dataset_12()

    sampler = CuboidBatchSampler(
        dataset=dataset,
        batch_size=4,
        auto_shuffle=True,
        sort_by_axis=None,
    )

    # we do 5 different sampling / shuffling. The prob they are all the same
    # by chance is astronomical, unless it's not shuffling properly
    same = True
    last = None
    for i in range(5):
        assert len(sampler) == 3
        samples = list(sampler)
        assert len(samples) == 3
        indices = np.concatenate(samples)

        if last is not None:
            same = same and np.array_equal(last, indices)
            if not same:
                break
        last = indices

    assert not same


def test_sampler_shuffle_sort():
    """
    Checks that when shuffling and sorting then only within batch is shuffled.
    Across batches the data stays the same.
    """
    dataset, points, _ = get_sample_dataset_12()

    sampler = CuboidBatchSampler(
        dataset=dataset,
        batch_size=4,
        auto_shuffle=True,
        sort_by_axis="z",
    )

    # prob of a particular ordering is 1 / (4! * 4! * 4!) because each batch
    # is individually reshuffled. This 1 / 13,824. Doing this 5 times is
    # also astronomical, since they are all independent events
    same = True
    last = None
    last_raw = None
    for i in range(5):
        assert len(sampler) == 3
        samples = list(sampler)
        assert len(samples) == 3
        indices = np.concatenate(samples)

        if last is not None:
            # check batches were only shuffled within batch
            for i in range(3):
                assert set(last_raw[i]) == set(samples[i])

            same = same and np.array_equal(last, indices)
            if not same:
                break

        last = indices
        last_raw = samples

    assert not same


@pytest.mark.parametrize("data_thread", [True, False])
@pytest.mark.parametrize("num_workers", [0, 1, 4])
def test_dataset_dataloader_threads(unique_int, num_workers, data_thread):
    """
    Checks that the torch/keras dataloaders can load the data properly under
    different threading conditions including whether data is loaded in separate
    threads or only the main thread.

    Also check that we can load the data in each sub-process, without a main
    data thread.
    """
    dataset, points, cubes = get_sample_dataset_12(unique_int)
    dataloader = DataLoader(
        dataset,
        batch_size=5,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )

    try:
        if data_thread:
            dataset.start_dataset_thread(num_workers)
        batches = list(dataloader)
    finally:
        dataset.stop_dataset_thread()

    cubes = [torch.from_numpy(cube)[None, ...] for cube in cubes]
    for k, batch in enumerate([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11]]):
        # compare batch of cubes with cubes from data loader
        cube_batch = torch.concatenate([cubes[i] for i in batch], 0)
        assert torch.equal(cube_batch, batches[k])


@pytest.mark.parametrize(
    "batch_size,batch_idx",
    [
        (1, [[i] for i in range(12)]),
        (4, [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]),
    ],
)
def test_dataset_dataloader_sampler(
    unique_int, batch_size, batch_idx, mocker: MockerFixture
):
    """
    Checks that the torch/keras dataloaders can load the data properly with
    different batch size using the provided sampler.
    """
    dataset, points, cubes = get_sample_dataset_12(unique_int)

    sampler = CuboidBatchSampler(
        dataset=dataset,
        batch_size=batch_size,
        sort_by_axis=None,
        auto_shuffle=False,
    )
    dataloader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=None,
        num_workers=0,
    )

    spy = mocker.spy(dataset, "_get_multiple_items")

    batches = list(dataloader)

    cubes = [torch.from_numpy(cube)[None, ...] for cube in cubes]
    for k, batch in enumerate(batch_idx):
        # compare batch of cubes with cubes from data loader
        cube_batch = torch.concatenate([cubes[i] for i in batch], 0)
        assert torch.equal(cube_batch, batches[k])

    assert len(spy.call_args_list) == len(batch_idx)
    args = [call.args[0].tolist() for call in spy.call_args_list]
    assert args == batch_idx


@pytest.mark.parametrize("target_output", ["cell", "label", None])
def test_dataset_target_output(unique_int, target_output):
    """Checks that dataset's output for different requested target labels."""
    stack, points, cubes = get_sample_dataset_12(unique_int, target_output)

    # get the batches or single items
    data = {k: stack[k] for k in [(1, 5), (1,), 1, 5]}
    target = {}
    if target_output is not None:
        # we have some labels, break it into target labels and data
        target = {k: v[1] for k, v in data.items()}
        data = {k: v[0] for k, v in data.items()}

    assert_dataset_cubes_matches_cubes(cubes, data, [(1, 5), (1,)])
    if target_output == "cell":
        # labels should be cells
        cells = {
            (1, 5): [Cell(points[1], 1), Cell(points[5], 1)],
            (1,): [
                Cell(points[1], 1),
            ],
            1: Cell(points[1], 1),
            5: Cell(points[5], 1),
        }
        assert cells == target
    elif target_output == "label":
        # labels are one-hot vectors
        assert torch.equal(
            target[(1, 5)], torch.tensor([[1, 0], [1, 0]], dtype=torch.int)
        )
        assert torch.equal(
            target[(1,)], torch.tensor([[1, 0]], dtype=torch.int)
        )
        assert torch.equal(target[1], torch.tensor([1, 0], dtype=torch.int))
        assert torch.equal(target[5], torch.tensor([1, 0], dtype=torch.int))


def test_dataset_augment(unique_int):
    """Checks that augment works using axis flipping."""
    stack, points, cubes = get_sample_dataset_12(
        unique_int, augment=True, augment_likelihood=1, flippable_axis=(1,)
    )
    # torch can't handle negative views, so copy
    cubes = [np.flip(c, 1).copy() for c in cubes]
    assert_dataset_cubes_matches_cubes(
        cubes, stack, [[0, 5], [0], [3], [0], [5, 0], [2, 2]]
    )


def test_dataset_dim_switch(unique_int):
    """Checks that input/output axis ordering works."""
    stack, points, cubes = get_sample_dataset_12(
        unique_int,
        output_axis_order=("y", "z", "x", "c"),
    )
    # all our test data is x, y, z order
    cubes = [np.moveaxis(c, [0, 1, 2], [2, 0, 1]).copy() for c in cubes]
    assert_dataset_cubes_matches_cubes(
        cubes, stack, [[0, 5], [0], [3], [0], [5, 0], [2, 2]]
    )


@pytest.mark.parametrize(
    "scale,axis",
    [
        (0.5, 0),
        (2, 0),
        (2, 1),
        (2, 2),
    ],
)
def test_dataset_voxel_scale(scale, axis):
    """
    Checks that cube scaling works properly when the input/network voxel size
    are different. Try different scale factors, as well as scaling only one
    axis at a time for each axis.
    """
    # generate points whose cubes don't much overlap, even when cube is 2x size
    cube_size = 15, 15, 17
    volume = np.zeros((60, 35, 40, 2), dtype=np.uint16)
    points = [(15, 17, 20), (45, 17, 20)]
    # fill cube of 5x5x5x around center of point
    for x, y, z in points:
        volume[x - 2 : x + 3, y - 2 : y + 3, z - 2 : z + 3, :] = 100

    network_voxel_sizes = (6, 8, 10)
    data_voxel_sizes = list(network_voxel_sizes)
    data_voxel_sizes[axis] = int(data_voxel_sizes[axis] * scale)
    stack = CuboidStackDataset(
        points=[Cell(pos, Cell.UNKNOWN) for pos in points],
        data_voxel_sizes=data_voxel_sizes,
        network_voxel_sizes=network_voxel_sizes,
        network_cuboid_voxels=cube_size,
        axis_order=("x", "y", "z"),
        output_axis_order=("x", "y", "z", "c"),
        augment=False,
        signal_array=volume[..., 0],
        background_array=volume[..., 1],
        max_axis_0_cuboids_buffered=3,
    )

    # get cubes from data sets, batch and individual items
    scaled_cubes = [
        stack[0],
        stack[1],
        *stack[(0, 1)],
    ]
    # we test a voxel in the cube along the given axis, for each axis
    for ax_i in (0, 1, 2):
        # both before and after the center in that axis
        for offset_sign in (-1, 1):
            # and at different offsets from center. Offset of 1 and 2 should be
            # set, offset of 4 should be blank. Because we set 5x5x5x
            for offset, f in ((1, torch.gt), (2, torch.gt), (4, torch.lt)):
                # for only the axis that was scaled, also scale the offset
                if ax_i == axis:
                    offset = int(offset * scale)

                pos = [s // 2 for s in cube_size]
                # keep it within cube in case of large scaling
                pos[ax_i] = min(
                    max(pos[ax_i] + offset_sign * offset, 0),
                    cube_size[ax_i] - 1,
                )

                for cube in scaled_cubes:
                    # check that for all channels, the voxel is set/unset
                    assert torch.all(f(cube[tuple(pos)], 25))


def get_single_point_dataset():
    volume = np.empty((30, 60, 60, 2), dtype=np.uint16)
    dataset = CuboidStackDataset(
        points=[Cell((30, 30, 15), Cell.UNKNOWN)],
        data_voxel_sizes=(5, 1, 1),
        augment=False,
        network_cuboid_voxels=(20, 50, 50),
        signal_array=volume[..., 0],
        background_array=volume[..., 1],
    )
    return dataset


def test_dataset_thread_exception():
    """
    Checks that the external thread that reads the data for all the
    requesting threads/processes properly forwards exceptions.
    """
    dataset = get_single_point_dataset()

    try:
        dataset.start_dataset_thread(1)
        with pytest.raises(ExecutionFailure):
            dataset.get_point_data(5)
    finally:
        dataset.stop_dataset_thread()


def test_dataset_thread_bad_msg():
    """
    Checks that the external thread can handle bad queue msg.
    """
    dataset = get_single_point_dataset()

    try:
        dataset.start_dataset_thread(1)
        with pytest.raises(ExecutionFailure):
            dataset._send_rcv_thread_msg(
                dataset.cuboid_with_channels_size, "baaad", 0
            )
    finally:
        dataset.stop_dataset_thread()


@pytest.mark.parametrize("index", [1, [1]])
@pytest.mark.parametrize("do_thread", [True, False])
def test_dataset_single_point_access(unique_int, do_thread, index):
    """
    Checks getting one and a batch of cubes with / without external thread.
    """
    dataset, _, cubes = get_sample_dataset_12(unique_int)

    try:
        if do_thread:
            dataset.start_dataset_thread(1)

        res = dataset[index]
        if isinstance(index, list):
            # remove batch dim
            res = res[0]
        assert np.array_equal(res.numpy(), cubes[1])
    finally:
        dataset.stop_dataset_thread()


def test_get_data_cuboid_range():
    """Validate input parameters."""
    assert get_data_cuboid_range(10, 10, "x") == (5, 15)
    assert get_data_cuboid_range(10, 10, "y") == (5, 15)
    assert get_data_cuboid_range(10, 10, "z") == (5, 15)

    with pytest.raises(ValueError):
        get_data_cuboid_range(10, 10, "m")


def test_img_data_base_bad_args():
    """Validate input parameters."""
    points = np.empty(5, dtype=[("x", "<f8"), ("y", "<f8"), ("z", "<f8")])

    with pytest.raises(ValueError):
        ImageDataBase(points_arr=points, data_axis_order=("x", "y"))


def test_img_data_not_impl():
    """Validate calling base, not-implemented functions."""
    points = np.empty(5, dtype=[("x", "<f8"), ("y", "<f8"), ("z", "<f8")])
    data = ImageDataBase(points_arr=points)

    with pytest.raises(NotImplementedError):
        data.get_point_cuboid_data(0)
    with pytest.raises(NotImplementedError):
        data.get_point_batch_cuboid_data(
            torch.empty(1, *data.cuboid_with_channels_size), [0]
        )


def test_img_stack_not_impl():
    """Validate calling base, not-implemented functions."""
    points = np.empty(5, dtype=[("x", "<f8"), ("y", "<f8"), ("z", "<f8")])
    data = CachedStackImageDataBase(points_arr=points)

    with pytest.raises(NotImplementedError):
        data.read_plane(0, 0)


def test_img_cuboid_not_impl():
    """Validate calling base, not-implemented functions."""
    points = np.empty(5, dtype=[("x", "<f8"), ("y", "<f8"), ("z", "<f8")])
    data = CachedCuboidImageDataBase(points_arr=points)

    with pytest.raises(NotImplementedError):
        data.read_cuboid(0, 0)


def test_img_cuboid_bad_arg(tmp_path):
    """Validate input parameters."""
    # 2 points but only 1 filename set
    points = np.empty(2, dtype=[("x", "<f8"), ("y", "<f8"), ("z", "<f8")])
    filenames = np.array(
        [(str(tmp_path / "a.tif"), str(tmp_path / "b.tif"))]
    ).astype(np.str_)

    tifffile.imwrite(
        filenames[0, 0].item(), np.empty((10, 10, 10), dtype=np.uint16)
    )
    tifffile.imwrite(
        filenames[0, 1].item(), np.empty((10, 10, 10), dtype=np.uint16)
    )

    with pytest.raises(ValueError):
        # no filenames, 1 point
        CachedTiffCuboidImageData(
            points_arr=points[:1], filenames_arr=filenames[:0]
        )

    with pytest.raises(ValueError):
        # 1 filename set, but 2 points
        CachedTiffCuboidImageData(points_arr=points, filenames_arr=filenames)

    # one of each - should work
    data = CachedTiffCuboidImageData(
        points_arr=points[:1], filenames_arr=filenames
    )
    assert len(data.points_arr)


def test_dataset_base_bad_args():
    """Validate input parameters."""
    points = [Cell((0, 0, 0), Cell.CELL)]

    with pytest.raises(ValueError):
        # needs 3 axis xyz
        CuboidDatasetBase(
            points=points,
            data_voxel_sizes=(5, 1, 1),
            axis_order=("x", "y"),
        )

    with pytest.raises(ValueError):
        # needs 4 dims xyzc
        CuboidDatasetBase(
            points=points,
            data_voxel_sizes=(5, 1, 1),
            output_axis_order=("x", "y", "c"),
        )

    with pytest.raises(ValueError):
        # c must be last dim
        CuboidDatasetBase(
            points=points,
            data_voxel_sizes=(5, 1, 1),
            output_axis_order=("x", "y", "c", "z"),
        )

    with pytest.raises(ValueError):
        # sizes must be 3-tuple
        CuboidDatasetBase(
            points=points,
            data_voxel_sizes=(5, 1, 1),
            network_voxel_sizes=(20, 50),
        )


def test_dataset_manual_image_data():
    """Check that we can manually pass an image data instance to dataset."""
    points = np.zeros(1, dtype=[("x", "<f8"), ("y", "<f8"), ("z", "<f8")])
    data = CachedStackImageDataBase(
        points_arr=points,
        data_axis_order=("x", "y", "z"),
        cuboid_size=(20, 50, 50),
    )

    dataset = CuboidDatasetBase(
        points=[Cell((0, 0, 0), Cell.CELL)],
        data_voxel_sizes=(5, 1, 1),
        axis_order=("z", "y", "x"),
        network_cuboid_voxels=(20, 50, 50),
        src_image_data=data,
    )
    # should have 5 elements (x, y, z, c, b) because we are re-ordering
    assert len(dataset._output_data_dim_reordering) == 5


def test_dataset_target_bad_values():
    """Validate that target must be valid both for single point and batch."""
    dataset = get_single_point_dataset()
    dataset.target_output = "blah"

    with pytest.raises(ValueError):
        dataset[0]

    with pytest.raises(ValueError):
        # batch of points
        dataset[[0]]


def test_rescale_dataset_bad_arg():
    """Validate input parameters."""
    volume = np.empty((30, 60, 60, 2), dtype=np.uint16)
    dataset = CuboidStackDataset(
        points=[Cell((30, 30, 15), Cell.UNKNOWN)],
        data_voxel_sizes=(4, 1, 1),
        augment=False,
        network_cuboid_voxels=(20, 50, 50),
        signal_array=volume[..., 0],
        background_array=volume[..., 1],
    )

    with pytest.raises(ValueError):
        # must be 5-dim, batch, xyzc
        dataset.convert_to_output(torch.zeros((50, 50, 20, 2)))


def test_stack_dataset_bad_arg_diff_shapes():
    """Validates that we check signal and background must have same shape."""
    volume = np.empty((30, 60, 60, 2, 2), dtype=np.uint16)
    with pytest.raises(ValueError):
        CuboidStackDataset(
            points=[Cell((30, 30, 15), Cell.UNKNOWN)],
            data_voxel_sizes=(5, 1, 1),
            signal_array=volume[..., 0, 0],
            background_array=volume[..., 1],
        )


def test_stack_dataset_bad_arg_bad_shape():
    """Validates that we check signal/background are 4-dim."""
    volume = np.empty((30, 60, 60, 2, 2), dtype=np.uint16)
    with pytest.raises(ValueError):
        CuboidStackDataset(
            points=[Cell((30, 30, 15), Cell.UNKNOWN)],
            data_voxel_sizes=(5, 1, 1),
            signal_array=volume[..., 0],
            background_array=volume[..., 1],
        )


def test_dataset_cuboid_bad_arg(tmp_path):
    """Validate input parameters."""
    filenames = [(str(tmp_path / "a.tif"), str(tmp_path / "b.tif"))]

    tifffile.imwrite(filenames[0][0], np.empty((10, 10, 10), dtype=np.uint16))
    tifffile.imwrite(filenames[0][1], np.empty((10, 10, 10), dtype=np.uint16))

    with pytest.raises(ValueError):
        # filenames must have at least one sample
        CuboidTiffDataset(
            points=[Cell((30, 30, 15), Cell.UNKNOWN)],
            data_voxel_sizes=(5, 1, 1),
            points_filenames=filenames[:0],
        )

    with pytest.raises(ValueError):
        # filenames (1) must have same num as points (2)
        CuboidTiffDataset(
            points=[
                Cell((30, 30, 15), Cell.UNKNOWN),
                Cell((32, 30, 15), Cell.UNKNOWN),
            ],
            data_voxel_sizes=(5, 1, 1),
            points_filenames=filenames,
        )


def test_point_has_full_cuboid_unscaled():
    """
    Tests that only cuboids that have full cubes around the point's center
    are included. Tested under condition where network and data have same
    voxel size.
    """
    volume = np.empty((30, 30, 30, 2), dtype=np.uint16)
    dataset = CuboidStackDataset(
        points=[
            Cell((5, 15, 15), Cell.UNKNOWN),
            Cell((3, 15, 15), Cell.UNKNOWN),
            Cell((27, 15, 15), Cell.UNKNOWN),
        ],
        data_voxel_sizes=(1, 1, 1),
        network_voxel_sizes=(1, 1, 1),
        network_cuboid_voxels=(10, 10, 10),
        axis_order=("x", "y", "z"),
        signal_array=volume[..., 0],
        background_array=volume[..., 1],
    )
    assert len(dataset.points_arr) == 1
    p = dataset.points_arr[0]
    assert (p["x"], p["y"], p["z"]) == (5, 15, 15)


def test_point_has_full_cuboid_scaled():
    """
    Tests that only cuboids that have full cubes around the point's center
    are included. Tested under condition where data is half of network's
    voxel size.
    """
    volume = np.empty((30, 30, 30, 2), dtype=np.uint16)
    dataset = CuboidStackDataset(
        points=[
            Cell((10, 15, 15), Cell.UNKNOWN),
            Cell((6, 15, 15), Cell.UNKNOWN),
            Cell((24, 15, 15), Cell.UNKNOWN),
        ],
        data_voxel_sizes=(1, 1, 1),
        network_voxel_sizes=(2, 1, 1),
        network_cuboid_voxels=(10, 10, 10),
        axis_order=("x", "y", "z"),
        signal_array=volume[..., 0],
        background_array=volume[..., 1],
    )
    assert len(dataset.points_arr) == 1
    p = dataset.points_arr[0]
    assert (p["x"], p["y"], p["z"]) == (10, 15, 15)


def test_points_unchanged():
    volume = np.empty((30, 60, 60, 2), dtype=np.uint16)
    cell = Cell((30, 30, 15), Cell.UNKNOWN)
    dataset = CuboidStackDataset(
        points=[cell],
        data_voxel_sizes=(5, 1, 1),
        augment=False,
        network_cuboid_voxels=(20, 50, 50),
        signal_array=volume[..., 0],
        background_array=volume[..., 1],
    )

    assert len(dataset.points) == 1
    assert dataset.points[0] is cell
    x, y, z, tp = dataset.points_arr[0]
    assert (x, y, z, tp) == (30, 30, 15, Cell.UNKNOWN)
