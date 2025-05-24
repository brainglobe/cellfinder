import numpy as np
import pytest
import tifffile
import torch
from brainglobe_utils.cells.cells import Cell
from pytest_mock.plugin import MockerFixture

from cellfinder.core.classify.cube_generator import (
    CachedArrayStackImageData,
    CachedTiffCuboidImageData,
)

try:
    from brainglobe_utils.cells.cells import file_name_from_cell
except ImportError:

    def file_name_from_cell(cell: Cell, channel: int) -> str:
        name = f"x{int(cell.x)}_y{int(cell.y)}_z{int(cell.z)}Ch{channel}.tif"
        return name


@pytest.fixture
def sample_volume() -> np.ndarray:
    x, y, z, c = 20, 20, 50, 2
    data = np.arange(x * y * z * c).reshape((x, y, z, c)).astype(np.int16)
    return data


@pytest.fixture
def numpy_cubes(sample_volume):
    points = np.empty(2, dtype=[("x", "<f8"), ("y", "<f8"), ("z", "<f8")])
    points[0] = 5, 5, 10
    points[1] = 10, 10, 20

    cubes = [
        sample_volume[4:7, 4:7, 8:13, :],
        sample_volume[8:11, 8:11, 18:23, :],
    ]

    return sample_volume, points, cubes, (3, 3, 5)


@pytest.fixture
def tiff_cubes(sample_volume, tmp_path):
    points = [[5, 5, 10], [10, 10, 20]]

    # create the tiff files, one per channel per point
    filenames = []
    for x, y, z in points:
        # don't have to center cube on point, just get a unique cube
        sig = tmp_path / file_name_from_cell(Cell([x, y, z], 0), channel=0)
        tifffile.imwrite(
            sig, sample_volume[x : x + 3, y : y + 3, z : z + 5, 0]
        )

        back = tmp_path / file_name_from_cell(Cell([x, y, z], 0), channel=1)
        tifffile.imwrite(
            back, sample_volume[x : x + 3, y : y + 3, z : z + 5, 1]
        )

        filenames.append((str(sig), str(back)))

    cubes = [
        sample_volume[5:8, 5:8, 10:15, :],
        sample_volume[10:13, 10:13, 20:25, :],
    ]

    return filenames, points, cubes, (3, 3, 5)


def test_array_image_data(numpy_cubes):
    sample_volume, points, cubes, size = numpy_cubes

    stack = CachedArrayStackImageData(
        input_arrays=[sample_volume[:, :, :, 0], sample_volume[:, :, :, 1]],
        max_axis_0_cuboids_buffered=3,
        data_axis_order=("x", "y", "z"),
        cuboid_size=size,
        points_arr=points,
    )

    assert stack.cuboid_with_channels_size == (*size, 2)
    assert stack.data_with_channels_axis_order == ("x", "y", "z", "c")

    # manually get the cubes around the points
    cube1 = torch.from_numpy(cubes[0])
    cube2 = torch.from_numpy(cubes[1])
    cube_batch = torch.concatenate([cube1[None, ...], cube2[None, ...]], 0)
    batch = torch.zeros_like(cube_batch)

    # compare manual cubes with cubes from stack, as a batch and individually
    assert torch.equal(cube1, stack.get_point_cuboid_data(0))
    assert torch.equal(cube2, stack.get_point_cuboid_data(1))
    stack.get_point_batch_cuboid_data(batch, [0, 1])
    assert torch.equal(cube_batch, batch)


def test_tiff_image_data(tiff_cubes):
    filenames, points, cubes, size = tiff_cubes
    tiffs = CachedTiffCuboidImageData(
        filenames_arr=np.array(filenames).astype(np.str_),
        max_cuboids_buffered=3,
        data_axis_order=("x", "y", "z"),
        cuboid_size=size,
        points_arr=points,
    )

    assert tiffs.cuboid_with_channels_size == (*size, 2)
    assert tiffs.data_with_channels_axis_order == ("x", "y", "z", "c")

    # manually get the cubes around the points
    cube1 = torch.from_numpy(cubes[0])
    cube2 = torch.from_numpy(cubes[1])
    cube_batch = torch.concatenate([cube1[None, ...], cube2[None, ...]], 0)
    batch = torch.zeros_like(cube_batch)

    # compare manual cubes with cubes from tiffs, as a batch and individually
    assert torch.equal(cube1, tiffs.get_point_cuboid_data(0))
    assert torch.equal(cube2, tiffs.get_point_cuboid_data(1))
    tiffs.get_point_batch_cuboid_data(batch, [0, 1])
    assert torch.equal(cube_batch, batch)


@pytest.mark.parametrize("cached", [0, 1])
def test_array_image_data_cache(numpy_cubes, cached, mocker: MockerFixture):
    sample_volume, points, cubes, size = numpy_cubes

    stack = CachedArrayStackImageData(
        input_arrays=[sample_volume[:, :, :, 0], sample_volume[:, :, :, 1]],
        max_axis_0_cuboids_buffered=cached,
        data_axis_order=("x", "y", "z"),
        cuboid_size=size,
        points_arr=points,
    )

    spy = mocker.spy(stack, "read_plane")
    stack.get_point_cuboid_data(0)
    stack.get_point_cuboid_data(1)
    stack.get_point_cuboid_data(0)
    batch = torch.zeros((2, *size, 2))
    stack.get_point_batch_cuboid_data(batch, [0, 1])

    # number of planes per cache dim cube
    n = size[0]
    match cached:
        # cache is always in addition to one cube
        case 0:
            # only one cube is ever cached in memory. Plus 2 channels
            assert spy.call_count == (n + n + n + n) * 2
        case 1:
            # should only ever read each plane once as there's enough cache
            assert spy.call_count == n * 2 * 2


@pytest.mark.parametrize("cached", [0, 1])
def test_tiff_image_data_cache(tiff_cubes, cached, mocker: MockerFixture):
    filenames, points, cubes, size = tiff_cubes
    tiffs = CachedTiffCuboidImageData(
        filenames_arr=np.array(filenames).astype(np.str_),
        max_cuboids_buffered=cached,
        data_axis_order=("x", "y", "z"),
        cuboid_size=size,
        points_arr=points,
    )

    spy = mocker.spy(tiffs, "read_cuboid")
    tiffs.get_point_cuboid_data(0)
    tiffs.get_point_cuboid_data(1)
    tiffs.get_point_cuboid_data(0)
    batch = torch.zeros((2, *size, 2))
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
