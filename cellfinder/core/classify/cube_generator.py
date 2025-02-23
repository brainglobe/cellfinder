import math
from collections import OrderedDict, defaultdict
from numbers import Integral
from typing import Any, Literal

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from brainglobe_utils.cells.cells import Cell
from torch.multiprocessing import Queue
from torch.utils.data import Dataset, Sampler, get_worker_info

from cellfinder.core import types
from cellfinder.core.tools.threading import (
    EOFSignal,
    ExecutionFailure,
    ProcessWithException,
)


class StackSizeError(Exception):
    pass


def get_axis_reordering(
    in_order: tuple[str, ...], out_order: tuple[str, ...]
) -> list[int]:
    indices = []
    for value in out_order:
        indices.append(in_order.index(value))
    return indices


def _read_planes_send_cubes(
    process: ProcessWithException,
    dataset: "CachedDatasetBase",
    queues: list[Queue],
) -> None:
    while True:
        msg = process.get_msg_from_mainthread()
        if msg == EOFSignal:
            return

        key, cell, buffer, queue_id = msg
        queue = queues[queue_id]
        try:
            if key == "cell":
                buffer[:] = dataset.get_cuboid(cell)
            elif key == "cells":
                dataset.fill_batch(buffer, cell)
            else:
                raise ValueError(f"Bad message {key}")
        except BaseException as e:
            queue.put(("exception", e), block=True, timeout=None)
        else:
            queue.put((key, cell), block=True, timeout=None)


def _get_cube_indices(
    cell: Cell, axis: Literal["x", "y", "z"], size: int
) -> tuple[int, int]:
    match axis:
        case "x" | "y":
            start = int(round(getattr(cell, axis) - size / 2))
        case "z":
            start = int(cell.z - size // 2)
        case _:
            raise ValueError(f"Unknown axis {axis}")

    return start, start + size


class CachedDatasetBase:
    """
    We buffer along axis 0, which is the data_axis_order[0] axis.
    The size of each cube in voxels of the dataset is given by cuboid_size.
    Data returned is in data_axis_order, with channel the last axis
    """

    data_shape: tuple[int, int, int]

    num_channels: int = 1

    max_axis_0_planes_buffered: int = 0

    cuboid_size: tuple[int, int, int] = (1, 1, 1)
    # in units of voxels, not um. in data_axis_order

    data_axis_order: tuple[str, str, str] = ("z", "y", "x")

    full_data_axis_order: tuple[str, str, str, str] = ("z", "y", "x", "c")

    _buffer: torch.Tensor | None = None
    # channels-last order

    _buf_plane_start: int | None = None

    _planes_buffer: OrderedDict[int, torch.Tensor]

    def __init__(
        self,
        max_axis_0_cubes_buffered: int = 0,
        data_axis_order: tuple[str, str, str] = ("z", "y", "x"),
        cuboid_size: tuple[int, int, int] = (1, 1, 1),
    ):
        self.max_axis_0_planes_buffered = int(
            round(max_axis_0_cubes_buffered * cuboid_size[0])
        )
        self.cuboid_size = cuboid_size

        if len(data_axis_order) != 3 or set(data_axis_order) != {
            "z",
            "y",
            "x",
        }:
            raise ValueError(
                f"Expected the axis order to list x, y, z, but got "
                f"{data_axis_order}"
            )
        self.data_axis_order = data_axis_order
        self.full_data_axis_order = *data_axis_order, "c"

        self._planes_buffer = OrderedDict()

    @property
    def full_cuboid_size(self) -> tuple[int, int, int, int]:
        return *self.cuboid_size, self.num_channels

    def fill_batch(self, batch: torch.Tensor, cells: list[Cell]) -> None:
        if len(cells) != batch.shape[0]:
            raise ValueError(
                "Expected the number of cells to match the batch size"
            )

        for i, cell in enumerate(cells):
            batch[i, ...] = self.get_cuboid(cell)

    def get_cuboid(self, cell: Cell) -> torch.Tensor:
        buffer = self._buffer
        buf_start = self._buf_plane_start
        n_planes = self.cuboid_size[0]
        cube_indices = []
        for ax, size in zip(self.data_axis_order, self.cuboid_size):
            cube_indices.append(_get_cube_indices(cell, ax, size))

        # buffered axis start/end
        cuboid_start, cuboid_end = cube_indices[0]

        if buf_start is None:
            shape = (
                self.cuboid_size[0],
                *self.data_shape[1:],
                self.num_channels,
            )
            buffer = torch.empty(shape, dtype=torch.float32)
            self._pop_plane_buffer(buffer, 0, cuboid_start, cuboid_end)
        elif buf_start + n_planes <= cuboid_start or buf_start > cuboid_end:
            # buffer ends before cuboid starts or it starts after cuboid's end
            self._return_plane_buffer(
                buffer, 0, buf_start, buf_start + n_planes
            )
            self._pop_plane_buffer(buffer, 0, cuboid_start, cuboid_end)
        elif buf_start < cuboid_start:
            # cuboid starts midway through buffer. Drop start of buffer, shift
            # buffer left and fill in end
            dropping = cuboid_start - buf_start
            self._return_plane_buffer(buffer, 0, buf_start, cuboid_start)
            buffer = torch.roll(buffer, -dropping, 0)
            self._pop_plane_buffer(
                buffer,
                n_planes - dropping,
                cuboid_start + n_planes - dropping,
                cuboid_end,
            )
        elif buf_start > cuboid_start:
            # cuboid starts before buffer start. Drop end of buffer, shift
            # buffer right, and fill in start
            dropping = buf_start - cuboid_start
            self._return_plane_buffer(
                buffer,
                n_planes - dropping,
                cuboid_start + n_planes - dropping,
                cuboid_end,
            )
            buffer = torch.roll(buffer, dropping, 0)
            self._pop_plane_buffer(buffer, 0, cuboid_start, buf_start)
        else:
            assert buf_start == cuboid_start

        self._buffer = buffer
        self._buf_plane_start = cuboid_start

        # 3 dim plus channels at the end
        slices = [
            slice(None, None),
            slice(*cube_indices[1]),
            slice(*cube_indices[2]),
            slice(None, None),
        ]
        return buffer[slices]

    def _return_plane_buffer(
        self,
        buffer: torch.Tensor,
        offset_start: int,
        plane_start: int,
        plane_end: int,
    ) -> None:
        planes_buffer = self._planes_buffer
        max_planes = self.max_axis_0_planes_buffered
        offset_end = offset_start + plane_end - plane_start

        if not max_planes:
            # no buffer available
            return

        for i, plane in zip(
            range(offset_start, offset_end), range(plane_start, plane_end)
        ):
            assert len(planes_buffer) <= max_planes
            if len(planes_buffer) == max_planes:
                # fifo when last=False
                planes_buffer.popitem(last=False)
            planes_buffer[plane] = buffer[i, ...].detach().clone()

    def _pop_plane_buffer(
        self,
        buffer: torch.Tensor,
        offset_start: int,
        plane_start: int,
        plane_end: int,
    ) -> None:
        planes_buffer = self._planes_buffer
        offset_end = offset_start + plane_end - plane_start

        for i, plane in zip(
            range(offset_start, offset_end), range(plane_start, plane_end)
        ):
            if plane in planes_buffer:
                buffer[i, ...] = planes_buffer.pop(plane)
            else:
                for channel in range(self.num_channels):
                    buffer[i, ..., channel] = self.read_plane(plane, channel)

    def read_plane(self, plane: int, channel: int) -> torch.Tensor:
        raise NotImplementedError


class CachedStackDataset(CachedDatasetBase):

    input_arrays: list[types.array]

    def __init__(
        self,
        input_arrays: list[types.array],
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.input_arrays = input_arrays
        self.data_shape = input_arrays[0].shape
        self.num_channels = len(self.input_arrays)

    def read_plane(self, plane: int, channel: int) -> torch.Tensor:
        return torch.from_numpy(
            np.asarray(self.input_arrays[channel][plane, ...])
        )


class CubeDatasetBase(Dataset):
    """
    Input voxel order is Width, Height, Depth
    Returned data order is (Batch) Height, Width, Depth, Channel.
    """

    def __init__(
        self,
        points: list[Cell],
        data_voxel_sizes: tuple[int, int, int],
        network_voxel_sizes: tuple[int, int, int],
        network_cube_voxels: tuple[int, int, int] = (20, 50, 50),
        axis_order: tuple[str, str, str] = ("z", "y", "x"),
        output_axis_order: tuple[str, str, str, str] = ("y", "x", "z", "c"),
        classes: int = 2,
        sort_by_axis: str | None = "z",
        transform=None,
        target_output: Literal["cell", "dict", "label"] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if len(axis_order) != 3 or set(axis_order) != {"z", "y", "x"}:
            raise ValueError(
                f"Expected the axis order to list x, y, z, but got "
                f"{axis_order}"
            )
        if len(output_axis_order) != 4 or set(output_axis_order) != {
            "z",
            "y",
            "x",
            "c",
        }:
            raise ValueError(
                f"Expected the axis order to list x, y, z, c, but got "
                f"{output_axis_order}"
            )

        if sort_by_axis is None:
            self.points = points
        else:
            self.points = list(
                sorted(points, key=lambda p: getattr(p, sort_by_axis))
            )

        self.data_voxel_sizes = data_voxel_sizes
        self.network_voxel_sizes = network_voxel_sizes
        self.network_cube_voxels = network_cube_voxels
        self.axis_order = axis_order
        self.output_axis_order = output_axis_order

        self.data_cube_voxels = []
        for data_um, network_um, cube_voxels in zip(
            data_voxel_sizes, network_voxel_sizes, network_cube_voxels
        ):
            self.data_cube_voxels.append(
                int(round(cube_voxels * network_um / data_um))
            )

        self.classes = classes
        self.transform = transform
        self.target_output = target_output

    def __len__(self):
        return len(self.points)

    def __getitem__(self, idx):
        if isinstance(idx, Integral):
            return self._get_single_item(idx)
        return self._get_multiple_items(idx)

    def _get_single_item(
        self, idx: int
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        cell = self.points[idx]
        data = self.get_cell_data(cell)

        data = self.rescale_to_output_size(data)
        if self.transform:
            data = self.transform(data)

        match self.target_output:
            case None:
                return data

            case "cell":
                label = cell
            case "dict":
                label = cell.to_dict()
            case _:
                cls = torch.tensor(cell.type - 1)
                label = F.one_hot(cls, num_classes=self.classes)

        return data, label

    def _get_multiple_items(
        self, indices: list[int]
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        cells = [self.points[i] for i in indices]
        data = self.get_cells_data(cells)

        data = self.rescale_to_output_size(data)
        if self.transform:
            data = self.transform(data)

        match self.target_output:
            case None:
                return data

            case "cell":
                labels = cells
            case "dict":
                labels = [cell.to_dict() for cell in cells]
            case _:
                cls = torch.tensor([cell.type - 1 for cell in cells])
                labels = F.one_hot(cls, num_classes=self.classes)

        return data, labels

    def rescale_to_output_size(self, data: torch.Tensor) -> torch.Tensor:
        if self.data_voxel_sizes == self.network_voxel_sizes:
            return data

        # batch dimension
        added_batch = len(data.shape) == 4
        if added_batch:
            data = torch.unsqueeze(data, 0)

        torch_order = "b", "c", "z", "y", "x"
        data_order = "b", *self.output_axis_order
        voxel_order = self.axis_order

        torch_voxel_map = get_axis_reordering(voxel_order, torch_order[2:])
        scaled_cube_size = [
            self.network_cube_voxels[i] for i in torch_voxel_map
        ]

        data = torch.permute(
            data, get_axis_reordering(data_order, torch_order)
        )
        assert len(data.shape) == 5, "expected (batch, channel, z, y, x) shape"
        data = F.interpolate(data, size=scaled_cube_size, mode="trilinear")

        data = torch.permute(
            data, get_axis_reordering(torch_order, data_order)
        )
        if added_batch:
            data = torch.squeeze(data, 0)

        return data

    def get_cell_data(self, cell: Cell) -> torch.Tensor:
        """

        :param cell:
        :return: In the output_axis_order order.
        """
        raise NotImplementedError

    def get_cells_data(self, cells: list[Cell]) -> torch.Tensor:
        """

        :param cells:
        :return: In the output_axis_order order.
        """
        raise NotImplementedError


class CubeStackDataset(CubeDatasetBase):

    cached_dataset: CachedDatasetBase | None = None

    _dataset_process: ProcessWithException | None = None

    _worker_queues: list[Queue] = None

    def __init__(
        self,
        signal_array: types.array,
        background_array: types.array,
        max_axis_0_cubes_buffered: int = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if signal_array.shape != background_array.shape:
            raise ValueError(
                f"Shape of signal images ({signal_array.shape}) does not "
                f"match the shape of the background images "
                f"({background_array.shape}"
            )
        if len(signal_array.shape) != 3:
            raise ValueError("Expected a 3d in data array")

        self.dataset_shape = signal_array.shape
        self.data_arrays = [signal_array, background_array]
        self._worker_queues = []

        self.cached_dataset = CachedStackDataset(
            input_arrays=self.data_arrays,
            data_axis_order=self.axis_order,
            max_axis_0_cubes_buffered=max_axis_0_cubes_buffered,
            cuboid_size=self.data_cube_voxels,
        )

        data_axis_order = self.cached_dataset.full_data_axis_order
        self._data_axis_reordering = None
        if data_axis_order != self.output_axis_order:
            self._data_axis_reordering = get_axis_reordering(
                ("b", *data_axis_order), ("b", *self.output_axis_order)
            )

        self.points = [p for p in self.points if self.point_has_full_cube(p)]

    def point_has_full_cube(self, point: Cell) -> bool:
        for ax, data_size, cube_size in zip(
            self.axis_order, self.dataset_shape, self.data_cube_voxels
        ):
            start, end = _get_cube_indices(point, ax, cube_size)
            if start < 0:
                return False
            if end > data_size:
                # if it's data_size it's fine because end is not inclusive
                return False

        return True

    def get_cell_data(self, cell: Cell) -> torch.Tensor:
        return self.get_cells_data([cell])[0, ...]

    def get_cells_data(self, cells: list[Cell]) -> torch.Tensor:
        data = torch.empty((len(cells), *self.cached_dataset.full_cuboid_size))

        process = self._dataset_process
        if process is None:
            self.cached_dataset.fill_batch(data, cells)
        else:
            queues = self._worker_queues
            # for host, it's the last queue
            queue_id = len(queues) - 1
            if get_worker_info() is not None:
                queue_id = get_worker_info().id
            queue = queues[queue_id]

            process.send_msg_to_thread(("cells", cells, data, queue_id))
            msg, value = queue.get(block=True)
            # if there's no error, we just sent back the cell
            if msg == "exception":
                raise ExecutionFailure(
                    "Reporting failure from data process"
                ) from value
            assert msg == "cells"

        if self._data_axis_reordering is not None:
            data = torch.permute(data, self._data_axis_reordering)
        return data

    def start_dataset_process(self, num_workers: int):
        # include queue for host process
        ctx = mp.get_context("spawn")
        queues = [ctx.Queue(maxsize=0) for _ in range(num_workers + 1)]
        self._worker_queues = queues

        self._dataset_process = ProcessWithException(
            target=_read_planes_send_cubes,
            args=(self.cached_dataset, queues),
            pass_self=True,
        )
        self._dataset_process.start()

    def stop_dataset_process(self):
        process = self._dataset_process
        if process is None:
            return

        self._dataset_process = None
        self._worker_queues = []

        process.notify_to_end_thread()
        process.clear_remaining()
        process.join()


class CubeBatchSampler(Sampler):

    def __init__(
        self,
        dataset: CubeDatasetBase,
        batch_size: int,
        auto_shuffle: bool = False,
        sort_by_axis: str | None = "z",
        segregate_by_axis: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if segregate_by_axis and sort_by_axis is None:
            raise ValueError(
                "Asked to segregate by axis, but an axis for sorting was "
                "not provided"
            )

        self.batch_size = batch_size
        self.auto_shuffle = auto_shuffle
        self.segregate_by_axis = segregate_by_axis

        if sort_by_axis is None:
            plane_indices = [np.arange(len(dataset.points), dtype=np.int64)]
        else:
            points_raw = defaultdict(list)
            for i, point in enumerate(dataset.points):
                points_raw[getattr(point, sort_by_axis)].append(i)
            points_sorted = sorted(points_raw.items(), key=lambda x: x[0])

            plane_indices = [
                np.array(indices, dtype=np.int64)
                for plane, indices in points_sorted
            ]

        self.plane_indices = plane_indices

        self.n_batches = len(self.get_batches(False))

    def get_batches(self, shuffle):
        indices = self.plane_indices
        batch_size = self.batch_size

        if shuffle:
            rng = np.random.default_rng()
            # permuted creates copy that is shuffled
            indices = [rng.permuted(items) for items in indices]

        batches = []
        if self.segregate_by_axis:
            for arr in indices:
                for i in range(math.ceil(len(arr) / batch_size)):
                    batches.append(arr[i * batch_size : (i + 1) * batch_size])
        else:
            arr = np.concatenate(indices)
            for i in range(math.ceil(len(arr) / batch_size)):
                batches.append(arr[i * batch_size : (i + 1) * batch_size])

        return batches

    def __len__(self) -> int:
        return self.n_batches

    def __iter__(self):
        yield from self.get_batches(self.auto_shuffle)
