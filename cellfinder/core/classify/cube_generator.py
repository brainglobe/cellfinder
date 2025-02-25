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
    ThreadWithException,
)
from cellfinder.core.tools.tools import get_data_converter

AXIS = Literal["x", "y", "z"]
DIM = Literal["x", "y", "z", "C"]


class StackSizeError(Exception):
    pass


def get_axis_reordering(
    in_order: tuple[Any, ...], out_order: tuple[Any, ...]
) -> list[int]:
    indices = []
    for value in out_order:
        indices.append(in_order.index(value))
    return indices


def _read_planes_send_cuboids(
    thread: ThreadWithException,
    dataset: "CachedDatasetBase",
    queues: list[Queue],
) -> None:
    while True:
        msg = thread.get_msg_from_mainthread()
        if msg == EOFSignal:
            return

        key, cell, buffer, queue_id = msg
        queue = queues[queue_id]
        try:
            if key == "cell":
                buffer[:] = dataset.get_cell_cuboid_data(cell)
            elif key == "cells":
                dataset.get_cell_batch_cuboid_data(buffer, cell)
            else:
                raise ValueError(f"Bad message {key}")
        except BaseException as e:
            queue.put(("exception", e), block=True, timeout=None)
        else:
            queue.put((key, cell), block=True, timeout=None)


def get_data_cuboid_voxels(
    network_cuboid_voxels: int,
    network_voxel_size_um: float,
    data_voxel_size_um: float,
) -> int:
    return int(
        round(
            network_cuboid_voxels * network_voxel_size_um / data_voxel_size_um
        )
    )


def get_data_cuboid_range(
    pos: float, num_voxels: int, axis: AXIS
) -> tuple[int, int]:
    match axis:
        case "x" | "y":
            start = int(round(pos - num_voxels / 2))
        case "z":
            start = int(pos - num_voxels // 2)
        case _:
            raise ValueError(f"Unknown axis {axis}")

    return start, start + num_voxels


class CachedDatasetBase:
    """
    We buffer along axis 0, which is the data_axis_order[0] axis.
    The size of each cuboid in voxels of the dataset is given by cuboid_size.
    Data returned is in data_axis_order, with channel the last axis
    """

    data_shape: tuple[int, int, int]

    num_channels: int = 1

    max_axis_0_planes_buffered: int = 0

    cuboid_size: tuple[int, int, int] = (1, 1, 1)
    # in units of voxels, not um. in data_axis_order

    data_axis_order: tuple[AXIS, AXIS, AXIS] = ("z", "y", "x")

    full_data_axis_order: tuple[DIM, DIM, DIM, DIM] = ("z", "y", "x", "c")

    _planes_buffer: OrderedDict[int, torch.Tensor]

    def __init__(
        self,
        max_axis_0_cuboids_buffered: float = 0,
        data_axis_order: tuple[AXIS, AXIS, AXIS] = ("z", "y", "x"),
        cuboid_size: tuple[int, int, int] = (1, 1, 1),
    ):
        self.max_axis_0_planes_buffered = int(
            round((max_axis_0_cuboids_buffered + 1) * cuboid_size[0])
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

    def get_cell_cuboid_data(self, cell: Cell) -> torch.Tensor:
        shape = *self.data_shape, self.num_channels
        data = torch.empty(shape, dtype=torch.float32)

        for j, plane in enumerate(self._get_cuboid_planes(cell)):
            data[j, ...] = plane

        return data

    def get_cell_batch_cuboid_data(
        self, batch: torch.Tensor, cells: list[Cell]
    ) -> None:
        if len(cells) != batch.shape[0]:
            raise ValueError(
                "Expected the number of cells to match the batch size"
            )

        for i, cell in enumerate(cells):
            for j, plane in enumerate(self._get_cuboid_planes(cell)):
                batch[i, j, ...] = plane

    def _get_cuboid_planes(self, cell: Cell) -> list[torch.Tensor]:
        max_planes = self.max_axis_0_planes_buffered
        planes_buffer = self._planes_buffer

        cuboid_indices = []
        for ax, size in zip(self.data_axis_order, self.cuboid_size):
            cuboid_indices.append(
                get_data_cuboid_range(getattr(cell, ax), size, ax)
            )
        ax1 = slice(*cuboid_indices[1])
        ax2 = slice(*cuboid_indices[2])

        planes = []
        # buffered axis
        for i in range(*cuboid_indices[0]):
            if i not in planes_buffer:
                plane_shape = *self.data_shape[1:], self.num_channels
                plane = torch.empty(plane_shape, dtype=torch.float32)
                for channel in range(self.num_channels):
                    plane[:, :, channel] = self.read_plane(i, channel)

                if len(planes_buffer) == max_planes:
                    # fifo when last=False
                    planes_buffer.popitem(last=False)
                assert len(planes_buffer) < max_planes

                planes_buffer[i] = plane

            planes.append(planes_buffer[i][ax1, ax2, :])

        return planes

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
        self._converters = [
            get_data_converter(arr.dtype, np.float32)
            for arr in self.input_arrays
        ]

    def read_plane(self, plane: int, channel: int) -> torch.Tensor:
        converter = self._converters[channel]
        return torch.from_numpy(
            converter(self.input_arrays[channel][plane, ...])
        )


class CubeDatasetBase(Dataset):
    """
    Input voxel order is Width, Height, Depth
    Returned data order is (Batch) Height, Width, Depth, Channel.
    """

    full_cuboid_size: tuple[int, int, int, int] = 1, 1, 1, 1

    def __init__(
        self,
        points: list[Cell],
        data_voxel_sizes: tuple[int, int, int],
        network_voxel_sizes: tuple[int, int, int],
        network_cuboid_voxels: tuple[int, int, int] = (20, 50, 50),
        axis_order: tuple[AXIS, AXIS, AXIS] = ("z", "y", "x"),
        output_axis_order: tuple[AXIS, AXIS, AXIS] = ("y", "x", "z", "c"),
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
        self.network_cuboid_voxels = network_cuboid_voxels
        self.axis_order = axis_order
        self.output_axis_order = output_axis_order

        self.data_cuboid_voxels = []
        for data_um, network_um, cuboid_voxels in zip(
            data_voxel_sizes, network_voxel_sizes, network_cuboid_voxels
        ):
            self.data_cuboid_voxels.append(
                get_data_cuboid_voxels(cuboid_voxels, network_um, data_um)
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
        scaled_cuboid_size = [
            self.network_cuboid_voxels[i] for i in torch_voxel_map
        ]

        data = torch.permute(
            data, get_axis_reordering(data_order, torch_order)
        )
        assert len(data.shape) == 5, "expected (batch, channel, z, y, x) shape"
        data = F.interpolate(data, size=scaled_cuboid_size, mode="trilinear")

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

    _dataset_thread: ThreadWithException | None = None

    _worker_queues: list[Queue] = None

    def __init__(
        self,
        signal_array: types.array,
        background_array: types.array,
        max_axis_0_cuboids_buffered: float = 0,
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
        self._worker_queues = []

        data_arrays = [signal_array, background_array]
        self.cached_dataset = CachedStackDataset(
            input_arrays=data_arrays,
            data_axis_order=self.axis_order,
            max_axis_0_cuboids_buffered=max_axis_0_cuboids_buffered,
            cuboid_size=self.data_cuboid_voxels,
        )

        data_axis_order = self.cached_dataset.full_data_axis_order
        self._data_axis_reordering = None
        if data_axis_order != self.output_axis_order:
            self._data_axis_reordering = get_axis_reordering(
                ("b", *data_axis_order), ("b", *self.output_axis_order)
            )

        self.points = [p for p in self.points if self.point_has_full_cuboid(p)]
        self.full_cuboid_size = self.cached_dataset.full_cuboid_size

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["cached_dataset"]
        return state

    def point_has_full_cuboid(self, point: Cell) -> bool:
        for ax, data_size, cuboid_size in zip(
            self.axis_order, self.dataset_shape, self.data_cuboid_voxels
        ):
            start, end = get_data_cuboid_range(
                getattr(point, ax), cuboid_size, ax
            )
            if start < 0:
                return False
            if end > data_size:
                # if it's data_size it's fine because end is not inclusive
                return False

        return True

    def get_cell_data(self, cell: Cell) -> torch.Tensor:
        return self.get_cells_data([cell])[0, ...]

    def get_cells_data(self, cells: list[Cell]) -> torch.Tensor:
        data = torch.empty(
            (len(cells), *self.full_cuboid_size),
            dtype=torch.float32,
        )

        thread = self._dataset_thread
        if thread is None:
            self.cached_dataset.get_cell_batch_cuboid_data(data, cells)
        else:
            queues = self._worker_queues
            # for host, it's the last queue
            queue_id = len(queues) - 1
            if get_worker_info() is not None:
                queue_id = get_worker_info().id
            queue = queues[queue_id]

            thread.send_msg_to_thread(("cells", cells, data, queue_id))
            msg, value = queue.get(block=True)
            # if there's no error, we just sent back the cell
            if msg == "exception":
                raise ExecutionFailure(
                    "Reporting failure from data thread"
                ) from value
            assert msg == "cells"

        if self._data_axis_reordering is not None:
            data = torch.permute(data, self._data_axis_reordering)
        return data

    def start_dataset_thread(self, num_workers: int):
        # include queue for host thread
        ctx = mp.get_context("spawn")
        queues = [ctx.Queue(maxsize=0) for _ in range(num_workers + 1)]
        self._worker_queues = queues

        self._dataset_thread = ThreadWithException(
            target=_read_planes_send_cuboids,
            args=(self.cached_dataset, queues),
            pass_self=True,
        )
        self._dataset_thread.start()

    def stop_dataset_thread(self):
        thread = self._dataset_thread
        if thread is None:
            return

        self._dataset_thread = None
        self._worker_queues = []

        thread.notify_to_end_thread()
        thread.clear_remaining()
        thread.join()


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
