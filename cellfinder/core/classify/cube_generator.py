import math
from collections import OrderedDict, defaultdict
from collections.abc import Sequence
from numbers import Integral
from typing import Any, Hashable, Literal

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from brainglobe_utils.cells.cells import Cell
from tifffile import imread
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


def _read_data_send_cuboids(
    thread: ThreadWithException,
    dataset: "ImageDataBase",
    queues: list[Queue],
) -> None:
    while True:
        msg = thread.get_msg_from_mainthread()
        if msg == EOFSignal:
            return

        key, point, buffer, queue_id = msg
        queue = queues[queue_id]
        try:
            if key == "point":
                buffer[:] = dataset.get_point_cuboid_data(point)
            elif key == "points":
                dataset.get_point_batch_cuboid_data(buffer, point)
            else:
                raise ValueError(f"Bad message {key}")
        except BaseException as e:
            queue.put(("exception", e), block=True, timeout=None)
        else:
            queue.put((key, point), block=True, timeout=None)


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


class ImageDataBase:
    """
    We buffer along axis 0, which is the data_axis_order[0] axis.
    The size of each cuboid in voxels of the dataset is given by cuboid_size.
    Data returned is in data_axis_order, with channel the last axis
    """

    num_channels: int = 1

    cuboid_size: tuple[int, int, int] = (1, 1, 1)
    # in units of voxels, not um. in data_axis_order

    data_axis_order: tuple[AXIS, AXIS, AXIS] = ("z", "y", "x")

    data_with_channels_axis_order: tuple[DIM, DIM, DIM, DIM] = (
        "z",
        "y",
        "x",
        "c",
    )

    def __init__(
        self,
        data_axis_order: tuple[AXIS, AXIS, AXIS] = ("z", "y", "x"),
        cuboid_size: tuple[int, int, int] = (1, 1, 1),
    ):
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
        self.data_with_channels_axis_order = *data_axis_order, "c"

    @property
    def cuboid_with_channels_size(self) -> tuple[int, int, int, int]:
        return *self.cuboid_size, self.num_channels

    def get_point_cuboid_data(self, point_key: int) -> torch.Tensor:
        raise NotImplementedError

    def get_point_batch_cuboid_data(
        self, batch: torch.Tensor, points_key: list[int]
    ) -> None:
        raise NotImplementedError


class CachedStackImageDataBase(ImageDataBase):

    points: list[Cell] = []

    stack_shape: tuple[int, int, int]

    max_axis_0_planes_buffered: int = 0

    _planes_buffer: OrderedDict[int, torch.Tensor]

    def __init__(
        self,
        points: list[Cell],
        max_axis_0_cuboids_buffered: float = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.points = points
        self.max_axis_0_planes_buffered = int(
            round((max_axis_0_cuboids_buffered + 1) * self.cuboid_size[0])
        )

        self._planes_buffer = OrderedDict()

    def get_point_cuboid_data(self, point_key: int) -> torch.Tensor:
        data = torch.empty(self.cuboid_with_channels_size, dtype=torch.float32)

        point = self.points[point_key]
        for j, plane in enumerate(self._get_cuboid_planes(point)):
            data[j, ...] = plane

        return data

    def get_point_batch_cuboid_data(
        self, batch: torch.Tensor, points_key: list[int]
    ) -> None:
        if len(points_key) != batch.shape[0]:
            raise ValueError(
                "Expected the number of points to match the batch size"
            )

        points = self.points
        for i, point_key in enumerate(points_key):
            point = points[point_key]
            for j, plane in enumerate(self._get_cuboid_planes(point)):
                batch[i, j, ...] = plane

    def _get_cuboid_planes(self, point: Cell) -> list[torch.Tensor]:
        max_planes = self.max_axis_0_planes_buffered
        planes_buffer = self._planes_buffer

        cuboid_indices = []
        for ax, size in zip(self.data_axis_order, self.cuboid_size):
            cuboid_indices.append(
                get_data_cuboid_range(getattr(point, ax), size, ax)
            )
        ax1 = slice(*cuboid_indices[1])
        ax2 = slice(*cuboid_indices[2])

        planes = []
        # buffered axis
        for i in range(*cuboid_indices[0]):
            if i not in planes_buffer:
                plane_shape = *self.stack_shape[1:], self.num_channels
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


class CachedArrayStackImageData(CachedStackImageDataBase):

    input_arrays: list[types.array]

    def __init__(
        self,
        input_arrays: list[types.array],
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.input_arrays = input_arrays
        self.stack_shape = input_arrays[0].shape
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


class CachedCuboidImageDataBase(ImageDataBase):

    max_cuboids_buffered: int = 0

    _cuboids_buffer: OrderedDict[Hashable, torch.Tensor]

    def __init__(
        self,
        max_cuboids_buffered: int = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.max_cuboids_buffered = max_cuboids_buffered
        self._cuboids_buffer = OrderedDict()

    def get_point_cuboid_data(self, point_key: int) -> torch.Tensor:
        max_cuboids = self.max_cuboids_buffered
        cuboids_buffer = self._cuboids_buffer

        if point_key not in cuboids_buffer:
            cuboid = torch.empty(
                self.cuboid_with_channels_size, dtype=torch.float32
            )
            for channel in range(self.num_channels):
                cuboid[:, :, :, channel] = self.read_cuboid(point_key, channel)

            if not max_cuboids:
                return cuboid

            if len(cuboids_buffer) == max_cuboids:
                # fifo when last=False
                cuboids_buffer.popitem(last=False)
            assert len(cuboids_buffer) < max_cuboids

            cuboids_buffer[point_key] = cuboid

        return cuboids_buffer[point_key]

    def get_point_batch_cuboid_data(
        self, batch: torch.Tensor, points_key: list[int]
    ) -> None:
        if len(points_key) != batch.shape[0]:
            raise ValueError(
                "Expected the number of points to match the batch size"
            )

        for i, point_key in enumerate(points_key):
            batch[i, ...] = self.get_point_cuboid_data(point_key)

    def read_cuboid(self, point_key: int, channel: int) -> torch.Tensor:
        raise NotImplementedError


class CachedTiffCuboidImageData(CachedCuboidImageDataBase):

    points: list[Cell] = []

    filenames: Sequence[Sequence[str]] = []

    def __init__(
        self,
        points: list[Cell],
        filenames: Sequence[Sequence[str]],
        **kwargs,
    ):
        super().__init__(**kwargs)
        if not filenames:
            raise ValueError("No data was provided")
        if len(filenames) != len(points):
            raise ValueError(
                "Points and filenames must have same number of elements"
            )

        self.points = points
        self.filenames = filenames
        self.num_channels = len(filenames[0])
        self._converters = [
            get_data_converter(imread(channel_filenames).dtype, np.float32)
            for channel_filenames in filenames[0]
        ]

    def read_cuboid(self, point_key: int, channel: int) -> torch.Tensor:
        converter = self._converters[channel]
        data = imread(self.filenames[point_key][channel])
        return torch.from_numpy(converter(data))


class CuboidDatasetBase(Dataset):
    """
    Input voxel order is Width, Height, Depth
    Returned data order is (Batch) Height, Width, Depth, Channel.
    """

    num_channels: int = 1

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

    @property
    def cuboid_with_channels_size(self) -> tuple[int, int, int, int]:
        return *self.data_cuboid_voxels, self.num_channels

    def __len__(self):
        return len(self.points)

    def __getitem__(self, idx):
        if isinstance(idx, Integral):
            return self._get_single_item(idx)
        return self._get_multiple_items(idx)

    def _get_single_item(
        self, idx: int
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        point = self.points[idx]
        data = self.get_point_data(idx)

        data = self.rescale_to_output_size(data)
        if self.transform:
            data = self.transform(data)

        match self.target_output:
            case None:
                return data

            case "cell":
                label = point
            case "dict":
                label = point.to_dict()
            case "label":
                cls = torch.tensor(point.type - 1)
                label = F.one_hot(cls, num_classes=self.classes)
            case _:
                raise ValueError(f"Unknown target value {self.target_output}")

        return data, label

    def _get_multiple_items(
        self, indices: list[int]
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        points = [self.points[i] for i in indices]
        data = self.get_points_data(indices)

        data = self.rescale_to_output_size(data)
        if self.transform:
            data = self.transform(data)

        match self.target_output:
            case None:
                return data

            case "cell":
                labels = points
            case "dict":
                labels = [point.to_dict() for point in points]
            case "label":
                cls = torch.tensor([point.type - 1 for point in points])
                labels = F.one_hot(cls, num_classes=self.classes)
            case _:
                raise ValueError(f"Unknown target value {self.target_output}")

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

    def get_point_data(self, point_key: int) -> torch.Tensor:
        """

        :param point_key:
        :return: In the output_axis_order order.
        """
        raise NotImplementedError

    def get_points_data(self, points_key: list[int]) -> torch.Tensor:
        """

        :param points_key:
        :return: In the output_axis_order order.
        """
        raise NotImplementedError


class CuboidThreadedDatasetBase(CuboidDatasetBase):

    src_dataset: ImageDataBase | None = None

    _dataset_thread: ThreadWithException | None = None

    _worker_queues: list[Queue] = None

    _data_axis_reordering: list[int] | None = None

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._worker_queues = []

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["src_dataset"]
        return state

    def _set_data_axis_reordering(self) -> None | list[int]:
        data_axis_order = self.src_dataset.data_with_channels_axis_order
        if data_axis_order == self.output_axis_order:
            self._data_axis_reordering = None

        self._data_axis_reordering = get_axis_reordering(
            ("b", *data_axis_order), ("b", *self.output_axis_order)
        )

    def get_point_data(self, point_key: int) -> torch.Tensor:
        return self.get_points_data([point_key])[0, ...]

    def get_points_data(self, points_key: list[int]) -> torch.Tensor:
        data = torch.empty(
            (len(points_key), *self.cuboid_with_channels_size),
            dtype=torch.float32,
        )

        thread = self._dataset_thread
        if thread is None:
            self.src_dataset.get_point_batch_cuboid_data(data, points_key)
        else:
            queues = self._worker_queues
            # for host, it's the last queue
            queue_id = len(queues) - 1
            if get_worker_info() is not None:
                queue_id = get_worker_info().id
            queue = queues[queue_id]

            thread.send_msg_to_thread(("points", points_key, data, queue_id))
            msg, value = queue.get(block=True)
            # if there's no error, we just sent back the point
            if msg == "exception":
                raise ExecutionFailure(
                    "Reporting failure from data thread"
                ) from value
            assert msg == "points"

        if self._data_axis_reordering is not None:
            data = torch.permute(data, self._data_axis_reordering)
        return data

    def start_dataset_thread(self, num_workers: int):
        # include queue for host thread
        ctx = mp.get_context("spawn")
        queues = [ctx.Queue(maxsize=0) for _ in range(num_workers + 1)]
        self._worker_queues = queues

        self._dataset_thread = ThreadWithException(
            target=_read_data_send_cuboids,
            args=(self.src_dataset, queues),
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


class CuboidStackDataset(CuboidThreadedDatasetBase):

    def __init__(
        self,
        signal_array: types.array,
        background_array: types.array,
        max_axis_0_cuboids_buffered: float = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_channels = 2

        if signal_array.shape != background_array.shape:
            raise ValueError(
                f"Shape of signal images ({signal_array.shape}) does not "
                f"match the shape of the background images "
                f"({background_array.shape}"
            )
        if len(signal_array.shape) != 3:
            raise ValueError("Expected a 3d in data array")

        self.stack_shape = signal_array.shape

        self.points = [p for p in self.points if self.point_has_full_cuboid(p)]

        data_arrays = [signal_array, background_array]
        self.src_dataset = CachedArrayStackImageData(
            points=self.points,
            input_arrays=data_arrays,
            data_axis_order=self.axis_order,
            max_axis_0_cuboids_buffered=max_axis_0_cuboids_buffered,
            cuboid_size=self.data_cuboid_voxels,
        )
        self._set_data_axis_reordering()

    def point_has_full_cuboid(self, point: Cell) -> bool:
        for ax, axis_size, cuboid_size in zip(
            self.axis_order, self.stack_shape, self.data_cuboid_voxels
        ):
            start, end = get_data_cuboid_range(
                getattr(point, ax), cuboid_size, ax
            )
            if start < 0:
                return False
            if end > axis_size:
                # if it's axis_size it's fine because end is not inclusive
                return False

        return True


class CuboidTiffDataset(CuboidThreadedDatasetBase):

    def __init__(
        self,
        points_filenames: Sequence[Sequence[str]],
        max_cuboids_buffered: int = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if not len(points_filenames):
            raise ValueError("No data provided")
        if len(points_filenames) != len(self.points):
            raise ValueError(
                "Points and filenames must have same number of elements"
            )

        self.num_channels = len(points_filenames[0])
        self.src_dataset = CachedTiffCuboidImageData(
            points=self.points,
            filenames=points_filenames,
            data_axis_order=self.axis_order,
            max_cuboids_buffered=max_cuboids_buffered,
            cuboid_size=self.data_cuboid_voxels,
        )
        self._set_data_axis_reordering()


class CuboidBatchSampler(Sampler):

    def __init__(
        self,
        dataset: CuboidDatasetBase,
        batch_size: int,
        auto_shuffle: bool = False,
        sort_by_axis: str | None = None,
        segregate_by_axis: bool = False,
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
