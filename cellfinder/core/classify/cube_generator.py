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
from cellfinder.core.classify.augment import DataAugmentation
from cellfinder.core.tools.threading import (
    EOFSignal,
    ExecutionFailure,
    ThreadWithException,
)
from cellfinder.core.tools.tools import get_axis_reordering, get_data_converter

AXIS = Literal["x", "y", "z"]
DIM = Literal[AXIS, "c"]
RandRange = Sequence[float] | Sequence[tuple[float, float]] | None


class StackSizeError(Exception):
    pass


def _read_data_send_cuboids(
    thread: ThreadWithException,
    dataset: "ImageDataBase",
    queues: list[Queue],
) -> None:
    """
    Function run by sub-thread that reads data from a dataset, extracts the
    cuboids, and sends them back to the main thread for further processing.

    Each thread listens on its own main queue, via its
    `thread.get_msg_from_mainthread`, for requests. We also pass in a list of
    response queues to which the response of a given request is sent. Each
    request indicates which queue in the list to send back the read data.

    Requests also pass in a torch buffer into which we read the cube data,
    saving us having to send back newly allocated buffers.

    If there's an exception serving a particular request, the indicated queue
    is sent back the exception, instead of the normal data response.

    :param thread: The `ThreadWithException`, automatically passed to the func.
    :param dataset: The `ImageDataBase` used to read the data and cubes.
    :param queues: A list of queues for sending data. Each request to the
        sub-thread indicates which queue in the list to send the cubes. This
        allows one thread to serve multiple consumers.
    """
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
    """In a given dimension, the network is trained with
    `network_cuboid_voxels` voxels and at `network_voxel_size_um` um per voxel.
    This returns the corresponding number of voxels of the input data with its
    `data_voxel_size_um` um per voxel.

    :param network_cuboid_voxels: The trained network's cube number of voxels
        in that dimension.
    :param network_voxel_size_um: The network's um per voxel in the dim.
    :param data_voxel_size_um: The input data's um per voxel in the dim.
    """
    return int(
        round(
            network_cuboid_voxels * network_voxel_size_um / data_voxel_size_um
        )
    )


def get_data_cuboid_range(
    pos: float, num_voxels: int, axis: AXIS
) -> tuple[int, int]:
    """For a given dim, takes the location of a point in the input data and
    returns the start and end index in the data that centers a cube on the
    point.

    :param pos: The position, in voxels, of the point in the input data. For
        the given dim.
    :param num_voxels: The number of voxels of the cube in this dim.
    :param axis: The name of the axis we're calculating. Can be the str
        `"x"`, `"y"`, or `"z"`. The axes are centered differently for backward
        compatibility.
    :return: tuple of ints, `(start, end)`. Cube can be extracted then with
        `data[start:end]`.
    """
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
    A base class for extracting cuboids out of image data.

    At the base level we initialize with an array of 3d positions (center
    location of potential cells) and we can return a cuboid centered on a
    position, given its index in the list.

    The returned cuboid is of size `cuboid_with_channels_size` with the
    corresponding axis order given by `data_with_channels_axis_order`. These
    correspond to the provided `cuboid_size` and `data_axis_order`, only with
    the addition of the channels dimension.
    """

    points_arr: np.ndarray = None
    """An Nx3 array. Each row is a 3d point with the position of a potential
    cell. Units are voxels of the input data.
    """

    num_channels: int = 1
    """The number of channels contained in the input data."""

    cuboid_size: tuple[int, int, int] = (1, 1, 1)
    """Size of the cuboid in 3d. Cuboids of this size, centered at a given
    point will be returned.

    Units are voxels in the input data. The axis order corresponds
    to `data_axis_order`.
    """

    data_axis_order: tuple[AXIS, AXIS, AXIS] = ("z", "y", "x")
    """The axis order of the input date. It's a tuple of `"x"`, `"y"`, and
    `"z"` matching the dim order of the input data.
    """

    def __init__(
        self,
        points_arr: np.ndarray,
        data_axis_order: tuple[AXIS, AXIS, AXIS] = ("z", "y", "x"),
        cuboid_size: tuple[int, int, int] = (1, 1, 1),
    ):
        self.points_arr = points_arr
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

    @property
    def data_with_channels_axis_order(self) -> tuple[DIM, DIM, DIM, DIM]:
        """Same as `data_axis_order`, but it's a 4-tuple because we also
        include `"c"`. The output cube ordering is 4 dimensional, x, y, z,
        and c. This specifies the order.

        By default, it's just `data_axis_order` plus `c` at
        the end. But it could be different for different loaders.
        """
        return *self.data_axis_order, "c"

    @property
    def cuboid_with_channels_size(self) -> tuple[int, int, int, int]:
        """Similar to `cuboid_size`, but it also includes channels size."""
        return *self.cuboid_size, self.num_channels

    def get_point_cuboid_data(self, point_key: int) -> torch.Tensor:
        """
        Takes a key used to identify a specific point and returns the cuboid
        centered around the point.

        :param point_key: A unique key used to identify the point. E.g. the
            index in `points_arr`.
        :return: The torch Tensor of size `cuboid_with_channels_size` centered
            on the point.
        """
        raise NotImplementedError

    def get_point_batch_cuboid_data(
        self, batch: torch.Tensor, points_key: list[int]
    ) -> None:
        """
        Similar to `get_point_cuboid_data` except it passes a sequence of keys
        that identifies a list of points. We then fill `batch` with the cuboids
        centered around the points.

        :param batch: A 5d torch tensor of size N x cuboid_with_channels_size
            into which the cuboids corresponding to the points will be filled
            into. N is the number of points (the length of `points_key`) and is
            the first dimension.
        :param points_key: A list of unique keys used to identify the points.
            E.g. their indices in `points_arr`.
        """
        raise NotImplementedError


class CachedStackImageDataBase(ImageDataBase):
    """
    Takes a 3d image stack (potentially a folder of tiffs or a 3d numpy array)
    and extracts requested cuboids from the stack as torch Tensors. It also
    buffers some planes (at the first axis of the data stack) in memory so they
    don't have to be repeatedly read.

    This is especially efficient when we request cuboids sequentially ordered
    by the first axis. E.g. if the first axis is z, then requesting cuboids
    ordered by increasing z will be very fast. With a buffer size at least as
    large as the number of processing workers, even if we read z planes
    slightly out of order e.g. if multiple workers request them in parallel
    which are serialized slightly out of order, it should still be very fast.
    """

    stack_shape: tuple[int, int, int]
    """
    The 3d input array size. Its axis order is `data_axis_order`.
    """

    max_axis_0_planes_buffered: int = 0
    """
    The number of planes to buffer in memory for the first axis, in addition to
    the number of planes in a cube corresponding to this axis. This
    corresponds to the first axis in `data_axis_order`. The assumption is that
    data is read from disk in planes of that axis.

    A good default is the max of the number of planes in a cube and the number
    of workers used by the torch data loaders.
    """

    _planes_buffer: OrderedDict[int, torch.Tensor]
    """
    Cache that maps plane numbers to the plane tensor. Plane axis and max dict
    size is as in `max_axis_0_planes_buffered`.
    """

    def __init__(
        self,
        max_axis_0_cuboids_buffered: float = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.max_axis_0_planes_buffered = int(
            round((max_axis_0_cuboids_buffered + 1) * self.cuboid_size[0])
        )

        self._planes_buffer = OrderedDict()

    def get_point_cuboid_data(self, point_key: int) -> torch.Tensor:
        """
        See base-class, except `point_key` *is* the plane index in the array of
        planes.
        """
        data = torch.empty(self.cuboid_with_channels_size, dtype=torch.float32)

        point = self.points_arr[point_key]
        for j, plane in enumerate(self._get_cuboid_planes(point)):
            data[j, ...] = plane

        return data

    def get_point_batch_cuboid_data(
        self, batch: torch.Tensor, points_key: list[int]
    ) -> None:
        """
        See base-class, except `points_key` *is* the plane index in the array
        of planes.
        """
        if len(points_key) != batch.shape[0]:
            raise ValueError(
                "Expected the number of points to match the batch size"
            )

        points_arr = self.points_arr
        for i, point_key in enumerate(points_key):
            point = points_arr[point_key]
            for j, plane in enumerate(self._get_cuboid_planes(point)):
                batch[i, j, ...] = plane

    def _get_cuboid_planes(self, point: np.ndarray) -> list[torch.Tensor]:
        """
        Takes a 3d point and returns a list of sequential planes, where when
        concatenated in the first axis yields a cube of the correct size.
        This is more efficient, for the calling function to copy plane by plane
        into its buffer than use concatenating and calling function copying.
        """
        max_planes = self.max_axis_0_planes_buffered
        planes_buffer = self._planes_buffer

        cuboid_indices = []
        for ax, size in zip(self.data_axis_order, self.cuboid_size):
            cuboid_indices.append(get_data_cuboid_range(point[ax], size, ax))
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
        """
        Takes a plane number along the first axis and a channel number and
        it returns the torch tensor containing that plane for that channel.

        :param plane: The plane index in the first dim of `stack_shape` to
            read.
        :param channel: The channel number `<num_channels` we want to read.
        :return: The plane data as a tensor.
        """
        raise NotImplementedError


class CachedArrayStackImageData(CachedStackImageDataBase):
    """
    Implements `CachedStackImageDataBase` for a stack represented by an array
    data type. E.g. an Dask array.
    """

    input_arrays: list[types.array]
    """
    List of the arrays. Each items corresponds to a channel.
    """

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
    """
    Takes a collection of cuboids (e.g. a folder of tiff files, each a cuboid
    or a list of 3d numpy arrays) and returns a requested cuboid as a torch
    Tensor.

    It also buffers `max_cuboids_buffered` recent cuboids so they
    don't have to be repeatedly read.
    """

    max_cuboids_buffered: int = 0
    """
    The number of most recently read cuboids to keep in memory.
    """

    _cuboids_buffer: OrderedDict[Hashable, torch.Tensor]
    """
    Maps the cuboids hashable point keys to the cuboids. The cuboid buffered
    includes the channels in the `data_with_channels_axis_order` order.
    """

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
        """
        Takes a key used to identify a point and a channel number and
        returns the torch tensor containing the cuboid for that channel.

        :param point_key: A key used to identify the point.
        :param channel: The channel number `<num_channels` we want to read.
        :return: The cuboid's data as a tensor.
        """
        raise NotImplementedError


class CachedTiffCuboidImageData(CachedCuboidImageDataBase):
    """
    Implements `CachedCuboidImageDataBase` for a list of tiff filenames. Each
    tiff contains a single channel 3d cuboid.
    """

    filenames_arr: np.ndarray = None
    """
    A 2d numpy array containing the list of tiff filenames. Each item is the
    list of the filenames of that point's channels.
    """

    def __init__(
        self,
        filenames_arr: np.ndarray,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if not len(filenames_arr):
            raise ValueError("No data was provided")
        if len(filenames_arr) != len(self.points_arr):
            raise ValueError(
                "Points and filenames must have same number of elements"
            )

        self.filenames_arr = filenames_arr
        self.num_channels = len(filenames_arr[0])
        self._converters = [
            get_data_converter(
                imread(str(channel_filenames)).dtype, np.float32
            )
            for channel_filenames in filenames_arr[0]
        ]

    def read_cuboid(self, point_key: int, channel: int) -> torch.Tensor:
        """
        See super-class. `point_key` is the cuboid's index in `filenames_arr`
        that we want to read.
        """
        converter = self._converters[channel]
        data = imread(str(self.filenames_arr[point_key][channel]))
        return torch.from_numpy(converter(data))


class CuboidDatasetBase(Dataset):
    """
    Implements a pytorch `Dataset` that takes a list of 3d point coordinates
    (centers of potential cells) and a `ImageDataBase` instance that contains
    voxel data. The dataset yields batches of torch Tensors with cuboids
    of the voxel data centered at these points.

    Data is accessed similar to normal torch Dataset. With e.g. `len(dataset)`
    or `dataset[i]`. If `i` is a single int it returns the corresponding 4d
    item (includes channel), otherwise it's a sequence of ints and it returns
    a 5d batch of those items. The batch dimension is always the first
    dimension followed by `output_axis_order` dimensions.

    The data output is either just the data or it also includes the labels,
    depending on `target_output`.

    :param points: A list of `Cell`instances containing the cell centers.
    :param data_voxel_sizes: A 3-tuple indicating the input data's 3d voxel
        size in `um`. The tuple's order corresponds to `axis_order`.
    :param network_voxel_sizes: A 3-tuple indicating the trained network's 3d
        voxel size in `um`. The tuple's order corresponds to `axis_order`.
    :param network_cuboid_voxels: A 3-tuple indicating the cuboid size used to
        train the network in voxels The tuple's order corresponds to
        `axis_order`.
    :param axis_order: A 3-tuple indicating the input data's three
        dimensions' axis order. It's any permutations of `("x", "y", "z")`.
    :param output_axis_order: A 4-tuple indicating the desired output data's
        three dimensions plus channel axis order. It's any permutations of
        `("x", "y", "z", "c")`. For now, `"c"` is assumed last.
    :param src_image_data: The `ImageDataBase` that will be used to read the
        voxel cuboids for a given point.
    :param classes: The number of classes used by the network when classifying
        cuboids.
    :param target_output: A literal indicating the type of label output dataset
        should return during training / testing. It is one of `"cell"`
        (it returns the `Cell` instance), `"label"` (it returns a one-hot
        vector labeling `Cell.type - 1` as the correct class), or None (not
        label is returned).

        I.e. if it's None, we do `batch_data = dataset[i]`. Otherwise, it's
        `batch_data, batche_label = dataset[i]`.
    :param augment: Whether to augment the dataset with the subsequent
        parameters.
    :param augment_likelihood: Value `[0, 1]` with the probability of a data
        item being augmented. I.e. `0.9` means 90% of the data will have been
        augmented.
    :param flippable_axis: A sequence of the dimensions in the output
        data to reverse, if any, with probability `augment_likelihood`.
    :param rotate_range: A sequence of floats or sequence of 2-tuples with the
        radian angle or range of angles to rotate the output data. Each item
        for the corresponding output data dim, with probability
        `augment_likelihood`. Or `None` if there's no rotation.
    :param translate_range: A sequence of floats or sequence of 2-tuples with
        the pixel distance or range of distance to translate the output data.
        Each item for the corresponding output data dim, with probability
        `augment_likelihood`. Or `None` if there's no translation.
    :param scale_range: A sequence of floats or sequence of 2-tuples with
        the amount or range of amount to scale the output data. `1` means
        no scaling. Each item for the corresponding output data dim, with
        probability `augment_likelihood`. Or `None` if there's no scaling.
    """

    points_arr: np.ndarray = None
    """A generated Nx4 array. Each row is `(x, y, z, type)` with the 3d
    position of the potential cell and the type of the point (`Cell.type`).

    Units are voxels of the input data (`data_voxel_sizes`).
    """

    src_image_data: ImageDataBase | None = None
    """
    The `ImageDataBase` that will be used to read the voxel cuboids for a given
    point.
    """

    data_cuboid_voxels: tuple[int, int, int] = 1, 1, 1
    """A 3-tuple of the number of voxels in a cuboid of the output data.

    The order corresponds to the `output_axis_order`, excluding the channel
    dim. See also `cuboid_with_channels_size`.
    """

    num_channels: int = 1
    """
    The number of channels in the image data / cuboids.
    """

    augmentation: DataAugmentation | None = None
    """
    If provided, used to augment the data during training.
    """

    _output_data_dim_reordering: list[int] | None = None
    """
    Cached indices that is used with `get_axis_reordering` to convert the
    cuboids from the input `axis_order` /
    `src_image_data.data_with_channels_axis_order` to the `output_axis_order`.
    """

    def __init__(
        self,
        points: list[Cell],
        data_voxel_sizes: tuple[float, float, float],
        network_voxel_sizes: tuple[float, float, float],
        network_cuboid_voxels: tuple[int, int, int] = (20, 50, 50),
        axis_order: tuple[AXIS, AXIS, AXIS] = ("z", "y", "x"),
        output_axis_order: tuple[DIM, DIM, DIM, DIM] = ("y", "x", "z", "c"),
        src_image_data: ImageDataBase | None = None,
        classes: int = 2,
        target_output: Literal["cell", "label"] | None = None,
        augment: bool = False,
        augment_likelihood: float = 0.9,
        flippable_axis: Sequence[int] = (0, 1, 2),
        rotate_range: RandRange = (math.pi / 4,) * 3,
        translate_range: RandRange = (0.05,) * 3,
        scale_range: RandRange = ((0.6, 1.4),) * 3,
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

        self.points_arr = np.empty(
            len(points),
            dtype=[("x", "<f8"), ("y", "<f8"), ("z", "<f8"), ("type", "<i8")],
        )
        data = self.points_arr
        for i, cell in enumerate(points):
            data[i]["x"] = cell.x
            data[i]["y"] = cell.y
            data[i]["z"] = cell.z
            data[i]["type"] = cell.type

        self.src_image_data = src_image_data
        self.data_voxel_sizes = data_voxel_sizes
        self.network_voxel_sizes = network_voxel_sizes
        self.network_cuboid_voxels = network_cuboid_voxels
        self.axis_order = axis_order
        self.output_axis_order = output_axis_order

        data_cuboid_voxels = []
        for data_um, network_um, cuboid_voxels in zip(
            data_voxel_sizes, network_voxel_sizes, network_cuboid_voxels
        ):
            data_cuboid_voxels.append(
                get_data_cuboid_voxels(cuboid_voxels, network_um, data_um)
            )
        self.data_cuboid_voxels = tuple(data_cuboid_voxels)

        self.classes = classes
        self.target_output = target_output

        if augment:
            vol_size = {
                ax: n for ax, n in zip(axis_order, network_cuboid_voxels)
            }
            self.augmentation = DataAugmentation(
                vol_size,
                output_axis_order,
                augment_likelihood,
                flippable_axis,
                translate_range,
                scale_range,
                rotate_range,
            )

        if src_image_data is not None:
            self._set_output_data_dim_reordering(src_image_data)

    @property
    def cuboid_with_channels_size(self) -> tuple[int, int, int, int]:
        """A 4-tuple of the number of voxels in the cuboid in the output data
        and the number of channels.

        The order corresponds to the `output_axis_order`. For now, `"c"` is
        assumed last.
        """
        return *self.data_cuboid_voxels, self.num_channels

    def __len__(self):
        return len(self.points_arr)

    def __getitem__(self, idx):
        if isinstance(idx, Integral):
            return self._get_single_item(idx)
        return self._get_multiple_items(idx)

    def _set_output_data_dim_reordering(
        self, src_image_data: ImageDataBase
    ) -> None:
        """
        Sets `_output_data_dim_reordering`.
        """
        if src_image_data.data_axis_order != self.output_axis_order:
            self._output_data_dim_reordering = get_axis_reordering(
                ("b", *src_image_data.data_with_channels_axis_order),
                ("b", *self.output_axis_order),
            )

    def _get_single_item(
        self, idx: int
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        """
        Handles `dataset[i]`, when `i` is an int.
        """
        point = self.points_arr[idx]
        data = self.get_point_data(idx)

        # batch dim
        data = data[None, ...]
        if self._output_data_dim_reordering is not None:
            data = torch.permute(data, self._output_data_dim_reordering)

        data = self.rescale_to_output_size(data)
        data = data[0, ...]

        augmentation = self.augmentation
        if augmentation is not None and augmentation.update_parameters():
            data[:] = augmentation(data)

        match self.target_output:
            case None:
                return data

            case "cell":
                label = Cell(
                    pos=[
                        float(point["x"]),
                        float(point["y"]),
                        float(point["z"]),
                    ],
                    cell_type=int(point["type"]),
                )
            case "label":
                cls = torch.tensor(point["type"] - 1)
                label = F.one_hot(cls, num_classes=self.classes)
            case _:
                raise ValueError(f"Unknown target value {self.target_output}")

        return data, label

    def _get_multiple_items(
        self, indices: list[int]
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        """
        Handles `dataset[i]`, when `i` is a list of ints.
        """
        points = self.points_arr[indices]

        data = self.get_points_data(indices)
        if self._output_data_dim_reordering is not None:
            data = torch.permute(data, self._output_data_dim_reordering)
        data = self.rescale_to_output_size(data)

        augmentation = self.augmentation
        if augmentation is not None:
            # batch is always first index
            for b in range(len(indices)):
                if augmentation.update_parameters():
                    data[b, ...] = augmentation(data[b, ...])

        match self.target_output:
            case None:
                return data

            case "cell":
                labels = [
                    Cell(
                        pos=[
                            float(point["x"]),
                            float(point["y"]),
                            float(point["z"]),
                        ],
                        cell_type=int(point["type"]),
                    )
                    for point in points
                ]
            case "label":
                cls = torch.tensor([point["type"] - 1 for point in points])
                labels = F.one_hot(cls, num_classes=self.classes)
            case _:
                raise ValueError(f"Unknown target value {self.target_output}")

        return data, labels

    def rescale_to_output_size(self, data: torch.Tensor) -> torch.Tensor:
        """
        Takes the input cuboids, ordered already according to the
        `output_axis_order` with the additional batch dim at the start and
        scales the cuboids to the network cuboid size.
        """
        if self.data_voxel_sizes == self.network_voxel_sizes:
            return data

        # batch dimension
        if len(data.shape) != 5:
            raise ValueError("Needs 5 dimensions: batch, channel and space")

        # our data comes in in output_axis_order order. To scale we need to
        # convert first to torch_order, which is torch's expected order. We
        # then re-order back to the output_axis_order before returning.
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
        data = F.interpolate(data, size=scaled_cuboid_size, mode="trilinear")
        data = torch.permute(
            data, get_axis_reordering(torch_order, data_order)
        )

        return data

    def get_point_data(self, point_key: int) -> torch.Tensor:
        """
        Takes a key used to identify a specific point (typically the index in
        the points list) and returns the cuboid centered around the point.

        This handles getting the cuboids for `dataset[i]`, when `i` is an int.

        :param point_key: A unique key used to identify the point. E.g. the
            index in `points_arr`.
        :return: The 5d cuboid in the output_axis_order order. With the
            batch dim as the first axis.
        """
        return self.get_points_data([point_key])[0, ...]

    def get_points_data(self, points_key: list[int]) -> torch.Tensor:
        """
        Takes a list of keys used to identify point (typically the indices in
        the points list) and returns the cuboids centered around the points.

        This handles getting the cuboids for `dataset[i]`, when `i` is a list
        of ints.

        :param points_key: A list of the unique key used to identify the
            points. E.g. the indices in `points_arr`.
        :return: The 5d cuboids in the output_axis_order order. With the
            batch dim as the first axis.
        """
        data = torch.empty(
            (len(points_key), *self.cuboid_with_channels_size),
            dtype=torch.float32,
        )
        self.src_image_data.get_point_batch_cuboid_data(data, points_key)

        return data


class CuboidThreadedDatasetBase(CuboidDatasetBase):
    """
    `CuboidDatasetBase` gets the data directly from its `src_image_data`
    (`ImageDataBase`) instance. If this is run with multiple workers, each
    in its own sub-process, each worker gets its own copy of `src_image_data`.

    This class adds the ability for the dataset to get the data from the
    `src_image_data` via a sub-thread in the main process. Each worker uses
    a queue to request data from this thread, which reads it from disk quickly
    and sends it back to the worker to process. This ensures that only one
    thread reads the data so there's no hard drive contention among the
    workers.

    Additionally, each worker allocates the batch buffer, which is shared
    in memory with the main thread. The main thread can then directly write
    to it without having to send data back and forth.
    """

    _dataset_thread: ThreadWithException | None = None

    _worker_queues: list[Queue] = None

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._worker_queues = []

    def __getstate__(self):
        state = self.__dict__.copy()
        if self._dataset_thread is not None:
            del state["src_image_data"]
        return state

    def get_points_data(self, points_key: list[int]) -> torch.Tensor:
        thread = self._dataset_thread
        if thread is None:
            return super().get_points_data(points_key)

        data = torch.empty(
            (len(points_key), *self.cuboid_with_channels_size),
            dtype=torch.float32,
        )

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

        return data

    def start_dataset_thread(self, num_workers: int):
        # include queue for host thread
        ctx = mp.get_context("spawn")
        queues = [ctx.Queue(maxsize=0) for _ in range(num_workers + 1)]
        self._worker_queues = queues

        self._dataset_thread = ThreadWithException(
            target=_read_data_send_cuboids,
            args=(self.src_image_data, queues),
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
    """
    Implements `CuboidThreadedDatasetBase` using a `CachedArrayStackImageData`
    to read the cuboids from array type data (e.g. Dask arrays).
    """

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

        mask = np.array(
            [self.point_has_full_cuboid(p) for p in self.points_arr],
            dtype=np.bool_,
        )
        self.points_arr = self.points_arr[mask]

        data_arrays = [signal_array, background_array]
        self.src_image_data = CachedArrayStackImageData(
            points_arr=self.points_arr,
            input_arrays=data_arrays,
            data_axis_order=self.axis_order,
            max_axis_0_cuboids_buffered=max_axis_0_cuboids_buffered,
            cuboid_size=self.data_cuboid_voxels,
        )
        self._set_output_data_dim_reordering(self.src_image_data)

    def point_has_full_cuboid(self, point: np.ndarray) -> bool:
        for ax, axis_size, cuboid_size in zip(
            self.axis_order, self.stack_shape, self.data_cuboid_voxels
        ):
            start, end = get_data_cuboid_range(point[ax], cuboid_size, ax)
            if start < 0:
                return False
            if end > axis_size:
                # if it's axis_size it's fine because end is not inclusive
                return False

        return True


class CuboidTiffDataset(CuboidThreadedDatasetBase):
    """
    Implements `CuboidThreadedDatasetBase` using a `CachedTiffCuboidImageData`
    to read the cuboids from individual tiff files.
    """

    def __init__(
        self,
        points_filenames: Sequence[Sequence[str]],
        max_cuboids_buffered: int = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if not len(points_filenames):
            raise ValueError("No data provided")
        if len(points_filenames) != len(self.points_arr):
            raise ValueError(
                "Points and filenames must have same number of elements"
            )

        self.num_channels = len(points_filenames[0])
        filenames_arr = np.array(points_filenames).astype(np.str_)
        self.src_image_data = CachedTiffCuboidImageData(
            points_arr=self.points_arr,
            filenames_arr=filenames_arr,
            data_axis_order=self.axis_order,
            max_cuboids_buffered=max_cuboids_buffered,
            cuboid_size=self.data_cuboid_voxels,
        )
        self._set_output_data_dim_reordering(self.src_image_data)


class CuboidBatchSampler(Sampler):
    """
    Custom Sampler for our `CuboidDatasetBase`. It can randomize the order in
    which it samples the points, while respecting that each batch contains
    samples from the same plane (so each batch can be efficiently read). Or
    it can sort the sampler by a specific axis in the data to load the data
    efficiently.
    """

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
            plane_indices = [
                np.arange(len(dataset.points_arr), dtype=np.int64)
            ]
        else:
            points_raw = defaultdict(list)
            for i, point in enumerate(dataset.points_arr):
                points_raw[point[sort_by_axis]].append(i)
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
