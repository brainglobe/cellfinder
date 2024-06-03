from pathlib import Path
from random import shuffle
from typing import Dict, List, Optional, Tuple, Union

import keras
import numpy as np
from brainglobe_utils.cells.cells import Cell, group_cells_by_z
from brainglobe_utils.general.numerical import is_even
from keras.utils import Sequence
from scipy.ndimage import zoom
from skimage.io import imread

from cellfinder.core import types
from cellfinder.core.classify.augment import AugmentationParameters, augment

# TODO: rename, as now using dask arrays -
#  actually should combine to one generator


class StackSizeError(Exception):
    pass


class CubeGeneratorFromFile(Sequence):
    """
    Reads cubes (defined as e.g. xml, csv) from raw data to pass to
    keras.Model.fit_generator() or keras.Model.predict_generator()

    If augment=True, each augmentation selected has an "augment_likelihood"
    chance of being applied to each cube
    """

    # TODO: shuffle within (and maybe between) batches
    # TODO: limit workers based on RAM

    def __init__(
        self,
        points: List[Cell],
        signal_array: types.array,
        background_array: types.array,
        voxel_sizes: Tuple[int, int, int],
        network_voxel_sizes: Tuple[int, int, int],
        batch_size: int = 64,
        cube_width: int = 50,
        cube_height: int = 50,
        cube_depth: int = 20,
        channels: int = 2,  # No other option currently
        classes: int = 2,
        extract: bool = False,
        train: bool = False,
        augment: bool = False,
        augment_likelihood: float = 0.1,
        flip_axis: Tuple[int, int, int] = (0, 1, 2),
        rotate_max_axes: Tuple[float, float, float] = (1, 1, 1),  # degrees
        # scale=[0.5, 2],  # min, max
        translate: Tuple[float, float, float] = (0.05, 0.05, 0.05),
        shuffle: bool = False,
        interpolation_order: int = 2,
        *args,
        **kwargs,
    ):
        # pass any additional arguments not specified in signature to the
        # constructor of the superclass (e.g.: `use_multiprocessing` or
        # `workers`)
        super().__init__(*args, **kwargs)

        self.points = points
        self.signal_array = signal_array
        self.background_array = background_array
        self.batch_size = batch_size
        self.axis_2_pixel_um = float(voxel_sizes[2])
        self.axis_1_pixel_um = float(voxel_sizes[1])
        self.axis_0_pixel_um = float(voxel_sizes[0])
        self.network_axis_2_pixel_um = float(network_voxel_sizes[2])
        self.network_axis_1_pixel_um = float(network_voxel_sizes[1])
        self.network_axis_0_pixel_um = float(network_voxel_sizes[0])
        self.cube_width = cube_width
        self.cube_height = cube_height
        self.cube_depth = cube_depth
        self.channels = channels
        self.classes = classes

        # saving training data to file
        self.extract = extract

        self.train = train
        self.augment = augment
        self.augment_likelihood = augment_likelihood
        self.flip_axis = flip_axis
        self.rotate_max_axes = rotate_max_axes
        # self.scale = scale
        self.translate = translate
        self.shuffle = shuffle
        self.interpolation_order = interpolation_order

        self.scale_cubes = False

        self.rescaling_factor_axis_2: float = 1
        self.rescaling_factor_axis_1: float = 1
        self.rescaled_cube_width: float = self.cube_width
        self.rescaled_cube_height: float = self.cube_height

        self.__check_image_sizes()
        self.__get_image_size()
        self.__check_z_scaling()
        self.__check_in_plane_scaling()
        self.__remove_outlier_points()
        self.__get_batches()
        if shuffle:
            self.on_epoch_end()

    def __check_image_sizes(self) -> None:
        if len(self.signal_array) != len(self.background_array):
            raise ValueError(
                f"Number of signal images ({len(self.signal_array)}) does not "
                f"match the number of background images "
                f"({len(self.background_array)}"
            )

    def __get_image_size(self) -> None:
        self.image_z_size = len(self.signal_array)
        self.image_height, self.image_width = self.signal_array[0].shape

    def __check_in_plane_scaling(self) -> None:
        if self.axis_2_pixel_um != self.network_axis_2_pixel_um:
            self.rescaling_factor_axis_2 = (
                self.network_axis_2_pixel_um / self.axis_2_pixel_um
            )
            self.rescaled_cube_width = (
                self.cube_width * self.rescaling_factor_axis_2
            )
            self.scale_cubes = True
        if self.axis_1_pixel_um != self.network_axis_1_pixel_um:
            self.rescaling_factor_axis_1 = (
                self.network_axis_1_pixel_um / self.axis_1_pixel_um
            )
            self.rescaled_cube_height = (
                self.cube_height * self.rescaling_factor_axis_1
            )
            self.scale_cubes = True

    def __check_z_scaling(self) -> None:
        if self.axis_0_pixel_um != self.network_axis_0_pixel_um:
            plane_scaling_factor = (
                self.network_axis_0_pixel_um / self.axis_0_pixel_um
            )
            self.num_planes_needed_for_cube = round(
                self.cube_depth * plane_scaling_factor
            )
        else:
            self.num_planes_needed_for_cube = self.cube_depth

        if self.num_planes_needed_for_cube > self.image_z_size:
            raise StackSizeError(
                f"The number of planes provided ({self.image_z_size}) "
                "is not sufficient for any cubes to be extracted "
                f"(need at least {self.num_planes_needed_for_cube}). "
                "Please check the input data"
            )

    def __remove_outlier_points(self) -> None:
        """
        Remove points that won't get extracted (i.e too close to the edge)
        """
        self.points = [
            point for point in self.points if self.extractable(point)
        ]

    def extractable(self, point: Cell) -> bool:
        x0, x1, y0, y1, z0, z1 = self.__get_boundaries()
        return (
            x0 <= point.x <= x1 and y0 <= point.y <= y1 and z0 <= point.z <= z1
        )

    def __get_boundaries(self) -> Tuple[int, int, int, int, int, int]:
        x0 = int(round((self.cube_width / 2) * self.rescaling_factor_axis_2))
        x1 = int(round(self.image_width - x0))

        y0 = int(round((self.cube_height / 2) * self.rescaling_factor_axis_1))
        y1 = int(round(self.image_height - y0))

        z0 = int(round(self.num_planes_needed_for_cube / 2))
        z1 = self.image_z_size - z0
        return x0, x1, y0, y1, z0, z1

    def __get_batches(self) -> None:
        self.points_groups = group_cells_by_z(self.points)
        # TODO: add optional shuffling of each group here
        self.batches = []
        for centre_plane in self.points_groups.keys():
            points_per_plane = self.points_groups[centre_plane]
            for i in range(0, len(points_per_plane), self.batch_size):
                self.batches.append(points_per_plane[i : i + self.batch_size])

        self.ordered_points = []
        for batch in self.batches:
            for cell in batch:
                self.ordered_points.append(cell)

    def __len__(self) -> int:
        """
        Number of batches
        :return: Number of batches per epoch
        """
        return len(self.batches)

    def __getitem__(self, index: int) -> Union[
        np.ndarray,
        Tuple[np.ndarray, List[Dict[str, float]]],
        Tuple[np.ndarray, Dict],
    ]:
        """
        Generates a single batch of cubes
        :param index:
        :return:
        """
        if not self.batches:
            raise IndexError("Empty batch. Were any cells detected?")

        cell_batch = self.batches[index]
        signal_stack, background_stack = self.__get_stacks(index)
        images = self.__generate_cubes(
            cell_batch, signal_stack, background_stack
        )

        if self.train:
            batch_labels = [cell.type - 1 for cell in cell_batch]
            batch_labels = keras.utils.to_categorical(
                batch_labels, num_classes=self.classes
            )
            return images, batch_labels.astype(np.float32)
        elif self.extract:
            batch_info = self.__get_batch_dict(cell_batch)
            return images, batch_info
        else:
            return images

    def __get_stacks(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        centre_plane = self.batches[index][0].z

        min_plane, max_plane = get_cube_depth_min_max(
            centre_plane, self.num_planes_needed_for_cube
        )

        signal_stack = np.array(self.signal_array[min_plane:max_plane])
        background_stack = np.array(self.background_array[min_plane:max_plane])

        return signal_stack, background_stack

    def __generate_cubes(
        self,
        cell_batch: List[Cell],
        signal_stack: np.ndarray,
        background_stack: np.ndarray,
    ) -> np.ndarray:
        number_images = len(cell_batch)
        images = np.empty(
            (
                (number_images,)
                + (self.cube_height, self.cube_width, self.cube_depth)
                + (self.channels,)
            ),
            dtype=np.float32,
        )

        for idx, cell in enumerate(cell_batch):
            images = self.__populate_array_with_cubes(
                images, idx, cell, signal_stack, background_stack
            )

        return images

    def __populate_array_with_cubes(
        self,
        images: np.ndarray,
        idx: int,
        cell: Cell,
        signal_stack: np.ndarray,
        background_stack: np.ndarray,
    ) -> np.ndarray:
        if self.augment:
            self.augmentation_parameters = AugmentationParameters(
                self.flip_axis,
                self.translate,
                self.rotate_max_axes,
                self.interpolation_order,
                self.augment_likelihood,
            )
        images[idx, :, :, :, 0] = self.__get_oriented_image(cell, signal_stack)
        images[idx, :, :, :, 1] = self.__get_oriented_image(
            cell, background_stack
        )
        return images

    def __get_oriented_image(
        self, cell: Cell, image_stack: np.ndarray
    ) -> np.ndarray:
        x0 = int(round(cell.x - (self.rescaled_cube_width / 2)))
        x1 = int(x0 + self.rescaled_cube_width)
        y0 = int(round(cell.y - (self.rescaled_cube_height / 2)))
        y1 = int(y0 + self.rescaled_cube_height)
        image = image_stack[:, y0:y1, x0:x1]
        image = np.moveaxis(image, 0, 2)

        if self.augment:
            # scale to isotropic, but don't scale back
            image = augment(
                self.augmentation_parameters, image, scale_back=False
            )

        pixel_scalings = [
            self.cube_height / image.shape[0],
            self.cube_width / image.shape[1],
            self.cube_depth / image.shape[2],  # type: ignore[misc]
            # Not sure why mypy thinks .shape[2] is out of bounds above?
        ]

        # TODO: ensure this is always the correct size
        image = zoom(image, pixel_scalings, order=self.interpolation_order)
        return image

    @staticmethod
    def __get_batch_dict(cell_batch: List[Cell]) -> List[Dict[str, float]]:
        return [cell.to_dict() for cell in cell_batch]

    def on_epoch_end(self) -> None:
        """
        Shuffle data for each epoch
        :return: Shuffled indexes
        """
        shuffle(self.batches)


class CubeGeneratorFromDisk(Sequence):
    """
    Reads in cubes from a list of paths for keras.Model.fit_generator() or
    keras.Model.predict_generator()

    If augment=True, each augmentation selected has a 50/50 chance of being
    applied to each cube
    """

    def __init__(
        self,
        signal_list: List[Union[str, Path]],
        background_list: List[Union[str, Path]],
        labels: Optional[List[int]] = None,  # only if training or validating
        batch_size: int = 64,
        shape: Tuple[int, int, int] = (50, 50, 20),
        channels: int = 2,
        classes: int = 2,
        shuffle: bool = False,
        augment: bool = False,
        augment_likelihood: float = 0.1,
        flip_axis: Tuple[int, int, int] = (0, 1, 2),
        rotate_max_axes: Tuple[int, int, int] = (45, 45, 45),  # degrees
        # scale=[0.5, 2],  # min, max
        translate: Tuple[float, float, float] = (0.2, 0.2, 0.2),
        train: bool = False,  # also return labels
        interpolation_order: int = 2,
        *args,
        **kwargs,
    ):
        # pass any additional arguments not specified in signature to the
        # constructor of the superclass (e.g.: `use_multiprocessing` or
        # `workers`)
        super().__init__(*args, **kwargs)

        self.im_shape = shape
        self.batch_size = batch_size
        self.labels = labels
        self.signal_list = signal_list
        self.background_list = background_list
        self.channels = channels
        self.classes = classes
        self.augment = augment
        self.augment_likelihood = augment_likelihood
        self.flip_axis = flip_axis
        self.rotate_max_axes = rotate_max_axes
        # self.scale = scale
        self.translate = translate
        self.train = train
        self.interpolation_order = interpolation_order
        self.indexes = np.arange(len(self.signal_list))
        if shuffle:
            self.on_epoch_end()

    # TODO: implement scale and shear

    def on_epoch_end(self) -> None:
        """
        Shuffle data for each epoch
        :return: Shuffled indexes
        """
        self.indexes = np.arange(len(self.signal_list))
        np.random.shuffle(self.indexes)

    def __len__(self) -> int:
        """
        Number of batches
        :return: Number of batches per epoch
        """
        return int(np.ceil(len(self.signal_list) / self.batch_size))

    def __getitem__(self, index: int) -> Union[
        np.ndarray,
        Tuple[np.ndarray, List[Dict[str, float]]],
        Tuple[np.ndarray, Dict],
    ]:
        """
        Generates a single batch of cubes
        :param index:
        :return:
        """
        # Generate indexes of the batch
        start_index = index * self.batch_size
        end_index = start_index + self.batch_size
        indexes = self.indexes[start_index:end_index]

        # Get data corresponding to batch
        list_signal_tmp = [self.signal_list[k] for k in indexes]
        list_background_tmp = [self.background_list[k] for k in indexes]

        images = self.__generate_cubes(list_signal_tmp, list_background_tmp)

        if self.train and self.labels is not None:
            batch_labels = [self.labels[k] for k in indexes]
            batch_labels = keras.utils.to_categorical(
                batch_labels, num_classes=self.classes
            )
            return images, batch_labels.astype(np.float32)
        else:
            return images

    def __generate_cubes(
        self,
        list_signal_tmp: List[Union[str, Path]],
        list_background_tmp: List[Union[str, Path]],
    ) -> np.ndarray:
        number_images = len(list_signal_tmp)
        images = np.empty(
            ((number_images,) + self.im_shape + (self.channels,)),
            dtype=np.float32,
        )

        for idx, signal_im in enumerate(list_signal_tmp):
            background_im = list_background_tmp[idx]
            images = self.__populate_array_with_cubes(
                images, idx, signal_im, background_im
            )

        return images

    def __populate_array_with_cubes(
        self,
        images: np.ndarray,
        idx: int,
        signal_im: Union[str, Path],
        background_im: Union[str, Path],
    ) -> np.ndarray:
        if self.augment:
            self.augmentation_parameters = AugmentationParameters(
                self.flip_axis,
                self.translate,
                self.rotate_max_axes,
                self.interpolation_order,
                self.augment_likelihood,
            )
        images[idx, :, :, :, 0] = self.__get_oriented_image(signal_im)
        images[idx, :, :, :, 1] = self.__get_oriented_image(background_im)

        return images

    def __get_oriented_image(self, image_path: Union[str, Path]) -> np.ndarray:
        # if paths are pathlib objs, skimage only reads one plane
        image = np.moveaxis(imread(image_path), 0, 2)
        if self.augment:
            image = augment(self.augmentation_parameters, image)
        return image


def get_cube_depth_min_max(
    centre_plane: int, num_planes_needed_for_cube: int
) -> Tuple[int, int]:
    half_cube_depth = num_planes_needed_for_cube // 2
    min_plane = centre_plane - half_cube_depth

    if is_even(num_planes_needed_for_cube):
        # WARNING: not centered because even
        max_plane = centre_plane + half_cube_depth
    else:
        # centered
        max_plane = centre_plane + half_cube_depth + 1

    return min_plane, max_plane
