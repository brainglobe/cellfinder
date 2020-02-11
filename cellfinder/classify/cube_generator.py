import logging
import numpy as np
import tensorflow as tf

from random import shuffle
from tifffile import tifffile
from skimage.io import imread
from scipy.ndimage import zoom
from tensorflow.python.keras.utils.data_utils import Sequence
from imlib.IO.cells import get_cells
from imlib.cells.cells import group_cells_by_z
from imlib.general.numerical import is_even

from cellfinder.extract.extract_cubes import StackSizeError
from cellfinder.classify.augment import AugmentationParameters, augment


class CubeGeneratorFromFile(Sequence):
    """
    Reads cubes (defined as e.g. xml, csv) from raw data to pass to
    keras.Model.fit_generator() or keras.Model.predict_generator()

    If augment=True, each augmentation selected has an "augment_likelihood"
    chance of beingapplied to each cube
    """

    # TODO: shuffle within (and maybe between) batches
    # TODO: limit workers based on RAM

    def __init__(
        self,
        cells_file,
        signal_planes,
        background_planes,
        batch_size=16,
        x_pixel_um=1,
        y_pixel_um=1,
        z_pixel_um=5,
        x_pixel_um_network=1,
        y_pixel_um_network=1,
        z_pixel_um_network=5,
        cube_width=50,
        cube_height=50,
        cube_depth=20,
        channels=2,  # No other option currently
        classes=2,
        train=False,
        augment=False,
        augment_likelihood=0.1,
        flip_axis=[0, 1, 2],
        rotate_max_axes=[1, 1, 1],  # degrees
        # scale=[0.5, 2],  # min, max
        translate=[0.05, 0.05, 0.05],
        shuffle=False,
        interpolation_order=2,
    ):
        self.cells_file = cells_file
        self.signal_planes = signal_planes
        self.background_planes = background_planes
        self.batch_size = batch_size
        self.x_pixel_um = x_pixel_um
        self.y_pixel_um = y_pixel_um
        self.z_pixel_um = z_pixel_um
        self.x_pixel_um_network = x_pixel_um_network
        self.y_pixel_um_network = y_pixel_um_network
        self.z_pixel_um_network = z_pixel_um_network
        self.cube_width = cube_width
        self.cube_height = cube_height
        self.cube_depth = cube_depth
        self.channels = channels
        self.classes = classes

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

        self.rescaling_factor_x = 1
        self.rescaling_factor_y = 1
        self.rescaled_cube_width = self.cube_width
        self.rescaled_cube_height = self.cube_height

        self.__check_image_paths()
        self.__get_image_size()
        self.__load_cells()
        self.__check_z_scaling()
        self.__check_in_plane_scaling()
        self.__remove_outlier_cells()
        self.__get_batches()
        if shuffle:
            self.on_epoch_end()

    def __check_image_paths(self):
        if len(self.signal_planes) != len(self.background_planes):
            raise ValueError(
                f"Number of signal images ({len(self.signal_planes)} does not"
                f"match the number of background images "
                f"({len(self.background_planes)}"
            )

    def __get_image_size(self):
        self.image_z_size = len(self.signal_planes)
        first_plane = tifffile.imread(self.signal_planes[0])
        self.image_height, self.image_width = first_plane.shape

    def __load_cells(self):
        self.cells = get_cells(self.cells_file)
        if not self.cells:
            logging.error(
                f"No cells found, exiting. "
                f"Please check the file: {self.cells_file}"
            )
            raise ValueError(
                f"No cells found, exiting. "
                f"Please check the file: {self.cells_file}"
            )

    def __check_in_plane_scaling(self):
        if self.x_pixel_um != self.x_pixel_um_network:
            self.rescaling_factor_x = self.x_pixel_um_network / self.x_pixel_um
            self.rescaled_cube_width = (
                self.cube_width * self.rescaling_factor_x
            )
            self.scale_cubes = True
        if self.y_pixel_um != self.y_pixel_um_network:
            self.rescaling_factor_y = self.y_pixel_um_network / self.y_pixel_um
            self.rescaled_cube_height = (
                self.cube_height * self.rescaling_factor_y
            )
            self.scale_cubes = True

    def __check_z_scaling(self):
        if self.z_pixel_um != self.z_pixel_um_network:
            plane_scaling_factor = self.z_pixel_um_network / self.z_pixel_um
            self.num_planes_needed_for_cube = round(
                self.cube_depth * plane_scaling_factor
            )
        else:
            self.num_planes_needed_for_cube = self.cube_depth

        if self.num_planes_needed_for_cube > self.image_z_size:
            raise StackSizeError(
                "The number of planes provided is not sufficient "
                "for any cubes to be extracted. Please check the "
                "input data"
            )

    def __remove_outlier_cells(self):
        """
        Remove cells that won't get extracted (i.e too close to the edge)
        """
        self.cells = [cell for cell in self.cells if self.extractable(cell)]

    def extractable(self, cell):
        x0, x1, y0, y1, z0, z1 = self.__get_boundaries()
        if (
            cell.z < z0
            or cell.z > z1
            or cell.x < x0
            or cell.x > x1
            or cell.y < y0
            or cell.y > y1
        ):
            return False
        else:
            return True

    def __get_boundaries(self):
        x0 = int(round((self.cube_width / 2) * self.rescaling_factor_x))
        x1 = int(round(self.image_width - x0))

        y0 = int(round((self.cube_height / 2) * self.rescaling_factor_y))
        y1 = int(round(self.image_height - y0))

        z0 = int(round(self.num_planes_needed_for_cube / 2))
        z1 = self.image_z_size - z0
        return x0, x1, y0, y1, z0, z1

    def __get_batches(self):
        self.cells_groups = group_cells_by_z(self.cells)
        # TODO: add optional shuffling of each group here
        self.batches = []
        for centre_plane in self.cells_groups.keys():
            cells_per_plane = self.cells_groups[centre_plane]
            for i in range(0, len(cells_per_plane), self.batch_size):
                self.batches.append(cells_per_plane[i : i + self.batch_size])

        self.ordered_cells = []
        for batch in self.batches:
            for cell in batch:
                self.ordered_cells.append(cell)

    def __len__(self):
        """
        Number of batches
        :return: Number of batches per epoch
        """
        return len(self.batches)

    def __getitem__(self, index):
        """
        Generates a single batch of cubes
        :param index:
        :return:
        """

        cell_batch = self.batches[index]
        signal_stack, background_stack = self.__get_stacks(index)
        images = self.__generate_cubes(
            cell_batch, signal_stack, background_stack
        )

        if self.train:
            batch_labels = [cell.type - 1 for cell in cell_batch]
            batch_labels = tf.keras.utils.to_categorical(
                batch_labels, num_classes=self.classes
            )
            return images, batch_labels
        else:
            return images

    def __get_stacks(self, index):
        centre_z = self.batches[index][0].z

        half_cube_depth = self.num_planes_needed_for_cube // 2
        min_plane = centre_z - half_cube_depth

        if is_even(self.num_planes_needed_for_cube):
            # WARNING: not centered because even
            max_plane = centre_z + half_cube_depth
        else:
            # centered
            max_plane = centre_z + half_cube_depth + 1

        signal_stack = np.empty(
            (
                self.num_planes_needed_for_cube,
                self.image_height,
                self.image_width,
            )
        )
        background_stack = np.empty_like(signal_stack)
        for plane, plane_path in enumerate(
            self.signal_planes[min_plane:max_plane]
        ):
            signal_stack[plane] = tifffile.imread(plane_path)

        for plane, plane_path in enumerate(
            self.background_planes[min_plane:max_plane]
        ):
            background_stack[plane] = tifffile.imread(plane_path)

        return signal_stack, background_stack

    def __generate_cubes(self, cell_batch, signal_stack, background_stack):
        number_images = len(cell_batch)
        images = np.empty(
            (
                (number_images,)
                + (self.cube_height, self.cube_width, self.cube_depth)
                + (self.channels,)
            )
        )

        for idx, cell in enumerate(cell_batch):
            images = self.__populate_array_with_cubes(
                images, idx, cell, signal_stack, background_stack
            )

        return images

    def __populate_array_with_cubes(
        self, images, idx, cell, signal_stack, background_stack
    ):
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

    def __get_oriented_image(self, cell, image_stack):
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
            self.cube_depth / image.shape[2],
        ]

        # TODO: ensure this is always the correct size
        image = zoom(image, pixel_scalings, order=self.interpolation_order)
        return image

    def on_epoch_end(self):
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
        signal_list,
        background_list,
        labels=None,  # only if training or validating
        batch_size=16,
        shape=(50, 50, 20),
        channels=2,
        classes=2,
        shuffle=False,
        augment=False,
        augment_likelihood=0.1,
        flip_axis=[0, 1, 2],
        rotate_max_axes=[45, 45, 45],  # degrees
        # scale=[0.5, 2],  # min, max
        translate=[0.2, 0.2, 0.2],
        train=False,  # also return labels
        interpolation_order=2,
    ):
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

    def on_epoch_end(self):
        """
        Shuffle data for each epoch
        :return: Shuffled indexes
        """
        self.indexes = np.arange(len(self.signal_list))
        np.random.shuffle(self.indexes)

    def __len__(self):
        """
        Number of batches
        :return: Number of batches per epoch
        """
        return int(np.ceil(len(self.signal_list) / self.batch_size))

    def __getitem__(self, index):
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

        if self.train:
            batch_labels = [self.labels[k] for k in indexes]
            batch_labels = tf.keras.utils.to_categorical(
                batch_labels, num_classes=self.classes
            )
            return images, batch_labels
        else:
            return images

    def __generate_cubes(self, list_signal_tmp, list_background_tmp):
        number_images = len(list_signal_tmp)
        images = np.empty(
            ((number_images,) + self.im_shape + (self.channels,))
        )

        for idx, signal_im in enumerate(list_signal_tmp):
            background_im = list_background_tmp[idx]
            images = self.__populate_array_with_cubes(
                images, idx, signal_im, background_im
            )

        return images.astype(np.float16)

    def __populate_array_with_cubes(
        self, images, idx, signal_im, background_im
    ):
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

    def __get_oriented_image(self, image_path):
        # if paths are pathlib objs, skimage only reads one plane
        image = np.moveaxis(imread(image_path), 0, 2)
        if self.augment:
            image = augment(self.augmentation_parameters, image)
        return image
