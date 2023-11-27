from typing import List, Tuple

import numpy as np
from scipy.ndimage import rotate, zoom

from cellfinder.core.tools.tools import (
    all_elements_equal,
    random_bool,
    random_probability,
    random_sign,
)

all_axes = np.array((0, 1, 2))


def augment(
    augmentation_parameters: "AugmentationParameters",
    image: np.ndarray,
    scale_back: bool = True,
) -> np.ndarray:
    pixel_sizes = image.shape
    min_pixel_size = min(pixel_sizes)
    relative_pixel_sizes = []
    for pixel_size in pixel_sizes:
        relative_pixel_sizes.append(pixel_size / min_pixel_size)

    image, normalised_pixel_sizes = rescale_to_isotropic(
        image,
        relative_pixel_sizes,
        augmentation_parameters.interpolation_order,
    )
    # TODO: is this a sensible order?
    if augmentation_parameters.flip_axis is not None:
        image = flip_image(image, augmentation_parameters.axes_to_flip)

    if augmentation_parameters.translate is not None:
        image = translate_image(
            image,
            augmentation_parameters.translate_axes,
            augmentation_parameters.random_translate_multipliers,
        )

    # if augmentation_parameters.scale is not None:
    #     image = scale_image(image, augmentation_parameters.scale)

    if augmentation_parameters.rotate_max_axes is not None:
        image = rotate_image(image, augmentation_parameters.rotation_angles)

    if scale_back:
        image = rescale_to_original_size(
            image,
            relative_pixel_sizes,
            normalised_pixel_sizes,
            augmentation_parameters.interpolation_order,
        )
    return image


def rescale_to_isotropic(
    image: np.ndarray,
    relative_pixel_sizes: List[float],
    interpolation_order: int,
) -> Tuple[np.ndarray, List[float]]:
    if not all_elements_equal(relative_pixel_sizes):
        min_pixel_size = min(relative_pixel_sizes)
        normalised_pixel_sizes = []
        for pixel_size in relative_pixel_sizes:
            normalised_pixel_sizes.append(
                round(pixel_size / min_pixel_size, 2)
            )

        image = zoom(image, normalised_pixel_sizes, order=interpolation_order)
    else:
        normalised_pixel_sizes = relative_pixel_sizes
    return image, normalised_pixel_sizes


def rescale_to_original_size(
    image: np.ndarray,
    relative_pixel_sizes: List[float],
    normalised_pixel_sizes: List[float],
    interpolation_order: int,
) -> np.ndarray:
    if not all_elements_equal(relative_pixel_sizes):
        inverse_pixel_sizes = []
        for pixel_size in normalised_pixel_sizes:
            inverse_pixel_sizes.append(round(1 / pixel_size, 2))

        image = zoom(image, inverse_pixel_sizes, order=interpolation_order)
    return image


def flip_image(image: np.ndarray, axes_to_flip: List[int]) -> np.ndarray:
    for axis in axes_to_flip:
        image = np.flip(image, axis)
    return image


def translate_image(
    image: np.ndarray,
    translate_axes: List[int],
    random_translate_multipliers: List[float],
) -> np.ndarray:
    pixel_shifts = []
    for idx, axis in enumerate(translate_axes):
        pixel_shifts.append(
            int(round(random_translate_multipliers[idx] * image.shape[axis]))
        )

    image = np.roll(image, pixel_shifts, axis=translate_axes)
    return image


def rotate_image(
    image: np.ndarray, rotation_angles: List[float]
) -> np.ndarray:
    for axis, angle in enumerate(rotation_angles):
        if angle != 0:
            rotate_axes = all_axes[all_axes != axis]
            image = rotate(
                image, angle, axes=rotate_axes, reshape=False, mode="constant"
            )
    return image


# def scale_image(image, scale, ndigits=2):
#     scale_factor = round(
#         uniform(scale[0], scale[1]), ndigits=ndigits
#     )
#     return image

# def shear_image(image):
#     return image


class AugmentationParameters:
    # precomputed, so both channels are treated identically
    def __init__(
        self,
        flip_axis: Tuple[int, int, int],
        translate: Tuple[float, float, float],
        rotate_max_axes: Tuple[float, float, float],
        interpolation_order: int,
        augment_likelihood: float,
    ):
        # this is a clumsy way of passing parameters to the augment function
        self.flip_axis = flip_axis
        self.translate = translate
        self.rotate_max_axes = rotate_max_axes
        self.interpolation_order = interpolation_order

        self.augment_likelihood = augment_likelihood

        self.axes_to_flip: List[int] = []
        self.translate_axes: List[int] = []
        self.random_translate_multipliers: List[float] = []
        self.rotation_angles: List[float] = []

        if flip_axis:
            self.get_flip_parameters(flip_axis)
        if translate:
            self.get_translation_parameters(translate)
        if rotate_max_axes:
            self.get_rotation_parameters(rotate_max_axes)

    def get_flip_parameters(self, flip_axis: Tuple[int, int, int]) -> None:
        self.axes_to_flip = []
        for axis in all_axes:
            if axis in flip_axis:
                if random_bool(likelihood=self.augment_likelihood):
                    self.axes_to_flip.append(axis)

    def get_translation_parameters(
        self, translate: Tuple[float, float, float]
    ) -> None:
        self.translate_axes = []
        self.random_translate_multipliers = []
        for axis, translate_mag in enumerate(translate):
            if translate_mag > 0:
                if random_bool(likelihood=self.augment_likelihood):
                    self.translate_axes.append(axis)
                    self.random_translate_multipliers.append(
                        random_sign() * random_probability() * translate_mag
                    )

    def get_rotation_parameters(
        self, rotate_max_axes: Tuple[float, float, float]
    ) -> None:
        self.rotation_angles = []
        for max_rotation in rotate_max_axes:
            if random_bool(likelihood=self.augment_likelihood):
                angle = int(
                    round(
                        -max_rotation + 2 * random_probability() * max_rotation
                    )
                )
            else:
                angle = 0
            self.rotation_angles.append(angle)
