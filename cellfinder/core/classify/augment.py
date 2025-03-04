import torch
import torch.nn.functional as F
from monai.transforms import RandAffine

from cellfinder.core.tools.tools import random_bool


class DataAugmentation:
    """
    Data must be 4 dim, with order of Channels, Y, X, Z,
    where spatial is the 3 dims.
    """

    AXIS_ORDER = "c", "y", "x", "z"

    # precomputed, so both channels are treated identically
    def __init__(
        self,
        volume_size: tuple[int, int, int],
        flippable_axis: list[int],
        translate: tuple[float, float, float],
        scale: tuple[
            tuple[float, float], tuple[float, float], tuple[float, float]
        ],
        rotate_max_axes: tuple[float, float, float],
        augment_likelihood: float,
    ):
        self.needs_isotropy = max(volume_size) != min(volume_size)
        self.volume_size = volume_size
        self.isotropic_volume_size = (max(self.volume_size),) * 3

        translate_range = [int(p * s) for p, s in zip(translate, volume_size)]
        # RandAffine adds one after sampling given random interval
        scale = [(s0 - 1, s1 - 1) for s0, s1 in scale]
        # do prob = 1 because we decide when to apply it
        self.affine = RandAffine(
            prob=1,
            rotate_range=rotate_max_axes,
            shear_range=None,
            translate_range=translate_range,
            scale_range=scale,
            cache_grid=True,
            lazy=False,
        )

        self.flippable_axis = flippable_axis
        self.augment_likelihood = augment_likelihood

        self.axes_to_flip: list[int] = []
        self.do_affine = False

    def update_parameters(self) -> bool:
        self.do_affine = random_bool(likelihood=self.augment_likelihood)
        self.update_flip_parameters()

        return bool(self.do_affine or self.axes_to_flip)

    def update_flip_parameters(self) -> None:
        flippable_axis = self.flippable_axis
        if not flippable_axis:
            return

        axes_to_flip = self.axes_to_flip = []
        for axis in flippable_axis:
            if random_bool(likelihood=self.augment_likelihood):
                # add 1 because of initial channel dim
                axes_to_flip.append(axis + 1)

    def rescale_to_isotropic(self, data: torch.Tensor) -> torch.Tensor:
        if not self.needs_isotropy:
            return data

        shape = data.shape[:1] + self.isotropic_volume_size
        data = F.interpolate(data, size=shape, mode="trilinear")
        return data

    def rescale_to_original(self, data: torch.Tensor) -> torch.Tensor:
        if not self.needs_isotropy:
            return data

        shape = data.shape[:1] + self.volume_size
        data = F.interpolate(data, size=shape, mode="trilinear")
        return data

    def apply_affine(self, data: torch.Tensor) -> torch.Tensor:
        if not self.do_affine:
            return data

        return self.affine(data, padding_mode="border")

    def flip_axis(self, data: torch.Tensor) -> torch.Tensor:
        if not self.axes_to_flip:
            return data

        return torch.flip(data, self.axes_to_flip)

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        data = self.rescale_to_isotropic(data)

        data = self.flip_axis(data)
        data = self.apply_affine(data)

        data = self.rescale_to_original(data)

        return data
