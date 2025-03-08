from typing import Sequence

import torch
import torch.nn.functional as F
from monai.transforms import RandAffine

from cellfinder.core.tools.tools import random_bool

RandRange = Sequence[float] | Sequence[tuple[float, float]] | None


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
        augment_likelihood: float,
        flippable_axis: Sequence[int] = (),
        translate_range: RandRange = None,
        scale_range: RandRange = None,
        rotate_range: RandRange = None,
    ):
        self.needs_isotropy = max(volume_size) != min(volume_size)
        self.volume_size = volume_size
        self.isotropic_volume_size = (max(self.volume_size),) * 3

        if translate_range is not None:
            translate_range = list(translate_range)
            for i, (val, size) in enumerate(
                zip(translate_range, self.isotropic_volume_size)
            ):
                # we expect the values as fraction of the size of the volume in
                # the given dim. monai expects it as pixel offsets so we need
                # to multiply by dim size. Also, it does negative translation
                if isinstance(val, Sequence):
                    translate_range[i] = -val[0] * size, -val[1] * size
                else:
                    translate_range[i] = -val * size

        if scale_range is not None:
            scale_range = list(scale_range)
            for i, val in enumerate(scale_range):
                # we get scale values where 1 means original size. monai
                # expects values around 0, where 0 means original size
                if isinstance(val, Sequence):
                    scale_range[i] = 1 / val[0] - 1, 1 / val[1] - 1
                else:
                    scale_range[i] = 1 / val - 1

        # do prob = 1 because we decide when to apply it
        self.affine = RandAffine(
            prob=1,
            rotate_range=rotate_range,
            shear_range=None,
            translate_range=translate_range,
            scale_range=scale_range,
            cache_grid=True,
            spatial_size=self.isotropic_volume_size,
            lazy=False,
        )

        self.flippable_axis = flippable_axis
        self.augment_likelihood = augment_likelihood

        self.axes_to_flip: list[int] = []
        self.do_affine = False

    def update_parameters(self) -> bool:
        self.do_affine = random_bool(likelihood=1 - self.augment_likelihood)
        self.update_flip_parameters()

        return bool(self.do_affine or self.axes_to_flip)

    def update_flip_parameters(self) -> None:
        flippable_axis = self.flippable_axis
        if not flippable_axis:
            return

        axes_to_flip = self.axes_to_flip = []
        for axis in flippable_axis:
            if random_bool(likelihood=1 - self.augment_likelihood):
                # add 1 because of initial channel dim
                axes_to_flip.append(axis + 1)

    def rescale_to_isotropic(self, data: torch.Tensor) -> torch.Tensor:
        if not self.needs_isotropy:
            return data

        # needs batch dim
        data = data.unsqueeze(0)
        data = F.interpolate(
            data, size=self.isotropic_volume_size, mode="trilinear"
        )
        data = data.squeeze(0)
        return data

    def rescale_to_original(self, data: torch.Tensor) -> torch.Tensor:
        if not self.needs_isotropy:
            return data

        # needs batch dim
        data = data.unsqueeze(0)
        data = F.interpolate(data, size=self.volume_size, mode="trilinear")
        data = data.squeeze(0)
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

        data = self.apply_affine(data)
        data = self.flip_axis(data)

        data = self.rescale_to_original(data)

        return data
