from typing import Literal, Sequence

import torch
import torch.nn.functional as F
from monai.transforms import RandAffine

from cellfinder.core.tools.tools import get_axis_reordering, random_bool

DIM = Literal["x", "y", "z", "c"]
AXIS = Literal["x", "y", "z"]
RandRange = Sequence[float] | Sequence[tuple[float, float]] | None


class DataAugmentation:
    """
    Randomly augments the input data when called.

    Typical example::

        augmenter = DataAugmentation(...)
        augmented_data = augmenter(data)

    Parameters
    ----------
    volume_size : dict
        Dict whose keys are x, y, and z and whose values are the size of the
        input data at the given dimension.
    """

    DIM_ORDER = "c", "y", "x", "z"
    """
    The dimension order we internally expect the data to be. The user passes
    in data in `data_axis_order` order that we convert to this order before
    handling.
    """

    AXIS_ORDER = "y", "x", "z"
    """
    Similar to `DIM_ORDER`, except it's the order of the 3 axes of the cuboid.
    """

    # precomputed, so both channels are treated identically
    def __init__(
        self,
        volume_size: dict[str, int],
        data_axis_order: tuple[DIM, DIM, DIM, DIM],
        augment_likelihood: float,
        flippable_axis: Sequence[int] = (),
        translate_range: RandRange = None,
        scale_range: RandRange = None,
        rotate_range: RandRange = None,
    ):
        volume_values = list(volume_size.values())
        self.needs_isotropy = max(volume_values) != min(volume_values)
        self.isotropic_volume_size = (max(volume_values),) * 3

        self._volume_size = [volume_size[ax] for ax in self.AXIS_ORDER]

        self._data_reordering = []
        self._data_reordering_back = []
        self._compute_data_reordering(data_axis_order)

        # do prob = 1 because we decide when to apply it
        self.affine = RandAffine(
            prob=1,
            rotate_range=rotate_range,
            shear_range=None,
            translate_range=self._fix_translate_range(translate_range),
            scale_range=self._fix_scale_range(scale_range),
            cache_grid=True,
            spatial_size=self.isotropic_volume_size,
            lazy=False,
        )

        self.flippable_axis = flippable_axis
        self.augment_likelihood = augment_likelihood

        self.axes_to_flip: list[int] = []
        self.do_affine = False

    def _compute_data_reordering(
        self, data_axis_order: tuple[DIM, DIM, DIM, DIM]
    ) -> None:
        self._data_reordering = []
        self._data_reordering_back = []

        if data_axis_order != self.DIM_ORDER:
            self._data_reordering = get_axis_reordering(
                data_axis_order,
                self.DIM_ORDER,
            )
            self._data_reordering_back = get_axis_reordering(
                self.DIM_ORDER,
                data_axis_order,
            )

    def _fix_translate_range(self, translate_range: RandRange) -> RandRange:
        if translate_range is None:
            return None

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

        return translate_range

    def _fix_scale_range(self, scale_range: RandRange) -> RandRange:
        if scale_range is None:
            return None

        scale_range = list(scale_range)
        for i, val in enumerate(scale_range):
            # we get scale values where 1 means original size. monai
            # expects values around 0, where 0 means original size
            if isinstance(val, Sequence):
                scale_range[i] = 1 / val[0] - 1, 1 / val[1] - 1
            else:
                scale_range[i] = 1 / val - 1

        return scale_range

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
        data = F.interpolate(data, size=self._volume_size, mode="trilinear")
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
        if self._data_reordering:
            data = torch.permute(data, self._data_reordering)

        data = self.rescale_to_isotropic(data)

        data = self.apply_affine(data)
        data = self.flip_axis(data)

        data = self.rescale_to_original(data)

        if self._data_reordering_back:
            data = torch.permute(data, self._data_reordering_back)

        return data
