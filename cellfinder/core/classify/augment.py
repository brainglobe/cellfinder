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

        augmentor = DataAugmentation(...)
        if augmentor.update_parameters():
            augmented_data = augmentor(data)

    The input data should be 4-dimensional (channel and 3 spatial dimensions),
    with the order specified by ``data_dim_order``. Calling
    ``update_parameters`` will evaluate random variables and return
    True if either the affine or flipping augmentation is to be applied. Then
    calling the augmentor with the data will return the augmented data. If it
    returned False, calling the augmentor will return the original data. The
    augmentation is applied identically to each channel in the cuboid.

    Using the random variables, each call to ``update_parameters`` refreshes
    the axes to be flipped (if any) in the subsequent augmentor call. It also
    refreshes whether the affine transformation will occur. But each subsequent
    call to the augmentor will, if the affine transformation was evaluated to
    occur, randomly pick a different value in the given translation, scale, and
    rotation intervals.

    Using the ``augment_likelihood`` it is evaluated whether the affine
    transformation will be applied. If so, all of them (translation, scaling,
    rotation) that were not None will be applied together.

    :param volume_size: Dict whose keys are x, y, and z and whose values are
        the size of the input data for the given dimension.
    :param data_dim_order: A 4-tuple with order of the 4 dimensions of the data
        passed to the augmentor. It is a tuple of ``("x", "y", "z", "c")`` in
        any order.
    :param augment_likelihood: A value in [0, 1] with the likelihood of the
        data to be augmented by the transformation provided. E.g. of the
        available transformations (flipping, affine [translation, scaling,
        rotation]), it's the likelihood of applying that transformation. E.g.
        if a scaling and rotation value is provided, ``flippable_axis`` is
        ``(0, )``, and ``augment_likelihood`` is ``0.4``, then the input data
        will have a 40% chance of being scaled and rotated and independently
        have a 40% chance of the first axis to be flipped.
    :param flippable_axis: A sequence of axes to potentially flip around its
        center during augmentation. The axes values are indexes in
        ``data_dim_order`` **without** the channel dimension. E.g. if
        ``data_dim_order`` is ``("c", "z", "y", "x")`` and ``flippable_axis``
        is ``(0, 2)`` and ``augment_likelihood`` is ``0.3``, then axes 0 (z)
        and 2 (x) have each independently a 30% chance of being flipped around
        their centers.
    :param translate_range: The per-axis translation factor. If None there's no
        translation. Otherwise, it's a sequence of size 3 corresponding to the
        data axis in ``data_dim_order`` (channel axis is excluded). Each
        element is either a float or a 2-tuple of floats indicating the
        translation for that axis. Floats are in the ``[-1, 1]`` interval,
        where ``-1`` means negative translation by a full ``volume_size`` of
        that axis and zero means no translation. A single float means there's
        a ``augment_likelihood`` of the given axis to be translated by that
        value. A 2-tuple of floats means there's a ``augment_likelihood`` of
        the given axis to be translated to a random value in that interval.
    :param scale_range: The per-axis scaling factor. If None there's no
        scaling. Otherwise, it's a sequence of size 3 corresponding to the data
        axis in ``data_dim_order`` (channel axis is excluded). Each element is
        either a float or a 2-tuple of floats indicating the scaling for
        that axis. Floats are in the ``(0, inf)`` interval, where ``1`` means
        unscaled. A single float means there's a ``augment_likelihood`` of the
        given axis to be scaled by that value. A 2-tuple of floats means
        there's a ``augment_likelihood`` of the given axis to be scaled to a
        random value in that interval.
    :param rotate_range: The per-axis rotation factor in radians. If None
        there's no rotation. Otherwise, it's a sequence of size 3 corresponding
        to the data axis in ``data_dim_order`` (channel axis is excluded). Each
        element is either a float or a 2-tuple of floats indicating the
        rotation **around** that axis. Floats are in the ``[0, 2pi)`` interval,
        where the value indicates a clockwise or counter-clockwise rotation
        (depending on if it's x/y or z axis) in radians around that axis. And
        where zero means no rotation. A single float means there's a
        ``augment_likelihood`` of a rotation around the given axis by that
        value. A 2-tuple of floats means there's an ``augment_likelihood`` of a
        rotation around the given axis by a random value in that interval.
    """

    DIM_ORDER = "c", "y", "x", "z"
    """
    The dimension order we internally expect the data to be. The user passes
    in data in `data_dim_order` order that we convert to this order before
    handling.
    """

    AXIS_ORDER = "y", "x", "z"
    """
    Similar to `DIM_ORDER`, except it's the order of the 3 axes of the cuboid.
    """

    _data_reordering: Sequence[int]
    """
    The indexing used to re-order the input data from its original ordering to
    ``DIM_ORDER``.
    """

    _data_reordering_back: Sequence[int]
    """
    The indexing used to re-order the internally re-ordered data from
    ``DIM_ORDER`` back to its original ordering.
    .
    """

    _input_axis_order: tuple[str, str, str]
    """
    The names of the input data axes, ordered according to the input data axes,
    excluding the channel dimension.
    """

    _volume_size: tuple[int, int, int]
    """
    The input data volume sizes ordered by ``AXIS_ORDER`` (excluding the
    channel dimension).
    """

    _isotropic_volume_size: Sequence[int]
    """
    The isotropic size of the input data. This will be the maximum size of the
    dimensions of the input data (excluding the channel dimension)..
    """

    _asked_affine: bool
    """If the any of the affine transformations were requested."""

    _is_isotropic: bool
    """
    Whether the input data is isotropic (excluding the channel dimension).
    """

    _do_affine: bool
    """
    After randomization, if we should do affine transformation on the next
    augmentation call based on the random value.
    """

    _affine: RandAffine
    """The RandAffine used to do the affine transformations."""

    augment_likelihood: float
    """The likelihood to use for each augmentation randomization. See above."""

    _flippable_axis: Sequence[int]
    """
    The axes, in ``AXIS_ORDER`` indexing, that the user listed as those to be
    randomly flipped.
    """

    _axes_to_flip: Sequence[int]
    """
    After randomization, which of the axes to actually flip in the next
    augmentation call.
    """

    def __init__(
        self,
        volume_size: dict[str, int],
        data_dim_order: tuple[DIM, DIM, DIM, DIM],
        augment_likelihood: float,
        flippable_axis: Sequence[int] = (),
        translate_range: RandRange = None,
        scale_range: RandRange = None,
        rotate_range: RandRange = None,
    ):
        volume_values = list(volume_size.values())
        self._asked_affine = bool(
            translate_range or scale_range or rotate_range
        )
        self._is_isotropic = max(volume_values) == min(volume_values)
        self._isotropic_volume_size = (max(volume_values),) * 3

        self._volume_size = tuple(volume_size[ax] for ax in self.AXIS_ORDER)

        self._input_axis_order = tuple(d for d in data_dim_order if d != "c")
        self._data_reordering = []
        self._data_reordering_back = []
        self._compute_data_reordering(data_dim_order)

        # do prob = 1 because we decide when to apply the affine later
        self._affine = RandAffine(
            prob=1,
            rotate_range=self._fix_rotate_range(rotate_range),
            shear_range=None,
            translate_range=self._fix_translate_range(translate_range),
            scale_range=self._fix_scale_range(scale_range),
            cache_grid=True,
            spatial_size=self._isotropic_volume_size,
            lazy=False,
        )

        self._flippable_axis = self._fix_flippable_axis(flippable_axis)
        self.augment_likelihood = augment_likelihood

        self._axes_to_flip = []
        self._do_affine = False

    def _compute_data_reordering(
        self, data_dim_order: tuple[DIM, DIM, DIM, DIM]
    ) -> None:
        """
        If the ``data_dim_order`` and ``DIM_ORDER`` are not the same, this
        creates the indexing used to re-order the input data to the
        ``DIM_ORDER`` back.
        """
        self._data_reordering = []
        self._data_reordering_back = []

        if data_dim_order != self.DIM_ORDER:
            self._data_reordering = get_axis_reordering(
                data_dim_order,
                self.DIM_ORDER,
            )
            self._data_reordering_back = get_axis_reordering(
                self.DIM_ORDER,
                data_dim_order,
            )

    def _fix_flippable_axis(
        self, flippable_axis: Sequence[int]
    ) -> Sequence[int]:
        """
        Users provide the axis in ``data_dim_order``, but during augmentation
        we expect the data to have been re-ordered to ``DIM_ORDER``. So we
        transform and return the input ``flippable_axis`` from
        ``data_dim_order`` to ``DIM_ORDER`` so it can be directly used.
        """
        if not flippable_axis:
            return flippable_axis
        if self._input_axis_order == self.AXIS_ORDER:
            return flippable_axis

        fixed_axis = []
        for ax_i in flippable_axis:
            ax = self._input_axis_order[ax_i]
            fixed_axis.append(self.AXIS_ORDER.index(ax))
        return fixed_axis

    def _fix_rotate_range(self, rotate_range: RandRange) -> RandRange:
        """
        Users provide the axis in ``data_dim_order``, but during augmentation
        we expect the data to have been re-ordered to ``DIM_ORDER``. So we
        re-order and return the input ``rotate_range`` from
        ``data_dim_order`` to ``DIM_ORDER`` so it can be directly used.

        ``test_monai_rotate_input`` verifies the monai input as doing
        rotations. However, clockwise or anti-clockwise rotations occur in
        MONAI depending on whether we rotate around x/y or z axis, which won't
        matter for augmentation.
        """
        if rotate_range is None:
            return None
        if len(rotate_range) != 3:
            raise ValueError("Must specify rotate value for each dimension")

        rotate_range = list(rotate_range)
        for i, val in enumerate(rotate_range):
            if not isinstance(val, Sequence):
                rotate_range[i] = val, val

        if self._input_axis_order == self.AXIS_ORDER:
            return rotate_range

        fixed_range = [None, None, None]
        for i in range(3):
            new_i = self.AXIS_ORDER.index(self._input_axis_order[i])
            fixed_range[new_i] = rotate_range[i]
        return fixed_range

    def _fix_translate_range(self, translate_range: RandRange) -> RandRange:
        """
        Users provide the axis in ``data_dim_order``, but during augmentation
        we expect the data to have been re-ordered to ``DIM_ORDER``. So we
        re-order and return the input ``translate_range`` from
        ``data_dim_order`` to ``DIM_ORDER`` so it can be directly used.

        We also transform the range from fraction to (negative) pixels as
        expected by monai. This is verified by ``test_monai_translate_input``.
        """
        if translate_range is None:
            return None
        if len(translate_range) != 3:
            raise ValueError("Must specify translate value for each dimension")

        translate_range = list(translate_range)
        for i, (val, size) in enumerate(
            zip(translate_range, self._isotropic_volume_size)
        ):
            # we expect the values as fraction of the size of the volume in
            # the given dim. monai expects it as pixel offsets so we need
            # to multiply by dim size. Also, monai does negative translation
            if isinstance(val, Sequence):
                # interval order seems irrelevant in monai
                translate_range[i] = -val[0] * size, -val[1] * size
            else:
                translate_range[i] = -val * size, -val * size

        if self._input_axis_order == self.AXIS_ORDER:
            return translate_range

        fixed_range = [None, None, None]
        for i in range(3):
            new_i = self.AXIS_ORDER.index(self._input_axis_order[i])
            fixed_range[new_i] = translate_range[i]
        return fixed_range

    def _fix_scale_range(self, scale_range: RandRange) -> RandRange:
        """
        Users provide the axis in ``data_dim_order``, but during augmentation
        we expect the data to have been re-ordered to ``DIM_ORDER``. So we
        re-order and return the input ``scale_range`` from
        ``data_dim_order`` to ``DIM_ORDER`` so it can be directly used.

        We also transform the scale from the range (0, inf), where 1 means
        original size, to (inf, -1), where 0 means original size, as expected
        by monai. Yes the interval is nonsensical and yes, this is what monai
        expects. We verify this in ``test_monai_scale_input``.
        """
        if scale_range is None:
            return None
        if len(scale_range) != 3:
            raise ValueError("Must specify scale value for each dimension")

        scale_range = list(scale_range)
        for i, val in enumerate(scale_range):
            # we get scale values where 1 means original size. monai
            # expects values around 0, where 0 means original size
            if isinstance(val, Sequence):
                # interval order seems irrelevant so (inf, -1) would become
                # (-1, inf) in monai
                scale_range[i] = 1 / val[0] - 1, 1 / val[1] - 1
            else:
                scale_range[i] = 1 / val - 1, 1 / val - 1

        if self._input_axis_order == self.AXIS_ORDER:
            return scale_range

        fixed_range = [None, None, None]
        for i in range(3):
            new_i = self.AXIS_ORDER.index(self._input_axis_order[i])
            fixed_range[new_i] = scale_range[i]
        return fixed_range

    def update_parameters(self) -> bool:
        """
        Evaluates random variables to decide if the data should be augmented
        using affine or flipping transformations according to
        ``augment_likelihood``. If so, it returns True and augmentor should
        be called with data, which will be augmented. Otherwise, it's False
        and if the augmentor is called it will return the original data.
        """
        self._do_affine = False
        if self._asked_affine:
            self._do_affine = random_bool(likelihood=self.augment_likelihood)
        self._update_flip_parameters()

        return bool(self._do_affine or self._axes_to_flip)

    def _update_flip_parameters(self) -> None:
        """
        Evaluates random variables to decide which of the requested axes to
        flip in the next augmentation call, if any.
        """
        flippable_axis = self._flippable_axis
        if not flippable_axis:
            return

        axes_to_flip = self._axes_to_flip = []
        for axis in flippable_axis:
            if random_bool(likelihood=self.augment_likelihood):
                # add 1 because of initial channel dim
                axes_to_flip.append(axis + 1)

    def rescale_to_isotropic(self, data: torch.Tensor) -> torch.Tensor:
        """
        Rescales and returns the data as isotropic (all spatial dimensions are
        the same and are given the size of the largest spatial dimension), if
        the input data is not already isotropic. Otherwise, it just returns the
        input data.

        The ``data`` shape must already be in ``DIM_ORDER`` and its size must
        match ``volume_size``.
        """
        if self._is_isotropic:
            return data

        # needs batch dim
        data = data.unsqueeze(0)
        data = F.interpolate(
            data, size=self._isotropic_volume_size, mode="trilinear"
        )
        data = data.squeeze(0)
        return data

    def rescale_to_original(self, data: torch.Tensor) -> torch.Tensor:
        """
        Rescales ``data`` from isotropic as returned by
        ``rescale_to_isotropic`` to the size of the original input data, if the
        original input data was not isotropic. Otherwise, it just returns
        ``data``.

        The returned ``data`` shape will stay in ``DIM_ORDER`` and its size
        will match ``volume_size``.
        """
        if self._is_isotropic:
            return data

        # needs batch dim
        data = data.unsqueeze(0)
        data = F.interpolate(data, size=self._volume_size, mode="trilinear")
        data = data.squeeze(0)
        return data

    def apply_affine(self, data: torch.Tensor) -> torch.Tensor:
        """
        Applies and returns the affine transformed data if it was requested and
        if the randomization via ``update_parameters`` requires it. Otherwise,
        it returns the data unchanged.

        Data must be isotropic and in ``DIM_ORDER``.
        """
        if not self._do_affine:
            return data

        return self._affine(data, padding_mode="border")

    def flip_axis(self, data: torch.Tensor) -> torch.Tensor:
        """
        Applies and returns axes flipped data if it was requested and
        if the randomization via ``update_parameters`` requires it. Otherwise,
        it returns the data unchanged.

        Data must be in ``DIM_ORDER``.
        """
        if not self._axes_to_flip:
            return data

        return torch.flip(data, self._axes_to_flip)

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        if len(data.shape) != 4:
            raise ValueError(
                f"Expected 4 dimensional data (channel + 3 spatial dimensions)"
                f" but got {len(data.shape)}"
            )
        # make sure it's in DIM_ORDER
        if self._data_reordering:
            data = torch.permute(data, self._data_reordering)

        # first dim is channel now
        if data.shape[1:] != self._volume_size:
            raise ValueError(
                "Input data size does not match expected volume_size"
            )

        # if we do affine, it must be isotropic
        if self._do_affine:
            data = self.rescale_to_isotropic(data)

        # apply affine and axes flipping based on last update_parameters
        data = self.apply_affine(data)
        data = self.flip_axis(data)

        # undo isotropism
        if self._do_affine:
            data = self.rescale_to_original(data)

        # make sure it's in data_dim_order
        if self._data_reordering_back:
            data = torch.permute(data, self._data_reordering_back)

        return data
