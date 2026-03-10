import math

import pytest
import torch
from monai.transforms import RandAffine

from cellfinder.core.classify.augment import DataAugmentation


@pytest.fixture
def cube_with_side_dot() -> torch.Tensor:
    # DataAugmentation.DIM_ORDER of (c, y, x, z)
    data = torch.zeros((2, 11, 11, 7))
    # put dot in channel 0 at pixel in plane z = 5, on the x/y diagonal - both x and y are 8
    data[0, 8, 8, 5] = 1
    # put dot in channel 1 at pixel in plane z = 5, but x is 2 and y is 8
    data[1, 8, 2, 5] = 1
    return data


@pytest.fixture
def cube_with_center_dot() -> torch.Tensor:
    # DataAugmentation.DIM_ORDER of (c, y, x, z)
    data = torch.zeros((2, 11, 11, 7))
    data[:, 5, 5, 3] = 1
    return data


@pytest.mark.parametrize("offset", [-1, 1])
def test_monai_translate_input(offset):
    """
    This tests that the translate values expected by monai are negative of the
    real values. See ``DataAugmentation._fix_translate_range`` for details.
    """
    data = torch.zeros((1, 5, 5, 5))
    data[0, 2, 2, 2] = 1

    affine = RandAffine(
        prob=1,
        translate_range=((-offset, -offset), 0, 0),
        cache_grid=True,
        spatial_size=(5, 5, 5),
        lazy=False,
    )
    transformed = affine(data, padding_mode="border")
    assert math.isclose(transformed[0, 2, 2, 2], 0)
    assert transformed[0, 2 + offset, 2, 2] > 0


@pytest.mark.parametrize(
    "interval,growth",
    [
        ((100, 100), "smaller"),
        ((-1, -1), "bigger"),
        ((100, 0), "not bigger"),
        ((0, -1), "not smaller"),
    ],
)
def test_monai_scale_input(interval, growth):
    """
    This tests that the unexpected scale interval is indeed expected by monai by checking values near the foreground border of a 5x5x5 test data block that contains a 3x3x3 foreground cube in its middle.
    See ``DataAugmentation._fix_scale_range`` for details.
    """
    data = torch.zeros((1, 5, 5, 5))
    data[0, 1:4, 1:4, 1:4] = 1

    affine = RandAffine(
        prob=1,
        scale_range=(interval,) * 3,
        cache_grid=True,
        spatial_size=(5, 5, 5),
        lazy=False,
    )
    transformed = affine(data, padding_mode="border")

    match growth:
        case "smaller":
            # it got scaled down, check edge became zero
            assert math.isclose(transformed[0, 3, 3, 3], 0)
        case "bigger":
            # it got scaled up, check outside became non-zero
            assert transformed[0, 4, 4, 4] > 0
        case "not bigger":
            # it got either smaller or stayed the same - outside stayed zero
            assert math.isclose(transformed[0, 4, 4, 4], 0)
        case "not smaller":
            # it got either bigger or stayed the same - edge stayed non-zero
            assert transformed[0, 3, 3, 3] > 0


@pytest.mark.parametrize("angle", [math.pi / 2, 3 * math.pi / 2])
def test_monai_rotate_input(angle):
    """
    This tests the angle values expected by monai is normal, but clockwise
    around x-axis.
    """
    data = torch.zeros((1, 5, 5, 5))
    data[0, 2, 2, 1] = 1

    # rotate around the first axis
    affine = RandAffine(
        prob=1,
        rotate_range=((angle, angle), 0, 0),
        cache_grid=True,
        spatial_size=(5, 5, 5),
        lazy=False,
    )
    transformed = affine(data, padding_mode="border")

    assert math.isclose(transformed[0, 2, 2, 1], 0)
    if angle < math.pi:
        assert transformed[0, 2, 1, 2] > 0
    else:
        assert transformed[0, 2, 3, 2] > 0


def test_needs_isotropy():
    """Check that converting cuboid to isotropy makes it isotropic."""
    data = torch.zeros((2, 11, 11, 7))
    augmenter = DataAugmentation(
        volume_size={"x": 11, "y": 11, "z": 7},
        augment_likelihood=0.5,
        data_dim_order=DataAugmentation.DIM_ORDER,
    )

    assert not augmenter._is_isotropic
    iso = augmenter.rescale_to_isotropic(data)
    # should now be isotropic with max axis size
    assert set(iso.shape[1:]) == {11}


def test_not_needs_isotropy():
    """Check already isotropic data is returned unchanged."""
    data = torch.zeros((2, 11, 11, 11))
    augmenter = DataAugmentation(
        volume_size={"x": 11, "y": 11, "z": 11},
        augment_likelihood=0.5,
        data_dim_order=DataAugmentation.DIM_ORDER,
    )

    assert augmenter._is_isotropic
    iso = augmenter.rescale_to_isotropic(data)
    assert iso is data

    orig_data = augmenter.rescale_to_original(iso)
    assert orig_data is data


def test_isotropy_round_trip(cube_with_side_dot):
    """
    Check that converting cuboid to isotropy and back results in original data.
    """
    c, y, x, z = cube_with_side_dot.shape
    augmenter = DataAugmentation(
        volume_size={"x": x, "y": y, "z": z},
        augment_likelihood=0.5,
        data_dim_order=DataAugmentation.DIM_ORDER,
    )
    assert not augmenter._is_isotropic

    # forward to iso
    iso = augmenter.rescale_to_isotropic(cube_with_side_dot)
    assert iso.shape != cube_with_side_dot.shape
    assert set(iso.shape[1:]) == {max([x, y, z])}

    # back to original
    round_trip = augmenter.rescale_to_original(iso)
    assert round_trip.shape == cube_with_side_dot.shape
    # The input has two impulse values of 1. So we can expect some distortion,
    # but the sum should be low and only around the impulse
    assert torch.sum(cube_with_side_dot - round_trip) < 0.3
    assert torch.sum(torch.not_equal(cube_with_side_dot, round_trip)) < 10


@pytest.mark.parametrize(
    "name,value",
    [
        (
            "translate_range",
            [(1 / 11, 1 / 11), (2 / 11, 2 / 11), (1 / 7, 1 / 7)],
        ),
        ("rotate_range", [(0, 0), (0, 0), (math.pi / 2, math.pi / 2)]),
        ("scale_range", ((3, 3),) * 3),
        ("flippable_axis", (0, 1)),
    ],
)
def test_no_augment(cube_with_side_dot, name, value):
    """Test that setting augment likelihood of zero results in no augment."""
    c, y, x, z = cube_with_side_dot.shape
    augmenter = DataAugmentation(
        volume_size={"x": x, "y": y, "z": z},
        augment_likelihood=0,
        data_dim_order=DataAugmentation.DIM_ORDER,
        **{name: value},
    )
    assert (
        not augmenter.update_parameters()
    ), "Parameters should not be randomized"
    assert not augmenter._do_affine
    assert not augmenter._axes_to_flip
    augmented = augmenter(cube_with_side_dot)

    assert augmented.shape == cube_with_side_dot.shape
    assert torch.equal(cube_with_side_dot, augmented)


@pytest.mark.parametrize("reorder", [False, True])
def test_augment_translate(cube_with_side_dot, reorder):
    """
    Test that translate augmentation works, with likelihood of 1.

    With re-order we check that input data of different axes order works.
    """
    c, y, x, z = cube_with_side_dot.shape
    translate_range = [(1 / 11, 1 / 11), (2 / 11, 2 / 11), (1 / 7, 1 / 7)]
    data_dim_order = "c", "y", "x", "z"
    if reorder:
        data_dim_order = "c", "y", "z", "x"
        z, x = x, z

    augmenter = DataAugmentation(
        volume_size={"x": x, "y": y, "z": z},
        augment_likelihood=1,
        translate_range=translate_range,
        data_dim_order=data_dim_order,
    )
    assert augmenter.update_parameters(), "Parameters should be randomized"
    # affine is needed for translate/rotate/scale
    assert augmenter._asked_affine
    assert augmenter._do_affine
    # the cube is not cube but cuboid, so need to make iso before transforming
    assert not augmenter._is_isotropic
    assert not augmenter._axes_to_flip
    augmented = augmenter(cube_with_side_dot)

    assert augmented.shape == cube_with_side_dot.shape
    # check it was translated correctly converting percent to voxels offset
    assert augmented[0, 8, 8, 5] < 0.1
    assert augmented[1, 8, 2, 5] < 0.1
    assert augmented[0, 8 + 1, 8 + 2, 5 + 1] > 0.1
    assert augmented[1, 8 + 1, 2 + 2, 5 + 1] > 0.1


@pytest.mark.parametrize("reorder", [False, True])
def test_augment_rotate(cube_with_side_dot, reorder):
    """
    Test that rotation augmentation works, with likelihood of 1.

    With re-order we check that input data of different axes order works.
    """
    c, y, x, z = cube_with_side_dot.shape
    data_dim_order = "c", "y", "x", "z"
    if reorder:
        data_dim_order = "c", "y", "z", "x"
        z, x = x, z

    augmenter = DataAugmentation(
        volume_size={"x": x, "y": y, "z": z},
        augment_likelihood=1,
        rotate_range=[(0, 0), (0, 0), (math.pi / 2, math.pi / 2)],
        data_dim_order=data_dim_order,
    )
    assert augmenter.update_parameters(), "Parameters should be randomized"
    # afine is needed for translate/rotate/scale
    assert augmenter._asked_affine
    assert augmenter._do_affine
    # the cube is not cube but cuboid, so need to make iso before transforming
    assert not augmenter._is_isotropic
    assert not augmenter._axes_to_flip
    augmented = augmenter(cube_with_side_dot)

    assert augmented.shape == cube_with_side_dot.shape
    # these 2 points that had ones, should now be zeros
    assert augmented[0, 8, 8, 5] < 0.1
    assert augmented[1, 8, 2, 5] < 0.1
    # we rotated around last axis by either +45 or -45 degree, not both. The
    # axis of the points depend on re-ordering which determines the rotation
    # direction. First two axis points at 8, if doing positive rotation it just
    # reflects around one of the first two axis and if doing negative it
    # reflects around the other. Positive or negative rotation depends on
    # whether we rotate around x,y or z. We do XOR to check either +/- rotation
    # occurred.
    assert bool(augmented[0, 8, 10 - 8, 5] > 0.1) != bool(
        augmented[0, 10 - 8, 8, 5] > 0.1
    )
    # similarly for the other point
    assert bool(augmented[1, 2, 10 - 8, 5] > 0.1) != bool(
        augmented[1, 10 - 2, 8, 5] > 0.1
    )


@pytest.mark.parametrize("reorder", [False, True])
def test_augment_scale(cube_with_center_dot, reorder):
    """
    Test that scaling augmentation works, with likelihood of 1.

    With re-order we check that input data of different axes order works.
    """
    c, y, x, z = cube_with_center_dot.shape
    data_dim_order = "c", "y", "x", "z"
    if reorder:
        data_dim_order = "c", "y", "z", "x"
        z, x = x, z

    augmenter = DataAugmentation(
        volume_size={"x": x, "y": y, "z": z},
        augment_likelihood=1,
        scale_range=((3, 3),) * 3,
        data_dim_order=data_dim_order,
    )
    assert augmenter.update_parameters(), "Parameters should be randomized"
    # affine is needed for translate/rotate/scale
    assert augmenter._asked_affine
    assert augmenter._do_affine
    # the cube is not cube but cuboid, so need to make iso before transforming
    assert not augmenter._is_isotropic
    assert not augmenter._axes_to_flip
    augmented = augmenter(cube_with_center_dot)

    assert augmented.shape == cube_with_center_dot.shape
    assert augmented[0, 5, 5, 3] > 0.1
    assert augmented[0, 5 + 1, 5, 3] > 0.1
    assert augmented[0, 5, 5 + 1, 3] > 0.1
    assert augmented[0, 5, 5, 3 + 1] > 0.1

    assert augmented[1, 5, 5, 3] > 0.1
    assert augmented[1, 5 + 1, 5, 3] > 0.1
    assert augmented[1, 5, 5 + 1, 3] > 0.1
    assert augmented[1, 5, 5, 3 + 1] > 0.1


def test_affine_2_interval_same_as_single_value(cube_with_side_dot):
    """
    Checks that for all the affine transformations, using a 2-tuple interval
    with the same value is the same as providing a single value.
    """
    c, y, x, z = cube_with_side_dot.shape

    augmenter_1 = DataAugmentation(
        volume_size={"x": x, "y": y, "z": z},
        augment_likelihood=1,
        translate_range=[1 / 11, 2 / 11, 1 / 7],
        rotate_range=[0, 0, math.pi / 2],
        scale_range=(3, 3, 3),
        data_dim_order=DataAugmentation.DIM_ORDER,
    )
    augmenter_2 = DataAugmentation(
        volume_size={"x": x, "y": y, "z": z},
        augment_likelihood=1,
        translate_range=[(1 / 11, 1 / 11), (2 / 11, 2 / 11), (1 / 7, 1 / 7)],
        rotate_range=[(0, 0), (0, 0), (math.pi / 2, math.pi / 2)],
        scale_range=((3, 3),) * 3,
        data_dim_order=DataAugmentation.DIM_ORDER,
    )
    assert augmenter_1.update_parameters()
    assert augmenter_2.update_parameters()

    augmented_1 = augmenter_1(cube_with_side_dot)
    augmented_2 = augmenter_2(cube_with_side_dot)

    assert not torch.allclose(cube_with_side_dot, augmented_1)
    assert torch.allclose(augmented_1, augmented_2)


@pytest.mark.parametrize("reorder", [False, True])
def test_augment_axis_flip(cube_with_side_dot, reorder):
    c, y, x, z = cube_with_side_dot.shape
    data_dim_order = "c", "y", "x", "z"
    if reorder:
        data_dim_order = "c", "y", "z", "x"
        z, x = x, z

    augmenter = DataAugmentation(
        volume_size={"x": x, "y": y, "z": z},
        augment_likelihood=1,
        flippable_axis=(0, 2),
        data_dim_order=data_dim_order,
    )
    assert augmenter.update_parameters(), "Parameters should be randomized"
    # afine is not needed when only flipping axes
    assert not augmenter._asked_affine
    assert not augmenter._do_affine
    assert not augmenter._is_isotropic
    assert augmenter._axes_to_flip
    augmented = augmenter(cube_with_side_dot)

    assert augmented.shape == cube_with_side_dot.shape
    # first and last axes are flip location of the 1
    assert augmented[0, 8, 8, 5] < 0.1
    assert augmented[1, 8, 2, 5] < 0.1
    assert augmented[0, 2, 8, 1] > 0.1
    assert augmented[1, 2, 2, 1] > 0.1


@pytest.mark.parametrize("reorder", [False, True])
def test_channels_are_identically_augmented(reorder):
    """
    Checks that all the channels are augmented identically and with identical
    parameters. And we also test all the augmentations possible.
    """
    data = torch.zeros((2, 11, 11, 7))
    data[:, 8, 8, 5] = 1

    y, x, z = 11, 11, 7
    data_dim_order = "c", "y", "x", "z"
    if reorder:
        data_dim_order = "c", "y", "z", "x"
        y, z, x = 11, 11, 7

    augmenter = DataAugmentation(
        volume_size={"x": x, "y": y, "z": z},
        augment_likelihood=1,
        translate_range=[(1 / 11, 1 / 11), (2 / 11, 2 / 11), (1 / 7, 1 / 7)],
        rotate_range=[(0, 0), (0, 0), (math.pi / 2, math.pi / 2)],
        scale_range=((3, 3),) * 3,
        flippable_axis=(0, 2),
        data_dim_order=data_dim_order,
    )
    assert augmenter.update_parameters(), "Parameters should be randomized"
    assert augmenter._asked_affine
    assert augmenter._do_affine
    assert not augmenter._is_isotropic
    assert augmenter._axes_to_flip

    augmented = augmenter(data)

    assert augmented.shape == data.shape
    assert torch.equal(augmented[0, ...], augmented[1, ...])


def test_bad_augment_parameters():
    """Checks that passing bad parameters to augmentor raises errors."""
    with pytest.raises(ValueError):
        DataAugmentation(
            volume_size={"x": 11, "y": 11, "z": 7},
            augment_likelihood=1,
            translate_range=[
                (0, 0),
                (0, 0),
            ],
            data_dim_order=DataAugmentation.DIM_ORDER,
        )
    with pytest.raises(ValueError):
        DataAugmentation(
            volume_size={"x": 11, "y": 11, "z": 7},
            augment_likelihood=1,
            rotate_range=[
                (0, 0),
                (0, 0),
            ],
            data_dim_order=DataAugmentation.DIM_ORDER,
        )
    with pytest.raises(ValueError):
        DataAugmentation(
            volume_size={"x": 11, "y": 11, "z": 7},
            augment_likelihood=1,
            scale_range=((3, 3),) * 2,
            data_dim_order=DataAugmentation.DIM_ORDER,
        )


def test_augment_call_bad_input(cube_with_side_dot):
    c, y, x, z = cube_with_side_dot.shape

    augmenter = DataAugmentation(
        volume_size={"x": x, "y": y, "z": z},
        augment_likelihood=1,
        flippable_axis=(0, 2),
        data_dim_order=DataAugmentation.DIM_ORDER,
    )
    assert augmenter.update_parameters()

    # call with good data
    augmenter(torch.empty((c, y, x, z), dtype=cube_with_side_dot.dtype))

    # and now with bad
    with pytest.raises(ValueError):
        # wrong num of dims
        augmenter(torch.empty((c, y, x, z, 2), dtype=cube_with_side_dot.dtype))
    with pytest.raises(ValueError):
        # wrong axis size
        augmenter(
            torch.empty((c, y + 1, x, z), dtype=cube_with_side_dot.dtype)
        )
