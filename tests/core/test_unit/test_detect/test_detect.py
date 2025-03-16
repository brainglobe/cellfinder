from unittest.mock import MagicMock

import numpy as np
import pytest
from pytest_mock.plugin import MockerFixture

from cellfinder.core.detect.detect import main


@pytest.fixture
def mocked_main(mocker: MockerFixture):
    from cellfinder.core.detect.filters.volume.volume_filter import (
        VolumeFilter,
    )

    process = mocker.patch.object(VolumeFilter, "process", autospec=True)
    get_results = mocker.patch.object(
        VolumeFilter, "get_results", autospec=True
    )

    return process, get_results


def test_main_bad_signal_arg(mocked_main):
    # should work
    main(signal_array=np.empty((5, 50, 50)))

    with pytest.raises(ValueError):
        main(signal_array=np.empty((1, 1, 1, 1)))

    with pytest.raises(ValueError):
        main(signal_array=np.empty((1, 1)))

    with pytest.raises(TypeError):
        main(signal_array=np.empty((5, 50, 50), dtype=np.str_))

    with pytest.raises(TypeError):
        main(signal_array=np.empty((5, 50, 50), dtype=np.uint64))

    with pytest.raises(TypeError):
        main(signal_array=np.empty((5, 50, 50), dtype=np.int64))


@pytest.mark.parametrize(
    "dtype",
    [
        np.uint8,
        np.uint16,
        np.uint32,
        np.int8,
        np.int16,
        np.int32,
        np.float32,
        np.float64,
    ],
)
def test_main_good_signal_arg(mocked_main, dtype):
    main(signal_array=np.empty((5, 50, 50)))


def test_main_bad_or_default_args(mocked_main):
    process, get_results = mocked_main
    main(
        signal_array=np.empty((5, 8, 19), dtype=np.uint16),
        end_plane=-1,
        batch_size=None,
        torch_device="cpu",
    )
    process.assert_called()
    get_results.assert_called()

    vol_filter, mp_tile_processor, signal_array = process.call_args.args
    (
        _,
        splitting_settings,
    ) = get_results.call_args.args
    settings = vol_filter.settings

    assert settings.plane_shape == (8, 19)
    assert settings.plane_original_np_dtype == np.uint16
    # for uint16 input we should use float32
    assert settings.filtering_dtype == np.float32
    assert settings.detection_dtype == np.uint64
    assert settings.end_plane == 5
    assert settings.n_planes == 5
    assert settings.batch_size == 4  # cpu default is 4

    assert splitting_settings.torch_device == "cpu"


def test_main_planes_crop_size(mocked_main):
    process, get_results = mocked_main
    main(
        signal_array=np.empty((5, 50, 41)),
        start_width=2,
        end_width=40,
        start_height=8,
        end_height=21,
    )

    process.assert_called()
    vol_filter, mp_tile_processor, signal_array = process.call_args.args
    settings = vol_filter.settings

    assert settings.plane_shape == (13, 38)
    assert settings.end_height == 21
    assert settings.end_width == 40
    assert settings.n_planes == 5


def test_main_planes_size(mocked_main):
    process, get_results = mocked_main
    main(signal_array=np.empty((5, 8, 19)), end_plane=4, start_plane=1)

    process.assert_called()
    vol_filter, mp_tile_processor, signal_array = process.call_args.args
    settings = vol_filter.settings

    assert settings.plane_shape == (8, 19)
    assert settings.end_plane == 4
    assert settings.n_planes == 3


def test_main_splitting_cpu_cuda(mocker: MockerFixture):
    # checks that even if main filtering runs on cuda, the structure splitting
    # only runs on cpu
    # patch anything that would do with cuda - in case there's no cuda
    vol: MagicMock = mocker.patch(
        "cellfinder.core.detect.detect.VolumeFilter", autospec=True
    )
    mocker.patch("cellfinder.core.detect.detect.TileProcessor", autospec=True)

    main(
        signal_array=np.empty((5, 8, 19)), batch_size=None, torch_device="cuda"
    )

    settings = vol.call_args.kwargs["settings"]
    (splitting_settings,) = vol.return_value.get_results.call_args.args

    assert settings.torch_device == "cuda"
    assert settings.batch_size == 1  # cuda default is 1

    assert splitting_settings.torch_device == "cpu"
