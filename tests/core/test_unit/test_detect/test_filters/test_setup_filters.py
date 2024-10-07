import pickle

import numpy as np
import pytest

import cellfinder.core.tools.tools as tools
from cellfinder.core.detect.filters.setup_filters import DetectionSettings


@pytest.mark.parametrize(
    "in_dtype,filter_dtype",
    [
        (np.uint8, np.float32),
        (np.uint16, np.float32),
        (np.uint32, np.float64),
        (np.int8, np.float32),
        (np.int16, np.float32),
        (np.int32, np.float64),
        (np.float32, np.float32),
        (np.float64, np.float64),
    ],
)
@pytest.mark.parametrize(
    "detect_dtype",
    [
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.float32,
        np.float64,
    ],
)
def test_good_input_dtype(in_dtype, filter_dtype, detect_dtype):
    """
    These input and filter types doesn't require any conversion because the
    filter type we use for the given input type is large enough to not need
    scaling. So data should be identical after conversion to filtering type.
    """
    settings = DetectionSettings(
        plane_original_np_dtype=in_dtype, detection_dtype=detect_dtype
    )
    converter = settings.filter_data_converter_func
    detection_converter = settings.detection_data_converter_func

    assert settings.filtering_dtype == filter_dtype
    assert settings.detection_dtype == detect_dtype

    # min input value can be converted to filter/detection type
    src = np.full(
        5, tools.get_min_possible_int_value(in_dtype), dtype=in_dtype
    )
    dest = converter(src)
    assert np.array_equal(src, dest)
    assert dest.dtype == filter_dtype
    # it is safe to do this conversion for any data type because it only uses
    # the soma_centre_value, and ignores everything else (e.g. outside range)
    assert detection_converter(dest).dtype == detect_dtype

    # typical input value can be converted to filter/detection type
    src = np.full(5, 3, dtype=in_dtype)
    dest = converter(src)
    assert np.array_equal(src, dest)
    assert dest.dtype == filter_dtype
    assert detection_converter(dest).dtype == detect_dtype

    # max input value can be converted to filter/detection type
    src = np.full(
        5, tools.get_max_possible_int_value(in_dtype), dtype=in_dtype
    )
    dest = converter(src)
    assert np.array_equal(src, dest)
    assert dest.dtype == filter_dtype
    assert detection_converter(dest).dtype == detect_dtype

    # soma_centre_value can be converted to filter/detection type
    src = np.full(5, settings.soma_centre_value, dtype=in_dtype)
    dest = converter(src)
    # data type is larger - so value is unchanged
    assert np.array_equal(src, dest)
    assert dest.dtype == filter_dtype
    # for detect, we convert soma_centre_value to detection_soma_centre_value
    detect = detection_converter(dest)
    assert detect.dtype == detect_dtype
    assert np.all(detect == settings.detection_soma_centre_value)


@pytest.mark.parametrize("in_dtype", [np.uint64, np.int64])
def test_bad_input_dtype(in_dtype):
    """
    For this input type, to be able to fit it into our largest filtering
    type - float64 we'd need to scale the data. Although `converter` can do
    it, we don't support it right now (maybe as an option?)
    """
    settings = DetectionSettings(plane_original_np_dtype=in_dtype)

    with pytest.raises(TypeError):
        # do assert to quiet linter complaints
        assert settings.filter_data_converter_func

    with pytest.raises(TypeError):
        assert settings.filtering_dtype

    # detection type should be available
    assert settings.detection_dtype == np.uint64


def test_pickle_settings():
    settings = DetectionSettings()

    # get some properties, both cached and not cached
    assert settings.filter_data_converter_func is not None
    assert settings.filtering_dtype is not None
    assert settings.detection_dtype is not None
    assert settings.threshold_value is not None
    assert settings.plane_shape is not None

    # make sure pickle works
    s = pickle.dumps(settings)
    assert s


def test_bad_ball_z_size():
    settings = DetectionSettings(ball_z_size_um=0)
    with pytest.raises(ValueError):
        # do something with value to quiet linter
        assert settings.ball_z_size
