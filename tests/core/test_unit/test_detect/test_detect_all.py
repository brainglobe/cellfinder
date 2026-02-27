import numpy as np

from cellfinder.core.detect.detect import main, parse_range
from cellfinder.core.detect.filters.plane import TileProcessor
from cellfinder.core.detect.filters.setup_filters import DetectionSettings
from cellfinder.core.detect.python_api import main as api_main

# ---------- Dummy data for tests ----------
dummy_3d = np.zeros((5, 10, 10), dtype=np.float32)
dummy_2d = np.zeros((10, 10), dtype=np.float32)


# ---------- 1. Test parse_range ----------
def test_parse_range_basic():
    s = parse_range("1:3", 10)
    assert s.start == 1 and s.stop == 3
    s2 = parse_range(None, 5)
    assert s2.start == 0 and s2.stop == 5


# ---------- 2. Test main() in detect.py ----------
def test_main_basic():
    result = main(dummy_3d)
    assert isinstance(result, list)


def test_main_slices():
    result = main(
        dummy_3d, start_plane=1, end_plane=3, x_range="0:5", y_range="0:5"
    )
    assert isinstance(result, list)


# ---------- 3. Test python_api.py ----------
def test_api_main_basic():
    result = api_main(dummy_3d)
    assert isinstance(result, list)


# ---------- 4. Test DetectionSettings ----------
def test_detection_settings():
    settings = DetectionSettings(
        plane_shape=(10, 10),
        plane_original_np_dtype=np.float32,
        voxel_sizes=[1, 1, 1],
        soma_diameter_um=16,
        ball_xy_size_um=6,
        ball_z_size_um=15,
    )
    assert settings.plane_shape == (10, 10)


# ---------- 5. Optional: test dummy 2D filtering ----------
def test_tile_processor_dummy():
    # create settings for TileProcessor
    settings = DetectionSettings(
        plane_shape=(10, 10),
        plane_original_np_dtype=np.float32,
        voxel_sizes=[1, 1, 1],
        soma_diameter_um=16,
        ball_xy_size_um=6,
        ball_z_size_um=15,
    )
    tp = TileProcessor(
        plane_shape=settings.plane_shape,
        clipping_value=255,
        threshold_value=254,
        n_sds_above_mean_thresh=10,
        n_sds_above_mean_tiled_thresh=10,
        tiled_thresh_tile_size=1,
        log_sigma_size=0.2,
        soma_diameter=settings.soma_diameter,
        torch_device="cpu",
        dtype="float32",
        use_scipy=True,
    )
    assert tp is not None
