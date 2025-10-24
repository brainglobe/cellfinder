import numpy as np
import pytest
import torch
from brainglobe_utils.IO.image.load import read_with_dask

from cellfinder.core.detect.filters.plane import TileProcessor
from cellfinder.core.detect.filters.plane.classical_filter import PeakEnhancer
from cellfinder.core.detect.filters.plane.tile_walker import TileWalker
from cellfinder.core.detect.filters.setup_filters import DetectionSettings
from cellfinder.core.tools.IO import fetch_pooch_directory
from cellfinder.core.tools.tools import (
    get_max_possible_int_value,
    inference_wrapper,
)


def load_pooch_dir(test_data_registry, path):
    data_path = fetch_pooch_directory(test_data_registry, path)
    return read_with_dask(data_path)


@pytest.mark.parametrize(
    "signal,enhanced,soma_diameter",
    [
        ("edge_cells_brain/signal", "edge_cells_brain/peak_enhanced", 16),
        ("bright_brain/signal", "bright_brain/peak_enhanced", 30),
    ],
)
@pytest.mark.parametrize(
    "torch_device,use_scipy", [("cpu", False), ("cpu", True), ("cuda", False)]
)
@inference_wrapper
def test_2d_filtering_peak_enhance_parity(
    signal,
    enhanced,
    soma_diameter,
    torch_device,
    use_scipy,
    test_data_registry,
):
    # test that the pure 2d plane filtering (median, gauss, laplacian) matches
    # exactly. We use float64 in the test, like original code we compare to
    # used
    if torch_device == "cuda" and not torch.cuda.is_available():
        pytest.skip("Cuda is not available")

    # check input data size/type is as expected
    data = np.asarray(load_pooch_dir(test_data_registry, signal))
    enhanced = np.asarray(load_pooch_dir(test_data_registry, enhanced))
    assert data.dtype == np.uint16
    assert enhanced.dtype == np.uint32
    assert data.shape == enhanced.shape

    # convert to working type and send to cpu/cuda, use float64, the type
    # used originally
    data = data.astype(np.float64)
    data = torch.from_numpy(data)
    data = data.to(torch_device)

    # first check that the raw filters produce the same output
    clip = get_max_possible_int_value(np.uint32)
    enhancer = PeakEnhancer(
        torch_device=torch_device,
        dtype=torch.float64,
        clipping_value=clip,
        laplace_gaussian_sigma=soma_diameter * 0.2,
        use_scipy=use_scipy,
    )
    enhanced_our = enhancer.enhance_peaks(data)
    enhanced_our = enhanced_our.cpu().numpy().astype(np.uint32)

    assert enhanced_our.shape == enhanced.shape
    # the number of pixels per plane that are different
    different = np.sum(
        np.sum(np.logical_not(np.isclose(enhanced_our, enhanced)), axis=2),
        axis=1,
    )
    assert np.all(different == 0)


@pytest.mark.parametrize(
    "signal,filtered,tiles,soma_diameter,max_different",
    [
        (
            "edge_cells_brain/signal",
            "edge_cells_brain/2d_filter",
            "edge_cells_brain/tiles",
            16,
            1,
        ),
        (
            "bright_brain/signal",
            "bright_brain/2d_filter",
            "bright_brain/tiles",
            30,
            2,
        ),
    ],
)
@pytest.mark.parametrize(
    "torch_device,use_scipy", [("cpu", False), ("cpu", True), ("cuda", False)]
)
@inference_wrapper
def test_2d_filtering_parity(
    signal,
    filtered,
    tiles,
    soma_diameter,
    max_different,
    torch_device,
    use_scipy,
    test_data_registry,
):
    # test that the pixels marked as bright after 2d plane filtering matches
    # now we don't always use float64, but the bright pixels will still stay
    # the same. Unlike test_2d_filtering_peak_enhance, we use float32 if the
    # input data fits in it, like we do in the codebase. So we want to be sure
    # that the number of bright pixels doesn't change much. Because running at
    # full float64 is expensive
    if torch_device == "cuda" and not torch.cuda.is_available():
        pytest.skip("Cuda is not available")

    # check input data size/type is as expected
    data = np.asarray(load_pooch_dir(test_data_registry, signal))
    filtered = np.asarray(load_pooch_dir(test_data_registry, filtered))
    tiles = np.asarray(load_pooch_dir(test_data_registry, tiles))
    assert data.dtype == np.uint16
    assert filtered.dtype == np.uint16
    assert data.shape == filtered.shape

    settings = DetectionSettings(plane_original_np_dtype=np.uint16)
    # convert to working type and send to cpu/cuda
    data = torch.from_numpy(settings.filter_data_converter_func(data))
    data = data.to(torch_device)

    tile_processor = TileProcessor(
        plane_shape=data[0, :, :].shape,
        clipping_value=settings.clipping_value,
        threshold_value=settings.threshold_value,
        soma_diameter=soma_diameter,
        log_sigma_size=0.2,
        n_sds_above_mean_thresh=10,
        n_sds_above_mean_tiled_thresh=10,
        tiled_thresh_tile_size=0,
        torch_device=torch_device,
        dtype=settings.filtering_dtype.__name__,
        use_scipy=use_scipy,
    )

    # apply filter and get data back
    filtered_our, tiles_our = tile_processor.get_tile_mask(data)
    filtered_our = filtered_our.cpu().numpy().astype(np.uint16)
    tiles_our = tiles_our.cpu().numpy()

    assert filtered_our.shape == filtered.shape
    # we don't care about exact pixel values, only which pixels are marked
    # bright and which aren't. Bright per plane
    bright = np.sum(
        np.sum(filtered == settings.threshold_value, axis=2), axis=1
    )
    bright_our = np.sum(
        np.sum(filtered_our == settings.threshold_value, axis=2), axis=1
    )
    # the number of pixels different should be less than 2!
    assert np.all(np.less(np.abs(bright - bright_our), max_different + 1))

    # the in/out of brain tiles though should be identical
    assert tiles_our.shape == tiles.shape
    assert tiles_our.dtype == tiles.dtype
    assert np.array_equal(tiles_our, tiles)


@pytest.mark.parametrize(
    "plane_size",
    [(1, 2), (2, 1), (2, 2), (2, 3), (3, 3), (2, 5), (22, 33), (200, 200)],
)
@inference_wrapper
def test_2d_filter_padding(plane_size):
    # check that filter padding works correctly for different sized inputs -
    # even if the input is smaller than filter sizes
    settings = DetectionSettings(plane_original_np_dtype=np.uint16)
    data = np.random.randint(0, 500, size=(1, *plane_size))
    data = data.astype(settings.filtering_dtype)

    tile_processor = TileProcessor(
        plane_shape=plane_size,
        clipping_value=settings.clipping_value,
        threshold_value=settings.threshold_value,
        soma_diameter=16,
        log_sigma_size=0.2,
        n_sds_above_mean_thresh=10,
        n_sds_above_mean_tiled_thresh=10,
        tiled_thresh_tile_size=0,
        torch_device="cpu",
        dtype=settings.filtering_dtype.__name__,
        use_scipy=False,
    )

    filtered, _ = tile_processor.get_tile_mask(torch.from_numpy(data))
    filtered = filtered.numpy()
    assert filtered.shape == data.shape


@inference_wrapper
def test_even_filter_kernel():
    with pytest.raises(ValueError):
        try:
            n = PeakEnhancer.median_filter_size
            PeakEnhancer.median_filter_size = 4
            PeakEnhancer(
                "cpu",
                torch.float32,
                clipping_value=5,
                laplace_gaussian_sigma=3.0,
                use_scipy=False,
            )
        finally:
            PeakEnhancer.median_filter_size = n

    enhancer = PeakEnhancer(
        "cpu",
        torch.float32,
        clipping_value=5,
        laplace_gaussian_sigma=3.0,
        use_scipy=False,
    )

    assert enhancer.gaussian_filter_size % 2

    _, _, x, y = enhancer.lap_kernel.shape
    assert x % 2, "Should be odd"
    assert y % 2, "Should be odd"
    assert x == y


@pytest.mark.parametrize(
    "sizes",
    [((1, 1), (1, 1)), ((1, 2), (1, 1)), ((2, 1), (1, 1)), ((22, 33), (3, 4))],
)
@inference_wrapper
def test_tile_walker_size(sizes, soma_diameter=5):
    plane_size, tile_size = sizes
    walker = TileWalker(plane_size, soma_diameter=soma_diameter)
    assert walker.tile_height == 10
    assert walker.tile_width == 10

    data = torch.rand((1, *plane_size), dtype=torch.float32)
    tiles = walker.get_bright_tiles(data)
    assert tiles.shape == (1, *tile_size)


def get_filtered_data(
    data: np.ndarray,
    soma_diameter=16,
    log_sigma_size=0.2,
    n_sds_above_mean_thresh=10.0,
    n_sds_above_mean_tiled_thresh=10.0,
    tiled_thresh_tile_size=0.0,
) -> np.ndarray:
    settings = DetectionSettings(plane_original_np_dtype=np.uint16)
    data = data.astype(settings.filtering_dtype)

    tile_processor = TileProcessor(
        plane_shape=data.shape[1:],
        clipping_value=settings.clipping_value,
        threshold_value=settings.threshold_value,
        soma_diameter=soma_diameter,
        log_sigma_size=log_sigma_size,
        n_sds_above_mean_thresh=n_sds_above_mean_thresh,
        n_sds_above_mean_tiled_thresh=n_sds_above_mean_tiled_thresh,
        tiled_thresh_tile_size=tiled_thresh_tile_size,
        torch_device="cpu",
        dtype=settings.filtering_dtype.__name__,
        use_scipy=True,
    )

    filtered, _ = tile_processor.get_tile_mask(torch.from_numpy(data))
    return (filtered == settings.threshold_value).numpy()


def test_2d_filter_plane_threshold_single_spot():
    # make bright area of 5x5 = 25
    data = np.zeros((1, 50, 50))
    data[0, 23:28, 23:28] = 10

    # use normal threshold
    filtered = get_filtered_data(
        data,
        soma_diameter=5,
        log_sigma_size=0.2,
        n_sds_above_mean_thresh=1,
        n_sds_above_mean_tiled_thresh=1,
        tiled_thresh_tile_size=0,
    )
    # about 25 pixels should be marked
    assert 20 <= np.sum(filtered) <= 30

    # use very high threshold
    filtered = get_filtered_data(
        data,
        soma_diameter=5,
        log_sigma_size=0.2,
        n_sds_above_mean_thresh=50,
        n_sds_above_mean_tiled_thresh=1,
        tiled_thresh_tile_size=0,
    )
    # with high threshold, should be no marked pixels
    assert not np.sum(filtered)


def test_2d_filter_plane_threshold_2_spots():
    # create 2 bright areas of 5x5 = 25px, one bright, one darker
    data = np.zeros((1, 50, 50))
    data[0, 13:18, 13:18] = 5
    data[0, 33:38, 33:38] = 20

    # low threshold should get both areas
    filtered = get_filtered_data(
        data,
        soma_diameter=5,
        log_sigma_size=0.2,
        n_sds_above_mean_thresh=0.1,
        n_sds_above_mean_tiled_thresh=1,
        tiled_thresh_tile_size=0,
    )
    assert 40 <= np.sum(filtered) <= 60

    # medium threshold should get very bright area
    filtered = get_filtered_data(
        data,
        soma_diameter=5,
        log_sigma_size=0.2,
        n_sds_above_mean_thresh=2,
        n_sds_above_mean_tiled_thresh=1,
        tiled_thresh_tile_size=0,
    )
    assert 20 <= np.sum(filtered) <= 35

    # high threshold should get no area
    filtered = get_filtered_data(
        data,
        soma_diameter=5,
        log_sigma_size=0.2,
        n_sds_above_mean_thresh=50,
        n_sds_above_mean_tiled_thresh=1,
        tiled_thresh_tile_size=0,
    )
    assert not np.sum(filtered)


def test_2d_filter_tiled_threshold_2_spots():
    # create 2 bright areas of 5x5 = 25px, one bright, one darker
    data = np.zeros((1, 50, 50))
    data[0, 3:8, 3:8] = 5
    data[0, 43:48, 43:48] = 20

    # medium plane threshold should get only very bright area
    filtered = get_filtered_data(
        data,
        soma_diameter=5,
        log_sigma_size=0.2,
        n_sds_above_mean_thresh=2,
        n_sds_above_mean_tiled_thresh=2,
        tiled_thresh_tile_size=0,
    )
    assert 20 <= np.sum(filtered) <= 35

    # with small tiles (size of soma) the mean would be high for the tiles with
    # both bright areas so we should get no pixels
    filtered = get_filtered_data(
        data,
        soma_diameter=5,
        log_sigma_size=0.2,
        n_sds_above_mean_thresh=2,
        n_sds_above_mean_tiled_thresh=2,
        tiled_thresh_tile_size=1,
    )
    assert not np.sum(filtered)

    # but with a very low tiled threshold we should get same as with plane
    # threshold only
    filtered = get_filtered_data(
        data,
        soma_diameter=5,
        log_sigma_size=0.2,
        n_sds_above_mean_thresh=2,
        n_sds_above_mean_tiled_thresh=-2,
        tiled_thresh_tile_size=1,
    )
    assert 20 <= np.sum(filtered) <= 35

    # and with a low plane threshold as well we should get everything
    filtered = get_filtered_data(
        data,
        soma_diameter=5,
        log_sigma_size=0.2,
        n_sds_above_mean_thresh=0,
        n_sds_above_mean_tiled_thresh=-2,
        tiled_thresh_tile_size=1,
    )
    assert 40 <= np.sum(filtered) <= 60


@pytest.mark.parametrize(
    "shape", [(1, 50, 23), (1, 23, 50), (1, 25, 25), (1, 57, 57)]
)
def test_2d_filter_tiled_threshold_odd_shapes(shape):
    # our tile size is 5 * 5 = 25, check that plane shapes that don't fit two
    # tiles or are not multiple of tile size still works
    # create bright area of 5x5 = 25px
    data = np.zeros(shape)
    data[0, 3:8, 3:8] = 5

    # use tiles size of 25 (5 x soma diameter of 5)
    filtered = get_filtered_data(
        data,
        soma_diameter=5,
        log_sigma_size=0.2,
        n_sds_above_mean_thresh=1,
        n_sds_above_mean_tiled_thresh=1,
        tiled_thresh_tile_size=5,
    )
    # about 25 pixels should be marked
    assert 20 <= np.sum(filtered) <= 30


@pytest.mark.parametrize("size", [0, 1, 2, 3])
def test_2d_filter_tiled_threshold_odd_tile_size(size):
    # check that tiny tile sizes works.
    data = np.zeros((1, 10, 10))

    # use tiles size of 25 (5 x soma diameter of 5)
    filtered = get_filtered_data(
        data,
        soma_diameter=1,
        log_sigma_size=0.2,
        n_sds_above_mean_thresh=1,
        n_sds_above_mean_tiled_thresh=1,
        tiled_thresh_tile_size=size,
    )
    assert filtered.shape == (1, 10, 10)
