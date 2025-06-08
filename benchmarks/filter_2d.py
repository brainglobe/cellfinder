import os
import sys

sys.path.append(os.path.dirname(__file__))

import numpy as np
import torch
from benchmark_tools import (
    compare_results,
    get_test_data_path,
    profile_cpu,
    profile_cuda,
    time_filters,
)
from brainglobe_utils.IO.image.load import read_with_dask

from cellfinder.core.detect.filters.plane import TileProcessor
from cellfinder.core.detect.filters.setup_filters import DetectionSettings


def setup_filter(
    signal_path,
    batch_size=1,
    num_z=None,
    torch_device="cpu",
    dtype=np.uint16,
    use_scipy=False,
):
    signal_array = read_with_dask(signal_path)
    num_z = num_z or len(signal_array)
    signal_array = np.asarray(signal_array[:num_z]).astype(dtype)
    shape = signal_array.shape

    settings = DetectionSettings(
        plane_original_np_dtype=dtype,
        plane_shape=shape[1:],
        torch_device=torch_device,
        voxel_sizes=(5.06, 4.5, 4.5),
        soma_diameter_um=30,
        ball_xy_size_um=6,
        ball_z_size_um=15,
    )
    signal_array = settings.filter_data_converter_func(signal_array)
    signal_array = torch.from_numpy(signal_array).to(torch_device)

    tile_processor = TileProcessor(
        plane_shape=shape[1:],
        clipping_value=settings.clipping_value,
        threshold_value=settings.threshold_value,
        soma_diameter=settings.soma_diameter,
        log_sigma_size=settings.log_sigma_size,
        n_sds_above_mean_thresh=settings.n_sds_above_mean_thresh,
        n_sds_above_mean_tiled_thresh=settings.n_sds_above_mean_tiled_thresh,
        tiled_thresh_tile_size=settings.tiled_thresh_tile_size,
        torch_device=torch_device,
        dtype=settings.filtering_dtype.__name__,
        use_scipy=use_scipy,
    )

    return tile_processor, signal_array, batch_size


def run_filter(tile_processor, signal_array, batch_size):
    for i in range(0, len(signal_array), batch_size):
        tile_processor.get_tile_mask(signal_array[i : i + batch_size])


if __name__ == "__main__":
    with torch.inference_mode(True):
        n = 5
        batch_size = 2
        signal_path = get_test_data_path("bright_brain/signal")

        compare_results(
            time_filters(
                n,
                run_filter,
                setup_filter(
                    signal_path,
                    batch_size=batch_size,
                    torch_device="cpu",
                    use_scipy=False,
                ),
                "cpu-no_scipy",
            ),
            time_filters(
                n,
                run_filter,
                setup_filter(
                    signal_path,
                    batch_size=batch_size,
                    torch_device="cpu",
                    use_scipy=True,
                ),
                "cpu-scipy",
            ),
            time_filters(
                n,
                run_filter,
                setup_filter(
                    signal_path, batch_size=batch_size, torch_device="cuda"
                ),
                "cuda",
            ),
        )

        profile_cpu(
            n,
            run_filter,
            setup_filter(
                signal_path,
                batch_size=batch_size,
                torch_device="cpu",
                use_scipy=False,
            ),
        )
        profile_cpu(
            n,
            run_filter,
            setup_filter(
                signal_path,
                batch_size=batch_size,
                torch_device="cpu",
                use_scipy=True,
            ),
        )
        profile_cuda(
            n,
            run_filter,
            setup_filter(
                signal_path, batch_size=batch_size, torch_device="cuda"
            ),
        )
