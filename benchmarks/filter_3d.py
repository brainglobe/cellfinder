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

from cellfinder.core.detect.filters.setup_filters import DetectionSettings
from cellfinder.core.detect.filters.volume.ball_filter import BallFilter


def setup_filter(
    plane_path,
    tiles_path,
    batch_size=1,
    num_z=None,
    torch_device="cpu",
    dtype=np.uint16,
):
    filtered = read_with_dask(plane_path)
    tiles = read_with_dask(tiles_path)
    num_z = num_z or len(filtered)
    filtered = np.asarray(filtered[:num_z])
    tiles = np.asarray(tiles[:num_z])
    shape = filtered.shape

    settings = DetectionSettings(
        plane_original_np_dtype=dtype,
        plane_shape=shape[1:],
        torch_device=torch_device,
        soma_diameter_um=30,
        ball_xy_size_um=6,
        ball_z_size_um=15,
    )
    filtered = filtered.astype(settings.filtering_dtype)
    filtered = torch.from_numpy(filtered).to(torch_device)
    tiles = tiles.astype(np.bool_)
    tiles = torch.from_numpy(tiles).to(torch_device)

    ball_filter = BallFilter(
        plane_height=settings.plane_height,
        plane_width=settings.plane_width,
        ball_xy_size=settings.ball_xy_size,
        ball_z_size=settings.ball_z_size,
        overlap_fraction=settings.ball_overlap_fraction,
        threshold_value=settings.threshold_value,
        soma_centre_value=settings.soma_centre_value,
        tile_height=settings.tile_height,
        tile_width=settings.tile_width,
        dtype=settings.filtering_dtype.__name__,
        batch_size=batch_size,
        torch_device=torch_device,
        use_mask=True,
    )

    return ball_filter, filtered, tiles, batch_size


def run_filter(ball_filter, filtered, tiles, batch_size):
    for i in range(0, len(filtered), batch_size):
        ball_filter.append(
            filtered[i : i + batch_size], tiles[i : i + batch_size]
        )
        if ball_filter.ready:
            ball_filter.walk()
            ball_filter.get_processed_planes()


if __name__ == "__main__":
    with torch.inference_mode(True):
        n = 5
        batch_size = 4
        plane_path = get_test_data_path("bright_brain/2d_filter")
        tiles_path = get_test_data_path("bright_brain/tiles")

        compare_results(
            time_filters(
                n,
                run_filter,
                setup_filter(
                    plane_path,
                    tiles_path,
                    batch_size=batch_size,
                    torch_device="cpu",
                ),
                "cpu",
            ),
            time_filters(
                n,
                run_filter,
                setup_filter(
                    plane_path,
                    tiles_path,
                    batch_size=batch_size,
                    torch_device="cuda",
                ),
                "cuda",
            ),
        )

        profile_cpu(
            n,
            run_filter,
            setup_filter(
                plane_path,
                tiles_path,
                batch_size=batch_size,
                torch_device="cpu",
            ),
        )
        profile_cuda(
            n,
            run_filter,
            setup_filter(
                plane_path,
                tiles_path,
                batch_size=batch_size,
                torch_device="cuda",
            ),
        )
