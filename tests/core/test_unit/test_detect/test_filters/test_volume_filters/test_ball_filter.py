import numpy as np
import pytest
import torch

from cellfinder.core.detect.filters.volume.ball_filter import BallFilter

bf_kwargs = {
    "plane_height": 50,
    "plane_width": 50,
    "ball_xy_size": 3,
    "ball_z_size": 3,
    "overlap_fraction": 0.5,
    "threshold_value": 1,
    "soma_centre_value": 1,
    "tile_height": 10,
    "tile_width": 10,
    "dtype": "float32",
    "torch_device": "cpu",
}


def test_filter_not_ready():
    bf = BallFilter(**bf_kwargs)
    assert not bf.ready

    with pytest.raises(TypeError):
        bf.get_processed_planes()

    with pytest.raises(TypeError):
        bf.get_raw_planes()

    with pytest.raises(TypeError):
        bf.walk()


@pytest.mark.parametrize(
    "sizes", [(1, 0, 0), (2, 1, 0), (3, 1, 1), (4, 2, 1), (5, 2, 2), (6, 3, 2)]
)
def test_filter_plane_params(sizes):
    kernel_size, start_offset, remaining = sizes
    # we get exactly one plane out of a volume that is the same size as the
    # kernel start_offset is index of first valid plane. Plus remaining is last
    # index. Plus 1 is size
    assert kernel_size == start_offset + 1 + remaining

    kwargs = bf_kwargs.copy()
    kwargs["ball_z_size"] = kernel_size
    bf = BallFilter(**kwargs)

    assert bf.first_valid_plane == start_offset
    assert bf.remaining_planes == remaining


@pytest.mark.parametrize("batch_size", [1, 2, 5, 10])
@pytest.mark.parametrize("kernel_size", [1, 2, 3, 5])
def test_filtered_planes(kernel_size, batch_size):
    kwargs = bf_kwargs.copy()
    kwargs["ball_z_size"] = kernel_size
    bf = BallFilter(**kwargs, use_mask=False)

    num_planes = 20
    n_batches = num_planes // batch_size
    total_planes = n_batches * batch_size
    sent_planes = 0
    gotten_planes = 0
    num_padded_planes = kernel_size - 1

    h, w = kwargs["plane_height"], kwargs["plane_width"]
    data = torch.arange(total_planes * h * w).reshape((total_planes, h, w))
    data = data.to(
        dtype=getattr(torch, kwargs["dtype"]), device=kwargs["torch_device"]
    )
    data_np = data.numpy()

    all_raw_planes = []
    for i in range(n_batches):
        bf.append(
            data[i * batch_size : (i + 1) * batch_size],
            raw_planes=data_np[i * batch_size : (i + 1) * batch_size],
        )
        sent_planes += batch_size
        # volume only includes batch and some padding from end of last batch
        assert bf.volume.shape[0] <= batch_size + kernel_size - 1

        if bf.ready:
            # no need to walk because walking only modifies the contents not
            # size of volume
            planes = bf.get_processed_planes()
            raw_planes = bf.get_raw_planes()
            all_raw_planes.extend(raw_planes)
            # first batch is 1 or batch minus padding. Remaining is batch size
            assert planes.shape[0] in (
                1,
                batch_size,
                batch_size - num_padded_planes,
            )
            assert len(raw_planes) == planes.shape[0]

            for raw_plane in raw_planes:
                assert raw_plane.shape == planes.shape[1:]

            gotten_planes += planes.shape[0]

    assert gotten_planes == sent_planes - num_padded_planes
    all_raw_planes_np = np.stack(all_raw_planes, axis=0)
    p1 = bf.first_valid_plane
    data_np_unpadded = data_np[p1 : total_planes - (num_padded_planes - p1)]

    assert data_np_unpadded.shape == all_raw_planes_np.shape
    assert np.array_equal(data_np_unpadded, all_raw_planes_np)
