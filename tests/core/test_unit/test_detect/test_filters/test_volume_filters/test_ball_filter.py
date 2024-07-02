import pytest

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
}


def test_filter_not_ready():
    bf = BallFilter(**bf_kwargs)
    assert not bf.ready

    with pytest.raises(TypeError):
        bf.get_processed_planes()

    with pytest.raises(TypeError):
        bf.walk()


@pytest.mark.parametrize(
    "sizes", [(1, 0, 0), (2, 1, 0), (3, 1, 1), (4, 2, 1), (5, 2, 2), (6, 3, 2)]
)
def test_filter_unprocessed_planes(sizes):
    kernel_size, start_offset, remaining = sizes
    assert kernel_size == start_offset + 1 + remaining

    kwargs = bf_kwargs.copy()
    kwargs["ball_z_size"] = kernel_size
    bf = BallFilter(**kwargs)

    assert bf.first_valid_plane == start_offset
    assert bf.remaining_planes == remaining
