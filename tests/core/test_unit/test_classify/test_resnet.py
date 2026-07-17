import numpy as np
import pytest

from cellfinder.core.classify.resnet import build_model


@pytest.mark.parametrize(
    "shape",
    [
        (50, 50, 1),
        (50, 50, 2),
        (50, 50, 20, 1),
        (50, 50, 20, 2),
    ],
)
def test_build_model_dimensionality(shape):
    """The network rank follows the input shape, and it runs a batch."""
    model = build_model(shape=shape, network_depth="18-layer")
    assert model.input_shape == (None, *shape)
    assert model.output_shape == (None, 2)

    out = model(np.zeros((2, *shape), dtype="float32"), training=False)
    assert tuple(out.shape) == (2, 2)


def test_unsupported_dimensionality_raises():
    with pytest.raises(ValueError, match="Unsupported spatial dimensionality"):
        build_model(shape=(50,))
