import numpy as np
import pytest

from cellfinder.core.classify.resnet import build_model


@pytest.mark.parametrize(
    "shape, dimensions",
    [
        ((50, 50, 1), 2),
        ((50, 50, 2), 2),
        ((50, 50, 20, 1), 3),
        ((50, 50, 20, 2), 3),
    ],
)
def test_build_model_dimensionality(shape, dimensions):
    model = build_model(shape=shape, network_depth="18-layer")
    assert model.input_shape == (None, *shape)
    assert model.output_shape == (None, 2)

    out = model(np.zeros((2, *shape), dtype="float32"), training=False)
    assert tuple(out.shape) == (2, 2)


def test_dimensions_inferred_from_shape():
    assert build_model(shape=(50, 50, 1)).input_shape == (None, 50, 50, 1)
    assert build_model(shape=(50, 50, 20, 2)).input_shape == (
        None,
        50,
        50,
        20,
        2,
    )


def test_explicit_dimensions_override():
    model = build_model(shape=(50, 50, 3), dimensions=2)
    assert model.output_shape == (None, 2)


def test_unsupported_dimensionality_raises():
    with pytest.raises(ValueError, match="Unsupported spatial dimensionality"):
        build_model(shape=(50,), dimensions=1)
