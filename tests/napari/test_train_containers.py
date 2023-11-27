from inspect import signature

import pytest

from cellfinder.core.train.train_yml import run
from cellfinder.napari.train.train_containers import (
    MiscTrainingInputs,
    OptionalNetworkInputs,
    OptionalTrainingInputs,
    TrainingDataInputs,
)


@pytest.mark.parametrize(
    argnames="input_container",
    argvalues=[
        MiscTrainingInputs(),
        OptionalNetworkInputs(),
        OptionalTrainingInputs(),
        TrainingDataInputs(),
    ],
)
def test_core_args_passed(input_container):
    """
    Check that any keyword argument that napari passes
    to the training backend actually is also expected by the backend
    """
    backend_signature = signature(run)
    expected_kwargs_set = set(backend_signature.parameters.keys())
    actual_kwargs_set = set(input_container.as_core_arguments().keys())
    # check all actual keywords are in expected (but not the other way around.)
    assert actual_kwargs_set <= expected_kwargs_set
