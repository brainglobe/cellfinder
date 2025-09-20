import os

import pytest
from pytest_mock.plugin import MockerFixture

from cellfinder.core.train.train_yaml import cli as train_run

data_dir = os.path.join(
    os.getcwd(), "tests", "data", "integration", "training"
)
cell_cubes = os.path.join(data_dir, "cells")
non_cell_cubes = os.path.join(data_dir, "non_cells")
training_yaml_file = os.path.join(data_dir, "training.yaml")
training_yaml_file_stats = os.path.join(data_dir, "training_with_stats.yaml")


EPOCHS = "2"

# only checks that the model is trained, and then saved.
# doesn't check that it works etc


@pytest.mark.slow
def test_train(mocker, tmpdir):
    tmpdir = str(tmpdir)

    train_args = [
        "cellfinder_train",
        "-y",
        training_yaml_file,
        "-o",
        tmpdir,
        "--epochs",
        EPOCHS,
    ]
    mocker.patch("sys.argv", train_args)

    train_run()

    model_file = os.path.join(tmpdir, "model.keras")
    assert os.path.exists(model_file)


@pytest.mark.parametrize("normalize", [True, False])
@pytest.mark.parametrize("has_norms", [True, False])
def test_train_normalization_missing_stats(
    mocker: MockerFixture, tmpdir, has_norms, normalize
):
    tmpdir = str(tmpdir)

    train_args = [
        "cellfinder_train",
        "-y",
        training_yaml_file_stats if has_norms else training_yaml_file,
        "-o",
        tmpdir,
        "--epochs",
        EPOCHS,
    ]
    if normalize:
        train_args.append("--normalize-channels")

    mocker.patch("sys.argv", train_args)
    get_model = mocker.patch(
        "cellfinder.core.train.train_yaml.get_model", autospec=True
    )

    if normalize and not has_norms:
        # if the yaml doesn't have normalization info an error will be raised
        with pytest.raises(ValueError):
            train_run()
    else:
        train_run()
        # get the data sets passed to fit() to verify if it has norm data
        # there's no clear name property of the mock fit call, so use its repr
        (fit_mock,) = [
            m
            for m in get_model.mock_calls
            if repr(m).startswith("call().fit(")
        ]
        train_dataset = fit_mock.kwargs["x"].dataset
        val_dataset = fit_mock.kwargs["validation_data"].dataset

        if normalize:
            # if we normalize, the normalization data should be in dataset
            assert train_dataset.points_norm_arr is not None
            assert val_dataset.points_norm_arr is not None
        else:
            # otherwise, no normalization data should have been passed, even if
            # the yaml has it
            assert train_dataset.points_norm_arr is None
            assert val_dataset.points_norm_arr is None
