import os
from unittest.mock import patch

import keras
import pytest
from pytest_mock.plugin import MockerFixture

from cellfinder.core.classify.tools import model_input_channels
from cellfinder.core.train.train_yaml import cli as train_run

data_dir = os.path.join(
    os.getcwd(), "tests", "data", "integration", "training"
)
cell_cubes = os.path.join(data_dir, "cells")
non_cell_cubes = os.path.join(data_dir, "non_cells")
training_yaml_file = os.path.join(data_dir, "training.yaml")
training_yaml_single_channel = os.path.join(
    data_dir, "training_single_channel.yaml"
)
training_yaml_file_stats = os.path.join(data_dir, "training_with_stats.yaml")


def test_cmd_args(mocker: MockerFixture, tmpdir):
    """
    Checks that training is run with expected set of parameters.
    """
    with patch("cellfinder.core.train.train_yaml.run") as train_yaml:
        yaml_files = "a_file_1.yaml", "a_file_2.yaml"

        train_args = [
            "cellfinder_train",
            "-y",
            yaml_files[0],
            yaml_files[1],
            "-o",
            str(tmpdir),
            "--epochs",
            "77",
            "--batch-size",
            "27",
            "--test-fraction",
            "0.21",
            "--learning-rate",
            "0.0023",
            "--lr-schedule",
            "12",
            "37",
            "57",
            "--lr-multiplier",
            "0.61",
            "--augment-likelihood",
            "0.83",
            "--flippable-axis",
            "0",
            "2",
            "--rotate-range",
            "33",
            "157",
            "--translate-range",
            "-0.33",
            "0.47",
            "--scale-range",
            "0.67",
            "1.37",
            "--intensity-range",
            "0.49",
            "1.83",
            "--n-free-cpus",
            "7",
        ]

        mocker.patch("sys.argv", train_args)
        train_run()

        train_yaml.assert_called_once()
        called_kwargs = train_yaml.call_args.kwargs

        assert str(called_kwargs["output_dir"]) == str(tmpdir)
        assert list(called_kwargs["yaml_file"]) == list(yaml_files)
        assert called_kwargs["epochs"] == 77
        assert called_kwargs["batch_size"] == 27
        assert called_kwargs["test_fraction"] == 0.21
        assert called_kwargs["learning_rate"] == 0.0023
        assert list(called_kwargs["lr_schedule"]) == [12, 37, 57]
        assert called_kwargs["lr_multiplier"] == 0.61
        assert called_kwargs["augment_likelihood"] == 0.83
        assert set(called_kwargs["flippable_axis"]) == {0, 2}
        assert (
            list(called_kwargs["rotate_range"])
            == [
                (33, 157),
            ]
            * 3
        )
        assert (
            list(called_kwargs["translate_range"])
            == [
                (-0.33, 0.47),
            ]
            * 3
        )
        assert (
            list(called_kwargs["scale_range"])
            == [
                (0.67, 1.37),
            ]
            * 3
        )
        assert list(called_kwargs["intensity_range"]) == [0.49, 1.83]
        assert called_kwargs["n_free_cpus"] == 7


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


@pytest.mark.slow
def test_train_single_channel(mocker, tmpdir):
    tmpdir = str(tmpdir)

    train_args = [
        "cellfinder_train",
        "-y",
        training_yaml_single_channel,
        "-o",
        tmpdir,
        "--epochs",
        EPOCHS,
    ]
    mocker.patch("sys.argv", train_args)
    train_run()

    model_file = os.path.join(tmpdir, "model.keras")
    assert os.path.exists(model_file)

    model = keras.models.load_model(model_file)
    assert model_input_channels(model) == 1


@pytest.mark.parametrize("lr_schedule", [True, False])
def test_train_lr_schedule(mocker: MockerFixture, tmpdir, lr_schedule):
    tmpdir = str(tmpdir)

    train_args = [
        "cellfinder_train",
        "-y",
        training_yaml_file,
        "-o",
        tmpdir,
        "--epochs",
        EPOCHS,
        "--lr-multiplier",
        "0.3",
    ]
    if lr_schedule:
        train_args.extend(["--lr-schedule", "10", "20"])

    mocker.patch("sys.argv", train_args)
    get_model = mocker.patch(
        "cellfinder.core.train.train_yaml.get_model", autospec=True
    )

    train_run()
    # get the data sets passed to fit(). There's no clear name property of
    # the mock fit call, so use its repr
    (fit_mock,) = [
        m for m in get_model.mock_calls if repr(m).startswith("call().fit(")
    ]
    callbacks = fit_mock.kwargs["callbacks"]

    # locate the scheduler callback, if any
    from keras.callbacks import LearningRateScheduler

    callbacks = [c for c in callbacks if isinstance(c, LearningRateScheduler)]
    if lr_schedule:
        assert len(callbacks) == 1
        # the callback is a partial function with these args
        partial_callback = callbacks[0].schedule
        assert partial_callback.keywords["multiplier"] == 0.3
        assert partial_callback.keywords["epoch_list"] == [10, 20]
    else:
        assert not callbacks


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
