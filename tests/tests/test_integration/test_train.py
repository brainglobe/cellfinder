import os
import sys
import pytest

from cellfinder_core.train.train_yml import cli as train_run

data_dir = os.path.join(
    os.getcwd(), "tests", "data", "integration", "training"
)
cell_cubes = os.path.join(data_dir, "cells")
non_cell_cubes = os.path.join(data_dir, "non_cells")
training_yml_file = os.path.join(data_dir, "training.yml")


EPOCHS = "2"

# only checks that the model is trained, and then saved.
# doesn't check that it works etc


@pytest.mark.slow
def test_train(tmpdir):
    tmpdir = str(tmpdir)

    train_args = [
        "cellfinder_train",
        "-y",
        training_yml_file,
        "-o",
        tmpdir,
        "--epochs",
        EPOCHS,
    ]
    sys.argv = train_args
    train_run()

    model_file = os.path.join(tmpdir, "model.h5")
    assert os.path.exists(model_file)
