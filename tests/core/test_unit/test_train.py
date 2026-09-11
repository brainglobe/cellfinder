import os

import pytest
import tifffile
import torch
from brainglobe_utils.IO.yaml import save_yaml

from cellfinder.core.train.train_yaml import (
    CUBE_HEIGHT,
    CUBE_WIDTH,
    _squeeze_z_collate,
    get_dataloader,
    get_tiff_files,
    make_tiff_lists,
    parse_yaml,
    run,
)

data_dir = os.path.join(
    os.getcwd(), "tests", "data", "integration", "training"
)
cell_cubes = os.path.join(data_dir, "cells")
training_yaml_file = os.path.join(data_dir, "training.yaml")


def _cells_and_filenames(yaml_file):
    tiff_files = get_tiff_files(parse_yaml([yaml_file]))
    filenames, cells = make_tiff_lists(tiff_files)
    return cells, filenames


@pytest.fixture
def yaml_2d(tmp_path):
    """A yaml pointing at depth-1 cubes, as 2D curation produces."""
    cubes_2d = tmp_path / "cells"
    cubes_2d.mkdir()
    for fname in os.listdir(cell_cubes):
        if fname.endswith(".tif"):
            cube = tifffile.imread(os.path.join(cell_cubes, fname))
            tifffile.imwrite(cubes_2d / fname, cube[10:11])

    yaml_file = tmp_path / "training_2d.yaml"
    save_yaml(
        {
            "data": [
                {
                    "bg_channel": 1,
                    "cell_def": "",
                    "cube_dir": str(cubes_2d),
                    "signal_channel": 0,
                    "type": "cell",
                },
            ]
        },
        yaml_file,
    )
    return yaml_file


def _loader(cells, filenames, **kwargs):
    return get_dataloader(
        cells,
        filenames,
        batch_size=2,
        n_processes=0,
        pin_memory=False,
        auto_shuffle=False,
        augment=False,
        augment_likelihood=0.0,
        normalize_channels=False,
        **kwargs,
    )


def test_squeeze_z_collate_drops_the_z_axis():
    data = torch.empty((2, 1, 50, 50, 2))
    squeezed, label = _squeeze_z_collate((data, "label"), z_axis=1)

    assert squeezed.shape == (2, 50, 50, 2)
    assert label == "label"


def test_2d_dataloader_yields_planes(yaml_2d):
    cells, filenames = _cells_and_filenames(yaml_2d)
    loader, dataset = _loader(cells, filenames, dimensions=2)

    assert dataset.network_cuboid_voxels[0] == 1

    data, _ = next(iter(loader))
    # the depth-1 z axis is squeezed away, leaving (batch, y, x, channels)
    assert data.ndim == 4


def test_3d_dataloader_yields_cubes():
    cells, filenames = _cells_and_filenames(training_yaml_file)
    loader, dataset = _loader(cells, filenames)

    assert dataset.network_cuboid_voxels[0] > 1

    data, _ = next(iter(loader))
    assert data.ndim == 5


@pytest.mark.parametrize(
    "dimensions,expected_shape",
    [(2, (CUBE_HEIGHT, CUBE_WIDTH, 2)), (3, None)],
)
def test_model_shape_follows_dimensions(
    yaml_2d, tmp_path, mocker, dimensions, expected_shape
):
    """Only 2D passes an explicit in-plane shape to the model builder."""
    get_model = mocker.patch(
        "cellfinder.core.train.train_yaml.get_model", autospec=True
    )
    mocker.patch(
        "cellfinder.core.train.train_yaml.prep_model_weights",
        return_value=None,
    )

    run(
        output_dir=tmp_path / "out",
        yaml_file=[yaml_2d],
        dimensions=dimensions,
        epochs=0,
        test_fraction=0,
        no_augment=True,
        max_workers=1,
        pin_memory=False,
    )

    assert get_model.call_args.kwargs["shape"] == expected_shape
    assert get_model.call_args.kwargs["dimensions"] == dimensions
