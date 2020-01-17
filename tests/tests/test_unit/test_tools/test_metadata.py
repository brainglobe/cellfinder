import pytest

from pathlib import Path
from math import isclose

import cellfinder.tools.metadata as meta
from cellfinder.tools.exceptions import CommandLineInputError

data_dir = Path("tests", "data")
metadata_dir = data_dir / "metadata"

cellfinder_metadata = metadata_dir / "cellfinder_metadata.ini"
baking_tray_metadata = metadata_dir / "BakingTray_recipe.yml"
mesoSPIM_metadata = metadata_dir / "mesoSPIM.raw_meta.txt"
unsupported_metadata = metadata_dir / "unsupported_metadata.txt"
missing_metadata = metadata_dir / "cellfinder_metadata_missing.ini"

VOX_DIM_TOLERANCE = 0.1


class Args:
    def __init__(self):
        self.x_pixel_um = self.y_pixel_um = self.z_pixel_um = None
        self.metadata = None

    def set_all_pixel_sizes(self):
        self.x_pixel_um = 1
        self.y_pixel_um = 2
        self.z_pixel_um = 3

    def set_some_pixel_sizes_w_meta_baking_tray(self):
        self.x_pixel_um = 10
        self.y_pixel_um = None
        self.z_pixel_um = 40

        self.metadata = baking_tray_metadata

    def set_some_pixel_sizes_w_meta_mesospim(self):
        self.x_pixel_um = 100
        self.y_pixel_um = None
        self.z_pixel_um = None

        self.metadata = mesoSPIM_metadata

    def set_some_pixel_sizes_w_meta_cellfinder(self):
        self.x_pixel_um = None
        self.y_pixel_um = 0.2
        self.z_pixel_um = 40

        self.metadata = cellfinder_metadata

    def set_some_pixel_sizes_no_meta(self):
        self.x_pixel_um = None
        self.y_pixel_um = 100
        self.z_pixel_um = 3
        self.metadata = None

    def set_some_pixel_sizes_unsupported_meta(self):
        self.x_pixel_um = None
        self.y_pixel_um = 100
        self.z_pixel_um = 3
        self.metadata = unsupported_metadata

    def set_some_pixel_sizes_missing_meta(self):
        self.x_pixel_um = None
        self.y_pixel_um = 100
        self.z_pixel_um = 3
        self.metadata = missing_metadata


def test_define_pixel_sizes():
    args = Args()
    args.set_all_pixel_sizes()
    args = meta.define_pixel_sizes(args)
    assert isclose(1, args.x_pixel_um, abs_tol=VOX_DIM_TOLERANCE)
    assert isclose(2, args.y_pixel_um, abs_tol=VOX_DIM_TOLERANCE)
    assert isclose(3, args.z_pixel_um, abs_tol=VOX_DIM_TOLERANCE)

    # baking tray
    args = Args()
    args.set_some_pixel_sizes_w_meta_baking_tray()
    args = meta.define_pixel_sizes(args)
    assert isclose(10, args.x_pixel_um, abs_tol=VOX_DIM_TOLERANCE)
    assert isclose(2.14, args.y_pixel_um, abs_tol=VOX_DIM_TOLERANCE)
    assert isclose(40, args.z_pixel_um, abs_tol=VOX_DIM_TOLERANCE)

    # mesospim
    args = Args()
    args.set_some_pixel_sizes_w_meta_mesospim()
    args = meta.define_pixel_sizes(args)
    assert isclose(100, args.x_pixel_um, abs_tol=VOX_DIM_TOLERANCE)
    assert isclose(8.23, args.y_pixel_um, abs_tol=VOX_DIM_TOLERANCE)
    assert isclose(10, args.z_pixel_um, abs_tol=VOX_DIM_TOLERANCE)

    # cellfinder
    args = Args()
    args.set_some_pixel_sizes_w_meta_cellfinder()
    args = meta.define_pixel_sizes(args)
    assert isclose(2, args.x_pixel_um, abs_tol=VOX_DIM_TOLERANCE)
    assert isclose(0.2, args.y_pixel_um, abs_tol=VOX_DIM_TOLERANCE)
    assert isclose(40, args.z_pixel_um, abs_tol=VOX_DIM_TOLERANCE)

    # no metadata
    args = Args()
    args.set_some_pixel_sizes_no_meta()
    with pytest.raises(CommandLineInputError):
        assert meta.define_pixel_sizes(args)

    # unsupported metadata
    args = Args()
    args.set_some_pixel_sizes_unsupported_meta()
    with pytest.raises(CommandLineInputError):
        assert meta.define_pixel_sizes(args)

    # missing metadata
    args = Args()
    args.set_some_pixel_sizes_missing_meta()
    with pytest.raises(CommandLineInputError):
        assert meta.define_pixel_sizes(args)
