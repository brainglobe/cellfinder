import glob
import os

from typing import Tuple

import numpy as np
from dask import array as da
from dask import delayed
from tifffile import TiffFile, imread

from imlib.general.system import get_sorted_file_paths


def get_tiff_meta(
    path: str,
) -> Tuple[Tuple[int, int], np.dtype]:
    with TiffFile(path) as tfile:
        nz = len(tfile.pages)
        if not nz:
            raise ValueError(f"tiff file {path} has no pages!")
        first_page = tfile.pages[0]

    return tfile.pages[0].shape, first_page.dtype


lazy_imread = delayed(imread)  # lazy reader


def read_with_dask(path):
    """
    Based on https://github.com/tlambert03/napari-ndtiffs
    :param path:
    :return:
    """

    filenames = glob.glob(os.path.join(path, "*.tif"))

    shape, dtype = get_tiff_meta(filenames[0])
    lazy_arrays = [lazy_imread(fn) for fn in get_sorted_file_paths(filenames)]
    dask_arrays = [
        da.from_delayed(delayed_reader, shape=shape, dtype=dtype)
        for delayed_reader in lazy_arrays
    ]
    stack = da.stack(dask_arrays, axis=0)
    return stack
