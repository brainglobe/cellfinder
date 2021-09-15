import dask.array as d_array

from cellfinder_core.tools import IO

BRAIN_DIR = "tests/data/brain"
BRAIN_PATHS = f"{BRAIN_DIR}/brain_paths.txt"


def test_read_with_dask_txt():
    stack = IO.read_with_dask(BRAIN_PATHS)
    assert type(stack) == d_array.Array


def test_read_with_dask_glob_txt_equal():
    txt_stack = IO.read_with_dask(BRAIN_PATHS)
    glob_stack = IO.read_with_dask(BRAIN_DIR)

    assert d_array.equal(txt_stack, glob_stack).all()
