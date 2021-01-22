import pytest
import os

from pathlib import Path
from math import isclose
from imlib.general.system import ensure_directory_exists

import cellfinder_core.tools.system as system
from imlib.general.exceptions import CommandLineInputError

data_dir = Path("tests", "data")
background_im_dir = os.path.join(data_dir, "background")


def test_get_subdirectories():
    subdirs = system.get_subdirectories(data_dir)
    assert len(subdirs) == 9
    assert Path(data_dir / "cells") in subdirs
    assert Path(data_dir / "brain") in subdirs

    subdir_names = system.get_subdirectories(data_dir, names_only=True)
    assert len(subdir_names) == 9
    assert "cells" in subdir_names
    assert "brain" in subdir_names


def test_get_number_of_files_in_dir():
    assert system.get_number_of_files_in_dir(background_im_dir) == 26


def write_file_single_size(directory, file_size):
    with open(os.path.join(directory, str(file_size)), "wb") as fout:
        fout.write(os.urandom(file_size))


def test_check_path_exists(tmpdir):
    num = 10
    tmpdir = str(tmpdir)

    assert system.check_path_exists(os.path.join(tmpdir))
    no_exist_dir = os.path.join(tmpdir, "i_dont_exist")
    with pytest.raises(FileNotFoundError):
        assert system.check_path_exists(no_exist_dir)

    write_file_single_size(tmpdir, num)
    assert system.check_path_exists(os.path.join(tmpdir, str(num)))
    with pytest.raises(FileNotFoundError):
        assert system.check_path_exists(os.path.join(tmpdir, "20"))


def test_catch_input_file_error(tmpdir):
    tmpdir = str(tmpdir)
    # check no error is raised:
    system.catch_input_file_error(tmpdir)

    no_exist_dir = os.path.join(tmpdir, "i_dont_exist")
    with pytest.raises(CommandLineInputError):
        system.catch_input_file_error(no_exist_dir)


def test_ensure_directory_exists():
    home = os.path.expanduser("~")
    exist_dir = os.path.join(home, ".cellfinder_test_dir")
    ensure_directory_exists(exist_dir)
    assert os.path.exists(exist_dir)
    os.rmdir(exist_dir)


def test_memory_in_bytes():
    memory_detection_tolerance = 1  # byte

    assert isclose(
        system.memory_in_bytes(1, "kb"),
        1000,
        abs_tol=memory_detection_tolerance,
    )
    assert isclose(
        system.memory_in_bytes(1.2, "MB"),
        1200000,
        abs_tol=memory_detection_tolerance,
    )
    assert isclose(
        system.memory_in_bytes(0.00065, "gb"),
        650000,
        abs_tol=memory_detection_tolerance,
    )
    assert isclose(
        system.memory_in_bytes(0.000000000234, "TB"),
        234,
        abs_tol=memory_detection_tolerance,
    )
    assert isclose(
        system.memory_in_bytes(1000, "pb"),
        10 ** 18,
        abs_tol=memory_detection_tolerance,
    )

    with pytest.raises(NotImplementedError):
        system.memory_in_bytes(1000, "ab")
