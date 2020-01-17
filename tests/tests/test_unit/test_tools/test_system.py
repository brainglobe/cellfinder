import pytest
import random
import os

from pathlib import Path
from math import isclose
from random import shuffle

import cellfinder.tools.tools as tools
import cellfinder.tools.system as system
from cellfinder.tools.exceptions import CommandLineInputError

data_dir = Path("tests", "data")
cubes_dir = data_dir / "classify" / "cubes"
jabberwocky = data_dir / "general" / "jabberwocky.txt"
jabberwocky_sorted = data_dir / "general" / "jabberwocky_sorted.txt"
background_im_dir = os.path.join(data_dir, "background")

cubes = [
    "pCellz222y2805x9962Ch1.tif",
    "pCellz222y2805x9962Ch2.tif",
    "pCellz258y3892x10559Ch1.tif",
    "pCellz258y3892x10559Ch2.tif",
    "pCellz413y2308x9391Ch1.tif",
    "pCellz413y2308x9391Ch2.tif",
    "pCellz416y2503x5997Ch1.tif",
    "pCellz416y2503x5997Ch2.tif",
    "pCellz418y5457x9489Ch1.tif",
    "pCellz418y5457x9489Ch2.tif",
    "pCellz433y4425x7552Ch1.tif",
    "pCellz433y4425x7552Ch2.tif",
]


sorted_cubes_dir = [os.path.join(str(cubes_dir), cube) for cube in cubes]


def write_n_random_files(n, dir, min_size=32, max_size=2048):
    sizes = random.sample(range(min_size, max_size), n)
    for size in sizes:
        with open(os.path.join(dir, str(size)), "wb") as fout:
            fout.write(os.urandom(size))


def test_delete_directory_contents(tmpdir):
    delete_dir = os.path.join(str(tmpdir), "delete_dir")
    os.mkdir(delete_dir)
    write_n_random_files(10, delete_dir)

    # check the directory isn't empty first
    assert not os.listdir(delete_dir) == []

    system.delete_directory_contents(delete_dir, progress=True)
    assert os.listdir(delete_dir) == []


def test_get_sorted_file_paths():
    # test list
    shuffled = sorted_cubes_dir.copy()
    shuffle(shuffled)
    assert system.get_sorted_file_paths(shuffled) == sorted_cubes_dir

    # test dir
    assert system.get_sorted_file_paths(cubes_dir) == sorted_cubes_dir
    assert (
        system.get_sorted_file_paths(cubes_dir, file_extension=".tif")
        == sorted_cubes_dir
    )

    # test text file
    # specifying utf8, as written on linux
    assert system.get_sorted_file_paths(
        jabberwocky, encoding="utf8"
    ) == tools.get_text_lines(jabberwocky_sorted, encoding="utf8")

    # test unsupported
    with pytest.raises(NotImplementedError):
        system.get_sorted_file_paths(shuffled[0])


def test_get_subdirectories():
    subdirs = system.get_subdirectories(data_dir)
    assert len(subdirs) == 11
    assert Path(data_dir / "general") in subdirs
    assert Path(data_dir / "IO") in subdirs

    subdir_names = system.get_subdirectories(data_dir, names_only=True)
    assert len(subdir_names) == 11
    assert "general" in subdir_names
    assert "IO" in subdir_names


def test_get_number_of_files_in_dir():
    assert system.get_number_of_files_in_dir(background_im_dir) == 26


def test_check_path_in_dir():
    assert system.check_path_in_dir(jabberwocky, data_dir / "general")


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
    system.ensure_directory_exists(exist_dir)
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


def test_replace_extension():
    test_file = "test_file.sh"
    test_ext = "txt"
    test_ext_w_dot = ".txt"
    validate_file = "test_file.txt"
    assert validate_file == system.replace_extension(test_file, test_ext)
    assert validate_file == system.replace_extension(test_file, test_ext_w_dot)


def test_remove_leading_character():
    assert ".ext" == system.remove_leading_character("..ext", ".")


def check_get_num_processes():
    assert len(os.sched_getaffinity(0)) == system.get_num_processes()


def check_max_processes():
    max_proc = 5
    correct_n = min(len(os.sched_getaffinity(0)), max_proc)
    assert correct_n == system.get_num_processes(n_max_processes=max_proc)


def check_slurm_n_processes():
    rand_n = random.randint(1, 100)
    rand_min = random.randint(1, 30)
    os.environ["SLURM_NPROCS"] = str(rand_n)
    correct_n_procs = rand_n - rand_min
    assert correct_n_procs == system.get_num_processes(
        min_free_cpu_cores=rand_min
    )
