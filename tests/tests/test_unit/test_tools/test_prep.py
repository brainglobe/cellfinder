import os
import pytest
import shutil

from pathlib import Path

from cellfinder.tools import prep
from imlib.general.exceptions import CommandLineInputError


data_dir = os.path.join("tests", "data")


def test_check_return_ch_ids():
    signal_ch = [0, 1, 3]
    bg_ch = 6
    signal_list = ["file1.txt", "file_2.txt", "file_3.txt"]
    # None given
    assert ([0, 1, 2], 3) == prep.check_and_return_ch_ids(
        None, None, signal_list
    )
    # Only signal given
    assert (signal_ch, 4) == prep.check_and_return_ch_ids(
        signal_ch, None, signal_list
    )

    # Only background given
    assert ([7, 8, 9], bg_ch) == prep.check_and_return_ch_ids(
        None, bg_ch, signal_list
    )

    # Both given (no overlap)
    assert (signal_ch, bg_ch) == prep.check_and_return_ch_ids(
        signal_ch, bg_ch, signal_list
    )

    # Both given (overlap)
    with pytest.raises(CommandLineInputError):
        assert prep.check_and_return_ch_ids(signal_ch, 3, signal_list)


class Args:
    def __init__(
        self,
        model_dir=None,
        empty=None,
        no_detection=False,
        no_classification=False,
        register=False,
        summarise=False,
        no_standard_space=False,
        figures=False,
        output_dir=None,
        registration_output_folder=None,
        cells_file_path=None,
        cubes_output_dir=None,
        classification_out_file=None,
        cells_in_standard_space=None,
        figures_dir=None,
    ):
        self.cell_count_model_dir = model_dir
        self.empty = empty

        self.no_detection = no_detection
        self.no_classification = no_classification
        self.register = register
        self.summarise = summarise
        self.no_standard_space = no_standard_space
        self.figures = figures

        self.output_dir = output_dir

        self.paths = Paths(
            output_dir=registration_output_folder,
            cells_file_path=cells_file_path,
            cubes_output_dir=cubes_output_dir,
            classification_out_file=classification_out_file,
            cells_in_standard_space=cells_in_standard_space,
            figures_dir=figures_dir,
        )


class Paths:
    def __init__(
        self,
        output_dir=None,
        cells_file_path=None,
        cubes_output_dir=None,
        classification_out_file=None,
        cells_in_standard_space=None,
        figures_dir=None,
    ):
        self.registration_output_folder = output_dir
        self.cells_file_path = cells_file_path
        self.tmp__cubes_output_dir = cubes_output_dir
        self.classification_out_file = classification_out_file
        self.cells_in_standard_space = cells_in_standard_space
        self.figures_dir = figures_dir


def get_dict_of_what_to_run(what_to_run):
    what_to_run_dict = {
        "detect": what_to_run.detect,
        "classify": what_to_run.classify,
        "register": what_to_run.register,
        "summarise": what_to_run.summarise,
        "standard_space": what_to_run.standard_space,
        "figures": what_to_run.figures,
    }
    return what_to_run_dict


def test_calc_what_to_run(tmpdir):
    tmpdir = str(tmpdir)

    registered_atlas_path = os.path.join(tmpdir, "registered_atlas.nii")
    cells_file_path = os.path.join(tmpdir, "cells.xml")
    classification_out_file_path = os.path.join(
        tmpdir, "cell_classification.xml"
    )
    cells_in_standard_space_file_path = os.path.join(
        tmpdir, "cells_in_standard_space.xml"
    )
    summary_cell_counts_path = os.path.join(tmpdir, "summary_cell_counts.csv")
    figures_dir = os.path.join(tmpdir, "figures")

    args = Args(
        output_dir=tmpdir,
        registration_output_folder=tmpdir,
        cells_file_path=cells_file_path,
        classification_out_file=classification_out_file_path,
        figures_dir=figures_dir,
        cells_in_standard_space=cells_in_standard_space_file_path,
    )

    # default
    what_to_run = prep.CalcWhatToRun(args)
    default_validation = {
        "detect": True,
        "classify": True,
        "register": False,
        "summarise": False,
        "standard_space": False,
        "figures": False,
    }
    default_test = get_dict_of_what_to_run(what_to_run)
    assert default_validation == default_test

    # options
    args.no_classification = True
    what_to_run = prep.CalcWhatToRun(args)
    validation = {
        "detect": True,
        "classify": False,
        "register": False,
        "summarise": False,
        "standard_space": False,
        "figures": False,
    }
    test = get_dict_of_what_to_run(what_to_run)
    assert validation == test

    # hierarchy
    args.summarise = True
    args.figures = True
    what_to_run = prep.CalcWhatToRun(args)
    validation = {
        "detect": True,
        "classify": True,
        "register": True,
        "summarise": True,
        "standard_space": True,
        "figures": True,
    }
    test = get_dict_of_what_to_run(what_to_run)
    assert validation == test

    # existance
    Path(registered_atlas_path).touch()
    what_to_run = prep.CalcWhatToRun(args)
    validation = {
        "detect": True,
        "classify": True,
        "register": False,
        "summarise": True,
        "standard_space": True,
        "figures": True,
    }
    test = get_dict_of_what_to_run(what_to_run)
    assert validation == test

    Path(cells_file_path).touch()
    what_to_run.update(args)
    validation = {
        "detect": False,
        "classify": True,
        "register": False,
        "summarise": True,
        "standard_space": True,
        "figures": True,
    }
    test = get_dict_of_what_to_run(what_to_run)
    assert validation == test

    os.remove(cells_file_path)
    what_to_run.update(args)
    validation = {
        "detect": True,
        "classify": True,
        "register": False,
        "summarise": True,
        "standard_space": True,
        "figures": True,
    }
    test = get_dict_of_what_to_run(what_to_run)
    assert validation == test

    Path(classification_out_file_path).touch()
    what_to_run.update(args)
    validation = {
        "detect": False,
        "classify": False,
        "register": False,
        "summarise": True,
        "standard_space": True,
        "figures": True,
    }
    test = get_dict_of_what_to_run(what_to_run)
    assert validation == test

    Path(summary_cell_counts_path).touch()
    what_to_run.update(args)
    validation = {
        "detect": False,
        "classify": False,
        "register": False,
        "summarise": False,
        "standard_space": True,
        "figures": True,
    }
    test = get_dict_of_what_to_run(what_to_run)
    assert validation == test

    make_and_fill_directory(figures_dir)
    args.figures = True
    what_to_run.update(args)
    validation = {
        "detect": False,
        "classify": False,
        "register": False,
        "summarise": False,
        "standard_space": True,
        "figures": False,
    }
    test = get_dict_of_what_to_run(what_to_run)
    assert validation == test

    Path(cells_in_standard_space_file_path).touch()
    what_to_run.update(args)
    validation = {
        "detect": False,
        "classify": False,
        "register": False,
        "summarise": False,
        "standard_space": False,
        "figures": False,
    }
    test = get_dict_of_what_to_run(what_to_run)
    assert validation == test

    os.remove(registered_atlas_path)
    os.remove(summary_cell_counts_path)
    os.remove(cells_in_standard_space_file_path)
    shutil.rmtree(figures_dir)
    what_to_run.update(args)
    validation = {
        "detect": False,
        "classify": False,
        "register": True,
        "summarise": True,
        "standard_space": True,
        "figures": True,
    }
    test = get_dict_of_what_to_run(what_to_run)
    assert validation == test


def make_and_fill_directory(directory):
    os.mkdir(directory)
    for file_size in range(100, 200, 20):
        with open(os.path.join(directory, str(file_size)), "wb") as fout:
            fout.write(os.urandom(file_size))
