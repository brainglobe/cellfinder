import os

import pytest
from brainglobe_utils.general.exceptions import CommandLineInputError

from cellfinder.tools import prep

# import shutil

# from pathlib import Path


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
        no_register=False,
        no_analyse=False,
        no_standard_space=False,
        output_dir=None,
        registration_output_folder=None,
        cells_file_path=None,
        cubes_output_dir=None,
        classification_out_file=None,
        cells_in_standard_space=None,
    ):
        self.cell_count_model_dir = model_dir
        self.empty = empty

        self.no_detection = no_detection
        self.no_classification = no_classification
        self.no_register = no_register
        self.no_summarise = no_analyse
        self.no_standard_space = no_standard_space

        self.output_dir = output_dir

        self.paths = Paths(
            output_dir=registration_output_folder,
            cells_file_path=cells_file_path,
            cubes_output_dir=cubes_output_dir,
            classification_out_file=classification_out_file,
            cells_in_standard_space=cells_in_standard_space,
        )


class Paths:
    def __init__(
        self,
        output_dir=None,
        cells_file_path=None,
        cubes_output_dir=None,
        classification_out_file=None,
        cells_in_standard_space=None,
    ):
        self.registration_output_folder = output_dir
        self.cells_file_path = cells_file_path
        self.tmp__cubes_output_dir = cubes_output_dir
        self.classification_out_file = classification_out_file
        self.cells_in_standard_space = cells_in_standard_space


# def get_dict_of_what_to_run(what_to_run):
#     what_to_run_dict = {
#         "detect": what_to_run.detect,
#         "classify": what_to_run.classify,
#         "register": what_to_run.register,
#         "figures": what_to_run.figures,
#     }
#     return what_to_run_dict
#
#
# def test_calc_what_to_run(tmpdir):
#     tmpdir = str(tmpdir)
#
#     # registered_atlas_path = os.path.join(tmpdir, "registered_atlas.tiff")
#     cells_file_path = os.path.join(tmpdir, "cells.xml")
#     classification_out_file_path = os.path.join(
#         tmpdir, "cell_classification.xml"
#     )
#     summary_cell_counts_path = os.path.join(
#     tmpdir, "summary_cell_counts.csv")
#     figures_dir = os.path.join(tmpdir, "figures")
#
#     args = Args(
#         output_dir=tmpdir,
#         registration_output_folder=tmpdir,
#         cells_file_path=cells_file_path,
#         classification_out_file=classification_out_file_path,
#         figures_dir=figures_dir,
#     )
#
#     # default
#     what_to_run = prep.CalcWhatToRun(args)
#     default_validation = {
#         "detect": True,
#         "classify": True,
#         "register": False,
#         "figures": False,
#     }
#     default_test = get_dict_of_what_to_run(what_to_run)
#     assert default_validation == default_test
#
#     # options
#     args.no_classification = True
#     what_to_run = prep.CalcWhatToRun(args)
#     validation = {
#         "detect": True,
#         "classify": False,
#         "register": False,
#         "figures": False,
#     }
#     test = get_dict_of_what_to_run(what_to_run)
#     assert validation == test
#
#
#     # # existance
#     # Path(registered_atlas_path).touch()
#     # what_to_run = prep.CalcWhatToRun(args)
#     # validation = {
#     #     "detect": True,
#     #     "classify": True,
#     #     "register": False,
#     #     "figures": True,
#     # }
#     # test = get_dict_of_what_to_run(what_to_run)
#     # assert validation == test
#
#     Path(cells_file_path).touch()
#     what_to_run.update(args)
#     validation = {
#         "detect": False,
#         "classify": True,
#         "register": False,
#         "figures": True,
#     }
#     test = get_dict_of_what_to_run(what_to_run)
#     assert validation == test
#
#     os.remove(cells_file_path)
#     what_to_run.update(args)
#     validation = {
#         "detect": True,
#         "classify": True,
#         "register": False,
#         "figures": True,
#     }
#     test = get_dict_of_what_to_run(what_to_run)
#     assert validation == test
#
#     Path(classification_out_file_path).touch()
#     what_to_run.update(args)
#     validation = {
#         "detect": False,
#         "classify": False,
#         "register": False,
#         "figures": True,
#     }
#     test = get_dict_of_what_to_run(what_to_run)
#     assert validation == test
#
#     Path(summary_cell_counts_path).touch()
#     what_to_run.update(args)
#     validation = {
#         "detect": False,
#         "classify": False,
#         "register": False,
#         "summarise": False,
#         "figures": True,
#     }
#     test = get_dict_of_what_to_run(what_to_run)
#     assert validation == test
#
#     make_and_fill_directory(figures_dir)
#     args.figures = True
#     what_to_run.update(args)
#     validation = {
#         "detect": False,
#         "classify": False,
#         "register": False,
#         "summarise": False,
#         "figures": False,
#     }
#     test = get_dict_of_what_to_run(what_to_run)
#     assert validation == test
#
#
#     # os.remove(registered_atlas_path)
#     os.remove(summary_cell_counts_path)
#     shutil.rmtree(figures_dir)
#     what_to_run.update(args)
#     validation = {
#         "detect": False,
#         "classify": False,
#         "register": True,
#         "summarise": True,
#         "figures": True,
#     }
#     test = get_dict_of_what_to_run(what_to_run)
#     assert validation == test


def make_and_fill_directory(directory):
    os.mkdir(directory)
    for file_size in range(100, 200, 20):
        with open(os.path.join(directory, str(file_size)), "wb") as fout:
            fout.write(os.urandom(file_size))
