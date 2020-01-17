# NOT RUNNING AS SPECIFIC TO OLD CLASSIFICATION

import os
import shutil
import pytest

from cellfinder.classify import classify
import cellfinder.IO.cells as cell_io
from cellfinder.cells.cells import Cell
from cellfinder.tools.prep import prep_classification

data_dir = os.path.join(os.getcwd(), "tests", "data")
cubes_dir = os.path.join(data_dir, "classify", "cubes")

signal_channel = 1
background_channel = 2

cells_validation = [
    Cell("z222y2805x9962", 2),
    Cell("z258y3892x10559", 2),
    Cell("z413y2308x9391", 2),
    Cell("z416y2503x5997", 1),
    Cell("z418y5457x9489", 1),
    Cell("z433y4425x7552", 1),
]


class ClassifyArgs:
    def __init__(
        self,
        tmpdir,
        cubes_output_dir,
        cell_classification_file="cell_classification.xml",
        debug=False,
    ):
        self.save_csv = True
        self.debug = debug
        self.output_dir = tmpdir
        self.empty = None
        self.signal_channel = signal_channel
        self.background_ch_id = background_channel
        self.classification_out_file = os.path.join(
            self.output_dir, cell_classification_file
        )
        self.trained_model = None
        self.model_weights = None
        self.batch_size = 32
        self.n_free_cpus = 1
        self.network_depth = "50"
        self.paths = Paths(
            cubes_output_dir=cubes_output_dir,
            classification_out_file=os.path.join(
                self.output_dir, cell_classification_file
            ),
        )


class Paths:
    def __init__(self, cubes_output_dir=None, classification_out_file=None):

        self.tmp__cubes_output_dir = cubes_output_dir
        self.classification_out_file = classification_out_file


@pytest.mark.slow
def test_classify(tmpdir):
    tmpdir = str(tmpdir)
    tmpdir = os.path.join(os.getcwd(), tmpdir)
    new_cubes_dir = os.path.join(tmpdir, "cubes")

    # copying so we can test the removal of intermediate files with debug set
    # to false. Otherwise will delete cubes in tests/data
    shutil.copytree(cubes_dir, new_cubes_dir)

    args = ClassifyArgs(tmpdir, new_cubes_dir)
    args = prep_classification(args)

    classify.main(args)

    cells_test = cell_io.get_cells(args.paths.classification_out_file)

    cells_test.sort(key=lambda x: x.z)
    cells_validation.sort(key=lambda x: x.z)

    assert cells_validation == cells_test
