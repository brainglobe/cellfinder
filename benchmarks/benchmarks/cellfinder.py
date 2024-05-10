import json
import os
import shutil
from pathlib import Path

from brainglobe_utils.IO.cells import save_cells

from cellfinder.core.main import main as cellfinder_run
from cellfinder.core.tools.IO import read_with_dask
from cellfinder.core.tools.prep import prep_models
from workflows.cellfinder import (
    CellfinderConfig,
    run_workflow_from_cellfinder_run,
)
from workflows.cellfinder import setup as setup_cellfinder_workflow
from workflows.utils import DEFAULT_JSON_CONFIG_PATH_CELLFINDER


class TimeBenchmark:
    """

    A base class for timing benchmarks for the cellfinder workflow.

    It includes:
     - a setup_cache function that downloads the GIN data specified in the
       default_config.json to a local directory (created by asv). This function
       runs only once before all repeats of the benchmark.
    -  a setup function, that runs the setup steps for the workflow.
    - a teardown function, that removes the output directory.

    Notes
    -----
    The class includes some predefined attributes for timing benchmarks. For
    the full list see
    https://asv.readthedocs.io/en/stable/benchmarks.html#benchmark-attributes

    Some asv benchmarking nomenclature:
    - repeat: a benchmark repeat is made up of the following steps:
      1- the `setup` is run,
      2- then the timed benchmark routine is called for `n` iterations, and
      3- finally that teardown function is run.
      Each repeat generates a sample, which is the average time that the
      routine took across all iterations. A new process is started for each
      repeat of each benchmark. A calibration phase before running the repeat
      computes the number of iterations that will be executed. Each benchmark
      is run for a number of repeats. The setup_cache function is run only once
      for all repeats of a benchmark (but it is discarded before the next
      benchmark). By default `repeat` is set to 0, which means:
        - if rounds==1 the default is
            (min_repeat, max_repeat, max_time) = (1, 10, 20.0),
        - if rounds != 1 the default is
            (min_repeat, max_repeat, max_time) = (1, 5, 10.0)

    - iterations (`number`): the number of iterations in each sample. Note that
      `setup` and `teardown` are not run between iterations. asv will
      automatically select the number of iterations so that each sample takes
      approximately `sample_time` seconds.

    - round: at each round, each benchmark is run for the specified number of
      repeats. The idea is that we sample each benchmark over longer periods of
      background performance variations.

    - warmup time: asv will spend this time (in seconds) in calling the
      benchmarked function repeatedly, before starting to run the actual
      benchmark. If not specified, warmup_time defaults to 0.1 seconds

    """

    # Timing attributes
    timeout = 600  # default: 60 s
    version = (
        None  # benchmark version. Default:None (i.e. hash of source code)
    )
    warmup_time = 0.1  # seconds
    rounds = 2
    repeat = 0
    sample_time = 0.01  # default: 10 ms = 0.01 s;
    min_run_count = 2  # default:2

    # Input config file
    # use environment variable CONFIG_PATH if exists, otherwise use default
    input_config_path = os.getenv(
        "CELLFINDER_CONFIG_PATH",
        default=str(DEFAULT_JSON_CONFIG_PATH_CELLFINDER),
    )

    def setup_cache(self):
        """
        Download the input data from the GIN repository to the local
        directory specified in the default_config.json.

        Notes
        -----
        The `setup_cache` method only performs the computations once
        per benchmark round and then caches the result to disk [1]_. It cannot
        be parametrised [2]_. Therefore, if we sweep across different input
        JSON files, we need to ensure all data for all configs is made
        available with this setup function.


        [1] https://asv.readthedocs.io/en/latest/writing_benchmarks.html#setup-and-teardown-functions
        [2] https://asv.readthedocs.io/en/latest/writing_benchmarks.html#parameterized-benchmarks
        """

        # Check config file exists
        assert Path(self.input_config_path).exists()

        # Instantiate a CellfinderConfig from the input json file
        # (fetches data from GIN if required)
        with open(self.input_config_path) as cfg:
            config_dict = json.load(cfg)
        config = CellfinderConfig(**config_dict)

        # Check paths to input data exist in config now
        assert Path(config._signal_dir_path).exists()
        assert Path(config._background_dir_path).exists()

        # Ensure cellfinder model is downloaded to default path
        _ = prep_models(
            model_weights_path=config.model_weights,
            install_path=None,  # Use default,
            model_name=config.model,
        )

    def setup(self):
        """
        Run the cellfinder workflow setup steps.

        The command line input arguments are injected as dependencies.
        """

        # Run setup
        cfg = setup_cellfinder_workflow(self.input_config_path)

        # Save configuration as attribute
        self.cfg = cfg

    def teardown(self):
        """
        Remove the cellfinder output directory.

        The input data is kept for all repeats of the same benchmark,
        to avoid repeated downloads from GIN.
        """
        shutil.rmtree(Path(self.cfg._output_path).resolve())


class TimeFullWorkflow(TimeBenchmark):
    """
    Time the full cellfinder workflow.

    It includes reading the signal and background arrays with dask,
    detecting the cells and saving the results to an XML file

    Parameters
    ----------
    TimeBenchmark : _type_
        A base class for timing benchmarks for the cellfinder workflow.
    """

    def time_workflow(self):
        run_workflow_from_cellfinder_run(self.cfg)


class TimeReadInputDask(TimeBenchmark):
    """
    Time the reading input data operations with dask

    Parameters
    ----------
    TimeBenchmark : _type_
        A base class for timing benchmarks for the cellfinder workflow.
    """

    def time_read_signal_with_dask(self):
        read_with_dask(str(self.cfg._signal_dir_path))

    def time_read_background_with_dask(self):
        read_with_dask(str(self.cfg._background_dir_path))


class TimeDetectAndClassifyCells(TimeBenchmark):
    """
    Time the cell detection main pipeline (`cellfinder_run`)

    Parameters
    ----------
    TimeBenchmark : _type_
        A base class for timing benchmarks for the cellfinder workflow.
    """

    # extend basic setup function
    def setup(self):
        # basic setup
        TimeBenchmark.setup(self)

        # add input data as arrays to the config
        self.signal_array = read_with_dask(str(self.cfg._signal_dir_path))
        self.background_array = read_with_dask(
            str(self.cfg._background_dir_path)
        )

    def time_cellfinder_run(self):
        cellfinder_run(
            self.signal_array,
            self.background_array,
            self.cfg.voxel_sizes,
            self.cfg.start_plane,
            self.cfg.end_plane,
            self.cfg.trained_model,
            self.cfg.model_weights,
            self.cfg.model,
            self.cfg.batch_size,
            self.cfg.n_free_cpus,
            self.cfg.network_voxel_sizes,
            self.cfg.soma_diameter,
            self.cfg.ball_xy_size,
            self.cfg.ball_z_size,
            self.cfg.ball_overlap_fraction,
            self.cfg.log_sigma_size,
            self.cfg.n_sds_above_mean_thresh,
            self.cfg.soma_spread_factor,
            self.cfg.max_cluster_size,
            self.cfg.cube_width,
            self.cfg.cube_height,
            self.cfg.cube_depth,
            self.cfg.network_depth,
        )


class TimeSaveCells(TimeBenchmark):
    # extend basic setup function
    def setup(self):
        # basic setup
        TimeBenchmark.setup(self)

        # add input data as arrays to config
        self.signal_array = read_with_dask(str(self.cfg._signal_dir_path))
        self.background_array = read_with_dask(
            str(self.cfg._background_dir_path)
        )

        # detect cells
        self.detected_cells = cellfinder_run(
            self.signal_array,
            self.background_array,
            self.cfg.voxel_sizes,
            self.cfg.start_plane,
            self.cfg.end_plane,
            self.cfg.trained_model,
            self.cfg.model_weights,
            self.cfg.model,
            self.cfg.batch_size,
            self.cfg.n_free_cpus,
            self.cfg.network_voxel_sizes,
            self.cfg.soma_diameter,
            self.cfg.ball_xy_size,
            self.cfg.ball_z_size,
            self.cfg.ball_overlap_fraction,
            self.cfg.log_sigma_size,
            self.cfg.n_sds_above_mean_thresh,
            self.cfg.soma_spread_factor,
            self.cfg.max_cluster_size,
            self.cfg.cube_width,
            self.cfg.cube_height,
            self.cfg.cube_depth,
            self.cfg.network_depth,
        )

    def time_save_cells(self):
        save_cells(self.detected_cells, self.cfg._detected_cells_path)
