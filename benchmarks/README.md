# README

## Overview
We use `asv` to benchmark some representative brainglobe workflows. The `asv` workflow is roughly as follows:
1. `asv` creates a virtual environment (as defined in the `asv.conf.json` file).
1. It installs the version of the `cellfinder` software package corresponding to the tip of the locally checked-out branch.
1. It runs the benchmarks defined (locally) under `cellfinder/benchmarks/benchmarks` and saves the results to `cellfinder/benchmarks/results` as json files.
1. With `asv publish`, the output json files are 'published' into an html directory (`cellfinder/benchmarks/html`)
1. With `asv preview` the html directory can be visualised in a static site.


There are three main ways in which these benchmarks can be useful to developers:
1. Developers can run the available benchmarks locally [on a small test dataset](#running-benchmarks-locally-on-default-small-dataset).
1. Developers can run these benchmarks on [data they have stored locally](#running-benchmarks-locally-on-custom-data).
1. We also plan to run the benchmarks internally on a large dataset, and make the results publicly available.

## Installation
To [install asv](https://asv.readthedocs.io/en/stable/installing.html):
```
pip install asv
```
Note that to run the benchmarks you only need to have `asv` in your virtual environment. You _do not_ need to install a development version of `cellfinder`, since `asv` will create a separate Python virtual environment to run the benchmarks on it. However, for convenience we include `asv` as part of the `[asv]` dependencies, so an environment with `cellfinder[asv]` can be used to run the benchmarks.


## Running benchmarks on a default small dataset


1. Git clone the `cellfinder` repository
    ```
    git clone https://github.com/brainglobe/cellfinder.git
    ```
2. Install the `asv` version of the `cellfinder` package:
    ```
    pip install .[asv]
    ```
    This is mostly for convenience: the `[asv]` specification includes `asv` as a dependency, but to run the benchmarks it would be sufficient to use an environment with `asv` only.
3. Launch the benchmarks, by running from the directory where `asv.conf.json` is at:
    ```
    asv run
    ```
    This will run the local benchmarks, with the default config at `cellfinder/configs/cellfinder.json`, on a small dataset downloaded from [GIN](https://gin.g-node.org/G-Node/info/wiki).

    See the [asv docs](https://asv.readthedocs.io/en/v0.6.1/using.html#running-benchmarks) for further details on the `asv run` command and others.


## Running benchmarks on custom data available locally

1. Git clone the `cellfinder` repository
    ```
    git clone https://github.com/brainglobe/cellfinder.git
    ```
1. Define a config file for the cellfinder workflow to benchmark.
    - You can use the default one at `cellfinder/configs/cellfinder.json` as reference.
    - Ensure your config file includes an `input_data_dir` field pointing to the data of interest.
    - The signal and background data are assumed to be in `signal` and `background` directories, under the `input_data_dir` directory. If they are under directories with a different name, you can specify their names with the `signal_subdir` and `background_subdir` fields.

1. Run the benchmarks, passing the path to your config file as an environment variable `CELLFINDER_CONFIG_PATH`. In Unix systems:
    ```
    CELLFINDER_CONFIG_PATH=/path/to/your/config/file asv run
    ```

## Running benchmarks in development
The following flags to `asv run` are often useful in development:
- `--quick`: will only run one repetition per benchmark, and no results to disk.
- `--verbose`: provides further info on intermediate steps.
- `--show-stderr`: will print out stderr.
- `--dry-run`: will not write results to disk.
- `--bench`: to specify a subset of benchmarks (e.g., `tools.prep.PrepTF`). Regexp can be used.
- `--python=same`: runs the benchmarks in the same environment that `asv` was launched from

Example:
```
asv run --bench TimeFullWorkflow --dry-run --show-stderr --quick
```
