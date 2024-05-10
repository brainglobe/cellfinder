# README

## Overview
We use `asv` to benchmark some representative brainglobe workflows. To [install asv](https://asv.readthedocs.io/en/stable/installing.html):
```
pip install asv
```
Note that to run the benchmarks you only need to have `asv` in your virtual environment. You do not need to install a development version of `brainglobe-workflows` (since `asv` will create a separate Python virtual environment to run the benchmarks on it). However, for convenience we include `asv` as part of the `[dev]` dependencies, so an environment with `brainglobe-workflows[dev]` can be used to run the benchmarks.


The `asv` workflow is roughly as follows:
1. `asv` creates a virtual environment (as defined in the `asv.conf.json` file).
1. It installs the version of the `brainglobe-workflows` software package corresponding to the tip of the locally checked-out branch.
1. It runs the benchmarks defined (locally) under `brainglobe-workflows/benchmarks/benchmarks` and saves the results to `brainglobe-workflows/benchmarks/results` as json files.
1. With `asv publish`, the output json files are 'published' into an html directory (`brainglobe-workflows/benchmarks/html`)
1. With `asv preview` the html directory can be visualised in a static site.


There are three main ways in which these benchmarks can be useful to developers:
1. Developers can run the available benchmarks locally on a small test dataset. See [here](#running-benchmarks-locally-on-default-small-dataset).
1. Developers can run these benchmarks on data they have stored locally. See [here](#running-benchmarks-locally-on-custom-data).
1. We also plan to run the benchmarks on an internal runner using a larger dataset, of the scale we expect users to be handling. The result of these benchmarks will be made publicly available.

## Running benchmarks locally on default small dataset

 To do so:
    - Git clone repo locally
    - Install the developer version of the package:
        ```
        pip install .[dev]
        ```
        This is mostly for convenience: the `[dev]` specification includes `asv` as a dependency, but to run the benchmarks it would be sufficient to use an environment with `asv` only. This is because `asv` creates its own virtual environment for the benchmarks, building and installing the relevant version of the `brainglobe-workflows` package in it. By default, the version at the tip of the currently checked out branch is installed.
    - Run the benchmarks, From the directory where asv config is at:
        ```
        asv run
        ```
       This will run the locally defined benchmarks with the default parameters defined at `brainglobe_workflows/configs/cellfinder.json`, on a small dataset downloaded from [GIN](https://gin.g-node.org/G-Node/info/wiki). See the [asv docs](https://asv.readthedocs.io/en/v0.6.1/using.html#running-benchmarks) for further guidance on how to run benchmarks.


## Running benchmarks locally on custom data
For data available locally.

Run commands From the directory where asv config is at.

To do so:
    - Git clone repo locally
    - Define a config file for the workflow to benchmark. You can use the default one at `brainglobe_workflows/configs/cellfinder.json` for reference.
    - Ensure your config file includes an `input_data_dir` field pointing to the data of interest.
    - Edit the names of the signal and background directories if required. By default, they are assumed to be in `signal` and `background` subdirectories under `input_data_dir`. However, these defaults can be overwritten with the `signal_subdir` and `background_subdir` fields.
    - Run the benchmarks, passing the path to your config file as an environment variable `CELLFINDER_CONFIG_PATH`. In Unix systems:
        ```
        CELLFINDER_CONFIG_PATH=/path/to/your/config/file asv run
        ```

## Quick run

For a quick check, you can run one iteration per benchmark with
```
asv run -q
```
You can add -v --show-stderr for a more verbose output. --dry-run for not saving to disk.

`asv run -q -v --show-stderr`

In development, the following flags to `asv run` are often useful:
- `--bench`: to specify a subset of benchmarks (e.g., `tools.prep.PrepTF`). Regexp can be used.
- `--dry-run`: will not write results to disk
- `--quick`: will only run one repetition, and no results to disk
- `--show-stderr`: will print out stderr
- `--verbose`: provides further info on intermediate steps
- `--python=same`: runs the benchmarks in the same environment that `asv` was launched from

E.g.:
```
asv run --bench bench tools.prep.PrepTF --dry-run --show-stderr --quick
```



## Useful commands for running benchmarks
To run benchmarks on a specific commit:
```
$ asv run 88fbbc33^!
```

To run them up to a specific commit:
```
$ asv run 88fbbc33
```

To run them on a range of commits:
```
$ asv run 827f322b..729abcf3
```

To collate the benchmarks' results into a viewable website:
```
$ asv publish
```
This will create a tree of files in the `html` directory, but this cannot be viewed directly from the local filesystem, so we need to put them in a static site. `asv publish` also detects statistically significant decreases of performance, the results can be inspected in the 'Regression' tab of the static site.

To visualise the results in a static site:
```
$ asv preview
```
To share the website on the internet, put the files in the `html` directory on any webserver that can serve static content (e.g. GitHub pages).

To put the results in the `gh-pages` branch and push them to GitHub:
```
$ asv gh-pages
```

## Useful commands for managing the results

To remove benchmarks from the database, for example, for a specific commit:

```
$ asv rm commit_hash=a802047be
```

This will remove the selected results from the files in the `results` directory. To update the results in the static site, remember to run `publish` again!

See more options for `asv rm` in the [asv documentation](https://asv.readthedocs.io/en/stable/using.html#managing-the-results-database).

To compare the results of running the benchmarks on two commits:
```
$ asv compare 88fbbc33 827f322b
```

### Writing benchmarks: `setup` and `setup_cache`

- `setup` includes initialisation bits that should not be included
in the timing of the benchmark. It can be added as:
    - a method of a class, or
    - an attribute of a free function, or
    - a module-level setup function (run for every benchmark in the
    module, prior to any function-specific setup)

    If `setup` raises `NotImplementedError`, the benchmark is skipped

- `setup_cache` only performs the setup calculation once
(for each benchmark and each repeat) and caches the
result to disk. This may be useful if the setup is computationally
expensive.

    A separate cache is used for each environment and each commit. The cache is thrown out between benchmark runs.

    There are two options to persist the data for the benchmarks:
    - `setup_cache` returns a data structure, which asv pickles to disk,
        and then loads and passes as the first argument to each benchmark (not
        automagically though), or
    - `setup_cache` saves files to the cwd (which is a temp dir managed by
        asv), which are then explicitly loaded in each benchmark. The recommended practice is to actually read the data in a `setup` function, so that loading time is not part of the benchmark timing.



## Other handy commands
To update the machine information
```
$ asv machine
```

To display results from the previous runs on the command line
```
$ asv show
```

To use binary search to find a commit within the benchmarked range that produced a large regression
```
$ asv find
```
Note this will only find the global maximum if runtimes over the range are more-or-less monotonic. See the [asv docs](https://asv.readthedocs.io/en/stable/using.html#finding-a-commit-that-produces-a-large-regression) for further details.

To check the validity of written benchmarks
```
$ asv check
```

`asv` has features to run a given benchmark in the Python standard profiler `cProfile`, and then visualise the results in the tool of your choice. For example:
```
$ asv profile time_units.time_very_simple_unit_parse 10fc29cb
```
See the [asv docs on profiling](https://asv.readthedocs.io/en/stable/using.html#running-a-benchmark-in-the-profiler) for further details


----
## References
- [astropy-benchmarks repository](https://github.com/astropy/astropy-benchmarks/tree/main)
- [numpy benchmarks](https://github.com/numpy/numpy/tree/main/benchmarks/benchmarks)
- [asv documentation](https://asv.readthedocs.io/en/stable/index.html)
