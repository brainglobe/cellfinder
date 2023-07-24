# Benchmarking with asv
[Install asv](https://asv.readthedocs.io/en/stable/installing.html) by running:
```
pip install asv
```

`asv` works roughly as follows:
1. It creates a virtual environment (as defined in the config)
2. It installs the software package version of a specific commit (or of a local commit)
3. It times the benchmarking tests and saves the results to json files
4. The json files are 'published' into an html dir
5. The html dir can be visualised in a static website

## Running benchmarks
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
