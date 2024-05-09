# Benchmarks
`detect_and_classify.py` contains a simple script that runs
detection and classification with the small test dataset.

## Memory
[memory_profiler](https://github.com/pythonprofilers/memory_profiler)
can be used to profile memory useage. Install, and then run
`mprof run --include-children --multiprocess detect_and_classify.py`. It is **very**
important to use these two flags to capture memory usage by the additional
processes that cellfinder.core uses.

To show the results of the latest profile run, run `mprof plot`.
