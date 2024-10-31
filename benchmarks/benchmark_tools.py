from pathlib import Path

import pooch
import torch
from torch.profiler import ProfilerActivity, profile
from torch.utils.benchmark import Compare, Timer

from cellfinder.core.tools.IO import fetch_pooch_directory


def get_test_data_path(path):
    """
    Create a test data registry for BrainGlobe.

    Returns:
        pooch.Pooch: The test data registry object.

    """
    registry = pooch.create(
        path=pooch.os_cache("brainglobe_test_data"),
        base_url="https://gin.g-node.org/BrainGlobe/test-data/raw/master/cellfinder/",
        env="BRAINGLOBE_TEST_DATA_DIR",
    )

    registry.load_registry(
        Path(__file__).parent.parent / "tests" / "data" / "pooch_registry.txt"
    )

    return fetch_pooch_directory(registry, path)


def time_filters(repeat, run, run_args, label):
    timer = Timer(
        stmt="run(*args)",
        globals={"run": run, "args": run_args},
        label=label,
        num_threads=4,
        description="",  # must be not None due to pytorch bug
    )
    return timer.timeit(number=repeat)


def compare_results(*results):
    # prints the results of all the timed tests
    compare = Compare(results)
    compare.trim_significant_figures()
    compare.colorize()
    compare.print()


def profile_cpu(repeat, run, run_args):
    with profile(
        activities=[ProfilerActivity.CPU],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_modules=True,
    ) as prof:
        for _ in range(repeat):
            run(*run_args)

    print(
        prof.key_averages(group_by_stack_n=1).table(
            sort_by="self_cpu_time_total", row_limit=20
        )
    )


def profile_cuda(repeat, run, run_args):
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_modules=True,
    ) as prof:
        for _ in range(repeat):
            run(*run_args)
            # make sure it's fully done filtering
            torch.cuda.synchronize("cuda")

    print(
        prof.key_averages(group_by_stack_n=1).table(
            sort_by="self_cuda_time_total", row_limit=20
        )
    )
