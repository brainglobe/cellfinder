import shutil
from pathlib import Path

from brainglobe_utils.general.system import get_num_processes

from cellfinder.core.tools.prep import (
    prep_model_weights,
    prep_models,
    prep_tensorflow,
)


class PrepModels:
    # parameters to sweep across
    param_names = ["model_name"]
    params = ["resnet50_tv", "resnet50_all"]

    # increase default timeout to allow for download
    timeout = 600

    # install path
    def benchmark_install_path(self):
        # also allow to run as "user" on GH actions?
        return Path(Path.home() / ".cellfinder-benchmarks")

    def setup(self, model_name):
        self.n_free_cpus = 2
        self.n_processes = get_num_processes(
            min_free_cpu_cores=self.n_free_cpus
        )
        self.trained_model = None
        self.model_weights = None
        self.install_path = self.benchmark_install_path()
        self.model_name = model_name

        # remove .cellfinder-benchmarks dir if it exists
        shutil.rmtree(self.install_path, ignore_errors=True)

    def teardown(self, model_name):
        # remove .cellfinder-benchmarks dir after benchmarks
        shutil.rmtree(self.install_path)

    def time_prep_models(self, model_name):
        prep_models(
            self.model_weights,
            self.install_path,
            model_name,
        )

    def time_prep_classification(self, model_name):
        prep_model_weights(
            self.model_weights,
            self.install_path,
            model_name,
            self.n_free_cpus,
        )


class PrepTF:
    def setup(self):
        n_free_cpus = 2
        self.n_processes = get_num_processes(min_free_cpu_cores=n_free_cpus)

    def time_prep_tensorflow(self):
        prep_tensorflow(self.n_processes)
