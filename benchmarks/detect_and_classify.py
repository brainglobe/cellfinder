from pathlib import Path

import dask.array as da

from cellfinder_core.main import main
from cellfinder_core.tools.IO import read_with_dask

data_dir = (
    Path(__file__).parent
    / ".."
    / "tests"
    / "data"
    / "integration"
    / "detection"
).resolve()
signal_data_path = data_dir / "crop_planes" / "ch0"
background_data_path = data_dir / "crop_planes" / "ch1"

voxel_sizes = [5, 2, 2]

# Read data
signal_array = read_with_dask(str(signal_data_path))
background_array = read_with_dask(str(background_data_path))

# Artificially increase size of the test data
repeats = 5
signal_array = da.repeat(signal_array, repeats=repeats, axis=0)
background_array = da.repeat(background_array, repeats=repeats, axis=0)

if __name__ == "__main__":
    # Run detection & classification
    main(signal_array, background_array, voxel_sizes, n_free_cpus=0)
