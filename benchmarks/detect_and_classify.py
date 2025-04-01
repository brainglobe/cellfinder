import dask.array as da
import numpy as np

from cellfinder.core.main import main
from cellfinder.core import logger

voxel_sizes = (5, 2, 2)

# Use random data for signal/background data
repeats = 2
shape = (30 * repeats, 510, 667)

signal_array = da.random.random(shape)
signal_array = (signal_array * 65535).astype(np.uint16)

background_array = da.random.random(shape)
background_array = (signal_array * 65535).astype(np.uint16)

array_size_MB = signal_array.nbytes / 1024 / 1024
logger.debug(f"Signal array size = {array_size_MB:.02f} MB")

if __name__ == "__main__":
    # Run detection & classification
    main(signal_array, background_array, voxel_sizes, n_free_cpus=0)
