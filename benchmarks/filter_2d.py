import numpy as np
from pyinstrument import Profiler

from cellfinder_core.detect.filters.plane import TileProcessor
from cellfinder_core.detect.filters.setup_filters import setup_tile_filtering

# Use random 16-bit integer data for signal plane
shape = (10000, 10000)

signal_array_plane = np.random.randint(low=0, high=65535, size=shape, dtype=np.uint16)

clipping_value, threshold_value = setup_tile_filtering(signal_array_plane)
tile_processor = TileProcessor(
    clipping_value=clipping_value,
    threshold_value=threshold_value,
    soma_diameter=16,
    log_sigma_size=0.2,
    n_sds_above_mean_thresh=10,
)
if __name__ == "__main__":
    profiler = Profiler()
    profiler.start()
    plane, tiles = tile_processor.get_tile_mask(signal_array_plane)
    profiler.stop()
    profiler.print(show_all=True)
