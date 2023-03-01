import numpy as np
from pyinstrument import Profiler

from cellfinder_core.detect.filters.volume.volume_filter import VolumeFilter

# Use random data for signal data
repeats = 2
shape = (30 * repeats, 667, 510)

signal_array = np.random.random(shape)
signal_array = (signal_array * 65535).astype(np.uint16)


soma_diameter = 8
setup_params = [
    signal_array[0, :, :].T,
    soma_diameter,
    3,  # ball_xy_size,
    3,  # ball_z_size,
    0.6,  # ball_overlap_fraction,
    0,  # start_plane,
]

mp_3d_filter = VolumeFilter(
    soma_diameter=soma_diameter,
    setup_params=setup_params,
    planes_paths_range=signal_array,
)

# Use random data for mask data
mask_shape = (42, 32)
mask = np.random.randint(2, size=42 * 32, dtype=np.uint8).reshape(mask_shape)

# Fill up the 3D filter with planes
for plane in signal_array:
    mp_3d_filter.ball_filter.append(plane, mask)
    if mp_3d_filter.ball_filter.ready:
        break

# Run the 3D filter
profiler = Profiler()
profiler.start()
for i in range(100):
    mp_3d_filter._run_filter()

profiler.stop()
profiler.print(show_all=True)
