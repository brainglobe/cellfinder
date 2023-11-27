import numpy as np
from pyinstrument import Profiler

from cellfinder.core.detect.filters.volume.volume_filter import VolumeFilter

# Use random data for signal data
ball_z_size = 3


def gen_signal_array(ny, nx):
    shape = (ball_z_size, ny, nx)
    return np.random.randint(low=0, high=65536, size=shape, dtype=np.uint16)


signal_array = gen_signal_array(667, 510)

soma_diameter = 8
setup_params = (
    signal_array[0, :, :].T,
    soma_diameter,
    3,  # ball_xy_size,
    ball_z_size,
    0.6,  # ball_overlap_fraction,
    0,  # start_plane,
)

mp_3d_filter = VolumeFilter(
    soma_diameter=soma_diameter,
    setup_params=setup_params,
    n_planes=len(signal_array),
    n_locks_release=1,
)

# Use random data for mask data
mask = np.random.randint(low=0, high=2, size=(42, 32), dtype=bool)

# Fill up the 3D filter with planes
for plane in signal_array:
    mp_3d_filter.ball_filter.append(plane, mask)
    if mp_3d_filter.ball_filter.ready:
        break

# Run the 3D filter
profiler = Profiler()
profiler.start()
for i in range(10):
    # Repeat same filter 10 times to increase runtime
    mp_3d_filter._run_filter()

profiler.stop()
profiler.print(show_all=True)
