# Cell detection

## Methodology
Cell detection in cellfinder has three stages:

1. 2D filter each image plane independently.
2. 3D filter small batches of planes.
3. Merge detected cell candidate voxels into into structures.

### 2D filtering
Code to do 2D filtering can be found in ``cellfinder_core/detect/filters/plane``.
This of processing performs two tasks:
1. Applys a filter to enhance peaks in the data (``cellfinder_core/detect/filters/plane/classical_filter.py``).
   This consists of (in order)
   1. a median filter (`scipy.signal.medfilt2d`)
   1. a gaussian filter (`scipy.ndimage.gaussian_filter`)
   1. a laplacian filter (`scipy.signal.laplace`),
   1. inverting the data
   1. normalising to [0, 1]
   1. scaling to *clipping_value*.
1.

   Because applying several of the filters is more time efficient when done on floating point data types, each plane is cast to `float64` in this step.


## Memory usage
### 2D filtering
For each plane, the following memory is used:
- The plane itself is read into memory
- During filtering, a copy of the plane is made and cast to `float64`
- A small `uint8` mask is created to mark areas of the plane that are inside/outside of the brain

### 3D filtering
`ball_z_size` planes at a time (defaults to 15)
Cell detection - two planes at a time

In addition the first two processes need an extra plane to write to.
This means the maximum memory used by a single worker is `ball_z_size + 1` planes.
