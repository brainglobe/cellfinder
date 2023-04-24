# Cell detection

## Methodology
Cell detection in cellfinder has three stages:

1. 2D filter each image plane independently.
2. 3D filter small batches of planes.
3. Merge detected cell candidate voxels into into structures.

### 2D filtering
Code can be found in `cellfinder_core/detect/filters/plane`.
Each plane of data is filtered independently, and in parallel across a number of processes.

This part of processing performs two tasks:
1. Applys a filter to enhance peaks in the data (``cellfinder_core/detect/filters/plane/classical_filter.py``).
   This consists of (in order)
   1. a median filter (`scipy.signal.medfilt2d`)
   1. a gaussian filter (`scipy.ndimage.gaussian_filter`)
   1. a laplacian filter (`scipy.signal.laplace`),
   1. inverting the data
   1. normalising to [0, 1]
   1. scaling to *clipping_value*.

   Because applying several of the filters is more time efficient when done on floating point data types, each plane is cast to `float64` in this step.

1. Works out which areas of the plane are inside or outside of the brain. To do this the plane is divied into square tiles that have edge length `2 * soma_diameter`. The lower corner tile is assumed to be outside the brain, and any tiles that have a mean intensity less than `1 + mean + (2 * stddev)` of the corner tile are marked as being outside the brain. This speeds up processing in later steps by automatically skipping over tiles marked as outside the brain in this step.

### 3D filtering
Code can be found in `cellfinder_core/detect/filters/volume/ball_filter.py`.
Both this step and the structure detection step take place in the main `Python` process, with no parallelism. As the planes are processed in the 2D filtering step, they are passed to this step. When `ball_z_size` planes have been handed over, 3D filtering begins.

The 3D filter stores a 3D array that has depth `ball_z_size`, and contains `ball_z_size` number of planes. This is a small 3D slice of the original data. A spherical kernel runs across the x, y dimensions, and where enough intensity overlaps with the spherical kernel the voxel at the centre of the kernel is marked as being part of a cell. The output of this step is the central plane of the array, with marked cells.


### Structure detection
Code can be found in `cellfinder_core/detect/filters/volume/structure_detection.py`.
This step takes the planes output from 3D filtering with marked cell voxels, and detects collections of voxels that are adjacent.

## Memory usage
### 2D filtering
For each plane, the following memory is used:
- The plane itself is read into memory
- During filtering, a copy of the plane is made and cast to `float64`
- A small `uint8` mask is created to mark areas of the plane that are inside/outside of the brain

### 3D filtering
- `ball_z_size` planes at a time are stored.
- Twice this amount of memory is required to roll the array each time a new array is fed to the 3D filter stage.

### Structure detection
- Two planes are cast to `uint64` and are stored at the same time
