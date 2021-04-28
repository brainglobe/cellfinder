[![Python Version](https://img.shields.io/pypi/pyversions/cellfinder-core.svg)](https://pypi.org/project/cellfinder-core)
[![PyPI](https://img.shields.io/pypi/v/cellfinder-core.svg)](https://pypi.org/project/cellfinder-core)
[![Downloads](https://pepy.tech/badge/cellfinder-core)](https://pepy.tech/project/cellfinder-core)
[![Wheel](https://img.shields.io/pypi/wheel/cellfinder-core.svg)](https://pypi.org/project/cellfinder-core)
[![Development Status](https://img.shields.io/pypi/status/cellfinder-core.svg)](https://github.com/brainglobe/cellfinder-core)
[![Tests](https://img.shields.io/github/workflow/status/brainglobe/cellfinder-core/tests)](
    https://github.com/brainglobe/cellfinder-core/actions)
[![Coverage Status](https://coveralls.io/repos/github/brainglobe/cellfinder-core/badge.svg?branch=main)](https://coveralls.io/github/brainglobe/cellfinder-core?branch=main)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/python/black)
[![Gitter](https://badges.gitter.im/brainglobe.svg)](https://gitter.im/BrainGlobe/cellfinder/?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)
[![Contributions](https://img.shields.io/badge/Contributions-Welcome-brightgreen.svg)](https://docs.brainglobe.info/cellfinder/contributing)
[![Website](https://img.shields.io/website?up_message=online&url=https%3A%2F%2Fcellfinder.info)](https://cellfinder.info)
[![Twitter](https://img.shields.io/twitter/follow/findingcells?style=social)](https://twitter.com/findingcells)
# cellfinder-core
Standalone cellfinder cell detection algorithm 

This package implements the cell detection algorithm from 
[Tyson, Rousseau & Niedworok et al. (2021)](https://www.biorxiv.org/content/10.1101/2020.10.21.348771v2) 
without any dependency on data type (i.e. it can be used outside of 
whole-brain microscopy). 

`cellfinder-core` supports the 
[cellfinder](https://github.com/brainglobe/cellfinder) software for 
whole-brain microscopy analysis, and the algorithm can also be implemented in 
[napari](https://napari.org/index.html) using the 
[cellfinder napari plugin](https://github.com/brainglobe/cellfinder-napari).

---

## Instructions

### Installation
`cellfinder-core` supports Python 3.7, 3.8 (3.9 when supported by TensorFlow), 
and works across Linux, Windows, and should work on most versions of macOS 
(although this is not tested).

Assuming you have a Python 3.7 or 3.8 environment set up 
(e.g. [using conda](https://docs.brainglobe.info/cellfinder/using-conda)), 
you can install `cellfinder-core` with:
```bash
pip install cellfinder-core
```

Once you have [installed napari](https://napari.org/index.html#installation). 
You can install napari either through the napari plugin installation tool, or 
directly from PyPI with:
```bash
pip install cellfinder-napari
```

N.B. To speed up cellfinder, you need CUDA & cuDNN installed. Instructions 
[here](https://docs.brainglobe.info/cellfinder/installation/using-gpu).

### Usage
Before using cellfinder-core, it may be useful to take a look at the 
[preprint](https://www.biorxiv.org/content/10.1101/2020.10.21.348771v2) which
outlines the algorithm.

The API is not yet fully documented. For an idea of what the parameters do, 
see the documentation for the cellfinder whole-brain microscopy image analysis 
command-line tool ([cell candidate detection](https://docs.brainglobe.info/cellfinder/user-guide/command-line/candidate-detection),
[cell candidate classification](https://docs.brainglobe.info/cellfinder/user-guide/command-line/classification)).
It may also be useful to try the
[cellfinder napari plugin](https://github.com/brainglobe/cellfinder-napari)
so you can adjust the parameters in a GUI. 

#### To run the full pipeline (cell candidate detection and classification)
```python
from cellfinder_core.main import main as cellfinder_run
import tifffile

signal_array = tifffile.imread("/path/to/signal_image.tif")
background_array = tifffile.imread("/path/to/background_image.tif")

voxel_sizes = [5, 2, 2] # in microns
detected_cells = cellfinder_run(signal_array,background_array,voxel_sizes)
```

The output is a list of
[imlib Cell objects](https://github.com/adamltyson/imlib/blob/51ec5a8053e738776ceaa8d44e531b3c4b0e29d8/imlib/cells/cells.py#L15).
Each `Cell` has a centroid coordinate, and a type:

```python
print(detected_cells[0])
# Cell: x: 132, y: 308, z: 10, type: 2
```

Cell type 2 is a "real" cell, and Cell type 1 is a "rejected" object (i.e. 
not classified as a cell):

```python
from imlib.cells.cells import Cell
print(Cell.CELL)
# 2

print(Cell.NO_CELL)
# 1
```

#### Using dask for lazy loading
`cellfinder-core` supports most array-like objects. Using 
[Dask arrays](https://docs.dask.org/en/latest/array.html) allows for lazy 
loading of data, allowing large (e.g. TB) datasets to be processed. 
`cellfinder-core` comes with a function 
(based on [napari-ndtiffs](https://github.com/tlambert03/napari-ndtiffs)) to 
load a series of image files (e.g. a directory of 2D tiff files) as a Dask 
array. `cellfinder-core` can then be used in the same way as with a numpy array.

```python
from cellfinder_core.main import main as cellfinder_run
from cellfinder_core.tools.IO import read_with_dask

signal_array = read_with_dask("/path/to/signal_image_directory")
background_array = read_with_dask("/path/to/background_image_directory")

voxel_sizes = [5, 2, 2] # in microns
detected_cells = cellfinder_run(signal_array,background_array,voxel_sizes)

```

#### Running the cell candidate detection and classification separately.
```python
import tifffile
from pathlib import Path

from cellfinder_core.detect import detect
from cellfinder_core.classify import classify

signal_array = tifffile.imread("/path/to/signal_image.tif")
background_array = tifffile.imread("/path/to/background_image.tif")
voxel_sizes = [5, 2, 2] # in microns

home = Path.home()
install_path = home / ".cellfinder"

start_plane=0
end_plane=-1
trained_model=None
model_weights=None
model="resnet50_tv"
batch_size=32
n_free_cpus=2
network_voxel_sizes=[5, 1, 1]
soma_diameter=16
ball_xy_size=6
ball_z_size=15
ball_overlap_fraction=0.6
log_sigma_size=0.2
n_sds_above_mean_thresh=10
soma_spread_factor=1.4
max_cluster_size=100000
cube_width=50
cube_height=50
cube_depth=20
network_depth="50"

cell_candidates = detect.main(
    signal_array,
    start_plane,
    end_plane,
    voxel_sizes,
    soma_diameter,
    max_cluster_size,
    ball_xy_size,
    ball_z_size,
    ball_overlap_fraction,
    soma_spread_factor,
    n_free_cpus,
    log_sigma_size,
    n_sds_above_mean_thresh,
)

if len(cell_candidates) > 0: # Don't run if there's nothing to classify
    classified_cells = classify.main(
        cell_candidates,
        signal_array,
        background_array,
        n_free_cpus,
        voxel_sizes,
        network_voxel_sizes,
        batch_size,
        cube_height,
        cube_width,
        cube_depth,
        trained_model,
        model_weights,
        network_depth,
    )
```
---
### More info

More documentation about cellfinder and other BrainGlobe tools can be 
found [here](https://docs.brainglobe.info). 
 
This software is at a very early stage, and was written with our data in mind. 
Over time we hope to support other data types/formats. If you have any 
questions or issues, please get in touch by 
[email](mailto:code@adamltyson.com?subject=cellfinder-core), 
[gitter](https://gitter.im/BrainGlobe/cellfinder) or by 
[raising an issue](https://github.com/brainglobe/cellfinder-core/issues).

---
## Illustration

### Introduction
cellfinder takes a stitched, but otherwise raw dataset with at least 
two channels:
 * Background channel (i.e. autofluorescence)
 * Signal channel, the one with the cells to be detected:
 
![raw](https://raw.githubusercontent.com/brainglobe/cellfinder/master/resources/raw.png)
**Raw coronal serial two-photon mouse brain image showing labelled cells**


### Cell candidate detection
Classical image analysis (e.g. filters, thresholding) is used to find 
cell-like objects (with false positives):

![raw](https://raw.githubusercontent.com/brainglobe/cellfinder/master/resources/detect.png)
**Candidate cells (including many artefacts)**


### Cell candidate classification
A deep-learning network (ResNet) is used to classify cell candidates as true 
cells or artefacts:

![raw](https://raw.githubusercontent.com/brainglobe/cellfinder/master/resources/classify.png)
**Cassified cell candidates. Yellow - cells, Blue - artefacts**

---
## Citing cellfinder
If you find this plugin useful, and use it in your research, please cite the preprint outlining the cell detection algorithm:
> Tyson, A. L., Rousseau, C. V., Niedworok, C. J., Keshavarzi, S., Tsitoura, C., Cossell, L., Strom, M. and Margrie, T. W. (2021) “A deep learning algorithm for 3D cell detection in whole mouse brain image datasets’ bioRxiv, [doi.org/10.1101/2020.10.21.348771](https://doi.org/10.1101/2020.10.21.348771)


**If you use this, or any other tools in the brainglobe suite, please
 [let us know](mailto:code@adamltyson.com?subject=cellfinder-core), and 
 we'd be happy to promote your paper/talk etc.**
