# cellfinder-napari

[![License](https://img.shields.io/pypi/l/cellfinder-napari.svg?color=green)](https://github.com/napari/cellfinder-napari/raw/master/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/cellfinder-napari.svg?color=green)](https://pypi.org/project/cellfinder-napari)
[![Python Version](https://img.shields.io/pypi/pyversions/cellfinder-napari.svg?color=green)](https://python.org)
[![tests](https://github.com/brainglobe/cellfinder-napari/workflows/tests/badge.svg)](https://github.com/brainglobe/cellfinder-napari/actions)
[![codecov](https://codecov.io/gh/brainglobe/cellfinder-napari/branch/master/graph/badge.svg)](https://codecov.io/gh/brainglobe/cellfinder-napari)
[![Downloads](https://pepy.tech/badge/cellfinder-napari)](https://pepy.tech/project/cellfinder-napari)
[![Wheel](https://img.shields.io/pypi/wheel/cellfinder.svg)](https://pypi.org/project/cellfinder)
[![Development Status](https://img.shields.io/pypi/status/cellfinder-napari.svg)](https://github.com/brainglobe/cellfinder-napari)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/python/black)
[![Contributions](https://img.shields.io/badge/Contributions-Welcome-brightgreen.svg)](https://docs.brainglobe.info/cellfinder/contributing)
[![Website](https://img.shields.io/website?up_message=online&url=https%3A%2F%2Fcellfinder.info)](https://cellfinder.info)
[![Twitter](https://img.shields.io/twitter/follow/findingcells?style=social)](https://twitter.com/findingcells)



### Efficient cell detection in large images (e.g. whole mouse brain images)

This package implements the cell detection algorithm from 
[Tyson, Rousseau & Niedworok et al. (2021)](https://www.biorxiv.org/content/10.1101/2020.10.21.348771v2) 
for [napari](https://napari.org/index.html), based on the 
[cellfinder-core](https://github.com/brainglobe/cellfinder-core) package.

This algorithm can also be used within the original 
[cellfinder](https://github.com/brainglobe/cellfinder) software for 
whole-brain microscopy analysis.

----
![raw](https://raw.githubusercontent.com/brainglobe/cellfinder-napari/master/resources/cellfinder-napari.gif)

**Visualising detected cells in the cellfinder napari plugin**

----
## Instructions

### Installation
Once you have [installed napari](https://napari.org/index.html#installation). 
You can install napari either through the napari plugin installation tool, or 
directly from PyPI with:
```bash
pip install cellfinder-napari
```

### Usage
Full documentation can be 
found [here](https://docs.brainglobe.info/cellfinder-napari). 
 
This software is at a very early stage, and was written with our data in mind. 
Over time we hope to support other data types/formats. If you have any 
questions or issues, please get in touch by 
[email](mailto:code@adamltyson.com?subject=cellfinder-napari), 
[on the forum](https://forum.image.sc/tag/brainglobe) or by 
[raising an issue](https://github.com/brainglobe/cellfinder-napari/issues).


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

## Citing cellfinder

If you find this plugin useful, and use it in your research, please cite the preprint outlining the cell detection algorithm:
> Tyson, A. L., Rousseau, C. V., Niedworok, C. J., Keshavarzi, S., Tsitoura, C., Cossell, L., Strom, M. and Margrie, T. W. (2021) “A deep learning algorithm for 3D cell detection in whole mouse brain image datasets’ PLOS Computational Biology, 17(5), e1009074
[https://doi.org/10.1371/journal.pcbi.1009074](https://doi.org/10.1371/journal.pcbi.1009074)


**If you use this, or any other tools in the brainglobe suite, please
 [let us know](mailto:code@adamltyson.com?subject=cellfinder-napari), and 
 we'd be happy to promote your paper/talk etc.**
