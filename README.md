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
