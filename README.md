[![Python Version](https://img.shields.io/pypi/pyversions/cellfinder-core.svg)](https://pypi.org/project/cellfinder)
[![PyPI](https://img.shields.io/pypi/v/cellfinder-core.svg)](https://pypi.org/project/cellfinder)
[![Downloads](https://pepy.tech/badge/cellfinder-core)](https://pepy.tech/project/cellfinder)
[![Wheel](https://img.shields.io/pypi/wheel/cellfinder-core.svg)](https://pypi.org/project/cellfinder)
[![Development Status](https://img.shields.io/pypi/status/cellfinder-core.svg)](https://github.com/brainglobe/cellfinder)
[![Tests](https://img.shields.io/github/actions/workflow/status/brainglobe/cellfinder/test_and_deploy.yml?branch=main)](https://github.com/brainglobe/cellfinder/actions)
[![codecov](https://codecov.io/gh/brainglobe/cellfinder-core/branch/main/graph/badge.svg?token=nx1lhNI7ox)](https://codecov.io/gh/brainglobe/cellfinder)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/python/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Contributions](https://img.shields.io/badge/Contributions-Welcome-brightgreen.svg)](https://brainglobe.info/developers/index.html)
[![Twitter](https://img.shields.io/twitter/follow/brain_globe?style=social)](https://twitter.com/brain_globe)

# cellfinder

cellfinder is software for automated 3D cell detection in very large 3D images (e.g., serial two-photon or lightsheet volumes of whole mouse brains).
There are three different ways to interact and use it, each with different user interfaces and objectives in mind.
For more details, head over to [the documentation website](https://brainglobe.info/documentation/cellfinder/index.html).

At a glance:

- There is a command-line interface called [brainmapper](https://brainglobe.info/documentation/brainglobe-workflows/brainmapper/index.html) that integrates [with `brainreg`](https://github.com/brainglobe/brainreg) for automated cell detection and classification. You can install it through [`brainglobe-workflows`](https://brainglobe.info/documentation/brainglobe-workflows/index.html).
- There is a [napari plugin](https://brainglobe.info/documentation/cellfinder/user-guide/napari-plugin/index.html) for interacting graphically with the cellfinder tool.
- There is a [Python API](https://brainglobe.info/documentation/cellfinder/user-guide/cellfinder-core.html) to allow users to integrate BrainGlobe tools into their custom workflows.

## Installation

You can find [the installation instructions](https://brainglobe.info/documentation/cellfinder/installation.html#installation) on the BrainGlobe website, which will go into more detail about the installation process if you want to minimise your installation to suit your needs.
However, we recommend that users install `cellfinder` either through installing BrainGlobe version 1, or (if you also want the command-line interface) installing `brainglobe-workflows`.

```bash
# If you want to install all BrainGlobe tools, including cellfinder, in a consistent manner with one command:
pip install brainglobe>=1.0.0
# If you want to install the brainmapper CLI tool as well:
pip install brainglobe-workflows>=1.0.0
```

If you only want the `cellfinder` package by itself, you can `pip install` it alone:

```bash
pip install cellfinder>=1.0.0
```

Be sure to specify a version greater than version `v1.0.0` - prior to this version the `cellfinder` package had a very different structure that is incompatible with BrainGlobe version 1 and the other tools in the BrainGlobe suite.
See [our blog posts](https://brainglobe.info/blog/) for more information on the release of BrainGlobe version 1.

## Contributing

If you have encountered a bug whilst using cellfinder, please [open an issue on GitHub](https://github.com/brainglobe/cellfinder/issues).

If you are interested in contributing to cellfinder (thank you!) - please head over to our [developer documentation](https://brainglobe.info/community/developers/index.html).
