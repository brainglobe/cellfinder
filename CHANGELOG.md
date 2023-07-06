# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## 0.5.0 - 2023-07-06

### Changes
- `cellfinder-core` cannot be imported without `tensorflow` present, [#186](https://github.com/brainglobe/cellfinder-core/pull/186). This is in [light of a bug](https://github.com/conda-forge/cellfinder-core-feedstock/issues/13) with the `conda-forge` `cellfinder-core` package and the `tensorflow` dependency being unavailable through these channels. As a compromise, `cellfinder-core` [can still be installed from `conda-forge`](https://github.com/conda-forge/cellfinder-core-feedstock/pull/14), however Windows users will have to manually install `tensorflow` into their environment themselves.
- The `imlib` library has been retired, [#182](https://github.com/brainglobe/cellfinder-core/pull/182). We now [depend on `brainglobe-utils`](https://github.com/brainglobe/brainglobe-utils), a library of functions used across a number of [BrainGlobe](https://brainglobe.info/) tools.

### Developer changes
- `mypy` has been updated to avoid errors whilst type checking, [#180](https://github.com/brainglobe/cellfinder-core/pull/180).
- Typehints have been added to `cellfinder-core.classificataion`, [#181](https://github.com/brainglobe/cellfinder-core/pull/181).

## 0.4.0 - 2023-05-17

### Changed
- The cell detection code has been re-written to use [numba](https://numba.readthedocs.io/en/stable/) instead of Cython. This means we no longer have to pre-compile the cell detection code when making a new release of `cellfinder-core`, but does mean `numba` is now a dependency that users have to have installed. The cell detection algorithm remains the same, and perfomance has been tested to make sure it is as fast after this change.

## 0.3.0 - 2022-04-25

### Added
- Re-worked the way processes are created during the detection stage to remove
  ~20 seconds of overhead when running cell detection.
- Support for Python 3.10.

### Removed
- Support for Python 3.7.

### Bug fixes
- Fixed macOS issues where cellfinder could hang during the detection stage.
