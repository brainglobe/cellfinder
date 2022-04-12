# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Python version support
Support for installing and running on Python 3.10 has been added. Support for
Python 3.7 has been dropped.

### Added
- Re-worked the way processes are created during the detection stage to remove
  ~20 seconds of overhead when running cell detection.

### Bug fixes
- Fixed macOS issues where cellfinder could hang during the detection stage.
