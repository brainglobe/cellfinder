import Cython.Build
from setuptools import Extension, setup

base_tile_filter_extension = Extension(
    name="cellfinder_core.detect.filters.plane.base_tile_filter",
    sources=["src/cellfinder_core/detect/filters/plane/base_tile_filter.pyx"],
    language="c++",
)

ball_filter_extension = Extension(
    name="cellfinder_core.detect.filters.volume.ball_filter",
    sources=["src/cellfinder_core/detect/filters/volume/ball_filter.pyx"],
)

structure_detection_extension = Extension(
    name="cellfinder_core.detect.filters.volume.structure_detection",
    sources=[
        "src/cellfinder_core/detect/filters/volume/structure_detection.pyx"
    ],
    language="c++",
)

extensions = [
    base_tile_filter_extension,
    ball_filter_extension,
    structure_detection_extension,
]


setup(
    ext_modules=Cython.Build.cythonize(extensions),
)
