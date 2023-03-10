import Cython.Build
from setuptools import Extension, setup

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
    ball_filter_extension,
    structure_detection_extension,
]


setup(
    ext_modules=Cython.Build.cythonize(extensions),
)
