from os import path

import Cython.Build
from setuptools import Extension, setup

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

requirements = [
    "dask[array]",
    "fancylog>=0.0.7",
    "imlib>=0.0.26",
    "natsort",
    "numpy",
    "scikit-image",
    "scikit-learn",
    "tensorflow>=2.5.0; "
    + "platform_system!='Darwin' or "
    + "platform_machine!='arm64'",
    "tensorflow-macos>=2.5.0; "
    + "platform_system=='Darwin' and "
    + "platform_machine=='arm64'",
    "tifffile",
    "tqdm",
]


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
    name="cellfinder-core",
    version="0.3.1",
    description="Automated 3D cell detection in large microscopy images",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=requirements,
    extras_require={
        "dev": [
            "black",
            "bump2version",
            "coverage>=5.0.3",
            "flake8",
            "gitpython",
            "pre-commit",
            "pytest-cov",
            "pytest-timeout",
            "pytest",
            "tox",
        ]
    },
    setup_requires=["cython"],
    python_requires=">=3.8",
    include_package_data=True,
    ext_modules=Cython.Build.cythonize(extensions),
    entry_points={
        "console_scripts": [
            "cellfinder_download = cellfinder_core.download.cli:main",
            "cellfinder_train = cellfinder_core.train.train_yml:cli",
            # "cellfinder_curate_new = cellfinder_core.train.curation:main",
            # "cellfinder_curate = cellfinder_core.train.curation_old:main",
        ]
    },
    url="https://brainglobe.info/cellfinder",
    project_urls={
        "Source Code": "https://github.com/brainglobe/cellfinder-core",
        "Bug Tracker": "https://github.com/brainglobe/cellfinder-core/issues",
        "Documentation": "https://docs.brainglobe.info/cellfinder",
    },
    author="Adam Tyson, Christian Niedworok, Charly Rousseau",
    author_email="code@adamltyson.com",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
    ],
    zip_safe=False,
)
