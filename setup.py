import platform
from setuptools import setup, find_namespace_packages, Extension
from os import path

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

requirements = [
    "numpy<1.19.0,>=1.16.0",  # until tensorflow supports numpy 1.19
    "scikit-learn",
    "configparser",
    "pandas",
    "packaging",
    "scikit-image",
    "tifffile",
    "natsort",
    "tqdm",
    "multiprocessing-logging",
    "psutil",
    "configobj",
    "tensorflow>=2.3.1",
    "napari[pyside2]>=0.3.7",
    "napari-cellfinder>=0.1.1",
    "slurmio>=0.0.4",
    "fancylog>=0.0.7",
    "imlib>=0.0.26",
    "brainreg",
    "imio",
]


if platform.system() == "Windows":
    # FIXME: There must be a better way of doing this.
    base_tile_filter_extension = Extension(
        name="cellfinder.detect.filters.plane_filters.base_tile_filter",
        sources=[
            "cellfinder/detect/filters/plane_filters/" "base_tile_filter.pyx"
        ],
        language="c++",
    )

    ball_filter_extension = Extension(
        name="cellfinder.detect.filters.volume_filters.ball_filter",
        sources=[
            "cellfinder/detect/filters/volume_filters/" "ball_filter.pyx"
        ],
    )

    structure_detection_extension = Extension(
        name="cellfinder.detect.filters.volume_filters.structure_detection",
        sources=[
            "cellfinder/detect/filters/volume_filters/"
            "structure_detection.pyx"
        ],
        language="c++",
    )
else:
    base_tile_filter_extension = Extension(
        name="cellfinder.detect.filters.plane_filters.base_tile_filter",
        sources=[
            "cellfinder/detect/filters/plane_filters/" "base_tile_filter.pyx"
        ],
        libraries=["m"],
        language="c++",
    )

    ball_filter_extension = Extension(
        name="cellfinder.detect.filters.volume_filters.ball_filter",
        sources=[
            "cellfinder/detect/filters/volume_filters/" "ball_filter.pyx"
        ],
        libraries=["m"],
    )

    structure_detection_extension = Extension(
        name="cellfinder.detect.filters.volume_filters.structure_detection",
        sources=[
            "cellfinder/detect/filters/volume_filters/"
            "structure_detection.pyx"
        ],
        libraries=["m"],
        language="c++",
    )


setup(
    name="cellfinder",
    version="0.4.2",
    description="Automated 3D cell detection and registration of "
    "whole-brain images",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=requirements,
    extras_require={
        "dev": [
            "black",
            "pytest-cov",
            "pytest",
            "gitpython",
            "coverage>=5.0.3",
            "bump2version",
            "pre-commit",
            "flake8",
        ]
    },
    setup_requires=["cython"],
    python_requires=">=3.7",
    packages=find_namespace_packages(exclude=("docs", "doc_build", "tests")),
    include_package_data=True,
    ext_modules=[
        ball_filter_extension,
        structure_detection_extension,
        base_tile_filter_extension,
    ],
    entry_points={
        "console_scripts": [
            "cellfinder = cellfinder.main:main",
            "cellfinder_download = cellfinder.download.cli:main",
            "cellfinder_train = cellfinder.train.train_yml:main",
            "cellfinder_curate = cellfinder.train.curation:main",
        ]
    },
    url="https://cellfinder.info",
    project_urls={
        "Source Code": "https://github.com/brainglobe/cellfinder",
        "Bug Tracker": "https://github.com/brainglobe/cellfinder/issues",
        "Documentation": "https://docs.brainglobe.info/cellfinder",
    },
    author="Adam Tyson, Christian Niedworok, Charly Rousseau",
    author_email="adam.tyson@ucl.ac.uk",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Operating System :: POSIX :: Linux",
        "Operating System :: Microsoft :: Windows :: Windows 10",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
    ],
    zip_safe=False,
)
