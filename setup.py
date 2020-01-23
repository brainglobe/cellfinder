import platform
from setuptools import setup, find_namespace_packages, Extension

requirements = [
    "numpy",
    "scikit-learn",
    "scipy >= 0.18",
    "configparser",
    "pandas",
    "packaging",
    "scikit-image",
    "tifffile",
    "natsort",
    "tqdm",
    "anytree",
    "h5py",
    "PyYAML",
    "multiprocessing-logging",
    "psutil",
    "nibabel",
    "configobj",
    "read-roi",
    "tables",
    "slurmio",
    "brainio>=0.0.9",
    "fancylog",
    "micrometa",
    "amap>=0.0.11",
    "napari>=0.2.8",
    # "brainrender",
    "toolz>=0.7.3",
    "tensorflow>=2.1.0",
    "six>=1.12.0",
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
    version="0.3.4rc1",
    description="Cell detection for whole-brain microscopy",
    install_requires=requirements,
    extras_require={
        "dev": [
            "sphinx",
            "recommonmark",
            "sphinx_rtd_theme",
            "pydoc-markdown",
            "black",
            "pytest-cov",
            "pytest",
            "gitpython",
            "coveralls",
            "coverage>=5.0.3",
        ]
    },
    setup_requires=["cython"],
    python_requires=">=3.6, <3.8",
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
            "cellfinder_train = cellfinder.classify.train_yml:main",
            "cellfinder_view_cells = cellfinder.viewer.two_dimensional:main",
            "cellfinder_view_3D = cellfinder.viewer.three_dimensional:main",
            "cellfinder_gen_cubes = cellfinder.extract.cli:main",
            "cellfinder_batch_extract = cellfinder.batch.extract:main",
            "cellfinder_cells_combine = cellfinder.batch.cells_combine:main",
            "cellfinder_count_summary = "
            "cellfinder.summarise.count_summary:main",
            "cellfinder_region_summary = "
            "cellfinder.analyse.group.region_summary:main",
            "cellfinder_xml_crop = cellfinder.utils.xml_crop:main",
            "cellfinder_xml_scale = cellfinder.utils.xml_scale:main",
            "cellfinder_cell_standard = "
            "cellfinder.standard_space.cells_to_standard_space:main",
            "cellfinder_roi_transform = cellfinder.utils.roi_transform:main",
            "cellfinder_gen_region_vol = "
            "cellfinder.utils.generate_region_volume:main",
            "cellfinder_cells_to_brainrender = "
            "cellfinder.utils.cells_to_brainrender:main",
        ]
    },
    url="https://github.com/SainsburyWellcomeCentre/cellfinder",
    author="Adam Tyson, Christian Niedworok, Charly Rousseau",
    author_email="adam.tyson@ucl.ac.uk",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
    ],
    zip_safe=False,
)
