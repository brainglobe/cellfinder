from os import path

from setuptools import find_packages, setup

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

requirements = [
    "brainreg",
    "cellfinder-core>=0.2.4",
    "configobj",
    "fancylog>=0.0.7",
    "imio",
    "brainglobe-utils>=0.2.5",
    "multiprocessing-logging>=0.3.4",
    "natsort",
    "numpy",
    "pandas",
    "packaging",
    "scikit-image",
    "tifffile",
    "tqdm",
]


setup(
    name="cellfinder",
    version="0.7.0",
    description="Automated 3D cell detection and registration of "
    "whole-brain images",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=requirements,
    extras_require={
        "napari": [
            "napari[pyside2]",
            "brainglobe-napari-io",
            "cellfinder-napari",
        ],
        "dev": [
            "black",
            "pytest-cov",
            "pytest",
            "gitpython",
            "coverage>=5.0.3",
            "bump2version",
            "pre-commit",
            "flake8",
        ],
    },
    python_requires=">=3.8",
    packages=find_packages(),
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "cellfinder = cellfinder.main:main",
            "cellfinder_curate_new = cellfinder.train.curation:main",
            "cellfinder_curate = cellfinder.train.curation_old:main",
        ]
    },
    url="https://brainglobe.info/cellfinder",
    project_urls={
        "Source Code": "https://github.com/brainglobe/cellfinder",
        "Bug Tracker": "https://github.com/brainglobe/cellfinder/issues",
        "Documentation": "https://docs.brainglobe.info/cellfinder",
    },
    author="Adam Tyson, Christian Niedworok, Charly Rousseau",
    author_email="code@adamltyson.com",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Operating System :: POSIX :: Linux",
        "Operating System :: Microsoft :: Windows :: Windows 10",
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
