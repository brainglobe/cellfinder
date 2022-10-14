from os import path

from setuptools import find_packages, setup

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

requirements = [
    "napari",
    "napari-plugin-engine >= 0.1.4",
    "napari-ndtiffs",
    "brainglobe-napari-io",
    "cellfinder-core>=0.3",
    "pooch>=1",  # For downloading sample data
]

setup(
    name="cellfinder-napari",
    version="0.0.20-rc0",
    author="Adam Tyson",
    author_email="code@adamltyson.com",
    license="BSD-3-Clause",
    description="Efficient cell detection in large images",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "black",
            "bump2version",
            "gitpython",
            "pre-commit",
            "pytest",
            "pytest-cov",
            "pytest-qt",
        ]
    },
    url="https://brainglobe.info/cellfinder",
    project_urls={
        "Source Code": "https://github.com/brainglobe/cellfinder-napari",
        "Bug Tracker": "https://github.com/brainglobe/cellfinder-napari/issues",
        "Documentation": "https://docs.brainglobe.info/cellfinder-napari/",
        "User Support": "https://forum.image.sc/tag/brainglobe",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Framework :: napari",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
    ],
    entry_points={
        "napari.manifest": [
            "cellfinder-napari = cellfinder_napari:napari.yaml",
        ],
    },
    package_data={"cellfinder_napari": ["napari.yaml"]},
)
