from setuptools import setup, find_packages
from os import path

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

requirements = [
    "napari",
    "napari-plugin-engine >= 0.1.4",
    "napari-ndtiffs",
    "brainglobe-napari-io",
    "cellfinder-core>=0.2.4",
]

setup(
    name="cellfinder-napari",
    version="0.0.11",
    author="Adam Tyson",
    author_email="code@adamltyson.com",
    license="GPL-3.0",
    description="Efficient cell detection in large images",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.7",
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
            "check-manifest",
        ]
    },
    url="https://cellfinder.info",
    project_urls={
        "Source Code": "https://github.com/brainglobe/cellfinder-napari",
        "Bug Tracker": "https://github.com/brainglobe/cellfinder-napari-issues",
        "Documentation": "https://docs.brainglobe.info/cellfinder-napari/",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Framework :: napari",
        "Topic :: Software Development :: Testing",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: BSD License",
    ],
    entry_points={"napari.plugin": ["cellfinder = cellfinder_napari.plugins"]},
)
