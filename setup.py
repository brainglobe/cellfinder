#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import codecs
from setuptools import setup, find_packages


def read(fname):
    file_path = os.path.join(os.path.dirname(__file__), fname)
    return codecs.open(file_path, encoding='utf-8').read()

requirements = [
    "napari-plugin-engine >= 0.1.4",
    "numpy",
    "cellfinder-core"
]

# https://github.com/pypa/setuptools_scm
use_scm = {"write_to": "cellfinder_napari/_version.py"}

setup(
    name='cellfinder-napari',
    author='Adam Tyson',
    author_email='adam.tyson@ucl.ac.uk',
    license='BSD-3',
    url='https://github.com/brainglobe/cellfinder-napari',
    description='Efficient cell detection in large images',
    long_description=read('README.md'),
    long_description_content_type='text/markdown',
    packages=find_packages(),
    python_requires='>=3.6',
    install_requires=requirements,
    use_scm_version=use_scm,
    setup_requires=['setuptools_scm'],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Framework :: napari',
        'Topic :: Software Development :: Testing',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Operating System :: OS Independent',
        'License :: OSI Approved :: BSD License',
    ],
    entry_points={
        'napari.plugin': [
            'cellfinder-napari = cellfinder_napari',
        ],
    },
)
