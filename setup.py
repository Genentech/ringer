#!/usr/bin/env python

from setuptools import find_packages, setup

with open("README.md") as source:
    long_description = source.read()

setup(
    name="ringer",
    version="1.1.0",
    description="Rapid conformer generation for macrocycles with internal coordinate diffusion",
    author="Colin Grambow, Hayley Weir, Kangway Chuang",
    author_email="grambow.colin@gene.com",
    url="https://github.com/Genentech/ringer",
    install_requires=[],
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "train = ringer.train:main",
            "evaluate = ringer.eval:main",
        ]
    },
)
