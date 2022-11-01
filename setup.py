#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data and analysis of the furusato nouzei program

@author: Sam Passaglia
"""

from setuptools import setup, find_packages

setup(
    name="furusato",
    version="0.0.1",
    description="",
    author="Sam Passaglia",
    license="MIT",
    packages=find_packages(),
    install_requires=["numpy", "pandas", "japandata", "geopandas"],
)
