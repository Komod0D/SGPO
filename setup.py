#!/usr/bin/env python

from setuptools import setup, find_packages


setup(name='sgpo',
      version='1.0',
      description='Steered Generation for Protein Optimization (SGPO)',
      author='misc',
      packages=find_packages(include=['sgpo', 'sgpo.*']),
      include_package_data=True,
      install_requires=['importlib_resources']
     )
