# !/usr/bin/env python

from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

setup(
    name='neuralnetwork',
    version='1.0.0',
    description='Neural Network Implementation',
    long_description=readme,
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    keywords=['neural network', 'machine learning', 'perceptron']
)
