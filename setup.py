""" growcut package configuration """

import numpy
from setuptools import setup, Extension
from Cython.Distutils import build_ext

setup(
    name='fastgc',
    version='0.1',
    description='Fast GrowCut with shortest path',
    author='Rui Shen',
    author_email='ruishen@seas.upenn.edu',
    packages=['growcut'],
    cmdclass={'build_ext': build_ext},
    ext_modules=[
        Extension("growcut.growcut_cy", ["growcut/_growcut_cy.pyx"])
        ],
    include_dir=[numpy.get_include()],
    install_requires=[
        'matplotlib',
        'scikit-image',
        'cython',
        'numpy'])
