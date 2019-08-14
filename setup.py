""" growcut package configuration """

import numpy
from setuptools import setup, Extension
from Cython.Distutils import build_ext

setup(
    name='growcut',
    version='0.1',
    description='GrowCut - A cellular automata image segmentation.',
    packages=['growcut'],
    cmdclass={'build_ext': build_ext},
    ext_modules=[
        Extension("growcut.growcut_cy", ["growcut/_growcut_cy.pyx"])
        ],
    include_dirs=[numpy.get_include(), ],
    install_requires=['matplotlib', 'scikit-image', 'cython', 'numpy'])