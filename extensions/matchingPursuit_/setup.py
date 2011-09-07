#command line build instruction: python setup.py build_ext --inplace
#to generate html markup: cython source.pyx -a

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np

ext_modules = [Extension("matchingPursuit_", ["matchingPursuit_.pyx"], include_dirs=[np.get_include()])]

setup(cmdclass = {'build_ext': build_ext}, ext_modules = ext_modules)
