from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext


setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = [Extension("atom_", ["atom_.pyx"], include_dirs = [numpy.get_include(), '/usr/local/include'])]
)
