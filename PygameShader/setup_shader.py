# encoding: utf-8

from Cython.Build import cythonize

from distutils.core import setup
from distutils.extension import Extension

import numpy

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# /O2 sets a combination of optimizations that optimizes code for maximum speed.
# /Ot (a default setting) tells the compiler to favor optimizations for speed over
# optimizations for size.
# /Oy suppresses the creation of frame pointers on the call stack for quicker function calls.
setup(
    name='SHADER',
    ext_modules=cythonize([
        Extension("shader", ["shader.pyx"],
                  extra_compile_args=["/openmp", "/Qpar", "/fp:fast", "/O2", "/Oy", "/Ot"],
                  language="c"),
        Extension("misc", ["misc.pyx"],
                  extra_compile_args=["/openmp", "/Qpar", "/fp:fast", "/O2", "/Oy", "/Ot"],
                  language="c"),
        Extension("gaussianBlur5x5", ["gaussianBlur5x5.pyx"],
                  extra_compile_args=["/openmp", "/Qpar", "/fp:fast", "/O2", "/Oy", "/Ot"],
                  language="c")
    ]),


    include_dirs=[numpy.get_include(), '../Include'],

)
