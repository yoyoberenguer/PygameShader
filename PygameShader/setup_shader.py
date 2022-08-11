# encoding: utf-8

from Cython.Build import cythonize

from distutils.core import setup
from distutils.extension import Extension
import platform
import sys
import numpy

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

OPENMP = False
__VERSION__ = "1.0.7"  # check the file shader.pyx and make sure the version is identical


py_requires = "PygameShader requires python3 version 3.6 or above."
py_minor_versions = [x for x in range(6, 11)]

if hasattr(sys, 'version_info'):
    try:
        if not hasattr(sys.version_info, "major") \
                or not hasattr(sys.version_info, "minor"):
            raise AttributeError
        py_major_ver = sys.version_info.major
        py_minor_ver = sys.version_info.minor
    except AttributeError:
        raise SystemExit(py_requires)
else:
    raise SystemExit(py_requires)

if py_major_ver != 3 or py_minor_ver not in py_minor_versions:
    raise SystemExit(
        "PygameShader support python3 versions 3.6 or above got version %s"
        % str(py_major_ver)+"."+str(py_minor_ver))

if hasattr(platform, "architecture"):
    arch = platform.architecture()
    if isinstance(arch, tuple):
        proc_arch_bits = arch[0].upper()
        proc_arch_type = arch[1].upper()
    else:
        raise AttributeError("Platform library is not install correctly")
else:
    raise AttributeError("Platform library is missing attribute <architecture>")


if hasattr(platform, "machine"):
    machine_type = platform.machine().upper()
else:
    raise AttributeError("Platform library is missing attribute <machine>")

if hasattr(platform, "platform"):
    plat = platform.platform().upper()

else:
    raise AttributeError("Platform library is missing attribute <platform>")


if plat.startswith("WINDOWS"):
    ext_compile_args = ["/openmp" if OPENMP else "", "/Qpar", "/fp:fast", "/O2", "/Oy", "/Ot"]

elif plat.startswith("LINUX"):
    if OPENMP:
        ext_compile_args = \
            ["-DPLATFORM=linux", "-march=i686" if proc_arch_bits == "32BIT" else "-march=x86-64",
             "-m32" if proc_arch_bits == "32BIT" else "-m64", "-O3", "-Wall", "-lgomp"]
    else:
        ext_compile_args = \
            ["-DPLATFORM=linux", "-march=i686" if proc_arch_bits == "32BIT" else "-march=x86-64",
             "-m32" if proc_arch_bits == "32BIT" else "-m64", "-O3", "-Wall"]
else:
    raise ValueError("PygameShader can be build on Windows and Linux systems only.")

# /O2 sets a combination of optimizations that optimizes code for maximum speed.
# /Ot (a default setting) tells the compiler to favor optimizations for speed over
# optimizations for size.
# /Oy suppresses the creation of frame pointers on the call stack for quicker function calls.
setup(
    name='SHADER',
    ext_modules=cythonize([
        Extension("shader", ["shader.pyx"],
                  extra_compile_args=ext_compile_args,
                  language="c"),
        Extension("misc", ["misc.pyx"],
                  extra_compile_args=ext_compile_args,
                  language="c"),
        Extension("gaussianBlur5x5", ["gaussianBlur5x5.pyx"],
                  extra_compile_args=ext_compile_args,
                  language="c"),
        Extension("Palette", ["Palette.pyx"],
                  extra_compile_args=ext_compile_args,
                  language="c"),
        Extension("shader_gpu", ["shader_gpu.pyx"],
                  extra_compile_args=ext_compile_args,
                  language="c"),

    ]),


    include_dirs=[numpy.get_include(), '../Include'],

)
