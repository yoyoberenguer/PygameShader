# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
# "This lightweight setup file facilitates the rebuilding of PYX files following any user modifications."

import setuptools
try:
    import Cython
except ImportError:
    raise ImportError("\n<Cython> library is missing on your system."
          "\nTry: \n   C:\\pip install Cython")

try:
    import numpy
except ImportError:
    raise ImportError("\n<numpy> library is missing on your system."
          "\nTry: \n   C:\\pip install numpy")

import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
try:
    import pygame
except ImportError:
    raise ImportError("\n<pygame> library is missing on your system."
          "\nTry: \n   C:\\pip install pygame")

__CUPY = True
try:
    import cupy
except ImportError:
    __CUPY = False
    print("\n** CUPY is not installed on your system **")

from Cython.Build import cythonize
from setuptools import Extension
import platform
import warnings
import sys
from config import THREAD_NUMBER, OPENMP, OPENMP_PROC, LANGUAGE, __VERSION__, TEST_VERSION

print("\n---PYTHON COPYRIGHT---\n")
print(sys.copyright)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=ImportWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# NUMPY IS REQUIRED
try:
    import numpy
except ImportError:
    raise ImportError("\n<numpy> library is missing on your system."
                      "\nTry: \n   C:\\pip install numpy on a window command prompt.")


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# version 1.0.1 Yank, latest version 1.0.9
# pypitest latest version 1.0.29

ext_link_args = ""
PythonVerMax = 14
py_requires = "PygameShader requires python3 version 3.6 - %s" % ("3." + str(PythonVerMax - 1))
# If you are building the project from source with a python version
# > python 3.11 you can extend the range to force the build process
# e.g py_minor_versions = [x for x in range(6, 15)] ** compatible until
# python 3.14
py_minor_versions = [x for x in range(6, PythonVerMax)]

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
        "PygameShader support python3 versions 3.6 - %s got version %s"
        % (("3." + str(PythonVerMax - 1)), str(py_major_ver)+"."+str(py_minor_ver)))

if hasattr(platform, "architecture"):
    arch = platform.architecture()
    if isinstance(arch, tuple):
        proc_arch_bits = arch[ 0 ].upper()
        proc_arch_type = arch[ 1 ].upper()
    else:
        raise AttributeError("Platform library is not install correctly")
else:
    raise AttributeError("Platform library is missing attribute <architecture>")

machine_type, plat, system = None, None, None

if hasattr(platform, "machine"):
    machine_type = platform.machine().upper()
else:
    raise AttributeError("Platform library is missing attribute <machine>")

if hasattr(platform, "platform"):
    plat = platform.platform().upper()

else:
    raise AttributeError("Platform library is missing attribute <platform>")

if hasattr(platform, "system"):
    system = platform.system()

release, versioninfo, machine = None, None, None

if plat.startswith("WINDOWS"):

    if hasattr(platform, "win32_ver"):
        if len(platform.win32_ver()) == 3:
            release, versioninfo, machine = platform.win32_ver()

    ext_compile_args = [ "/openmp" if OPENMP else "", "/Qpar", "/fp:fast", "/O2", "/Oy", "/Ot", "/W3" ]



elif plat.startswith("LINUX"):

    if hasattr(platform, "libc_ver"):
        if len(platform.libc_ver()) == 3:
            release, versioninfo, machine = platform.libc_ver()

    if OPENMP:
        ext_compile_args = \
            [ "-DPLATFORM=linux", "-march=i686" if proc_arch_bits == "32BIT" else "-march=x86-64",
              "-m32" if proc_arch_bits == "32BIT" else "-m64", "-O3", "-ffast-math",
              "--param=max-vartrack-size=1500000",
              "-Wall", OPENMP_PROC, "-static" ]
        ext_link_args = [ OPENMP_PROC ]
    else:
        ext_compile_args = \
            [ "-DPLATFORM=linux", "-march=i686" if proc_arch_bits == "32BIT" else "-march=x86-64",
              "-m32" if proc_arch_bits == "32BIT" else "-m64", "-O3", "-ffast-math", "-Wall", "-static",
              "--param=max-vartrack-size=1500000" ]
        ext_link_args = ""

elif plat.startswith("MACOS") or plat.startswith("DARWIN"):

    if hasattr(platform, "mac_ver"):
        if len(platform.mac_ver()) == 3:
            release, versioninfo, machine = platform.mac_ver()

    if OPENMP:
        ext_compile_args = \
            [ "-DPLATFORM=Darwin", "-march=i686" if proc_arch_bits == "32BIT" else "-march=x86-64",
              "-m32" if proc_arch_bits == "32BIT" else "-m64", "-O3", "-ffast-math",
              "-Wall", OPENMP_PROC ]
        ext_link_args = [ OPENMP_PROC ]

    else:
        ext_compile_args = \
            [ "-DPLATFORM=Darwin", "-march=i686" if proc_arch_bits == "32BIT" else "-march=x86-64",
              "-m32" if proc_arch_bits == "32BIT" else "-m64", "-O3", "-ffast-math", "-Wall"
              ]
        ext_link_args = ""

else:
    print("\n---SYSTEM IDENTIFICATION---\n")
    print("SYSTEM                : %s " % system)
    print("SYSTEM VERBOSE        : %s " % plat)
    print("SYS VERSION           : %s " % release if release is not None
          else "SYS VERSION           : undetermined")
    print("BUILD                 : %s " % proc_arch_bits)
    raise ValueError("PygameShader can be build on Windows, Linux and Mac systems only.")


print("\n---COMPILATION---\n")
print("SYSTEM                : %s " % system)
print("SYSTEM VERBOSE        : %s " % plat)
print("SYS VERSION           : %s " % release if release is not None
      else "SYS VERSION           : undetermined")
print("BUILD                 : %s " % proc_arch_bits)
print("FLAGS                 : %s " % ext_compile_args)
print("EXTRA LINK FLAGS      : %s " % ext_link_args if ext_link_args != ""
      else "EXTRA LINK FLAGS      : None")
print("LANGUAGE              : %s " % LANGUAGE)
print("MULTITPROCESSING      : %s " % OPENMP)
print("MULTITPROCESSING FLAG : %s " % OPENMP_PROC if OPENMP is True else
      "MULTITPROCESSING FLAG : %s " % OPENMP_PROC + " ignored")
if OPENMP:
    print("MAX THREADS           : %s " % THREAD_NUMBER)

print("\n")
print("PYTHON VERSION        : %s.%s " % (sys.version_info.major, sys.version_info.minor))
print("SETUPTOOLS VERSION    : %s " % setuptools.__version__)
print("CYTHON VERSION        : %s " % Cython.__version__)
print("NUMPY VERSION         : %s " % numpy.__version__)
print("PYGAME VERSION        : %s " % pygame.__version__)
if __CUPY:
    print("CUPY VERSION          : %s " % cupy.__version__)
else:
    print("CUPY IS NOT INSTALL")



try:
    print("SDL VERSION           : %s.%s.%s " % pygame.version.SDL)
except:
    pass  # ignore SDL versioning issue

if TEST_VERSION:
    print("\n*** BUILDING PYGAMESHADER TESTING VERSION ***  : %s \n" % __VERSION__)
else:
    print("\n*** BUILDING PYGAMESHADER VERSION ***  : %s \n" % __VERSION__)

print(
    "GPU SHADERS INFO"
    "\nIn order to use the GPU shaders the library CUPY has to be installed on your system\n"
    "in addition to CUDA Toolkit: v10.2 / v11.0 / v11.1 / v11.2 / v11.3 / v11.4 / v11.5 / "
    "v11.6 or above.\n"
    "Please refer to the below link for the full installation guide https://docs.cupy.dev/"
    "en/stable/install.html\n"
    "To check if CUDA is installed type nvcc --version in a command prompt.\n")

# define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]),
setuptools.setup(
    name="PygameShader",
    version= __VERSION__,       # testing version "1.0.27",
    author="Yoann Berenguer",
    author_email="yoyoberenguer@hotmail.com",
    description="Pygame effects for 2D video game and arcade game",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yoyoberenguer/PygameShader",
    # packages=setuptools.find_packages(),
    packages=['PygameShader'],
    ext_modules=cythonize(module_list=[
        Extension("shader", ["shader.pyx"],
                  extra_compile_args=ext_compile_args, extra_link_args=ext_link_args,
                  language=LANGUAGE,
                  define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]),
        Extension("misc", ["misc.pyx"],
                  extra_compile_args=ext_compile_args, extra_link_args=ext_link_args,
                  language=LANGUAGE,
                  define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]),
        Extension("gaussianBlur5x5", ["gaussianBlur5x5.pyx"],
                  extra_compile_args=ext_compile_args, extra_link_args=ext_link_args,
                  language=LANGUAGE,
                  define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]),
        Extension("Palette", ["Palette.pyx"],
                  extra_compile_args=ext_compile_args, extra_link_args=ext_link_args,
                  language=LANGUAGE,
                  define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]),
        Extension("shader_gpu", ["shader_gpu.pyx"],
                  extra_compile_args=ext_compile_args, extra_link_args=ext_link_args,
                  language=LANGUAGE,
                  define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]),
        Extension("BurstSurface", ["BurstSurface.pyx"],
                  extra_compile_args=ext_compile_args, extra_link_args=ext_link_args,
                  language=LANGUAGE,
                  define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]),
        Extension("BlendFlags", ["BlendFlags.pyx"],
                  extra_compile_args=ext_compile_args, extra_link_args=ext_link_args,
                  language=LANGUAGE,
                  define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]),
        Extension("Sprites", ["Sprites.pyx"],
                  extra_compile_args=ext_compile_args, extra_link_args=ext_link_args,
                  language=LANGUAGE,
                  define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]),
        Extension("PygameTools", ["PygameTools.pyx"],
                  extra_compile_args=ext_compile_args, extra_link_args=ext_link_args,
                  language=LANGUAGE,
                  define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]),
        Extension("RGBConvert", ["RGBConvert.pyx"],
                  extra_compile_args=ext_compile_args, extra_link_args=ext_link_args,
                  language=LANGUAGE,
                  define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]),
        Extension("Fire", ["Fire.pyx"],
                  extra_compile_args=ext_compile_args, extra_link_args=ext_link_args,
                  language=LANGUAGE,
                  define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")])
    ], quiet=True),


    include_dirs=[ numpy.get_include() ])
