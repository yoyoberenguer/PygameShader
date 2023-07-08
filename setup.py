# encoding: utf-8

"""
Setup.py file

Configure the project, build the package and upload the package to PYPI


python_version setup.py sdist bdist_wheel (to include the source)

[TEST PYPI]
repository = https://test.pypi.org/

[PRODUCTION]
repository = https://upload.pypi.org/legacy/
"""
# twine upload --repository testpypi dist/*

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
    print("\n**CUPY is not installed on your system**")

from Cython.Build import cythonize
from setuptools import Extension
import platform
import warnings
import sys

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

OPENMP = True
OPENMP_PROC = "-fopenmp"  # "-lgomp"
__VERSION__ = "1.0.9"  # check the file shader.pyx and make sure the version is identical
LANGUAGE = "c++"
ext_link_args = ""

py_requires = "PygameShader requires python3 version 3.6 or above."
py_minor_versions = [x for x in range(6, 12)]

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
             "-m32" if proc_arch_bits == "32BIT" else "-m64", "-O3",
             "-Wall", OPENMP_PROC, "-static"]
        ext_link_args = [OPENMP_PROC]
    else:
        ext_compile_args = \
            ["-DPLATFORM=linux", "-march=i686" if proc_arch_bits == "32BIT" else "-march=x86-64",
             "-m32" if proc_arch_bits == "32BIT" else "-m64", "-O3", "-Wall", "-static"]
        ext_link_args = ""
else:
    raise ValueError("PygameShader can be build on Windows and Linux systems only.")


print("\n---COMPILATION---\n")
print("SYSTEM                : %s " % plat)
print("BUILD                 : %s " % proc_arch_bits)
print("FLAGS                 : %s " % ext_compile_args)
print("EXTRA LINK FLAGS      : %s " % ext_link_args)
print("LANGUAGE              : %s " % LANGUAGE)
print("MULTITPROCESSING      : %s " % OPENMP)
print("MULTITPROCESSING FLAG : %s " % OPENMP_PROC)

print("\n")
print("PYTHON VERSION        : %s.%s " % (sys.version_info.major, sys.version_info.minor))
print("SETUPTOOLS VERSION    : %s " % setuptools.__version__)
print("CYTHON VERSION        : %s " % Cython.__version__)
print("NUMPY VERSION         : %s " % numpy.__version__)
print("PYGAME VERSION        : %s " % pygame.__version__)
if __CUPY:
    print("CUPY VERSION          : %s " % cupy.__version__)
try:
    print("SDL VERSION           : %s.%s.%s " % pygame.version.SDL)
except:
    pass  # ignore SDL versioning issue

print("\n*** BUILDING PYGAMESHADER VERSION ***  : %s \n" % __VERSION__)

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
        Extension("PygameShader.shader", ["PygameShader/shader.pyx"],
                  extra_compile_args=ext_compile_args, extra_link_args=ext_link_args,
                  language=LANGUAGE),
        Extension("PygameShader.misc", ["PygameShader/misc.pyx"],
                  extra_compile_args=ext_compile_args, extra_link_args=ext_link_args,
                  language=LANGUAGE),
        Extension("PygameShader.gaussianBlur5x5", ["PygameShader/gaussianBlur5x5.pyx"],
                  extra_compile_args=ext_compile_args, extra_link_args=ext_link_args,
                  language=LANGUAGE),
        Extension("PygameShader.Palette", ["PygameShader/Palette.pyx"],
                  extra_compile_args=ext_compile_args, extra_link_args=ext_link_args,
                  language=LANGUAGE),
        Extension("PygameShader.shader_gpu", ["PygameShader/shader_gpu.pyx"],
                  extra_compile_args=ext_compile_args, extra_link_args=ext_link_args,
                  language=LANGUAGE)
    ]),

    include_dirs=[numpy.get_include()],
    license='GNU General Public License v3.0',

    classifiers=[
        'Development Status :: 4 - Beta',

        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        "Operating System :: Microsoft :: Windows",

        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',

        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10'
    ],

    install_requires=[
        'setuptools>=49.2.1',
        'Cython>=0.28',
        'numpy>=1.18',
        'pygame>=2.0'
        # cupy

    ],
    python_requires='>=3.6',
    platforms=['Windows'],
    include_package_data=True,
    data_files=[
        ('./lib/site-packages/PygameShader',
         ['LICENSE',
          'MANIFEST.in',
          'pyproject.toml',
          'README.md',
          'requirements.txt',
          'PygameShader/__init__.py',
          'PygameShader/__init__.pxd',
          'PygameShader/setup_shader.py',
          'PygameShader/shader.pyx',
          'PygameShader/shader.pxd',
          'PygameShader/misc.pyx',
          'PygameShader/misc.pxd',
          'PygameShader/gaussianBlur5x5.pyx',
          'PygameShader/Palette.pyx',
          'PygameShader/shader_gpu.pyx',
          'PygameShader/shader_gpu.pxd'
          ]),
        ('./lib/site-packages/PygameShader/Include',
         ['PygameShader/Include/Shaderlib.c'
          ]),
        ('./lib/site-packages/PygameShader/tests',
         [
             'PygameShader/tests/test_shader.py',
             'PygameShader/tests/__init__.py',
          ]),
        ('./lib/site-packages/PygameShader/Assets',
         [
             'PygameShader/Assets/Aliens.jpg',
             'PygameShader/Assets/background.jpg',
             'PygameShader/Assets/background2.jpg',
             'PygameShader/Assets/city.jpg',
             'PygameShader/Assets/ES_WaterDrip1.wav',
             'PygameShader/Assets/ES_WaterDrip2.wav',
             'PygameShader/Assets/ES_WaterDrip3.wav',
             'PygameShader/Assets/img.png',
             'PygameShader/Assets/Radial4.png',
             'PygameShader/Assets/Radial8.png',
             'PygameShader/Assets/redvignette.png',
             'PygameShader/Assets/space1.jpg',
             'PygameShader/Assets/space2.jpg',
             'PygameShader/Assets/Bokeh__Lens_Dirt_9.jpg',
             'PygameShader/Assets/Bokeh__Lens_Dirt_38.jpg',
             'PygameShader/Assets/Bokeh__Lens_Dirt_46.jpg',
             'PygameShader/Assets/Bokeh__Lens_Dirt_50.jpg',
             'PygameShader/Assets/Bokeh__Lens_Dirt_54.jpg',
             'PygameShader/Assets/Bokeh__Lens_Dirt_67.jpg',
             'PygameShader/Assets/Parrot.jpg',
             'PygameShader/Assets/space2_seamless_alpha.jpg'

         ]),
        ('./lib/site-packages/PygameShader/Demo',
         [
             'PygameShader/Demo/cloud_smoke_effect.py',
             'PygameShader/Demo/demo_chromatic.py',
             'PygameShader/Demo/demo_bloom.py',
             'PygameShader/Demo/demo_fire.py',
             'PygameShader/Demo/demo_glitch.py',
             'PygameShader/Demo/demo_transition.py',
             'PygameShader/Demo/demo_wave.py',
             'PygameShader/Demo/demo_zoom.py',
             'PygameShader/Demo/GPU_chromatic.py',
             'PygameShader/Demo/GPU_demo_ripple.py',
             'PygameShader/Demo/GPU_fisheye.py',
             'PygameShader/Demo/GPU_hsl.py',
             'PygameShader/Demo/GPU_light.py',
             'PygameShader/Demo/GPU_wave.py',
             'PygameShader/Demo/GPU_zoom.py'
         ])
    ],

    project_urls={  # Optional
        'Bug Reports': 'https://github.com/yoyoberenguer/PygameShader/issues',
        'Source': 'https://github.com/yoyoberenguer/PygameShader',
    },
)

