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
# twine upload --repository pypi dist/pygameshader-1.0.11* --verbose

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
    name="pygameshader",
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
                  language=LANGUAGE,
                  define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]),
        Extension("PygameShader.misc", ["PygameShader/misc.pyx"],
                  extra_compile_args=ext_compile_args, extra_link_args=ext_link_args,
                  language=LANGUAGE,
                  define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]),
        Extension("PygameShader.gaussianBlur5x5", ["PygameShader/gaussianBlur5x5.pyx"],
                  extra_compile_args=ext_compile_args, extra_link_args=ext_link_args,
                  language=LANGUAGE,
                  define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]),
        Extension("PygameShader.Palette", ["PygameShader/Palette.pyx"],
                  extra_compile_args=ext_compile_args, extra_link_args=ext_link_args,
                  language=LANGUAGE,
                  define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]),
        Extension("PygameShader.shader_gpu", ["PygameShader/shader_gpu.pyx"],
                  extra_compile_args=ext_compile_args, extra_link_args=ext_link_args,
                  language=LANGUAGE,
                  define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]),
        Extension("PygameShader.BurstSurface", ["PygameShader/BurstSurface.pyx"],
                  extra_compile_args=ext_compile_args, extra_link_args=ext_link_args,
                  language=LANGUAGE,
                  define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]),
        Extension("PygameShader.BlendFlags", ["PygameShader/BlendFlags.pyx"],
                  extra_compile_args=ext_compile_args, extra_link_args=ext_link_args,
                  language=LANGUAGE,
                  define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]),
        Extension("PygameShader.Sprites", ["PygameShader/Sprites.pyx"],
                  extra_compile_args=ext_compile_args, extra_link_args=ext_link_args,
                  language=LANGUAGE,
                  define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]),
        Extension("PygameShader.PygameTools", ["PygameShader/PygameTools.pyx"],
                  extra_compile_args=ext_compile_args, extra_link_args=ext_link_args,
                  language=LANGUAGE,
                  define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]),
        Extension("PygameShader.RGBConvert", ["PygameShader/RGBConvert.pyx"],
                  extra_compile_args=ext_compile_args, extra_link_args=ext_link_args,
                  language=LANGUAGE,
                  define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]),
        Extension("PygameShader.Fire", ["PygameShader/Fire.pyx"],
                  extra_compile_args=ext_compile_args, extra_link_args=ext_link_args,
                  language=LANGUAGE,
                  define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")])
    ], quiet=True),

    include_dirs=[numpy.get_include()],
    license='GNU General Public License v3.0',

    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Intended Audience :: End Users/Desktop",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX :: Linux",
        "Operating System :: POSIX",
        "Operating System :: MacOS",
        "Programming Language :: Cython",
        "Programming Language :: C",
        "Programming Language :: C++",
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Topic :: Software Development :: Libraries :: pygame',
        'Topic :: Software Development :: Build Tools',
    ],

    install_requires=[
        'setuptools>=39.2.1',
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
          'Compilation tips.txt',
          'requirements.txt',
          'PygameShader/__init__.py',
          'PygameShader/__init__.pxd',
          'PygameShader/Dithering.py',
          'PygameShader/setup_shader.py',
          'PygameShader/BlendFlags.pxd',
          'PygameShader/BlendFlags.pyx',
          'PygameShader/shader.pyx',
          'PygameShader/shader.pxd',
          'PygameShader/misc.pyx',
          'PygameShader/misc.pxd',
          'PygameShader/gaussianBlur5x5.pyx',
          'PygameShader/gaussianBlur5x5.pxd',
          'PygameShader/Palette.pyx',
          'PygameShader/Palette.pxd',
          'PygameShader/PygameTools.pyx',
          'PygameShader/PygameTools.pxd',
          'PygameShader/shader_gpu.pyx',
          'PygameShader/shader_gpu.pxd',
          'PygameShader/config.py',
          'PygameShader/BurstSurface.pxd',
          'PygameShader/BurstSurface.pyx',
          'PygameShader/Sprites.pxd',
          'PygameShader/Sprites.pyx',
          'PygameShader/RGBConvert.pyx',
          'PygameShader/RGBConvert.pxd',
          'PygameShader/Fire.pyx',
          'PygameShader/Fire.pxd'
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
             'PygameShader/Assets/px.png',
             'PygameShader/Assets/Aliens.jpg',
             'PygameShader/Assets/AliensLuma.png',
             'PygameShader/Assets/background.jpg',
             'PygameShader/Assets/background2.jpg',
             'PygameShader/Assets/city.jpg',
             'PygameShader/Assets/ES_WaterDrip1.wav',
             'PygameShader/Assets/ES_WaterDrip2.wav',
             'PygameShader/Assets/ES_WaterDrip3.wav',
             'PygameShader/Assets/film_strip2.png',
             'PygameShader/Assets/firepit.ogg',
             'PygameShader/Assets/jump_flight.ogg',
             'PygameShader/Assets/predator.ogg',
             'PygameShader/Assets/vision_swap.ogg',
             'PygameShader/Assets/img.jpg',
             'PygameShader/Assets/Radial4.png',
             'PygameShader/Assets/Radial8.png',
             'PygameShader/Assets/redvignette.png',
             'PygameShader/Assets/space1.jpg',
             'PygameShader/Assets/space1_alpha.jpg',
             'PygameShader/Assets/space2.jpg',
             'PygameShader/Assets/Bokeh__Lens_Dirt_9.jpg',
             'PygameShader/Assets/Bokeh__Lens_Dirt_38.jpg',
             'PygameShader/Assets/Bokeh__Lens_Dirt_46.jpg',
             'PygameShader/Assets/Bokeh__Lens_Dirt_50.jpg',
             'PygameShader/Assets/Bokeh__Lens_Dirt_54.jpg',
             'PygameShader/Assets/Bokeh__Lens_Dirt_67.jpg',
             'PygameShader/Assets/Parrot.jpg',
             'PygameShader/Assets/space2_seamless_alpha.jpg',
             'PygameShader/Assets/space2_seamless.jpg',
             'PygameShader/Assets/space3.jpg',
             'PygameShader/Assets/space5.jpg',
             'PygameShader/Assets/space7.jpg',
             'PygameShader/Assets/teslaColor03_m.png',
             'PygameShader/Assets/alpha.png',

         ]),
        ('./lib/site-packages/PygameShader/Demo',
         [
             'PygameShader/Demo/__init__.py',
             'PygameShader/Demo/cloud_smoke_effect.py',
             'PygameShader/Demo/demo_bloom.py',
             'PygameShader/Demo/demo_bloom_mask.py',
             'PygameShader/Demo/demo_burst.py',
             'PygameShader/Demo/demo_burst_exp.py',
             'PygameShader/Demo/demo_chromatic.py',
             'PygameShader/Demo/demo_fire.py',
             'PygameShader/Demo/demo_fire_border.py',
             'PygameShader/Demo/demo_fisheye.py',
             'PygameShader/Demo/demo_glitch.py',
             'PygameShader/Demo/demo_hsl.py',
             'PygameShader/Demo/demo_hsv.py',
             'PygameShader/Demo/demo_light.py',
             'PygameShader/Demo/demo_magnifier.py',
             'PygameShader/Demo/demo_predator.py',
             'PygameShader/Demo/demo_rain.py',
             'PygameShader/Demo/demo_ripple.py',
             'PygameShader/Demo/demo_ripple1.py',
             'PygameShader/Demo/demo_ripple_seabed.py',
             'PygameShader/Demo/demo_scroll.py',
             'PygameShader/Demo/demo_scroll_32bit.py',
             'PygameShader/Demo/demo_transition.py',
             'PygameShader/Demo/demo_transition_inplace.py',
             'PygameShader/Demo/demo_tunnel.py',
             'PygameShader/Demo/demo_wave.py',
             'PygameShader/Demo/demo_wave_static.py',
             'PygameShader/Demo/demo_zoom.py',
             'PygameShader/Demo/GPU_cartoon.py',
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

