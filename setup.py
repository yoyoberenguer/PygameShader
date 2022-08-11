# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION


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
from Cython.Build import cythonize
from setuptools import Extension
import platform
import warnings
import sys

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

# version 1.0.1 Yank, latest version 1.0.2
# pypitest latest version 1.0.17

OPENMP = False
__VERSION__ = "1.0.8"  # check the file shader.pyx and make sure the version is identical


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


setuptools.setup(
    name="PygameShader",
    version= __VERSION__,       # testing version "1.0.27",
    author="Yoann Berenguer",
    author_email="yoyoberenguer@hotmail.com",
    description="Pygame CPU/GPU shader effects for 2D video game and arcade game",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yoyoberenguer/PygameShader",
    # packages=setuptools.find_packages(),
    packages=['PygameShader'],
    ext_modules=cythonize([
        Extension("PygameShader.shader", ["PygameShader/shader.pyx"],
                  extra_compile_args=ext_compile_args,
                  language="c"),
        Extension("PygameShader.misc", ["PygameShader/misc.pyx"],
                  extra_compile_args=ext_compile_args,
                  language="c"),
        Extension("PygameShader.gaussianBlur5x5", ["PygameShader/gaussianBlur5x5.pyx"],
                  extra_compile_args=ext_compile_args,
                  language="c"),
        Extension("PygameShader.Palette", ["PygameShader/Palette.pyx"],
                  extra_compile_args=ext_compile_args,
                  language="c"),
        Extension("PygameShader.shader_gpu", ["PygameShader/shader_gpu.pyx"],
                  extra_compile_args=ext_compile_args,
                  language="c"),
    ]),

    include_dirs=[numpy.get_include()],
    define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    license='GNU General Public License v3.0',

    classifiers=[  # Optional
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 4 - Beta',

        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        "Operating System :: Microsoft :: Windows",
        "Operating System :: Unix",
        # "Operating System :: MacOS",

        # Pick your license as you wish
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate you support Python 3. These classifiers are *not*
        # checked by 'pip install'. See instead 'python_requires' below.
        # 'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        "Programming Language :: Python :: Implementation :: CPython",
        'Programming Language :: Python',
        'Programming Language :: Cython',
        'Programming Language :: C',

        'Topic :: Software Development :: Libraries :: Python Modules'
        "Topic :: Multimedia :: Graphics",
        "Topic :: Multimedia :: Graphics :: Capture :: Digital Camera",
        "Topic :: Multimedia :: Graphics :: Capture :: Screen Capture",
        "Topic :: Multimedia :: Graphics :: Graphics Conversion",
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
             'PygameShader/Assets/Parrot.jpg'

         ]),
        ('./lib/site-packages/PygameShader/Demo',
         [
             'PygameShader/Demo/cloud_smoke_effect.py',
             'PygameShader/Demo/demo_bloom.py',
             'PygameShader/Demo/demo_fire.py',
             'PygameShader/Demo/demo_glitch.py',
             'PygameShader/Demo/demo_transition.py',
             'PygameShader/Demo/demo_wave.py',
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

