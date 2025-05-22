.. include:: .special.rst

Welcome to PygameShader's documentation!
========================================

:mod:`info.pyx`

=====================

1. Pygame Shader Project
========================

**PygameShader** is a **wrapper around the Pygame library**, written in **Python**, **Cython** and **C**,
designed to add **advanced**
**visual effects** to multimedia applications, including **2D Indie video games, arcade games,
and real-time image processing** for video and camera feeds.

This library extends **Pygame** by enabling developers to apply **shaders** or texture processor to **sprite textures**,
**SDL surfaces, and the entire game display**, significantly enhancing the visual experience with minimal performance overhead.

**Supported Surface/Image Formats**

PygameShader supports multiple image formats, including **BMP, GIF (non-animated), JPEG, and PNG**.
However, Pygame may not always be built with support for all formats. At a minimum, **BMP** is always available.

To check if additional image formats are supported, run:

.. code-block:: python

    import pygame
    print(pygame.image.get_extended())

If it returns ``True``, then **PNG, JPG, and GIF** formats are also supported.

**Performance & Optimization**

PygameShader is optimized for **real-time rendering at 60 FPS**, particularly for **medium resolutions (1024x768)**.
However, performance depends on the complexity of the shader.

Some effects, like **median filtering, predator vision and cartoon effect**, require multiple shaders to create composite
effects, making them computationally more demanding. Most shaders will maintain **smooth performance at medium
resolutions**, but higher resolutions may affect frame rates.

For **sprite texturing and special effects**, PygameShader remains **extremely fast**, thanks to **Cython-based optimizations**.
To maintain a high frame rate, it's recommended to keep SDL sprite texture sizes within **200x200 pixels** to avoid 
unnecessary processing overhead.

**Features**

- **Shader Effects** – Enhance your game's visuals with advanced shading techniques.
- **Real-time Rendering** – Apply effects to the entire game display at high frame rates.
- **Optimized for Speed** – Efficient **Cython** implementation ensures smooth performance.
- **Sprite & Surface Customization** – Modify textures and surfaces dynamically.

PygameShader provides powerful tools to **improve the visual appeal** of your game,
whether by enhancing sprite textures or applying full-screen visual effects.

2. GPU Shaders with CuPy & CUDA
===============================

Some scripts have been ported to the **GPU** using **CuPy** and **CUDA raw kernels**, allowing them to run on
**NVIDIA graphics cards (Compute Capability 3.0 or higher)**.

.. warning::
   These shaders are compatible **only with NVIDIA chipsets**.

**Requirements**

To use GPU shaders, you must have:

- The **CuPy** library installed.
- The **CUDA Toolkit** installed (supported versions: ``v10.2 - v11.6``).

For installation instructions, see the `CuPy installation guide <https://docs.cupy.dev/en/stable/install.html>`_.

**Experimental Status & Performance Considerations**

The **GPU shaders are experimental** and may be **limited by PCI Express bandwidth**,
especially for **real-time rendering**.

For an example of real-time GPU rendering of a ``pygame.Surface``, check out:

.. code-block:: bash

    Demo/GPU_demo_ripple.py

**Comparison: GLSL vs. CUDA**

While **GLSL** outperforms CuPy/CUDA in raw graphics performance, CUDA excels at **accelerating general-purpose parallel computations** originally designed for **CPU architectures** (Python & Cython).

.. list-table::
   :header-rows: 1

   * - Feature
     - GLSL (Shading Language)
     - CUDA
   * - **Primary Use**
     - Graphics Rendering
     - Compute Shaders
   * - **Compilation**
     - Shader Code
     - PTX (NVIDIA Assembly)
   * - **Flexibility**
     - Requires Graphics API
     - Independent Execution

For further insights, check this `GPU programming guide <https://carpentries-incubator.github.io/lesson-gpu-programming/aio/index.html>`_.

**Usage in Python**

To import GPU shaders:

.. code-block:: python

    from PygameShader.shader_gpu import *

**Demo**

To run demos from the ``Demo`` directory (Press ``ESC`` to quit):

**CPU Shader Demos**

.. code-block:: bash

    python demo_fire.py
    python demo_transition.py
    python demo_wave.py

**GPU Shader Demos (Requires CuPy & CUDA)**

.. code-block:: bash

    python gpu_chromatic.py
    python gpu_zoom.py
    python gpu_wave.py

**Easy integration with various library such as Pillow, OpenCV, Scikit-image**


**Pygame**

.. code-block:: python
   :emphasize-lines: 6

   WIDTH = 1280
   HEIGHT = 1024
   SCREEN = pygame.display.set_mode((WIDTH, HEIGHT))
   # load an image
   image = pygame.image.load("../Assets/Aliens.jpg")
   bgr(image)  # use bgr(image, True) for 32-bit images

**Pillow**

.. code-block:: python
   :emphasize-lines: 8

   import PIL
   from PIL import Image
   # load 24-bit image
   im = Image.open("../Assets/background.jpg")
   width, height = im.size
   rgb_array_ = numpy.frombuffer(im.tobytes(), dtype=numpy.uint8)
   arr = rgb_array_.copy()
   bgr_1d(arr)
   image = Image.frombytes('RGB', (width, height), arr)
   image.show()
   # load 32-bit image
   im = Image.open("../Assets/px.png")
   width, height = im.size
   rgb_array_ = numpy.frombuffer(im.tobytes(), dtype=numpy.uint8)
   arr = rgb_array_.copy()
   bgr_1d(arr, True)
   image = Image.frombytes('RGBA', (width, height), arr)
   image.show()

**OpenCV**

.. code-block:: python
   :emphasize-lines: 3

   import cv2
   # load 24-bit image
   img = cv2.imread("../Assets/background.jpg")
   bgr_1d(img)
   cv2.imshow('image', img.astype(numpy.uint8))
   cv2.waitKey(0)
   cv2.destroyAllWindows()
   # load 32-bit image
   img = cv2.imread("../Assets/px.png")
   bgr_1d(img, True)
   cv2.imshow('image', img.astype(numpy.uint8))
   cv2.waitKey(0)
   cv2.destroyAllWindows()

**Scikit-image**

.. code-block:: python
   :emphasize-lines: 8

   import skimage as ski
   import os
   import matplotlib
   from matplotlib import pyplot as plt
   matplotlib.use('TkAgg')
   # load 24-bit image
   filename = '../Assets/background.jpg'
   rgb_array = ski.io.imread(filename)
   bgr_1d(rgb_array)
   plt.imshow(rgb_array)
   plt.show()
   # load 32-bit image
   filename = '../Assets/px.png'
   rgb_array = ski.io.imread(filename)
   bgr_1d(rgb_array, True)
   plt.imshow(rgb_array)
   plt.show()



3. Installation
===============

**Install via pip**

Check the latest version: `PygameShader on PyPI <https://pypi.org/project/PygameShader/>`_

.. code-block:: bash

    pip install PygameShader

**Supported Python Versions:** ``3.6 - 3.11``
**Platforms:** Windows & Linux (``x86``, ``x86_64``)

Check installed version:

.. code-block:: python

    from PygameShader.shader import __VERSION__
    print(__VERSION__)

**Install from Source**

Linux
~~~~~

.. code-block:: bash

    tar -xvf source-1.0.8.tar.gz
    cd PygameShader-1.0.8
    python3 setup.py bdist_wheel
    cd dist
    pip3 install PygameShader-xxxxxx

Windows
~~~~~~~

.. code-block:: bash

    python setup.py bdist_wheel
    cd dist
    pip install PygameShader-xxxxxx

**Building Cython & C Code**

**When Should You Compile Cython Code?**

If you modify any ``.pyx`` or ``.pxd`` files (e.g., ``shader.pyx``, ``shader.pxd``, ``__init__.pxd``),
recompile using:

.. code-block:: bash

    python setup_shader.py build_ext --inplace --force

For a specific Python version, use:

.. code-block:: bash

    python3.8 setup_shader.py build_ext --inplace

If compilation fails, ensure:

- **Cython & a C-compiler** (MSVC, MinGW, GCC) are installed.
- **Python development headers** are available.

For Windows users, refer to this guide:
`VC++ Compiler Setup <https://devblogs.microsoft.com/python/unable-to-find-vcvarsall-bat/>`_.

4. OpenMP for Multi-threaded Processing
=======================================

By default, PygameShader is built with **multi-processing support (OpenMP)**.

To **disable OpenMP**, update ``config.py``:

.. code-block:: python

    # Enable/disable multi-processing
    OPENMP = False
    OPENMP_PROC = "-fopenmp"

Then rebuild:

.. code-block:: bash

    python setup.py bdist_wheel
    cd dist
    pip install PygameShader-xxxxxx.whl

5. Credits
==========

**Author:** Yoann Berenguer

6. Dependencies
===============

.. code-block:: bash

    numpy >= 1.18
    pygame >= 2.4.0
    cython >= 3.0.2
    setuptools ~= 54.1.1
    cupy >= 9.6.0  # (Optional, for GPU shaders)

7. License
==========

**GNU General Public License Version 3 (GPLv3)**

.. code-block:: text

    © 2019 Yoann Berenguer
    © 2007 Free Software Foundation, Inc. <https://fsf.org/>
    Everyone is permitted to copy and distribute verbatim copies
    of this license document, but changing it is not allowed.

8. Running Tests
================

.. code-block:: python

    import PygameShader
    from PygameShader.tests.test_shader import run_testsuite
    run_testsuite()



9. Libraries
============


.. toctree::
   :maxdepth: 2

   Shader
   Shader_gpu
   PygameTools
   RGBConvert
   Sprites
   Palette
   Misc
   GaussianBlur5x5
   BurstSurface
   BlendFlags
   Fire

|

.. currentmodule:: shader

|


