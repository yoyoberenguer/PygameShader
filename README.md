# Pygame Shaders Library 


*New Version 1.0.10


```
pip install PygameShader==1.0.10
```

<p align="left">
    <img src="https://github.com/yoyoberenguer/PygameShader/blob/main/zoom1.gif?raw=true">
</p>

<p align="left">
    <img src="https://github.com/yoyoberenguer/PygameShader/blob/main/lens.gif?raw=true">
</p>

<p align="left">
    <img src="https://github.com/yoyoberenguer/PygameShader/blob/main/bloom.gif?raw=true">
</p>

<p align="left">
    <img src="https://github.com/yoyoberenguer/PygameShader/blob/main/fire.gif?raw=true">
</p>

<p align="left">
    <img src="https://github.com/yoyoberenguer/PygameShader/blob/main/swirl.gif?raw=true">
</p>

## Version 1.0.10 is out 
Fastest and improved version of **PygameShader**, 15-20% faster CPU algorithms

**New CPU demonstrations available:** 
- demo_burst
- demo_burst_exp (experimental)
- demo_fire_border
- demo_predator
- demo_magnifier
- demo_rain 
- demo_ripple
- demo_transition_inplace
- demo_tunnel
- demo_wave_static

Added following Cython flags to all libraries and methods
```
@cython.profile(False)
@cython.initializedcheck(False)
```
Fast math operations by using C float precision e.g cosf, sinf, atanf etc.<br>
This changes apply only for windows version AMD64 and Win32 (all libraries using libc.math).<br>
Linux versions is still using double precision<br>

Added Fast RGB to HSL color conversion model for a given color
```
cpdef inline hsl _rgb_to_hsl(unsigned char r, unsigned char g, unsigned char b)nogil
cpdef inline rgb _hsl_to_rgb(float h, float s, float l)nogil
```
Added fast RGB to HSV color conversion model for a given color
```
cpdef inline hsv _rgb_to_hsv(unsigned char r, unsigned char g, unsigned char b)nogil
cpdef inline rgb _hsv_to_rgb(float h, float s, float v)nogil
```

## What's changed
Renamed various Cython methods (cdef & cpdef methods) to simplify

Improved and simplified many CPU algorithms. 
Version 1.0.10 is 10-20% faster than 1.0.9

### New BlendFlags library. 
This library is similar to Pygame special flags attribute.<br>
Unlike Pygame, where only surface can be blend together, this library allow you to blend directly 3d arrays(w, h, 3) 
and 2d arrays shape (w, h) together (texture vs texture or alpha channels vs alpha). <br>
Blending array together is much faster than converting both surfaces into equivalent arrays or converting 
arrays into Surfaces to blend pixels together. <br>
Removing unnecessary steps improved the performances and can make your game or code run much faster when you
need to apply transformation to the array level.<br>
```
Added blit_s function (blend a sprite to an image or surface)
Added blend_add_surface (blend surfaces, equivalent to BLEND_RGB_ADD)
Added blend_add_array (blend two 3d arrays together, equivalent to BLEND_RGB_ADD for arrays)
Added blend_add_alpha (blend two 2d arrays together, equivalent to BLEND_RGB_ADD  for alpha channels)
Added blend_sub_surface (blend surfaces, equivalent to BLEND_RGB_SUB)
Added blend_sub_array (same for arrays)
Added blend_min_surface (blend surfaces, equ to BLEND_ADD_MIN)
Added blend_min_array (same for arrays)
Added blend_max_surface (blend surface with BLEND_RGB_MAX flag)
Added blend_max_array (same for arrays)
```
### New BurstSurface library
This library provides new tools to transform PNG & JPG images into multiple sub-surfaces or pixels block.<br>
It contains tools to produce images explosion/burst into pixels or pixel's block, check the demo `demo_burst` and <br>
`demo_burt_exp` (experimental version with _sdl library).<br>
Tools for disassembling or reassembling images from exploded pixels<br>
```
Added pixel_block_rgb (extract sprites from a sprite-sheet) 
Added surface_split used by burst method to decompose an image into pixel blocks)
Added burst (explode a Pygame surface into multiple sub-surface )
Added display_burst (Display an exploded image on the Pygame display)
Added rebuild_from_frame (Rebuild an exploded image)
Added burst_into_memory (Burst image in memory)
Added rebuild_from_memory (Rebuild image from memory)

-- experimental with _sdl library--
Added burst_experimental (explode a surface into multiple sub-surfaces)
Added db_experimental (display burst)
Added rff_experimental (rebuild from a specific frame number)
Added rfm_experimental (rebuild from memory)
Added build_surface_inplace (build a surface from a sprite group inplace)
Added build_surface_inplace_fast (build surface from a sprite group, same than above but faster)
```
### New library Sprites
This is the Pygame sprite module Cythonized 

### Library misc
New algorithm for scrolling surfaces or arrays, check demo `demo_scroll` 
```
Added scroll24 (scroll surface horizontally / vertically)
Added scroll24_inplace (same but inplace)
Added scroll24_arr_inplace (same but for 3d arrays)
Added surface_copy (equivalent tp pygame surface copy)
```
### Library Palette
Changed the palettes arrays to with dataset float32 types<br>
Changed C file Shaderlib.c to perform fast math operations instead of double <br>



---

Some scripts have been ported to GPU using CUPY and CUDA raw Kernels for running 
on NVIDIA graphics cards (NVIDIA CUDA GPU with the compute Capability 3.0 or larger.). 
These shaders are only compatible with NVIDIA chipset.

In order to use the GPU shaders the library `CUPY` has to be 
installed on your system in addition to CUDA Toolkit: 
v10.2 / v11.0 / v11.1 / v11.2 / v11.3 / v11.4 / v11.5 / v11.6. 
Please refer to the below link for the full installation 
https://docs.cupy.dev/en/stable/install.html

The GPU shaders are still experimental, and the performances are restricted
by the pci-express bandwidth due to the amount of data sent from the CPU to 
the GPU, especially when the shaders are use for real time rendering.

Check the Shader GPU_demo_ripple.py in the `Demo folder` for an example of real time 
rendering of a pygame.Surface.

How this shaders compare to GLSL (GL shading language): 
You may ask yourself how fast these shaders perform against the shading language GLSL,
well without any doubt GLSL outperform CUPY & CUDA for graphics performance. 
However, CUDA is taking advantage of its highly parallelized architecture and will improve 
the speed of pygame, python & cython algorithms designed for CPU architecture only.

Difference between shading language and CUDA:
CUDA is essentially just a way to run compute shaders without the graphics API and
without requiring a particular language. CUDA programs are compiled into PTX (NVIDIA's
analoque to x86 assembly for the GPU.

For the curious, please check this excellent post :

https://carpentries-incubator.github.io/lesson-gpu-programming/aio/index.html 

*In python*
```
from PygameShader.shader_gpu import *
```

---


Pygame shader project is a `2D game library` written in Python and Cython containing
`special effects` for development of multimedia applications like video games, arcade game, 
video and camera image processing or to customize your sprites textures/surfaces.

This library is compatible with BMP, GIF (non - animated), JPEG, PNG image format.
```
pygame may not always be built to support all image formats. At minimum it will support 
uncompressed BMP. If pygame.image.get_extended() returns 'True', you should be able to
load most images (including PNG, JPG and GIF).
```

The shaders can be applied to the `entire game display` for a real time rendering @ 60 fps
for games running in medium resolution such as `1024 x 768`. 
Some algorithms are more demanding than others in terms of processing power 
ex : median filtering and predator vision (due to the fact that it is built with more
than one shader to provide a composite effect). Consequently, not all shader will run at
the same speed at medium resolutions. Feel free to experiment with higher display resolutions
while the shader provides 60 fps or above.

If you are using the shader library for sprites texturing and special effects
then the overall processing time should be extremely fast due to code optimization with
cython. Nevertheless, to keep a good frame rate, it is advised to keep the sprites below
the screen display resolution e,g 200x200 texture size.

PygameShader provide tools to improve your overall game appearance by changing 
Sprites texture/surface and or by using great special effects that will affect 
the entire screen. 

The project is under the `GNU GENERAL PUBLIC LICENSE Version 3`

---
## Demo

In the PygameShader `Demo` directory 

(press ESC to quit the demo)

```bash
C:\>python demo_fire.py
C:\>python demo_transition.py
C:\>python demo_wave.py
```
*if cupy and CUDA are installed correctly on your system you can run the GPU shaders*
```bash
C:\>python gpu_chromatic.py
C:\>python gpu_zoom.py
C:\>python gpu_wave.py

```
---

## Installation from pip
Check the link for newest version https://pypi.org/project/PygameShader/

* Available python build 3.6, 3.7, 3.8, 3.9, 3.10, 3.11 and source build
* Compatible WINDOWS and LINUX for platform x86, x86_64
```
pip install PygameShader 
```

* Checking the installed version 
  (*Imported module is case sensitive*) 
```python
>>>from PygameShader.shader import __VERSION__
>>>__VERSION__
```
---
## Installation from source code

*Download the source code and decompress the Tar or zip file*
* ### Linux
```bash
tar -xvf source-1.0.8.tar.gz
cd PygameShader-1.0.8
python3 setup.py bdist_wheel
cd dist 
pip3 install PygameShader-xxxxxx 
```
* ### Windows 

*Decompress the archive and enter PygameShader directory* 
```bash
python setup.py bdist_wheel 
cd dist
pip install PygameShader-xxxxxx
```

---

## Building Cython & C code 

#### When do you need to compile the cython code ? 

Each time you are modifying any of the pyx files such as 
shader.pyx, shader.pxd, __init__.pxd or any external C code if applicable

1) open a terminal window
2) Go in the main project directory where (shader.pyx & 
   shader.pxd files are located)
3) run : `C:\>python setup_shader.py build_ext --inplace --force`

If you have to compile the code with a specific python 
version, make sure to reference the right python version 
in (`python38 setup_shader.py build_ext --inplace`)

If the compilation fail, refers to the requirement section and 
make sure Cython and a C-compiler are correctly install on your
 system.
- A compiler such visual studio, MSVC, CGYWIN setup correctly on 
  your system.
  - a C compiler for windows (Visual Studio, MinGW etc) install 
  on your system and linked to your Windows environment.
  Note that some adjustment might be needed once a compiler is 
  install on your system, refer to external documentation or 
  tutorial in order to setup this process.e.g :
    
https://devblogs.microsoft.com/python/unable-to-find-vcvarsall-bat/

*Edit the file config.py and check the variable OPENMP.*
*You can enable or disable multi-processing*
```python
# Build the cython code with mutli-processing (OPENMP) 
OPENMP : bool = True
```
*Save the change and build the cython code with the following instruction:*
```bash
C:\PygameShader\PygameShader\python setup_shader.py build_ext --inplace --force
````
*If the project build successfully, the compilation will end up with the following lines*
```
Generating code
Finished generating code
```
If you have any compilation error(s) refer to the section ```Building cython code```, make sure 
your system has the following program & libraries installed. Check also that the code is not 
running in a different thread. 
- Python > 3.6
- numpy >= 1.18
- pygame >=2.4.0
- cython >=3.0.2
- setuptools~=54.1.1
- cupy >=9.6.0
- A C compiler for windows (Visual Studio, MinGW etc)
---
## OPENMP for Linux and Windows

The pip packages (including LINUX architectures i686 and x86_64), are build by default with multiprocessing for 
the CPU's shader. If you need to build the package without multiprocessing, you can change the flag OPENMP  
in the setup.py file such as :

To build the package without multiprocessing (OPENMP=False)


*in the config.py file*
```bash
# True enable the multiprocessing
OPENMP : bool = True
OPENMP_PROC : str = "-fopenmp" 
```
*Then compile the code (e.g : Version 1.0.8, 64-bit python3.7)*
```cmdline
C:\PygameShader\python setup.py bdist_wheel 
cd dist
pip install PygameShader-1.0.8-cp37-cp37m-win_amd64.whl
```

*The same variable `OPENMP` exist also in the setup_config.py file when building the Cython code*

* Building PygameShader package will automatically check and compile the source code, you do not 
need to build manually the Cython code.
---

## Credit
Yoann Berenguer 

## Dependencies :
```
numpy >= 1.18
pygame >=2.4.0
cython >=3.0.2
setuptools~=54.1.1
*cupy >=9.6.0  
```
(*) Used for GPU shader (not compulsory during installation). In order to use the GPU shaders 
you would need to have a NVIDIA graphic card, CUDA and CUPY install sucessfully on your platform. 

## License :

GNU GENERAL PUBLIC LICENSE Version 3

Copyright (c) 2019 Yoann Berenguer

Copyright (C) 2007 Free Software Foundation, Inc. <https://fsf.org/>
Everyone is permitted to copy and distribute verbatim copies
of this license document, but changing it is not allowed.


## Testing: 
```python
>>> import PygameShader
>>> from PygameShader.tests.test_shader import run_testsuite
>>> run_testsuite()
```
