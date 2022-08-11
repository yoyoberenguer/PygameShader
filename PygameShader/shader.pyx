# cython: binding=False, boundscheck=False, wraparound=False, nonecheck=False, cdivision=True,
# cython: optimize.use_switch=True
# encoding: utf-8

"""
                 GNU GENERAL PUBLIC LICENSE
                       Version 3, 29 June 2007

 Copyright (C) 2007 Free Software Foundation, Inc. <https://fsf.org/>
 Everyone is permitted to copy and distribute verbatim copies
 of this license document, but changing it is not allowed.

Copyright Yoann Berenguer
"""


"""
Version 1.0.1 (yank)
+ new cartoon effect added to library for offline surface processing. This shader cannot  
  be used in real time due to the amount of transformation. 

Version 1.0.2 same than 1.0.1
+ new cartoon effect added to library for offline surface processing. This shader cannot  
  be used in real time due to the amount of transformation. 


Version 1.0.3 
++ New transition effect (blend effect) 

++ BUG CORRECTED IN rgb_split
#         unsigned char [:] red   = numpy.empty(z, uint8, order='C')
#         unsigned char [:] green = numpy.empty(z, uint8, order='C')
#         unsigned char [:] blue  = numpy.empty(z, uint8, order='C')
# TO
#         unsigned char [:] red   = numpy.zeros(z, uint8, order='C')
#         unsigned char [:] green = numpy.zeros(z, uint8, order='C')
#         unsigned char [:] blue  = numpy.zeros(z, uint8, order='C')

# ++ Predator mode has now a blending mode
# int blend = pygame.BLEND_RGB_ADD
# Also change declaration in pxd file and pyx bint inv_colormap and bint fast
cpdef predator_vision(object surface_, unsigned int sobel_threshold=*,
                           unsigned int bpf_threshold=*, unsigned int bloom_threshold=*,
                           bint inv_colormap=*, bint fast=*, int blend=*)
                           
++ Renamed all the methods with a quicker and easier name to remember

++ corrected algos swirl and swirl2 (divided r by r_max)

++ new algorithm dirt_lens effect

++ Change brightness and brightness_exclude, no change if shift_ == 0 

++ Changed /corrected bug with color_reduction algorithm.

++ Added Dithering algorithm 
++ Added Color_palette
++ added bilateral filter 
++ added emboss filter
++ pixelation 
++ convert to 27 colors
++ spectrum surface
++ horizontal gradient (unit misc.pyx)

++ create misc.pxd file

"""


"""
VERSION 1.0.5
Bug correction in shader_median_filter24_inplace_c 
The value returned previously was a minimal value and not the median value
rgb_array_[i, j, 0] = tmpr[(index -1) >> 1]
rgb_array_[i, j, 1] = tmpg[(index -1) >> 1]
rgb_array_[i, j, 2] = tmpb[(index -1) >> 1]

Bug correction in shader_plasma24bit_inplace_c  (replace t by t_)
if i == 0:
    r, g, b =  v, t_, p
if i == 1:
     r, g, b = q, v, p
if i == 2:
     r, g, b = p, v, t_
if i == 3:
     r, g, b = p, q, v
if i == 4:
     r, g, b = t_, p, v
if i == 5:
     r, g, b = v, p, q

++ Added bilinear filter 
++ Added alpha_blending
++ Light effect 24 bit
++ added shader_bloom_fast (much faster)
++ bug corrected in fisheye lens (different width and height would have cause an exception)
"""


""" 
VERSION 1.0.6 
same than 1.0.6 
+ removed the cupy import from the __init__.py file to avoid user not being able to 
use CPU shader without installing CUPY and CUDA

"""

"""
`Version 1.0.7`

new GPU rgb_split_gpu
new GPU chromatic 
new GPU zoom 

improved bloom effect for moving objects and particles, 
this version is also faster (less sub-surface used to generate the bloom)
new CPU shader_bloom_fast1  (works for moving objects) 
new CPU chromatic 
new CPU zoom

Removed CPU median_avg 
"""


"""
`Version 1.0.8`
New GPU wavelength_map_gpu
New GPU heatmap_gpu
new CPU shader_rgb_to_yiq_inplace
new CPU shader_rgb_to_yiq_i_comp_inplace
new CPU shader_rgb_to_yiq_q_comp_inplace

ADDED CPU METHOD FOR 
create_horizontal_gradient_1d
create_horizontal_gradient_1d_alpha
object horizontal_grad3d
object horizontal_grad3d_alpha
create_radial_gradient
create_radial_gradient_alpha
create_quarter_radial_gradient
create_quarter_radial_gradient_alpha

Removed fast cpdef object bilateral_fast_gpu(surface_, unsigned int kernel_size_)
Removed cdef bilateral_fast_cupy(gpu_array_, unsigned int kernel_size_)

"""


__VERSION__ = "1.0.8"

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=ImportWarning)

try:
    import numpy
    from numpy import empty, uint8, int16, float32, asarray, linspace, \
        ascontiguousarray, zeros, uint16, uint32, int32, int8
except ImportError:
    raise ImportError("\n<numpy> library is missing on your system."
          "\nTry: \n   C:\\pip install numpy on a window command prompt.")

cimport numpy as np

try:
    cimport cython
    from cython.parallel cimport prange

except ImportError:
    raise ImportError("\n<cython> library is missing on your system."
          "\nTry: \n   C:\\pip install cython on a window command prompt.")

# PYGAME IS REQUIRED
try:
    import pygame
    from pygame import Color, Surface, SRCALPHA, RLEACCEL, BufferProxy, HWACCEL, HWSURFACE, \
    QUIT, K_SPACE, BLEND_RGB_ADD, Rect, BLEND_RGB_MAX, BLEND_RGB_MIN
    from pygame.surfarray import pixels3d, array_alpha, pixels_alpha, array3d, \
        make_surface, blit_array, pixels_red, \
    pixels_green, pixels_blue
    from pygame.image import frombuffer, fromstring, tostring
    from pygame.math import Vector2
    from pygame import _freetype
    from pygame._freetype import STYLE_STRONG, STYLE_NORMAL
    from pygame.transform import scale, smoothscale, rotate, scale2x
    from pygame.pixelcopy import array_to_surface

except ImportError:
    raise ImportError("\n<Pygame> library is missing on your system."
          "\nTry: \n   C:\\pip install pygame on a window command prompt.")

try:
    cimport cython
    from cython.parallel cimport prange
    from cpython cimport PyObject_CallFunctionObjArgs, PyObject, \
        PyList_SetSlice, PyObject_HasAttr, PyObject_IsInstance, \
        PyObject_CallMethod, PyObject_CallObject
    from cpython.dict cimport PyDict_DelItem, PyDict_Clear, PyDict_GetItem, PyDict_SetItem, \
        PyDict_Values, PyDict_Keys, PyDict_Items
    from cpython.list cimport PyList_Append, PyList_GetItem, PyList_Size, PyList_SetItem
    from cpython.object cimport PyObject_SetAttr

except ImportError:
    raise ImportError("\n<cython> library is missing on your system."
          "\nTry: \n   C:\\pip install cython on a window command prompt.")

from PygameShader.gaussianBlur5x5 import canny_blur5x5_surface24_c
from PygameShader.misc cimport color_diff_hsv, color_diff_hsl, close_color

cimport numpy as np

from libc.stdlib cimport rand, malloc
from libc.math cimport sqrt, atan2, sin, cos, nearbyint, exp, pow, floor
from libc.stdlib cimport malloc, free
from libc.math cimport round as round_c, fmin, fmax

from libc.stdio cimport printf

cdef float M_PI = 3.14159265358979323846
cdef float M_PI2 =3.14159265358979323846/2.0
cdef float M_2PI =2 * 3.14159265358979323846

cdef float RAD_TO_DEG=<float>(180.0/M_PI)
cdef float DEG_TO_RAD=<float>(M_PI/180.0)

cdef float C1 = <float>7.0/<float>16.0
cdef float C2 = <float>3.0/<float>16.0
cdef float C3 = <float>5.0/<float>16.0
cdef float C4 = <float>1.0/<float>16.0

cdef int THREADS = 8

DEF HALF         = 1.0/2.0
DEF ONE_THIRD    = 1.0/3.0
DEF ONE_FOURTH   = 1.0/4.0
DEF ONE_FIFTH    = 1.0/5.0
DEF ONE_SIXTH    = 1.0/6.0
DEF ONE_SEVENTH  = 1.0/7.0
DEF ONE_HEIGHT   = 1.0/8.0
DEF ONE_NINTH    = 1.0/9.0
DEF ONE_TENTH    = 1.0/10.0
DEF ONE_ELEVENTH = 1.0/11.0
DEF ONE_TWELVE   = 1.0/12.0
DEF ONE_32       = 1.0/32.0
DEF ONE_64       = 1.0/64.0
DEF ONE_128      = 1.0/128.0
DEF ONE_255      = 1.0/255.0
DEF ONE_360      = 1.0/360.0
DEF TWO_THIRD    = 2.0/3.0

cdef float[360] COS_TABLE
cdef float[360] SIN_TABLE
cdef int ANGLE
for ANGLE in range(0, 360):
    COS_TABLE[ANGLE] = <float>cos(<float>(ANGLE * DEG_TO_RAD))
    SIN_TABLE[ANGLE] = <float>sin(<float>(ANGLE * DEG_TO_RAD))

COLORS_CPC64 = numpy.array(
    [[0, 0, 0],
    [0, 0, 128],
    [0, 0, 255],
    [128, 0, 0],
    [128, 0, 128],
    [128, 0, 255],
    [255, 0, 0],
    [255, 0, 128],
    [255, 0, 255],
    [0, 128, 0],
    [0, 128, 128],
    [0, 128, 255],
    [128, 128, 0],
    [128, 128, 128],
    [128, 128, 255],
    [255, 128, 0],
    [255, 128, 128],
    [255, 128, 255],
    [0, 255, 0],
    [0, 255, 128],
    [0, 255, 255],
    [128, 255, 0],
    [128, 255, 128],
    [128, 255, 255],
    [255, 255, 0],
    [255, 255, 128],
    [255, 255, 255]], dtype=numpy.uint8)

cdef:
    float [:, :] COLORS_CPC64_C = numpy.divide(COLORS_CPC64, 255.0).astype(dtype=float32)



cpdef tuple rgb_2_yiq(unsigned char r, unsigned char g, unsigned char b):
    """
    CONVERT RGB VALUES INTO YIQ COLOR MODEL 
    
    * formulas NTSC 1953 
    
    :param r: integer; (unsigned char) red value in range [0.255]
    :param g: integer; (unsigned char) green value in range [0.255]
    :param b: integer; (unsigned char) blue value in range [0.255]
    :return: tuple representing the color in YIQ color model
    """
    cdef yiq yiq_
    yiq_ = rgb_to_yiq(r/<float>255.0, g/<float>255.0, b/<float>255.0)
    return yiq_.y, yiq_.i, yiq_.q

cpdef tuple yiq_2_rgb(float y, float i, float q):
    """
    CONVERT YIQ COLOR MODEL INTO EQUIVALENT RGB VALUES
    * NTSC 1953
    
    :param y: float; LUMA
    :param i: float; I stands for in-phase
    :param q: float; Q stands for quadrature, 
    :return: tuple representing the RGB values [0...255] unsigned char values
    """
    cdef rgb rgb_

    rgb_ = yiq_to_rgb(y, i, q)
    return <unsigned char>(rgb_.r * <float>255.0), \
           <unsigned char>(rgb_.g * <float>255.0), \
           <unsigned char>(rgb_.b * <float>255.0)


cpdef inline void rgb_to_bgr(object surface_):
    """  
    SHADER RGB to BGR
  
    Convert your game display from RGB to BGR format
    This algorithm can also be used to transform pygame texture in the equivalent bgr format
    
    * The changes are automatically applied inplace to the surface you do not need to create a 
    new surface.

    e.g:
    rgb_to_bgr(surface)

    :param surface_    : Pygame surface or display surface compatible (image 24-32 bit with or 
                         without per-pixel transparency / alpha channel)
    :return             : void
    """
    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)

    try:
        rgb_array = pixels3d(surface_)

    except Exception as e:
        raise ValueError("Cannot reference source pixels into a 3d array.\n %s " % e)


    shader_rgb_to_bgr_inplace_c(rgb_array)


cpdef inline void rgb_to_brg(object surface_):
    """
    SHADER RGB TO BRG

    Convert your game display from RGB to BRG format.
    This algorithm can also be used to transform pygame texture in the equivalent BRG format
    
    * The changes are automatically applied inplace to the surface you do not need to create a 
      new surface.
    
    e.g:
    rgb_to_brg(surface)

    :param surface_: Pygame surface or display surface compatible (image 24-32 bit with or without 
                     per-pixel transparency / alpha channel)
    :return: void 
    """
    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)
    try:
        rgb_array = pixels3d(surface_)

    except Exception as e:
        raise ValueError("Cannot reference source pixels into a 3d array.\n %s " % e)

    shader_rgb_to_brg_inplace_c(rgb_array)

cpdef inline void greyscale(object surface_):
    """
    SHADER GRAYSCALE (CONSERVE LUMINOSITY)

    This shader transform the game display on a grayscale video game effect
    
    * This shader can also be applied to pygame textures/surface to transform them into
      an equivalent grayscale model
    * The changes are automatically applied inplace to the surface you do not need to create a 
      new surface.

    e.g:
    greyscale(surface)

    :param surface_  : Pygame surface or display surface compatible (image 24-32 bit with 
                       or without per-pixel transparency / alpha channel)
    :return          : void
    
    """
    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)

    try:
        rgb_array = pixels3d(surface_)

    except Exception as e:
        raise ValueError("Cannot reference source pixels into a 3d array.\n %s " % e)

    shader_greyscale_luminosity24_inplace_c(rgb_array)


cpdef inline void sepia(object surface_):
    """
    SHADER SEPIA MODEL

    Transform your video game into an equivalent sepia model
    
    * The changes are automatically applied inplace to the surface you do not need to create a 
      new surface.

    e.g:
    sepia(surface)


    :param surface_  : Pygame surface or display surface compatible (image 24-32 bit with 
                       or without per-pixel transparency / alpha channel)
    :return:         : void
    """
    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)

    try:
        rgb_array = pixels3d(surface_)

    except Exception as e:
        raise ValueError("Cannot reference source pixels into a 3d array.\n %s " % e)

    shader_sepia24_inplace_c(rgb_array)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline void median_fast(
        object surface_,
        unsigned short int kernel_size_=2,
        unsigned short int reduce_factor_=1
):
    """
    This function cannot be called directly from python script (cdef)

    :param surface_: pygame.surface; Surface compatible 24-32 bit 
    :param kernel_size_: integer; size of the kernel 
    :param reduce_factor_: integer; value of 1 divide the image by 2, value of 2 div the image by 4
    :return: void
    """
    try:
        rgb_array = pixels3d(surface_)

    except Exception as e:
        raise ValueError("Cannot reference source pixels into a 3d array.\n %s " % e)


    surface_cp = surface_.copy()
    cdef:
        int w, h
    w, h = surface_cp.get_size()

    surface_cp = smoothscale(surface_cp, (w >> reduce_factor_, h >> reduce_factor_))

    try:
        cp_array = pixels3d(surface_cp)

    except Exception as e:
        raise ValueError("Cannot reference source pixels into a 3d array.\n %s " % e)

    cdef:
        int i, j
        unsigned char[:, :, :] org_surface = rgb_array
        unsigned char[:, :, :] surface_cp_arr = cp_array

    shader_median_filter24_inplace_c(surface_cp_arr, kernel_size_)
    surface_cp_arr = scale_array24_c(surface_cp_arr, w, h)

    with nogil:
        for i in prange(w, schedule='static', num_threads=THREADS):
            for j in range(h):
                org_surface[i, j, 0] = surface_cp_arr[i, j, 0]
                org_surface[i, j, 1] = surface_cp_arr[i, j, 1]
                org_surface[i, j, 2] = surface_cp_arr[i, j, 2]


cpdef inline void median(
        object surface_,
        unsigned short int kernel_size_=2,
        bint fast_=True,
        unsigned short int reduce_factor_=1
) except *:
    """
    SHADER MEDIAN FILTER COMPATIBLE 24-32 bit IMAGE

    This shader cannot be used for real time display rendering as the performance 
    of the algorithm is not satisfactory < 50 fps. 
    The code would have to be changed and improved with C or assembler in order to
    be adapted for a real time application. 

    In the state, this shader can be used for texture/surface transformation offline
    
    * The changes are automatically applied inplace to the surface you do not need to create a 
      new surface.  

    :param surface_      : Pygame surface or display surface compatible (image 24-32 bit with 
                           or without per-pixel transparency / alpha channel)
    :param kernel_size_  : integer; Kernel size or neighbourhood pixels to be included default is 2
                           Increase the effect with kernel size > 2 (effect overall speed is 
                           degrading quickly with large kernel size e.g > 2)
    :param fast_         : boolean; Flag for fast calculation (default True). Improve overall speed 
                           performance by using smaller texture sizes (see reduce_factor_ option)
    :param reduce_factor_: integer; Int value to reduce the size of the original surface to 
                           process. A value of 1, divide the original surface by 2 and a value of 2
                           reduce the surface by 4 (value of 1 and 2 are acceptable, over 2 the 
                           image quality is too pixelated and blur) default value is 1 (div by 2).
                           This argument as no effect if flag fast_=False
    :return:             : void
    """
    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)

    if kernel_size_ <= 0:
        raise ValueError('\nArgument kernel_size_ cannot be <= 0')
    if not 0 < reduce_factor_ < 9:
        raise ValueError('\nArgument reduce_factor_ must be in range [1 ... 8] ')

    if fast_:
        median_fast(surface_, kernel_size_, reduce_factor_)
    else:
        try:
            rgb_array = pixels3d(surface_)

        except Exception as e:
            raise ValueError("Cannot reference source pixels into a 3d array.\n %s " % e)

        shader_median_filter24_inplace_c(rgb_array, kernel_size_)


cpdef inline void median_grayscale(
        object surface_,
        int kernel_size_=2
):
    """
    SHADER MEDIAN FILTER (GRAYSCALE)

    This shader cannot be used for real time rendering as the performance of the algorithm are not
    satisfactory. The code would have to be changed and improved with C or assembler in order to
    be adapted for a real time application. 

    In the state, this shader can be used for texture/surface transformation offline

    The surface is compatible 24 - 32 bit with or without alpha layer
    
    * The changes are automatically applied inplace to the surface you do not need to create a 
      new surface.  

    :param surface_: pygame.Surface; compatible 24 - 32 bit with or without alpha layer
    :param kernel_size_: integer; Kernel size (must be > 0), default value = 2
    :return: void 
    """
    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)

    assert kernel_size_ > 0, "\nArgument kernel_size_ cannot be <= 0"
    try:
        rgb_array = pixels3d(surface_)

    except Exception as e:
        raise ValueError("Cannot reference source pixels into a 3d array.\n %s " % e)

    shader_median_grayscale_filter24_inplace_c(rgb_array, kernel_size_)




cpdef inline void color_reduction(
        object surface_,
        int color_=8
):
    """
     COLOR REDUCTION SHADER

    Decrease the amount of colors in the display or texture.
    The method of color reduction is very simple: every color of the original picture is replaced
    by an appropriate color from the limited palette that is accessible.
    
    The surface is compatible 24 - 32 bit with or without alpha layer
    
    * The changes are automatically applied inplace to the surface you do not need to create a 
      new surface.  
      
    e.g:
    color_reduction(surface, 8)

    :param surface_: pygame.Surface; compatible 24 - 32 bit 
    :param color_: integer must be > 0 default 8
    :return: void 
    """
    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)

    assert color_ > 0, "Argument color_number must be > 0"

    try:
        rgb_array = pixels3d(surface_)

    except Exception as e:
        raise ValueError("Cannot reference source pixels into a 3d array.\n %s " % e)

    shader_color_reduction24_inplace_c(rgb_array, color_)


cpdef inline void sobel(
        object surface_,
        int threshold_ = 64
):
    """
    SHADER SOBEL (EDGE DETECTION)

    Transform the game display or a pygame surface into a sobel equivalent model
    (surface edge detection)

    The surface is compatible 24 - 32 bit with or without alpha layer
    
    * The changes are automatically applied inplace to the surface, you do not need to create a 
      new surface.  

    e.g:
    sobel(surface, 64)

    :param surface_: pygame.Surface; compatible 24 - 32 bit 
    :param threshold_: integer; Value for detecting the edges default 64
    :return:
    """
    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)

    assert -1 < threshold_ < 256, "\nArgument threshold must be an integer in range [0 ... 255]"

    try:
        rgb_array = pixels3d(surface_)

    except Exception as e:
        raise ValueError("Cannot reference source pixels into a 3d array.\n %s " % e)

    shader_sobel24_inplace_c(rgb_array, threshold_)


cpdef inline void sobel_fast(
        object surface_,
        int threshold_ = 64,
        unsigned short factor_ = 1
):
    """
    SHADER FAST SOBEL (EDGE DETECTION)

    Transform the game display or a pygame surface into a sobel equivalent model
    (surface edge detection).This version is slightly fastest than shader_sobel24_inplace_c as
    it down-scale the array containing all the pixels and apply the sobel algorithm to a smaller
    sample. When the processing is done, the array is re-scale to its original dimensions.
    If this method is in theory faster than shader_sobel24_inplace_c, down-scaling and up-scaling
    an array does have a side effect of decreasing the overall image definition
    (jagged lines non-antialiasing)
    
    Compatible 24 - 32 bit with or without alpha layer
    
    * The changes are automatically applied inplace to the surface, you do not need to create a 
      new surface.  
      
    e.g:
    sobel_fast(surface, 64, factor_=1)

    :param surface_: pygame.surface compatible 24-32 bit 
    :param threshold_: integer; default value is 24 
    :param factor_: integer; default value is 1 (div by 2)
    :return:
    """
    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)

    assert -1 < threshold_ < 256, "\nArgument threshold must be an integer in range [0 ... 255]"
    assert 0 < factor_ < 9, "\nArgument factor_ must be in range [1 ... 8]"

    shader_sobel24_fast_inplace_c(surface_, threshold_, factor_)




cpdef inline void invert(object surface_):
    """
    SHADER INVERT PIXELS
    
    Invert all pixels of the display or a given texture
    
    Compatible 24 - 32 bit with or without alpha layer
    
    * The changes are automatically applied inplace to the surface, you do not need to create a 
      new surface.  
    
    e.g:
    invert(surface)
    
    :param surface_: pygame.surface; compatible 24 - 32 bit surfaces
    :return: void
    """
    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)

    try:
        rgb_array = pixels3d(surface_)

    except Exception as e:
        raise ValueError("Cannot reference source pixels into a 3d array.\n %s " % e)

    shader_invert_surface_24bit_inplace_c(rgb_array)



cpdef inline void hsl_effect(object surface_, float shift_):
    """
    ROTATE THE HUE OF THE GAME DISPLAY OR GIVEN TEXTURE
    
    Compatible 24 - 32 bit with or without alpha layer
    
    * The changes are automatically applied inplace to the surface, you do not need to create a 
      new surface.  
      
    e.g:
    hsl_effect(surface, 0.2)
    
    :param surface_: pygame.Surface; Compatible 24 - 32 bit surfaces
    :param shift_: float; float value in range [-1.0 ... 1.0]
    :return: void 
    """
    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)

    assert -1.0 <= shift_ <= 1.0, \
        "Argument shift must be in range[-1.0 ... 1.0]"

    try:
        rgb_array = pixels3d(surface_)

    except Exception as e:
        raise ValueError("Cannot reference source pixels into a 3d array.\n %s " % e)

    shader_hsl_surface24bit_inplace_c(rgb_array, shift_)


cpdef inline void hsl_fast(
        object surface_,
        float shift_,
        float [:, :, :, ::1] hsl_model_,
        unsigned char [:, :, :, ::1] rgb_model_
):
    """    
    ROTATE THE HUE OF AN IMAGE USING STORED HSL TO RGB AND RGB TO HSL VALUES
    
    Compatible 24 - 32 bit with or without alpha layer
    
    * The changes are automatically applied inplace to the surface, you do not need to create a 
      new surface.  

    e.g:
    rgb2hsl_model = hsl_to_rgb_model()
    hsl2rgb_model = rgb_to_hsl_model()
    hsl_fast(
                image,
                0.1,
                hsl_model_=hsl2rgb_model,
                rgb_model_=rgb2hsl_model)

    :param surface_: pygame.Surface; compatible 24 - 32 bit surfaces
    :param shift_: float; value must be in range [ -1.0 ... + 1.0]
    :param hsl_model_: 3d numpy.ndarray shape (256, 256, 256, 3) see hsl_to_rgb_model function 
    :param rgb_model_: 3d numpy.ndarray shape (256, 256, 256, 3) see rgb_to_hsl_model function
    :return:
    """
    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)
    assert -1.0 <= shift_ <= 1.0, \
        "Argument shift must be in range[-1.0 ... 1.0]"
    assert PyObject_IsInstance(hsl_model_, (numpy.ndarray, cython.view.memoryview)), \
        "\nArgument hsl_model_ must be a numpy.ndarray or memoryview type, got %s " % type(
            hsl_model_)
    assert PyObject_IsInstance(rgb_model_, (numpy.ndarray, cython.view.memoryview)), \
        "\nArgument rgb_model_ must be a numpy.ndarray or memoryview type, got %s " % type(
            rgb_model_)

    try:
        rgb_array = pixels3d(surface_)

    except Exception as e:
        raise ValueError("Cannot reference source pixels into a 3d array.\n %s " % e)

    shader_hsl_surface24bit_fast_inplace_c(rgb_array, shift_, hsl_model_, rgb_model_)


cpdef inline void blur(object surface_, t_=1):
    """
    APPLY A GAUSSIAN BLUR EFFECT TO THE GAME DISPLAY OR TO A GIVEN TEXTURE (KERNEL 5x5)

    This method is using convolution property and process the image in two passes,
    first the horizontal convolution and last the vertical convolution
    pixels convoluted outside image edges will be set to adjacent edge value
    
    Compatible 24 - 32 bit with or without alpha layer
    
    * The changes are automatically applied inplace to the surface, you do not need to create a 
      new surface.  

    :param surface_: pygame.Surface; compatible 24 - 32 bit surfaces
    :param t_      : integer; must be >0; number of passes (default 1)
    :return: void 
    """
    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)
    assert t_ > 0, \
        "\nArgument t_ must be > 0, got %s " % t_

    try:
        rgb_array = pixels3d(surface_)

    except Exception as e:
        raise ValueError("Cannot reference source pixels into a 3d array.\n %s " % e)

    shader_blur5x5_array24_inplace_c(rgb_array, None, t_)


cpdef inline void wave(object surface_, float rad, int size):
    """
    CREATE A WAVE EFFECT TO THE GAME DISPLAY OR TO A GIVEN SURFACE

    Compatible 24 - 32 bit with or without alpha layer
    
    * The changes are automatically applied inplace to the surface, you do not need to create a 
      new surface.  

    e.g:
    wave(surface, 8 * math.pi/180.0 + frame_number, 5)
    
    :param surface_: pygame.Surface; pygame surface compatible 24 - 32 bit  
    :param rad     : float; angle in rad to rotate over time
    :param size    : int; Number of sub-surfaces
    :return        : void
    """
    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)
    assert size > 0, "Argument size must be > 0"

    try:
        rgb_array = pixels3d(surface_)

    except Exception as e:
        raise ValueError("Cannot reference source pixels into a 3d array.\n %s " % e)

    shader_wave24bit_inplace_c(rgb_array, rad, size)


cpdef inline void swirl(object surface_, float degrees):
    """
    SWIRL AN IMAGE (ANGLE APPROXIMATION METHOD)

    This algorithm uses a table of cos and sin.
    
    Compatible 24 - 32 bit with or without alpha layer
    
    * The changes are automatically applied inplace to the surface, you do not need to create a 
      new surface.  

    e.g:
    swirl(surface, 1.0)
    
    :param surface_: pygame.Surface, compatible 24 - 32 bit 
    :param degrees : float; angle in degrees 
    :return        : void 
    """
    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)

    try:
        rgb_array = pixels3d(surface_)

    except Exception as e:
        raise ValueError("Cannot reference source pixels into a 3d array.\n %s " % e)

    shader_swirl24bit_inplace_c(rgb_array, degrees)



cpdef inline void swirl2(object surface_, float degrees):
    """
    SWIRL AN IMAGE WITHOUT ANGLE APPROXIMATION

    Compatible 24 - 32 bit with or without alpha layer
    
    * The changes are automatically applied inplace to the surface, you do not need to create a 
      new surface.  
      
    e.g:
    swirl(surface_, frame_number)
    
    :param surface_: pygame.Surface, compatible 24 - 32 bit 
    :param degrees : float; angle in degrees
    :return        : void 
    """
    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)

    try:
        rgb_array = pixels3d(surface_)

    except Exception as e:
        raise ValueError("Cannot reference source pixels into a 3d array.\n %s " % e)

    shader_swirl24bit_inplace_c1(rgb_array, degrees)



cpdef inline void plasma_config(
        object surface_,
        int frame,
        float hue_=1.0/6.0,
        float sat_=1.0/6.0,
        float value_=1.0/8.0,
        float a_=1.0/255.0,
        float b_=1.0/12.0,
        float c_=1.0/12.0
):
    """

    CREATE A BASIC PLASMA EFFECT ON THE TOP OF A PYGAME SURFACE

    Compatible 24 - 32 bit with or without alpha layer
    
    * The changes are automatically applied inplace to the surface, you do not need to create a 
      new surface.  

    e.g:
    plasma_config(surface, frame_number)

    :param a_           : float; default value 1.0/255.0 control the plasma equation
    :param b_           : float; default value 1.0/12.0 control the plasma equation
    :param c_           : float; default value 1.0/12.0 control the plasma equation
    :param value_       : float; default value 1.0/8.0 value factor
    :param sat_         : float; default value 1.0/6.0 saturation value
    :param hue_         : float; default value 1.0/6.0 hue value factor
    :param surface_     : pygame.surface; compatible 24 - 32 bit
    :param frame        : integer; Variable that need to change over time
    :return             : void
    """
    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)

    try:
        rgb_array = pixels3d(surface_)

    except Exception as e:
        raise ValueError("Cannot reference source pixels into a 3d array.\n %s " % e)

    shader_plasma24bit_inplace_c(rgb_array, frame, hue_, sat_, value_, a_, b_, c_)


cpdef inline void plasma(surface_, float frame, unsigned int [::1] palette_):
    """
    CREATE A PLASMA EFFECT INPLACE

    e.g:
    plasma(surface, frame_number, palette_)
    
    :param surface_: pygame.Surface; compatible 24 - 32 bit 
    :param frame   : float; frame number
    :param palette_: 1d array containing colors
    :return:
    """
    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)
    shader_plasma_c(surface_, frame, palette_)



cpdef inline float [:, :, :, ::1] rgb_to_hsl_model():
    """
    Create an HSL model containing all the values
    :return: Return a cython.view.memoryview shape (256, 256, 256, 3)
    """
    return rgb_to_hsl_model_c()


cpdef inline unsigned char [:, :, :, ::1] hsl_to_rgb_model():
    """
    Create an RGB model containing all the values
    :return: Return a cython.view.memoryview shape (256, 256, 256, 3)
    """
    return hsl_to_rgb_model_c()



cpdef inline void brightness(object surface_, float shift_):
    """
    SHADER BRIGHTNESS

    This shader control the pygame display brightness level
    It uses two external functions coded in C, struct_rgb_to_hsl & struct_hsl_to_rgb
    
    Parameter shift_ is a float value in range [ -1.0 ... 1.0]. with +1.0 for the 
    maximum brightness. A value of 0.0 will not perform any changes to the original 
    surface
    
    Compatible 24 - 32 bit with or without alpha layer
    
    * The changes are automatically applied inplace to the surface, you do not need to create a 
      new surface.  
      
    e.g:
    brightness(surface, 0.2)
    
    :param surface_ : pygame.surface; 
    :param shift_   : float must be in range [ -1.0 ... 1.0 ]
    :return         : void
    """
    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)

    if shift_ == 0.0:
        return

    assert -1.0 <= shift_ <= 1.0, "\nArgument shift_ must be in range [-1.0 ... 1.0]"

    try:
        rgb_array = pixels3d(surface_)

    except Exception as e:
        raise ValueError("Cannot reference source pixels into a 3d array.\n %s " % e)

    shader_brightness24_inplace_c(rgb_array, shift_)


cpdef inline void brightness_exclude(
        object surface_,
        float shift_,
        color_=(0, 0, 0)
):
    """

    INCREASE/DECREASE A SURFACE BRIGHTNESS (OPTIONAL EXCLUDE COLOR)
    
    The optional setting (color_) allow you to select a color that will not 
    be included in the processing. This can be useful if you know the background 
    color RGB values and do not wish the background to undergo a change in brightness    
    
    Parameter shift_ is a float value in range [ -1.0 ... 1.0]. with +1.0 for the 
    maximum brightness. A value of 0.0 will not perform any changes to the original 
    surface
   
    Parameter color_ is a tuple of RGB colors e.g (1, 1, 1) 

    :param surface_ : pygame.surface; 
    :param shift_   : float must be in range [ -1.0 ... 1.0 ]
    :param color_   : tuple RGB to be excluded from the process
    :return         : void
    """
    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)

    if shift_ == 0.0:
        return

    assert -1.0 <= shift_ <= 1.0, \
        "Argument shift_ must be in range[-1.0 ... 1.0]"

    try:
        rgb_array = pixels3d(surface_)

    except Exception as e:
        raise ValueError("Cannot reference source pixels into a 3d array.\n %s " % e)

    shader_brightness24_exclude_inplace_c(rgb_array, shift_, color_)



cpdef inline void brightness_bpf(
        object surface_,
        float shift_,
        unsigned char bpf_threshold = 64):
    """

    INCREASE/DECREASE SURFACE BRIGHTNESS 

    bpf_threshold is an integer value in range [0..255] that 
    determines the pixels threshold for the brightness algorithm. 
    The sum RGB below this threshold will not be included in the process  
    R + G + B < Threshold 


    :param surface_: Pygame.Surface compatible 24 - 32 bit 

    :param shift_: float, must be in range [-1.00 ... +1.00]

    :param bpf_threshold : integer value in range [0 ... 255].
    threshold RGB. Values R+G+B < threshold will not be included in the process
    :return: void 

    """
    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)

    assert -1.0 <= shift_ <= 1.0, \
        "Argument shift must be in range[-1.0 ... 1.0]"

    try:
        rgb_array = pixels3d(surface_)

    except Exception as e:
        raise ValueError("Cannot reference source pixels into a 3d array.\n %s " % e)

    shader_brightness24_bpf_c(rgb_array, shift_, bpf_threshold)


cpdef inline void brightness_model(
        object surface_,
        float shift_,
        float [:, :, :, :] rgb_to_hsl_model
):
    """
    
    SHADER BRIGHTNESS (EXCLUDE A SPECIFIC COLOR FROM THE PROCESS, DEFAULT BLACK COLOR)

    This shader control the pygame display brightness level
    It uses two external functions coded in C, struct_rgb_to_hsl & struct_hsl_to_rgb

    e.g:
    brightness_exclude(surface, 0.2)
    
    :param surface_ : pygame.surface; compatible 24 - 32 bit 
    :param shift_   : float in range [-1.0 ... 1.0 ]
    :param rgb_to_hsl_model : numpy.ndarray shape (256, 256, 256, 3)
    :return : void 
    """
    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)

    warnings.warn("Deprecated version, use shader_brightness_24_inplace (fastest version)",
                  DeprecationWarning)
    assert -1.0 <= shift_ <= 1.0, "\nArgument shift_ must be in range [-1.0 ... 1.0]"

    try:
        rgb_array = pixels3d(surface_)

    except Exception as e:
        raise ValueError("Cannot reference source pixels into a 3d array.\n %s " % e)

    shader_brightness_24_inplace1_c(rgb_array, shift_, rgb_to_hsl_model)




cpdef inline void saturation(object surface_, float shift_):
    """
    SHADER SATURATION

    This shader control the saturation level of the pygame display or surface/texture

    e.g:
    saturation(surface, 0.2)
    
    
    :param surface_: pygame.Surface; compatible 24 - 32 bit
    :param shift_  : float must be in range [ -1.0 ... 1.0] 
    :return:
    """
    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)

    assert -1.0 <= shift_ <= 1.0, "\nArgument shift_ must be in range [-1.0 ... 1.0]"

    try:
        rgb_array = pixels3d(surface_)

    except Exception as e:
        raise ValueError("Cannot reference source pixels into a 3d array.\n %s " % e)

    shader_saturation_array24_inplace_c(rgb_array, shift_)



cpdef inline void heatwave_vertical(
        object surface_,
        unsigned char [:, :] mask,
        float factor_,
        float center_,
        float sigma_,
        float mu_):
    """

    APPLY A GAUSSIAN TRANSFORMATION TO A SURFACE

    This effect can be use to simulate air turbulence or heat flow/convection

    :param surface_  : pygame.Surface; compatible 24 - 32 bit 
    :param mask      : numpy.ndarray shape (x, y) uint8, (values 255 or 0).
                       Apply transformation to the original array
                       if the value @(x, y) is 255 else remain unchanged.
    :param factor_   : Control the maximum of the gaussian equation.
                       No transformation if factor_ equal zero
    :param center_   : Control the center of the gaussian equation (if center_ equal zero,
                       the Gauss equation is centered
                       at x=0 and maximum is 0.4 with amplitude_ = 1.0)
    :param sigma_    : float; sigma value of the gauss equation
    :param mu_       : float; mu value of the gauss equation
    """
    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)

    assert PyObject_IsInstance(mask, (numpy.ndarray, cython.view.memoryview)), \
        "\nArgument mask must be a numpy.array or memoryview type, got %s " % type(mask)

    try:
        rgb_array = pixels3d(surface_)

    except Exception as e:
        raise ValueError("Cannot reference source pixels into a 3d array.\n %s " % e)

    shader_heatwave24_vertical_inplace_c(rgb_array, mask, factor_, center_, sigma_, mu_)



cpdef inline void horizontal_glitch(
        object surface_,
        float rad1_,
        float frequency_,
        float amplitude_
):
    """
    SHADER GLITCH EFFECT

    Deform the pygame display to create a glitch effect

    :param surface_  : pygame.Surface; compatible 24 - 32 bit 
    :param rad1_     : float; Angle in radians, this value control the angle variation over the time
    :param frequency_: float; signal frequency, factor that amplify the angle variation
    :param amplitude_: float; cos amplitude value
    :return: void
    """
    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)

    try:
        rgb_array = pixels3d(surface_)

    except Exception as e:
        raise ValueError("Cannot reference source pixels into a 3d array.\n %s " % e)

    shader_horizontal_glitch24_inplace_c(rgb_array, rad1_, frequency_, amplitude_)



cpdef inline void bpf(object surface_, int threshold = 128):
    """
    
    SHADER BRIGHT PASS FILTER (INPLACE)

    Conserve only the brightest pixels in a surface

    :param surface_ : pygame.Surface; compatible 24 - 32 bit 
    :param threshold: integer; Bright pass threshold default 128
    :return: void 
    """
    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)
    try:
        rgb_array = pixels3d(surface_)

    except Exception as e:
        raise ValueError("Cannot reference source pixels into a 3d array.\n %s " % e)

    shader_bpf24_inplace_c(rgb_array, threshold)



cpdef inline void bloom(object surface_, int threshold_, bint fast_=False, object mask_=None):
    """
    
    CREATE A BLOOM EFFECT

    * Surface must be a pygame Surface 24-32 bit format

    :param surface_     : pygame.Surface; Game display or texture
    :param threshold_   : integer; Threshold value uint8 in range [0 ... 255].
                          The threshold value is used by a bright
                          pass filter to determine the bright pixels above the given threshold.
                          Below 128 the bloom effect will be more
                          noticeable and above 128 a bit less.
    :param fast_        : bool; True | False; If True the bloom effect will be approximated
                          and only the x16 subsurface
                          will be processed to maximize the overall processing time, 
                          default is False).
    :param mask_        : 
    :return             : void
    
    """
    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)

    shader_bloom_effect_array24_c(surface_, threshold_, fast_, mask_)



cpdef inline fisheye_footprint(int w, int h, unsigned int centre_x, unsigned int centre_y):
    """

    :param w: integer; width of the fisheye model
    :param h: integer; height of the fisheye model
    :return: Return a numpy.ndarray type (w, h, 2) representing the fisheye model (coordinates
    of all surface pixels passing through the fisheye lens model)
    """
    return shader_fisheye24_footprint_c(w, h, centre_x, centre_y)




cpdef inline void fisheye(
        object surface_, unsigned int [:, :, :] fisheye_model):
    """
    THIS SHADER CAN BE USE TO DISPLAY THE GAME THROUGH A LENS EFFECT

    Display a fisheye effect in real time given a surface referencing the
    pixels RGB. In order to accomplish a real time calculation, 
    this algorithm is using a pre-calculated transformation stored
    in the array fisheye_model.
    
    The function shader_fisheye24_footprint_c has to be called prior
    shader_fisheye24_inplace_c in order to store the transformation values.

    This shader can be applied directly to the pygame display

    :param fisheye_model    : numpy.ndarray shape (width, height, 2) int32, fisheye model
    containing the pixels
    coordinates after the fisheye transformation
    :return                 : void
    
    :param surface_      : pygame.Surface; compatible 24 - 32 bit 
    :param fisheye_model : numpy.ndarray or cython memoryview shape (width, height, 2) int32, 
                           fisheye model containing the pixels
    :return: void 
    """

    # TODO EXPERIMENT WITH ORDER = 'C'
    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)

    assert PyObject_IsInstance(fisheye_model, (cython.view.memoryview, numpy.ndarray)), \
        "\nArgument fisheye_model must be a numpy.ndarray or a cython.view.memoryview  type, " \
        "got %s " % type(fisheye_model)

    try:
        rgb_array = pixels3d(surface_)

    except Exception as e:
        raise ValueError("Cannot reference source pixels into a 3d array.\n %s " % e)

    shader_fisheye24_inplace_c(rgb_array, fisheye_model)



# TODO DOC
cpdef inline tuple rain_footprint(int w, int h):
    """
    
    CREATE A FISH EYE LENS DEFORMATION MAP/TEXTURE

    * This function create a texture and its equivalent numpy.ndarray containing the coordinates
      for each pixels after deformation.
    * This method must be called once outside of your game main loop
    * The model can be re-use to display your video game animation without being re-calculated for
      each frame. This method allow a high FPS rate

    :param w    : integer; Width of the fish eye effect
    :param h    : integer; height of the fish eye effect
    :return     : Pygame Surface representing the fish-eye effect and its
                  equivalent numpy.ndarray
    """
    return shader_rain_footprint_inplace_c(w, h)




# TODO DOC
cpdef inline void rain_fisheye(
        object surface_,
        unsigned int [:, :, ::1] rain_fisheye_model
):
    """
    THIS SHADER CAN BE USED TO SIMULATE RAIN DROPLET OR BUBBLE DISPLAYED ON THE TOP OF
    THE SCREEN SURFACE.

    Both surface and rain_fisheye_model must have the exact same size

    1) Always call the method shader_rain_footprint_inplace_c before the main loop.
       The transformation model doesn't have to be calculated every frames.
       The above method will generate a pygame texture (24bit) containing the location
       of each pixels after deformation. It does also return a numpy.ndarray equivalent
       pixel format that can be used instead of the surface if needed.

    It uses a fish eye lens deformation to reproduce the deformed background image onto
    the final image. The operation apply inplace and the surface referenced by the rgb_array_
    will be modified directly.
    The fish-eye lens deformation will recreate you game scene into the rain droplet or bubble
    and create the illusion of animation inside the bubble.

    * This shader cannot be applied directly to the pygame display as the array passed to the
    function is a scaled format of the pygame.display (copy not referencing directly the
    surface pixels)

    * This algorithm use a pre-calculated fish-eye lens deformation model to boost the overall
    FPS performances, the texture pixel is then transformed with the model without any intensive
    math calculation.

    :param surface_             : pygame.Surface compatible 24 - 32 bit 
    :param rain_fisheye_model   : numpy.ndarray or memoryview type (w, h, 3) unsigned int 
                                  containing the the coordinate for each pixels
    :return                     : void

    """
    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)

    assert PyObject_IsInstance(rain_fisheye_model, (numpy.ndarray, cython.view.memoryview)), \
        "\nArgument rain_fisheye_model must be a " \
        "numpy.ndarray or a cython memoryview type, got %s " % type(rain_fisheye_model)

    try:
        rgb_array = pixels3d(surface_)

    except Exception as e:
        raise ValueError("Cannot reference source pixels into a 3d array.\n %s " % e)

    shader_rain_fisheye24_inplace_c(rgb_array, rain_fisheye_model)




cpdef inline void tv_scan(surface_, int space=5):
    """
    
    SHADER CREATING A TV SCANLINE EFFECT ON PYGAME SURFACE

    The space between each scanline can by adjusted with the space value.
    The scanline intensity/colors is lower that the original image

    :param surface_     : pygame.Surface compatible 24-32 bit 
    :param space        : integer; space between the lines
    :return             : void
    
    """
    # TODO SCANLINE VERTICAL | HORIZONTAL

    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)

    assert space > 0, "Argument space cannot be <=0"

    try:
        rgb_array = pixels3d(surface_)

    except Exception as e:
        raise ValueError("Cannot reference source pixels into a 3d array.\n %s " % e)

    shader_tv_scanline_inplace_c(rgb_array, space)




cpdef inline void rgb_split(object surface_, int offset_=10):
    """
    
    THIS SHADER CREATE AN RGB SPLIT EFFECT (SUPERPOSED CHANNEL R, G, B WITH GIVEN OFFSET)
    The transformation apply inplace

    The original surface will be used and used for the subsurface blit operation.
    Each channels will be blit sequentially in the following order RGB
    Note that channel green and blue will be blit with an additional flag BLEND_RGB_ADD, to mix
    the channel with the lower layers.

    * FPS BOOST
    In order to boost the fps frame rate the original surface to process can be downscale x2
    and rescale after processing.


    :param surface_ : pygame Surface to process (24bit format)
    :param offset_  : integer; offset for (x, y) to add to each channels RGB
    :return         : void
    
    """
    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)

    shader_rgb_split_inplace_c(surface_, offset_)



cpdef object rgb_split_clean(object surface_, int offset_=10):
    """

    THIS SHADER CREATE AN RGB SPLIT EFFECT (SUPERPOSED CHANNEL R, G, B WITH GIVEN OFFSET)
    The final image has a different width and height since the offset value is removed to keep only 
    the overlapping R, G, B channels 
    Setting the Offset_ to zero will have no effect to the original image.

    :param surface_ : pygame Surface to process (24bit format)
    :param offset_  : integer; offset for (x, y) to add to each channels RGB
    :return         : void

    """
    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)

    assert isinstance(offset_, int), \
        "\nArgument offset_ must be an int type, got %s" % type(offset_)

    if offset_==0:
        return surface_

    return shader_rgb_split_c(surface_, offset_)




cpdef inline tuple ripple(
        int rows_,
        int cols_,
        float [:, ::1] previous_,
        float [:, ::1] current_,
        unsigned char [:, :, ::1] array_
):
    """
    
    THIS SHADER CREATE A WATER EFFECT ON A PYGAME SURFACE
    This version does not include any background deformation to keep a reasonable FPS rate

    * NOTE this shader cannot be apply to the screen directly (screen referencing
    pygame.display.get_surface()),

    :param rows_        : integer; Array width
    :param cols_        : integer; Array height
    :param previous_    : numpy.ndarray type (w, h) type float; array use for the transformation
    :param current_     : numpy.ndarray type (w, h) type float; array use for the transformation
    :param array_       : numpy.ndarray type (w, h, 3) type unsigned char
    :return             : void
    
    """
    assert PyObject_IsInstance(previous_, (numpy.ndarray, cython.view.memoryview)), \
        "\nArgument previous must be a numpy.ndarray type got %s " % type(previous_)

    assert PyObject_IsInstance(current_, (numpy.ndarray, cython.view.memoryview)), \
        "\nArgument current must be a numpy.ndarray type got %s " % type(current_)

    assert PyObject_IsInstance(array_, (numpy.ndarray, cython.view.memoryview)), \
        "\nArgument array must be a numpy.ndarray type got %s " % type(array_)

    cdef Py_ssize_t prev_w, prev_h
    prev_w, prev_h = previous_.shape[:2]

    cdef Py_ssize_t curr_w, curr_h
    curr_w, curr_h = current_.shape[:2]

    cdef Py_ssize_t arr_w, arr_h
    arr_w, arr_h = array_.shape[:2]

    assert prev_w == curr_w and prev_w == arr_w \
           and prev_h == curr_h and prev_h == arr_h, \
        "\n Array sizes mismatch (previous w: %s, h: %s; " \
        "current w: %s, h: %s; array_ w: %s, h: %s " % (prev_w, prev_h, curr_w, curr_h,
        arr_w, arr_h)

    return shader_ripple_c(rows_, cols_, previous_, current_, array_)




cpdef inline void heatmap(object surface_, bint rgb_=True):
    """
    TRANSFORM AN IMAGE INTO A HEATMAP EQUIVALENT

    :param surface_ : pygame.Surface
    :param rgb_     : boolean; True transformed the image into a RGB heatmap model of False (BGR)
    :return         : void
    """
    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)

    heatmap_convert(surface_, rgb_)





cpdef inline predator_vision(
        object surface_,
        unsigned int sobel_threshold=12,
        unsigned int bpf_threshold=50,
        unsigned int bloom_threshold=50,
        bint inv_colormap=False,
        bint fast=False,
        int blend=pygame.BLEND_RGB_ADD
):
    """
    CONVERT A SURFACE OR DISPLAY TO AN EQUIVALENT PREDATOR MODE 

    :param surface_        : pygame.Surface; compatible 24 - 32 bit 
    :param sobel_threshold : integer; value for sobel edge detection, default is 12 
    :param bpf_threshold   : integer; value for the bright pass filter pixel detection, 
                             default is 50
    :param bloom_threshold : integer; Value for the bloom effect intensity default is 50
    :param inv_colormap    : boolean True | False inverse the colormap aspect, default is False
    :param fast            : boolean True | False for a fast process, default is False
    :param blend           : boolean True | False; final blending mode (New in version 1.0.4)  
    :return                : Return a pygame surface  
    """

    surface_copy = surface_.copy()

    if fast:
        sobel_fast(surface_copy, sobel_threshold, factor_=1)
    else:
        sobel(surface_copy, sobel_threshold)

    bpf(surface_, bpf_threshold)
    shader_bloom_fast1(surface_, bloom_threshold)
    heatmap_convert(surface_, inv_colormap)
    surface_.blit(surface_copy, (0, 0), special_flags=blend)

    return surface_.convert()




cpdef inline blood(object surface_, float [:, :] mask_, float perc_):
    """
    SHADER 2D GAME "HURT EFFECT"
    
    This effect is used in 2D game when the player is being hurt
    THE MASK DETERMINE THE CONTOUR USED FOR THE BLOOD EFFECT.

    :param surface_ : pygame.Surface; compatible surface 24 - 32 bit
    :param mask_    : numpy.ndarray shape (w, h) of float values in range [0.0...1.0]
    :param perc_    : Percentage value in range [0.0 ... 1.0] with 1.0 being 100%
    :return         : void
    
    """
    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)

    assert PyObject_IsInstance(mask_, (numpy.ndarray, cython.view.memoryview)), \
        "\nArgument mask_ must be a numpy.ndarray or cython memoryview types got %s " % type(mask_)

    cdef Py_ssize_t w, h
    w, h = surface_.get_size()

    cdef Py_ssize_t mask_w, mask_h
    mask_w, mask_h = mask_.shape[:2]

    assert w == mask_w and h == mask_h, "\nSurface size and mask size mismatch"

    try:
        rgb_array = pixels3d(surface_)

    except Exception as e:
        raise ValueError("Cannot reference source pixels into a 3d array.\n %s " % e)

    shader_blood_inplace_c(rgb_array, mask_, perc_)


# TODO DOC
cpdef inline object make_palette(int width, float fh, float fs, float fl):
    """
    
    CREATE A PALETTE OF RGB COLORS

    e.g:
        # below: palette of 256 colors & surface (width=256, height=50).
        # hue * 6, saturation = 255.0, lightness * 2.0
        palette, surf = make_palette(256, 50, 6, 255, 2)
        palette, surf = make_palette(256, 50, 4, 255, 2)

    :param width  : integer, Palette width
    :param fh     : float, hue factor
    :param fs     : float, saturation factor
    :param fl     : float, lightness factor
    :return       : Return a 1D array palette

    """

    return make_palette_c(width, fh, fs, fl)



cpdef inline fire_sub(
        int width,
        int height,
        float factor,
        unsigned int [::1] palette,
        float [:, ::1] fire
):
    """

    CREATE A FIRE EFFECT

    :param width    : integer; max width of the effect
    :param height   : integer; max height of the effect
    :param factor   : float; factor to reduce the flame effect
    :param palette  : ndarray; Color palette 1d numpy array (colors buffer unsigned int values)
    :param fire     : ndarray; 2d array (x, y) (contiguous) containing float values
    :return         : Return a numpy array containing the fire effect array shape
     (w, h, 3) of RGB pixels
     
    """

    return fire_surface24_c(width, height, factor, palette, fire)




cpdef inline fire_effect(
        int width_,
        int height_,
        float factor_,
        unsigned int [::1] palette_,
        float [:, ::1] fire_,

        # OPTIONAL
        unsigned short int reduce_factor_ = 3,
        unsigned short int fire_intensity_= 32,
        bint smooth_                      = True,
        bint bloom_                       = True,
        bint fast_bloom_                  = True,
        unsigned char bpf_threshold_      = 0,
        unsigned int low_                 = 0,
        unsigned int high_                = 600,
        bint brightness_                  = True,
        float brightness_intensity_       = 0.15,
        object surface_                   = None,
        bint adjust_palette_              = False,
        tuple hsl_                        = (10, 80, 1.8),
        bint transpose_                   = False,
        bint border_                      = False,
        bint blur_                        = True
        ):
    """
    FIRE SHADER EFFECT 

    * FIRE TEXTURE SIZES 
    
    input width_  : integer,  
    input height_ : integer
    
    width_ and height_ values define the size of the texture e.g Surface(width x height)

    * FIRE ASPECT (CONTROL OVER THE WIDTH): 
    
    inputs low_ : integer  
    input high_ : integer 
    
    Optional arguments low_ & high_ (integer values) define the width 's limits of the fire effect. 
    low_ for the starting point and high_ for the ending of the effect.
    e.g low_ = 10 and high_ = 200. The fire effect will be contain within width = 10 and 200
    low_ & high_ values must be in range [0 ... width_]  
        
    * FIRE HEIGHT:
    
    input factor_ : float
    
    The fire maximum height can be adjust with the variable factor_ (float value)
    value > 3.95 will contain the effect within the display 
    value < 3.95 will enlarge the effect over the display height  
    Recommended value is 3.95 with reduce_factor_ = 3 otherwise adjust the value manually 
    to contain the fire effect within the display
        
    * SPEED CONSIDERATION
    
    input reduce_factor_ : integer
    
    The argument reduce_factor_ control the size of the texture to be processed 
    e.g : a value of 2, divide by 4 the pygame surface define by the values (width_ & height_)
    Smaller texture improve the overall performances but will slightly degrade the fire aspect, 
    especially if the blur and smooth option are not enabled.
    Recommended value for reduce_factor_ is 3 (fast process)   
    reduce_factor_ values must be an integer in range [ 0 ... 4] 
    The reduce_factor_ value will have a significant impact on the fire effect maximum height, 
    adjust the argument factor_ accordingly

    * FIRE INTENSITY AT THE SOURCE
    
    input fire_intensity_: integer
    
    Set the fire intensity with the variable fire_intensity_, 0 low flame,
    32 maximum flame effect
    Values must be an int in range [0 ... 32] 

    * SMOOTHING THE EFFECT
    
    input smooth_: True | False
    
    When smooth_ is True the algorithm will use the pygame function smoothscale (bi-linear 
    filtering) or False the final texture will be adjust with the scale function.
    Set this variable to False if you need the best performance for the effect or if you require
    a pixelated fire effect. Otherwise set the variable to True for a more realistic effect. 

    
    * BLOOM EFFECT 
    
    input bloom_         : True | False
    input fast_bloom_    : True | False
    input bpf_threshold_ : integer
       
    Fire effect produce a bright and smooth light effect to the background texture where the fire 
    intensity is at its maximum.
    Use the flag fast_bloom_ for a compromise between a realistic effect and the best performances
    The flag fast_bloom_ define a very fast bloom algo using only the smallest texture 
    to create a bloom effect (all the intermediate textures will be bypassed). See the bloom effect 
    project for more details.
    When fast_bloom is False, all the sub-surfaces will be blit to the final effect and will 
    produce a more realistic fire effect (this will slightly degrade the overall performances). 
    If the fire effect is too bright, you can always adjust the bright pass filter value
    bpf_threshold_(this will adjust the bloom intensity)
    bpf_threshold_ value must be in range [ 0 ... 255]   
    Below 128 the bloom effect will be more noticeable and above 128 only the brightest
    area will be enhanced.

    * LIGHT EFFECT INTENSITY

    input brightness_            : True | False
    input brightness_intensity_  : float

    When the flag is set to True, the algorithm will use an external function, 
    <shader_brightness24_exclude_inplace_c> to increase the brightness of the effect / texture
    A custom color can be passed to the function defining the pixels to be ignored during the 
    process (default is black color).
    the value must be in range [-1.0 ... 1.0]. Values below zero will decrease the brightness 
    of the flame effect and positive values will increase the brightness of the effect (causing
    bright white patches on the fire texture). 
    Values below -0.4 will cause the fire effect to be translucent and this effect can also be 
    used for simulating ascending heat convection effects on a background texture.
    
    
    * OPTIONAL SURFACE
      
    input surface_ : pygame.Surface
      
    This is an optional surface that can be passed to the shader to improve the performances 
    and to avoid a new surface to be generated every iterations. The surface size must match 
    exactly the reduce texture dimensions otherwise an exception will be raise. 
    see reduce_factor_ option to determine the fire texture size that will be processed.
    
    * COLOR PALETTE ADJUSTMENT  
    
    input adjust_palette_ : True | False
    input hsl_            : (10, 80, 1.8)

    Set this flag to True to modify the color palette of the fire texture. 
    This allow the HSL color model to be apply to the palette values
    You can redefine the palette when the flag is True and by customizing a tuple of 3 float 
    values, default is (10, 80, 1.8). 
    The first value control the palette hue value, the second is for the saturation and last, 
    the palette color lightness. 
    With the variable hsl_ you can rotate the palette colors and define a new flame
    aspect/color/intensity
    If adjust_palette_ is True the original palette define by the argument palette_, will 
    be disregarded.Instead a new palette will be created with the hsl values

    * FLAME ORIENTATION / DIRECTION & BORDER FLAME EFFECT
     
    input transpose_ = True | False,
    input border_    = True | False,
    
    transpose_ = True, this will transpose the final array 
    for e.g :  
    If the final fire texture is (w, h) after setting the transpose flag, the final 
    fire texture will become (h, w). As a result the fire effect will be transversal (starting 
    from the right of the display to the left side). 
    You can always transpose / flip the texture to get the right flame orientation  
    BORDER FLAME EFFECT 
    border_ = True to create a flame effect burning the edge of the display. This version is only
    compatible with symmetrical display or textures (same width & height). If the display 
    is asymmetric, the final border fire effect will be shown within the display and not neccessary 
    on the frame border 
    
    * FINAL TOUCH
    
    input blur_ : True | False
    
    This will will blur the fire effect for a more realistic appearance, remove all the jagged 
    edge when and pixelated effect
    
    
    :param width_           : integer; Size (width) of the surface or display in pixels
    :param height_          : integer; size (height) of the surface or display in pixels
    :param factor_          : float; Value controlling the fire height value
                              must be in range [3.95 ... 4.2].
                              The value 3.95 gives the highest flame effect
    :param palette_         : numpy.ndarray, buffer containing mapped RGB colors (uint values)
    :param fire_            : numpy.ndarray shape (w, h) containing float values (fire intensity).
                              For better performance it is advised to set the array to the size 
                              of the texture after applying the reduction_factor_.
                              For example if the reduction_factor_ is 2, the texture would have 
                              width >> 1 and height >> 1 and the fire_array should be set to 
                              numpy.empty((height >> 1, width >> 1), float32)
    :param reduce_factor_   : unsigned short int ; Can be either 0, 1, 2, 3, 4. 
                              2 and 3 provide the best performance and the best looking effect.
    :param fire_intensity_  : Integer; Control the original amount of energy at the
                              bottom of the fire, must be in range of [0 ... 32]. 
                              32 being the maximum value and the maximum fire intensity
    :param smooth_          : boolean; True smoothscale (bi-linear filtering) or
                              scale algorithm jagged edges (mush faster)
    :param bloom_           : boolean; True or False, True apply a bloom effect to the fire effect
    :param fast_bloom_      : boolean; Fastest bloom. This reduce the amount of calculation
    :param bpf_threshold_   : integer; control the bright pass filter threshold
                              value, must be in range [0 ... 255].
                              Maximum brightness amplification with threshold = 0, 
                              when bpf_threshold_ = 255, no change.
    :param low_             : integer; Starting position x for the fire effect
    :param high_            : integer; Ending position x for the fire effect
    :param brightness_      : boolean; True apply a bright filter shader to the array.
                              Increase overall brightness of the effect
    :param brightness_intensity_: float; must be in range [-1.0 ... 1.0] control
                              the brightness intensity
                              of the effect
    :param surface_         : pygame.Surface. Pass a surface to the shader for
                              better performance, otherwise a new surface will be created each 
                              calls.
    :param adjust_palette_  : boolean; True adjust the palette setting HSL
                              (hue, saturation, luminescence).
                              Be aware that if adjust_palette is True, the optional palette 
                              passed to the Shader will be disregarded
    :param hsl_             : tuple; float values of hue, saturation and luminescence.
                              Hue in range [0.0 ... 100],  saturation [0...100], 
                              luminescence [0.0 ... 2.0]
    :param transpose_       : boolean; Transpose the array (w, h) become (h, w).
                              The fire effect will start from the left and move to the right
    :param border_          : boolean; Flame effect affect the border of the texture
    :param blur_            : boolean; Blur the fire effect
    :return                 : Return a pygame surface that can be blit directly to the game display

    """
    # todo reduce_factor=0 and border = True crash

    assert reduce_factor_ in (0, 1, 2, 3, 4), \
        "Argument reduce factor must be in range 0 ... 4 " \
        "\n reduce_factor_ = 1 correspond to dividing the image size by 2" \
        "\n reduce_factor_ = 2 correspond to dividing the image size by 4"
    assert 0 <= fire_intensity_ < 33, \
        "Argument fire_intensity_ must be in range [0 ... 32] got %s" % fire_intensity_

    assert width_ > 0 and height_ > 0, "Argument width or height cannot be null or < 0"
    assert factor_ > 0, "Argument factor_ cannot be null or < 0"

    return shader_fire_effect_c(
        width_, height_, factor_, palette_, fire_,
        reduce_factor_, fire_intensity_, smooth_,
        bloom_, fast_bloom_, bpf_threshold_, low_, high_, brightness_,
        brightness_intensity_, surface_, adjust_palette_,
        hsl_, transpose_, border_, blur_
    )



cpdef inline cloud_effect(
        int width_,
        int height_,
        float factor_,
        unsigned int [::1] palette_,
        float [:, ::1] cloud_,

        # OPTIONAL
        unsigned short int reduce_factor_   = 2,
        unsigned short int cloud_intensity_ = 16,
        bint smooth_                        = True,
        bint bloom_                         = False,
        bint fast_bloom_                    = True,
        unsigned char bpf_threshold_        = 128,
        unsigned int low_                   = 0,
        unsigned int high_                  = 0,
        bint brightness_                    = False,
        float brightness_intensity_         = 0.0,
        object surface_                     = None,
        bint transpose_                     = False,
        bint blur_                          = True
        ):
    """
    GENERATE CLOUD /SMOKE ON THE GAME DISPLAY 
    
    * CLOUD TEXTURE SIZES 
    
    input width_  : integer,  
    input height_ : integer
    
    width_ and height_ values define the size of the texture e.g Surface(width x height)

    * CLOUD ASPECT (CONTROL OVER THE WIDTH): 
    
    inputs low_ : integer  
    input high_ : integer 
    
    Optional arguments low_ & high_ (integer values) define the width 's limits of the cloud 
    effect. low_ for the starting point and high_ for the ending of the effect.
    e.g low_ = 10 and high_ = 200. The cloud effect will be contain within width = 10 and 200
    low_ & high_ values must be in range [0 ... width_]  
        
    * CLOUD HEIGHT:
    
    input factor_ : float
    
    The cloud maximum height can be adjust with the variable factor_ (float value)
    value > 3.95 will contain the effect within the display 
    value < 3.95 will enlarge the effect over the display height  
    Recommended value is 3.95 with reduce_factor_ = 3 otherwise adjust the value manually 
    to contain the cloud effect within the display
        
    * SPEED CONSIDERATION
    
    input reduce_factor_ : integer
    
    The argument reduce_factor_ control the size of the texture to be processed 
    e.g : a value of 2, divide by 4 the pygame surface define by the values (width_ & height_)
    Smaller texture improve the overall performances but will slightly degrade the cloud aspect, 
    especially if the blur and smooth option are not enabled.
    Recommended value for reduce_factor_ is 3 (fast process)   
    reduce_factor_ values must be an integer in range [ 0 ... 4] 
    The reduce_factor_ value will have a significant impact on the cloud effect maximum height, 
    adjust the argument factor_ accordingly

    * CLOUD INTENSITY AT THE SOURCE
    
    input cloud_intensity_: integer
    
    Set the cloud intensity with the variable cloud_intensity_, 0 low flame,
    32 maximum flame effect
    Values must be an int in range [0 ... 32] 

    * SMOOTHING THE EFFECT
    
    input smooth_: True | False
    
    When smooth_ is True the algorithm will use the pygame function smoothscale (bi-linear 
    filtering) or False the final texture will be adjust with the scale function.
    Set this variable to False if you need the best performance for the effect or if you require
    a pixelated cloud effect. Otherwise set the variable to True for a more realistic effect. 
   
    * BLOOM EFFECT 
    
    input bloom_         : True | False
    input fast_bloom_    : True | False
    input bpf_threshold_ : integer
       
    Bloom effect produce a bright and smooth light effect to the background texture where the cloud 
    intensity is at its maximum.
    Use the flag fast_bloom_ for a compromise between a realistic effect and the best performances
    The flag fast_bloom_ define a very fast bloom algo using only the smallest texture 
    to create a bloom effect (all the intermediate textures will be bypassed). See the bloom effect 
    project for more details.
    When fast_bloom is False, all the sub-surfaces will be blit to the final effect and will 
    produce a more realistic cloud effect (this will slightly degrade the overall performances). 
    If the cloud effect is too bright, you can always adjust the bright pass filter value
    bpf_threshold_(this will adjust the bloom intensity)
    bpf_threshold_ value must be in range [ 0 ... 255]   
    Below 128 the bloom effect will be more noticeable and above 128 only the brightest
    area will be enhanced.

    * LIGHT EFFECT INTENSITY

    input brightness_            : True | False
    input brightness_intensity_  : float

    When the flag is set to True, the algorithm will use an external function, 
    <shader_brightness24_exclude_inplace_c> to increase the brightness of the effect / texture
    A custom color can be passed to the function defining the pixels to be ignored during the 
    process (default is black color).
    the value must be in range [-1.0 ... 1.0]. Values below zero will decrease the brightness 
    of the cloud effect and positive values will increase the brightness of the effect (causing
    bright white patches on the cloud texture). 
    Values below -0.4 will cause the cloud effect to be translucent 
    
    
    * OPTIONAL SURFACE
      
    input surface_ : pygame.Surface
      
    This is an optional surface that can be passed to the shader to improve the performances 
    and to avoid a new surface to be generated every iterations. The surface size must match 
    exactly the reduce texture dimensions otherwise an exception will be raise. 
    see reduce_factor_ option to determine the cloud texture size that will be processed.
    

    * CLOUD ORIENTATION / DIRECTION 
     
    input transpose_ = True | False,
    
    transpose_ = True, this will transpose the final array 
    for e.g :  
    If the final cloud texture is (w, h) after setting the transpose flag, the final 
    cloud texture will become (h, w). As a result the cloud effect will be transversal (starting 
    from the right of the display to the left side). 
    You can always transpose / flip the texture to get the right cloud orientation  
    
    * FINAL TOUCH
    
    input blur_ : True | False
    
    This will will blur the cloud effect for a more realistic appearance, remove all the jagged 
    edge when and pixelated effect
    
    :param width_               : integer; Texture size (width) 
    :param height_              : integer; Texture size (height)
    :param factor_              : float; Floating value used to control the size of the cloud
                                  effect. Value must be in range [3.95 ... 4.2]. Value > 3.95 
                                  will contain the smoke/ cloud effect within the display. 
                                  Values < 3.95 will enlarge the smoke effect.                              
    :param palette_             : numpy.ndarray or cython memoryview containing the color for the 
                                  cloud effect (buffer containing mapped RGB colors (uint values))
    :param cloud_               : numpy.ndarray shape (w, h) containing float values 
                                  (cloud intensity). For better performance it is advised to set the
                                  array to the size of the texture after applying the 
                                  reduction_factor_. For example if the reduction_factor_ is 2, 
                                  the texture would have to be width >> 1 and height >> 1 and the 
                                  cloud_ array should be equivalent to numpy.empty((height >> 1, 
                                  width >> 1), float32)
    :param reduce_factor_       : integer; unsigned short int ; Can be either 0, 1, 2, 3, 4. 
                                  2 and 3 provide the best performance and the best looking effect.
    :param cloud_intensity_     : integer; Determine the amount of smoke the cloud
                                  effect will generate at the base of the effect (value must be in 
                                  range [0 .. 260]). If you provide zero a random value between 
                                  0 ... 260 will be assigned. If you provide 250, a random value 
                                  between 250 and 260 will be set for the amount of smoke. 
                                  The highest the value, the more dense the cloud effect will be
    :param smooth_              : boolean; True use a smoothscale (bi-linear filtering) or
                                  False -> scale algorithm jagged edges (mush faster)
    :param bloom_               : True | False, Add a bloom effect when the flag is set to True
                                  The bloom effect will smooth the cloud and create a dense smoke 
                                  areas where the cloud is the brightest.  
    :param fast_bloom_          : True | False; This set a fast algorithm for the bloom effect (the 
                                  bloom effect will use the smallest texture)
    :param bpf_threshold_       : integer; Bright pass filter value must be in range [ 0 ... 255]
                                  0 produce the maximum bloom effect
    :param low_                 : integer; must be in range [ 0 ... width_], left position of the 
                                  cloud effect 
    :param high_                : integer; must be in range [ 0 ... height_], right position of the
                                  cloud effect
    :param brightness_          : True | False; Increase the brightness of the cloud effect when 
                                  True
    :param brightness_intensity_: float; Set the brightness intensity of the cloud. The value must 
                                  be in range [-1.0 ... +1.0]. Changing the value overtime will 
                                  generate a realistic cloud effect. Negative value will generate 
                                  translucent patch of smoke on the background image
    :param surface_             : Pygame.Surface; Pass a surface to the shader for
                                  better performance, otherwise a new surface will be created each 
                                  calls.
    :param transpose_           : boolean; Transpose the array (w, h) become (h, w).
                                  The cloud effect will start from the left and move to the right
    :param blur_                : boolean; Blur the cloud effect
    :return                     : Return a pygame surface that can be blit directly to the game 
                                  display
    """

    assert reduce_factor_ in (0, 1, 2, 3, 4), \
        "Argument reduce factor must be in range 0 ... 4 " \
        "\n reduce_factor_ = 1 correspond to dividing the image size by 2" \
        "\n reduce_factor_ = 2 correspond to dividing the image size by 4"

    assert width_ > 0 and height_ > 0, "Argument width or height cannot be null or < 0"
    assert factor_ > 0, "Argument factor_ cannot be null or < 0"

    return shader_cloud_effect_c(
        width_, height_, factor_, palette_, cloud_,
        reduce_factor_, cloud_intensity_, smooth_,
        bloom_, fast_bloom_, bpf_threshold_, low_, high_, brightness_,
        brightness_intensity_, surface_, transpose_, blur_
    )



cpdef inline mirroring_array(object surface_):
    """
    
    SHADER MIRRORING

    This method create a mirror image 
    
    Compatible 24 - 32 bit image / surface
    
    :param surface_ : pygame.Surface; compatible 24 - 32 bit 
    :return         : returns a numpy ndarray shape (w, h, 3) 
    
    """
    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)

    try:
        rgb_array = pixels3d(surface_)

    except Exception as e:
        raise ValueError("Cannot reference source pixels into a 3d array.\n %s " % e)

    return mirroring_c(rgb_array)




cpdef inline void mirroring(object surface_):
    """
    SHADER MIRRORING (INPLACE)

    This method create a mirror image 

    :param surface_ : pygame.Surface; compatible 24 - 32 bit 
    :return : void
    """
    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)

    try:
        rgb_array = pixels3d(surface_)

    except Exception as e:
        raise ValueError("Cannot reference source pixels into a 3d array.\n %s " % e)

    mirroring_inplace_c(rgb_array)

# cpdef inline transpose_inplace(object surface_):
#     return tranpose_c(pixels3d(surface_))




cpdef inline void sharpen(object surface_):
    """
    
    SHARPEN IMAGE APPLYING THE BELOW 3 X 3 KERNEL OVER EVERY PIXELS.

    :param surface_ : pygame.Surface; compatible 24 - 32 bit 
    :return         : void 
    """
    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)

    try:
        rgb_array = pixels3d(surface_)

    except Exception as e:
        raise ValueError("Cannot reference source pixels into a 3d array.\n %s " % e)

    shader_sharpen_filter_inplace_c(rgb_array)


cpdef inline void dirt_lens(
        object surface_,
        object lens_model_,
        int flag_=BLEND_RGB_ADD,
        float light_ = 0.0
):
    """
    DIRT LENS EFFECT 
    
    This function display a dirt lens texture on the top of your game display to 
    simulate a camera artefact or realistic camera effect when the light from the
    scene is oriented directly toward the camera. 
    
    Choose a lens texture from the Assets directory (free textures provided in Assets directory 
    of this project). All textures are sizes 5184x3456 and would have to be re-sized to  
    your game display and used by function `dirt_lens`. 
    The function dirt_lens will not resize the texture for you.
    
    The setting light_ is a float values cap between -1.0 to 0.2 and allow you to increase the 
    light source oriented toward the camera. Values <0.0 will decrease the lens dirt 
    effect and values >0.0 will increase the brightness of the display and increase the 
    amount of dirt on the camera lens (your display).
    
    Optionally the setting flag_ can be changed from BLEND_RGB_ADD to any other pygame optional 
    flags value. BLEND_RGB_ADD is the default setting and allow the pixels from the dirt lens 
    texture to be blended (added) to the display and provide the brightest and better looking 
    effect. 
    
    This effect can be used for real time rendering for surfaces resolution 1024x768  
    
    Assets/Bokeh__Lens_Dirt_9.jpg
    Assets/Bokeh__Lens_Dirt_38.jpg
    Assets/Bokeh__Lens_Dirt_46.jpg
    Assets/Bokeh__Lens_Dirt_50.jpg
    Assets/Bokeh__Lens_Dirt_54.jpg
    Assets/Bokeh__Lens_Dirt_67.jpg
    
    :param surface_   : Surface 24 - 32 bit represent the surface or the display 
    
    :param lens_model_: Surface The Lens model is a pygame Surface. PygameShader provide a 6 
     different surfaces that can be used as a layer to generate a dirt lens effect on your game 
     display. See above for the name of the free dirt lens textures. 
     The texture has to be loaded prior calling this effect and passed as an argument. By default 
     the textures sizes are 5184x3456 (width & height). The texture would have also to be re-scale 
     once to the game display dimensions (e.g 1027x768) or to the size of your texture.
     
    :param flag_      : integer; pygame flags such as BLEND_RGB_ADD, BLEND_RGB_MAX etc. These flags 
     will change the overall appearance of the effect blending the dirt lens
     image with a different mathematical expression. BLEND_RGB_ADD is the 
     default flag and blend together the dirt_lens and the game display 
     providing a very bright aspect and vivid effect.
    
    :param light_     : float; Float value cap between [-1.0 ... 0.2] to increase or decrease 
     the overall brightness of the dirt lens texture. Tis setting can be used to simulate a 
     texture transition when sweeping the values from -1.0 toward 0.2 by a small increment.
     Values < 0 will tend to diminish the effect and values > 0 will increase the brightness 
     and the dirt lens effect. 
     
    :return: void 
    """
    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)

    assert PyObject_IsInstance(lens_model_, pygame.Surface), \
        "\nArgument lens_model_ must be a pygame.Surface type, got %s " % type(lens_model_)

    assert PyObject_IsInstance(light_, float), \
        "\nArgument light_ must be a float type, got %s " % type(light_)

    if light_ > 0.2:
        light_ = 0.2
    elif light_ < -1.0:
        light_ = 1.0

    assert PyObject_IsInstance(flag_, int), \
        "\nArgument flag_ must be a int type, got %s " % type(flag_)

    dirt_lens_c(surface_, lens_model_, flag_, light_)


# *******************************************************************

@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef inline void end_game(object surface):
    """
    
    :param surface: 
    :return: 
    """
    raise NotImplementedError

@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef inline void level_clear(object surface):
    """
    
    :param surface: 
    :return: 
    """
    raise NotImplementedError

@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef object dithering(object surface_, int factor_=2):

    """
    Dithering is used in computer graphics to create the illusion of "color depth" in images with
    a limited color palette - a technique also known as color quantization. In a dithered image,
    colors that are not available in the palette are approximated by a diffusion of colored pixels
    from within the available palette. The human eye perceives the diffusion as a mixture of 
    the colors within it (see color vision). Dithered images, particularly those with relatively
     few colors, can often be distinguished by a characteristic graininess or speckled appearance
    
    Take a pygame surface as argument format 24-32 bit and convert it to a 3d array format 
    (w, h, 3) type float (float32, single precision). 
    As the image is converted to a different data type format (uint8 to float32), 
    the transformation cannot be applied inplace. The image returned by the method dithering 
    is a copy of the original image.   
    
    :param surface_: Pygame surface format 24-32 bit 
    :param factor_ : integer; Value must be > 1 otherwise an exception will be thrown 
    :return        : Surface; 
    
    """
    assert PyObject_IsInstance(surface_, Surface), \
        'Argument surface_ must be a pygame.Surface got %s ' % type(surface_)

    assert PyObject_IsInstance(factor_, int), \
        'Argument factor_ must be an int got %s ' % type(factor_)

    assert factor_ > 1, \
        "Argument factor_ must be > 1"

    try:
        rgb_array = pixels3d(surface_)

    except Exception as e:
        raise ValueError("Cannot reference source pixels into a 3d array.\n %s " % e)

    return dithering_c(numpy.divide(rgb_array, 255.0).astype(float32), factor_)

@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef inline void dithering_int(object surface_, int factor_=2):
    """
    Dithering is used in computer graphics to create the illusion of "color depth" in images with
    a limited color palette - a technique also known as color quantization. In a dithered image,
    colors that are not available in the palette are approximated by a diffusion of colored pixels
    from within the available palette. The human eye perceives the diffusion as a mixture of 
    the colors within it (see color vision). Dithered images, particularly those with relatively
     few colors, can often be distinguished by a characteristic graininess or speckled appearance

    Take a pygame surface as argument format 24-32 bit and convert it to a 3d array format 
    (w, h, 3) type float (float32, single precision). 
    As the image is converted to a different data type format (uint8 to float32), 
    the transformation cannot be applied inplace. The image returned by the method dithering 
    is a copy of the original image.   

    :param surface_: Pygame surface format 24-32 bit 
    :param factor_ : integer; Value must be > 1 otherwise an exception will be thrown 
    :return        : Surface; 

    """
    assert PyObject_IsInstance(surface_, Surface), \
        'Argument surface_ must be a pygame.Surface got %s ' % type(surface_)

    assert PyObject_IsInstance(factor_, int), \
        'Argument factor_ must be an int got %s ' % type(factor_)

    assert factor_ > 1, \
        "Argument factor_ must be > 1"

    try:
        rgb_array = pixels3d(surface_)

    except Exception as e:
        raise ValueError("Cannot reference source pixels into a 3d array.\n %s " % e)

    dithering_int_c(rgb_array, factor_)

@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef object pixelation(object surface_):
    """
    Pixelate a pygame.Surface 
    
    This method cannot be used for the game display as the change 
    is applied in a new Surface
    
    :param surface_: pygame.Surface; 
    :return: pixelated surface
    """
    assert PyObject_IsInstance(surface_, Surface), \
        'Argument surface_ must be a pygame.Surface got %s ' % type(surface_)

    cdef Py_ssize_t w, h
    # todo percentage of pixelation
    w, h = surface_.get_size()
    cdef object small = smoothscale(surface_, (32, 32))
    return scale(small, (w, h))

@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef inline zoom_in(object surface_):
    """

    :param surface_ : pygame.Surface; compatible 24 - 32 bit 
    :return:
    """
    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)

    raise NotImplementedError


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef inline electric(object surface_):
    """

    :param surface_ : pygame.Surface; compatible 24 - 32 bit 
    :return:
    """
    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)

    raise NotImplementedError


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef inline filmstrips(object surface_):
    """

    :param surface_ : pygame.Surface; compatible 24 - 32 bit 
    :return:
    """
    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)

    raise NotImplementedError


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef cubism(object surface_):
    """

    :param surface_ : pygame.Surface; compatible 24 - 32 bit 
    :return:
    """
    assert isinstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)
    raise NotImplementedError


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef code_listing(object surface_, size_):
    """

    :param surface_: pygame.Surface; compatible 24 - 32 bit 
    :param size_:
    :return:
    """
    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)

    raise NotImplementedError

@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef ascii_char(object surface_):
    """
    
    :param surface_: 
    :return: 
    """
    raise NotImplementedError

@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef object blend(object source_, object destination_, float percentage_):
    """
    BLEND A SOURCE TEXTURE TOWARD A DESTINATION TEXTURE 
    
    The shader create a new image from both source_ and destination_

    * Video system must be initialised 
    * source_ & destination_ Textures must be same sizes
    * Compatible with 24 - 32 bit surface
    * Output create a new surface
    * Image returned is converted for fast blit (convert())

    :param source_     : pygame.Surface (Source)
    :param destination_: pygame.Surface (Destination)
    :param percentage_ : float; Percentage value between [0.0 ... 100.0]
    :return: return    : Return a 24 bit pygame.Surface and blended with a percentage
                         of the destination texture.
    """
    assert PyObject_IsInstance(source_, Surface), \
        'Argument source_ must be a pygame.Surface got %s ' % type(source_)

    assert PyObject_IsInstance(destination_, Surface), \
        'Argument destination_ must be a pygame.Surface got %s ' % type(destination_)

    assert 0.0 <= percentage_ <= 100.0, \
        "\nIncorrect value for argument percentage should be [0.0 ... 100.0] got %s " % percentage_

    if percentage_ == 0.0:
        return source_

    assert source_.get_size() == destination_.get_size(), \
        'Source and Destination surfaces must have same dimensions: ' \
        'Source (w:%s, h:%s), destination (w:%s, h:%s).' % \
        (*source_.get_size(), *destination_.get_size())

    return blending(source_, destination_, percentage_)



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef cartoon(
        object surface_,
        int sobel_threshold_ = 128,
        int median_kernel_   = 2,
        color_               = 8,
        flag_                = BLEND_RGB_ADD
):
    """
    CREATE A CARTOON EFFECT FROM A GIVEN SURFACE 
    
    * This shader cannot be use online or real time due to the amout of 
      transformation. You can use this shader while editing your textures 
      befre the main loop 
    
    * Compatible with 24 - 32 bit image 
    
    :param surface_: pygame.Surface compatible 24 - 32 bit 
    :param sobel_threshold_: integer sobel threshold
    :param median_kernel_  : integer median kernel  
    :param color_          : integer; color reduction value (max color)
    :param flag_           : integer; Blend flag e.g (BLEND_RGB_ADD, BLEND_RGB_SUB, 
                             BLEND_RGB_MULT, BLEND_RGB_MAX, BLEND_RGB_MIN  
    :return                : Return a pygame Surface with the cartoon effect 
    """

    return cartoon_effect(surface_, sobel_threshold_, median_kernel_, color_, flag_)


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef explode(object surface_):
    """

    :param surface_: 
    :return: 
    """
    raise NotImplementedError

@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef object spectrum(int width, int height, float gamma=1.0):
    """
    CREATE A PYGAME SURFACE DISPLAYING THE LIGHT SPECTRUM 380-750 nm

    Color   Wavelength(nm) Frequency(THz)
    Red     620-750        484-400
    Orange  590-620        508-484
    Yellow  570-590        526-508
    Green   495-570        606-526
    Blue    450-495        668-606
    Violet  380-450        789-668

    :param width: integer; width of the image
    :param height: integer; height of the image
    :param gamma: float; gamma value 
    :return: Return a pygame surface 24-bit (width, height) converted for fast 
    blit 

    """

    return spectrum_c(width, height, gamma)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef inline void convert_27colors(object surface_):

    """
    THIS ALGORITHM CONVERT AN IMAGE USING 27 COLORS ONLY

    :param surface_: numpy.ndarray; containing the pixels RGB. Array shape (w, h, 3)  
    :return: void 
    """
    assert isinstance(surface_, Surface), \
        'Argument surface_ must be a valid Surface, got %s ' % type(surface_)

    try:
        array_ = pixels3d(surface_)

    except (pygame.error, ValueError):
        raise ValueError('\nTexture/image is not compatible.')

    convert_27colors_c(array_)



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef object bilateral(object image, float sigma_s, float sigma_i, unsigned int kernel_size = 3):
    """
    A bilateral filter is a non-linear, edge-preserving, and noise-reducing
    smoothing filter for images. It replaces the intensity of each pixel with a
    weighted average of intensity values from nearby pixels. This weight can be
    based on a Gaussian distribution.

    Here, the normalization factor and the range weight are new terms added to 
    the previous equation. sigma_s  denotes the spatial extent of the kernel, i.e. 
    the size of the neighborhood, and sigma_r  denotes the minimum amplitude of an edge.
    It ensures that only those pixels with intensity values similar to that of the
    central pixel are considered for blurring, while sharp intensity changes are maintained.
    The smaller the value of sigma_i ( or sigma_r), the sharper the edge. As sigma_r  tends to 
    infinity, the equation tends to a Gaussian blur.
       
    e.g:
    
    bilateral(surface, sigma_s = 16, sigma_i = 18)
    
    :param kernel_size  : integer; kernel size, default is 3
    :param image: Surface, Pygame Surface format 24-32 bit format (alpha channel will be ignored)
    
    :param sigma_s: float sigma_s : Spatial extent of the kernel, size of the 
    considered neighborhood
    
    :param sigma_i: float sigma_i (also call sigma_r) range kernel, minimum amplitude of an edge.
    
    :return: return a filtered Surface
    """

    assert isinstance(image, Surface), \
        'Argument image must be a valid Surface, got %s ' % type(image)

    assert isinstance(sigma_s, float), \
        'Argument sigma_s must be a valid Surface, got %s ' % type(sigma_s)

    assert isinstance(sigma_i, float), \
        'Argument sigma_i must be a valid Surface, got %s ' % type(sigma_i)

    try:
        array_ = pixels3d(image)

    except (pygame.error, ValueError):
        raise ValueError('\nTexture/image is not compatible.')


    return bilateral_filter24_c(array_, sigma_s, sigma_i, kernel_size)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef object emboss(object surface_, unsigned int flag=0):
    """
    EMBOSS A PYGAME SURFACE 
    
    :param surface_: pygame.Surface; compatible 24-32 bit
    :param flag    : integer; special pygame flag such as BLEND_RGB_ADD, BLEND_RGB_MULT etc
    :return        : void
    """

    assert isinstance(surface_, Surface), \
        'Argument surface_ must be a valid Surface, got %s ' % type(surface_)

    try:
        array_ = pixels3d(surface_)

    except (pygame.error, ValueError):
        raise ValueError('\nTexture/image is not compatible.')

    cdef object emb = emboss5x5_c(array_)

    if flag != 0:
        del array_
        surface_.blit(emb, (0, 0), special_flags=flag)
        return surface_

    return emboss5x5_c(array_)




@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef void palette_change(
        object surface_,
        object palette_):
    """
    CHANGE AN IMAGE BY CHANGING THE COLOR PALETTE 
    
    LIST_PALETTES contains all the palettes available
    in the project.
    
    e.g: 
    from PygameShader.Palette import LIST_PALETTES
    print(LIST_PALETTES.keys())
    
    :param surface_: pygame.Surface; 
    :param palette_: numpy.ndarray containing all the RGB color values 
    :return: void
    """

    assert isinstance(surface_, Surface), \
        'Argument surface_ must be a valid Surface, got %s ' % type(surface_)

    assert isinstance(palette_, numpy.ndarray), \
        'Argument surface_ must be a valid Surface, got %s ' % type(palette_)
    try:
        array_ = pixels3d(surface_)

    except (pygame.error, ValueError):
        raise ValueError('\nTexture/image is not compatible.')

    palette_change_c(array_, palette_)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef object bilinear(object surface_, int new_width, int new_height, fx=None, fy=None):


    assert isinstance(surface_, Surface), \
        'Argument surface_ must be a valid Surface, got %s ' % type(surface_)

    try:
        rgb_array = pixels3d(surface_)

    except (pygame.error, ValueError):
        raise ValueError('\nTexture/image is not compatible.')


    return bilinear_c(rgb_array, new_width, new_height)

# ******************************************************************



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline float damped_oscillation(float t)nogil:
    """
    
    :param t: 
    :return: 
    """
    return <float>(exp(-t * 0.1) * <float>cos(M_PI * t))


cdef float C1_ = <float>1.0 / <float>sqrt(M_2PI)

cdef inline float gauss(float x, float c, float sigma=1.0, float mu=0.0)nogil:
    """
    
    :param x: 
    :param c: 
    :param sigma: 
    :param mu: 
    :return: 
    """
    x -= c
    return (1.0 / sigma * C1_) * exp(-0.5 * ((x - mu) * (x - mu)) / (sigma * sigma))


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline void shader_rgb_to_bgr_inplace_c(unsigned char [:, :, :] rgb_array):
    """
    SHADER RGB to BGR

    The Array (rgb_array) must be a numpy array shape (w, h, 3) containing RGB pixels,
    please refer to pygame function pixels3d or array3d to convert an image into a
    3d array (library surfarray)

    Convert your game display from RGB to BGR format
    This algorithm can also be used to transform pygame texture in the equivalent bgr format

    e.g:
    rgb_to_bgr(surface)

    :param rgb_array    : numpy.ndarray shape(w, h, 3) uint8 (unsigned char 0...255) containing the
    pygame display pixels format RGB. Apply the transformation inplace by swapping the channel
    Red to channel blue and vice versa
    :return             : void
    """

    cdef Py_ssize_t w, h
    w, h = rgb_array.shape[:2]

    cdef:
        int i=0, j=0
        unsigned char tmp

    with nogil:

        for j in prange(h, schedule='static', num_threads=THREADS):
            for i in range(w):
                tmp = rgb_array[i, j, 0]  # keep the blue color
                rgb_array[i, j, 0]  = rgb_array[i, j, 2]
                rgb_array[i, j, 2]  = tmp


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline void shader_rgb_to_brg_inplace_c(unsigned char [:, :, :] rgb_array):
    """

    SHADER RGB TO BRG

    Convert your game display from RGB to BRG format.
    This algorithm can also be used to transform pygame texture in the equivalent BRG format

    The Array (rgb_array) must be a numpy array shape (w, h, 3) containing RGB pixels,
    please refer to pygame function pixels3d or array3d to convert an image into a 3d array
    (library surfarray)

    e.g:
    rgb_to_brg(surface)

    :param rgb_array    : numpy.ndarray shape(w, h, 3) uint8 (unsigned char 0...255) containing the
    pygame display pixels format RGB
    :return             : void
    """

    cdef Py_ssize_t w, h
    w, h = rgb_array.shape[:2]

    cdef:
        int i=0, j=0
        unsigned char tmp_r, tmp_g
        unsigned char *r
        unsigned char *g
        unsigned char *b

    with nogil:

        for j in prange(h, schedule='static', num_threads=THREADS):
            for i in range(w):
                tmp_r = rgb_array[i, j, 0]  # keep the red color
                tmp_g = rgb_array[i, j, 1]  # keep the green color
                rgb_array[i, j, 0] = rgb_array[i, j, 2] # r-->b
                rgb_array[i, j, 1] = tmp_r  # g --> r
                rgb_array[i, j, 2] = tmp_g



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline void shader_greyscale_luminosity24_inplace_c(unsigned char [:, :, :] rgb_array):
    """
    SHADER GRAYSCALE (CONSERVE LUMINOSITY)

    This shader transform the game display on a grayscale video game effect
    This shader can also be applied to pygame textures/surface to transform them into
    an equivalent grayscale model

    The Array (rgb_array) must be a numpy array shape (w, h, 3) containing RGB pixels,
    please refer to pygame function pixels3d or array3d to convert an image into a 3d array
    (library surfarray)

    e.g:
    greyscale(surface)

    :param rgb_array    : numpy.ndarray shape(w, h, 3) uint8 (unsigned char 0...255) containing the
    pygame display pixels format RGB
    :return             : void
    """

    cdef Py_ssize_t w, h
    w, h = rgb_array.shape[:2]

    cdef:
        int i, j
        unsigned char luminosity
        unsigned char *r
        unsigned char *g
        unsigned char *b

    with nogil:
        for j in prange(h, schedule='static', num_threads=THREADS):
            for i in range(w):
                r = &rgb_array[i, j, 0]
                g = &rgb_array[i, j, 1]
                b = &rgb_array[i, j, 2]
                luminosity = <unsigned char>(r[0] * 0.2126 + g[0] * 0.7152 + b[0] * 0.072)
                r[0] = luminosity
                g[0] = luminosity
                b[0] = luminosity


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline void shader_sepia24_inplace_c(unsigned char [:, :, :] rgb_array):

    """
    SHADER SEPIA MODEL

    Transform your video game into an equivalent sepia model
    The Array (rgb_array) must be a numpy array shape (w, h, 3) containing RGB pixels,
    please refer to pygame function pixels3d or array3d to convert an image into a 3d array
    (library surfarray)

    e.g:
    sepia(surface)

    :param rgb_array    : numpy.ndarray shape(w, h, 3) uint8 (unsigned char 0...255) containing the
    pygame display pixels format RGB
    :return             : void
    """
    cdef Py_ssize_t w, h
    w, h = rgb_array.shape[:2]

    cdef:
        int i=0, j=0
        float rr, gg, bb
        unsigned char *r
        unsigned char *g
        unsigned char *b

    with nogil:
        for j in prange(h, schedule='static', num_threads=THREADS):
            for i in range(w):

                r = &rgb_array[i, j, 0]
                g = &rgb_array[i, j, 1]
                b = &rgb_array[i, j, 2]

                rr = r[0] * <float>0.393 + g[0] * <float>0.769 + b[0] * <float>0.189
                gg = r[0] * <float>0.349 + g[0] * <float>0.686 + b[0] * <float>0.168
                bb = r[0] * <float>0.272 + g[0] * <float>0.534 + b[0] * <float>0.131
                if rr > 255:
                    rr = 255
                if gg > 255:
                    gg = 255
                if bb > 255:
                    bb = 255
                r[0] = <unsigned char> rr
                g[0] = <unsigned char> gg
                b[0] = <unsigned char> bb

# ************* SORTING ALGORITHM FOR MEDIAN FILTER
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline void bubble_sort(unsigned char [::1] nums, int size)nogil:
    """
    
    :param nums: 
    :param size: 
    :return: 
    """
    # We set swapped to True so the loop looks runs at least once
    cdef:
        int i, j
        unsigned char *p
        unsigned char *p1
        bint swapped

    swapped = True
    while swapped:
        swapped = False
        for i in range(size - 1):
            p  = &nums[i]
            p1 = &nums[i+1]
            if p[0] > p1[0]:
                p[0], p1[0] = p1[0], p[0]
                swapped = True


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline void insertion_sort(unsigned char [::1] nums, int size)nogil:
    """
    
    :param nums: 
    :param size: 
    :return: 
    """

    cdef:
        int i, j
        unsigned char item_to_insert

    for i in prange(1, size, schedule='static', num_threads=THREADS):
        item_to_insert = nums[i]

        j = i - 1
        while j >= 0 and nums[j] > item_to_insert:
            nums[j + 1] = nums[j]
            j = j - 1
        # Insert the item
        nums[j + 1] = item_to_insert



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
# There are different ways to do a Quick Sort partition, this implements the
# Hoare partition scheme. Tony Hoare also created the Quick Sort algorithm.
cdef inline int partition_cython(unsigned char [::1] nums, int low, int high)nogil:
    """
    
    :param nums: 
    :param low: 
    :param high: 
    :return: 
    """
    cdef:
        int pivot
        int i, j
    pivot = nums[(low + high) >> 1]
    i = low - 1
    j = high + 1
    while True:
        i += 1
        while nums[i] < pivot:
            i += 1

        j -= 1
        while nums[j] > pivot:
            j -= 1

        if i >= j:
            return j

        nums[i], nums[j] = nums[j], nums[i]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline void _quick_sort(unsigned char [::1] items, int low, int high)nogil:
    """
    
    :param items: 
    :param low: 
    :param high: 
    :return: 
    """
    cdef int split_index
    if low < high:
        split_index = partition_cython(items, low, high)
        _quick_sort(items, low, split_index)
        _quick_sort(items, split_index + 1, high)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline void heapify(unsigned char [::1] nums, int heap_size, int root_index)nogil:
    """
    
    :param nums: 
    :param heap_size: 
    :param root_index: 
    :return: 
    """
    # Assume the index of the largest element is the root index
    cdef int largest = root_index
    cdef int left_child = (2 * root_index) + 1
    cdef int right_child = (2 * root_index) + 2

    if left_child < heap_size and nums[left_child] > nums[largest]:
        largest = left_child

    if right_child < heap_size and nums[right_child] > nums[largest]:
        largest = right_child

    if largest != root_index:
        nums[root_index], nums[largest] = nums[largest], nums[root_index]
        heapify(nums, heap_size, largest)



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline void heap_sort(unsigned char [::1] nums, int n)nogil:
    """
    
    :param nums: 
    :param n: 
    :return: 
    """
    cdef int i

    for i in range(n, -1, -1):
        heapify(nums, n, i)

    for i in range(n - 1, 0, -1):
        nums[i], nums[0] = nums[0], nums[i]
        heapify(nums, i, 0)



# *********** END OF SORTING ALGORITHM


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline void shader_median_filter24_inplace_heapsort_c(
        unsigned char [:, :, :] rgb_array_, 
        int kernel_size_ =2):

    """
    SHADER MEDIAN FILTER

    This shader cannot be used for real time rendering as the performance of the algorithm are not
    satisfactory. The code would have to be changed and improved with C or assembler in order to
    be adapted for a real time application. Another version can also be written with a surface
    downscale prior processing /rescale method that would boost the fps performance.

    In the state, this shader can be used for texture/surface transformation offline

    The Array (rgb_array) must be a numpy array shape (w, h, 3) containing RGB pixels,
    please refer to pygame function pixels3d or array3d to convert an image into a 3d array
    (library surfarray)

    :param rgb_array_   : numpy.ndarray shape(w, h, 3) uint8 (unsigned char 0...255) containing the
    pygame display pixels format RGB
    :param kernel_size_ : integer; size of the kernel
    :return             : void
    """

    cdef Py_ssize_t w, h
    w, h = rgb_array_.shape[:2]

    cdef:
        int k = kernel_size_ >> 1
        int k_size = kernel_size_ * kernel_size_

    cdef:
        unsigned char [:, :, ::1] rgb_array_copy = \
            ascontiguousarray(numpy.array(rgb_array_, copy=True))

        int i=0, j=0, ky, kx
        Py_ssize_t ii=0, jj=0

        unsigned char [::1] tmp_red   = empty(k_size, numpy.uint8, order='C')
        unsigned char [::1] tmp_green = empty(k_size, numpy.uint8, order='C')
        unsigned char [::1] tmp_blue  = empty(k_size, numpy.uint8, order='C')

        int index = 0
        Py_ssize_t w_1 = w, h_1 = h

    with nogil:
        for i in prange(w_1, schedule='static', num_threads=THREADS, chunksize=2048):
            for j in range(h_1):

                index = 0

                for kx in range(-k, k):
                    for ky in range(-k, k):

                        ii = i + kx
                        jj = j + ky

                        if ii < 0:
                            ii = 0
                        elif ii > w:
                            ii = w

                        if jj < 0:
                            jj = 0
                        elif jj > h:
                            jj = h

                        tmp_red[index]   = rgb_array_copy[ii, jj, 0]
                        tmp_green[index] = rgb_array_copy[ii, jj, 1]
                        tmp_blue[index]  = rgb_array_copy[ii, jj, 2]
                        index = index + 1

                heap_sort(tmp_red, k_size)
                heap_sort(tmp_green, k_size)
                heap_sort(tmp_blue, k_size)

                rgb_array_[i, j, 0] = tmp_red[k + 1]
                rgb_array_[i, j, 1] = tmp_green[k + 1]
                rgb_array_[i, j, 2]= tmp_blue[k + 1]



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline void shader_median_filter24_inplace_c(
        unsigned char [:, :, :] rgb_array_, 
        int kernel_size_=2
        ):

    """
    SHADER MEDIAN FILTER

    This shader cannot be used for real time rendering as the performance of the algorithm are not
    satisfactory. The code would have to be changed and improved with C or assembler in order to
    be adapted for a real time application. Another version can also be written with a surface
    downscale prior processing /rescale method that would boost the fps performance.

    In the state, this shader can be used for texture/surface transformation offline

    The Array (rgb_array) must be a numpy array shape (w, h, 3) containing RGB pixels,
    please refer to pygame function pixels3d or array3d to convert an image into a 3d array
    (library surfarray)

    :param rgb_array_   : numpy.ndarray shape(w, h, 3) uint8 (unsigned char 0...255) containing the
    pygame display pixels format RGB
    :param kernel_size_ : integer; size of the kernel
    :return             : void
    """

    cdef Py_ssize_t w, h
    w, h = rgb_array_.shape[:2]

    cdef:
        unsigned char [:, :, ::1] rgb_array_copy = \
            ascontiguousarray(numpy.array(rgb_array_, copy=True))

        int i=0, j=0, ky, kx
        Py_ssize_t ii=0, jj=0
        int k = kernel_size_ >> 1
        int k_size = kernel_size_ * kernel_size_

        # int [64] tmp_red   = empty(64, numpy.int16, order='C')
        # int [64] tmp_green = empty(64, numpy.int16, order='C')
        # int [64] tmp_blue  = empty(64, numpy.int16, order='C')

        int *tmp_red   = <int *> malloc(k_size * sizeof(int))
        int *tmp_green = <int *> malloc(k_size * sizeof(int))
        int *tmp_blue  = <int *> malloc(k_size * sizeof(int))

        int [::1] tmp_red_ = <int[:k_size]>tmp_red
        int [::1] tmp_green_ = <int[:k_size]>tmp_green
        int [::1] tmp_blue_ = <int[:k_size]>tmp_blue

        int *tmpr
        int *tmpg
        int *tmpb

        int index = 0
        Py_ssize_t w_1 = w, h_1 = h


    with nogil:
        for i in prange(w_1, schedule='static', num_threads=THREADS, chunksize=2048):
            for j in prange(h_1):

                index = 0

                for kx in range(-k, k):
                    for ky in range(-k, k):

                        ii = i + kx
                        jj = j + ky

                        if ii < 0:
                            ii = 0
                        elif ii > w:
                            ii = w

                        if jj < 0:
                            jj = 0
                        elif jj > h:
                            jj = h

                        tmp_red_[index]   = rgb_array_copy[ii, jj, 0]
                        tmp_green_[index] = rgb_array_copy[ii, jj, 1]
                        tmp_blue_[index]  = rgb_array_copy[ii, jj, 2]
                        index = index + 1

                # External C quicksort
                tmpr = quickSort(&tmp_red_[0], 0, k_size)
                tmpg = quickSort(&tmp_green[0], 0, k_size)
                tmpb = quickSort(&tmp_blue[0], 0, k_size)

                rgb_array_[i, j, 0] = tmpr[(index -1) >> 1]
                rgb_array_[i, j, 1] = tmpg[(index -1) >> 1]
                rgb_array_[i, j, 2] = tmpb[(index -1) >> 1]



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline void shader_median_grayscale_filter24_inplace_c(
        unsigned char [:, :, :] rgb_array_, 
        int kernel_size_=2
        ):

    """
    SHADER MEDIAN FILTER

    This shader cannot be used for real time rendering as the performance of the algorithm are not
    satisfactory. The code would have to be changed and improved with C or assembler in order to
    be adapted for a real time application. Another version can also be written with a surface
    downscale prior processing /rescale method that would boost the fps performance.

    In the state, this shader can be used for texture/surface transformation offline

    The Array (rgb_array) must be a numpy array shape (w, h, 3) containing RGB pixels,
    please refer to pygame function pixels3d or array3d to convert an image into a 3d
    array (library surfarray)

    :param rgb_array_   : numpy.ndarray shape(w, h, 3) uint8 (unsigned char 0...255) containing the
    pygame display pixels format RGB
    :param kernel_size_ : integer; size of the kernel
    :return             : void
    """

    cdef Py_ssize_t w, h
    w, h = rgb_array_.shape[:2]

    cdef:
        unsigned char [:, :, ::1] rgb_array_copy = \
            ascontiguousarray(numpy.array(rgb_array_, copy=True))

        int i=0, j=0, ky, kx
        Py_ssize_t ii=0, jj=0

        int k = kernel_size_ >> 1
        int k_size = kernel_size_ * kernel_size_
        # int [64] tmp_  = empty(64, numpy.int16, order='C')
        int *tmp_   = <int *> malloc(k_size * sizeof(int))
        int index = 0


    with nogil:
        for i in prange(0, w, schedule='static', num_threads=THREADS, chunksize=2048):
            for j in range(0, h):

                index = 0

                for kx in range(-k, k):
                    for ky in range(-k, k):

                        ii = i + kx
                        jj = j + ky

                        if ii < 0:
                            ii = 0
                        elif ii > w:
                            ii = w

                        if jj < 0:
                            jj = 0
                        elif jj > h:
                            jj = h

                        tmp_[index]   = rgb_array_copy[ii, jj, 0]

                        index = index + 1

                tmp = quickSort(tmp_, 0, k_size)

                rgb_array_[i, j, 0] = tmp[k + 1]
                rgb_array_[i, j, 1] = tmp[k + 1]
                rgb_array_[i, j, 2] = tmp[k + 1]



cdef float ONE_255 = <float>1.0 / <float>255.0

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline void shader_color_reduction24_inplace_c(
        unsigned char [:, :, :] rgb_array, int color_number):
    """
    COLOR REDUCTION SHADER

    Decrease the amount of colors in the display or texture.
    The method of color reduction is very simple: every color of the original picture is replaced
    by an appropriate color from the limited palette that is accessible.

    The Array (rgb_array) must be a numpy array shape (w, h, 3) containing RGB pixels,
    please refer to pygame function pixels3d or array3d to convert an image into a 3d array
     (library surfarray)

    e.g:
    color_reduction(surface, 8)

    :param rgb_array    : numpy.ndarray shape(w, h, 3) uint8 (unsigned char 0...255) containing the
    pygame display pixels format RGB
    :param color_number : integer; color number color_number^2
    :return             : void
    """


    cdef Py_ssize_t w, h
    w, h = rgb_array.shape[:2]


    cdef:
        int x=0, y=0
        unsigned char *r
        unsigned char *g
        unsigned char *b
        float f = <float> 255.0 / <float> color_number
        float c1 = <float>color_number / <float>255.0

    with nogil:
        for y in prange(h, schedule='static', num_threads=THREADS):
            for x in range(0, w):

                r = &rgb_array[x, y, 0]
                g = &rgb_array[x, y, 1]
                b = &rgb_array[x, y, 2]

                r[0] = <unsigned char>(<int>(<float>round_c(c1 * <float>r[0]) * f))
                g[0] = <unsigned char>(<int>(<float>round_c(c1 * <float>g[0]) * f))
                b[0] = <unsigned char>(<int>(<float>round_c(c1 * <float>b[0]) * f))



cdef:
    short [:, ::1] GY = numpy.array(
        ([-1, 0, 1],
         [-2, 0, 2],
         [-1, 0, 1])).astype(dtype=int16, order='C')

    short [:, ::1] GX = numpy.array(
        ([-1, -2, -1],
         [0,   0,  0],
         [1,   2,  1])).astype(dtype=int16, order='c')

    unsigned short int KERNEL_HALF = 1


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline void shader_sobel24_inplace_c(unsigned char [:, :, :] rgb_array, float threshold=20.0):
    """
    SHADER SOBEL (EDGE DETECTION)

    Transform the game display or a pygame surface into a sobel equivalent model
    (surface edge detection)

    The Array (rgb_array) must be a numpy array shape (w, h, 3) containing RGB pixels,
    please refer to pygame function pixels3d or array3d to convert an image into a 3d
    array (library surfarray)

    e.g:
    sobel(surface, 64)

    :param rgb_array    : numpy.ndarray shape (w, h, 3) containing RGB values
    :param threshold    : float; Threshold value (Default value = 20.0)
    :return             : void
    """

    cdef Py_ssize_t w, h
    w, h = rgb_array.shape[:2]

    cdef:
        Py_ssize_t w_1 = w - 1
        Py_ssize_t h_1 = h - 1
        int kernel_offset_x, kernel_offset_y
        unsigned char [:, :, :] source_array = numpy.array(rgb_array, copy=True)
        int x, y
        Py_ssize_t xx, yy
        float r_gx, r_gy
        unsigned char *gray
        unsigned char m
        float magnitude

    with nogil:

        for y in prange(h, schedule='static', num_threads=THREADS):

            for x in range(w):

                r_gx, r_gy = <float>0.0, <float>0.0

                for kernel_offset_x in range(-KERNEL_HALF, KERNEL_HALF + 1):

                    for kernel_offset_y in range(-KERNEL_HALF, KERNEL_HALF + 1):

                        xx = x + kernel_offset_x
                        yy = y + kernel_offset_y

                        if xx > w_1:
                            xx = w_1
                        if xx < 0:
                            xx = 0

                        if yy > h_1:
                            yy = h_1
                        if yy < 0:
                            yy = 0

                        # grayscale image red = green = blue
                        gray = &source_array[xx, yy, 0]

                        # if kernel_offset_x != 0:

                        r_gx = r_gx + <float> gray[0] * \
                               <float> GX[kernel_offset_x + KERNEL_HALF,
                                          kernel_offset_y + KERNEL_HALF]

                        # if kernel_offset_y != 0:

                        r_gy = r_gy + <float> gray[0] * \
                               <float> GY[kernel_offset_x + KERNEL_HALF,
                                          kernel_offset_y + KERNEL_HALF]

                magnitude = min(<float> sqrt(r_gx * r_gx + r_gy * r_gy), <float>255.0)

                # if magnitude > 255:
                #     magnitude = 255.0

                m = <unsigned char> magnitude if magnitude > threshold else 0

                # update the pixel if the magnitude is above threshold else black pixel
                rgb_array[x, y, 0] = m
                rgb_array[x, y, 1] = m
                rgb_array[x, y, 2] = m



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef unsigned char [:, :, ::1] scale_array24_c(
        unsigned char [:, :, :]
        rgb_array,
        int w2,
        int h2
):
    """
    ARRAY RE-SCALING ; ARRAY SHAPE (W, H, 3)

    This is an internal tool that cannot be access outside of this library (cpdef hook missing)
    Re-scale an array (rgb_array) of size (w, h, 3) into an equivalent array size (w2, h2, 3).
    This function is identical to a surface rescaling but uses the array instead

    The Array (rgb_array) must be a numpy array shape (w, h, 3) containing RGB pixels,
     please refer to pygame function pixels3d or array3d to convert an image into a 3d
      array (library surfarray)

    e.g:
    memview_array = scale_array24_c(my_array, 300, 300)

    :param rgb_array    : RGB numpy.ndarray, format (w, h, 3) numpy.uint8
    :param w2           : new width
    :param h2           : new height
    :return             : Return a MemoryViewSlice 3d numpy.ndarray format (w, h, 3) uint8
    """
    assert w2 > 0, "Argument w2 cannot be <=0"
    assert h2 > 0, "Argument h2 cannot be <=0"

    cdef Py_ssize_t w1, h1, s

    try:
        w1, h1, s = rgb_array.shape[:3]

    except (ValueError, pygame.error) as e:
        raise ValueError('\nArray shape not understood.')

    cdef:
        unsigned char [:, :, ::1] new_array = numpy.empty((w2, h2, 3), numpy.uint8)
        float fx = <float>w1 / <float>w2
        float fy = <float>h1 / <float>h2
        int x, y
        Py_ssize_t xx, yy
    with nogil:
        for x in prange(w2, schedule='static', num_threads=THREADS):
            xx = <int>(x * fx)
            for y in range(h2):
                yy = <int>(y * fy)
                new_array[x, y, 0] = rgb_array[xx, yy, 0]
                new_array[x, y, 1] = rgb_array[xx, yy, 1]
                new_array[x, y, 2] = rgb_array[xx, yy, 2]

    return new_array


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline void shader_sobel24_fast_inplace_c(
        surface_, int threshold_=20, unsigned short factor_=1):
    """
    SHADER FAST SOBEL (EDGE DETECTION)

    Transform the game display or a pygame surface into a sobel equivalent model
    (surface edge detection).This version is slightly fastest than shader_sobel24_inplace_c as
    it down-scale the array containing all the pixels and apply the sobel algorithm to a smaller
    sample. When the processing is done, the array is re-scale to its original dimensions.
    If this method is in theory faster than shader_sobel24_inplace_c, down-scaling and up-scaling
    an array does have a side effect of decreasing the overall image definition
    (jagged lines non-antialiasing)

    e.g:
    sobel_fast(surface, 64, factor_=1)

    :param surface_     :  pygame.Surface
    :param threshold_   : integer; threshold value for the sobel filter
    :param factor_      : unsigned short (default value =1). Define the
    reduction factor of an image. 1 divide by 2, 2 divide by 4 etc
    :return             : void
    """

    cdef:
        Py_ssize_t w, h, w_1, h_1
    w, h = surface_.get_size()

    cdef:
        int kernel_offset_x, kernel_offset_y
        int x, y
        Py_ssize_t xx, yy
        float r_gx, r_gy
        unsigned char *gray
        unsigned char m
        float magnitude

        unsigned char [:, :, :] source_array = pixels3d(surface_)
        unsigned char [:, :, :] rescale_array = \
            numpy.array(pixels3d(scale(surface_, (w >> factor_, h >> factor_))))
        unsigned char [:, :, :] new_array = empty((w >> factor_, h >> factor_, 3), uint8)

    h = h >> factor_
    w = w >> factor_
    w_1 = w - 1
    h_1 = h - 1

    with nogil:

        for y in prange(h, schedule='static', num_threads=THREADS):

            for x in range(w):

                r_gx, r_gy = <float>0.0, <float>0.0

                for kernel_offset_y in range(-KERNEL_HALF, KERNEL_HALF + 1):

                    for kernel_offset_x in range(-KERNEL_HALF, KERNEL_HALF + 1):

                        xx = x + kernel_offset_x
                        yy = y + kernel_offset_y

                        if xx > w_1:
                            xx = w_1
                        if xx < 0:
                            xx = 0

                        if yy > h_1:
                            yy = h_1
                        if yy < 0:
                            yy = 0

                        # grayscale image red = green = blue
                        gray = &rescale_array[xx, yy, 0]

                        if kernel_offset_x != 0:

                            r_gx = r_gx + <float> gray[0] * \
                                   <float> GX[kernel_offset_x + KERNEL_HALF,
                                              kernel_offset_y + KERNEL_HALF]

                        if kernel_offset_y != 0:

                            r_gy = r_gy + <float> gray[0] * \
                                   <float> GY[kernel_offset_x + KERNEL_HALF,
                                              kernel_offset_y + KERNEL_HALF]

                magnitude = <float> sqrt(r_gx * r_gx + r_gy * r_gy)

                if magnitude > 255:
                    magnitude = <float>255.0

                m = <unsigned char> magnitude if magnitude > threshold_ else 0

                # update the pixel if the magnitude is above threshold else black pixel
                new_array[x, y, 0] = m
                new_array[x, y, 1] = m
                new_array[x, y, 2] = m

    w = w << factor_
    h = h << factor_

    cdef unsigned char [:, :, :] new_ = scale_array24_c(new_array, w, h)

    with nogil:

        for y in prange(h, schedule='static', num_threads=THREADS):
            for x in range(w):

                source_array[x, y, 0] = new_[x, y, 0]
                source_array[x, y, 1] = new_[x, y, 1]
                source_array[x, y, 2] = new_[x, y, 2]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline void shader_invert_surface_24bit_inplace_c(
        unsigned char [:, :, :] rgb_array):

    """
    SHADER INVERT PIXELS

    The Array (rgb_array) must be a numpy array shape (w, h, 3) containing RGB pixels,
     please refer to pygame function pixels3d or array3d to convert an image into a 3d
     array (library surfarray)

    Invert all pixels of the display or a given texture
    e.g:
    invert(surface)

    :param rgb_array    : numpy.ndarray containing all the RGB color values. Array shape (w, h, 3)
    :return             : void
    """

    cdef Py_ssize_t w, h
    w, h = rgb_array.shape[:2]

    cdef:
        int i=0, j=0
        unsigned char *r
        unsigned char *g
        unsigned char *b

    with nogil:
        for j in prange(h, schedule='static', num_threads=THREADS):
            for i in range(w):
                r = &rgb_array[i, j, 0]
                g = &rgb_array[i, j, 1]
                b = &rgb_array[i, j, 2]
                r[0] = 255 - r[0]
                g[0] = 255 - g[0]
                b[0] = 255 - b[0]


cdef float[5] GAUSS_KERNEL = \
    [<float>(1.0/16.0), <float>(4.0/16.0),
     <float>(6.0/16.0), <float>(4.0/16.0), <float>(1.0/16.0)]

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline void shader_blur5x5_array24_inplace_c(
        unsigned char [:, :, :] rgb_array_, mask=None, t=1):
    """
    APPLY A GAUSSIAN BLUR EFFECT TO THE GAME DISPLAY OR TO A GIVEN TEXTURE (KERNEL 5x5)

    # Gaussian kernel 5x5
        # |1   4   6   4  1|
        # |4  16  24  16  4|
        # |6  24  36  24  6|  x 1/256
        # |4  16  24  16  4|
        # |1  4    6   4  1|
    This method is using convolution property and process the image in two passes,
    first the horizontal convolution and last the vertical convolution
    pixels convoluted outside image edges will be set to adjacent edge value

    The Array (rgb_array) must be a numpy array shape (w, h, 3) containing RGB pixels,
    please refer to pygame function pixels3d or array3d to convert an image into a 3d array (
    library surfarray)

    :param rgb_array_   : numpy.ndarray type (w, h, 3) uint8
    :param mask         : numpy.ndarray default None
    :param t            : integer; number of times
    :return             : Return 24-bit a numpy.ndarray type (w, h, 3) uint8
    """

    cdef Py_ssize_t w, h
    w, h = rgb_array_.shape[:2]

    # kernel 5x5 separable
    cdef:

        short int kernel_half = 2
        unsigned char [:, :, :] convolve = numpy.empty((w, h, 3), dtype=uint8)
        int x, y, xx, yy
        float r, g, b, s
        char kernel_offset
        unsigned char red, green, blue
        float *k
        unsigned char *c1
        unsigned char *c2
        unsigned char *c3
        unsigned char *c4
        unsigned char *c5
        unsigned char *c6

    for r in range(t):
        with nogil:

            # horizontal convolution
            for y in prange(0, h, schedule='static', num_threads=THREADS):

                c1 = &rgb_array_[0, y, 0]
                c2 = &rgb_array_[0, y, 1]
                c3 = &rgb_array_[0, y, 2]
                c4 = &rgb_array_[w-1, y, 0]
                c5 = &rgb_array_[w-1, y, 1]
                c6 = &rgb_array_[w-1, y, 2]

                for x in range(0, w):  # range [0..w-1]

                    r, g, b = 0, 0, 0

                    for kernel_offset in range(-kernel_half, kernel_half + 1):

                        k = &GAUSS_KERNEL[kernel_offset + kernel_half]

                        xx = x + kernel_offset

                        # check boundaries.
                        # Fetch the edge pixel for the convolution
                        if xx < 0:
                            red, green, blue = c1[0], c2[0], c3[0]
                        elif xx > (w - 1):
                            red, green, blue = c4[0], c5[0], c6[0]
                        else:
                            red, green, blue = rgb_array_[xx, y, 0],\
                                rgb_array_[xx, y, 1], rgb_array_[xx, y, 2]
                            if red + green + blue == 0:
                                continue

                        r = r + red * k[0]
                        g = g + green * k[0]
                        b = b + blue * k[0]

                    convolve[x, y, 0] = <unsigned char>r
                    convolve[x, y, 1] = <unsigned char>g
                    convolve[x, y, 2] = <unsigned char>b

            # Vertical convolution
            for x in prange(0,  w, schedule='static', num_threads=THREADS):

                c1 = &convolve[x, 0, 0]
                c2 = &convolve[x, 0, 1]
                c3 = &convolve[x, 0, 2]
                c4 = &convolve[x, h-1, 0]
                c5 = &convolve[x, h-1, 1]
                c6 = &convolve[x, h-1, 2]

                for y in range(0, h):
                    r, g, b = 0, 0, 0

                    for kernel_offset in range(-kernel_half, kernel_half + 1):

                        k = &GAUSS_KERNEL[kernel_offset + kernel_half]
                        yy = y + kernel_offset

                        if yy < 0:
                            red, green, blue = c1[0], c2[0], c3[0]
                        elif yy > (h -1):
                            red, green, blue = c4[0], c5[0], c6[0]
                        else:
                            red, green, blue = convolve[x, yy, 0],\
                                convolve[x, yy, 1], convolve[x, yy, 2]
                            if red + green + blue == 0:
                                continue

                        r = r + red * k[0]
                        g = g + green * k[0]
                        b = b + blue * k[0]

                    rgb_array_[x, y, 0], rgb_array_[x, y, 1], rgb_array_[x, y, 2] = \
                        <unsigned char>r, <unsigned char>g, <unsigned char>b


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline void shader_wave24bit_inplace_c(
        unsigned char [:, :, :] rgb_array_,
        float rad,
        int size):
    """
    CREATE A WAVE EFFECT TO THE GAME DISPLAY OR TO A GIVEN SURFACE

    The Array (rgb_array) must be a numpy array shape (w, h, 3) containing RGB pixels,
     please refer to pygame function pixels3d or array3d to convert an image into a
     3d array (library surfarray)

    e.g:
    wave(surface, 8 * math.pi/180.0 + frame_number, 5)

    :param rgb_array_   : numpy.ndarray shape (w, h, 3) containing all the RGB values
    :param rad          : float; angle in rad to rotate over time
    :param size         : int; Number of sub-surfaces
    :return             : void
    """



    cdef Py_ssize_t w, h
    w, h = rgb_array_.shape[:2]

    cdef:
        unsigned char [:, :, ::1] rgb = numpy.array(rgb_array_, copy=True, order='C')
        int x, y, x_pos, y_pos, xx, yy
        int i=0, j=0
        float c1 = <float>1.0 / <float>(size * size)
        int w_1 = <int>w - 1
        int h_1 = <int>h - 1

    with nogil:

        for y in prange(0, h_1 - size, size, schedule='static', num_threads=THREADS):

            y_pos = y + size + <int>(<float>sin(rad + <float>y * c1) * <float>size)

            for x in prange(0, w_1 - size, size, schedule='static', num_threads=THREADS):

                x_pos = x + size + <int> (<float>sin(rad + <float> x * c1) * <float> size)

                for i in range(0, size + 1):

                    for j in range(0, size + 1):

                        xx = x_pos + i
                        yy = y_pos + j

                        if xx > w_1:
                            xx = w_1
                        elif xx < 0:
                            xx = 0
                        if yy > h_1:
                            yy = h_1
                        elif yy < 0:
                            yy = 0
                        rgb_array_[xx, yy, 0] = rgb[x + i, y + j, 0]
                        rgb_array_[xx, yy, 1] = rgb[x + i, y + j, 1]
                        rgb_array_[xx, yy, 2] = rgb[x + i, y + j, 2]




@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(False)
cdef inline void shader_swirl24bit_inplace_c(
        unsigned char [:, :, :] rgb_array_,
        float degrees
):
    """
    SWIRL AN IMAGE (ANGLE APPROXIMATION METHOD)

    This algorithm uses a table of cos and sin.

    The Array (rgb_array) must be a numpy array shape (w, h, 3) containing RGB pixels,
     please refer to pygame function pixels3d or array3d to convert an image into a
     3d array (library surfarray)

    e.g:
    swirl(surface, frame_number / 1000.0)

    :param rgb_array_   : numpy.ndarray shape (w, h, 3) containing all the RGB color values
    :param degrees      : float; swirl angle in degrees
    :return             : void
    """

    cdef Py_ssize_t w, h
    w, h = rgb_array_.shape[:2]

    cdef:
        int i, j, diffx, diffy, angle
        float columns, rows, r, di, dj, c1, c2
        unsigned char [:, :, ::1] rgb = numpy.array(rgb_array_, copy=True, order='C')
        float r_max


    columns = <float>0.5 * (<float>w - <float>1.0)
    rows    = <float>0.5 * (<float>h - <float>1.0)

    r_max = <float>sqrt(columns * columns + rows * rows)

    with nogil:
        for j in prange(h, schedule='static', num_threads=THREADS):
            for i in range(w):

                di = <float>i - columns
                dj = <float>j - rows

                r = <float>sqrt(di * di + dj * dj) / <float>r_max
                angle = <int>(degrees * r % 360)

                c1 = COS_TABLE[angle]
                c2 = SIN_TABLE[angle]
                diffx = <int>(di * c1 - dj * c2 + columns)
                diffy = <int>(di * c2 + dj * c1 + rows)

                if (diffx >-1) and (diffx < w) and \
                   (diffy >-1) and (diffy < h):
                    rgb_array_[i, j, 0], rgb_array_[i, j, 1],\
                        rgb_array_[i, j, 2] = rgb[diffx, diffy, 0], \
                                              rgb[diffx, diffy, 1], rgb[diffx, diffy, 2]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline void shader_swirl24bit_inplace_c1(unsigned char [:, :, :] rgb_array_, float degrees):
    """
    SWIRL AN IMAGE WITHOUT ANGLE APPROXIMATION

    The Array (rgb_array) must be a numpy array shape (w, h, 3) containing RGB pixels,
     please refer to pygame function pixels3d or array3d to convert an image into a 3d
     array (library surfarray)

    e.g:
    swirl(surface_, frame_number / 1000)

    :param rgb_array_   : numpy.ndarray shape (w, h, 3) containing all the RGB color values
    :param degrees      : float; swirl angle in degrees
    :return             : void
    """

    cdef Py_ssize_t w, h
    w, h = rgb_array_.shape[:2]

    cdef:
        int i, j, diffx, diffy
        float columns, rows, r, di, dj, c1, c2, angle
        unsigned char [:, :, :] rgb = numpy.array(rgb_array_, copy=True)
        float rad = degrees * DEG_TO_RAD
        float r_max


    columns = <float>0.5 * (w - <float>1.0)
    rows    = <float>0.5 * (h - <float>1.0)
    r_max   = <float>sqrt(columns * columns + rows * rows)

    with nogil:
        for i in prange(w, schedule='static', num_threads=THREADS):
            for j in range(h):

                di = <float>i - columns
                dj = <float>j - rows

                r = <float>sqrt(di * di + dj * dj)
                angle = <float>(rad * r/r_max)

                c1 = <float>cos(angle)
                c2 = <float>sin(angle)
                diffx = <int>(di * c1 - dj * c2 + columns)
                diffy = <int>(di * c2 + dj * c1 + rows)

                if (diffx >-1) and (diffx < w) and \
                   (diffy >-1) and (diffy < h):
                    rgb_array_[i, j, 0], rgb_array_[i, j, 1],\
                        rgb_array_[i, j, 2] = rgb[diffx, diffy, 0], \
                                              rgb[diffx, diffy, 1], rgb[diffx, diffy, 2]




@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(False)
cdef inline void shader_plasma24bit_inplace_c(
        unsigned char [:, :, :] rgb_array_,
        int frame,
        float hue_=1.0/6.0,
        float sat_=1.0/6.0,
        float value_=1.0/8.0,
        float a_=ONE_255,
        float b_=ONE_TWELVE,
        float c_=ONE_TWELVE):
    """
    CREATE A BASIC PLASMA EFFECT ON THE TOP OF A PYGAME SURFACE

    The Array (rgb_array) must be a numpy array shape (w, h, 3) containing RGB pixels,
     please refer to pygame function pixels3d or array3d to convert an image into a
     3d array (library surfarray)

    e.g:
    plasma_config(surface, frame_number)

    :param a_           : float; default value 1.0/255.0 control the plasma equation
    :param b_           : float; default value 1.0/12.0 control the plasma equation
    :param c_           : float; default value 1.0/12.0 control the plasma equation
    :param value_       : float; default value 1.0/8.0 value factor
    :param sat_         : float; default value 1.0/6.0 saturation value
    :param hue_         : float; default value 1.0/6.0 hue value factor
    :param rgb_array_   : numpy.ndarray shape( w, h, 3) containing all the RGB color values
    :param frame        : integer; Variable that need to change over time
    :return             : void
    """

    cdef Py_ssize_t width, height
    width, height = rgb_array_.shape[:2]

    cdef:
        float xx, yy, t
        float h, s, v
        int i = 0, x, y
        float f, p, q, t_
        float hue, r, g, b
        unsigned char *rr
        unsigned char *gg
        unsigned char *bb

    t = <float>frame

    with nogil:
        for x in prange(width, schedule='static', num_threads=THREADS):
            for y in range(height):

                xx = <float>x * <float>0.5
                yy = <float>y * <float>0.5

                hue = <float>4.0 + <float>sin((xx * <float>0.5 + yy * <float>0.5) * a_) \
                      + <float>sin((xx * b_ + t) * c_)

                h, s, v = hue * hue_, hue * sat_, hue * value_

                i = <int>(h * <float>6.0)
                f = (h * <float>6.0) - i
                p = v*(<float>1.0 - s)
                q = v*(<float>1.0 - s * f)
                t_ = v*(<float>1.0 - s * (<float>1.0 - f))
                i = i % 6

                if i == 0:
                    r, g, b =  v, t_, p
                if i == 1:
                     r, g, b = q, v, p
                if i == 2:
                     r, g, b = p, v, t_
                if i == 3:
                     r, g, b = p, q, v
                if i == 4:
                     r, g, b = t_, p, v
                if i == 5:
                     r, g, b = v, p, q

                if s == 0.0:
                     r, g, b = v, v, v

                rr = &rgb_array_[x, y, 0]
                gg = &rgb_array_[x, y, 1]
                bb = &rgb_array_[x, y, 2]

                if rr[0] < 256:
                    rr[0] = <unsigned char>(min(rr[0] + r * <float>128.0, <float>255.0))

                if gg[0] < 256:
                    gg[0] = <unsigned char>(min(gg[0] + g * <float>128.0, <float>255.0))

                if bb[0] < 256:
                    bb[0] = <unsigned char>(min(bb[0] + b * <float>128.0, <float>255.0))


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline void shader_plasma_c(surface_, float frame, unsigned int [::1] palette_):

    """
    CREATE A PLASMA EFFECT INPLACE

    e.g:
    plasma(surface, frame_number)

    :param surface_: pygame Surface
    :param frame   : float; frame number
    :param palette_: color palette
    :return        : void
    """
    cdef Py_ssize_t width, height
    width, height = surface_.get_size()

    try:
        rgb_array = pixels3d(surface_)

    except Exception as e:
        raise ValueError("Cannot reference source pixels into a 3d array.\n %s " % e)

    cdef:
        int x, y, ii,c

        unsigned char [:, :, :] rgb_array_ = rgb_array

        unsigned char *rr
        unsigned char *gg
        unsigned char *bb
        float color_
        float w2 = <float>width * <float>HALF
        float h2 = <float>height * <float>HALF
        Py_ssize_t length = len(palette_)

    with nogil:

        for y in prange(height, schedule='static', num_threads=THREADS):
            for x in range(width):

                rr = &rgb_array_[x, y, 0]
                gg = &rgb_array_[x, y, 1]
                bb = &rgb_array_[x, y, 2]

                color_ = <float>128.0 + (<float>128.0 * <float>sin(x * ONE_255 + frame)) \
                    + <float>128.0 + (<float>128.0 * <float>sin(frame * ONE_32 + y * ONE_64)) \
                    + <float>128.0 + (<float>128.0 * <float>sin(
                    sqrt((x - w2) * (x - w2) + (y - h2) * (y - h2)) * ONE_255)) \
                    + <float>128.0 + (<float>128.0 * <float>sin(
                    <float>sqrt(x * x + y * y + frame) * <float>ONE_64))

                c = min(<int>(color_ / <float>8.0), <int>length)

                ii = palette_[c]

                rr[0] = (ii >> 16) & 255
                gg[0] = (ii >> 8) & 255
                bb[0] = ii & 255


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline float [:, :, :, ::1] rgb_to_hsl_model_c():
    """
    CONVERT RGB INTO HSL MODEL

    The array can be used when the rgb to hsl is extensively used

    All the values will be stored into an array shape (r, g, b, 3) type float
    :return: Numpy.ndarray shape (r, g, b, 3) type float
    """
    cdef float [:, :, :, ::1] rgb_to_hsl = numpy.empty((256, 256, 256, 3), float32)
    cdef hsl hsl_
    cdef int r, g, b
    with nogil:
        for r in prange(0, 256, schedule='static', num_threads=THREADS):
            for g in range(0, 256):
                for b in range(0, 256):
                    hsl_ = struct_rgb_to_hsl(
                        r * <float>ONE_255, g * <float>ONE_255, b * <float>ONE_255)
                    rgb_to_hsl[r, g, b, 0] = min(<float>(hsl_.h * <float>255.0), <float>255.0)
                    rgb_to_hsl[r, g, b, 1] = min(<float>(hsl_.s * <float>255.0), <float>255.0)
                    rgb_to_hsl[r, g, b, 2] = min(<float>(hsl_.l * <float>255.0), <float>255.0)

    return asarray(rgb_to_hsl, dtype=float32)



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline unsigned char [:, :, :, ::1] hsl_to_rgb_model_c():
    """
    CONVERT HSL INTO RGB MODEL

    The array can be used when the hsl to rgb is extensively used

    All the values will be stored into an array shape (r, g, b, 3) type float
    :return: Numpy.ndarray shape (r, g, b, 3) type float
    """
    cdef unsigned char [:, :, :, ::1] hsl_to_rgb = numpy.empty((256, 256, 256, 3), uint8)
    cdef rgb rgb_
    cdef int r, g, b
    cdef int h, s, l

    with nogil:
        for h in prange(0, 256, schedule='static', num_threads=THREADS):
            for s in range(0, 256):
                for l in range(0, 256):
                    rgb_ = struct_hsl_to_rgb(h * <float>ONE_255, s *
                                             <float>ONE_255, l * <float>ONE_255)
                    hsl_to_rgb[h, s, l, 0] =\
                        min(<unsigned char> (rgb_.r * <float>255.0), <unsigned char>255)
                    hsl_to_rgb[h, s, l, 1] = \
                        min(<unsigned char> (rgb_.g * <float>255.0), <unsigned char>255)
                    hsl_to_rgb[h, s, l, 2] = \
                        min(<unsigned char> (rgb_.b * <float>255.0), <unsigned char>255)

    return asarray(hsl_to_rgb, dtype=uint8)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline void shader_hsl_surface24bit_inplace_c(unsigned char [:, :, :] rgb_array, float shift_):

    """
    ROTATE THE HUE OF THE GAME DISPLAY OR GIVEN TEXTURE

    The Array (rgb_array) must be a numpy array shape (w, h, 3) containing RGB pixels,
    please refer to pygame function pixels3d or array3d to convert an image into a
    3d array (library surfarray)

    e.g:
    hsl(surface, 0.2)

    :param rgb_array    : numpy.ndarray of shape(w, h, 3) of unsigned char, rgb values
    :param shift_       : float; Hue value in range [-1.0 ... 1.0]
    :return             : void
    """

    cdef Py_ssize_t w, h
    w, h = rgb_array.shape[:2]

    cdef:
        int i=0, j=0
        hsl hsl_
        rgb rgb_
        float h_
        unsigned char *r
        unsigned char *g
        unsigned char *b


    with nogil:
        for j in prange(h, schedule='static', num_threads=THREADS):
            for i in range(w):
                r = &rgb_array[i, j, 0]
                g = &rgb_array[i, j, 1]
                b = &rgb_array[i, j, 2]
                hsl_ = struct_rgb_to_hsl(
                    r[0] * <float>ONE_255, g[0] *
                    <float>ONE_255, b[0] * <float>ONE_255)

                #h_ = min((hsl_.h + shift_), 1.0)
                #h_ = max(h_, 0.0)
                h_ = hsl_.h + shift_
                rgb_ = struct_hsl_to_rgb(h_, hsl_.s, hsl_.l)

                r[0] = <unsigned char>(rgb_.r * <float>255.0)
                g[0] = <unsigned char>(rgb_.g * <float>255.0)
                b[0] = <unsigned char>(rgb_.b * <float>255.0)



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline void shader_hsl_surface24bit_fast_inplace_c(
        unsigned char [:, :, :] rgb_array,
        float shift_,
        float [:, :, :, ::1] hsl_model_,
        unsigned char [:, :, :, ::1] rgb_model_):

    """
    ROTATE THE HUE OF AN IMAGE USING STORED HSL TO RGB AND RGB TO HSL VALUES

    The Array (rgb_array) must be a numpy array shape (w, h, 3) containing RGB pixels,
    please refer to pygame function pixels3d or array3d to convert an image into a
    3d array (library surfarray)

    e.g:
    hsl(surface, 0.2, hsl_model, rgb_model)

    :param rgb_array    : numpy.ndarray of shape(w, h, 3) of unsigned char, rgb values
    :param shift_       : float; Hue value in range [-1.0 ... 1.0]
    :param hsl_model_   : numpy.ndarray shape (r, g, b, 3) of hsl values r, g & b
    in range [0 ... 255]
    :param rgb_model_   : numpy.ndarray shape (h, s, l, 3) of hsl values h, s & l
    in range [0.0 ... 1.0]
    :return: void
    """



    cdef Py_ssize_t w, h
    w, h = rgb_array.shape[:2]

    cdef:
        int i=0, j=0

        float h_
        unsigned char h__, s__, l__

        unsigned char *r
        unsigned char *g
        unsigned char *b


    with nogil:
        for j in prange(h, schedule='dynamic', num_threads=THREADS):
            for i in range(w):

                r = &rgb_array[i, j, 0]
                g = &rgb_array[i, j, 1]
                b = &rgb_array[i, j, 2]

                h_ = hsl_model_[r[0], g[0], b[0], 0]

                h__ = <unsigned char> (<float>min((h_ * ONE_255 + shift_), <float>1.0) * <float>255.0)

                s__ = <unsigned char> hsl_model_[r[0], g[0], b[0], 1]
                l__ = <unsigned char> hsl_model_[r[0], g[0], b[0], 2]

                r[0] = (&rgb_model_[h__, s__, l__, 0])[0]
                g[0] = (&rgb_model_[h__, s__, l__, 1])[0]
                b[0] = (&rgb_model_[h__, s__, l__, 2])[0]



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline void shader_brightness24_inplace_c(
        unsigned char [:, :, :] rgb_array_, float shift_=0):
    """
    SHADER BRIGHTNESS

    This shader control the pygame display brightness level
    It uses two external functions coded in C, struct_rgb_to_hsl & struct_hsl_to_rgb

    The Array (rgb_array) must be a numpy array shape (w, h, 3) containing RGB pixels,
    please refer to pygame function pixels3d or array3d to convert an image into
    a 3d array (library surfarray)

    e.g:
    brightness(surface, 0.2)

    :param rgb_array_: numpy ndarray shape (w, h, 3) containing RGB pixels values
    :param shift_    : float; values in range [-1.0 ... 1.0], 0 no change,
    -1 lowest brightness, +1 highest brightness
    :return          : void
    """

    cdef Py_ssize_t width, height
    width, height = rgb_array_.shape[:2]

    cdef:
        int i=0, j=0
        unsigned char *r
        unsigned char *g
        unsigned char *b
        float l, h, s
        hsl hsl_
        rgb rgb_
        float high, low, high_
        unsigned char exclude_r, exclude_g, exclude_b
        bint exclude_color = False

    with nogil:
        for j in prange(height, schedule='static', num_threads=THREADS):
            for i in range(width):

                r, g, b = &rgb_array_[i, j, 0], &rgb_array_[i, j, 1], &rgb_array_[i, j, 2]

                hsl_ = struct_rgb_to_hsl(
                    r[0] * <float>ONE_255, g[0] *
                    <float>ONE_255, b[0] * <float>ONE_255)

                l = min((hsl_.l + shift_), <float>1.0)
                l = max(l, <float>0.0)

                rgb_ = struct_hsl_to_rgb(hsl_.h, hsl_.s, l)

                r[0] = <unsigned char> (rgb_.r * 255.0)
                g[0] = <unsigned char> (rgb_.g * 255.0)
                b[0] = <unsigned char> (rgb_.b * 255.0)


# TODO CHECK WHY NOT PRANGE
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline void shader_brightness24_exclude_inplace_c(
        unsigned char [:, :, :] rgb_array_, float shift_=0.0, color_=(0, 0, 0)):
    """
    SHADER BRIGHTNESS (EXCLUDE A SPECIFIC COLOR FROM THE PROCESS, DEFAULT BLACK COLOR)

    This shader control the pygame display brightness level
    It uses two external functions coded in C, struct_rgb_to_hsl & struct_hsl_to_rgb

    The Array (rgb_array) must be a numpy array shape (w, h, 3) containing RGB pixels,
     please refer to pygame function pixels3d or array3d to convert an image into a
     3d array (library surfarray)

    e.g:
    brightness_exclude(surface, 0.2)

    :param rgb_array_: numpy ndarray shape (w, h, 3) containing RGB pixels values
    :param shift_    : float; values in range [-1.0 ... 1.0], 0 no change,
    -1 lowest brightness, +1 highest brightness
    :param color_    : tuple; Color to exclude from the brightness process
    :return          : void
    """

    cdef Py_ssize_t width, height
    width, height = rgb_array_.shape[:2]

    cdef:
        int i=0, j=0
        unsigned char r
        unsigned char g
        unsigned char b
        float l, h, s
        hsl hsl_
        rgb rgb_
        float high, low, high_
        unsigned char rr=color_[0], gg=color_[1], bb=color_[2]

    with nogil:
        for j in range(height): #, schedule='static', num_threads=THREADS):
            for i in range(width):

                r, g, b = rgb_array_[i, j, 0], rgb_array_[i, j, 1], rgb_array_[i, j, 2]

                if not ((r==rr) and (g==gg) and (b==bb)):

                    hsl_ = struct_rgb_to_hsl(
                        r * <float>ONE_255, g * <float>ONE_255, b * <float>ONE_255)

                    l = min((hsl_.l + shift_), <float>1.0)
                    l = max(l, <float>0.0)

                    rgb_ = struct_hsl_to_rgb(hsl_.h, hsl_.s, l)

                    rgb_array_[i, j, 0] = <unsigned char> (rgb_.r * 255.0)
                    rgb_array_[i, j, 1] = <unsigned char> (rgb_.g * 255.0)
                    rgb_array_[i, j, 2] = <unsigned char> (rgb_.b * 255.0)



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline void shader_brightness24_bpf_c(
        unsigned char [:, :, :] rgb_array_,
        float shift_=0.0,
        unsigned char bpf_treshold_=64):
    """
    
    :param rgb_array_: numpy ndarray shape (w, h, 3) containing RGB pixels values
    :param shift_    : float; values in range [-1.0 ... 1.0], 0 no change,
    -1 lowest brightness, +1 highest brightness
    :param bpf_treshold_ : integer; Bright pass filter threshold value 
    :return          : void
    """


    cdef Py_ssize_t width, height
    width, height = rgb_array_.shape[:2]

    cdef:
        int i=0, j=0
        unsigned char r
        unsigned char g
        unsigned char b
        float l, h, s
        hsl hsl_
        rgb rgb_
        float high, low, high_


    with nogil:
        for j in prange(height, schedule='static', num_threads=THREADS):
            for i in range(width):

                r, g, b = rgb_array_[i, j, 0], rgb_array_[i, j, 1], rgb_array_[i, j, 2]

                if r + g + b < bpf_treshold_:
                    continue

                hsl_ = struct_rgb_to_hsl(
                    r * <float>ONE_255, g * <float>ONE_255, b * <float>ONE_255)

                l = min((hsl_.l + shift_), <float>1.0)
                l = max(l, <float>0.0)

                rgb_ = struct_hsl_to_rgb(hsl_.h, hsl_.s, l)

                rgb_array_[i, j, 0] = <unsigned char> (rgb_.r * 255.0 )
                rgb_array_[i, j, 1] = <unsigned char> (rgb_.g * 255.0 )
                rgb_array_[i, j, 2] = <unsigned char> (rgb_.b * 255.0 )



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline void shader_brightness_24_inplace1_c(
        unsigned char [:, :, :] rgb_array_, float shift_, float [:, :, :, :] rgb_to_hsl_model):
    """
    SHADER BRIGHTNESS USING STORED RGB TO HSL VALUES (SLOWEST VERSION)

    This method is fetching all the HSL values from an array instead
    In theory this method should be faster than the direct calculation therefore the size of the
    array degrade the performance somehow.

    The Array (rgb_array) must be a numpy array shape (w, h, 3) containing RGB pixels,
    please refer to pygame function pixels3d or array3d to convert an image into a
    3d array (library surfarray)

    e.g:
    brightness_model(surface, 0.2 rgb_to_hsl_model)

    :param rgb_array_      : numpy.ndarray containing RGB values array shapes (w, h, 3) uint8
    :param shift_          : float; value in range[-1.0 ... 1.0]
    :param rgb_to_hsl_model: Array shape (r, g, b, 3) containing all pre-calculated HSL values
    :return                : void
    """

    cdef Py_ssize_t width, height
    width, height = rgb_array_.shape[:2]

    cdef:
        int i=0, j=0
        unsigned char *r
        unsigned char *g
        unsigned char *b
        float l, h, s

        rgb rgb_
        float high, low, high_

    with nogil:

        for i in prange(width, schedule='static', num_threads=THREADS):
            for j in range(height):

                r, g, b = &rgb_array_[i, j, 0], &rgb_array_[i, j, 1], &rgb_array_[i, j, 2]

                h = rgb_to_hsl_model[r[0], g[0], b[0], 0]
                s = rgb_to_hsl_model[r[0], g[0], b[0], 1]
                l = rgb_to_hsl_model[r[0], g[0], b[0], 2]

                l = min((l + shift_), <float>1.0)
                l = max(l, <float>0.0)

                rgb_ = struct_hsl_to_rgb(h, s, l)
                r[0] = <unsigned char> (rgb_.r * <float>255.0)
                g[0] = <unsigned char> (rgb_.g * <float>255.0)
                b[0] = <unsigned char> (rgb_.b * <float>255.0)




@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline void shader_saturation_array24_inplace_c(
        unsigned char [:, :, :] rgb_array_, float shift_):
    """
    SHADER SATURATION

    This shader control the saturation level of the pygame display or surface/texture

    The Array (rgb_array) must be a numpy array shape (w, h, 3)
    containing RGB pixels, please refer to pygame
    function pixels3d or array3d to convert an image into a 3d array (library surfarray)

    e.g:
    saturation(surface, 0.2)

    :param rgb_array_: numpy.ndarray shape (w, h, 3) containing RGB values uint8
    :param shift_    : float; value in range[-1.0...1.0], control the saturation level
    :return          : void
    """

    assert -1.0 <= shift_ <= 1.0, \
        "Argument shift must be in range[-1.0 ... 1.0]"

    cdef Py_ssize_t width, height
    width, height = rgb_array_.shape[:2]

    cdef:
        int i=0, j=0
        unsigned char *r
        unsigned char *g
        unsigned char *b
        float s
        hsl hsl_
        rgb rgb_


    with nogil:
        for j in prange(height, schedule='static', num_threads=THREADS):
            for i in range(width):
                r, g, b = &rgb_array_[i, j, 0], &rgb_array_[i, j, 1], &rgb_array_[i, j, 2]
                hsl_ = struct_rgb_to_hsl(<float>r[0] * <float>ONE_255,
                                         <float>g[0] * <float>ONE_255,
                                         <float>b[0] * <float>ONE_255)

                s = min((hsl_.s + shift_), <float>0.5)
                s = max(s, <float>0.0)
                rgb_ = struct_hsl_to_rgb(hsl_.h, s, hsl_.l)
                r[0] = <unsigned char>(rgb_.r * <float>255.0)
                g[0] = <unsigned char>(rgb_.g * <float>255.0)
                b[0] = <unsigned char>(rgb_.b * <float>255.0)





@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
# e.g
# heatwave_vertical(
#         surface_, numpy.full((w, h), 255, dtype=numpy.uint8),
#         b*random.uniform(55.0, 100), 0, sigma_=random.uniform(0.4, 1), mu_=b*2)
cdef inline void shader_heatwave24_vertical_inplace_c(
        unsigned char [:, :, :] rgb_array_,
        unsigned char [:, :] mask,
        float amplitude_,
        float center_,
        float sigma_,
        float mu_):
    """
    APPLY A GAUSSIAN TRANSFORMATION TO AN ARRAY

    This effect can be use for simulate air turbulence or heat flow/convection

    The Array (rgb_array) must be a numpy array shape (w, h, 3) containing RGB pixels, please refer to pygame
    function pixels3d or array3d to convert an image into a 3d array (library surfarray)


    :param rgb_array_: numpy.ndarray shape (width, height, 3) uint8 containing RGB pixels
    :param mask      : numpy.ndarray shape (x, y) uint8, (values 255 or 0).
    Apply transformation to the original array
    if the value @(x, y) is 255 else remain unchanged.
    :param amplitude_: Control the maximum of the gaussian equation.
    No transformation if factor_ equal zero
    :param center_   : Control the center of the gaussian equation (if center_ equal zero,
     the Gauss equation is centered
    at x=0 and maximum is 0.4 with amplitude_ = 1.0)
    :param sigma_    : float; sigma value of the gauss equation
    :param mu_       : float; mu value of the gauss equation
    :return          : void
    """
    # TODO MASK

    cdef Py_ssize_t w, h
    w, h = rgb_array_.shape[:2]

    cdef:
        unsigned char [:, :, :] rgb_array_copy = numpy.array(rgb_array_, copy=False)
        int x = 0, y = 0
        int yy
        int h_1 = <int>h - 1
        unsigned char *r
        float [::1] f_gauss = linspace(-4, 4, w, dtype=float32)


    with nogil:

        for x in prange(w, schedule='static', num_threads=THREADS):

            for y in range(h):

                yy =<int>(gauss(f_gauss[x], center_, sigma_, mu_) * amplitude_ + y)

                # printf("\n %i ", yy)
                if yy > h_1:
                    yy = h_1

                if yy < 0:
                    yy = 0

                rgb_array_[x, y, 0] = rgb_array_copy[x, yy, 0]
                rgb_array_[x, y, 1] = rgb_array_copy[x, yy, 1]
                rgb_array_[x, y, 2] = rgb_array_copy[x, yy, 2]




@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(False)
# e.g horizontal_glitch(surface, 0.5, 0.08, frame % 20)
cdef inline void shader_horizontal_glitch24_inplace_c(
        unsigned char [:, :, :] rgb_array_,
        float rad1_,
        float frequency_,
        float amplitude_):

    """
    SHADER GLITCH EFFECT

    Deform the pygame display to create a glitch appearance

    The Array (rgb_array) must be a numpy array shape (w, h, 3) containing RGB pixels,
    please refer to pygame function pixels3d or array3d to convert an image into a
    3d array (library surfarray)

    :param rgb_array_: numpy.ndarray shape (w, h, 3) uint8 containing RGB pixels
    :param rad1_     : float; Angle in radians, this value control the angle variation over the time
    :param frequency_:  float; signal frequency, factor that amplify the angle variation
    :param amplitude_: float; cos amplitude value
    :return:
    """
    cdef Py_ssize_t w, h
    w, h = rgb_array_.shape[:2]

    cdef:
        int i=0, j=0
        float rad = <float>(3.14/180.0)
        float angle = 0
        float angle1 = 0
        unsigned char [:, :, :] rgb_array_copy = numpy.array(rgb_array_, copy=True)
        int ii=0

    with nogil:

        for j in range(h):

            for i in range(w):

                ii = (i + <int>(<float>cos(angle) * amplitude_))
                if ii > <int>w - 1:
                    ii = <int>w - 1
                if ii < 0:
                    ii = 0

                rgb_array_[i, j, 0],\
                rgb_array_[i, j, 1],\
                rgb_array_[i, j, 2] = rgb_array_copy[ii, j, 0],\
                    rgb_array_copy[ii, j, 1], rgb_array_copy[ii, j, 2]

            angle1 = angle1 + frequency_ * rad
            angle = angle + (rad1_ * rad + rand() % angle1 - rand() % angle1)




@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline void shader_bpf24_inplace_c(
        unsigned char [:, :, :] rgb_array_, int threshold = 128, bint transpose=False):
    """
    SHADER BRIGHT PASS FILTER (INPLACE)

    Conserve only the brightest pixels in an array

    The Array (rgb_array) must be a numpy array shape (w, h, 3) containing RGB pixels,
     please refer to pygame function pixels3d or array3d to convert an image into a
     3d array (library surfarray)

    :param rgb_array_: numpy.ndarray shape (w, h, 3) uint8 containing RGB pixels
    :param threshold : integer; Bright pass threshold default 128
    :param transpose : bool; True| False transpose the final array
    :return          :  void
    """

    assert 0 <= threshold <= 255, "Argument threshold must be in range [0 ... 255]"

    cdef:
        Py_ssize_t w, h
    w, h = rgb_array_.shape[:2]

    cdef:
        int i = 0, j = 0
        float lum, c
        unsigned char *r
        unsigned char *g
        unsigned char *b

    with nogil:
        for j in prange(0, h, schedule='static', num_threads=THREADS):
            for i in range(0, w):

                # ITU-R BT.601 luma coefficients
                r = &rgb_array_[i, j, 0]
                g = &rgb_array_[i, j, 1]
                b = &rgb_array_[i, j, 2]

                lum = r[0] * <float>0.299 + g[0] * <float>0.587 + b[0] * <float>0.114

                if lum > threshold:
                    c = (lum - threshold) / lum
                    r[0] = <unsigned char>(r[0] * c)
                    g[0] = <unsigned char>(g[0] * c)
                    b[0] = <unsigned char>(b[0] * c)
                else:
                    r[0], g[0], b[0] = 0, 0, 0



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline bpf24_c(
        unsigned char [:, :, :] input_array_,
        int threshold = 128,
        bint transpose=False):
    """
    SHADER BRIGHT PASS FILTER

    Conserve only the brightest pixels in an array

    The Array (rgb_array) must be a numpy array shape (w, h, 3) containing RGB pixels,
    please refer to pygame function pixels3d or array3d to convert an image into a
    3d array (library surfarray)

    :param input_array_: numpy.ndarray shape (w, h, 3) uint8 containing RGB pixels
    :param threshold   : float Bright pass threshold default 128
    :param transpose   : bool; True| False transpose the final array
    :return            :  Return the modified array shape (w, h, 3) uint8
    """
    assert 0 <= threshold <= 255, "Argument threshold must be in range [0 ... 255]"

    cdef:
        Py_ssize_t w, h
    w, h = input_array_.shape[:2]

    cdef:
        int i = 0, j = 0
        float lum, c
        unsigned char [:, :, :] output_array_ = numpy.zeros((h, w, 3), uint8)
        unsigned char *r
        unsigned char *g
        unsigned char *b

    with nogil:
        for j in prange(0, h, schedule='static', num_threads=THREADS):
            for i in range(0, w):

                # ITU-R BT.601 luma coefficients
                r = &input_array_[i, j, 0]
                g = &input_array_[i, j, 1]
                b = &input_array_[i, j, 2]

                lum = r[0] * <float>0.299 + g[0] * <float>0.587 + b[0] * <float>0.114

                if lum > threshold:
                    c = (lum - threshold) / lum
                    output_array_[j, i, 0] = <unsigned char>(r[0] * c)
                    output_array_[j, i, 1] = <unsigned char>(g[0] * c)
                    output_array_[j, i, 2] = <unsigned char>(b[0] * c)

    return pygame.image.frombuffer(output_array_, (w, h), 'RGB')





@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef inline filtering24_c(object surface_, mask_):
    """
    MULTIPLY MASK VALUES WITH AN ARRAY REPRESENTING THE SURFACE PIXELS (COMPATIBLE 24 BIT ONLY).
    Mask values are floats in range (0 ... 1.0)

    :param surface_: pygame.Surface compatible 24-bit
    :param mask_: 2d array (MemoryViewSlice) containing alpha values (float).
    The mask_ output image is monochromatic (values range [0 ... 1.0] and R=B=G.
    :return: Return a pygame.Surface 24 bit
    """
    cdef int w, h, w_, h_
    w, h = surface_.get_size()
    try:
        w_, h_ = mask_.shape[:2]
    except (ValueError, pygame.error):
       raise ValueError(
           '\nArgument mask_ type not understood, '
           'expecting numpy.ndarray type (w, h) got %s ' % type(mask_))


    assert w == w_ and h == h_, \
        '\nSurface and mask size does not match (w:%s, h:%s), ' \
        '(w:%s, h:%s) ' % (w, h, w_, h_)

    try:
        rgb_ = pixels3d(surface_)
    except (ValueError, pygame.error):
        try:
            rgb_ = array3d(surface_)
        except (ValueError, pygame.error):
            raise ValueError('Incompatible surface.')

    cdef:
        unsigned char [:, :, :] rgb = rgb_.transpose(1, 0, 2)
        unsigned char [:, :, ::1] rgb1 = numpy.empty((h, w, 3), numpy.uint8)
        float [:, :] mask = numpy.asarray(mask_, numpy.float32)
        int i, j
    with nogil:
        for i in prange(0, w, schedule='static', num_threads=THREADS):
            for j in range(h):
                rgb1[j, i, 0] = <unsigned char>(rgb[j, i, 0] * mask[i, j])
                rgb1[j, i, 1] = <unsigned char>(rgb[j, i, 1] * mask[i, j])
                rgb1[j, i, 2] = <unsigned char>(rgb[j, i, 2] * mask[i, j])

    return frombuffer(rgb1, (w, h), 'RGB')


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline void shader_bloom_effect_array24_c(
        surface_,
        int threshold_,
        bint fast_ = False,
        object mask_ = None
):
    """
    CREATE A BLOOM EFFECT

    * Surface must be a pygame Surface 24-32 bit format

    :param surface_     : pygame.Surface; Game display or texture
    :param threshold_   : integer; Threshold value uint8 in range [0 ... 255].
    The threshold value is used by a bright
     pass filter to determine the bright pixels above the given threshold.
      Below 128 the bloom effect will be more
     noticeable and above 128 a bit less.
    :param fast_        : bool; True | False; If True the bloom effect will be approximated
    and only the x16 subsurface
    will be processed to maximize the overall processing time, default is False).
    :param mask_        : 
    :return             : void
    """

    assert 0 <= threshold_ <= 255, "Argument threshold must be in range [0 ... 255]"

    cdef:
        Py_ssize_t  w, h
        int bit_size
        int w2, h2, w4, h4, w8, h8, w16, h16
        bint x2, x4, x8, x16 = False

    w, h = surface_.get_size()
    bit_size = surface_.get_bitsize()


    w2, h2   = <int>w >> 1, <int>h >> 1
    w4, h4   = w2 >> 1, h2 >> 1
    w8, h8   = w4 >> 1, h4 >> 1
    w16, h16 = w8 >> 1, h8 >> 1

    if w16 == 0 or h16 == 0:
        raise ValueError(
            "\nImage too small and cannot be processed.\n"
            "Try to increase the size of the image")

    if w2 > 0 and h2 > 0:
        x2 = True
    else:
        x2 = False

    if w4 > 0 and h4 > 0:
        x4 = True
    else:
        x4 = False

    if w8 > 0 and h8 > 0:
        x8 = True
    else:
        x8 = False

    if w16 > 0 and h16 > 0:
        x16 = True
    else:
        x16 = False

    s2, s4, s8, s16 = None, None, None, None

    # SUBSURFACE DOWNSCALE CANNOT
    # BE PERFORMED AND WILL RAISE AN EXCEPTION
    if not x2:
        return

    if fast_:
        x2, x4, x8 = False, False, False


    # FIRST SUBSURFACE DOWNSCALE x2
    # THIS IS THE MOST EXPENSIVE IN TERM OF PROCESSING TIME
    if x2:
        s2 = scale(surface_, (w2, h2))
        s2 = bpf24_c(pixels3d(s2), threshold=threshold_)
        s2_array = numpy.array(s2.get_view('3'), dtype=numpy.uint8)
        shader_blur5x5_array24_inplace_c(s2_array)
        # b2_blurred = frombuffer(numpy.array(s2_array.transpose(1, 0, 2),
        # order='C', copy=False), (w2, h2), 'RGB')
        b2_blurred = make_surface(s2_array)
        s2 = smoothscale(b2_blurred, (w, h))
        surface_.blit(s2, (0, 0), special_flags=BLEND_RGB_ADD)

    # SECOND SUBSURFACE DOWNSCALE x4
    # THIS IS THE SECOND MOST EXPENSIVE IN TERM OF PROCESSING TIME
    if x4:
        s4 = scale(surface_, (w4, h4))
        s4 = bpf24_c(pixels3d(s4), threshold=threshold_)
        s4_array = numpy.array(s4.get_view('3'), dtype=numpy.uint8)
        shader_blur5x5_array24_inplace_c(s4_array)
        # b4_blurred = frombuffer(numpy.array(s4_array.transpose(1, 0, 2),
        # order='C', copy=False), (w4, h4), 'RGB')
        b4_blurred = make_surface(s4_array)
        s4 = smoothscale(b4_blurred, (w, h))
        surface_.blit(s4, (0, 0), special_flags=BLEND_RGB_ADD)

    # THIRD SUBSURFACE DOWNSCALE x8
    if x8:
        s8 = scale(surface_, (w8, h8))
        s8 = bpf24_c(pixels3d(s8), threshold=threshold_)
        s8_array = numpy.array(s8.get_view('3'), dtype=numpy.uint8)
        shader_blur5x5_array24_inplace_c(s8_array)
        # b8_blurred = frombuffer(numpy.array(s8_array.transpose(1, 0, 2),
        # order='C', copy=False), (w8, h8), 'RGB')
        b8_blurred = make_surface(s8_array)
        s8 = smoothscale(b8_blurred, (w, h))
        surface_.blit(s8, (0, 0), special_flags=BLEND_RGB_ADD)

    # FOURTH SUBSURFACE DOWNSCALE x16
    # LESS SIGNIFICANT IN TERMS OF RENDERING AND PROCESSING TIME
    if x16:
        s16 = scale(surface_, (w16, h16))
        s16 = bpf24_c(pixels3d(s16), threshold=threshold_)
        s16_array = numpy.array(s16.get_view('3'), dtype=numpy.uint8)
        shader_blur5x5_array24_inplace_c(s16_array)
        shader_blur5x5_array24_inplace_c(s16_array)
        # b16_blurred = frombuffer(numpy.array(s16_array.transpose(1, 0, 2),
        # order='C', copy=False), (w16, h16), 'RGB')
        b16_blurred = make_surface(s16_array)
        s16 = smoothscale(b16_blurred, (w, h))
        surface_.blit(s16, (0, 0), special_flags=BLEND_RGB_ADD)

    # todo filtering is not an inplace function and will returned a surface
    # todo Bloom is an inplace method an cannot return a surface, so filtering here
    # todo is not working ?
    if mask_ is not None:
        surface_ = filtering24_c(surface_, mask_)





@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef inline object shader_bloom_fast(
        surface_,
        int threshold_,
        bint fast_ = False,
        unsigned short int factor_ = 2
):
    """
    BLOOM EFFECT

    :param surface_  : pygame.Surface; compatible 32-24 bit containing RGB pixel values
    :param threshold_: integer; Bloom threshold value, small value cause greater bloon effect
    :param fast_     : boolean; True will increase the speed of the algorithm since only the S16 surface is blit
    :param factor_   : integer; Texture reduction value, must be in range [0, 4] and correspond to the dividing texture
        factor (div 1, div 2, div 4, div 8)
    :return          : Return a pygame Surface with the bloom effect (24 bit format)
    """

    assert isinstance(surface_, pygame.Surface), "Argument surface_ must be a pygame.Surface got %s " % type(surface_)
    assert 0 <= threshold_ <= 255, "Argument threshold must be in range [0 ... 255] got %s " % threshold_
    assert 0 <= factor_ <= 4, "Argument factor_ must be in range [0 ... 4] got %s " % factor_
    assert isinstance(fast_, bool), "Argument fast_ must be boolean True | False got %s " % type(fast_)

    cdef:
        Py_ssize_t  w, h
        int bit_size
        int w2, h2, w4, h4, w8, h8, w16, h16
        bint x2, x4, x8, x16 = False

    surface_copy = surface_.copy()
    surface_ = smoothscale(surface_,
                           (surface_.get_width() >> factor_, surface_.get_height() >> factor_))

    w, h = surface_.get_size()
    bit_size = surface_.get_bitsize()



    w2, h2   = <int>w >> 1, <int>h >> 1
    w4, h4   = w2 >> 1, h2 >> 1
    w8, h8   = w4 >> 1, h4 >> 1
    w16, h16 = w8 >> 1, h8 >> 1

    if w16 == 0 or h16 == 0:
        raise ValueError(
            "\nImage too small and cannot be processed.\n"
            "Try to increase the size of the image or decrease the factor_ value (default 2)")

    if w2 > 0 and h2 > 0:
        x2 = True
    else:
        x2 = False

    if w4 > 0 and h4 > 0:
        x4 = True
    else:
        x4 = False

    if w8 > 0 and h8 > 0:
        x8 = True
    else:
        x8 = False

    if w16 > 0 and h16 > 0:
        x16 = True
    else:
        x16 = False

    s2, s4, s8, s16 = None, None, None, None

    # SUBSURFACE DOWNSCALE CANNOT
    # BE PERFORMED AND WILL RAISE AN EXCEPTION
    if not x2:
        return

    if fast_:
        x2, x4, x8 = False, False, False


    # FIRST SUBSURFACE DOWNSCALE x2
    # THIS IS THE MOST EXPENSIVE IN TERM OF PROCESSING TIME
    if x2:
        s2 = scale(surface_, (w2, h2))
        s2 = bpf24_c(pixels3d(s2), threshold=threshold_)
        s2_array = numpy.array(s2.get_view('3'), dtype=numpy.uint8)
        shader_blur5x5_array24_inplace_c(s2_array)
        b2_blurred = make_surface(s2_array)
        s2 = smoothscale(b2_blurred, (w, h))


    # SECOND SUBSURFACE DOWNSCALE x4
    # THIS IS THE SECOND MOST EXPENSIVE IN TERM OF PROCESSING TIME
    if x4:
        s4 = scale(surface_, (w4, h4))
        s4 = bpf24_c(pixels3d(s4), threshold=threshold_)
        s4_array = numpy.array(s4.get_view('3'), dtype=numpy.uint8)
        shader_blur5x5_array24_inplace_c(s4_array)
        b4_blurred = make_surface(s4_array)
        s4 = smoothscale(b4_blurred, (w, h))


    # THIRD SUBSURFACE DOWNSCALE x8
    if x8:
        s8 = scale(surface_, (w8, h8))
        s8 = bpf24_c(pixels3d(s8), threshold=threshold_)
        s8_array = numpy.array(s8.get_view('3'), dtype=numpy.uint8)
        shader_blur5x5_array24_inplace_c(s8_array)
        # order='C', copy=False), (w8, h8), 'RGB')
        b8_blurred = make_surface(s8_array)
        s8 = smoothscale(b8_blurred, (w, h))


    # FOURTH SUBSURFACE DOWNSCALE x16
    # LESS SIGNIFICANT IN TERMS OF RENDERING AND PROCESSING TIME
    if x16:
        s16 = scale(surface_, (w16, h16))
        s16 = bpf24_c(pixels3d(s16), threshold=threshold_)
        s16_array = numpy.array(s16.get_view('3'), dtype=numpy.uint8)
        shader_blur5x5_array24_inplace_c(s16_array)
        shader_blur5x5_array24_inplace_c(s16_array)
        b16_blurred = make_surface(s16_array)
        s16 = smoothscale(b16_blurred, (w, h))

    if fast_:
        s16 = smoothscale(s16, (w << factor_, h << factor_))
        surface_copy.blit(s16, (0, 0), special_flags=BLEND_RGB_ADD)
    else:
        s2.blit(s4, (0, 0), special_flags=BLEND_RGB_ADD)
        s2.blit(s8, (0, 0), special_flags=BLEND_RGB_ADD)
        s2.blit(s16, (0, 0), special_flags=BLEND_RGB_ADD)
        s2 = smoothscale(s2, (w << factor_, h << factor_))
        surface_copy.blit(s2, (0, 0), special_flags=BLEND_RGB_ADD)

    return surface_copy



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef void shader_bloom_fast1(
        object surface_,
        unsigned short int smooth_ = 3,
        unsigned int threshold_ = 0,
        unsigned short int flag_ = BLEND_RGB_ADD,
        bint saturation_ = False
):
    """
    BLOOM EFFECT (SIMPLIFY VERSION) 
    
    This version is compatible with moving object in the display.
    
    The other bloom versions cause the halo of light to be offset from moving 
    objects due to the re-scaling (down sampling) of the sub-surfaces and the loss 
    of accuracy.
    
    The quantity of bloom can be adjust with the bright pass filter threshold
     (variable threshold_). The variable can help you to choose what level of 
     light can trigger a bloom effect. 
    The lowest the value the brightest the effect. 
    
    The smooth factor will help to spread the light homogeneously around the objects. 
    A large number of smooth will cast the bloom over the entire scene and diminished 
    the overall bloom effect, while a small value will pixelate the hallo around objects 
    but will generate the brightest effect on objects. 
    When smooth is below 3, the halo appear to be slightly pixelated. 
    
    You can use the saturation to generate saturated colors within the light 
    effect.
    The flag can be used to create different special effect with the light within the 
    pygame display. The default value is pygame.BLEND_RGB_ADD and allow to blend 
    the bloom to the display. 
    Nevertheless you cann also use any of the other flags such as BLEND_RGB_MAX, BLEND_RGB_SUB etc 
    
    This effect is applied inplace
    
    :param surface_    : pygame.Surface; compatible 32-24 bit containing RGB pixel values 
    :param smooth_     : integer; Smooth the hallow default 3 (gaussian kernel)
    :param threshold_  : integer; BPF threshold default 20
    :param flag_       : integer; pygame flag to use (default is BLEND_RGB_ADD)
    :param saturation_ : bool; True | False include saturation effect to the halo  
    :return            : void 
    """

    cdef:
        Py_ssize_t  w, h
        unsigned int bit_size
        unsigned int  w16, h16
        int r

    assert isinstance(surface_, pygame.Surface), \
        "Argument surface_ must be a pygame.Surface got %s " % type(surface_)
    if flag_ < 0:
        raise ValueError("Argument flag_ cannot be < 0")
    if smooth_ < 0:
        raise ValueError("Argument smooth_ cannot be < 0")

    threshold_ %= 255

    w, h = surface_.get_size()
    bit_size = surface_.get_bitsize()

    w16, h16 = w >> 4, h >> 4

    if w16 == 0 or h16 == 0:
        raise ValueError(
            "\nImage too small and cannot be processed.\n"
                "Try to increase the size of the image")

    s2 = smoothscale(surface_, (w16, h16))
    s2.blit(s2, (0, 0), special_flags=BLEND_RGB_ADD)

    cdef unsigned char [ :, :, : ] s2_array = pixels3d(s2)
    shader_bpf24_inplace_c(s2_array, threshold=threshold_)

    for r in range(smooth_):
        shader_blur5x5_array24_inplace_c(s2_array)
        if saturation_ : shader_saturation_array24_inplace_c(s2_array, <float>0.3)

    b2_blurred = make_surface(numpy.asarray(s2_array))
    s2 = smoothscale(b2_blurred, (w, h))

    if flag_ is not None and flag_!=0:
        surface_.blit(s2, (0, 0), special_flags=flag_)
    else:
        surface_.blit(s2, (0, 0))


# cdef unsigned int [:, :, ::1] IMAGE_FISHEYE_MODEL = numpy.zeros((800, 1024, 2), uint32)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline shader_fisheye24_footprint_c(
        Py_ssize_t w,
        Py_ssize_t h,
        unsigned int centre_x,
        unsigned int centre_y
):

    """
    CREATE A FISHEYE MODEL TO HOLD THE PIXEL COORDINATES OF A SURFACE/ GAME DISPLAY

    * The surface and the model must have the same dimensions.

    Store the fisheye model into an external array image_fisheye_model shape (width, height, 2)

    IMAGE_FISHEYE_MODEL contains the fisheye transformation coordinate (x2 & y2) that reference
    the final image pixel position (fisheye model)
    This method has to be call once before the main loop in order to calculate
    the projected position for each pixels.

    :param centre_y: integer; centre y of the effect   
    :param centre_x: integer; centre x of the effect 
    :param w       : integer; width of the model
    :param h       : integer; height of the model
    :return        : Return a numpy.ndarray type (w, h, 2) representing the fisheye model (coordinates
        of all surface pixels passing through the fisheye lens model)
    """

    assert w > 0, "Argument w must be > 0"
    assert h > 0, "Argument h must be > 0"

    cdef:
        unsigned int [:, :, :] image_fisheye_model = numpy.zeros((w, h, 3), numpy.uint32)
        int y=0, x=0, v
        float ny, ny2, nx, nx2, r, theta, nxn, nyn, nr
        int x2, y2
        float s = <float>w * <float>h
        float c1 = <float>2.0 / <float>h
        float c2 = <float>2.0 / <float>w
        float w2 = <float>w * <float>centre_x/<float>w
        float h2 = <float>h * <float>centre_y/<float>h

    with nogil:
        for x in prange(w, schedule='static', num_threads=THREADS):
            nx = x * c2 - <float>1.0
            nx2 = nx * nx
            for y in range(h):
                ny = <float>y * c1 - <float>1.0
                ny2 = ny * ny
                r = <float>sqrt(nx2 + ny2)
                if 0.0 <= r <= 1.0:
                    nr = (r + <float>1.0 - <float>sqrt(1.0 - (nx2 + ny2))) * <float>0.5
                    if nr <= 1.0:
                        theta = <float>atan2(ny, nx)
                        nxn = nr * <float>cos(theta)
                        nyn = nr * <float>sin(theta)
                        x2 = <int>(nxn * w2 + w2)
                        y2 = <int>(nyn * h2 + h2)
                        v = <int>(y2 * w + x2)
                        image_fisheye_model[x, y, 0] = x2
                        image_fisheye_model[x, y, 1] = y2
                        image_fisheye_model[x, y, 2] = 0
                else:
                    image_fisheye_model[x, y, 0] = 0
                    image_fisheye_model[x, y, 1] = 0
                    image_fisheye_model[x, y, 2] = 0

    return asarray(image_fisheye_model)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline void shader_fisheye24_inplace_c(
        unsigned char [:, :, :] rgb_array_, unsigned int [:, :, :] fisheye_model):
    """
    THIS SHADER CAN BE USE TO DISPLAY THE GAME THROUGH A LENS EFFECT

    Display a fisheye effect in real time given a numpy ndarray referencing the
    pixels RGB of a Pygame.Surface. In order to accomplish a real time
    calculation, this algorithm is using a pre-calculated transformation stored
    in the array fisheye_model.
    The function shader_fisheye24_footprint_c has to be called prior
    shader_fisheye24_inplace_c in order to store the transformation values.

    This shader can be applied directly to the pygame display

    The Array (rgb_array) must be a numpy array shape (w, h, 3) containing RGB pixels,
     please refer to pygame
    function pixels3d or array3d to convert an image into a 3d array (library surfarray)

    :param rgb_array_       : numpy.ndarray shape (width, height, 3) containing RGB pixels
    :param fisheye_model    : numpy.ndarray shape (width, height, 2) int32, fisheye model
    containing the pixels
    coordinates after the fisheye transformation
    :return                 : void
    """

    cdef:
        Py_ssize_t w, h
    w, h = rgb_array_.shape[:2]

    cdef:
        int x, y
        unsigned char [:, :, :] rgb_array_copy = numpy.array(rgb_array_, copy=False, order='C')
        unsigned int *x2
        unsigned int *y2

    with nogil:
        for x in prange(w, schedule='static', num_threads=THREADS):
            for y in range(h):

                x2 = &fisheye_model[x, y, 0]
                y2 = &fisheye_model[x, y, 1]

                if x2[0]!=0 and y2[0]!=0:

                    rgb_array_[x, y, 0] = rgb_array_copy[x2[0] % w, y2[0] % h, 0]
                    rgb_array_[x, y, 1] = rgb_array_copy[x2[0] % w, y2[0] % h, 1]
                    rgb_array_[x, y, 2] = rgb_array_copy[x2[0] % w, y2[0] % h, 2]




@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline tuple shader_rain_footprint_inplace_c(Py_ssize_t w, Py_ssize_t h):
    """
    CREATE A FISH EYE LENS DEFORMATION MAP/TEXTURE

    * This function create a texture and its equivalent numpy.ndarray containing the coordinates
      for each pixels after deformation.
    * This method must be called once outside of your game main loop
    * The model can be re-use to display your video game animation without being re-calculated for
      each frame. This method allow a high fps rate

    :param w    : integer; Width of the fish eye effect
    :param h    : integer; height of the fish eye effect
    :return     : Pygame Surface representing the fish-eye effect and the equivalent numpy.ndarray
    """
    assert w > 0, "Argument w must be > 0"
    assert h > 0, "Argument h must be > 0"

    cdef:
        unsigned int [:, :, ::1] rain_fisheye_model = numpy.zeros((w, h, 3), numpy.uint)
        int y=0, x=0, v
        float ny, ny2, nx, nx2, r, theta, nxn, nyn, nr
        int x2, y2
        float s = <float>w * <float>h
        float c1 = <float>2.0 / <float>h
        float c2 = <float>2.0 / <float>w
        float w2 = <float>w * <float>0.5
        float h2 = <float>h * <float>0.5

    with nogil:
        for x in prange(w, schedule='static', num_threads=THREADS):
            nx = x * c2 - <float>1.0
            nx2 = nx * nx
            for y in range(h):
                ny = y * c1 - <float>1.0
                ny2 = ny * ny
                r = <float>sqrt(nx2 + ny2)
                if 0.0 <= r <= 1.0:
                    nr = (r + <float>1.0 - <float>sqrt(1.0 - (nx2 + ny2))) * <float>0.45
                    if nr <= 1.0:
                        theta = <float>atan2(ny, nx)
                        nxn = nr * <float>cos(theta)
                        nyn = nr * <float>sin(theta)
                        x2 = <int>(nxn * w2 + w2)
                        y2 = <int>(nyn * h2 + h2)
                        v = <int>(y2 * w + x2)
                        rain_fisheye_model[x, y, 0] = x2
                        rain_fisheye_model[x, y, 1] = y2
                        rain_fisheye_model[x, y, 2] = 0

    return make_surface(asarray(rain_fisheye_model)), asarray(rain_fisheye_model)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline void shader_rain_fisheye24_inplace_c(
        unsigned char [:, :, :] rgb_array_,
        unsigned int [:, :, ::1] rain_fisheye_model
):
    """
    THIS SHADER CAN BE USED TO SIMULATE RAIN DROPLET OR BUBBLE DISPLAYED ON THE TOP OF
    THE SCREEN SURFACE.

    Both array rgb_array_ and rain_fisheye_model must have the same size

    The Array (rgb_array) must be a numpy array shape (w, h, 3) containing RGB pixels,
    please refer to pygame
    function pixels3d or array3d to convert an image into a 3d array (library surfarray)

    1) Always call the method shader_rain_footprint_inplace_c before the main loop.
       The transformation model doesn't have to be calculated every frames.
       The above method will generate a pygame texture (24bit) containing the location
       of each pixels after deformation. It does also return a numpy.ndarray equivalent
       pixel format that can be used instead of the surface if needed.

    It uses a fish eye lens deformation to reproduce the deformed background image onto
    the final image. The operation apply inplace and the surface referenced by the rgb_array_
    will be modified directly.
    The fish-eye lens deformation will recreate you game scene into the rain droplet or bubble
    and create the illusion of animation inside the bubble.

    * This shader cannot be applied directly to the pygame display as the array passed to the
    function is a scaled format of the pygame.display (copy not referencing directly the
    surface pixels)

    * This algorithm use a pre-calculated fish-eye lens deformation model to boost the overall
    fps performances, the texture pixel is then transformed with the model without any intensive
    math calculation.

    :param rgb_array_           : numpy.ndarray type (w, h, 3) uint8 (unsigned char 0...255)
    Array referencing a
    pygame surface (compatible with 24bit format only RGB model)
    :param rain_fisheye_model   : numpy.ndarray type (w, h, 3) unsigned int containing the
    the coordinate for each pixels
    :return                     : void
    """

    cdef:
        Py_ssize_t w, h
    w, h = rgb_array_.shape[:2]

    cdef:
        int x, y
        unsigned char [:, :, ::1] rgb_array_copy = numpy.array(rgb_array_, copy=True, order='C')
        unsigned int *x2
        unsigned int *y2

    with nogil:
        for x in prange(w, schedule='static', num_threads=THREADS):
            for y in range(h):
                x2 = &rain_fisheye_model[x, y, 0]
                y2 = &rain_fisheye_model[x, y, 1]

                rgb_array_[x, y, 0] = rgb_array_copy[x2[0], y2[0], 0]
                rgb_array_[x, y, 1] = rgb_array_copy[x2[0], y2[0], 1]
                rgb_array_[x, y, 2] = rgb_array_copy[x2[0], y2[0], 2]





@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline void shader_tv_scanline_inplace_c(
        unsigned char [:, :, :] rgb_array_,
        int frame_):
    """
    SHADER CREATING A TV SCANLINE EFFECT ON PYGAME SURFACE

    The space between each scanline can by adjusted with the frame_ value.
    The scanline intensity/colors is lower that the original pixel value 
    
    * This shader can be apply directly to the pygame display as long as rgb_array_ array reference
    directly the screen pixels

    The Array (rgb_array) must be a numpy array shape (w, h, 3) containing RGB pixels,
    please refer to pygame
    function pixels3d or array3d to convert an image into a 3d array (library surfarray)

    :param rgb_array_   : numpy.ndarray shape (w, h, 3) containing RGB pixels
    :param frame_       : integer; Frame numbre (linear value)
    :return             : void
    """


    cdef:
        Py_ssize_t w, h
    w, h = rgb_array_.shape[:2]

    cdef:
        int x, y, j
        unsigned char *r
        unsigned char *g
        unsigned char *b
        int frame_2 = frame_ >> 1

    with nogil:
        for y in prange(0, h, frame_, schedule='static', num_threads=THREADS):
            for x in range(w):
                for j in range(frame_2):
                    if y + j < h - 1:
                        r = &rgb_array_[x, y + j, 0]
                        g = &rgb_array_[x, y + j, 1]
                        b = &rgb_array_[x, y + j, 2]
                    else:
                        r = &rgb_array_[x, y, 0]
                        g = &rgb_array_[x, y, 1]
                        b = &rgb_array_[x, y, 2]
                    r[0] = <unsigned char> (r[0] * <float>0.65)
                    g[0] = <unsigned char> (g[0] * <float>0.65)
                    b[0] = <unsigned char> (b[0] * <float>0.65)


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline void shader_rgb_split_inplace_c(object surface_, int offset_):
    """
    THIS SHADER CREATE AN RGB SPLIT EFFECT (SUPERPOSED CHANNEL R, G, B WITH GIVEN OFFSET)
    The transformation apply inplace

    The original surface will be used and used for the subsurface blit operation.
    Each channels will be blit sequentially in the following order RGB
    Note that channel green and blue will be blit with an additional flag BLEND_RGB_ADD, to mix
    the channel with the lower layers.

    * FPS BOOST
    In order to boost the fps frame rate the original surface to process can be downscale x2
    and rescale after processing.

    * This shader can be apply directly to the pygame display by passing the screen equ
    surface to the
    method. This is true if the surface passed to the method is not a screen copy or a
    modified/altered
    surface (e.g downscale / upscale surface)

    :param surface_ : pygame Surface to process (24bit format)
    :param offset_  : integer; offset to add to each channels RGB
    :return         : void
    """
    cdef:
        Py_ssize_t w, h
    w, h = surface_.get_size()

    cdef:
        int i, j
        int z = <int>h * <int>w * 3
        unsigned char [:] rgb   = numpy.frombuffer(surface_.get_buffer(), dtype=numpy.uint8)
        unsigned char [:] red   = numpy.zeros(z, uint8, order='C')
        unsigned char [:] green = numpy.zeros(z, uint8, order='C')
        unsigned char [:] blue  = numpy.zeros(z, uint8, order='C')


    with nogil:

        for i in prange(0, w * h * 4, 4,  schedule='static', num_threads=THREADS, chunksize=8):
            j = (i >> 2) * 3
            red[j]     = rgb[i + 2]
            green[j+1] = rgb[i + 1]
            blue[j+2]  = rgb[i    ]

    del rgb
    surface_.blit(fromstring(bytes(red), (w, h), 'RGB'), (0, 0))
    surface_.blit(fromstring(bytes(green), (w, h), 'RGB'), (offset_, offset_),
                  special_flags=BLEND_RGB_ADD)
    surface_.blit(fromstring(bytes(blue), (w, h), 'RGB'), (offset_ << 1, offset_ << 1),
                  special_flags=BLEND_RGB_ADD)



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef shader_rgb_split_c(object surface_, int offset_):
    """
    THIS SHADER CREATE AN RGB SPLIT EFFECT (SUPERPOSED CHANNEL R, G, B WITH GIVEN OFFSET)

    The final image has a different width and height since the offset value is removed to keep only 
    the overlapping R, G, B channels 
    Setting the Offset_ to zero will have no effect to the original image.

    :param surface_ : pygame Surface to process (24bit format)
    :param offset_  : integer; offset to add to each channels RGB
    :return         : void
    """

    cdef:
        Py_ssize_t w, h
    w, h = surface_.get_size()

    cdef:
        int i, j
        int z = <int>h * <int>w * 3
        unsigned char [:] rgb   = numpy.frombuffer(surface_.get_buffer(), dtype=numpy.uint8)
        unsigned char [:] red   = numpy.zeros(z, uint8, order='C')
        unsigned char [:] green = numpy.zeros(z, uint8, order='C')
        unsigned char [:] blue  = numpy.zeros(z, uint8, order='C')

    # Create a new surface (sizes - offset)
    new_surface = Surface((w-offset_, h-offset_))
    new_surface.convert()

    with nogil:

        for i in prange(0, w * h * 4, 4,  schedule='static', num_threads=THREADS, chunksize=8):
            j = (i >> 2) * 3
            red[j]     = rgb[i + 2]
            green[j+1] = rgb[i + 1]
            blue[j+2]  = rgb[i    ]

    del rgb

    red_ = fromstring(bytes(red), (w, h), 'RGB')
    green_ = fromstring(bytes(green), (w, h), 'RGB')
    blue_ = fromstring(bytes(blue), (w, h), 'RGB')

    new_surface.blit(red_, (-offset_, -offset_), special_flags=BLEND_RGB_ADD)
    new_surface.blit(green_, (0, 0), special_flags=BLEND_RGB_ADD)
    new_surface.blit(blue_, (offset_, offset_), special_flags=BLEND_RGB_ADD)
    return new_surface


cpdef inline void putmask_c(
        int [:, :, :] array_, Py_ssize_t rows_, Py_ssize_t cols_, int v_=0)nogil:
    """
    EQUIVALENT METHOD TO numpy.putmask BUT MUCH FASTER FOR OPERATION WITH OPERAND < V_

    * Cython cpdef function, this function can be called directly and do not require a
      hook function.

    numpy.putmask(array_, array_<0, 0) --> putmask_c(array_, w, h, 0)

    :param array_   : numpy.ndarray shape (w, h, 3) of integer
    :param rows_    : integer;
    :param cols_    : integer;
    :param v_       : Value for filter < v_
    :return         : void
    """
    cdef:
        int i, j
        int *r
        int *g
        int *b

    for i in prange(0, rows_, schedule='static', num_threads=THREADS):
        for j in range(0, cols_):
            r = &array_[i, j, 0]
            g = &array_[i, j, 1]
            b = &array_[i, j, 2]
            if r[0] < 0:
                r[0] = 0
                g[0] = 0
                b[0] = 0

            # if r[0] > 255:
            #     r[0] = 255
            # if g[0] > 255:
            #     g[0] = 255
            # if b[0] > 255:
            #     b[0] = 255



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline tuple shader_ripple_c(
       Py_ssize_t rows_, Py_ssize_t cols_,
       float [:, ::1] previous,
       float [:, ::1] current,
       unsigned char [:, :, ::1] array,
       ):
    """
    THIS SHADER CREATE A WATER EFFECT ON A PYGAME SURFACE
    This version does not include any background deformation to keep a reasonable fps rate

    * GLOBAL VARIABLE CONSIDERATION
    Set 3 arrays as global numpy arrays (not in the main loop, these arrays does not have
    to be created every frames).

    current = numpy.zeros((width, height), dtype=numpy.float32)
    previous = numpy.zeros((width, height), dtype=numpy.float32)
    array = numpy.full((width, height, 3), 0, numpy.uint8)

    * ADD A WATER DROP
    To add a water drop to the display just add an integer value to the array previous such as
    previous[random.randint(0, width - 1), random.randint(0, height - 1)] = 1024
    with width and height matching the size of the array (width, height = previous.get_size())

    * PROCESSING TIME
    Then update the transformation. The function below will start the blurring process
    (flattening the
    values across the array to absorb the drop energy ; 1024 in the example above.
    Finally the arrays are swapped : current become previous and previous become current
    Note that:
    1) w & h must match the array size
    2) previous & current & array must be identical sizes otherwise an error will be thrown

    previous, current, array = shader_ripple_c(w, h, previous, current, array)

    * CREATING THE SURFACE
    The array is then transformed into a numpy.ndarray of type uint8 (unsigned char 0..255)
    The pygame make_surface method will convert the array type (w, h, 3) into a surface that can
    be blit to the active display.
    surf = make_surface(asarray(array, dtype=uint8))

    * BLIT

    The newly surface containing the drops can then be added to the current background or
    display (blit process).
    Both surface should have the same size for a more realistic effect
    The special_flag here is BLEND_RGBA_ADD but can be set to a different value to achieve the
    special effect needed (BLEND_RGB_MULT, BLEND_RGB_SUB) etc.
    Refer to blending modes to understand the math operation behind the flags

    surface_.blit(surf, (0, 0), special_flags=pygame.BLEND_RGBA_ADD)

    * BOOSTING THE FPS
    In order to boost the processing time, I downscale the surface prior processing and
    upscale it x2 when
    the process is done, this method provide an additional overall performance.

    * NOTE this shader cannot be apply to the screen directly (screen referencing
    pygame.display.get_surface()),


    :param rows_        : integer; Array width
    :param cols_        : integer; Array height
    :param previous     : numpy.ndarray type (w, h) type float; array use for the transformation
    :param current      : numpy.ndarray type (w, h) type float; array use for the transformation
    :param array        : numpy.ndarray type (w, h, 3) type unsigned char
    :return             : Return a tuple containing 3 arrays
    """

    cdef:
        int i, j, a, b
        float data
        float *c
        float *d
        unsigned char *e
        float r

    with nogil:

        for i in prange(1, rows_ - 1, schedule='static', num_threads=THREADS):
            for j in prange(1, cols_ - 1, schedule='static', num_threads=THREADS):

                # data = <int>(previous[i + 1, j] + previous[i - 1, j] +
                #         previous[i, j - 1] + previous[i, j + 1])  >> 1

                data = (previous[i + 1, j] + previous[i - 1, j] +
                              previous[i, j - 1] + previous[i, j + 1]) * <float>0.5
                c = &current[i, j]
                data = data - <float>c[0]
                data = data - (data * <float>0.011)
                c[0] = data
                d = &previous[i,j]
                e = &array[i, j, 0]
                e[0] = <unsigned char> d[0] if d[0] > 0 else 0
                array[i, j, 1] = e[0]
                array[i, j, 2] = e[0]

    return current, previous, array



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(False)
cpdef tunnel_modeling32(Py_ssize_t screen_width, Py_ssize_t screen_height):
    """
    THIS METHOD CREATE A TUNNEL MODEL

    * This method must be called before rendering the tunnel in order to create
      all the necessary buffers that will be called during the rendering of the tunnel effect.
      tunnel_modeling32 must be call once only before the main loop of your game.

    * Cython cpdef function, this function can be called directly and do not require a
      hook function.

    * The default tunnel texture is internally loaded (the texture is 256x256 pixels 24-bit)
      check the line :
      surface = pygame.image.load("Assets\\Graphics\\Background\\space1.jpg").convert()

    * This algorithm uses a 256x256 texture but reshape it to 512x512 pixels for a
    better effect definition

    :param screen_width     : integer; Game display size (width in pixels)
    :param screen_height    : integer; Game display size (height in pixels)
    :return                 : return a tuple containing the following (distances, angles,
    shades, scr_data)
    distances is a numpy.ndarray buffer containing float values representing the distance
    of each pixels
    angles is a numpy.ndarray buffer containing float values representing the angle of each pixels
    shades is a numpy.ndarray buffer containing float values representing the shade of each pixels
    scr_data is a numpy.ndarray buffer containing uint8 values representing the BGR (not RGB)
    values of each pixels

    """

    assert screen_width > 0, "Argument screen_width must be > 0"
    assert screen_height > 0, "Argument screen_height must be > 0"

    cdef int [:] distances = numpy.empty((screen_width * screen_height * 4), int32)
    cdef int [:] angles    = numpy.empty((screen_width * screen_height * 4), int32)
    cdef int [:] shades    = numpy.empty((screen_width * screen_height * 4), int32)

    surface = pygame.image.load("../Assets/space1.jpg").convert()

    cdef int s_width  = 512
    cdef int s_height = 512
    surface = smoothscale(surface, (s_width, s_height))
    cdef unsigned char [::1] scr_data = surface.get_buffer()

    cdef  int x, y, i = 0


    for y in range(0, screen_height * 2):
        sqy = pow(y - screen_height, 2)
        for x in range(0, screen_width * 2):
            sqx = pow(x - screen_width, 2)
            if (sqx + sqy) == 0:
                distances[i] = 1
            else:
                distances[i] = <int>(floor(
                    32 * <float>s_height / <float>sqrt(sqx + sqy))) % s_height
            angles[i]    = <int>round_c(<float>s_width * atan2(y - screen_height,
                                                             x - screen_width) / (<float>M_PI))
            shades[i]    = <int>min(sqrt(sqx + sqy)*10, 255)
            i = i + 1

    return distances, angles, shades, scr_data




@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef tunnel_render32(
        int t,
        Py_ssize_t screen_width,
        Py_ssize_t screen_height,
        int screen_w2,
        int screen_h2,
        int [::1] distances,
        int [::1] angles,
        int [::1] shades,
        unsigned char [::1] scr_data,
        unsigned char [::1] dest_array):

    """
    TUNNEL EFFECT RENDERING METHOD

    * Always call the method tunnel_modeling32 outside of your game main loop
    before calling this method (from
      the main loop)

    * Return a pygame surface containing the tunnel rendering effect, the image
    is 32 bit (with per-pixel information)

    * Cython cpdef function, this function can be called directly and do not
    require a hook function.

    * The parameter t must change overtime

    :param t            : integer; linear value (frame number)
    :param screen_width : integer; Game display size (width in pixels)
    :param screen_height: integer; Game display size (height in pixels)
    :param screen_w2    : integer; Game display width / 2.0
    :param screen_h2    : integer; game display height / 2.0
    :param distances    : numpy.ndarray buffer containing float values
    representing the distance of each pixels
    :param angles       : numpy.ndarray buffer containing float values
    representing the angle of each pixels
    :param shades       : numpy.ndarray buffer containing float values
    representing the shade of each pixels
    :param scr_data     : numpy.ndarray buffer containing float values
    representing the BGR values  of each pixels
    :param dest_array   : numpy.ndarray buffer containing float values
    representing the RGB values of each pixels
    :return             : Return a pygame.Surface (w, h) 32 bit with per-pixel information
    """

    assert screen_width > 0, "Argument screen_width must be > 0"
    assert screen_height > 0, "Argument screen_height must be > 0"
    assert screen_w2 > 0, "Argument screen_w2 must be > 0"
    assert screen_h2 > 0, "Argument screen_h2 must be > 0"

    cdef:
        int s_width  = 512
        int s_height = 512
        float timer = t * <float>1e-3
        int shiftx  = <int>floor(s_width * timer)
        int shifty  = <int>floor(s_height * <float>0.25 * timer)
        int centerx = <int>(screen_w2 +
                            <float>floor((screen_w2 >> 1) * <float>sin(timer * <float>0.25)))
        int centery = <int>(screen_h2 + <float>floor(
            (screen_h2 >> 1) * <float>sin(timer * <float>0.5)))
        int stride  = <int>screen_width * 2
        int dest_ofs = 0
        int src_ofs
        int u, v, x, y
        int pix_ofs, shade

    # with nogil:
    #     for y in range(0,  screen_height):
    #         srcOfs = y * stride + centerx + centery * stride
    #         for x in range(0, screen_width):
    #             u = (distances[srcOfs] + shiftx) & 0xff
    #             v = (angles[srcOfs] + shifty) & 0xff
    #             while v < 0:
    #               v = v + s_height
    #
    #             shade = <int>(shades[srcOfs] * ONE_255)
    #
    #             pixOfs = (u + (v << 9)) << 3
    #             dest_array[dest_ofs    ] = scr_data[pixOfs + 2] * shade
    #             dest_array[dest_ofs + 1] = scr_data[pixOfs + 1] * shade
    #             dest_array[dest_ofs + 2] = scr_data[pixOfs + 0] * shade
    #             dest_array[dest_ofs + 3] = 255 # scr_data[pixOfs + 4] * shade
    #
    #             dest_ofs = dest_ofs + 4
    #             srcOfs  = srcOfs + 1
    #
    # return pygame.image.frombuffer(dest_array, (screen_width, screen_height), "RGBA")

    with nogil:

        for y in prange(0, screen_height, schedule='static', num_threads=4):

            for x in range(0, screen_width):

                src_ofs = y * stride + centerx + centery * stride + x
                dest_ofs = (y * screen_height + x) << 2

                u = (distances[src_ofs] + shiftx) & 0xff
                v = (angles[src_ofs] + shifty) & 0xff

                shade = <int> (shades[src_ofs] * ONE_255)

                pix_ofs = (u + (v << 9)) << 3

                dest_array[dest_ofs] = scr_data[pix_ofs + 2] * shade
                dest_array[dest_ofs + 1] = scr_data[pix_ofs + 1] * shade
                dest_array[dest_ofs + 2] = scr_data[pix_ofs + 0] * shade
                dest_array[dest_ofs + 3] = 255

    return pygame.image.frombuffer(dest_array, (screen_width, screen_height), "RGBA").convert_alpha()




# @cython.binding(True)
# @cython.boundscheck(True)
# @cython.wraparound(True)
# @cython.nonecheck(True)
# @cython.cdivision(False)
# cpdef tunnel_modeling24(int screen_width, int screen_height):
#
#     cdef int [:] distances = numpy.empty((screen_width * screen_height * 3), int32)
#     cdef int [:] angles    = numpy.empty((screen_width * screen_height * 3), int32)
#     cdef int [:] shades    = numpy.empty((screen_width * screen_height * 3), int32)
#
#     surface = pygame.image.load("Assets\\Graphics\\Background\\space1.jpg").convert()
#
#     cdef int s_width  = 512
#     cdef int s_height = 512
#     surface = pygame.transform.smoothscale(surface, (s_width, s_height))
#     cdef unsigned char [:] scr_data = numpy.frombuffer(
#     pygame.image.tostring(surface, 'RGB'), uint8).copy()
#
#     cdef  int x, y, i = 0
#
#     for y in range(0, screen_height * 2):
#         sqy = pow(y - screen_height, 2)
#         for x in range(0, screen_width * 2):
#             sqx = pow(x - screen_width, 2)
#
#             i /= 4
#             i *= 3
#             if (sqx + sqy) == 0:
#                 distances[i] = 1
#             else:
#                 distances[i] = <int>(
#                 floor(32 * <float>s_height / <float>sqrt(sqx + sqy))) % s_height
#             angles[i]    = <int>round(
#             <float>s_width * atan2(y - screen_height, x - screen_width) / M_PI)
#             shades[i]    = <int>min(sqrt(sqx + sqy)*10, 255)
#             i = i + 1
#
#     return distances, angles, shades, scr_data
#

# @cython.binding(True)
# @cython.boundscheck(True)
# @cython.wraparound(True)
# @cython.nonecheck(True)
# @cython.cdivision(True)
# cpdef tunnel_render24(int t,
#                     int screen_width,
#                     int screen_height,
#                     int screen_w2,
#                     int screen_h2,
#                     int [:] distances,
#                     int [:] angles,
#                     int [:] shades,
#                     unsigned char [:] scr_data,
#                     unsigned char [:] dest_array):
#     cdef:
#         int s_width  = 512
#         int s_height = 512
#         float timer = t * 1e-3
#         int shiftx  = <int>floor(s_width * timer)
#         int shifty  = <int>floor(s_height * 0.25 * timer)
#         int centerx = <int>(screen_w2 + floor((screen_w2 >> 1) * sin(timer * 0.25)))
#         int centery = <int>(screen_h2 + floor((screen_h2 >> 1) * sin(timer * 0.5)))
#         int stride  = screen_width * 2
#         int destOfs = 0
#         int srcOfs
#         int u, v, x, y
#         int pixOfs, shade
#
#
#     with nogil:
#
#         for y in prange(0,  screen_height,  schedule='static', num_threads=THREADS):
#             for x in range(0, screen_width):
#
#
#                 srcOfs = (y * stride * 3 + centerx + centery * stride + x * 3)
#
#                 destOfs = (y * screen_height*3 + x * 3)
#
#
#                 u = (distances[srcOfs] + shiftx) & 0xff
#                 v = (angles[srcOfs] + shifty) & 0xff
#
#
#                 #shade = <int>(shades[srcOfs] * ONE_255)
#
#                 pixOfs = (u + (v << 9)) << 1
#
#                 dest_array[destOfs    ] = scr_data[pixOfs + 2] #* shade
#                 dest_array[destOfs + 1] = scr_data[pixOfs + 1] #* shade
#                 dest_array[destOfs + 2] = scr_data[pixOfs + 0] #* shade
#
#
#     return pygame.image.frombuffer(dest_array, (screen_width, screen_height), "RGB")





@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef tuple heat_map(int wavelength, float gamma=1.0):
    """
    RETURN AN RGB COLOR VALUE MATCHING A SPECIFIC WAVELENGTH

    Cython cpdef function, this function can be called directly and do not require a
    hook function.

    This function return a tuple (R,G,B) corresponding to the
    color wavelength (wavelength_to_rgb is an External C
    routine with pre-defined wavelength such as :
    Color   Wavelength(nm) Frequency(THz)
    Red     620-750        484-400
    Orange  590-620        508-484
    Yellow  570-590        526-508
    Green   495-570        606-526
    Blue    450-495        668-606
    Violet  380-450        789-668
    e.g If the routine is called with a wavelength of 620, the returned color
    will be a red gradient

    :param wavelength   : integer; Wavelength
    :param gamma        : float; Gamma value
    :return             : tuple RGB values (0 ... 255)
    """
    cdef  rgb_color_int rgb_c
    rgb_c = wavelength_to_rgb(wavelength, gamma)
    return rgb_c.r, rgb_c.g, rgb_c.b

@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef tuple custom_map(int wavelength, int [:] color_array_, float gamma=1.0):
    """
    RETURN AN RGB COLOR VALUE MATCHING A CUSTOM WAVELENGTH

    Cython cpdef function, this function can be called directly and do not require a
    hook function.

    This function return a tuple (R,G,B) corresponding to the
    color wavelength define in color_array_
    (wavelength_to_rgb_custom is an External C
    routine with customize wavelength and allow the user to defined
    a customize palette according to an input value)

    example for a Fire palette
    arr = numpy.array(
        [0, 1,       # violet is not used
         0, 1,       # blue is not used
         0, 1,       # green is not used
         2, 619,     # yellow, return a yellow gradient for values [2...619]
         620, 650,   # orange return a orange gradient for values [620 ... 650]
         651, 660    # red return a red gradient for values [651 ... 660]
         ], numpy.int)


    :param wavelength   : integer; Wavelength
    :param gamma        : float; Gamma value
    :param color_array_ : numpy array containing the min and max of each color (red,
    orange, yellow, green, blue, violet)
    :return             : tuple RGB values (0 ... 255)
    """
    cdef  rgb_color_int rgb_c
    cdef int *p
    p = &color_array_[0]
    rgb_c = wavelength_to_rgb_custom(wavelength, p, gamma)
    return rgb_c.r, rgb_c.g, rgb_c.b




cdef int i = 0
HEATMAP = [heat_map(i, 1.0) for i in range(380, 750)]

cdef float f_map = (<float>750.0 - <float>380.0 -<float>1.0) / (<float>255.0 * <float>3.0)

cdef unsigned char[:, ::1] heatmap_array = numpy.zeros((750 - 380, 3), numpy.uint8)
cdef tuple t
i = 0
for t in HEATMAP:
    heatmap_array[i, 0] = t[0]
    heatmap_array[i, 1] = t[1]
    heatmap_array[i, 2] = t[2]
    i += 1



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef inline void heatmap_convert(object surface_, bint rgb_=True):
    """
    TRANSFORM AN IMAGE INTO A HEATMAP EQUIVALENT

    Cython cpdef function, this function can be called directly and do not require a
    hook function.

    :param surface_ : pygame.Surface
    :param rgb_     : boolean; True transformed the image into a RGB heatmap model of False (BGR)
    :return         : void
    """

    cdef:
        unsigned int w, h
    w, h = surface_.get_size()

    cdef:
        unsigned char [::1] rgb_array  = surface_.get_buffer()
        unsigned int s
        int i
        int size = rgb_array.size
        unsigned int index = 0
        unsigned char *r
        unsigned char *g
        unsigned char *b
        short int bitsize = surface_.get_bitsize()
        short int bytesize = surface_.get_bytesize()

    with nogil:
        # RGB
        if rgb_:
            for i in prange(0, size, bytesize, schedule='static', num_threads=THREADS):

                r = &rgb_array[i]
                g = &rgb_array[i + 1]
                b = &rgb_array[i + 2]

                s = r[0] + g[0] + b[0]
                index = <int>(s * f_map)

                r[0] = <unsigned char>heatmap_array[index, 0]
                g[0] = <unsigned char>heatmap_array[index, 1]
                b[0] = <unsigned char>heatmap_array[index, 2]
        # BGR
        else:
            for i in prange(0, size, bytesize, schedule='static', num_threads=THREADS):

                r = &rgb_array[i]
                g = &rgb_array[i + 1]
                b = &rgb_array[i + 2]

                s = r[0] + g[0] + b[0]
                index = <int>(s * f_map)

                r[0] = <unsigned char>heatmap_array[index, 2]
                g[0] = <unsigned char>heatmap_array[index, 1]
                b[0] = <unsigned char>heatmap_array[index, 0]



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef tuple blue_map(int wavelength, float gamma=1.0):
    """
    RETURN AN RGB COLOR VALUE MATCHING A SPECIFIC WAVELENGTH

    Cython cpdef function, this function can be called directly and do not require a
    hook function.

    :param wavelength   : integer; Wavelength
    :param gamma        : float; Gamma value
    :return             : tuple RGB values (0 ... 255)
    """
    cdef  rgb_color_int rgb_c
    rgb_c = wavelength_to_rgb(wavelength, gamma)
    return rgb_c.r, rgb_c.g, rgb_c.b

i = 0
BLUEMAP = [ blue_map(i, 1.0) for i in range(450, 495) ]

cdef float f_bluemap = (<float>495.0 - <float>450.0 - <float>1.0) / (<float>255.0 * <float>3.0)

cdef unsigned char[:, ::1] bluemap_array = numpy.zeros((495 - 450, 3), numpy.uint8)

i = 0
for t in BLUEMAP:
    bluemap_array[i, 0] = t[0]
    bluemap_array[i, 1] = t[1]
    bluemap_array[i, 2] = t[2]
    i += 1

@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef inline void bluescale(object surface_):
    """
    MAP AN IMAGE INTO A BLUE EQUIVALENT FORMAT
    THIS ALGORITHM IS USING THE WAVELENGTH FROM 450-495 NM TO
    REPRESENT THE IMAGE IN BLUE SHADES

    Cython cpdef function, this function can be called directly and do not require a
    hook function.

    :param surface_ : pygame.Surface to transform
    :return         : void
    """


    cdef:
        unsigned int w, h
    w, h = surface_.get_size()

    cdef:
        unsigned char [::1] rgb_array  = surface_.get_buffer()
        unsigned int s
        int i
        int size = rgb_array.size
        unsigned int index = 0
        unsigned char *r
        unsigned char *g
        unsigned char *b
        short int bitsize = surface_.get_bitsize()
        short int bytesize = surface_.get_bytesize()

    with nogil:
        for i in prange(0, size, bytesize, schedule='static', num_threads=THREADS):

            r = &rgb_array[i]
            g = &rgb_array[i + 1]
            b = &rgb_array[i + 2]

            s = r[0] + g[0] + b[0]
            index = <int>(s * f_bluemap)

            r[0] = <unsigned char>bluemap_array[index, 2]
            g[0] = <unsigned char>bluemap_array[index, 1]
            b[0] = <unsigned char>bluemap_array[index, 0]



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef tuple red_map(int wavelength, float gamma=1.0):
    """
    RETURN AN RGB COLOR VALUE MATCHING A SPECIFIC WAVELENGTH

    Cython cpdef function, this function can be called directly and do not require a
    hook function.

    :param wavelength   : integer; Wavelength
    :param gamma        : float; Gamma value
    :return             : tuple RGB values (0 ... 255)
    """
    cdef  rgb_color_int rgb_c
    rgb_c = wavelength_to_rgb(wavelength, gamma)
    return rgb_c.r, rgb_c.g, rgb_c.b

i = 0
REDMAP = [ red_map(i, 1.0) for i in range(620, 750) ]

cdef float f_redmap = (<float>750.0 - <float>620.0 - <float>1.0) / (<float>255.0 * <float>3.0)

cdef unsigned char[:, ::1] redmap_array = numpy.zeros((750 - 620, 3), numpy.uint8)

i = 0
for t in REDMAP:
    redmap_array[i, 0] = t[0]
    redmap_array[i, 1] = t[1]
    redmap_array[i, 2] = t[2]
    i += 1



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef inline void redscale(object surface_):
    """
    MAP AN IMAGE INTO A RED EQUIVALENT FORMAT
    THIS ALGORITHM IS USING THE WAVELENGTH FROM 620 TO 750 NM TO
    REPRESENT THE IMAGE IN RED SHADES

    Cython cpdef function, this function can be called directly and do not require a
    hook function.

    :param surface_ : pygame.Surface to transform
    :return         : void
    """


    cdef:
        unsigned int w, h
    w, h = surface_.get_size()

    cdef:
        unsigned char [::1] rgb_array  = surface_.get_buffer()
        unsigned int s
        int i
        int size = rgb_array.size
        unsigned int index = 0
        unsigned char *r
        unsigned char *g
        unsigned char *b
        short int bitsize = surface_.get_bitsize()
        short int bytesize = surface_.get_bytesize()

    with nogil:
        for i in prange(0, size, bytesize, schedule='static', num_threads=THREADS):

            r = &rgb_array[i]
            g = &rgb_array[i + 1]
            b = &rgb_array[i + 2]

            s = r[0] + g[0] + b[0]
            index = <int>(s * f_redmap)

            r[0] = <unsigned char>redmap_array[index, 2]
            g[0] = <unsigned char>redmap_array[index, 1]
            b[0] = <unsigned char>redmap_array[index, 0]



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline void shader_blood_inplace_c(
        unsigned char [:, :, :] rgb_array_, float [:, :] mask_, float perc_):

    """
    SHADER HURT EFFECT
    THE MASK DETERMINE THE CONTOUR USED FOR THE BLOOD EFFECT.

    The Array (rgb_array) must be a numpy array shape (w, h, 3)
    containing RGB pixels, please refer to pygame
    function pixels3d or array3d to convert an image into a
    3d array (library surfarray)

    :param rgb_array_   : numpy.ndarray shape (w, h, 3) of unsigned
    char representing the surface pixels
    :param mask_        : numpy.ndarray shape (w, h) of float values in range [0.0...1.0]
    :param perc_        : Percentage value in range [0.0 ... 1.0] with 1.0 being 100%

    :return: void
    """
    assert 0.0 <= perc_ <= 1.0, "perc_ variable must be in range[0.0 ... 1.0] got %s " % perc_

    cdef:
        int w, h, bytesize
    w, h, bytesize = (<object> rgb_array_).shape

    cdef:
        unsigned int s
        int i, j
        unsigned int index = 0
        unsigned char *r
        unsigned char *g
        unsigned char *b
        float theta


    with nogil:
        for j in prange(0, h, schedule='static', num_threads=THREADS):
            for i in range(0, w):

                r = &rgb_array_[i, j, 0]
                # g = &rgb_array_[i, j, 1]
                # b = &rgb_array_[i, j, 2]

                # s = r[0] + g[0] + b[0]
                # index = <int>(s * f_redmap)

                index = <int>(r[0] * f_redmap)
                theta = <float>(mask_[i, j] * perc_)

                # BEST METHOD (SLOW)
                # r[0] = <unsigned char> (r[0] * (1.0 - theta) +
                # <float>redmap_array[index, 0] * theta)
                # g[0] = <unsigned char> (g[0] * (1.0 - theta) +
                # <float>redmap_array[index, 1] * theta)
                # b[0] = <unsigned char> (b[0] * (1.0 - theta) +
                # <float>redmap_array[index, 2] * theta)

                # ALTERNATIVE WITH BEST PERFORMANCES
                r[0] = <unsigned char> (
                    min(r[0] + <float> redmap_array[index, 0] * theta, <unsigned char>255))



#
# def interpolant(t):
#     return t*t*t*(t*(t*6 - 15) + 10)
#
#
# def generate_perlin_noise_2d(
#         shape, res, tileable=(False, False), interpolant=interpolant
# ):
#     """Generate a 2D numpy array of perlin noise.
#
#     Args:
#         shape: The shape of the generated array (tuple of two ints).
#             This must be a multple of res.
#         res: The number of periods of noise to generate along each
#             axis (tuple of two ints). Note shape must be a multiple of
#             res.
#         tileable: If the noise should be tileable along each axis
#             (tuple of two bools). Defaults to (False, False).
#         interpolant: The interpolation function, defaults to
#             t*t*t*(t*(t*6 - 15) + 10).
#
#     Returns:
#         A numpy array of shape shape with the generated noise.
#
#     Raises:
#         ValueError: If shape is not a multiple of res.
#     """
#     delta = (res[0] / shape[0], res[1] / shape[1])
#     d = (shape[0] // res[0], shape[1] // res[1])
#     grid = numpy.mgrid[0:res[0]:delta[0], 0:res[1]:delta[1]]\
#              .transpose(1, 2, 0) % 1
#     # Gradients
#     angles = 2*numpy.pi*numpy.random.rand(res[0]+1, res[1]+1)
#     gradients = numpy.dstack((numpy.cos(angles), numpy.sin(angles)))
#     if tileable[0]:
#         gradients[-1,:] = gradients[0,:]
#     if tileable[1]:
#         gradients[:,-1] = gradients[:,0]
#     gradients = gradients.repeat(d[0], 0).repeat(d[1], 1)
#     g00 = gradients[    :-d[0],    :-d[1]]
#     g10 = gradients[d[0]:     ,    :-d[1]]
#     g01 = gradients[    :-d[0],d[1]:     ]
#     g11 = gradients[d[0]:     ,d[1]:     ]
#     # Ramps
#     n00 = numpy.sum(numpy.dstack((grid[:,:,0]  , grid[:,:,1]  )) * g00, 2)
#     n10 = numpy.sum(numpy.dstack((grid[:,:,0]-1, grid[:,:,1]  )) * g10, 2)
#     n01 = numpy.sum(numpy.dstack((grid[:,:,0]  , grid[:,:,1]-1)) * g01, 2)
#     n11 = numpy.sum(numpy.dstack((grid[:,:,0]-1, grid[:,:,1]-1)) * g11, 2)
#     # Interpolation
#     t = interpolant(grid)
#     n0 = n00*(1-t[:,:,0]) + t[:,:,0]*n10
#     n1 = n01*(1-t[:,:,0]) + t[:,:,0]*n11
#     return numpy.sqrt(2)*((1-t[:,:,1])*n0 + t[:,:,1]*n1)
#
#
#
# @cython.boundscheck(False)
# @cython.wraparound(False)
# @cython.nonecheck(False)
# @cython.cdivision(True)
# cdef inline cloud_effect_inplace_c(int frame):
#
#     cdef int w, h, i, j
#
#     # cdef float [:, :] noise_array = generate_perlin_noise_2d(
#     (800, 800), (16, 16), tileable=(True, True))
#
#     cdef float [:, :] noise_array = numpy.empty((200, 200), float32)
#
#     for j in range(0, 200):
#         for i in range(0, 200):
#             noise_array[i, j] = noise.pnoise2((i+ 50 * cos(frame * M_PI/180.0))/ 8.0,
#                           j / 8.0,
#                           octaves=8,
#                           persistence=0.25,
#                           lacunarity=2,
#                           repeatx=1024,
#                           repeaty=1024,
#                           base=0)
#
#
#     cdef:
#         unsigned char [:, :, :] dest_array = numpy.empty((800, 800, 3), numpy.uint8)
#         int ii, jj
#         int v
#
#     with nogil:
#         for i in prange(0, 800, schedule='static', num_threads=THREADS):
#             for j in range(0, 800):
#                 ii = <int>(200.0/800.0 * i)
#                 jj = <int>(200.0/800.0 * j)
#                 dest_array[i, j, 0] =<unsigned char>(noise_array[ii, jj] * 255)
#                 dest_array[i, j, 1] =<unsigned char>(noise_array[ii, jj] * 255)
#                 dest_array[i, j, 2] =<unsigned char>(noise_array[ii, jj] * 255)
#
#
#     return pygame.surfarray.make_surface(asarray(dest_array))
#


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef inline unsigned int rgb_to_int(int red, int green, int blue)nogil:
    """
    CONVERT RGB MODEL INTO A PYTHON INTEGER EQUIVALENT TO THE FUNCTION PYGAME MAP_RGB()

    Cython cpdef function, this function can be called directly and do not require a
    hook function.

    :param red   : Red color value,  must be in range [0..255]
    :param green : Green color value, must be in range [0..255]
    :param blue  : Blue color, must be in range [0.255]
    :return      : returns a positive python integer representing the RGB values(int32)
    """
    return 65536 * red + 256 * green + blue

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef inline rgb int_to_rgb(unsigned int n)nogil:
    """
    CONVERT A PYTHON INTEGER INTO A RGB COLOUR MODEL (UNSIGNED CHAR VALUES [0..255]).
    EQUIVALENT TO PYGAME UNMAP_RGB()

    Cython cpdef function, this function can be called directly and do not require a
    hook function.

    :param n : positive integer value to convert
    :return  : return a C structure rgb containing RGB values
    """
    cdef:
        rgb rgb_

    rgb_.r = <float>((n >> 16) & 255)
    rgb_.g = <float>((n >> 8) & 255)
    rgb_.b = <float>(n & 255)
    return rgb_


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline make_palette_c(int width, float fh, float fs, float fl):
    """
    CREATE A PALETTE (ARRAY) FROM HSL VALUES (HUE, SATURATION, LIGHTNESS)

    e.g:
        # below: palette of 256 colors & surface (width=256, height=50).
        # hue * 6, saturation = 255.0, lightness * 2.0
        palette, surf = make_palette(256, 50, 6, 255, 2)
        palette, surf = make_palette(256, 50, 4, 255, 2)

    :param width  : integer, Palette width
    :param fh     : float, hue factor
    :param fs     : float, saturation factor
    :param fl     : float, lightness factor
    :return       : Return a tuple ndarray type uint32 and pygame.Surface (width, height)
    """
    assert width > 0, "Argument width should be > 0, got %s " % width

    cdef:
        unsigned int [::1] palette = numpy.empty(width, uint32)
        int x, y
        float h, s, l
        rgb rgb_

    with nogil:
        for x in prange(width, schedule='static', num_threads=THREADS):
            h, s, l = <float>x * fh,  min(fs, <float>255.0), min(<float>x * fl, <float>255.0)
            rgb_ = struct_hsl_to_rgb(h * <float>ONE_360, s * <float>ONE_255, l * <float>ONE_255)
            # build the palette (1d buffer int values)
            palette[x] = rgb_to_int(<int>(rgb_.r * <float>255.0),
                                    <int>(rgb_.g * <float>255.0),
                                    <int>(rgb_.b * <float>255.0 * <float>0.5))

    return asarray(palette)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(False)
cdef fire_surface24_c(
        int width,
        int height,
        float factor,
        unsigned int [::1] palette,
        float [:, ::1] fire,
        int intensity = 0,
        int low       = 0,
        int high      = 0,
):
    """

    CREATE A FIRE EFFECT

    * Do not call this function directly

    :param width    : integer; max width of the effect
    :param height   : integer; max height of the effect
    :param factor   : float; factor to reduce the flame effect
    :param palette  : ndarray; Color palette 1d numpy array (colors buffer unsigned int values)
    :param fire     : ndarray; 2d array (x, y) (contiguous) containing float values
    :param intensity: integer; Control the flame intensity default 0 (low intensity), range [0...32]
    :param low      : integer; The x lowest position of the effect, x must be >=0 and < high
    :param high     : integer; The x highest position of the effect, x must be > low and <= high
    :return         : Return a numpy array containing the fire effect array shape
     (w, h, 3) of RGB pixels
    """

    cdef:
        # flame opacity palette
        unsigned char [:, :, ::1] out = zeros((width, height, 3), dtype=uint8)
        int x = 0, y = 0
        float d
        unsigned int ii=0
        unsigned c1 = 0, c2 = 0

    cdef int min_, max_, middle


    if low != 0 or high != 0:
        assert 0 <= low < high, "Argument low_ must be < high_"
        assert high <= width, "Argument high must be <= width"

        middle = low + ((high - low) >> 1)
        min_ = randRange(low, middle)
        max_ = randRange(middle + 1, high)
    else:
        middle = width >> 1
        min_ = randRange(0, middle)
        max_ = randRange(middle +1, width)


    with nogil:
        # POPULATE TO THE BASE OF THE FIRE (THIS WILL CONFIGURE THE FLAME ASPECT)
        for x in prange(min_, max_, schedule='static', num_threads=THREADS):
                fire[height-1, x] = randRange(intensity, 260) #260


        # DILUTE THE FLAME EFFECT (DECREASE THE MAXIMUM INT VALUE) WHEN THE FLAME TRAVEL UPWARD
        for y in prange(1, height-1, schedule='static', num_threads=THREADS):

            for x in range(0, width):

                    c1 = (y + 1) % height
                    c2 = x % width
                    d = (fire[c1, (x - 1 + width) % width]
                       + fire[c1, c2]
                       + fire[c1, (x + 1) % width]
                       + fire[(y + 2) % height, c2]) * factor

                    d = d - <float>(rand() * 0.0001) # 0.0001
                    
                    
                    # Cap the values
                    if d <0:
                        d = 0.0

                    # CAP THE VALUE TO 255
                    if d>255.0:
                        d = <float>255.0
                    fire[y, x] = d

                    ii = palette[<unsigned int>d % width]

                    out[x, y, 0] = (ii >> 16) & 255
                    out[x, y, 1] = (ii >> 8) & 255
                    out[x, y, 2] = ii & 255

    return asarray(out)



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(False)
cdef fire_surface24_c_border(
        int width,
        int height,

        float factor,
        unsigned int [::1] palette,
        float [:, ::1] fire,
        int intensity = 0,
        int low       = 0,
        int high      = 0,
):
    """

    CREATE A FIRE EFFECT (BORDER EFFECT)

    * Do not call this function directly

    :param width    : integer; max width of the effect
    :param height   : integer; max height of the effect
    :param factor   : float; factor to reduce the flame effect
    :param palette  : ndarray; Color palette 1d numpy array (colors buffer unsigned int values)
    :param fire     : ndarray; 2d array (x, y) (contiguous) containing float values
    :param intensity: integer; Control the flame intensity default 0 (low intensity), range [0...32]
    :param low      : integer; The x lowest position of the effect, x must be >=0 and < high
    :param high     : integer; The x highest position of the effect, x must be > low and <= high
    :return         : Return a numpy array containing the fire effect array
    shape (w, h, 3) of RGB pixels
    """

    cdef:
        # flame opacity palette
        unsigned char [:, :, ::1] out = zeros((width, height, 3), dtype=uint8)
        int x = 0, y = 0
        float d
        unsigned int ii=0
        unsigned c1 = 0, c2 = 0

    cdef int min_, max_, middle


    if low != 0 or high != 0:
        assert 0 <= low < high, "Argument low_ must be < high_"
        assert high <= width, "Argument high must be <= width"

        middle = low + ((high - low) >> 1)
        min_ = randRange(low, middle)
        max_ = randRange(middle + 1, high)
    else:
        middle = width >> 1
        min_ = randRange(0, middle)
        max_ = randRange(middle +1, width)


    with nogil:
        # POPULATE TO THE BASE OF THE FIRE (THIS WILL CONFIGURE THE FLAME ASPECT)
        # for x in prange(min_, max_, schedule='static', num_threads=THREADS
        #         fire[height - 1, x] = randRange(intensity, 260)

        # FIRE ARRAY IS [HEIGHT, WIDTH]
        for x in prange(min_, max_, schedule='static', num_threads=THREADS):
                fire[x % height, (height - 1) % width] = randRange(intensity, 260)


        # DILUTE THE FLAME EFFECT (DECREASE THE MAXIMUM INT VALUE) WHEN THE FLAME TRAVEL UPWARD
        for y in prange(1, height - 1, schedule='static', num_threads=THREADS):

            for x in range(0, width):

                    c1 = (y + 1) % height
                    c2 = x % width
                    d = (fire[c1, (x - 1 + width) % width]
                       + fire[c1, c2]
                       + fire[c1, (x + 1) % width]
                       + fire[(y + 2) % height, c2]) * factor

                    d = d - <float>(rand() * 0.0001)

                    # Cap the values
                    if d <0:
                        d = 0.0

                    if d>255.0:
                        d = <float>255.0

                    fire[x % height , y % width] = d

                    ii = palette[<unsigned int>d % width]

                    out[x, y, 0] = (ii >> 16) & 255
                    out[x, y, 1] = (ii >> 8) & 255
                    out[x, y, 2] = ii & 255

    return asarray(out)


# CUSTOM FIRE PALETTE
# arr = numpy.array([0, 1,       # violet
#                    0, 1,       # blue
#                    0, 1,       # green
#                    2, 619,     # yellow
#                    620, 650,   # orange
#                    651, 660],  # red
#                   numpy.int)
#
# HEATMAP = [custom_map(i - 20, arr, 1.0) for i in range(380, 800)]
# heatmap_array = numpy.zeros((800 - 380, 3), uint8)
# heatmap_rescale = numpy.zeros(255, numpy.uint)
# i = 0
# for t in HEATMAP:
#     heatmap_array[i, 0] = t[0]
#     heatmap_array[i, 1] = t[1]
#     heatmap_array[i, 2] = t[2]
#     i += 1
# for r in range(0, 255):
#     s = int(r * (800.0-380.0)/255)
#     heatmap_rescale[r] = rgb_to_int(heatmap_array[s][0], heatmap_array[s][1], heatmap_array[s][2])
# heatmap_rescale = numpy.ascontiguousarray(heatmap_rescale[::-1])
# -------- main loop ----
# if frame % 2 == 0:
# surface_ = fire_effect(
#     width, height, 3.95 + random.uniform(0.002, 0.008),
#     heatmap_rescale,
#     FIRE_ARRAY,
#     reduce_factor_=3,  fire_intensity_=8,
#     smooth_=True, bloom_=True, fast_bloom_=True,
#     bpf_threshold_=70, low_=0, high_=800, brightness_=True, brightness_intensity_=0.1,
#     adjust_palette_=False, hsv_=(10,  80, 1.8), transpose_=False, border_=False, surface_=None)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline shader_fire_effect_c(
        int width_,
        int height_,
        float factor_,
        unsigned int [::1] palette_,
        float [:, ::1] fire_,
        unsigned short int reduce_factor_ = 3,
        unsigned short int fire_intensity_= 32,
        bint smooth_                      = True,
        bint bloom_                       = True,
        bint fast_bloom_                  = False,
        unsigned char bpf_threshold_      = 0,
        unsigned int low_                 = 0,
        unsigned int high_                = 600,
        bint brightness_                  = True,
        float brightness_intensity_       = 0.15,
        object surface_                   = None,
        bint adjust_palette_              = False,
        tuple hsl_                        = (10, 80, 1.8),
        bint transpose_                   = False,
        bint border_                      = False,
        bint blur_                        = True
        ):
    """

    FIRE SHADER EFFECT 

    * FIRE TEXTURE SIZES 
    
    input width_  : integer,  
    input height_ : integer
    
    width_ and height_ values define the size of the texture e.g Surface(width x height)

    * FIRE ASPECT (CONTROL OVER THE WIDTH): 
    
    inputs low_ : integer  
    input high_ : integer 
    
    Optional arguments low_ & high_ (integer values) define the width 's limits of the fire effect. 
    low_ for the starting point and high_ for the ending of the effect.
    e.g low_ = 10 and high_ = 200. The fire effect will be contain within width = 10 and 200
    low_ & high_ values must be in range [0 ... width_]  
        
    * FIRE HEIGHT:
    
    input factor_ : float
    
    The fire maximum height can be adjust with the variable factor_ (float value)
    value > 3.95 will contain the effect within the display 
    value < 3.95 will enlarge the effect over the display height  
    Recommended value is 3.95 with reduce_factor_ = 3 otherwise adjust the value manually 
    to contain the fire effect within the display
        
    * SPEED CONSIDERATION
    
    input reduce_factor_ : integer
    
    The argument reduce_factor_ control the size of the texture to be processed 
    e.g : a value of 2, divide by 4 the pygame surface define by the values (width_ & height_)
    Smaller texture improve the overall performances but will slightly degrade the fire aspect, 
    especially if the blur and smooth option are not enabled.
    Recommended value for reduce_factor_ is 3 (fast process)   
    reduce_factor_ values must be an integer in range [ 0 ... 4] 
    The reduce_factor_ value will have a significant impact on the fire effect maximum height, 
    adjust the argument factor_ accordingly

    * FIRE INTENSITY AT THE SOURCE
    
    input fire_intensity_: integer
    
    Set the fire intensity with the variable fire_intensity_, 0 low flame,
    32 maximum flame effect
    Values must be an int in range [0 ... 32] 

    * SMOOTHING THE EFFECT
    
    input smooth_: True | False
    
    When smooth_ is True the algorithm will use the pygame function smoothscale (bi-linear 
    filtering) or False the final texture will be adjust with the scale function.
    Set this variable to False if you need the best performance for the effect or if you require
    a pixelated fire effect. Otherwise set the variable to True for a more realistic effect. 

    
    * BLOOM EFFECT 
    
    input bloom_         : True | False
    input fast_bloom_    : True | False
    input bpf_threshold_ : integer
       
    Fire effect produce a bright and smooth light effect to the background texture where the fire 
    intensity is at its maximum.
    Use the flag fast_bloom_ for a compromise between a realistic effect and the best performances
    The flag fast_bloom_ define a very fast bloom algo using only the smallest texture 
    to create a bloom effect (all the intermediate textures will be bypassed). See the bloom effect 
    project for more details.
    When fast_bloom is False, all the sub-surfaces will be blit to the final effect and will 
    produce a more realistic fire effect (this will slightly degrade the overall performances). 
    If the fire effect is too bright, you can always adjust the bright pass filter value
    bpf_threshold_(this will adjust the bloom intensity)
    bpf_threshold_ value must be in range [ 0 ... 255]   
    Below 128 the bloom effect will be more noticeable and above 128 only the brightest
    area will be enhanced.

    * LIGHT EFFECT INTENSITY

    input brightness_            : True | False
    input brightness_intensity_  : float

    When the flag is set to True, the algorithm will use an external function, 
    <shader_brightness24_exclude_inplace_c> to increase the brightness of the effect / texture
    A custom color can be passed to the function defining the pixels to be ignored during the 
    process (default is black color).
    the value must be in range [-1.0 ... 1.0]. Values below zero will decrease the brightness 
    of the flame effect and positive values will increase the brightness of the effect (causing
    bright white patches on the fire texture). 
    Values below -0.4 will cause the fire effect to be translucent and this effect can also be 
    used for simulating ascending heat convection effects on a background texture.
    
    
    * OPTIONAL SURFACE
      
    input surface_ : pygame.Surface
      
    This is an optional surface that can be passed to the shader to improve the performances 
    and to avoid a new surface to be generated every iterations. The surface size must match 
    exactly the reduce texture dimensions otherwise an exception will be raise. 
    see reduce_factor_ option to determine the fire texture size that will be processed.
    
    * COLOR PALETTE ADJUSTMENT  
    
    input adjust_palette_ : True | False
    input hsl_            : (10, 80, 1.8)

    Set this flag to True to modify the color palette of the fire texture. 
    This allow the HSL color model to be apply to the palette values
    You can redefine the palette when the flag is True and by customizing a tuple of 3 float 
    values, default is (10, 80, 1.8). 
    The first value control the palette hue value, the second is for the saturation and last, 
    the palette color lightness. 
    With the variable hsl_ you can rotate the palette colors and define a new flame
    aspect/color/intensity

    * FLAME ORIENTATION / DIRECTION & BORDER FLAME EFFECT
     
    input transpose_ = True | False,
    input border_    = True | False,
    
    transpose_ = True, this will transpose the final array 
    for e.g :  
    If the final fire texture is (w, h) after setting the transpose flag, the final 
    fire texture will become (h, w). As a result the fire effect will be transversal (starting 
    from the right of the display to the left side). 
    You can always transpose / flip the texture to get the right flame orientation  
    BORDER FLAME EFFECT 
    border_ = True to create a flame effect burning the edge of the display
    
    * FINAL TOUCH
    
    input blur_ : True | False
    
    This will will blur the fire effect for a more realistic appearance, remove all the jagged 
    edge when and pixelated effect
    
    
    :param width_           : integer; Size (width) of the surface or display in pixels
    :param height_          : integer; size (height) of the surface or display in pixels
    :param factor_          : float; Value controlling the fire height value
                              must be in range [3.95 ... 4.2].
                              The value 3.95 gives the highest flame effect
    :param palette_         : numpy.ndarray, buffer containing mapped RGB colors (uint values)
    :param fire_            : numpy.ndarray shape (w, h) containing float values (fire intensity).
                              For better performance it is advised to set the array to the size 
                              of the texture after applying the reduction_factor_.
                              For example if the reduction_factor_ is 2, the texture would have 
                              width >> 1 and height >> 1 and the fire_array should be set to 
                              numpy.empty((height >> 1, width >> 1), float32)
    :param reduce_factor_   : unsigned short int ; Can be either 0, 1, 2, 3, 4. 
                              2 and 3 provide the best performance and the best looking effect.
    :param fire_intensity_  : Integer; Control the original amount of energy at the
                              bottom of the fire, must be in range of [0 ... 32]. 
                              32 being the maximum value and the maximum fire intensity
    :param smooth_          : boolean; True smoothscale (bi-linear filtering) or
                              scale algorithm jagged edges (mush faster)
    :param bloom_           : boolean; True or False, True apply a bloom effect to the fire effect
    :param fast_bloom_      : boolean; Fastest bloom. This reduce the amount of calculation
    :param bpf_threshold_   : integer; control the bright pass filter threshold
                              value, must be in range [0 ... 255].
                              Maximum brightness amplification with threshold = 0, 
                              when bpf_threshold_ = 255, no change.
    :param low_             : integer; Starting position x for the fire effect
    :param high_            : integer; Ending position x for the fire effect
    :param brightness_      : boolean; True apply a bright filter shader to the array.
                              Increase overall brightness of the effect
    :param brightness_intensity_: float; must be in range [-1.0 ... 1.0] control
                              the brightness intensity
                              of the effect
    :param surface_         : pygame.Surface. Pass a surface to the shader for
                              better performance, otherwise algo is creating a new surface each 
                              calls.
    :param adjust_palette_  : boolean; True adjust the palette setting HSL
                              (hue, saturation, luminescence).
                              Be aware that if adjust_palette is True, the optional palette 
                              passed to the Shader will be disregarded
    :param hsl_             : tuple; float values of hue, saturation and luminescence.
                              Hue in range [0.0 ... 100],  saturation [0...100], 
                              luminescence [0.0 ... 2.0]
    :param transpose_       : boolean; Transpose the array (w, h) become (h, w).
                              The fire effect will start from the left and move to the right
    :param border_          : boolean; Flame effect affect the border of the texture
    :param blur_            : boolean; Blur the fire effect
    :return                 : Return a pygame surface that can be blit directly to the game display

    """


    cdef int w4, h4

    # TEXTURE DIVIDE BY POWER OF 2
    if reduce_factor_ in (0, 1, 2):
        w4, h4 = width_ >> reduce_factor_, height_ >> reduce_factor_

    # TEXTURE 150 x 150 * ratio
    elif reduce_factor_ == 3:
        # CUSTOM SIZE WIDTH 150 AND RATIO * HIGH
        w4 = 150
        h4 = <int>(150 * height_/width_)
        low_ = <int>(low_ * low_/width_)
        high_ = <int>(high_ * 150/width_)
        reduce_factor_ = 0

    # TEXTURE 100 x 100 * ratio
    elif reduce_factor_ == 4:
        w4 = 100
        h4 = <int> (100 * height_ / width_)
        low_ = <int> (low_ * low_ / width_)
        high_ = <int> (high_ * 100 / width_)
        reduce_factor_ = 0

    cdef int f_height, f_width
    f_height, f_width = (<object>fire_).shape[:2]

    assert f_width >= w4 or f_height >= h4,\
        "Fire array size mismatch the texture size.\n" \
        "Set fire_ array to numpy.empty((%s, %s), dtype=numpy.float32)" % (h4, w4)

    if surface_ is None:
        fire_surface_smallest = pygame.Surface((w4, h4)).convert()

    else:
        if PyObject_IsInstance(surface_, pygame.Surface):
            assert surface_.get_width() == w4 and surface_.get_height() == h4, \
            "Surface argument has incorrect dimension surface must be (w:%s, h:%s) got (%s, %s)\n" \
            "Set argument surface_ to None to avoid this error message"\
            % (w4, h4, surface_.get_width(), surface_.get_height())
            fire_surface_smallest = surface_
        else:
            raise ValueError("Argument surface_ must be a Surface type got %s " % type(surface_))

    if adjust_palette_:
        palette_= make_palette_c(w4, hsl_[0], hsl_[1], hsl_[2])

    if border_:
        # CREATE THE FIRE EFFECT ONTO A PYGAME SURFACE
        rgb_array_ = fire_surface24_c_border(
            w4, h4, <float>1.0 / factor_, palette_, fire_, fire_intensity_,
            low_ >> reduce_factor_, high_ >> reduce_factor_)
    else:
        rgb_array_ = fire_surface24_c(
            w4, h4, <float>1.0 / factor_, palette_, fire_, fire_intensity_,
                    low_ >> reduce_factor_, high_ >> reduce_factor_)

    # BRIGHTNESS SHADER
    if brightness_:
        # EXCLUDE BLACK COLORS (DEFAULT)
        assert -1.0 <= brightness_intensity_ <= 1.0, \
            "Argument brightness intensity must be in range [-1.0 ... 1.0]"
        shader_brightness24_exclude_inplace_c(rgb_array_=rgb_array_,
                                              shift_=brightness_intensity_, color_=(0, 0, 0))

    if blur_:
        shader_blur5x5_array24_inplace_c(rgb_array_)

    if transpose_:
        rgb_array_ = rgb_array_.transpose(1, 0, 2)
        fire_surface_smallest = rotate(fire_surface_smallest, 90)


    # CONVERT THE ARRAY INTO A PYGAME SURFACE
    array_to_surface(fire_surface_smallest, rgb_array_)


    # BLOOM SHADER EFFECT
    if bloom_:
        assert 0 <= bpf_threshold_ < 256, \
            "Argument bpf_threshold_ must be in range [0 ... 256] got %s " % bpf_threshold_
        # shader_bloom_effect_array24_c(fire_surface_smallest, bpf_threshold_, fast_=fast_bloom_)
        try:
            # fire_surface_smallest = shader_bloom_fast(
            #     fire_surface_smallest, bpf_threshold_, fast_=fast_bloom_, factor_=1)

            shader_bloom_fast1(
                fire_surface_smallest,
                threshold_ = bpf_threshold_,
                smooth_    = 0,
                saturation_= False
            )

        except ValueError:
            raise ValueError(
                "The surface is too small and cannot be bloomed with shader_bloom_fast1.\n"
                "Increase the size of the image")

    # RESCALE THE SURFACE TO THE FULL SIZE
    if smooth_:
        fire_effect = smoothscale(fire_surface_smallest, (width_, height_))
    else:
        fire_effect = scale(fire_surface_smallest, (width_, height_))

    return fire_effect



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(False)
cdef cloud_surface24_c(
        int width,
        int height,
        float factor,
        unsigned int [::1] palette,
        float [:, ::1] cloud_,
        int intensity = 0,
        int low       = 0,
        int high      = 0,
    ):

    """
    CLOUD PROCESSING FUNCTION

    * Do not call that function directly, this function is a sub function of cloud_effect

    :param width    : integer; Size (width) of the surface or display in pixels
    :param height   : integer; size (height) of the surface or display in pixels
    :param factor   : float; Value controlling the cloud size value must
                      be in range [3.95 ... 4.2].
                      value 3.95 will fill entirely the display with the cloud while value
                      above 3.95 will shrink the cloud effect
    :param palette  : numpy.ndarray, buffer containing mapped RGB colors (uint values)
    :param cloud_   : numpy.ndarray shape (w, h) containing float values (cloud intensity).
    :param intensity: integer; Determine the guaranteed amount of smoke the cloud
                      effect will generate at the base
                      of the effect (value must be in range [0 .. 260]). If you provide zero a 
                      random value between 0 ... 260 will be
                      assigned. If you provide 250, a random value between 250 and 260 will be set.
    :param low      : integer; low determine the X starting position on the display,
                      if you provide a value of
                      100 pixels, the effect will start at the position 100 from the display
                      (first 100 pixels will bot be affected by the cloud/smoke effect)
    :param high     : integer; high determine the X ending position on the display,
                      if you provide a value of 800 pixels, the effect will end at the 
                      position 800 from the display (last remaining pixels will not be affected
                      by the cloud/smoke effect)
    :return         : Return a numpy array shape (w, h, 3) containing the RGB pixels (smoke effect)
    """

    cdef:
        int new_height = height
        unsigned char [:, :, ::1] out = empty((width, new_height, 3), dtype=uint8)
        int x = 0, y = 0
        float d
        unsigned int ii=0
        unsigned c1 = 0, c2 = 0
        int p_length = (<object>palette).size

    cdef int min_, max_, middle

    if low != 0 or high != 0:
        assert 0 <= low < high, "Argument low_ must be < high_"
        assert high <= width,   "Argument high must be <= width"

        middle = low + ((high - low) >> 1)
        min_ = randRange(low, middle)
        max_ = randRange(middle + 1, high)
    else:
        middle = width >> 1
        min_ = randRange(0, middle)
        max_ = randRange(middle +1, width)


    with nogil:
        # POPULATE TO THE BASE OF THE FIRE (THIS WILL CONFIGURE THE FLAME ASPECT)
        for x in prange(min_, max_, schedule='static', num_threads=THREADS):
                cloud_[(new_height - 1) % height, x % width] = randRange(intensity, 260)


        # DILUTE THE FLAME EFFECT (DECREASE THE MAXIMUM INT VALUE) WHEN THE FLAME TRAVEL UPWARD
        for y in prange(0, new_height - 1, schedule='static', num_threads=THREADS):

            for x in range(0, width):

                    c1 = (y + 1) % height
                    c2 = x % width
                    d = (cloud_[c1, (x - 1 + width) % width]
                       + cloud_[c1, c2]
                       + cloud_[c1, (x + 1) % width]
                       + cloud_[(y + 2) % height, c2]) * factor

                    d = d - <float>(rand() * 0.0001)

                    # Cap the values
                    if d <0:
                        d = 0.0

                    # CAP THE VALUE TO 255
                    if d>512.0:
                        d = <float>512.0
                    cloud_[y % height, x % width] = d

                    ii = palette[<unsigned int>d % p_length]

                    out[x, y, 0] = (ii >> 16) & 255
                    out[x, y, 1] = (ii >> 8) & 255
                    out[x, y, 2] = ii & 255

    return asarray(out[:, 0:height, :])


# CLOUD ARRAY PALETTE
# cloud_rescale = numpy.zeros(256 * 2 * 3, numpy.uint)
#
# arr1 = create_horizontal_gradient_1d(255, (0, 0, 0), (255, 255, 255))
# arr2 = create_horizontal_gradient_1d(255, (255, 255, 255), (0, 0, 0))
# arr3 = numpy.concatenate((arr1, arr2), axis=None)
# i = 0
# for r in range(0, 1530, 3):
#     cloud_rescale[i] = rgb_to_int(arr3[r], arr3[r+1], arr3[r+2])
#     i += 1
# -------- main loop ----------
# surface_ = cloud_effect(
#             width, height, 3.956 + random.uniform(0.002, 0.008),
#             heatmap_rescale,
#             CLOUD_ARRAY,
#             reduce_factor_=2,  cloud_intensity_=75,
#             smooth_=True, bloom_=False, fast_bloom_=True,
#             bpf_threshold_=80, low_=0, high_=800, brightness_=False, brightness_intensity_=0.1,
#             transpose_=False, surface_=None, blur_=False)

# TODO MASK ? TO MOVE CLOUD ?

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(False)
cdef inline shader_cloud_effect_c(
        int width_,
        int height_,
        float factor_,
        unsigned int [::1] palette_,
        float [:, ::1] cloud_,

        # OPTIONAL
        unsigned short int reduce_factor_   = 2,
        unsigned short int cloud_intensity_ = 16,
        bint smooth_                        = True,
        bint bloom_                         = False,
        bint fast_bloom_                    = True,
        unsigned short int bpf_threshold_   = 128,
        unsigned int low_                   = 0,
        unsigned int high_                  = 0,
        bint brightness_                    = False,
        float brightness_intensity_         = 0.0,
        object surface_                     = None,
        bint transpose_                     = False,
        bint blur_                          = True
        ):
    """
    GENERATE CLOUD /SMOKE ON THE GAME DISPLAY 
    
    * CLOUD TEXTURE SIZES 
    
    input width_  : integer,  
    input height_ : integer
    
    width_ and height_ values define the size of the texture e.g Surface(width x height)

    * CLOUD ASPECT (CONTROL OVER THE WIDTH): 
    
    inputs low_ : integer  
    input high_ : integer 
    
    Optional arguments low_ & high_ (integer values) define the width 's limits of the cloud 
    effect. low_ for the starting point and high_ for the ending of the effect.
    e.g low_ = 10 and high_ = 200. The cloud effect will be contain within width = 10 and 200
    low_ & high_ values must be in range [0 ... width_]  
        
    * CLOUD HEIGHT:
    
    input factor_ : float
    
    The cloud maximum height can be adjust with the variable factor_ (float value)
    value > 3.95 will contain the effect within the display 
    value < 3.95 will enlarge the effect over the display height  
    Recommended value is 3.95 with reduce_factor_ = 3 otherwise adjust the value manually 
    to contain the cloud effect within the display
        
    * SPEED CONSIDERATION
    
    input reduce_factor_ : integer
    
    The argument reduce_factor_ control the size of the texture to be processed 
    e.g : a value of 2, divide by 4 the pygame surface define by the values (width_ & height_)
    Smaller texture improve the overall performances but will slightly degrade the cloud aspect, 
    especially if the blur and smooth option are not enabled.
    Recommended value for reduce_factor_ is 3 (fast process)   
    reduce_factor_ values must be an integer in range [ 0 ... 4] 
    The reduce_factor_ value will have a significant impact on the cloud effect maximum height, 
    adjust the argument factor_ accordingly

    * CLOUD INTENSITY AT THE SOURCE
    
    input cloud_intensity_: integer
    
    Set the cloud intensity with the variable cloud_intensity_, 0 low flame,
    32 maximum flame effect
    Values must be an int in range [0 ... 32] 

    * SMOOTHING THE EFFECT
    
    input smooth_: True | False
    
    When smooth_ is True the algorithm will use the pygame function smoothscale (bi-linear 
    filtering) or False the final texture will be adjust with the scale function.
    Set this variable to False if you need the best performance for the effect or if you require
    a pixelated cloud effect. Otherwise set the variable to True for a more realistic effect. 
   
    * BLOOM EFFECT 
    
    input bloom_         : True | False
    input fast_bloom_    : True | False
    input bpf_threshold_ : integer
       
    Bloom effect produce a bright and smooth light effect to the background texture where the cloud 
    intensity is at its maximum.
    Use the flag fast_bloom_ for a compromise between a realistic effect and the best performances
    The flag fast_bloom_ define a very fast bloom algo using only the smallest texture 
    to create a bloom effect (all the intermediate textures will be bypassed). See the bloom effect 
    project for more details.
    When fast_bloom is False, all the sub-surfaces will be blit to the final effect and will 
    produce a more realistic cloud effect (this will slightly degrade the overall performances). 
    If the cloud effect is too bright, you can always adjust the bright pass filter value
    bpf_threshold_(this will adjust the bloom intensity)
    bpf_threshold_ value must be in range [ 0 ... 255]   
    Below 128 the bloom effect will be more noticeable and above 128 only the brightest
    area will be enhanced.

    * LIGHT EFFECT INTENSITY

    input brightness_            : True | False
    input brightness_intensity_  : float

    When the flag is set to True, the algorithm will use an external function, 
    <shader_brightness24_exclude_inplace_c> to increase the brightness of the effect / texture
    A custom color can be passed to the function defining the pixels to be ignored during the 
    process (default is black color).
    the value must be in range [-1.0 ... 1.0]. Values below zero will decrease the brightness 
    of the cloud effect and positive values will increase the brightness of the effect (causing
    bright white patches on the cloud texture). 
    Values below -0.4 will cause the cloud effect to be translucent 
    
    
    * OPTIONAL SURFACE
      
    input surface_ : pygame.Surface
      
    This is an optional surface that can be passed to the shader to improve the performances 
    and to avoid a new surface to be generated every iterations. The surface size must match 
    exactly the reduce texture dimensions otherwise an exception will be raise. 
    see reduce_factor_ option to determine the cloud texture size that will be processed.
    

    * CLOUD ORIENTATION / DIRECTION 
     
    input transpose_ = True | False,
    
    transpose_ = True, this will transpose the final array 
    for e.g :  
    If the final cloud texture is (w, h) after setting the transpose flag, the final 
    cloud texture will become (h, w). As a result the cloud effect will be transversal (starting 
    from the right of the display to the left side). 
    You can always transpose / flip the texture to get the right cloud orientation  
    
    * FINAL TOUCH
    
    input blur_ : True | False
    
    This will will blur the cloud effect for a more realistic appearance, remove all the jagged 
    edge when and pixelated effect
    
    :param width_               : integer; Texture size (width) 
    :param height_              : integer; Texture size (height)
    :param factor_              : float; Floating value used to control the size of the cloud
                                  effect. Value must be in range [3.95 ... 4.2]. Value > 3.95 
                                  will contain the smoke/ cloud effect within the display. 
                                  Values < 3.95 will enlarge the smoke effect.                              
    :param palette_             : numpy.ndarray or cython memoryview containing the color for the 
                                  cloud effect (buffer containing mapped RGB colors (uint values))
    :param cloud_               : numpy.ndarray shape (w, h) containing float values 
                                  (cloud intensity). For better performance it is advised to set the
                                  array to the size of the texture after applying the 
                                  reduction_factor_. For example if the reduction_factor_ is 2, 
                                  the texture would have to be width >> 1 and height >> 1 and the 
                                  cloud_ array should be equivalent to numpy.empty((height >> 1, 
                                  width >> 1), float32)
    :param reduce_factor_       : integer; unsigned short int ; Can be either 0, 1, 2, 3, 4. 
                                  2 and 3 provide the best performance and the best looking effect.
    :param cloud_intensity_     : integer; Determine the amount of smoke the cloud
                                  effect will generate at the base of the effect (value must be in 
                                  range [0 .. 260]). If you provide zero a random value between 
                                  0 ... 260 will be assigned. If you provide 250, a random value 
                                  between 250 and 260 will be set for the amount of smoke. 
                                  The highest the value, the more dense the cloud effect will be
    :param smooth_              : boolean; True use a smoothscale (bi-linear filtering) or
                                  False -> scale algorithm jagged edges (mush faster)
    :param bloom_               : True | False, Add a bloom effect when the flag is set to True
                                  The bloom effect will smooth the cloud and create a dense smoke 
                                  areas where the cloud is the brightest.  
    :param fast_bloom_          : True | False; This set a fast algorithm for the bloom effect (the 
                                  bloom effect will use the smallest texture)
    :param bpf_threshold_       : integer; Bright pass filter value must be in range [ 0 ... 255]
                                  0 produce the maximum bloom effect
    :param low_                 : integer; must be in range [ 0 ... width_], left position of the 
                                  cloud effect 
    :param high_                : integer; must be in range [ 0 ... height_], right position of the
                                  cloud effect
    :param brightness_          : True | False; Increase the brightness of the cloud effect when 
                                  True
    :param brightness_intensity_: float; Set the brightness intensity of the cloud. The value must 
                                  be in range [-1.0 ... +1.0]. Changing the value overtime will 
                                  generate a realistic cloud effect. Negative value will generate 
                                  translucent patch of smoke on the background image
    :param surface_             : Pygame.Surface; Pass a surface to the shader for
                                  better performance, otherwise a new surface will be created each 
                                  calls.
    :param transpose_           : boolean; Transpose the array (w, h) become (h, w).
                                  The cloud effect will start from the left and move to the right
    :param blur_                : boolean; Blur the cloud effect
    :return                     : Return a pygame surface that can be blit directly to the game 
                                  display
    """


    cdef int w4, h4

    # TEXTURE DIVIDE BY POWER OF 2
    if reduce_factor_ in (0, 1, 2):
        w4, h4 = width_ >> reduce_factor_, height_ >> reduce_factor_

    # TEXTURE 150 x 150 * ratio
    elif reduce_factor_ == 3:
        # CUSTOM SIZE WIDTH 150 AND RATIO * HIGH
        w4 = 150
        h4 = <int>(150 * height_/width_)
        low_ = <int>(low_ * low_/width_)
        high_ = <int>(high_ * 150/width_)
        reduce_factor_ = 0

    # TEXTURE 100 x 100 * ratio
    elif reduce_factor_ == 4:
        w4 = 100
        h4 = <int> (100 * height_ / width_)
        low_ = <int> (low_ * low_ / width_)
        high_ = <int> (high_ * 100 / width_)
        reduce_factor_ = 0

    cdef int f_height, f_width
    f_height, f_width = (<object> cloud_).shape[:2]

    assert f_width >= w4 or f_height >= h4, \
        "Cloud array size mismatch the texture size.\n" \
        "Set cloud array to numpy.empty((%s, %s), dtype=numpy.float32)" % (h4, w4)

    if surface_ is None:
        cloud_surface_smallest = pygame.Surface((w4, h4)).convert()

    else:
        if PyObject_IsInstance(surface_, pygame.Surface):
            assert surface_.get_width() == w4 and surface_.get_height() == h4, \
            "Surface argument has incorrect dimension surface must be (w:%s, h:%s) got (%s, %s)\n" \
            "Set argument surface_ to None to avoid this error message"\
            % (w4, h4, surface_.get_width(), surface_.get_height())
            cloud_surface_smallest = surface_
        else:
            raise ValueError("Argument surface_ must be a Surface type got %s " % type(surface_))

    rgb_array_ = cloud_surface24_c(
        w4, h4, <float>1.0 / factor_, palette_, cloud_, cloud_intensity_,
                low_ >> reduce_factor_, high_ >> reduce_factor_)

    # BRIGHTNESS SHADER
    if brightness_:
        # EXCLUDE BLACK COLORS (DEFAULT)
        assert -1.0 <= brightness_intensity_ <= 1.0, \
            "Argument brightness intensity must be in range [-1.0 ... 1.0]"
        shader_brightness24_exclude_inplace_c(rgb_array_=rgb_array_,
                                              shift_=brightness_intensity_, color_=(0, 0, 0))

    if blur_:
        shader_blur5x5_array24_inplace_c(rgb_array_)

    if transpose_:
        rgb_array_ = rgb_array_.transpose(1, 0, 2)
        cloud_surface_smallest = make_surface(rgb_array_)
    else:
        # CONVERT THE ARRAY INTO A PYGAME SURFACE
        array_to_surface(cloud_surface_smallest, rgb_array_)


    # BLOOM SHADER EFFECT
    if bloom_:
        assert 0 <= bpf_threshold_ < 256, \
            "Argument bpf_threshold_ must be in range [0 ... 256] got %s " % bpf_threshold_
        shader_bloom_effect_array24_c(cloud_surface_smallest, bpf_threshold_, fast_=fast_bloom_)

    # RESCALE THE SURFACE TO THE FULL SIZE
    if smooth_:
        cloud_effect = smoothscale(cloud_surface_smallest, (width_, height_))
    else:
        cloud_effect = scale(cloud_surface_smallest, (width_, height_))

    return cloud_effect



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline mirroring_c(unsigned char[:, :, :] rgb_array_):

    """
    SHADER MIRRORING

    This method create a mirror image placed to the right side of the
     original image referenced by rgb_array_

    The Array (rgb_array) must be a numpy array shape (w, h, 3)
    containing RGB pixels, please refer to pygame
    function pixels3d or array3d to convert an image into a 3d
    array (library surfarray)

    :param rgb_array_: numpy ndarray shape (w, h, 3) containing RGB pixels
    :return          : returns a numpy ndarray shape (w, h, 3) with transformation
    """

    cdef:
        Py_ssize_t w, h
        int x2, x3
    w, h = rgb_array_.shape[:2]


    cdef:
        int x, y
        unsigned char [:, :, :] new_array = empty((w, h, 3), uint8)
        unsigned char *r
        unsigned char *g
        unsigned char *b

    with nogil:
        for x in prange(w, schedule='static', num_threads=THREADS):
            for y in range(h):

                r = &rgb_array_[x, y, 0]
                g = &rgb_array_[x, y, 1]
                b = &rgb_array_[x, y, 2]

                x2 = x >> 1
                new_array[x2, y, 0] = r[0]
                new_array[x2, y, 1] = g[0]
                new_array[x2, y, 2] = b[0]
                x3 = <int>w - x2 -1
                new_array[x3, y, 0] = r[0]
                new_array[x3, y, 1] = g[0]
                new_array[x3, y, 2] = b[0]

    return asarray(new_array)



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline mirroring_inplace_c(unsigned char[:, :, :] rgb_array_):

    """
    SHADER MIRRORING (INPLACE)

    This method create a mirror image placed to the right side of
     the original image referenced by rgb_array_

    The Array (rgb_array) must be a numpy array shape (w, h, 3)
    containing RGB pixels, please refer to pygame
    function pixels3d or array3d to convert an image into a 3d array (library surfarray)

    :param rgb_array_: numpy ndarray shape (w, h, 3) containing RGB pixels
    :return          : void
    """

    cdef:
        Py_ssize_t w, h
        int x2, x3
    w, h = rgb_array_.shape[:2]

    cdef:
        int x, y
        unsigned char [:, :, :] rgb_array_copy = numpy.array(rgb_array_, copy=True)
        unsigned char *r
        unsigned char *g
        unsigned char *b

    with nogil:
        for x in prange(w, schedule='static', num_threads=THREADS):
            for y in range(h):

                r = &rgb_array_copy[x, y, 0]
                g = &rgb_array_copy[x, y, 1]
                b = &rgb_array_copy[x, y, 2]

                x2 = x >> 1
                rgb_array_[x2, y, 0] = r[0]
                rgb_array_[x2, y, 1] = g[0]
                rgb_array_[x2, y, 2] = b[0]

                x3 = <int>w - x2 - 1

                rgb_array_[x3, y, 0] = r[0]
                rgb_array_[x3, y, 1] = g[0]
                rgb_array_[x3, y, 2] = b[0]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef tuple dampening(
        object surface_,
        int frame_,
        int display_width,
        int display_height_,
        float amplitude_=50.0,
        int duration_=30,
        float freq_=20.0):

    """
    DAMPENING EFFECT

    Cython cpdef function, this function can be called directly and do not require a
    hook function.

    Compatible with image 24-32 bit
    The length of the effect equal duration_ * freq_

    e.g :
    surf, xx, yy = dampening(BCK, frame, width, height,
    amplitude_=100, duration_=40, freq_=15)
    SCREEN.blit(surf, (xx, yy))

    :param surface_       : pygame.Surface (compatible 24 - 32 bit)
    :param frame_         : integer; Frame number (linear variable changing overtime)
    :param display_width  : integer; Size of your game display (width)
    :param display_height_: integer; size of your game display (height)
    :param amplitude_     : float; Amplitude of the dampening effect  (default is 50)
    :param duration_      : integer; Duration of the effect (default value is 30)
    :param freq_          : float; change the speed of the effect default value is 20.0.
    A small value will decrease
    the overall timing of the effect while a larger value will increase the duration of the effect.
    :return               : Tuple values containing the Surface and the position (x, y)
    with x & y are the top
     left corner of the
    image
    """

    assert freq_ > 0.0, "Argument freq_ must be > 0"
    assert duration_ > 0.0, "Argument duration_ must be > 0"

    cdef float t = damped_oscillation((frame_ / freq_) % duration_)
    cdef int width, height,
    cdef float tm = t * amplitude_

    width, height = surface_.get_size()

    if width + tm < 0:
        tm = 0
    if height + tm < 0:
        tm = 0
    cdef object surf = smoothscale(surface_, (<int>tm +<int> (width + <int>tm),
                                              <int>tm +<int> (height + <int>tm)))
    cdef int new_width, new_height
    new_width, new_height = surf.get_size()

    cdef int diff_x = display_width - new_width
    cdef int diff_y = display_height_ - new_height

    return surf, diff_x >> 1, diff_y >> 1


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(False)
cpdef inline float lateral_dampening(int frame_, float amplitude_=50.0,
                                     int duration_=30, float freq_=20.0):
    """
    DAMPENING EFFECT

    * This method return the lateral displacement (x)

    e.g:
    tm = lateral_dampening(frame, amplitude_=50.0, duration_=35, freq_=5.0)
    SCREEN.blit(BCK, (tm, 0), special_flags=0)

    Cython cpdef function, this function can be called directly and do not require a
    hook function.

    The length of the effect equal duration_ * freq_

    :param frame_    : integer; Your game frame number
    :param amplitude_: float; Represent the amplitude of the dampening effect.
                       An amplitude of 1.0 will have no effect.Default value is 50.0
    :param duration_ : float; This represent the duration of the effect, default value is 30
    :param freq_     : float; change the speed of the effect default value is 20.0.
                       A small value will decrease
                       the overall timing of the effect while a larger value will increase the 
                       duration of the effect.
    :return          : Return a float corresponding to the lateral displacement (x)
    """
    assert freq_ > 0, "Argument freq_ must be > 0"
    assert duration_ > 0, "Argument duration_ must be > 0"

    cdef float t = damped_oscillation((<float>frame_ / freq_) % duration_) * amplitude_
    return t

# --------------------------------------------------------------------------------------------------------
# KERNEL DEFINITION FOR SHARPEN ALGORITHM
cdef float [:, :] SHARPEN_KERNEL = numpy.array(([0, -1, 0],
                      [-1, 5, -1],
                      [0, -1, 0])).astype(dtype=float32)
cdef int HALF_KERNEL = len(SHARPEN_KERNEL) >> 1

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline void shader_sharpen_filter_inplace_c(unsigned char [:, :, :] rgb_array_):
        """
        SHARPEN IMAGE APPLYING THE BELOW 3 X 3 KERNEL OVER EVERY PIXELS.

        The Array (rgb_array) must be a numpy array shape (w, h, 3)
        containing RGB pixels, please refer to pygame
        function pixels3d or array3d to convert an image into a 3d array (library surfarray)

        pixels convoluted outside image edges will be set to adjacent edge value
        [0 , -1,  0]
        [-1,  5, -1]
        [0 , -1,  0]

        e.g
        sharpen(surface_)

        :param rgb_array_: numpy.ndarray shape (w, h, 3) containing all the RGB pixels
        :return          : void
        """

        # texture sizes
        cdef Py_ssize_t w, h
        w, h = rgb_array_.shape[:2]

        cdef:

            unsigned char [:, :, :] rgb_array_1 = numpy.empty((w, h, 3), uint8)
            int x, y, xx, yy
            short kernel_offset_y, kernel_offset_x
            float r, g, b, k
            unsigned char *rr
            unsigned char *gg
            unsigned char *bb
            int w_1 = <int>w - 1
            int h_1 = <int>h - 1

        with nogil:

            for y in prange(h, schedule='static', num_threads=THREADS):

                for x in range(w):

                    rr = &rgb_array_1[x, y, 0]
                    gg = &rgb_array_1[x, y, 1]
                    bb = &rgb_array_1[x, y, 2]

                    r, g, b = 0, 0, 0

                    for kernel_offset_y in range(-HALF_KERNEL, HALF_KERNEL + 1):

                        for kernel_offset_x in range(-HALF_KERNEL, HALF_KERNEL + 1):

                            xx = x + kernel_offset_x
                            yy = y + kernel_offset_y

                            if xx < 0:
                                xx = 0
                            elif xx > w_1:
                                xx = w_1

                            if yy < 0:
                                yy = 0
                            elif yy > h_1:
                                yy = h_1

                            k = SHARPEN_KERNEL[kernel_offset_y + HALF_KERNEL,
                                               kernel_offset_x + HALF_KERNEL]

                            r = r + rgb_array_[xx, yy, 0] * k
                            g = g + rgb_array_[xx, yy, 1] * k
                            b = b + rgb_array_[xx, yy, 2] * k

                    if r < 0:
                        r = 0
                    if g < 0:
                        g = 0
                    if b < 0:
                        b = 0
                    if r > 255:
                        r= 255
                    if g > 255:
                        g = 255
                    if b > 255:
                        b = 255

                    rr[0] = <unsigned char>r
                    gg[0] = <unsigned char>g
                    bb[0] = <unsigned char>b

            for y in prange(h, schedule='static', num_threads=THREADS):

                for x in range(w):
                    rgb_array_[x, y, 0] = rgb_array_1[x, y, 0]
                    rgb_array_[x, y, 1] = rgb_array_1[x, y, 1]
                    rgb_array_[x, y, 2] = rgb_array_1[x, y, 2]


# Added to version 1.0.1
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef cartoon_effect(
        object surface_,
        int sobel_threshold_,
        int median_kernel_,
        int color_, int flag_):
    """

    :param surface_: pygame.Surface compatible 24 - 32 bit 
    :param sobel_threshold_: integer sobel threshold
    :param median_kernel_  : integer median kernel  
    :param color_          : integer; color reduction value (max color)
    :param flag_           : integer; Blend flag e.g (BLEND_RGB_ADD, BLEND_RGB_SUB, 
                             BLEND_RGB_MULT, BLEND_RGB_MAX, BLEND_RGB_MIN  
    :return               : Return a pygame Surface with the cartoon effect 
    """

    # First branch from the surface_
    # surface_branch_1
    surface_branch_1 = surface_.copy()
    #array_ = canny_blur5x5_surface24_c(surface_branch_1)
    #shader_greyscale_luminosity24_inplace_c(array_)

    try:
        array_ = pixels3d(surface_branch_1)

    except Exception as e:
        raise ValueError("Cannot reference source pixels into a 3d array.\n %s " % e)

    shader_sobel24_inplace_c(array_, sobel_threshold_)

    # Second branch
    try:
        arr = pixels3d(surface_)

    except Exception as e:
        raise ValueError("Cannot reference source pixels into a 3d array.\n %s " % e)


    shader_median_filter24_inplace_c(arr, median_kernel_)
    #
    # # Color reduction
    # shader_color_reduction24_inplace_c(arr, color_)
    #

    #
    # Blend both branch
    del arr
    surface_.blit(make_surface(array_), (0, 0), special_flags=flag_)
    return surface_




# TODO ADD TO TESTING
# Added to version 1.0.1
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef object blending(object source_, object destination_, float percentage_):
    """
    BLEND A SOURCE TEXTURE TOWARD A DESTINATION TEXTURE (TRANSITION EFFECT)

    * Video system must be initialised 
    * source_ & destination_ Textures must be same sizes
    * Compatible with 24 - 32 bit surface
    * Output create a new surface
    * Image returned is converted for fast blit (convert())

    :param source_     : pygame.Surface (Source)
    :param destination_: pygame.Surface (Destination)
    :param percentage_ : float; Percentage value between [0.0 ... 100.0]
    :return: return    : Return a 24 bit pygame.Surface and blended with a percentage
                         of the destination texture.
    """

    cdef:
            unsigned char [:, :, :] source_array
            unsigned char [:, :, :] destination_array

    try:
        source_array      = pixels3d(source_)

    except Exception as e:
        raise ValueError("\nCannot reference source pixels into a 3d array.\n %s " % e)

    try:
        destination_array = pixels3d(destination_)

    except Exception as e:
        raise ValueError("\nCannot reference destination pixels into a 3d array.\n %s " % e)

    cdef:

        int c1, c2, c3
        int i=0, j=0
        Py_ssize_t w = source_array.shape[0]
        Py_ssize_t h = source_array.shape[1]
        unsigned char[:, :, ::1] final_array = empty((h, w, 3), dtype=uint8)
        float c4 = percentage_ / <float> 100.0
        unsigned char *r
        unsigned char *g
        unsigned char *b

    with nogil:
        for j in prange(h, schedule='static', num_threads=THREADS):
            for i in range(w):

                r = &source_array[i, j, 0]
                g = &source_array[i, j, 1]
                b = &source_array[i, j, 2]

                c1 = min(<int> (<float> destination_array[i, j, 0] * c4 +
                                r[0] * (<float> 1.0 - c4)), 255)
                c2 = min(<int> (<float> destination_array[i, j, 1] * c4 +
                                g[0] * (<float> 1.0 - c4)), 255)
                c3 = min(<int> (<float> destination_array[i, j, 2] * c4 +
                                b[0] * (<float> 1.0 - c4)), 255)
                if c1 < 0:
                    c1 = 0
                if c2 < 0:
                    c2 = 0
                if c3 < 0:
                    c3 = 0
                final_array[j, i, 0] = c1
                final_array[j, i, 1] = c2
                final_array[j, i, 2] = c3

    return pygame.image.frombuffer(final_array, (w, h), 'RGB').convert()


# new version 1.0.5
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef object alpha_blending(object source_, object destination_):
    """
    ALPHA BLENDING 

    * Video system must be initialised 
    * source_ & destination_ Textures must be same sizes
    * Compatible with 32 bit surfaces only
    * Output create a new surface
    * Image returned is converted for fast blit (convert())

    :param source_     : pygame.Surface (Source) 32-bit with alpha channel
    :param destination_: pygame.Surface (Destination) 32-bit with alpha channel
    :return: return    : Return a 24 bit pygame.Surface with alpha blending
    """

    cdef:
            float [:] source_array
            float [:] destination_array
            int w, h

    w, h = source_.get_size()

    # source & destination array are normalized
    try:
        source_array      = (numpy.frombuffer(source_.get_view('0').raw,
                                              dtype=numpy.uint8) / <float>255.0).astype(dtype=numpy.float32)

    except Exception as e:
        raise ValueError("\nCannot reference source pixels into a 3d array.\n %s " % e)

    try:
        destination_array = (numpy.frombuffer(destination_.get_view('0').raw,
                                              dtype=numpy.uint8) / <float>255.0).astype(dtype=numpy.float32)

    except Exception as e:
        raise ValueError("\nCannot reference destination pixels into a 3d array.\n %s " % e)

    cdef:

        float rr, gg, bb, alpha, tmp
        int i=0
        int l = w * h * 4
        unsigned char[:] final_array = empty(l, dtype=uint8)


        float *r
        float *g
        float *b
        float *a

    with nogil:
        # noinspection SpellCheckingInspection
        for i in prange(0, l, 4, schedule='static', num_threads=THREADS):

                r = &source_array[i+2]
                g = &source_array[i+1]
                b = &source_array[i]
                a = &source_array[i+3]

                # premult with alpha
                r[0] = r[0] * a[0]
                g[0] = g[0] * a[0]
                b[0] = b[0] * a[0]

                tmp = (<float>1.0 - a[0]) * destination_array[i+3]
                alpha = a[0] + tmp

                # premult with alpha
                rr = r[0] + destination_array[i+2] *  tmp
                gg = g[0] + destination_array[i+1] *  tmp
                bb = b[0] + destination_array[i  ] *  tmp

                # back to [0 ... 255]
                final_array[i]   = <unsigned char>min(rr * <float>255.0, 255)
                final_array[i+1] = <unsigned char>min(gg * <float>255.0, 255)
                final_array[i+2] = <unsigned char>min(bb * <float>255.0, 255)
                final_array[i+3] = <unsigned char>min(alpha * <float>255.0, 255)

    return pygame.image.frombuffer(numpy.asarray(
        final_array).reshape(w, h, 4), (w, h), 'RGBA').convert()

# new version 1.0.5
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef void alpha_blending_inplace(object source_, object destination_):
    """
    ALPHA BLENDING INPLACE

    * Video system must be initialised 
    * source_ & destination_ Textures must be same sizes
    * Compatible with 32 bit surfaces only
    * Output create a new surface
    * Image returned is converted for fast blit (convert())

    :param source_     : pygame.Surface (Source) 32-bit with alpha channel
    :param destination_: pygame.Surface (Destination) 32-bit with alpha channel
    :return: return    : Return a 24 bit pygame.Surface with alpha blending
    """

    cdef:
            float [:, :, :] source_array
            float [:, :, :] destination_array
            unsigned char [:, :, :] dest_rgb
            unsigned char [:, :] dest_alpha
            int w, h

    w, h = source_.get_size()

    # source & destination array are normalized
    try:
        source_rgb = (pixels3d(source_)/<float>255.0)
        source_alpha = (pixels_alpha(source_)/<float>255.0)
        source_array = numpy.dstack((source_rgb, source_alpha)).astype(dtype=numpy.float32)

    except Exception as e:
        raise ValueError("\nCannot reference source pixels into a 3d array.\n %s " % e)

    try:
        dest_rgb = pixels3d(destination_)
        dest_alpha = pixels_alpha(destination_)
        destination_array = (numpy.dstack(
            (numpy.asarray(dest_rgb), numpy.asarray(dest_alpha)))/255.0).astype(dtype=numpy.float32)


    except Exception as e:
        raise ValueError("\nCannot reference destination pixels into a 3d array.\n %s " % e)

    cdef:

        float rr, gg, bb, alpha, tmp
        int i=0, j=0

        float *r
        float *g
        float *b
        float *a

    with nogil:
        # noinspection SpellCheckingInspection
        for j in prange(h, schedule='static', num_threads=THREADS):
            for i in range(w):
                r = &source_array[i, j, 0]
                g = &source_array[i, j, 1]
                b = &source_array[i, j, 2]
                a = &source_array[i, j, 3]

                # premult with alpha
                r[0] = r[0] * a[0]
                g[0] = g[0] * a[0]
                b[0] = b[0] * a[0]

                tmp = (<float>1.0 - a[0]) * destination_array[i, j, 3]
                alpha = a[0] + tmp

                # premult with alpha
                rr = r[0] + destination_array[i, j, 0] *  tmp
                gg = g[0] + destination_array[i, j, 1] *  tmp
                bb = b[0] + destination_array[i, j, 2] *  tmp

                # back to [0 ... 255]
                dest_rgb[i, j, 0] = <unsigned char>min(rr * <float>255.0, 255)
                dest_rgb[i, j, 1] = <unsigned char>min(gg * <float>255.0, 255)
                dest_rgb[i, j, 2] = <unsigned char>min(bb * <float>255.0, 255)
                dest_alpha[i, j] = <unsigned char>min(alpha * <float>255.0, 255)




@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline void dirt_lens_c(
        object source_,
        object lens_model_,
        int flag_,
        float light_=0.0):

    if light_!=0.0:
        brightness(lens_model_, light_)
    source_.blit(lens_model_, (0, 0), special_flags=flag_)


# TODO DITHERING WITH KERNEL INSTEAD
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef object dithering_c(
    float [:, :, :] rgb_array_, 
    int factor_
    ):

    """
    Dithering is used in computer graphics to create the illusion of "color depth" in images with
    a limited color palette - a technique also known as color quantization. In a dithered image,
    colors that are not available in the palette are approximated by a diffusion of colored pixels
    from within the available palette. The human eye perceives the diffusion as a mixture of 
    the colors within it (see color vision). Dithered images, particularly those with relatively
    few colors, can often be distinguished by a characteristic graininess or speckled appearance.
    
    factor_ represent the color number per RGB channel, values must be (2, 3, 4 .. etc)
    and 2 means two colors per channels (2x2x2 = total 8 colors)
    
    
    :param rgb_array_: pygame.Surface compatible 24-32 bit
    :param factor_: integer; color per channel; must be > 1. 
    A value of 2 means a total of 8 colors
    :return: pygame surface 24-32 bit    
    """

    cdef Py_ssize_t w, h
    w = <object> rgb_array_.shape[ 0 ] - 1
    h = <object> rgb_array_.shape[ 1 ] - 1

    cdef:
        int x=0, y=0
        float new_red, new_green, new_blue
        float quantization_error_red, quantization_error_green, quantization_error_blue
        float oldr, oldg, oldb

    rgb_array_ = rgb_array_.astype(dtype=numpy.float32)
    
    with nogil:

        for y in prange(1, h, schedule='static', num_threads=THREADS, chunksize=256):

            for x in range(1, w):

                oldr = rgb_array_[x, y, 0]
                oldg = rgb_array_[x, y, 1]
                oldb = rgb_array_[x, y, 2]

                new_red   = <float>round_c(oldr *
                                           (factor_ - <float>1.0)/<float>255.0) / ((factor_ - <float>1.0)/<float>255.0)
                new_green = <float>round_c(oldg *
                                           (factor_ - <float>1.0)/<float>255.0) / ((factor_ - <float>1.0)/<float>255.0)
                new_blue  = <float>round_c(oldb *
                                           (factor_ - <float>1.0)/<float>255.0) / ((factor_ - <float>1.0)/<float>255.0)

                rgb_array_[x, y, 0] = new_red
                rgb_array_[x, y, 1] = new_green
                rgb_array_[x, y, 2] = new_blue

                quantization_error_red   = <float>(oldr - new_red)
                quantization_error_green = <float>(oldg - new_green)
                quantization_error_blue  = <float>(oldb - new_blue)

                rgb_array_[x + 1, y, 0] = \
                    rgb_array_[x + 1, y, 0] + quantization_error_red * C1
                rgb_array_[x + 1, y, 1] = \
                    rgb_array_[x + 1, y, 1] + quantization_error_green * C1
                rgb_array_[x + 1, y, 2] = \
                    rgb_array_[x + 1, y, 2] + quantization_error_blue * C1

                rgb_array_[x - 1, y + 1, 0] = \
                    rgb_array_[x - 1, y + 1, 0] + quantization_error_red * C2
                rgb_array_[x - 1, y + 1, 1] = \
                    rgb_array_[x - 1, y + 1, 1] + quantization_error_green * C2
                rgb_array_[x - 1, y + 1, 2] = \
                    rgb_array_[x - 1, y + 1, 2] + quantization_error_blue * C2

                rgb_array_[x, y + 1, 0] = \
                    rgb_array_[x, y + 1, 0] + quantization_error_red * C3
                rgb_array_[x, y + 1, 1] = \
                    rgb_array_[x, y + 1, 1] + quantization_error_green * C3
                rgb_array_[x, y + 1, 2] = \
                    rgb_array_[x, y + 1, 2] + quantization_error_blue * C3

                rgb_array_[x + 1, y + 1, 0] = \
                    rgb_array_[x + 1, y + 1, 0] + quantization_error_red * C4
                rgb_array_[x + 1, y + 1, 1] = \
                    rgb_array_[x + 1, y + 1, 1] + quantization_error_green * C4
                rgb_array_[x + 1, y + 1, 2] = \
                    rgb_array_[x + 1, y + 1, 2] + quantization_error_blue * C4

    return make_surface(numpy.multiply(asarray(rgb_array_), 255.0).astype(dtype=uint8)).convert()



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline void dithering_int_c(
    unsigned char[:, :, :] rgb_array_, 
    int factor_ = 2
    ):


    cdef Py_ssize_t w, h
    w = <object> rgb_array_.shape[ 0 ] - 1
    h = <object> rgb_array_.shape[ 1 ] - 1

    cdef:
        int x=0, y=0
        unsigned char new_red, new_green, new_blue
        int quantization_error_red, quantization_error_green, quantization_error_blue
        unsigned char oldr, oldg, oldb
        float c5 = <float>255.0 / <float>(factor_ -1)
        float c6 = <float>(factor_ -1) * ONE_255


    with nogil:

        for y in prange(1, h, schedule='static', num_threads=THREADS, chunksize=2**8):

            for x in range(1, w):

                oldr = rgb_array_[x, y, 0]
                oldg = rgb_array_[x, y, 1]
                oldb = rgb_array_[x, y, 2]


                new_red = <unsigned int> (<int>nearbyint(c6 * oldr) * <int>c5)
                new_green = <unsigned int> (<int>nearbyint(c6 * oldg) * <int>c5)
                new_blue = <unsigned int> (<int>nearbyint(c6 * oldb) * <int>c5)

                rgb_array_[x, y, 0] = new_red
                rgb_array_[x, y, 1] = new_green
                rgb_array_[x, y, 2] = new_blue

                quantization_error_red   = int(oldr - new_red)
                quantization_error_green = int(oldg - new_green)
                quantization_error_blue  = int(oldb - new_blue)

                rgb_array_[x + 1, y, 0] = \
                    <unsigned char>(rgb_array_[x + 1, y, 0] + quantization_error_red * C1)
                rgb_array_[x + 1, y, 1] = \
                    <unsigned char>(rgb_array_[x + 1, y, 1] + quantization_error_green * C1)
                rgb_array_[x + 1, y, 2] = \
                    <unsigned char>(rgb_array_[x + 1, y, 2] + quantization_error_blue * C1)

                rgb_array_[x - 1, y + 1, 0] = \
                    <unsigned char>(rgb_array_[x - 1, y + 1, 0] + quantization_error_red * C2)
                rgb_array_[x - 1, y + 1, 1] = \
                    <unsigned char>(rgb_array_[x - 1, y + 1, 1] + quantization_error_green * C2)
                rgb_array_[x - 1, y + 1, 2] = \
                    <unsigned char>(rgb_array_[x - 1, y + 1, 2] + quantization_error_blue * C2)

                rgb_array_[x, y + 1, 0] = \
                    <unsigned char>(rgb_array_[x, y + 1, 0] + quantization_error_red * C3)
                rgb_array_[x, y + 1, 1] = \
                    <unsigned char>(rgb_array_[x, y + 1, 1] + quantization_error_green * C3)
                rgb_array_[x, y + 1, 2] = \
                    <unsigned char>(rgb_array_[x, y + 1, 2] + quantization_error_blue * C3)

                rgb_array_[x + 1, y + 1, 0] = \
                    <unsigned char>(rgb_array_[x + 1, y + 1, 0] + quantization_error_red * C4)
                rgb_array_[x + 1, y + 1, 1] = \
                    <unsigned char>(rgb_array_[x + 1, y + 1, 1] + quantization_error_green * C4)
                rgb_array_[x + 1, y + 1, 2] = \
                    <unsigned char>(rgb_array_[x + 1, y + 1, 2] + quantization_error_blue * C4)



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline void convert_27colors_c(
        unsigned char [:, :, :] rgb_array):

    """
    THIS ALGORITHM CONVERT AN IMAGE USING 27 COLORS ONLY
    
    :param rgb_array: numpy.ndarray; containing the pixels RGB. Array shape (w, h, 3)  
    :return: void 
    """
    cdef Py_ssize_t w, h
    w, h = rgb_array.shape[:2]

    cdef:
        int x=0
        int y=0
        unsigned char *r
        unsigned char *g
        unsigned char *b
        float f = <float> 255.0 / <float> 2
        float c1 = <float>2 / <float>255.0
        int index = 0
        float rr, gg, bb

    with nogil:
        for y in prange(h, schedule='static', num_threads=THREADS):
            for x in range(0, w):

                r = &rgb_array[x, y, 0]
                g = &rgb_array[x, y, 1]
                b = &rgb_array[x, y, 2]

                rr = <float>round_c(c1 * <float> r[ 0 ] ) * f
                gg = <float>round_c(c1 * <float> g[ 0 ] ) * f
                bb = <float>round_c(c1 * <float> b[ 0 ] ) * f

                r[ 0 ] = <unsigned char>rr
                g[ 0 ] = <unsigned char>gg
                b[ 0 ] = <unsigned char>bb


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef object spectrum_c(int width, int height, float gamma=1.0):

    """
    CREATE A PYGAME SURFACE DISPLAYING THE LIGHT SPECTRUM 380-750 nm
    
    Color   Wavelength(nm) Frequency(THz)
    Red     620-750        484-400
    Orange  590-620        508-484
    Yellow  570-590        526-508
    Green   495-570        606-526
    Blue    450-495        668-606
    Violet  380-450        789-668
    
    :param width: integer; width of the image
    :param height: integer; height of the image
    :param gamma: float; gamma value 
    :return: Return a pygame surface 24-bit (width, height) converted for fast 
    blit 
    
    """

    cdef:
        int i, j, k
        rgb_color_int rgb_c
        unsigned char [:, :, :] spectrum_array =\
            numpy.empty((370, 1, 3), numpy.uint8)
        object surface

    with nogil:
        for i in prange(380, 750, schedule='static', num_threads=THREADS):
            rgb_c = wavelength_to_rgb(i, gamma)
            k = i - 380
            spectrum_array[ k, 0, 0 ] = rgb_c.r
            spectrum_array[ k, 0, 1 ] = rgb_c.g
            spectrum_array[ k, 0, 2 ] = rgb_c.b
            spectrum_array[ k, 1, 0 ] = rgb_c.r
            spectrum_array[ k, 1, 1 ] = rgb_c.g
            spectrum_array[ k, 1, 2 ] = rgb_c.b

    surface = make_surface(asarray(spectrum_array))
    surface = scale(surface, (width, height)).convert()
    return surface




@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline void palette_change_c(
        unsigned char [:, :, :] rgb_array,
        float [:, :] palette_):


    cdef:
        int i, j
        Py_ssize_t w = <object>rgb_array.shape[0]
        Py_ssize_t h = <object>rgb_array.shape[1]
        rgb rgb_
        rgb rgb_c
        unsigned char *r
        unsigned char *g
        unsigned char *b
        Py_ssize_t p_length = <object>palette_.shape[0]

    with nogil:
        for i in prange(w, schedule='static', num_threads=THREADS, chunksize=w * h):
            for j in range(h):
                r = &rgb_array[i, j, 0]
                g = &rgb_array[i, j, 1]
                b = &rgb_array[i, j, 2]

                rgb_c.r = r[0]
                rgb_c.g = g[0]
                rgb_c.b = b[0]

                rgb_ = close_color(rgb_c, palette_, p_length)

                r[0] = <unsigned char>rgb_.r
                g[0] = <unsigned char>rgb_.g
                b[0] = <unsigned char>rgb_.b




@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef float distance_ (float x, float y)nogil:
  return <float>sqrt(x*x + y*y)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef float gaussian_ (float v, float sigma2)nogil:
  return (<float>1.0 / (<float>M_PI * sigma2)) * <float>exp(-(v * v ) / sigma2)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef bilateral_filter24_c(
        unsigned char [:, :, :] rgb_array_,
        float sigma_s_,
        float sigma_i_,
        unsigned int kernel_size = 3
):
    """
    A bilateral filter is a non-linear, edge-preserving, and noise-reducing
    smoothing filter for images. It replaces the intensity of each pixel with a
    weighted average of intensity values from nearby pixels. This weight can be
    based on a Gaussian distribution.

    Here, the normalization factor and the range weight are new terms added to 
    the previous equation. sigma_s  denotes the spatial extent of the kernel, i.e. 
    the size of the neighborhood, and sigma_r  denotes the minimum amplitude of an edge.
    It ensures that only those pixels with intensity values similar to that of the
    central pixel are considered for blurring, while sharp intensity changes are maintained.
    The smaller the value of sigma_r  , the sharper the edge. As sigma_r  tends to infinity,  
    the equation tends to a Gaussian blur.
    
    :param kernel_size: integer; kernel size; default is 3 
    :param rgb_array_: Surface, 24-32 bit format (alpha channel will be ignored)
    
    :param sigma_s_: float sigma_s : Spatial extent of the kernel, size of the 
    considered neighborhood
    
    :param sigma_i_: float sigma_i (also call sigma_r) range kernel, minimum amplitude of an edge.
    
    :return: return a filtered Surface
    """

    cdef Py_ssize_t w, h
    w = <object>rgb_array_.shape[ 0 ]
    h = <object> rgb_array_.shape[ 1 ]

    cdef:
        unsigned char [:, :, :] bilateral = empty((h, w, 3), dtype=uint8)
        int x, y, xx, yy
        int k = kernel_size
        int kx, ky
        float gs, wr, wg, wb, ir, ig, ib , wpr, wpg, wpb
        unsigned char *r
        unsigned char *g
        unsigned char *b
        float sigma_i2 = 2 * sigma_i_ * sigma_i_
        float sigma_s2 = 2 * sigma_s_ * sigma_s_

    with nogil:

        for x in prange(0, w, schedule='static', num_threads=THREADS):
            for y in range(0, h):

                ir, ig, ib = 0, 0, 0
                wpr, wpg, wpb = 0, 0, 0

                for ky in range(-k, k + 1):

                    for kx in range(-k, k + 1):

                        xx = x + kx
                        yy = y + ky

                        if xx < 0:
                            xx = 0
                        elif xx > w:
                            xx = w

                        if yy < 0:
                            yy = 0
                        elif yy > h:
                            yy = h

                        gs = gaussian_(distance_(kx, ky), sigma_s2)

                        r = &rgb_array_[xx, yy, 0]
                        g = &rgb_array_[xx, yy, 1]
                        b = &rgb_array_[xx, yy, 2]

                        wr = gaussian_(r[0] - rgb_array_[x, y, 0], sigma_i2) * gs
                        wg = gaussian_(g[0] - rgb_array_[x, y, 1], sigma_i2) * gs
                        wb = gaussian_(b[0] - rgb_array_[x, y, 2], sigma_i2) * gs

                        ir = ir + r[0] * wr
                        ig = ig + g[0] * wg
                        ib = ib + b[0] * wb

                        wpr = wpr + wr
                        wpg = wpg + wg
                        wpb = wpb + wb

                ir = ir / wpr
                ig = ig / wpg
                ib = ib / wpb

                bilateral[y, x, 0] = <int>ir
                bilateral[y, x, 1] = <int>ig
                bilateral[y, x, 2] = <int>ib


    return frombuffer(bilateral, (w, h), 'RGB')




EMBOSS_KERNEL = \
    numpy.array((
        [-1, -1, -1, -1, 0],
        [-1, -1, -1, 0,  1],
        [-1, -1,  0, 1,  1],
        [-1,  0,  1, 1,  1],
        [ 0,  1,  1, 1,  1])).astype(dtype=numpy.float32, order='C')

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef object emboss5x5_c(unsigned char [:, :, :] rgb_array_):

    k_weight = numpy.sum(EMBOSS_KERNEL)
    k_length = len(EMBOSS_KERNEL)
    half_kernel = len(EMBOSS_KERNEL) >> 1

    # texture sizes
    cdef Py_ssize_t w, h
    w = <object>rgb_array_.shape[0]
    h = <object>rgb_array_.shape[1]

    cdef:
        float [:, :] kernel = EMBOSS_KERNEL
        float kernel_weight = k_weight
        short kernel_half = half_kernel
        unsigned char [:, :, ::1] emboss = empty((h, w, 3), order='C', dtype=uint8)
        int kernel_length = k_length
        int x, y, xx, yy
        unsigned short red, green, blue,
        short kernel_offset_y, kernel_offset_x
        float r, g, b, k

    with nogil:

        for x in prange(0, w, schedule='static', num_threads=THREADS):

            for y in range(0, h):

                r, g, b = 0, 0, 0

                for kernel_offset_y in range(-kernel_half, kernel_half + 1):

                    for kernel_offset_x in range(-kernel_half, kernel_half + 1):

                        xx = x + kernel_offset_x
                        yy = y + kernel_offset_y

                        if xx < 0:
                            xx = 0
                        elif xx > w:
                            xx = w

                        if yy < 0:
                            yy = 0
                        elif yy > h:
                            yy = h


                        red, green, blue = \
                            rgb_array_[xx, yy, 0], \
                            rgb_array_[xx, yy, 1],\
                            rgb_array_[xx, yy, 2]

                        k = kernel[kernel_offset_y + kernel_half, kernel_offset_x + kernel_half]
                        r = r + red * k
                        g = g + green * k
                        b = b + blue * k

                if r < 0:
                    r = 0
                if g < 0:
                    g = 0
                if b < 0:
                    b = 0
                if r > 255:
                    r= 255
                if g > 255:
                    g = 255
                if b > 255:
                    b = 255

                emboss[y, x, 0], emboss[y, x, 1], \
                emboss[y, x, 2] = <unsigned char>r, <unsigned char>g, <unsigned char>b

    return frombuffer(asarray(emboss), (w, h), 'RGB')





@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef object bilinear_c(
        unsigned char [:, :, :] rgb_array_,
        int new_width, int new_height,
        fx=None, fy=None):

    cdef Py_ssize_t w, h
    w, h = rgb_array_.shape[:2]

    cdef:
        float rowScale, colScale
        float original_x, original_y
        unsigned int bl, br, tl, tr,\
            modXiPlusOneLim, modYiPlusOneLim
        int modXi, modYi, x, y, chan
        float modXf, modYf, b, t, xf

    rowScale = <float>w / <float>new_width
    colScale = <float>h / <float>new_height

    if fx is not None:
        new_width = <int> (w * fx)
    if fy is not None:
       new_height = <int>(h * fy)

    cdef unsigned char [: , :, :] new_rgb = \
        numpy.empty((new_width, new_height, 3), dtype=numpy.uint8)

    with nogil:
        for x in prange(0, new_width, schedule='static', num_threads=THREADS):
            for y in prange(new_height):
                original_x = <float>x * rowScale
                original_y = <float>y * colScale

                modXi = <int>original_x
                modYi = <int>original_y
                modXf = original_x - modXi
                modYf = original_y - modYi
                modXiPlusOneLim = min(modXi + 1, h - 1)
                modYiPlusOneLim = min(modYi + 1, w - 1)
                xf = <float>1.0 - modXf

                # for chan in range(3):
                #     bl = rgb_array_[modYi, modXi, chan]
                #     br = rgb_array_[modYi, modXiPlusOneLim, chan]
                #     tl = rgb_array_[modYiPlusOneLim, modXi, chan]
                #     tr = rgb_array_[modYiPlusOneLim, modXiPlusOneLim, chan]
                #
                #     # Calculate interpolation
                #     b = modXf * br + xf * bl
                #     t = modXf * tr + xf * tl
                #     # pixel_value = modYf * t + (<float>1.0 - modYf) * b
                #     new_rgb[x, y, chan] = <int>(modYf * t + (<float>1.0 - modYf) * b + <float>0.5)

                bl = rgb_array_[modYi, modXi, 0]
                br = rgb_array_[modYi, modXiPlusOneLim, 0]
                tl = rgb_array_[modYiPlusOneLim, modXi, 0]
                tr = rgb_array_[modYiPlusOneLim, modXiPlusOneLim, 0]

                # Calculate interpolation
                b = modXf * br + xf * bl
                t = modXf * tr + xf * tl

                new_rgb[x, y, 0] = <int> (modYf * t + (<float> 1.0 - modYf) * b + <float> 0.5)

                bl = rgb_array_[modYi, modXi, 1]
                br = rgb_array_[modYi, modXiPlusOneLim, 1]
                tl = rgb_array_[modYiPlusOneLim, modXi, 1]
                tr = rgb_array_[modYiPlusOneLim, modXiPlusOneLim, 1]

                # Calculate interpolation
                b = modXf * br + xf * bl
                t = modXf * tr + xf * tl

                new_rgb[x, y, 1] = <int> (modYf * t + (<float> 1.0 - modYf) * b + <float> 0.5)

                bl = rgb_array_[modYi, modXi, 2]
                br = rgb_array_[modYi, modXiPlusOneLim, 2]
                tl = rgb_array_[modYiPlusOneLim, modXi, 2]
                tr = rgb_array_[modYiPlusOneLim, modXiPlusOneLim, 2]

                # Calculate interpolation
                b = modXf * br + xf * bl
                t = modXf * tr + xf * tl

                new_rgb[x, y, 2] = <int> (modYf * t + (<float> 1.0 - modYf) * b + <float> 0.5)



    return frombuffer(new_rgb, (new_height, new_width), 'RGB')


# -------------------------------------------------------------------------------------------------------------------




@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef heatwave_array24_horiz_c(unsigned char [:, :, :] rgb_array,
                            unsigned char [:, :] mask_array,
                            float frequency, float amplitude, float attenuation=0.10,
                            unsigned char threshold=64):
    """
    HORIZONTAL HEATWAVE 

    DISTORTION EQUATION: 
    distortion = sin(x * attenuation + frequency) * amplitude * mask_array[x, y]
    Amplitude is equivalent to ((frequency % 2) / 1000.0) and will define the maximum pixel displacement.
    The highest the frequency the lowest the heat wave  


    :param rgb_array: numpy.ndarray or MemoryViewSlice, array shape (w, h, 3) containing RGB values
    :param mask_array: numpy.ndarray or  MemoryViewSlice shape (w, h) containing alpha values
    :param frequency: float; increment value. The highest the frequency the lowest the heat wave  
    :param amplitude: float; variable amplitude. Max amplitude is 10e-3 * 255 = 2.55 
    when alpha is 255 otherwise 10e-3 * alpha.
    :param attenuation: float; default 0.10
    :param threshold: unsigned char; Compare the alpha value with the threshold.
     if alpha value > threshold, apply the displacement to the texture otherwise no change
    :return: Return a pygame.Surface 24 bit format 
    """


    cdef int w, h
    w, h = (<object>rgb_array).shape[:2]

    cdef:
        unsigned char [:, :, ::1] new_array = empty((h, w, 3), dtype=numpy.uint8)
        int x = 0, y = 0, xx, yy
        float distortion


    with nogil:
        for x in prange(0, w, schedule='static', num_threads=THREADS):
            for y in range(h):
                distortion = sin(x * attenuation + frequency) * amplitude * mask_array[x, y]

                xx = <int>(x  + distortion + rand() * 0.0002)
                if xx > w - 1:
                    xx = w - 1
                if xx < 0:
                    xx = 0

                if mask_array[x, y] > threshold:
                    new_array[y, x, 0] = rgb_array[xx, y, 0]
                    new_array[y, x, 1] = rgb_array[xx, y, 1]
                    new_array[y, x, 2] = rgb_array[xx, y, 2]
                else:
                    new_array[y, x, 0] = rgb_array[x, y, 0]
                    new_array[y, x, 1] = rgb_array[x, y, 1]
                    new_array[y, x, 2] = rgb_array[x, y, 2]

    return numpy.asarray(new_array)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)

cpdef unsigned char[::1] stack_buffer_c(rgb_array_, alpha_, int w, int h, bint transpose=False):
    """
    Stack RGB & ALPHA MemoryViewSlice C-buffers structures together.
    If transpose is True, the output MemoryViewSlice is flipped.

    :param h: integer; Texture height
    :param w: integer; Texture width
    :param transpose: boolean; Transpose rows and columns (default False)
    :param rgb_array_: MemoryViewSlice or pygame.BufferProxy (C-buffer type) representing the texture
    RGB values filled with uint8
    :param alpha_:  MemoryViewSlice or pygame.BufferProxy (C-buffer type) representing the texture
    alpha values filled with uint8 
    :return: Return a contiguous MemoryViewSlice representing RGBA pixel values
    """

    cdef:
        int b_length = w * h * 3
        int new_length = w * h * 4
        unsigned char [:] rgb_array = rgb_array_
        unsigned char [:] alpha = alpha_
        unsigned char [::1] new_buffer =  numpy.empty(new_length, dtype=numpy.uint8)
        unsigned char [::1] flipped_array = numpy.empty(new_length, dtype=numpy.uint8)
        int i=0, j=0, ii, jj, index, k
        int w4 = w * 4

    with nogil:

        for i in prange(0, b_length, 3, schedule='static', num_threads=THREADS):
                ii = i // 3
                jj = ii * 4
                new_buffer[jj]   = rgb_array[i]
                new_buffer[jj+1] = rgb_array[i+1]
                new_buffer[jj+2] = rgb_array[i+2]
                new_buffer[jj+3] = alpha[ii]

        if transpose:
            for i in prange(0, w4, 4, schedule='static', num_threads=THREADS):
                for j in range(0, h):
                    index = i + (w4 * j)
                    k = (j * 4) + (i * h)
                    flipped_array[k    ] = new_buffer[index    ]
                    flipped_array[k + 1] = new_buffer[index + 1]
                    flipped_array[k + 2] = new_buffer[index + 2]
                    flipped_array[k + 3] = new_buffer[index + 3]
            return flipped_array

    return new_buffer

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef inline float [:, :] array2d_normalized_c(unsigned char [:, :] array):

    """
    NORMALIZED AN ARRAY

    Transform/convert an array shapes (w, h) containing unsigned char values
    into a MemoryViewSlice (2d array) with float values rescale in range [0 ... 1.0]

    :param array: numpy.array shape (w, h) containing unsigned int values (uint8)
    :return     : a MemoryViewSlice 2d array shape (w, h) with float values in range [0 ... 1.0]

    """
    cdef:
        int w, h
    try:
        w, h = array.shape[:2]
    except (ValueError, pygame.error) as e:
        raise ValueError('\nArray shape not understood. Only 2d array shape (w, h) are compatible.')

    cdef:
        int i = 0, j = 0
        float [:, :] array_f = numpy.empty((w, h), numpy.float32)

    with nogil:
        for i in prange(w, schedule='static', num_threads=THREADS):
            for j in range(h):
                array_f[i, j] = <float>(array[i, j] * ONE_255)
    return array_f


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef area24_c(int x, int y, np.ndarray[np.uint8_t, ndim=3] background_rgb,
              np.ndarray[np.uint8_t, ndim=2] mask_alpha, float intensity=1.0,
              float [:] color=numpy.array([128.0, 128.0, 128.0], dtype=numpy.float32, copy=False),
              bint smooth=False, bint saturation=False, float sat_value=0.2, bint bloom=False,
              bint heat=False, float frequency=1):
    """
    Create a realistic light effect on a pygame.Surface or texture.

    You can blit the output surface with additive mode using pygame flag BLEND_RGBA_ADD.


    :param x: integer, light x coordinates (must be in range [0..max screen.size x]
    :param y: integer, light y coordinates (must be in range [0..max screen size y]
    :param background_rgb: numpy.ndarray (w, h, 3) uint8. 3d array shape containing all RGB values
    of the background surface (display background).
    :param mask_alpha: numpy.ndarray (w, h) uint8, 2d array with light texture alpha values.
    For better appearances, choose a texture with a radial mask shape (maximum light intensity in the center)
    :param color: numpy.array; Light color (RGB float), default
    array([128.0 / 255.0, 128.0 / 255.0, 128.0 / 255.0], float32, copy=False)
    :param intensity: float; Light intensity range [0.0 ... 20.0]
    :param bloom: boolean; Bloom effect, default False
    :param sat_value: float; Set the saturation value
    :param saturation: boolean; Saturation effect
    :param smooth: boolean; Blur effect
    :param frequency: float; frequency must be incremental
    :param heat: boolean; Allow heat wave effect
    :return: Return a pygame surface 24 bit without per-pixel information,
    surface with same size as the light texture. Represent the lit surface.
    """

    assert intensity >= 0.0, '\nIntensity value cannot be > 0.0'


    cdef int w, h, lx, ly, ax, ay
    try:
        w, h = background_rgb.shape[:2]
    except (ValueError, pygame.error) as e:
        raise ValueError('\nArray shape not understood.')

    try:
        ax, ay = (<object>mask_alpha).shape[:2]
    except (ValueError, pygame.error) as e:
        raise ValueError('\nArray shape not understood.')

    # Return an empty surface if the x or y are not within the normal range.
    if (x < 0) or (x > w - 1) or (y < 0) or (y > h - 1):
        return Surface((ax, ay), SRCALPHA), ax, ay

    # return an empty Surface when intensity = 0.0
    if intensity == 0.0:
        return Surface((ax, ay), SRCALPHA), ax, ay

    lx = ax >> 1
    ly = ay >> 1

    cdef:
        np.ndarray[np.uint8_t, ndim=3] rgb = empty((ax, ay, 3), uint8, order='C')
        np.ndarray[np.uint8_t, ndim=2] alpha = empty((ax, ay), uint8, order='C')
        int i=0, j=0
        float f
        int w_low = lx
        int w_high = lx
        int h_low = ly
        int h_high = ly

    if x < lx:
        w_low = x
    elif x > w - lx:
        w_high = w - x

    if y < ly:
        h_low = y
    elif y >  h - ly:
        h_high = h - y

    rgb = background_rgb[x - w_low:x + w_high, y - h_low:y + h_high, :]
    alpha = mask_alpha[lx - w_low:lx + w_high, ly - h_low:ly + h_high]

    ax, ay = rgb.shape[:2]
    cdef:
        unsigned char [:, :, ::1] new_array = empty((ay, ax, 3), numpy.uint8)

    # NOTE the array is transpose
    with nogil:
        for i in prange(ax, schedule='static', num_threads=THREADS):
            for j in range(ay):
                f = alpha[i, j] * ONE_255 * intensity
                new_array[j, i, 0] = <unsigned char>fmin(rgb[i, j, 0] * f * color[0], 255.0)
                new_array[j, i, 1] = <unsigned char>fmin(rgb[i, j, 1] * f * color[1], 255.0)
                new_array[j, i, 2] = <unsigned char>fmin(rgb[i, j, 2] * f * color[2], 255.0)

    ay, ax = new_array.shape[:2]

    if smooth:
        shader_blur5x5_array24_inplace_c(new_array, mask=None, t=1)

    if saturation:
        shader_saturation_array24_inplace_c(new_array, sat_value)

    if heat:
        new_array = heatwave_array24_horiz_c(numpy.asarray(new_array).transpose(1, 0, 2),
            alpha, frequency, (frequency % 8) / 1000.0, attenuation=100, threshold=10)

    surface = pygame.image.frombuffer(new_array, (ax, ay), "RGB")

    if bloom:
        mask = array2d_normalized_c(alpha)
        shader_bloom_effect_array24_c(surface, threshold_=190, fast_=True, mask_=mask)

    return surface, ax, ay




@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef object chromatic(
        surface_,
        unsigned int delta_x,
        unsigned int delta_y,
        float zoom=0.9999,
        float fx=0.02
):
    """
    CHROMATIC ABERRATION 
    
    Create a chromatic aberration with an amplitude proportional to the 
    distance from the centre of the effect
    
    :param surface_ : pygame.Surface
    :param delta_x  : int; chromatic centre effect coordinate X, must be in range [0 ... w]
    :param delta_y  : int; chromatic centre effect coordinate Y, must be in range [0 ... h]
    :param zoom     : float; zoom factor 0.9999 (no zoom, full image), < 1.0 zoom-in. Must 
        be in range [0.0 ... 0.9999] 
    :param fx       : channel rgb layer offset default 0.02. Must be in range [0.0 ... 0.2]
    :return         : a chromatic aberration effect
    """
    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)

    cdef Py_ssize_t w, h
    w, h = surface_.get_size()


    if w == 0 or h == 0:
        raise ValueError("Surface width or height cannot be null!")

    if delta_x < 0 or delta_y < 0:
        raise ValueError("Arguments delta_x and delta_y must be > 0")

    delta_x %= w
    delta_y %= h

    if zoom < 0 or <float>floor(zoom) > <float>0.99999999:
        raise ValueError("Argument zoom must be in range [0.0 ... 0.999]")

    if 0 > <float>floor(fx) > 0.2:
        raise ValueError("Argument fx must be in range [0.0 ... 0.2]")

    cdef unsigned char [:, :, :] rgb_array

    try:
        rgb_array = pixels3d(surface_)

    except Exception as e:
        raise ValueError("\nCannot reference source pixels into a 3d array.\n %s " % e)

    cdef:
        unsigned char [:, :, ::1] new_array = empty((h, w, 3), dtype=numpy.uint8)
        int i = 0, j = 0
        float dw = <float>delta_y / <float>w
        float dh = <float>delta_x / <float>h
        float nx, ny, theta_rad, nx2, ny2, dist, new_dist, new_ii, new_jj
        int new_j, new_i

    with nogil:
        for j in prange(0, h, schedule='static', num_threads=THREADS):
            for i in range(w):
                nx = <float>(<float>i / <float>h) - dh
                ny = <float>(<float>j / <float>w) - dw
                theta_rad = <float>atan2 (ny,nx)
                nx2 = nx * nx
                ny2 = ny * ny
                dist = <float>sqrt(nx2 + ny2)
                new_dist = dist * (zoom - fx)
                new_ii = <float>cos(<float>theta_rad) * new_dist
                new_jj = <float>sin(<float>theta_rad) * new_dist
                new_j = <int>((new_jj + dw) * <float>w)
                new_i = <int>((new_ii + dh) * <float>h)
                new_array[j, i, 0] = rgb_array[new_i, new_j, 0]

                new_dist = dist * (zoom  - fx * <float>2.0)

                new_j = <int>((<float>sin(<float>theta_rad) * new_dist + dw) * <float>w)
                new_i = <int>((<float>cos(<float>theta_rad) * new_dist + dh) * <float>h)
                new_array[j, i, 1] = rgb_array[new_i, new_j, 1]

                new_dist = dist * (zoom  - fx * <float>3.0)

                new_ii = <float>cos(<float>theta_rad) * new_dist
                new_jj = <float>sin(<float>theta_rad) * new_dist

                new_j = <int>((new_jj + dw) * <float>w)
                new_i = <int>((new_ii + dh) * <float>h)

                new_array[j, i, 2] = rgb_array[new_i, new_j, 2]

    return frombuffer(new_array, (w, h), 'RGB').convert()




@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef object zoom(surface_, unsigned int delta_x, unsigned int delta_y, float zx=0.9999):
    """
    ZOOM WITHIN AN IMAGE
    
    Zoom-in or zoom-out (factor zx) toward a given centre point (delta_x, delta_y) 
    
    :param surface_ : pygame.Surface
    :param delta_x  : int; Zoom centre x coordinate must be in range [0 ... w] 
    :param delta_y  : int; Zoom centre y coordinate must be in range [0 ... h]
    :param zx       : float; Zoom factor (0.9999 no zoom) must be in range [0.0 ... 0.9999]
    :return         : Returns an image with a zoom effect
    """

    cdef int w, h
    w, h = surface_.get_size()

    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)

    if w == 0 or h == 0:
        raise ValueError("Surface width or height cannot be null!")

    if delta_x < 0 or delta_y < 0:
        raise ValueError("Arguments delta_x and delta_y must be > 0")

    delta_x %= w
    delta_y %= h

    if zx < 0 or <float>floor(zx) > <float>0.99999999:
        raise ValueError("Argument zx must be in range [0.0 ... 0.999]")


    cdef unsigned char [:, :, :] rgb_array

    try:
        rgb_array = pixels3d(surface_)

    except Exception as e:
        raise ValueError("\nCannot reference source pixels into a 3d array.\n %s " % e)

    cdef:
        unsigned char [:, :, ::1] new_array = empty((h, w, 3), dtype=numpy.uint8)
        int i = 0, j = 0
        float dw = delta_y / <float>w
        float dh = delta_x / <float>h
        float nx, ny, theta, nx2, ny2, dist, new_dist
        int new_j, new_i, r, g, b

    with nogil:
        for j in prange(0, h, schedule='static', num_threads=THREADS):
            for i in range(w):
                nx = <float>(<float>i / <float>h) - dh
                ny = <float>(<float>j / <float>w) - dw

                theta = <float>atan2 (ny,nx)

                nx2 = nx * nx
                ny2 = ny * ny

                dist = <float>sqrt(nx2 + ny2)
                new_dist = dist * zx

                new_j = <int>((<float>sin(<float>theta) * new_dist + dw) * <float>w)
                new_i = <int>((<float>cos(<float>theta) * new_dist + dh) * <float>h)

                new_array[j, i, 0] = rgb_array[new_i, new_j, 0]
                new_array[j, i, 1] = rgb_array[new_i, new_j, 1]
                new_array[j, i, 2] = rgb_array[new_i, new_j, 2]


    return frombuffer(new_array, (w, h), 'RGB').convert()



cpdef void shader_rgb_to_yiq_inplace(object surface_):
    """
    CONVERT IMAGE INTO GREYSCALE USING YIQ (LUMA INFORMATION)
    
    :param surface_: pygame.Surface; 
    :return: void
    """

    shader_rgb_to_yiq_inplace_c(pixels3d(surface_))

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef inline void shader_rgb_to_yiq_inplace_c(unsigned char [:, :, :] rgb_array):

    cdef Py_ssize_t w, h
    w, h = rgb_array.shape[:2]

    cdef:
        int i=0, j=0
        yiq yiq_
        rgb rgb_
        float h_
        unsigned char *r
        unsigned char *g
        unsigned char *b


    with nogil:
        for j in prange(h, schedule='static', num_threads=THREADS):
            for i in range(w):
                r = &rgb_array[i, j, 0]
                g = &rgb_array[i, j, 1]
                b = &rgb_array[i, j, 2]
                yiq_ = rgb_to_yiq(r[0] * <float>ONE_255, g[0] * <float>ONE_255, b[0] * <float>ONE_255)

                r[0] = min(<unsigned char>(yiq_.y * <float>255.0), 255)
                g[0] = min(<unsigned char>(yiq_.y * <float>255.0), 255)
                b[0] = min(<unsigned char>(yiq_.y * <float>255.0), 255)



cpdef void shader_rgb_to_yiq_i_comp_inplace(object surface_):
    """
    CONVERT IMAGE INTO YIQ MODEL (REPRESENT IN-PHASE VALUE)

    :param surface_: pygame.Surface; 
    :return: void
    """

    shader_rgb_to_yiq_i_comp_inplace_c(pixels3d(surface_))

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef inline void shader_rgb_to_yiq_i_comp_inplace_c(unsigned char [:, :, :] rgb_array):

    cdef Py_ssize_t w, h
    w, h = rgb_array.shape[:2]

    cdef:
        int i=0, j=0
        yiq yiq_
        rgb rgb_
        float h_
        unsigned char *r
        unsigned char *g
        unsigned char *b


    with nogil:
        for j in prange(h, schedule='static', num_threads=THREADS):
            for i in range(w):
                r = &rgb_array[i, j, 0]
                g = &rgb_array[i, j, 1]
                b = &rgb_array[i, j, 2]
                yiq_ = rgb_to_yiq(r[0] * <float>ONE_255, g[0] * <float>ONE_255, b[0] * <float>ONE_255)
                rgb_ = yiq_to_rgb(yiq_.y, yiq_.i, 0)
                r[0] = <unsigned char>(rgb_.r * <float>255.0)
                g[0] = <unsigned char>(rgb_.g * <float>255.0)
                b[0] = <unsigned char>(rgb_.b * <float>255.0)

cpdef void shader_rgb_to_yiq_q_comp_inplace(object surface_):
    """
    CONVERT IMAGE INTO YIQ MODEL (REPRESENT IN-PHASE VALUE)

    :param surface_: pygame.Surface; 
    :return: void
    """

    shader_rgb_to_yiq_q_comp_inplace_c(pixels3d(surface_))

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef inline void shader_rgb_to_yiq_q_comp_inplace_c(unsigned char [:, :, :] rgb_array):

    cdef Py_ssize_t w, h
    w, h = rgb_array.shape[:2]

    cdef:
        int i=0, j=0
        yiq yiq_
        rgb rgb_
        float h_
        unsigned char *r
        unsigned char *g
        unsigned char *b


    with nogil:
        for j in prange(h, schedule='static', num_threads=THREADS):
            for i in range(w):
                r = &rgb_array[i, j, 0]
                g = &rgb_array[i, j, 1]
                b = &rgb_array[i, j, 2]
                yiq_ = rgb_to_yiq(r[0] * <float>ONE_255, g[0] * <float>ONE_255, b[0] * <float>ONE_255)
                rgb_ = yiq_to_rgb(yiq_.y, 0, yiq_.q)
                r[0] = <unsigned char>(rgb_.r * <float>255.0)
                g[0] = <unsigned char>(rgb_.g * <float>255.0)
                b[0] = <unsigned char>(rgb_.b * <float>255.0)

