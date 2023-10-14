# cython: binding=False, boundscheck=False, wraparound=False, nonecheck=False, cdivision=True,
# profile=False, initializedcheck=False
# cython: optimize.use_switch=True
# cython: warn.maybe_uninitialized=False
# cython: warn.unused=False
# cython: warn.unused_result=False
# cython: warn.unused_arg=False
# cython: language_level=3
# encoding: utf-8

"""
                 GNU GENERAL PUBLIC LICENSE
                       Version 3, 29 June 2007

 Copyright (C) 2007 Free Software Foundation, Inc. <https://fsf.org/>
 Everyone is permitted to copy and distribute verbatim copies
 of this license document, but changing it is not allowed.

Copyright Yoann Berenguer
"""



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
    from cython.parallel cimport prange, parallel

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
from PygameShader.misc cimport color_diff_hsv, color_diff_hsl, close_color, min_index, \
    surface_copy_c
from PygameShader.BlendFlags import blend_add_array
from PygameShader.BlendFlags cimport blend_add_surface_c, blend_add_array_c

cimport numpy as np

from libc.math cimport sqrtf as sqrt, atan2f as atan2, sinf as sin,\
    cosf as cos, nearbyintf as nearbyint, expf as exp, powf as pow, floorf as floor, \
roundf as round_c, fminf as fmin, fmaxf as fmax, rintf

from libc.stdlib cimport malloc, rand, free


from libc.stdio cimport printf
from PygameShader.config import OPENMP, THREAD_NUMBER, __VERSION__

cdef int THREADS = 1

if OPENMP:
     THREADS = THREAD_NUMBER

DEF SCHEDULE = 'static'

cdef float M_PI = 3.14159265358979323846
cdef float INV_M_PI = 1.0 / M_PI
cdef float M_PI2 =3.14159265358979323846/2.0
cdef float M_2PI =2 * 3.14159265358979323846

cdef float RAD_TO_DEG=<float>(180.0/M_PI)
cdef float DEG_TO_RAD=<float>(M_PI/180.0)

cdef float C1 = <float>7.0/<float>16.0
cdef float C2 = <float>3.0/<float>16.0
cdef float C3 = <float>5.0/<float>16.0
cdef float C4 = <float>1.0/<float>16.0


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
DEF ONE_1024     = 1.0/1024

cdef float[360] COS_TABLE
cdef float[360] SIN_TABLE
cdef int ANGLE

with nogil:
    for ANGLE in prange(0, 360):
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


print("PygameShader version %s " % __VERSION__)
print("OPENMP %s " % OPENMP)
print("THREADS %s " % THREADS)
print("SCHEDULE %s " % SCHEDULE)

cdef:
    float [:, :] COLORS_CPC64_C = numpy.divide(COLORS_CPC64, <float>255.0).astype(dtype=float32)

@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cpdef inline float _randf(float lower, float upper)nogil:
    """
    Equivalent to random.uniform (much faster)
    """
    return randRangeFloat(lower, upper)

@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cpdef inline int _randi(int lower, int upper)nogil:
    """
    Equivalent to random.randint (much faster)
    """
    return randRange(lower, upper)

@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cpdef inline hsl _rgb_to_hsl(unsigned char r, unsigned char g, unsigned char b)nogil:
    """
    CONVERT RGB color to HSL 
    
    """

    cdef:
        hsl hsl_
    with gil:
        assert 0 < r <255 and 0 < g < 255 and 0 < b < 255, \
            "\nRGB values are unsigned char type, range [0 ... 255]"

    # divide be 255.0
    hsl_ = struct_rgb_to_hsl(
        r * <float>ONE_255,
        g * <float>ONE_255,
        b * <float>ONE_255
    ) # struct_rgb_to_hsl returns values between 0.0 ... 1.0

    hsl_.h *= 360;
    hsl_.s *= 100;
    hsl_.l *= 100;

    return hsl_

@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cpdef inline rgb _hsl_to_rgb(float h, float s, float l)nogil:
    """
    CONVERT HSL model to RGB
    HSL values are normalized h/360, s/100, l/100
    """
    cdef:
        rgb rgb_
    with gil:
        assert 0 < h <1.0 and 0 < s < 1.0 and 0 < h < 1.0, \
            "\nHSL values are normalized float, range [0.0 ... 1.0]"

    rgb_ = struct_hsl_to_rgb(
        h, s, l
    ) # struct_hsl_to_rgb returns values in range 0.0 ... 1.0

    rgb_.r *= <float>255.0
    rgb_.g *= <float>255.0
    rgb_.b *= <float>255.0

    return rgb_

@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cpdef inline hsv _rgb_to_hsv(unsigned char r, unsigned char g, unsigned char b)nogil:
    """
    CONVERT RGB color to HSV

    """

    cdef:
        hsv hsv_
    with gil:
        assert 0 < r < 255 and 0 < g < 255 and 0 < b < 255, \
            "\nRGB values are unsigned char type, range [0 ... 255]"

    # divide be 255.0
    hsv_ = struct_rgb_to_hsv(
        r * <float> ONE_255,
        g * <float> ONE_255,
        b * <float> ONE_255
    )  # struct_rgb_to_hsv returns values between 0.0 ... 1.0

    hsv_.h *= 360;
    hsv_.s *= 100;
    hsv_.v *= 100;

    return hsv_

@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cpdef inline rgb _hsv_to_rgb(float h, float s, float v)nogil:
    """
    CONVERT HSV model to RGB
    HSV values are normalized h/360, s/100, v/100
    """
    cdef:
        rgb rgb_
    with gil:
        assert 0 < h < 1.0 and 0 < s < 1.0 and 0 < v < 1.0, \
            "\nHSL values are normalized float, range [0.0 ... 1.0]"

    rgb_ = struct_hsv_to_rgb(
        h, s, v
    )  # struct_hsv_to_rgb returns values in range 0.0 ... 1.0

    rgb_.r *= <float> 255.0
    rgb_.g *= <float> 255.0
    rgb_.b *= <float> 255.0

    return rgb_

@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
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

@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
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

@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cpdef inline void rgb_to_bgr(object surface_):
    """  
    SHADER RGB to BGR (INPLACE)
  
    Convert your game display from RGB to BGR format (blue, green, red)
    This algorithm can also be used to transform pygame surface in the equivalent bgr format
    

    e.g:
    rgb_to_bgr(surface)

    :param surface_    : Pygame surface or display surface compatible (image 24-32 bit with or 
                         without per-pixel transparency / alpha channel)
    :return             : void
    """
    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)

    cdef unsigned char [:, :, :] rgb_array
    try:
        rgb_array = surface_.get_view('3')

    except Exception as e:
        raise ValueError("Cannot reference source pixels into a 3d array.\n %s " % e)


    rgb_to_bgr_inplace_c(rgb_array)

@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cpdef inline void rgb_to_brg(object surface_):
    """
    SHADER RGB TO BRG (INPLACE)

    Convert your game display from RGB to BRG format (blue, red, green)
    This algorithm can also be used to transform pygame texture in the equivalent BRG format
    
    e.g:
    rgb_to_brg(surface)

    :param surface_: Pygame surface or display surface compatible (image 24-32 bit with or without 
                     per-pixel transparency / alpha channel)
    :return: void 
    """
    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)

    cdef unsigned char [:, :, :] rgb_array
    try:
        rgb_array = surface_.get_view('3')

    except Exception as e:
        raise ValueError("Cannot reference source pixels into a 3d array.\n %s " % e)

    rgb_to_brg_inplace_c(rgb_array)

@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cpdef inline void greyscale(object surface_):
    """
    SHADER GRAYSCALE (INPLACE &CONSERVE LUMINOSITY)

    This shader transform the game display or pygame surface into a grayscale

    e.g:
    greyscale(surface)

    :param surface_  : Pygame surface or display surface compatible (image 24-32 bit with 
                       or without per-pixel transparency / alpha channel)
    :return          : void
    
    """
    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)

    cdef unsigned char [:, :, :] rgb_array
    try:
        rgb_array = surface_.get_view('3')

    except Exception as e:
        raise ValueError("Cannot reference source pixels into a 3d array.\n %s " % e)

    greyscale_luminosity24_inplace_c(rgb_array)

@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cpdef inline void sepia(object surface_):
    """
    SHADER SEPIA MODEL (INPLACE)

    Transform your video game into an equivalent sepia model

    e.g:
    sepia(surface)


    :param surface_  : Pygame surface or display surface compatible (image 24-32 bit with 
                       or without per-pixel transparency / alpha channel)
    :return:         : void
    """
    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)
    cdef unsigned char [:, :, :] rgb_array
    try:
        rgb_array = surface_.get_view('3')

    except Exception as e:
        raise ValueError("Cannot reference source pixels into a 3d array.\n %s " % e)

    sepia_inplace_c(rgb_array)


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cpdef inline void median(
        object surface_,
        unsigned short int kernel_size_=2,
        bint fast_=True,
        unsigned short int reduce_factor_=1
) except *:
    """
    SHADER MEDIAN FILTER (INPLACE) 
    
    This shader cannot be used for real time display rendering as the performance 
    of the algorithm is not satisfactory.
    

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

    if kernel_size_ <= 1:
        raise ValueError('\nArgument kernel_size_ must be > 1')

    cdef unsigned char [:, :, :] rgb_array
    if fast_:
        if not 0 < reduce_factor_ < 9:
            raise ValueError('\nArgument reduce_factor_ must be > 1 and < 8 ')
        median_fast(surface_, kernel_size_, reduce_factor_)

    else:
        try:
            rgb_array = surface_.get_view('3')

        except Exception as e:
            raise ValueError("Cannot reference source pixels into a 3d array.\n %s " % e)

        median_inplace_c(rgb_array, kernel_size_)



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cpdef inline void painting(
        object surface_,
) except *:
    """
    PAINTING (INPLACE)

    This algorithm cannot be used for real time rendering, use it offline to modify 
    images/surface.


    :param surface_      : Pygame surface or display surface compatible (image 24-32 bit with 
                           or without per-pixel transparency / alpha channel)  
    :return:             : void
    """
    cdef:
        unsigned char kernel_size_ = 8
        bint fast_ = True
        unsigned char reduce_factor_ = 1

    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)

    median_fast(surface_, kernel_size_, reduce_factor_)



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cpdef inline void pixels(object surface_) except *:
    """
    PIXELATION (INPLACE)

    :param surface_      : Pygame surface or display surface compatible (image 24-32 bit with 
                           or without per-pixel transparency / alpha channel)  
    :return:             : void
    """
    cdef:
        unsigned char kernel_size_ = 2
        bint fast_ = True
        unsigned char reduce_factor_ = 4

    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)

    median_fast(surface_, kernel_size_, reduce_factor_)


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cdef inline void median_fast(
        object surface_,
        unsigned short int kernel_size_=2,
        unsigned short int reduce_factor_=1
):
    """
    This function is used by the algorithms median, painting and pixels (cannot be call directly 
    from python) 

    :param surface_: pygame.surface; Surface compatible 24-32 bit 
    :param kernel_size_: integer; size of the kernel 
    :param reduce_factor_: integer; value of 1 divide the image by 2, value of 2 div the image by 4
    :return: void
    """
    try:
        rgb_array = surface_.get_view('3')

    except Exception as e:
        raise ValueError("Cannot reference source pixels into a 3d array.\n %s " % e)


    # surface_cp = surface_.copy()
    surface_cp = surface_.copy()
    cdef:
        int w, h
    w, h = surface_cp.get_size()

    surface_cp = smoothscale(surface_cp, (w >> reduce_factor_, h >> reduce_factor_))

    try:
        cp_array = surface_cp.get_view('3')

    except Exception as e:
        raise ValueError("Cannot reference source pixels into a 3d array.\n %s " % e)

    cdef:
        int i, j
        unsigned char[:, :, :] org_surface = rgb_array
        unsigned char[:, :, :] surface_cp_arr = cp_array

    median_inplace_c(surface_cp_arr, kernel_size_)
    surface_cp_arr = scale_array24_c(surface_cp_arr, w, h)

    with nogil:
        for i in prange(w, schedule=SCHEDULE, num_threads=THREADS):
            for j in range(h):
                org_surface[i, j, 0] = surface_cp_arr[i, j, 0]
                org_surface[i, j, 1] = surface_cp_arr[i, j, 1]
                org_surface[i, j, 2] = surface_cp_arr[i, j, 2]

@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cpdef inline void median_grayscale(
        object surface_,
        int kernel_size_=2
):
    """
    SHADER MEDIAN GRAYSCALE (INPLACE)

    This shader cannot be used for real time rendering as the performance of the algorithm are not
    satisfactory. 

    The surface is compatible 24 - 32 bit with or without alpha layer
    
    :param surface_: pygame.Surface; compatible 24 - 32 bit with or without alpha layer
    :param kernel_size_: integer; Kernel size (must be > 0), default value = 2
    :return: void 
    """
    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)
    cdef unsigned char [:, :, :] rgb_array
    assert kernel_size_ > 1, "\nArgument kernel_size_ must be > 1"
    try:
        rgb_array = surface_.get_view('3')

    except Exception as e:
        raise ValueError("Cannot reference source pixels into a 3d array.\n %s " % e)

    shader_median_grayscale_filter24_inplace_c(rgb_array, kernel_size_)




@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cpdef inline void color_reduction(
        object surface_,
        int color_=8
):
    """
     COLOR REDUCTION SHADER (INPLACE)

    Decrease the amount of colors in the display or texture.
    The method of color reduction is very simple: every color of the original picture is replaced
    by an appropriate color from the limited palette that is accessible.
    
    The surface is compatible 24 - 32 bit with or without alpha layer
      
    e.g:
    color_reduction(surface, 8)

    :param surface_: pygame.Surface; compatible 24 - 32 bit 
    :param color_: integer must be > 0 default 8
    :return: void 
    """
    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)

    assert color_ > 0, "Argument color_number must be > 0"

    cdef unsigned char [:, :, :] rgb_array

    try:
        rgb_array = surface_.get_view('3')

    except Exception as e:
        raise ValueError("Cannot reference source pixels into a 3d array.\n %s " % e)

    color_reduction_inplace_c(rgb_array, color_)


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cpdef inline void sobel(
        object surface_,
        int threshold_ = 64
):
    """
    SHADER SOBEL (INPLACE) 

    Transform the game display or a pygame surface into a sobel equivalent model
    (surface edge detection)

    The surface is compatible 24 - 32 bit with or without alpha layer
    

    e.g:
    sobel(surface, 64)

    :param surface_: pygame.Surface; compatible 24 - 32 bit 
    :param threshold_: integer; Value for detecting the edges default 64
    :return:
    """
    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)

    assert -1 < threshold_ < 256, "\nArgument threshold must be an integer in range [0 ... 255]"
    cdef unsigned char [:, :, :] rgb_array
    try:
        rgb_array = surface_.get_view('3')

    except Exception as e:
        raise ValueError("Cannot reference source pixels into a 3d array.\n %s " % e)

    sobel_inplace_c(rgb_array, threshold_)

@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cpdef inline void sobel_fast(
        object surface_,
        int threshold_ = 64,
        unsigned short factor_ = 1
):
    """
    SHADER FAST SOBEL (INPLACE)

    Transform the game display or a pygame surface into a sobel equivalent model
    (surface edge detection).This version is slightly fastest than sobel_inplace_c as
    it down-scale the array containing all the pixels and apply the sobel algorithm to a smaller
    sample. When the processing is done, the array is re-scale to its original dimensions.
    If this method is in theory faster than sobel_inplace_c, down-scaling and up-scaling
    an array does have a side effect of decreasing the overall image definition
    (jagged lines non-antialiasing)
    
    Compatible 24 - 32 bit with or without alpha layer
      
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

    sobel_fast_inplace_c(surface_, threshold_, factor_)



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cpdef inline void invert(object surface_):
    """
    SHADER INVERT PIXELS (INPLACE)
    
    Invert all pixels of the display or a given texture
    
    Compatible 24 - 32 bit with or without alpha layer
    
    
    e.g:
    invert(surface)
    
    :param surface_: pygame.surface; compatible 24 - 32 bit surfaces
    :return: void
    """
    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)
    cdef unsigned char [:, :, :] rgb_array
    try:
        rgb_array = surface_.get_view('3')

    except Exception as e:
        raise ValueError("Cannot reference source pixels into a 3d array.\n %s " % e)

    invert_inplace_c(rgb_array)



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cpdef inline void hsl_effect(object surface_, float shift_):
    """
    ROTATE THE HUE OF THE GAME DISPLAY OR GIVEN TEXTURE (INPLACE)
    
    Compatible 24 - 32 bit with or without alpha layer
      
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
    cdef unsigned char [:, :, :] rgb_array
    try:
        rgb_array = surface_.get_view('3')

    except Exception as e:
        raise ValueError("Cannot reference source pixels into a 3d array.\n %s " % e)

    hsl_inplace_c(rgb_array, shift_)




@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cpdef inline void hsv_effect(object surface_, float shift_):
    """
    ROTATE THE HUE OF THE GAME DISPLAY OR GIVEN TEXTURE (INPLACE)

    Compatible 24 - 32 bit with or without alpha layer

    e.g:
    hsv_effect(surface, 0.2)

    :param surface_: pygame.Surface; Compatible 24 - 32 bit surfaces
    :param shift_: float; float value in range [-1.0 ... 1.0]
    :return: void
    """
    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)

    assert -1.0 <= shift_ <= 1.0, \
        "Argument shift must be in range[0.0 ... 1.0]"

    cdef unsigned char [:, :, :] rgb_array
    try:
        rgb_array = surface_.get_view('3')

    except Exception as e:
        raise ValueError("Cannot reference source pixels into a 3d array.\n %s " % e)

    hsv_inplace_c(rgb_array, shift_)


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cpdef inline void hsl_fast(
        object surface_,
        float shift_,
        float [:, :, :, ::1] rgb_to_hsl_,
        unsigned char [:, :, :, ::1] hsl_to_rgb_
):
    """    
    ROTATE THE HUE OF AN IMAGE USING STORED HSL TO 
    RGB AND RGB TO HSL VALUES (INPLACE)
    
    This method is 25% faster than hsl_effect
    
    To use this algorithm you would have to pre-cache the converted values of both models 
    RGB to HSL and HSL to RGB using the functions hsl_to_rgb_model & rgb_to_hsl_model.
    
    Compatible 24 - 32 bit with or without alpha layer

    e.g:
    rgb2hsl_model = hsl_to_rgb_model()
    hsl2rgb_model = rgb_to_hsl_model()

    while game:
        hsl_fast(
            image,
            hsl_value,
            rgb_to_hsl_=rgb2hsl_model,
            hsl_to_rgb_=hsl2rgb_model)

    :param surface_: pygame.Surface; compatible 24 - 32 bit surfaces
    :param shift_: float; value must be in range [ -1.0 ... + 1.0]
    :param hsl_to_rgb_: 3d numpy.ndarray shape (256, 256, 256, 3) see hsl_to_rgb_model function
    :param rgb_to_hsl_: 3d numpy.ndarray shape (256, 256, 256, 3) see rgb_to_hsl_model function
    :return:
    """
    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)
    assert -1.0 <= shift_ <= 1.0, \
        "Argument shift must be in range[-1.0 ... 1.0]"
    assert PyObject_IsInstance(hsl_to_rgb_, (numpy.ndarray, cython.view.memoryview)), \
        "\nArgument hsl_to_rgb_ must be a numpy.ndarray or memoryview type, got %s " % type(
            hsl_to_rgb_)
    assert PyObject_IsInstance(rgb_to_hsl_, (numpy.ndarray, cython.view.memoryview)), \
        "\nArgument rgb_to_hsl_ must be a numpy.ndarray or memoryview type, got %s " % type(
            rgb_to_hsl_)
    cdef unsigned char [:, :, :] rgb_array
    try:
        rgb_array = surface_.get_view('3')

    except Exception as e:
        raise ValueError("Cannot reference source pixels into a 3d array.\n %s " % e)

    hsl_fast_inplace_c(rgb_array, shift_, rgb_to_hsl_, hsl_to_rgb_)


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cpdef inline void blur(object surface_, t_=1):
    """
    APPLY A GAUSSIAN BLUR EFFECT TO THE GAME DISPLAY (INPLACE)

    This method is using convolution property and process the image in two passes,
    first the horizontal convolution and last the vertical convolution
    pixels convoluted outside image edges will be set to adjacent edge value
    Apply a 5x5 kernel.
    
    Compatible 24 - 32 bit with or without alpha layer
     
    :param surface_: pygame.Surface; compatible 24 - 32 bit surfaces
    :param t_      : integer; must be >0; number of passes (default 1)
    :return: void 
    """
    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)
    assert t_ > 0, \
        "\nArgument t_ must be > 0, got %s " % t_

    cdef unsigned char [:, :, :] rgb_array
    try:
        rgb_array = surface_.get_view('3')

    except Exception as e:
        raise ValueError("Cannot reference source pixels into a 3d array.\n %s " % e)

    blur_array_inplace_c(rgb_array, None, t_)


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cpdef void blur5x5_array24_inplace(rgb_array_, mask_=None, t_=1):
    """
    BLUR ARRAY DIRECTLY (INPLACE)
    HOOK FOR THE METHOD blur_array_inplace_c
    """
    blur_array_inplace_c(rgb_array_, mask_, t_)


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cpdef inline void wave(object surface_, float rad, int size=5):
    """
    CREATE A WAVE EFFECT TO THE GAME DISPLAY OR SURFACE (INPLACE)

    The variable rad represent the angle (in radian) changing overtime, 
    rad cannot be a constant. 
    
    Compatible 24 - 32 bit with or without alpha layer

    e.g:
    wave(surface, 8 * math.pi/180.0 + frame_number, 5)
    wave(surface, x * math.pi/180.0, 5)
    
    :param surface_: pygame.Surface; pygame surface compatible 24 - 32 bit  
    :param rad     : float; angle in rad to rotate over time, default 0.139 
    :param size    : int; Number of sub-surfaces, default is 5 
    :return        : void
    """
    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)
    assert size > 0, "Argument size must be > 0"
    cdef unsigned char [:, :, :] rgb_array
    try:
        rgb_array = surface_.get_view('3')

    except Exception as e:
        raise ValueError("Cannot reference source pixels into a 3d array.\n %s " % e)

    wave_inplace_c(rgb_array, rad, size)


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cpdef inline void wave_static(object surface_, array_, float rad, int size=5):
    """
    WAVE EFFECT FOR STATIC BACKGROUND (INPLACE)
    
    This method is much faster than wave method 

    The variable rad represent the angle (in radian) changing overtime, 
    rad cannot be a constant. 
    Array_ must have the same dimensions and size than the surface

    Compatible 24 - 32 bit with or without alpha layer

    e.g:
    
    # Outside of your game loop create convert your surface into a numpy array
    array_cp = pygame.surfarray.pixels3d(BCK_COPY) 
    
    # in the game loop   
    wave_static(surface, 8 * math.pi/180.0 + frame_number, 5)
    wave_static(surface, x * math.pi/180.0, 5)

    :param surface_: pygame.Surface; pygame surface compatible 24 - 32 bit  
    :param array_  : numpy.ndarray (array containing copied pixels of the surface)
    :param rad     : float; angle in rad to rotate over time, default 0.139 
    :param size    : int; Number of sub-surfaces, default is 5 
    :return        : void
    """
    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)
    assert PyObject_IsInstance(array_, numpy.ndarray), \
        "\nArgument surface_ must be a numpy ndarray type, got %s " % type(array_)
    assert size > 0, "Argument size must be > 0"
    cdef unsigned char [:, :, :] rgb_array
    try:
        rgb_array = surface_.get_view('3')

    except Exception as e:
        raise ValueError("Cannot reference source pixels into a 3d array.\n %s " % e)

    wave_static_inplace_c(rgb_array, array_, rad, size)


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cpdef inline void swirl(object surface_, float degrees):
    """
    SWIRL AN IMAGE (INPLACE WITH ANGLE APPROXIMATION)

    This algorithm uses a table of cos and sin.
    
    Compatible 24 - 32 bit with or without alpha layer

    e.g:
    swirl_static(BCK, angle)
    
    :param surface_: pygame.Surface, compatible 24 - 32 bit 
    :param degrees : float; angle in degrees 
    :return        : void 
    """
    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)
    cdef unsigned char [:, :, :] rgb_array
    try:
        rgb_array = surface_.get_view('3')

    except Exception as e:
        raise ValueError("Cannot reference source pixels into a 3d array.\n %s " % e)

    swirl_inplace_c(rgb_array, degrees)



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cpdef inline void swirl_static(object surface_, array_, float degrees):
    """
    SWIRL STATIC IMAGE/BACKGROUND (INPLACE WITH ANGLE APPROXIMATION)

    This algorithm uses a table of cos and sin.
    
    array_ must have the exact same size/dimension than the image/surface

    Compatible 24 - 32 bit with or without alpha layer

    e.g:
    # outside the main loop
    array_cp = pygame.surfarray.pixels3d(BCK_COPY)
    
    # in the main loop
    swirl_static(BCK, array_cp, angle)

    :param surface_: pygame.Surface, compatible 24 - 32 bit 
    :param array_  : numpy.ndarray (pixels copy) 
    :param degrees : float; angle in degrees 
    :return        : void 
    """
    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)
    assert PyObject_IsInstance(array_, numpy.ndarray), \
        "\nArgument surface_ must be a numpy ndarray type, got %s " % type(array_)


    cdef unsigned char [:, :, :] rgb_array
    try:
        rgb_array = surface_.get_view('3')

    except Exception as e:
        raise ValueError("Cannot reference source pixels into a 3d array.\n %s " % e)

    swirl_static_inplace_c(rgb_array, array_, degrees)




@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cpdef inline void swirl_inplace(object surface_, float degrees):
    """
    SWIRL AN IMAGE WITHOUT ANGLE APPROXIMATION (INPLACE)

    Compatible 24 - 32 bit with or without alpha layer
       
    e.g:
    swirl2(surface_, angle)
    
    :param surface_: pygame.Surface, compatible 24 - 32 bit 
    :param degrees : float; angle in degrees
    :return        : void 
    """
    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)
    cdef unsigned char [:, :, :] rgb_array
    try:
        rgb_array = surface_.get_view('3')

    except Exception as e:
        raise ValueError("Cannot reference source pixels into a 3d array.\n %s " % e)

    swirl_inplace_c1(rgb_array, degrees)




@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cpdef inline void swirl2_static(object surface_, array_, float degrees):
    """
    SWIRL STATIC IMAGE/BACKGROUND WITHOUT ANGLE APPROXIMATION (INPLACE)

    Compatible 24 - 32 bit with or without alpha layer

    e.g:
    # outside the main loop
    array_cp = pygame.surfarray.pixels3d(image)
    
    # in the main loop
    swirl2_static(BCK, array_cp, angle)

    :param surface_: pygame.Surface, compatible 24 - 32 bit 
    :param array_  : numpy.ndarray pixel copy
    :param degrees : float; angle in degrees
    :return        : void 
    """
    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)
    assert PyObject_IsInstance(array_, numpy.ndarray), \
        "\nArgument surface_ must be a numpy.ndarray type, got %s " % type(array_)

    cdef unsigned char [:, :, :] rgb_array
    try:
        rgb_array = surface_.get_view('3')

    except Exception as e:
        raise ValueError("Cannot reference source pixels into a 3d array.\n %s " % e)

    swirl2_static_inplace_c1(rgb_array, array_, degrees)





@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cpdef inline void plasma_config(
        object surface_,
        int frame,
        float hue_=<float>1.0/<float>6.0,
        float sat_=<float>1.0/<float>6.0,
        float value_=<float>1.0/<float>8.0,
        float a_=<float>1.0/<float>255.0,
        float b_=<float>1.0/<float>12.0,
        float c_=<float>1.0/<float>12.0
):
    """

    CREATE A BASIC PLASMA EFFECT ON THE TOP OF A PYGAME SURFACE (INPLACE)

    Compatible 24 - 32 bit with or without alpha layer
    
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
    cdef unsigned char [:, :, :] rgb_array
    try:
        rgb_array = surface_.get_view('3')

    except Exception as e:
        raise ValueError("Cannot reference source pixels into a 3d array.\n %s " % e)

    plasma_inplace_c(rgb_array, frame, hue_, sat_, value_, a_, b_, c_)

@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cpdef inline void plasma(surface_, float frame, unsigned int [::1] palette_):
    """
    CREATE A PLASMA EFFECT INPLACE

    e.g:
    plasma(surface, frame_number, palette_)
    
    :param surface_: pygame.Surface; compatible 24 - 32 bit 
    :param frame   : float; frame number
    :param palette_: 1d array containing colors
    :return: void
    """
    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)
    plasma_c(surface_, frame, palette_)


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cpdef inline float [:, :, :, ::1] rgb_to_hsl_model():
    """
    Create an HSL model containing all precalculate values
    :return: Return a cython.view.memoryview shape (256, 256, 256, 3)
    """
    return rgb_to_hsl_model_c()

@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cpdef inline unsigned char [:, :, :, ::1] hsl_to_rgb_model():
    """
    Create an RGB model containing all precalculate values
    :return: Return a cython.view.memoryview shape (256, 256, 256, 3)
    """
    return hsl_to_rgb_model_c()


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cpdef inline void brightness(object surface_, float shift_):
    """
    SHADER BRIGHTNESS (INPLACE)

    This shader control the pygame display brightness level
    It uses two external functions coded in C, struct_rgb_to_hsl & struct_hsl_to_rgb
    
    Parameter shift_ is a float value in range [ -1.0 ... 1.0]. with +1.0 for the 
    maximum brightness. A value of 0.0 will not perform any changes to the original 
    surface
    
    Compatible 24 - 32 bit with or without alpha layer
      
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
    cdef unsigned char [:,:,:] rgb_array
    try:
        rgb_array = surface_.get_view('3')

    except Exception as e:
        raise ValueError("Cannot reference source pixels into a 3d array.\n %s " % e)

    brightness_inplace_c(rgb_array, shift_)




@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cpdef inline brightness_(object surface_, float shift_):
    """
    SHADER BRIGHTNESS

    Apply the transformation to a new surface

    This shader control the pygame display brightness level
    It uses two external functions coded in C, struct_rgb_to_hsl & struct_hsl_to_rgb

    Parameter shift_ is a float value in range [ -1.0 ... 1.0]. with +1.0 for the
    maximum brightness. A value of 0.0 will not perform any changes to the original
    surface

    Compatible 24 - 32 bit with or without alpha layer

    e.g:
    brightness_(surface, 0.2)

    :param surface_ : pygame.surface;
    :param shift_   : float must be in range [ -1.0 ... 1.0 ]
    :return         : pygame surface
    """
    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)

    if shift_ == 0.0:
        return surface_

    assert -1.0 <= shift_ <= 1.0, "\nArgument shift_ must be in range [-1.0 ... 1.0]"
    cdef unsigned char [:,:,:] rgb_array
    try:
        rgb_array = surface_.get_view('3')

    except Exception as e:
        raise ValueError("Cannot reference source pixels into a 3d array.\n %s " % e)

    return brightness_c(rgb_array, shift_)


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cpdef inline void brightness_exclude(
        object surface_,
        float shift_,
        color_=(0, 0, 0)
):
    """

    INCREASE/DECREASE AN IMAGE BRIGHTNESS (OPTIONAL EXCLUDE COLOR)
    
    The optional setting (color_) allow you to select a color that will not 
    be included in the process. This can be useful if you know the background 
    color RGB values and do not wish the background to change
    
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
    cdef unsigned char [:,:,:] rgb_array
    try:
        rgb_array = surface_.get_view('3')

    except Exception as e:
        raise ValueError("Cannot reference source pixels into a 3d array.\n %s " % e)

    brightness_exclude_inplace_c(rgb_array, shift_, color_)


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
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
    cdef unsigned char [:,:,:] rgb_array
    try:
        rgb_array = surface_.get_view('3')

    except Exception as e:
        raise ValueError("Cannot reference source pixels into a 3d array.\n %s " % e)

    brightness_bpf_c(rgb_array, shift_, bpf_threshold)

@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
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
    cdef unsigned char [:,:,:] rgb_array
    try:
        rgb_array = surface_.get_view('3')

    except Exception as e:
        raise ValueError("Cannot reference source pixels into a 3d array.\n %s " % e)

    brightness_inplace1_c(rgb_array, shift_, rgb_to_hsl_model)



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
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

    if shift_ < -1.0:
        shift_ = -1.0

    if shift_ > 1.0:
        shift_ = 1.0

    cdef unsigned char [:, :, :] rgb_array

    try:
        rgb_array = surface_.get_view('3')

    except Exception as e:
        raise ValueError("Cannot reference source pixels into a 3d array.\n %s " % e)

    saturation_inplace_c(rgb_array, shift_)


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
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

    cdef unsigned char [:,:,:] rgb_array
    try:
        rgb_array = surface_.get_view('3')

    except Exception as e:
        raise ValueError("Cannot reference source pixels into a 3d array.\n %s " % e)

    heatwave24_vertical_inplace_c(rgb_array, mask, factor_, center_, sigma_, mu_)


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cpdef inline void horizontal_glitch(
        object surface_,
        float rad1_,
        float frequency_,
        float amplitude_
):
    """
    SHADER GLITCH EFFECT (INPLACE)

    Deform the pygame display to create a glitch effect
    
    e.g:
    horizontal_glitch(BCK, rad1_=0.5, frequency_=0.08, amplitude_=FRAME % 20)

    :param surface_  : pygame.Surface; compatible 24 - 32 bit 
    :param rad1_     : float; Angle in radians, this value control the angle variation over time
    :param frequency_: float; signal frequency, factor that amplify the angle variation
    :param amplitude_: float; cos amplitude value
    :return: void
    """
    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)

    cdef unsigned char [:,:,:] rgb_array
    try:
        rgb_array = surface_.get_view('3')

    except Exception as e:
        raise ValueError("Cannot reference source pixels into a 3d array.\n %s " % e)

    horizontal_glitch_inplace_c(rgb_array, rad1_, frequency_, amplitude_)



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cpdef inline void horizontal_static_glitch(
        object surface_,
        object array_,
        float rad1_,
        float frequency_,
        float amplitude_
):
    """
    SHADER GLITCH EFFECT ON STATIC IMAGE/BACKGROUND (INPLACE)

    Deform the pygame display to create a glitch effect

    e.g:
    horizontal_glitch(BCK, rad1_=0.5, frequency_=0.08, amplitude_=FRAME % 20)

    :param surface_  : pygame.Surface; compatible 24 - 32 bit 
    :param array_    : numpy.ndarray pixel copy
    :param rad1_     : float; Angle in radians, this value control the angle variation over time
    :param frequency_: float; signal frequency, factor that amplify the angle variation
    :param amplitude_: float; cos amplitude value
    :return: void
    """
    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)
    assert PyObject_IsInstance(array_, numpy.ndarray), \
        "\nArgument surface_ must be a numpy.ndarray type, got %s " % type(array_)

    cdef unsigned char [:,:,:] rgb_array

    try:
        rgb_array = surface_.get_view('3')

    except Exception as e:
        raise ValueError("Cannot reference source pixels into a 3d array.\n %s " % e)

    horizontal_glitch_static_inplace_c(rgb_array, array_, rad1_, frequency_, amplitude_)

@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
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

    cdef unsigned char [:,:,:] rgb_array
    try:
        rgb_array = surface_.get_view('3')

    except Exception as e:
        raise ValueError("Cannot reference source pixels into a 3d array.\n %s " % e)

    bpf24_inplace_c(rgb_array, threshold)


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cpdef inline void bloom(object surface_, int threshold_, bint fast_=False,
                        object mask_=None):
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

    bloom_array24_c(surface_, threshold_, fast_, mask_)



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cpdef void bloom_array24(surface_, threshold_, fast_, mask_):
    bloom_array24_c(surface_, threshold_, fast_, mask_)

@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cpdef inline fisheye_footprint(
        int w, int h,
        unsigned int centre_x,
        unsigned int centre_y):
    """
    CREATE A FISHEYE MODEL TO HOLD THE PIXEL COORDINATES OF A SURFACE/ GAME DISPLAY

    * The surface and the model must have the same dimensions.

    Store the fisheye model into an external array image_fisheye_model shape (width, height, 2)

    IMAGE_FISHEYE_MODEL contains the fisheye transformation coordinate (x2 & y2) that reference
    the final image pixel position (fisheye model)
    This method has to be call once before the main loop in order to calculate
    the projected position for each pixels.

    :param centre_y: centre y coordinate
    :param centre_x: centre x coordinate
    :param w: integer; width of the fisheye model
    :param h: integer; height of the fisheye model
    :return: Return a numpy.ndarray type (w, h, 2) representing the fisheye model (coordinates
    of all surface pixels passing through the fisheye lens model)
    """
    return fisheye_footprint_c(w, h, centre_x, centre_y)




@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cpdef inline void fisheye_footprint_param(
        tmp_array_,
        float centre_x,
        float centre_y,
        float param1_,
        float focal_length,
        ):
    """
    CREATE A FISHEYE MODEL TO HOLD THE PIXEL COORDINATES OF A PYGAME SURFACE/ GAME DISPLAY

    This version contains two additional variables param1_ & focal_length to control
    the fisheye model aspect.

    :param tmp_array_: numpy.ndarray shape (w, h, 2) of unsigned int
      e.g: tmp = numpy.ndarray((400, 400, 2), dtype=numpy.uint32, order='C').
      The tmp_array_ array shape will determine the fisheye model.
      (Declare the tmp outside the main loop)
    :param centre_x  : float; fisheye x centre coordinate
    :param centre_y  : float; fisheye y centre coordinate
    :param param1_   : float; Control the fisheye aspect. param1_ > 1.0 converge lens effect,
    :param focal_length : float; Control the fisheye type zoom_ > 1 diverging lens
      zoom_ < 0 converging lens
    """
    assert isinstance(tmp_array_, numpy.ndarray), \
        "\nArgument tmp_array_ must be a numpy.ndarray type, got %s " % type(tmp_array_)

    fisheye_footprint_param_c(
        tmp_array_, centre_x, centre_y, param1_, focal_length)


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cdef inline void fisheye_footprint_param_c(
        unsigned int [:, :, :] tmp_array_,
        float centre_x,
        float centre_y,
        float param1,
        float focal_length,
):

    cdef:
        Py_ssize_t w = <object>tmp_array_.shape[0]
        Py_ssize_t h = <object>tmp_array_.shape[1]
        int y, x
        float ny, ny2, nx, nx2, r, theta, nr
        float c1 = <float>2.0/w
        float c2 = <float>2.0/h
        float w2 = centre_x
        float h2 = centre_y

    with nogil:

        for x in prange(w, schedule=SCHEDULE, num_threads=THREADS):
            nx = x * c1 - <float>1.0
            nx2 = nx * nx

            for y in range(h):
                ny = y * c2 - <float>1.0
                ny2 = ny * ny

                r = <float>sqrt(nx2 + ny2)

                if r > 1:
                    tmp_array_[x, y, 0] = <unsigned int>0
                    tmp_array_[x, y, 1] = <unsigned int>0
                    continue

                nr = (r + <float>param1 - <float>sqrt(
                    <float>1.0 - (nx2 + ny2))) * <float>focal_length

                theta = <float>atan2(ny, nx)
                tmp_array_[x, y, 0] = <unsigned int> (nr * <float>cos(theta) * w2 + w2)
                tmp_array_[x, y, 1] = <unsigned int> (nr * <float>sin(theta) * h2 + h2)







@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cpdef inline void fisheye(
        object surface_, unsigned int [:, :, ::1] fisheye_model):
    """
    THIS SHADER CAN BE USE TO DISPLAY THE GAME THROUGH A LENS EFFECT (INPLACE)

    Display a fisheye effect in real time given a surface referencing the
    pixels RGB. In order to accomplish a real time calculation, 
    this algorithm is using a pre-calculated transformation stored
    in the array fisheye_model.
    
    The function fisheye_footprint_c has to be called prior
    fisheye_inplace_c in order to store the transformation values.

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

    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)

    assert PyObject_IsInstance(fisheye_model, (cython.view.memoryview, numpy.ndarray)), \
        "\nArgument fisheye_model must be a numpy.ndarray or a cython.view.memoryview  type, " \
        "got %s " % type(fisheye_model)

    cdef unsigned char [:,:,:] rgb_array
    try:
        rgb_array = surface_.get_view('3')

    except Exception as e:
        raise ValueError("Cannot reference source pixels into a 3d array.\n %s " % e)

    fisheye_inplace_c(rgb_array, fisheye_model)


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cpdef inline void tv_scan(surface_, int space=5):
    """
    
    SHADER CREATING A TV SCANLINE EFFECT ON PYGAME SURFACE (INPLACE)

    The space between each scanline can by adjusted with the space value.
    The scanline intensity/colors is lower that the original image

    :param surface_     : pygame.Surface compatible 24-32 bit 
    :param space        : integer; space between the lines
    :return             : void
    
    """

    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)

    assert space > 0, "Argument space cannot be <=0"
    cdef unsigned char [:, :, :] rgb_array
    try:
        rgb_array = surface_.get_view('3')

    except Exception as e:
        raise ValueError("Cannot reference source pixels into a 3d array.\n %s " % e)

    tv_scanline_inplace_c(rgb_array, space)



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cpdef inline void rgb_split(object surface_, int offset_=10):
    """
    
    THIS SHADER CREATE AN RGB SPLIT EFFECT (SUPERPOSED CHANNEL R, G, B WITH GIVEN OFFSET)
    
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

    rgb_split_inplace_c(surface_, offset_)


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cpdef object rgb_split_clean(object surface_, int offset_=10):
    """

    THIS SHADER CREATE AN RGB SPLIT EFFECT (SUPERPOSED CHANNEL R, G, B WITH GIVEN OFFSET)
    
    The final image has a different width and height since the offset value is removed 
    to keep only the overlapping R, G, B channels 
    Setting the Offset_ to zero will have no effect to the original image.
    

    :param surface_ : pygame Surface to process (24bit format)
    :param offset_  : integer; offset for (x, y) to add to each channels RGB
    :return         : pygame Surface

    """
    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)

    assert isinstance(offset_, int), \
        "\nArgument offset_ must be an int type, got %s" % type(offset_)

    if offset_==0:
        return surface_

    return rgb_split_c(surface_, offset_)



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cpdef inline tuple ripple(
        int rows_,
        int cols_,
        float [:, ::1] previous_,
        float [:, ::1] current_,
        unsigned char [:, :, :] array_,
        float dispersion_ = 0.008
):
    """
    
    THIS SHADER CREATE A WATER EFFECT ON A PYGAME SURFACE
    This version does not include any background deformation to keep a reasonable FPS rate

    Check demo_ripple.py for a demonstration (Demo folder)

    :param rows_        : integer; Array width
    :param cols_        : integer; Array height
    :param previous_    : numpy.ndarray type (w, h) type float; array use for the transformation
    :param current_     : numpy.ndarray type (w, h) type float; array use for the transformation
    :param array_       : numpy.ndarray type (w, h, 3) type unsigned char
    :param dispersion_  : float; ripple dampening factor, higher values decrease the ripple effect radius
    :return             : tuple
    
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

    return ripple_c(rows_, cols_, previous_, current_, array_, dispersion_)


@cython.binding(True)
@cython.boundscheck(True)
@cython.wraparound(True)
@cython.nonecheck(True)
@cython.cdivision(True)
@cython.profile(True)
@cython.initializedcheck(True)
cpdef inline tuple ripple_seabed(
    int cols_, int rows_,
    float [:, ::1] previous_,                       # type numpy.float32 (w, h)
    float [:, ::1] current_,                        # type numpy.float32 (w, h)
    unsigned char [:, :, ::1] texture_array_,       # type numpy.ndarray (w, h, 3)
    unsigned char [:, :, :] background_array_,    # type numpy.ndarray (w, h, 3)
    float dispersion_ = 0.008
):
    """

    WATER SURFACE RIPPLE EFFECT AND SEABED DISTORTION


    :param rows_          : integer; Array width
    :param cols_          : integer; Array height
    :param previous_      : numpy.ndarray type (w, h) type float; array use for the transformation
    :param current_       : numpy.ndarray type (w, h) type float; array use for the transformation
    :param texture_array_ : numpy.ndarray type (w, h, 3) type unsigned char
    :param background_array_: numpy.ndarray type (w, h, 3) type unsigned char
    :param dispersion_    : float; ripple dampening factor.
    :return             : tuple

    """
    assert PyObject_IsInstance(previous_, (numpy.ndarray, cython.view.memoryview)), \
        "\nArgument previous must be a numpy.ndarray type got %s " % type(previous_)

    assert PyObject_IsInstance(current_, (numpy.ndarray, cython.view.memoryview)), \
        "\nArgument current must be a numpy.ndarray type got %s " % type(current_)

    assert PyObject_IsInstance(texture_array_, (numpy.ndarray, cython.view.memoryview)), \
        "\nArgument array must be a numpy.ndarray type got %s " % type(texture_array_)

    cdef Py_ssize_t prev_w, prev_h
    prev_w, prev_h = previous_.shape[:2]

    cdef Py_ssize_t curr_w, curr_h
    curr_w, curr_h = current_.shape[:2]

    cdef Py_ssize_t arr_w, arr_h
    arr_w, arr_h = texture_array_.shape[:2]

    assert prev_w == curr_w and prev_w == arr_w \
           and prev_h == curr_h and prev_h == arr_h, \
        "\n Array sizes mismatch (previous w: %s, h: %s; " \
        "current w: %s, h: %s; texture_array_ w: %s, h: %s " % (prev_w, prev_h, curr_w, curr_h,
        arr_w, arr_h)

    return ripple_seabed_c(
        rows_, cols_,
        previous_,
        current_,
        texture_array_,
        background_array_,
        dispersion_)



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cpdef inline void heatmap(object surface_, bint rgb_=True):
    """
    TRANSFORM AN IMAGE INTO A HEATMAP EQUIVALENT (INPLACE)

    :param surface_ : pygame.Surface
    :param rgb_     : boolean; True transformed the image into a RGB heatmap model of False (BGR)
    :return         : void
    """
    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)

    heatmap_convert(surface_, rgb_)




@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
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


    cp = surface_.copy()

    if fast:
        sobel_fast(cp, sobel_threshold, factor_=1)
    else:
        sobel(cp, sobel_threshold)

    bpf(surface_, bpf_threshold)
    shader_bloom_fast1(surface_, bloom_threshold)
    heatmap_convert(surface_, inv_colormap)
    surface_.blit(cp, (0, 0), special_flags=blend)

    return surface_.convert()



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cpdef inline void blood(object surface_, float [:, :] mask_, float perc_):
    """
    SHADER 2D GAME "HURT EFFECT" (INPLACE)
    
    This effect is used in 2D game when the player is being hurt
    THE MASK DETERMINE THE CONTOUR USED FOR THE BLOOD EFFECT.
    
    e.g
    # Outside the main loop 
    blood_surface = pygame.image.load("../Assets/redvignette.png").convert_alpha()
    blood_surface = pygame.transform.smoothscale(blood_surface, (WIDTH, HEIGHT))
    BLOOD_MASK = numpy.asarray(
    pygame.surfarray.pixels_alpha(blood_surface) / 255.0, numpy.float32)
    
    # In the main loop (percentage must change overtime)
    blood(BCK, BLOOD_MASK, percentage)

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
    cdef unsigned char [:, :, :] rgb_array
    try:
        rgb_array = surface_.get_view('3')

    except Exception as e:
        raise ValueError("Cannot reference source pixels into a 3d array.\n %s " % e)

    blood_inplace_c(rgb_array, mask_, perc_)


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cpdef inline object make_palette(int width, float fh, float fs, float fl):
    """
    
    CREATE A PALETTE OF MAPPED RGB COLORS VALUES
    
    
    h, s, l = color[i] * fh,  min(fs, 255.0), min(color[i] * fl, 255.0)
    e.g:
        # below: palette of 256 colors (256 colors).
        # hue * 6, saturation = 255.0, lightness * 2.0
        palette, surf = make_palette(256, 6, 255, 2)
        palette, surf = make_palette(256, 4, 255, 2)

    :param width  : integer, Palette width
    :param fh     : float, hue factor
    :param fs     : float, saturation value must be in range (0 ... 255)
    :param fl     : float, lightness factor
    :return       : Return a 1D array palette of mapped RGB values

    """

    return numpy.asarray(make_palette_c(width, fh, fs, fl))



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cpdef object palette_to_surface(unsigned int [::1] palette_c):
    """
    THIS METHOD RETURNS A PYGAME SURFACE RESEMBLING TO THE
    GIVEN PALETTE (palette_c IS DEFINE WITH make_palette)

    :param palette_c: numpy.ndarray containing the palette colors type unsigned char

    """

    cdef:
        int i= 0
        int s = len(<object>palette_c)
        rgb rgb_
        unsigned char [:, :, :] array_ = numpy.empty((s, 1, 3), dtype=numpy.uint8)

    with nogil:

        for i in prange(s, schedule = SCHEDULE, num_threads = THREADS):
            rgb_ = int_to_rgb_c(palette_c[i])
            array_[i,0,0] = <unsigned char>rgb_.r
            array_[i,0,1] = <unsigned char>rgb_.g
            array_[i,0,2] = <unsigned char>rgb_.b

    return make_surface(numpy.asarray(array_))



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
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



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
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
    CPU FIRE SHADER EFFECT

    e.g
    ==================================================================
    # Create a palette
    palette = make_palette(256, 0.1, 350, 1.2)

    # To check the palette color
    surf = palette_to_surface(palette)
    surf = pygame.transform.smoothscale(surf, (800, 1024))

    # Create the fire array
    fire_array = numpy.zeros((HEIGHT, WIDTH), dtype=numpy.float32)

    # In the main loop
    SCREEN.fill((0, 0, 0))
    surf = fire_effect(
        width_ = 800,
        height_ = 1024,
        factor_ = 3.95,
        palette_ = palette,
        fire_ = fire_array,
        low_ = 30,
        high_ = WIDTH - 30
    )
    SCREEN.blit(surf, (0, 0))
    ==================================================================

    * FIRE TEXTURE SIZES
    input width_  : integer (screen width)
    input height_ : integer (screen height)


    * FIRE ASPECT (CONTROL OVER THE WIDTH):
    inputs low_ : integer (width xmin)
    input high_ : integer (width xmax)

    * FIRE HEIGHT:
    
    input factor_ : float
    
    The fire maximum height can be adjust with the variable factor_ (float value)
    value > 3.95 will squash the effect
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
    <brightness_exclude_inplace_c> to increase the brightness of the effect / texture
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
    If the final fire texture is (w, h) after setting the transpose flag, the final 
    fire texture will become (h, w). As a result the fire effect will be transversal (starting 
    from the right of the display propagating to the left).
    You can always transpose / flip the texture to get the right flame orientation

    BORDER FLAME EFFECT 
    border_ = True to create a flame effect burning the edge of the display. This version is only
    compatible with symmetrical display or textures (same width & height).

    Transpose and border flame effect cannot be combined

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

    return fire_effect_c(
        width_, height_, factor_, palette_, fire_,
        reduce_factor_, fire_intensity_, smooth_,
        bloom_, fast_bloom_, bpf_threshold_, low_, high_, brightness_,
        brightness_intensity_, surface_, adjust_palette_,
        hsl_, transpose_, border_, blur_
    )


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
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
    <brightness_exclude_inplace_c> to increase the brightness of the effect / texture
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

    return cloud_effect_c(
        width_, height_, factor_, palette_, cloud_,
        reduce_factor_, cloud_intensity_, smooth_,
        bloom_, fast_bloom_, bpf_threshold_, low_, high_, brightness_,
        brightness_intensity_, surface_, transpose_, blur_
    )


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cpdef inline mirroring_array(object surface_):
    """
    
    MIRRORING

    This method create a mirror image 
    
    Compatible 24 - 32 bit image / surface
    
    :param surface_ : pygame.Surface; compatible 24 - 32 bit 
    :return         : returns a numpy ndarray shape (w, h, 3) 
    
    """
    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)
    cdef unsigned char [:, :, :] rgb_array
    try:
        rgb_array = surface_.get_view('3')

    except Exception as e:
        raise ValueError("Cannot reference source pixels into a 3d array.\n %s " % e)

    return mirroring_c(rgb_array)



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cpdef inline void mirroring(object surface_):
    """
    MIRRORING (INPLACE)

    This method create a mirror image 

    :param surface_ : pygame.Surface; compatible 24 - 32 bit 
    :return : void
    """
    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)
    cdef unsigned char [:, :, :] rgb_array
    try:
        rgb_array = surface_.get_view('3')

    except Exception as e:
        raise ValueError("Cannot reference source pixels into a 3d array.\n %s " % e)

    mirroring_inplace_c(rgb_array)




@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cpdef inline void sharpen(object surface_):
    """
    
    SHARPEN IMAGE APPLYING THE BELOW 3 X 3 KERNEL OVER EVERY PIXELS (INPLACE)

    :param surface_ : pygame.Surface; compatible 24 - 32 bit 
    :return         : void 
    """
    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)
    cdef unsigned char [:, :, :] rgb_array
    try:
        rgb_array = surface_.get_view('3')

    except Exception as e:
        raise ValueError("Cannot reference source pixels into a 3d array.\n %s " % e)

    sharpen_inplace_c(rgb_array)


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cpdef inline void dirt_lens(
        object surface_,
        object lens_model_,
        int flag_=BLEND_RGB_ADD,
        float light_ = 0.0
):
    """
    DIRT LENS EFFECT (INPLACE)
    
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
        light_ = -1.0

    assert PyObject_IsInstance(flag_, int), \
        "\nArgument flag_ must be a int type, got %s " % type(flag_)

    dirt_lens_c(surface_, lens_model_, flag_, light_)


# *******************************************************************

@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
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
@cython.profile(False)
@cython.initializedcheck(False)
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
@cython.profile(False)
@cython.initializedcheck(False)
cpdef object dithering(object surface_):

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
    :return        : Surface; 
    
    """
    assert PyObject_IsInstance(surface_, Surface), \
        'Argument surface_ must be a pygame.Surface got %s ' % type(surface_)

    try:
        rgb_array = pixels3d(surface_)

    except Exception as e:
        raise ValueError("Cannot reference source pixels into a 3d array.\n %s " % e)

    return dithering_c(numpy.asarray(rgb_array/<float>255.0, dtype=numpy.float32))


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cpdef inline void dithering_inplace(object surface_):
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
    :return        : Surface; 

    """
    assert PyObject_IsInstance(surface_, Surface), \
        'Argument surface_ must be a pygame.Surface got %s ' % type(surface_)

    cdef unsigned char [:, :, :] rgb_array
    try:
        rgb_array = surface_.get_view('3')

    except Exception as e:
        raise ValueError("Cannot reference source pixels into a 3d array.\n %s " % e)

    dithering_inplace_c(numpy.asarray(rgb_array, dtype=numpy.float32)/<float>255.0, rgb_array)

@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cpdef object dithering_atkinson(object surface_):
    """
    Dithering atkinson

    Take a pygame surface as argument format 24-32 bit and convert it to a 3d array format 
    (w, h, 3) type float (float32, single precision). 
    As the image is converted to a different data type format (uint8 to float32), 
    the transformation cannot be applied inplace. The image returned by the method dithering 
    is a copy of the original image.   

    :param surface_: Pygame surface format 24-32 bit 
    :return        : Surface; 

    """
    assert PyObject_IsInstance(surface_, Surface), \
        'Argument surface_ must be a pygame.Surface got %s ' % type(surface_)

    try:
        rgb_array = pixels3d(surface_)

    except Exception as e:
        raise ValueError("Cannot reference source pixels into a 3d array.\n %s " % e)

    return dithering_atkinson_c(numpy.asarray(rgb_array / <float> 255.0, dtype=numpy.float32))


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cpdef object pixelation(object surface_, unsigned int blocksize_=64):
    """
    Pixelate a pygame.Surface 
    
    Return a new pixelated surface
    Blocksize represent the square pixel size (default is 64, 64x64 pixel block).
    
    :param surface_: pygame.Surface; 
    :param blocksize_: unsigned int; block size used for the pixelation process, default is 64
    :return: pixelated surface
    """

    assert PyObject_IsInstance(surface_, Surface), \
        'Argument surface_ must be a pygame.Surface got %s ' % type(surface_)

    assert blocksize_ > 4, 'Invalid block size, blocksize must be > 4  got %s ' % blocksize_

    cdef Py_ssize_t w, h
    w, h = surface_.get_size()
    cdef object small = smoothscale(surface_, (blocksize_, blocksize_))
    return scale(small, (w, h))



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cpdef object blend(object source_, object destination_, float percentage_):
    """
    BLEND A SOURCE TEXTURE TOWARD A DESTINATION TEXTURE 
    
    The shader create a new image from both source_ and destination_

    * Video system must be initialised 
    * source_ & destination_ Textures must be same sizes
    * Compatible with 24 - 32 bit surface
    * Output create a new surface


    :param source_     : pygame.Surface (Source)
    :param destination_: pygame.Surface (Destination)
    :param percentage_ : float; Percentage value between [0.0 ... 100.0]
    :return: return    : Return a 24 bit pygame.Surface and blended with a percentage
                         of the destination texture.
    """
    assert PyObject_IsInstance(source_, Surface), \
        'Argument source_ must be a pygame.Surface got %s ' % type(source_)

    assert PyObject_IsInstance(destination_, numpy.ndarray), \
        'Argument destination_ must be a numpy.ndarray got %s ' % type(destination_)

    assert 0.0 <= percentage_ <= 100.0, \
        "\nIncorrect value for argument percentage should be [0.0 ... 100.0] got %s " % percentage_

    if percentage_ == 0.0:
        return source_

    assert source_.get_size() == destination_.shape[:2], \
        'Source and Destination surfaces must have same dimensions: ' \
        'Source (w:%s, h:%s), destination (w:%s, h:%s).' % \
        (*source_.get_size(), *destination_.shape[:2])

    return blending(source_, destination_, percentage_)



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cpdef void blend_inplace(
        object source_,
        object destination_,
        float percentage_
        ):
    """
    BLEND A SOURCE TEXTURE TOWARD A DESTINATION TEXTURE 

    The shader create a new image from both source_ and destination_

    * Video system must be initialised 
    * source_ & destination_ Textures must be same sizes
    * Compatible with 24 - 32 bit surface
    * Output create a new surface

    :param source_     : pygame.Surface (Source)
    :param destination_: 3d array (Destination)
    :param percentage_ : float; Percentage value between [0.0 ... 100.0]
    :return: return    : Return a 24 bit pygame.Surface and blended with a percentage
                         of the destination texture.
    """
    assert PyObject_IsInstance(source_, Surface), \
        'Argument source_ must be a pygame.Surface got %s ' % type(source_)

    assert PyObject_IsInstance(destination_, numpy.ndarray), \
        'Argument destination_ must be a numpy.ndarray got %s ' % type(destination_)

    assert 0.0 <= percentage_ <= 100.0, \
        "\nIncorrect value for argument percentage should be [0.0 ... 100.0] got %s " % percentage_

    assert source_.get_size() == destination_.shape[ :2 ], \
        'Source and Destination surfaces must have same dimensions: ' \
        'Source (w:%s, h:%s), destination (w:%s, h:%s).' % \
        (*source_.get_size(), *destination_.shape[ :2 ])

    blend_inplace_c(source_, destination_, percentage_)


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cpdef cartoon(
        object surface_,
        unsigned int sobel_threshold_ = 128,
        unsigned int median_kernel_ = 2,
        unsigned int color_  = 8,
        unsigned int flag_  = BLEND_RGB_ADD
):
    """
    CREATE A CARTOON EFFECT FROM A GIVEN PYGAME SURFACE/IMAGE
    
    * Compatible with 24 - 32 bit image 
    
    :param surface_: pygame.Surface compatible 24 - 32 bit 
    :param sobel_threshold_: integer sobel threshold
    :param median_kernel_  : integer median kernel  
    :param color_          : integer; color reduction value (max color)
    :param flag_           : integer; Blend flag e.g (BLEND_RGB_ADD, BLEND_RGB_SUB, 
                             BLEND_RGB_MULT, BLEND_RGB_MAX, BLEND_RGB_MIN  
    :return                : Return a pygame Surface with the cartoon effect 
    """
    if median_kernel_ < 2:
        raise ValueError("\nKernel size median_kernel_ must be >=2")
    if not (0 <= sobel_threshold_<=255):
        raise ValueError("\nSobel threshold sobel_threshold_ must be in range 0...255")

    return cartoon_c(surface_, sobel_threshold_, median_kernel_, color_, flag_)



@cython.profile(False)
@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
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


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cpdef inline void convert_27colors(object surface_):

    """
    THIS ALGORITHM CONVERT AN IMAGE USING 27 COLORS ONLY

    :param surface_: numpy.ndarray; containing the pixels RGB. Array shape (w, h, 3)  
    :return: void 
    """
    assert isinstance(surface_, Surface), \
        'Argument surface_ must be a valid Surface, got %s ' % type(surface_)
    cdef unsigned char [:, :, :] array_
    try:
        array_ = surface_.get_view('3')

    except (pygame.error, ValueError):
        raise ValueError('\nTexture/image is not compatible.')

    convert_27colors_c(array_)



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cpdef object bilateral(object image, float sigma_s, float sigma_i, unsigned int kernel_size = 3):
    """
    BILATERAL FILTERING (CREATE A NEW SURFACE)

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
    
    surface = bilateral(surface, sigma_s = 16, sigma_i = 18)
    
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
    cdef unsigned char [:, :, :] array_
    try:
        array_ = image.get_view('3')

    except (pygame.error, ValueError):
        raise ValueError('\nTexture/image is not compatible.')


    return bilateral_c(array_, sigma_s, sigma_i, kernel_size)


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cpdef object emboss(object surface_, unsigned short int flag_=0):
    """
    EMBOSS A PYGAME SURFACE 
    
    :param surface_: pygame.Surface; compatible 24-32 bit
    :param flag_    : integer; special pygame flag such as BLEND_RGB_ADD, BLEND_RGB_MULT etc
    :return        : pygame.Surface; Emboss effect 
    """

    assert isinstance(surface_, Surface), \
        'Argument surface_ must be a valid Surface, got %s ' % type(surface_)

    cdef unsigned char [:, :, :] array_

    try:
        array_ = surface_.get_view('3')

    except (pygame.error, ValueError):
        raise ValueError('\nTexture/image is not compatible.')

    cdef object emb = emboss5x5_c(array_)

    if flag_ != 0:
        del array_
        surface_.blit(emb, (0, 0), special_flags=flag_)
        return surface_

    return emb




@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cpdef void palette_change(
        object surface_,
        object palette_,
        object tmp_v_
):
    """
    CHANGE AN IMAGE BY CHANGING THE COLOR PALETTE (INPLACE)
    
    LIST_PALETTES contains all the palettes available
    in PygameShader project.
    
    e.g: 
    from PygameShader.Palette import LIST_PALETTES
    print(LIST_PALETTES.keys())
   
    Temporary array to declare 
    tmp_v = numpy.ascontiguousarray(numpy.ndarray(
        (SURFACE.get_width()*SURFACE.get_height(),
        IRIDESCENTCRYSTAL.shape[0]), dtype=float32
    ))
    
    :param surface_: pygame.Surface; 
        
    :param palette_: numpy.ndarray containing the palette colors to use for
        substituting the image colors, array format (w, 3) of type float range (0.0 ... 255.0)
        e.g 
        from PygameShader import IRIDESCENTCRYSTAL
        
    :param tmp_v_ : numpy.ndarray (contiguous array) shape 
        (rgb_array_.shape[0] * rgb_array_.shape[1], len(palette_.shape[0])) of type float32
        Temporary array to increase performance (the array does not have to be redeclared every
        frames. 
        e.g 
        tmp_v = numpy.ascontiguousarray(numpy.ndarray(
            (SURFACE.get_width()*SURFACE.get_height(),
            IRIDESCENTCRYSTAL.shape[0]), dtype=float32
        ))
    :return: void
    """

    assert isinstance(surface_, Surface), \
        'Argument surface_ must be a valid pygame Surface, got %s ' % type(surface_)

    assert isinstance(palette_, numpy.ndarray), \
        'Argument palette_ must be a numpy.ndarray, got %s ' % type(palette_)

    assert isinstance(tmp_v_, numpy.ndarray), \
        'Argument tmp_v must be a numpy.ndarray, got %s ' % type(tmp_v_)

    cdef:
        unsigned char [:, :, :] array_
        float [:, :] palette = palette_
        float[ :, ::1 ] tmp_v = tmp_v_

    try:
        array_ = surface_.get_view('3')

    except (pygame.error, ValueError):
        raise ValueError('\nTexture/image is not compatible.')

    palette_change_c(array_, palette, tmp_v)




@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cpdef object bilinear(
    object surface_,
    tuple size_,
    fx=None,
    fy=None
    ):

    """
    BILINEAR FILTER (RESIZE IMAGE)

    Return a resized image using the bilinear filter algorithm
    This algorithm is 10 times faster than smoothscale when reducing surfaces but 7 times slower when 
    increasing surface sizes 
    
    :param surface_: pygame Surface
    :param size_ : tuple; (new_width, new_height)
    :param fx: float; new width (factor), the new width will be, current image width * fx
    :param fy:float; new height (factor), the new width will be, current image height * fy

    """

    assert isinstance(surface_, Surface), \
        'Argument surface_ must be a valid Surface, got %s ' % type(surface_)

    cdef unsigned char [:,:,:] rgb_array

    try:
        rgb_array = surface_.get_view('3')

    except (pygame.error, ValueError):
        raise ValueError('\nTexture/image is not compatible.')

    return bilinear_c(rgb_array, size_)



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cpdef tunnel_modeling24(int screen_width, int screen_height, object surface_):

    assert screen_width > 0, "Argument screen_width must be > 0"
    assert screen_height > 0, "Argument screen_height must be > 0"

    cdef:
        int length = screen_width * screen_height * 4
        int [:] distances = numpy.empty(length, int32)
        int [:] angles    = numpy.empty(length, int32)
        int [:] shades    = numpy.empty(length, int32)

    cdef:
        int s_width  = 512
        int s_height = 512

    surface = surface_.convert_alpha()
    surface = smoothscale(surface, (s_width, s_height))

    cdef:
        unsigned char [::1] scr_data = surface.get_buffer()
        float sqy, sqx
        int x, y, i = 0

    for y in range(0, screen_height * <unsigned short int>2):
        sqy = <float>pow(y - screen_height, <unsigned short int>2)
        for x in range(0, screen_width * <unsigned short int>2):
            sqx = <float>pow(x - screen_width, <unsigned short int>2)
            if (sqx + sqy) == 0:
                distances[i] = <unsigned short int>1
            else:
                distances[i] = <int>(<float>floor(
                    <float>32.0 * <float>s_height / <float>sqrt(sqx + sqy))) % s_height
            angles[i]    = <int>round_c(<float>s_width *
                                        <float>atan2(<float>y - <float>screen_height,
                                        <float>x - <float>screen_width) / (<float>M_PI))
            shades[i]    = <int>min(<float>sqrt(sqx + sqy)* <float>10.0, <unsigned char>255)
            i = i + <unsigned short int>1

    return distances, angles, shades, scr_data




@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cpdef tunnel_render24(int t,
                    int screen_width,
                    int screen_height,
                    int screen_w2,
                    int screen_h2,
                    int [::1] distances,
                    int [::1] angles,
                    int [::1] shades,
                    unsigned char [::1] scr_data,
                    unsigned char [::1] dest_array):
    cdef:
        unsigned int s_width  = 512
        unsigned int s_height = 512
        float timer = t * <float>1e-3
        unsigned int shiftx  = <unsigned int>floor(s_width * timer)
        unsigned int shifty  = <unsigned int>floor(s_height * <float>0.25 * timer)
        unsigned int centerx = <unsigned int>(screen_w2 +
                <float>floor((screen_w2 >> 1) * <float>sin(timer * <float>0.25)))
        unsigned int centery = \
            <unsigned int>(screen_h2 +
                           <float>floor((screen_h2 >> 1) * <float>sin(timer * <float>0.5)))
        unsigned int stride  = screen_width * <unsigned short int>2
        unsigned int dest_ofs = 0
        unsigned int pix_ofs = 0
        int u, v, x, y
        unsigned int shade, src_ofs

    with nogil:
        for y in prange(0,  screen_height,  schedule=SCHEDULE, num_threads=THREADS):
            for x in range(0, screen_width):

                src_ofs = y * stride + centerx + centery * stride + x
                dest_ofs = (y * <unsigned int> screen_height + x) << 2

                u = (distances[ src_ofs ] + shiftx) & 0xff
                v = (angles[ src_ofs ] + shifty) & 0xff

                shade = <int> (shades[ src_ofs ] * <float>ONE_255)

                pix_ofs = (u + (v << <unsigned short int> 9)) << <unsigned short int> 3

                dest_array[ dest_ofs ] = scr_data[ pix_ofs + <unsigned short int> 2 ] * shade
                dest_array[ dest_ofs + <unsigned short int> 1 ] = \
                    scr_data[ pix_ofs + <unsigned short int> 1 ] * shade
                dest_array[ dest_ofs + <unsigned short int> 2 ] = \
                    scr_data[ pix_ofs + <unsigned short int> 0 ] * shade
                # dest_array[ dest_ofs + <unsigned short int> 3 ] = <unsigned char> 255

    # !!! Convert improve the overall performances, do not remove (strip the alpha channel)
    return frombuffer(dest_array, (screen_width, screen_height), "RGBA").convert()



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cpdef tunnel_modeling32(Py_ssize_t screen_width, Py_ssize_t screen_height, object surface_):
    """
    THIS METHOD CREATE A TUNNEL MODEL

    * This method must be called before rendering the tunnel in order to create
      all the necessary buffers that will be called during the rendering of the tunnel effect.
      tunnel_modeling32 must be call once only before the main loop of your game.

    * Cython cpdef function, this function can be called directly and do not require a
      hook function.

    * This algorithm uses a 256x256 texture but reshape it to 512x512 pixels for a
    better effect definition

    :param surface_: Pygame Surface to pass for the tunnel effect (surface 256x256)
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

    surface = surface_.convert_alpha()

    cdef int s_width  = 512
    cdef int s_height = 512
    surface = smoothscale(surface, (s_width, s_height))
    cdef unsigned char [::1] scr_data = surface.get_buffer()
    cdef float sqy, sqx
    cdef int x, y, i = 0


    for y in range(0, screen_height * <unsigned short int>2):
        sqy = <float>pow(y - screen_height, <unsigned short int>2)
        for x in range(0, screen_width * <unsigned short int>2):
            sqx = <float>pow(x - screen_width, <unsigned short int>2)
            if (sqx + sqy) == 0:
                distances[i] = <unsigned short int>1
            else:
                distances[i] = <int>(<float>floor(
                    <float>32.0 * <float>s_height / <float>sqrt(sqx + sqy))) % s_height
            angles[i]    = <int>round_c(<float>s_width *
                                        <float>atan2(<float>y - <float>screen_height,
                                        <float>x - <float>screen_width) / (<float>M_PI))
            shades[i]    = <int>min(<float>sqrt(sqx + sqy)* <float>10.0, <unsigned char>255)
            i = i + <unsigned short int>1

    return distances, angles, shades, scr_data




@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
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
        int stride  = <int>screen_width * <unsigned short int>2
        int dest_ofs = 0
        int src_ofs
        int u, v, x, y
        int pix_ofs, shade


    with nogil:

        for y in prange(0, screen_height, schedule=SCHEDULE, num_threads=4):

            for x in range(0, screen_width):

                src_ofs = y * stride + centerx + centery * stride + x
                dest_ofs = (y * <int>screen_height + x) << 2

                u = (distances[src_ofs] + shiftx) & 0xff
                v = (angles[src_ofs] + shifty) & 0xff

                shade = <int> (shades[src_ofs] * <float>ONE_255)

                pix_ofs = (u + (v << <unsigned short int>9)) << <unsigned short int>3

                dest_array[dest_ofs] = scr_data[pix_ofs + <unsigned short int>2] * shade
                dest_array[dest_ofs + <unsigned short int>1] = \
                    scr_data[pix_ofs + <unsigned short int>1] * shade
                dest_array[dest_ofs + <unsigned short int>2] =\
                    scr_data[pix_ofs + <unsigned short int>0] * shade
                dest_array[dest_ofs + <unsigned short int>3] = <unsigned char>255

    return pygame.image.frombuffer(dest_array,
                                   (screen_width, screen_height), "RGBA").convert_alpha()





# ******************************************************************



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cdef inline float damped_oscillation(float t)nogil:
    """
    
    :param t: float (variable x use for the oscillation
    :return: float (value y = f(x))
    """
    return <float>(exp(-t * <float>0.1) * <float>cos(M_PI * t))


cdef float C1_ = <float>1.0 / <float>sqrt(M_2PI)


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cdef inline float gauss(float x, float c, float sigma=1.0, float mu=0.0)nogil:
    """
    Gauss function

    check https://en.wikipedia.org/wiki/Gaussian_function for more details

    :param x: 
    :param c: 
    :param sigma: 
    :param mu: 
    :return: 
    """
    x -= c
    return <float>((<float>1.0 / sigma * C1_) * exp(-<float>0.5 *
    ((x - mu) * (x - mu)) / (sigma * sigma)))


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cdef inline void rgb_to_bgr_inplace_c(unsigned char [:, :, :] rgb_array):
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

        for j in prange(h, schedule=SCHEDULE, num_threads=THREADS):
            for i in range(w):
                tmp = rgb_array[i, j, <unsigned short int>0]  # keep the blue color
                rgb_array[i, j, <unsigned short int>0]  = rgb_array[i, j, <unsigned short int>2]
                rgb_array[i, j, <unsigned short int>2]  = tmp


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cdef inline void rgb_to_brg_inplace_c(unsigned char [:, :, :] rgb_array):
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

    with nogil:

        for j in prange(h, schedule=SCHEDULE, num_threads=THREADS):
            for i in range(w):
                tmp_r = rgb_array[i, j, 0]  # keep the red color
                tmp_g = rgb_array[i, j, 1]  # keep the green color
                rgb_array[i, j, <unsigned short int>0] = \
                    rgb_array[i, j, <unsigned short int>2] # r-->b
                rgb_array[i, j, <unsigned short int>1] = tmp_r  # g --> r
                rgb_array[i, j, <unsigned short int>2] = tmp_g



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cdef inline void greyscale_luminosity24_inplace_c(unsigned char [:, :, :] rgb_array):
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
        for j in prange(h, schedule=SCHEDULE, num_threads=THREADS):
            for i in range(w):
                r = &rgb_array[i, j, <unsigned short int>0]
                g = &rgb_array[i, j, <unsigned short int>1]
                b = &rgb_array[i, j, <unsigned short int>2]
                luminosity = <unsigned char>(r[0] * <float>0.2126 + g[0] * <float>0.7152 + b[0] * <float>0.072)
                r[0] = luminosity
                g[0] = luminosity
                b[0] = luminosity


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cdef inline void sepia_inplace_c(unsigned char [:, :, :] rgb_array):

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
        for j in prange(h, schedule=SCHEDULE, num_threads=THREADS):
            for i in range(w):

                r = &rgb_array[i, j, <unsigned short int>0]
                g = &rgb_array[i, j, <unsigned short int>1]
                b = &rgb_array[i, j, <unsigned short int>2]

                rr = r[0] * <float>0.393 + g[0] * <float>0.769 + b[0] * <float>0.189
                gg = r[0] * <float>0.349 + g[0] * <float>0.686 + b[0] * <float>0.168
                bb = r[0] * <float>0.272 + g[0] * <float>0.534 + b[0] * <float>0.131

                r[0] = <unsigned char> rr if rr<255 else 255
                g[0] = <unsigned char> gg if gg<255 else 255
                b[0] = <unsigned char> bb if bb<255 else 255

# ************* SORTING ALGORITHM FOR MEDIAN FILTER
@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
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


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cdef inline void insertion_sort(unsigned char [::1] nums, int size)nogil:
    """
    
    :param nums: 
    :param size: 
    :return: 
    """

    cdef:
        int i, j
        unsigned char item_to_insert

    for i in prange(1, size, schedule=SCHEDULE, num_threads=THREADS):
        item_to_insert = nums[i]

        j = i - 1
        while j >= 0 and nums[j] > item_to_insert:
            nums[j + 1] = nums[j]
            j = j - 1
        # Insert the item
        nums[j + 1] = item_to_insert



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
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


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
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

@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
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

    if left_child < heap_size and nums[left_child] < nums[left_child]:
        largest = left_child

    if right_child < heap_size and nums[largest] < nums[right_child]:
        largest = right_child

    if largest != root_index:
        nums[root_index], nums[largest] = nums[largest], nums[root_index]
        heapify(nums, heap_size, largest)




@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cdef inline void heap_sort(unsigned char [::1] nums, int n)nogil:
    """

    :param nums:
    :param n:
    :return:
    """
    cdef int i

    for i in range(n>>1, -1, -1):
        heapify(nums, n, i)

    for i in range(n - 1, 0, -1):
        nums[i], nums[0] = nums[0], nums[i]
        heapify(nums, i, 0)





# *********** END OF SORTING ALGORITHM

@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cdef inline void median_inplace_heapsort_c(
        unsigned char [:, :, :] rgb_array_, int kernel_size_=2):

    cdef Py_ssize_t w, h
    w, h = rgb_array_.shape[:2]

    cdef int k = kernel_size_ >> 1
    cdef int v = 0, kx, ky


    for kx in range(-k, k):
       for ky in range(-k, k):
           v += 1

    cdef:
        # todo change to unsigned char [::1, :, :] rgb_array_copy = numpy.array(rgb_array_,
        #  copy=False, order='F')
        unsigned char [:, :, ::1] rgb_array_copy = \
           ascontiguousarray(numpy.array(rgb_array_, copy=False))

        int i=0, j=0
        Py_ssize_t ii=0, jj=0

        unsigned char [::1] tmp_red   = empty(v, numpy.uint8, order='C')
        unsigned char [::1] tmp_green = empty(v, numpy.uint8, order='C')
        unsigned char [::1] tmp_blue  = empty(v, numpy.uint8, order='C')


        int index = 0, val
        Py_ssize_t w_1 = w - 1, h_1 = h - 1


    with nogil:
       for j in prange(h, schedule=SCHEDULE, num_threads=THREADS, chunksize=2048):
           for i in range(w):

               index = 0

               for kx in range(-k, k):
                   for ky in range(-k, k):

                       ii = i + kx
                       jj = j + ky

                       if ii < 0:
                           ii = 0
                       elif ii >= w_1:
                           ii = <int>w_1

                       if jj < 0:
                           jj = 0
                       elif jj >= h_1:
                           jj = <int>h_1


                       tmp_red[index]   = rgb_array_copy[ii, jj, 0]
                       tmp_green[index] = rgb_array_copy[ii, jj, 1]
                       tmp_blue[index]  = rgb_array_copy[ii, jj, 2]

                       index = index + 1

               # External C quicksort
               heap_sort(tmp_red, v)
               heap_sort(tmp_green,v)
               heap_sort(tmp_blue, v)

               val = (index - 1) >> 1
               rgb_array_[i, j, 0] = tmp_red[val]
               rgb_array_[i, j, 1] = tmp_green[val]
               rgb_array_[i, j, 2] = tmp_blue[val]




@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cdef inline void median_inplace_c(
        unsigned char [:, :, :] rgb_array_, int kernel_size_=2):

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

    cdef int k = kernel_size_ >> 1
    cdef int v = 0, kx, ky


    for kx in range(-k, k):
        for ky in range(-k, k):
            v += 1

    cdef:
        # unsigned char [:, :, ::1] rgb_array_copy = \
        #     ascontiguousarray(numpy.array(rgb_array_, copy=False))
        unsigned char [::1, :, :] rgb_array_copy = \
                    numpy.array(rgb_array_, copy=False, order='F')

        int i=0, j=0
        Py_ssize_t ii=0, jj=0

        unsigned char *tmp_red   = <unsigned char *> malloc(v * sizeof(unsigned char))
        unsigned char *tmp_green = <unsigned char *> malloc(v * sizeof(unsigned char))
        unsigned char *tmp_blue  = <unsigned char *> malloc(v * sizeof(unsigned char))

        unsigned char [::1] tmp_red_ = <unsigned char[:v]>tmp_red
        unsigned char [::1] tmp_green_ = <unsigned char[:v]>tmp_green
        unsigned char [::1] tmp_blue_ = <unsigned char[:v]>tmp_blue

        unsigned char *tmpr
        unsigned char *tmpg
        unsigned char *tmpb

        int index = 0, val
        Py_ssize_t w_1 = w - 1, h_1 = h - 1


    with nogil:
        for i in prange(w, schedule=SCHEDULE, num_threads=THREADS, chunksize=2048):
            for j in range(h):

                index = 0

                for kx in range(-k, k):
                    ii = i + kx
                    if ii < 0:
                        ii = 0
                    elif ii >= w_1:
                        ii = <int> w_1

                    for ky in range(-k, k):
                        jj = j + ky
                        if jj < 0:
                            jj = 0
                        elif jj >= h_1:
                            jj = <int>h_1

                        tmp_red_[index]   = rgb_array_copy[ii, jj, 0]
                        tmp_green_[index] = rgb_array_copy[ii, jj, 1]
                        tmp_blue_[index]  = rgb_array_copy[ii, jj, 2]

                        index = index + 1

                # External C quicksort
                tmpr = new_quickSort(&tmp_red_[0], 0, v)
                tmpg = new_quickSort(&tmp_green_[0], 0, v)
                tmpb = new_quickSort(&tmp_blue_[0], 0, v)

                val = (index - 1) >> 1
                rgb_array_[i, j, 0] = tmpr[val]
                rgb_array_[i, j, 1] = tmpg[val]
                rgb_array_[i, j, 2] = tmpb[val]


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cdef inline void shader_median_grayscale_filter24_inplace_c(
        unsigned char [:, :, :] rgb_array_, int kernel_size_=2):

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

    cdef int k = kernel_size_ >> 1, ky, kx
    cdef unsigned char v

    v = 0
    for kx in range(-k, k):
        for ky in range(-k, k):
            v += 1

    cdef:
        unsigned char [:, :, ::1] rgb_array_copy = \
            ascontiguousarray(numpy.array(rgb_array_, copy=False))

        int i=0, j=0
        Py_ssize_t ii=0, jj=0

        unsigned char *tmp_   = <unsigned char *> malloc(v * sizeof(unsigned char))
        unsigned char *tmp
        int index = 0
        unsigned char val


    with nogil:
        for i in prange(0, w, schedule=SCHEDULE, num_threads=THREADS, chunksize=2048):
            for j in range(0, h):

                index = 0

                for kx in range(-k, k):
                    for ky in range(-k, k):

                        ii = i + kx
                        jj = j + ky

                        if ii < 0:
                            ii = 0
                        elif ii > w - 1:
                            ii = w - 1

                        if jj < 0:
                            jj = 0
                        elif jj > h - 1:
                            jj = h - 1

                        tmp_[index]   = rgb_array_copy[ii, jj, 0]

                        index = index + 1

                tmp = new_quickSort(tmp_, 0, v)

                val = (v >> 1) - 1
                rgb_array_[i, j, 0] = tmp[val]
                rgb_array_[i, j, 1] = tmp[val]
                rgb_array_[i, j, 2] = tmp[val]



cdef float ONE_255 = <float>1.0 / <float>255.0

@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cdef inline void color_reduction_inplace_c(
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
        float c1 = <float>color_number * <float>ONE_255

    with nogil:
        for y in prange(h, schedule=SCHEDULE, num_threads=THREADS):
            for x in range(0, w):

                r = &rgb_array[x, y, 0]
                g = &rgb_array[x, y, 1]
                b = &rgb_array[x, y, 2]

                r[0] = <unsigned char>(<int>(round_c(c1 * r[0]) * f))
                g[0] = <unsigned char>(<int>(round_c(c1 * g[0]) * f))
                b[0] = <unsigned char>(<int>(round_c(c1 * b[0]) * f))



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

@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cdef inline void sobel_inplace_c(unsigned char [:, :, :] rgb_array, float threshold=20.0):
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
        # unsigned char [:, :, :] source_array = numpy.array(rgb_array, copy=True)
        unsigned char [::1, :, :] source_array = numpy.array(rgb_array, copy=False, order='F')
        int x, y
        Py_ssize_t xx, yy
        float r_gx, r_gy
        unsigned char *gray
        unsigned char m
        float magnitude

    with nogil:

        for y in prange(h, schedule=SCHEDULE, num_threads=THREADS):

            for x in range(w):

                r_gx, r_gy = <float>0.0, <float>0.0

                for kernel_offset_x in range(-KERNEL_HALF, KERNEL_HALF + 1):
                    xx = x + kernel_offset_x
                    if xx > w_1:
                        xx = w_1
                    if xx < 0:
                        xx = <unsigned int>0

                    for kernel_offset_y in range(-KERNEL_HALF, KERNEL_HALF + 1):
                        yy = y + kernel_offset_y
                        if yy > h_1:
                            yy = h_1
                        if yy < 0:
                            yy = <unsigned int>0

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



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
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
        for x in prange(w2, schedule=SCHEDULE, num_threads=THREADS):
            xx = <int>(x * fx)
            for y in range(h2):
                yy = <int>(y * fy)
                new_array[x, y, 0] = rgb_array[xx, yy, 0]
                new_array[x, y, 1] = rgb_array[xx, yy, 1]
                new_array[x, y, 2] = rgb_array[xx, yy, 2]

    return new_array


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cdef inline void sobel_fast_inplace_c(
        surface_, int threshold_=20, unsigned short factor_=1):
    """
    SHADER FAST SOBEL (EDGE DETECTION)

    Transform the game display or a pygame surface into a sobel equivalent model
    (surface edge detection).This version is slightly fastest than sobel_inplace_c as
    it down-scale the array containing all the pixels and apply the sobel algorithm to a smaller
    sample. When the processing is done, the array is re-scale to its original dimensions.
    If this method is in theory faster than sobel_inplace_c, down-scaling and up-scaling
    an array does have a side effect such as decreasing the overall image definition
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
        int w, h, w_1, h_1
    w, h = surface_.get_size()

    cdef:
        int kernel_offset_x, kernel_offset_y
        int x, y
        Py_ssize_t xx, yy
        float r_gx, r_gy
        unsigned char *gray
        unsigned char m
        float magnitude

        unsigned char [:, :, :] source_array = surface_.get_view('3')
        unsigned char [:, :, :] rescale_array = \
            numpy.array(pixels3d(scale(surface_, (w >> factor_, h >> factor_))))
        unsigned char [:, :, :] new_array = empty((w >> factor_, h >> factor_, 3), uint8)

    h = h >> factor_
    w = w >> factor_
    w_1 = w - 1
    h_1 = h - 1

    with nogil:

        for y in prange(h, schedule=SCHEDULE, num_threads=THREADS):

            for x in range(w):

                r_gx, r_gy = <float>0.0, <float>0.0

                for kernel_offset_y in range(-KERNEL_HALF, KERNEL_HALF + 1):
                    yy = y + kernel_offset_y
                    if yy > h_1:
                        yy = h_1
                    if yy < 0:
                        yy = 0

                    for kernel_offset_x in range(-KERNEL_HALF, KERNEL_HALF + 1):

                        xx = x + kernel_offset_x
                        if xx > w_1:
                            xx = w_1
                        if xx < 0:
                            xx = 0

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

        for y in prange(h, schedule=SCHEDULE, num_threads=THREADS):
            for x in range(w):

                source_array[x, y, 0] = new_[x, y, 0]
                source_array[x, y, 1] = new_[x, y, 1]
                source_array[x, y, 2] = new_[x, y, 2]


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cdef inline void invert_inplace_c(
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
        for j in prange(h, schedule=SCHEDULE, num_threads=THREADS):
            for i in range(w):
                r = &rgb_array[i, j, 0]
                g = &rgb_array[i, j, 1]
                b = &rgb_array[i, j, 2]
                r[0] = 255 - r[0]
                g[0] = 255 - g[0]
                b[0] = 255 - b[0]


cdef float[::1] GAUSS_KERNEL = numpy.array(
    [<float>(1.0/16.0), <float>(4.0/16.0),
     <float>(6.0/16.0), <float>(4.0/16.0), <float>(1.0/16.0)], dtype=numpy.float32)

@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cdef inline void blur_array_inplace_c(
        unsigned char [:, :, :] rgb_array_, mask=None, t=1):
    """
    APPLY A GAUSSIAN BLUR EFFECT TO THE GAME DISPLAY OR SURFACE (KERNEL 5x5)

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
        unsigned char [:, :, ::1] convolve = \
            numpy.empty((w, h, 3), dtype=uint8)
        int x, y, xx, yy
        float r, g, b, s
        char kernel_offset
        unsigned char red, green, blue
        float *k

    for r in range(t):
        with nogil:

            # horizontal convolution
            for x in prange(0, w, schedule=SCHEDULE, num_threads=THREADS):


                for y in prange(0, h):  # range [0..w-1]

                    r, g, b = <unsigned char> 0, <unsigned char> 0, <unsigned char> 0

                    for kernel_offset in range(-kernel_half, kernel_half + 1):

                        k = &GAUSS_KERNEL[kernel_offset + kernel_half]

                        xx = x + kernel_offset

                        # check boundaries.
                        # Fetch the edge pixel for the convolution
                        if xx < 0:
                            continue

                        if xx > w - 1:
                            continue

                        red, green, blue = rgb_array_[xx, y, 0],\
                            rgb_array_[xx, y, 1], rgb_array_[xx, y, 2]
                        if red + green + blue == <unsigned short int>0:
                            continue

                        r = r + red * k[0]
                        g = g + green * k[0]
                        b = b + blue * k[0]

                    convolve[x, y, 0] = <unsigned char>r
                    convolve[x, y, 1] = <unsigned char>g
                    convolve[x, y, 2] = <unsigned char>b

            # Vertical convolution
            for x in prange(0,  w, schedule=SCHEDULE, num_threads=THREADS):

                for y in prange(0, h):

                    r, g, b = <unsigned char> 0, <unsigned char> 0, <unsigned char> 0
                    for kernel_offset in range(-kernel_half, kernel_half + 1):

                        k = &GAUSS_KERNEL[kernel_offset + kernel_half]
                        yy = y + kernel_offset

                        if yy < 0:
                            continue

                        if yy > h-1:
                            continue

                        red, green, blue = convolve[x, yy, 0],\
                            convolve[x, yy, 1], convolve[x, yy, 2]
                        if red + green + blue == <unsigned short int>0:
                            continue

                        r = r + red * k[0]
                        g = g + green * k[0]
                        b = b + blue * k[0]

                    rgb_array_[x, y, 0], \
                    rgb_array_[x, y, 1],\
                    rgb_array_[x, y, 2] = \
                        <unsigned char>r, <unsigned char>g, <unsigned char>b


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cdef inline void wave_inplace_c(
        unsigned char [:, :, :] rgb_array_,
        float rad,
        int size):
    """
    CREATE A WAVE EFFECT TO THE GAME DISPLAY OR TO A GIVEN SURFACE

    The Array (rgb_array) must be a numpy array shape (w, h, 3) containing RGB pixels,
     please refer to pygame function pixels3d or array3d to convert an image into a
     3d array (library surfarray)


    :param rgb_array_   : numpy.ndarray shape (w, h, 3) containing all the RGB values
    :param rad          : float; angle in rad to rotate over time
    :param size         : int; Number of sub-surfaces
    :return             : void
    """

    cdef Py_ssize_t w, h
    w, h = rgb_array_.shape[:2]

    cdef:
        # unsigned char [:, :, ::1] rgb = \
        # numpy.ascontiguousarray(numpy.array(rgb_array_, copy=False, order='C'))
        unsigned char [::1, :, :] rgb = \
                numpy.array(rgb_array_, copy=False, order='F')
        int x, y, x_pos, y_pos, xx, yy
        unsigned int i=0, j=0
        float c1 = <float>1.0 / <float>(size * size)
        unsigned int w_1 = <unsigned int>w - 1
        unsigned int h_1 = <unsigned int>h - 1

    with nogil:

        for y in prange(0, h_1 - size, size, schedule=SCHEDULE, num_threads=THREADS):

            y_pos = y + size + <int>(<float>sin(rad + <float>y * c1) * <float>size)

            for x in prange(0, w_1 - size, size, schedule=SCHEDULE, num_threads=THREADS):

                x_pos = x + size + <int> (<float>sin(rad + <float> x * c1) * <float> size)

                for j in range(0, size + 1):
                    yy = y_pos + j
                    if yy > h_1:
                        # yy = h_1
                        continue
                    if yy < 0:
                        # yy = 0
                        continue

                    for i in range(0, size + 1):
                        xx = x_pos + i
                        if xx > w_1:
                            # xx = w_1
                            continue
                        if xx < 0:
                            # xx = 0
                            continue

                        rgb_array_[xx, yy, 0] = rgb[x + i, y + j, 0]
                        rgb_array_[xx, yy, 1] = rgb[x + i, y + j, 1]
                        rgb_array_[xx, yy, 2] = rgb[x + i, y + j, 2]



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cdef inline void wave_static_inplace_c(
        unsigned char [:, :, :] rgb_array_,
        unsigned char [:, :, :] array_,
        float rad,
        int size):
    """
    CREATE A WAVE EFFECT TO THE GAME DISPLAY OR TO A GIVEN SURFACE

    The Array (rgb_array) must be a numpy array shape (w, h, 3) containing RGB pixels,
     please refer to pygame function pixels3d or array3d to convert an image into a
     3d array (library surfarray)


    :param rgb_array_   : numpy.ndarray shape (w, h, 3) containing all the RGB values
    :param array_       : numpy.ndarray shape (w, h, 3) copy
    :param rad          : float; angle in rad to rotate over time
    :param size         : int; Number of sub-surfaces
    :return             : void
    """



    cdef Py_ssize_t w, h, ww, hh
    w, h = rgb_array_.shape[:2]
    ww, hh = array_.shape[:2]
    if w!=ww or h!=hh:
        raise ValueError("\nBoth the surface and the array must have the same sizes/dimensions")

    cdef:
        int x, y, x_pos, y_pos, xx, yy
        unsigned int i=0, j=0
        float c1 = <float>1.0 / <float>(size * size)
        int w_1 = <int>w - 1
        int h_1 = <int>h - 1

    with nogil:

        for y in prange(0, h_1 - size, size, schedule=SCHEDULE, num_threads=THREADS):

            y_pos = y + size + <int>(<float>sin(rad + <float>y * c1) * <float>size)

            for x in prange(0, w_1 - size, size, schedule=SCHEDULE, num_threads=THREADS):

                x_pos = x + size + <int> (<float>sin(rad + <float> x * c1) * <float> size)

                for j in range(0, size + 1):
                    yy = y_pos + j
                    if yy > h_1:
                        # yy = h_1
                        continue
                    if yy < 0:
                        # yy = 0
                        continue

                    for i in range(0, size + 1):
                        xx = x_pos + i
                        if xx > w_1:
                            # xx = w_1
                            continue
                        if xx < 0:
                            # xx = 0
                            continue

                        rgb_array_[xx, yy, 0] = array_[x + i, y + j, 0]
                        rgb_array_[xx, yy, 1] = array_[x + i, y + j, 1]
                        rgb_array_[xx, yy, 2] = array_[x + i, y + j, 2]

@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cdef inline void swirl_inplace_c(
        unsigned char [:, :, :] rgb_array_,
        float degrees
):
    """
    SWIRL AN IMAGE (ANGLE APPROXIMATION METHOD)

    This algorithm uses a table of cos and sin.

    The Array (rgb_array) must be a numpy array shape (w, h, 3) containing RGB pixels,
     please refer to pygame function pixels3d or array3d to convert an image into a
     3d array (library surfarray)

    :param rgb_array_   : numpy.ndarray shape (w, h, 3) containing all the RGB color values
    :param degrees      : float; swirl angle in degrees
    :return             : void
    """

    cdef Py_ssize_t w, h
    w, h = rgb_array_.shape[:2]

    cdef:
        int i, j, angle
        int diffx, diffy
        float columns, rows, r, di, dj
        float * c1
        float * c2
        # unsigned char [:, :, ::1] rgb = numpy.array(rgb_array_, copy=False, order='C')
        unsigned char [::1, :, :] rgb = numpy.array(rgb_array_, copy=False, order='F')
        float r_max


    columns = <float>0.5 * (<float>w - <float>1.0)
    rows    = <float>0.5 * (<float>h - <float>1.0)

    r_max = <float>sqrt(columns * columns + rows * rows)

    with nogil:
        for j in prange(h, schedule=SCHEDULE, num_threads=THREADS):
            dj = <float> j - rows
            for i in range(w):
                di = <float>i - columns

                r = <float>sqrt(di * di + dj * dj) / <float>r_max
                angle = <int>(degrees * r % 360)

                c1 = &COS_TABLE[angle]
                c2 = &SIN_TABLE[angle]
                diffx = <int>(di * c1[0] - dj * c2[0] + columns)
                diffy = <int>(di * c2[0] + dj * c1[0] + rows)

                if (diffx >-1) and (diffx < w) and \
                   (diffy >-1) and (diffy < h):
                    rgb_array_[i, j, 0], rgb_array_[i, j, 1],\
                        rgb_array_[i, j, 2] = rgb[diffx, diffy, 0], \
                                              rgb[diffx, diffy, 1], rgb[diffx, diffy, 2]
                else:
                    rgb_array_[ i, j, 0 ] = 0
                    rgb_array_[ i, j, 1 ] = 0
                    rgb_array_[ i, j, 2 ] = 0


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cdef inline void swirl_static_inplace_c(
        unsigned char [:, :, :] rgb_array_,
        unsigned char [:, :, :] array_,
        float degrees
):
    """
    SWIRL A STATIC BACKGROUND (ANGLE APPROXIMATION METHOD)

    This algorithm uses a table of cos and sin.

    The Array (rgb_array) must be a numpy array shape (w, h, 3) containing RGB pixels,
     please refer to pygame function pixels3d or array3d to convert an image into a
     3d array (library surfarray)
    array_ must have the exact same size/dimension


    :param rgb_array_   : numpy.ndarray shape (w, h, 3) containing all the RGB color values
    :param array_       : numpy.ndarray shape (w, h, 3) copy
    :param degrees      : float; swirl angle in degrees
    :return             : void
    """

    cdef Py_ssize_t w, h, ww, hh
    w, h = rgb_array_.shape[:2]
    ww, hh = array_.shape[:2]

    if w!=ww or h!=hh:
        raise ValueError("\nBoth surface and array_ must have the same sizes/dimensions.")

    cdef:
        int i, j, diffx, diffy, angle
        float columns, rows, r, di, dj
        float * c1
        float * c2
        float r_max

    columns = <float>0.5 * (<float>w - <float>1.0)
    rows    = <float>0.5 * (<float>h - <float>1.0)

    r_max = <float>sqrt(columns * columns + rows * rows)

    with nogil:
        for j in prange(h, schedule=SCHEDULE, num_threads=THREADS):
            dj = <float> j - rows
            for i in range(w):
                di = <float>i - columns

                r = <float>sqrt(di * di + dj * dj) / <float>r_max
                angle = <int>(degrees * r % 360)

                c1 = &COS_TABLE[angle]
                c2 = &SIN_TABLE[angle]
                diffx = <int>(di * c1[0] - dj * c2[0] + columns)
                diffy = <int>(di * c2[0] + dj * c1[0] + rows)

                if (diffx >-1) and (diffx < w) and \
                   (diffy >-1) and (diffy < h):
                    rgb_array_[i, j, 0], rgb_array_[i, j, 1],\
                        rgb_array_[i, j, 2] = array_[diffx, diffy, 0], \
                                              array_[diffx, diffy, 1], array_[diffx, diffy, 2]
                else:
                    rgb_array_[ i, j, 0 ] = 0
                    rgb_array_[ i, j, 1 ] = 0
                    rgb_array_[ i, j, 2 ] = 0



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cdef inline void swirl_inplace_c1(unsigned char [:, :, :] rgb_array_, float degrees):
    """
    SWIRL AN IMAGE WITHOUT ANGLE APPROXIMATION

    The Array (rgb_array) must be a numpy array shape (w, h, 3) containing RGB pixels,
     please refer to pygame function pixels3d or array3d to convert an image into a 3d
     array (library surfarray)


    :param rgb_array_   : numpy.ndarray shape (w, h, 3) containing all the RGB color values
    :param degrees      : float; swirl angle in degrees
    :return             : void
    """

    cdef Py_ssize_t w, h
    w, h = rgb_array_.shape[:2]

    cdef:
        int i, j, diffx, diffy
        float columns, rows, r, di, dj, c1, c2, angle
        # unsigned char [:, :, :] rgb = numpy.array(rgb_array_, copy=True)
        unsigned char [::1, :, :] rgb = numpy.array(rgb_array_, copy=False, order='F')
        float rad = degrees * DEG_TO_RAD
        float r_max


    columns = <float>0.5 * (w - <float>1.0)
    rows    = <float>0.5 * (h - <float>1.0)
    r_max   = <float>sqrt(columns * columns + rows * rows)

    with nogil:
        for j in prange(h, schedule=SCHEDULE, num_threads=THREADS):
            dj = <float> j - rows
            for i in range(w):
                di = <float>i - columns

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

                else:
                    rgb_array_[ i, j, 0 ] = 0
                    rgb_array_[ i, j, 1 ] = 0
                    rgb_array_[ i, j, 2 ] = 0


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cdef inline void swirl2_static_inplace_c1(
        unsigned char [:, :, :] rgb_array_,
        unsigned char [:, :, :] array_,
        float degrees):

    """
    SWIRL STATIC IMAGE/BACKGROUND WITHOUT ANGLE APPROXIMATION (INPLACE)

    The Array (rgb_array) must be a numpy array shape (w, h, 3) containing RGB pixels,
     please refer to pygame function pixels3d or array3d to convert an image into a 3d
     array (library surfarray)
    array_ must have the exact same size/dimension than rgb_array_

    :param rgb_array_   : numpy.ndarray shape (w, h, 3) containing all the RGB color values
    :param array_       : numpy.ndarray shape (w, h, 3) copy
    :param degrees      : float; swirl angle in degrees
    :return             : void
    """

    cdef Py_ssize_t w, h
    w, h = rgb_array_.shape[:2]

    cdef:
        int i, j, diffx, diffy
        float columns, rows, r, di, dj, c1, c2, angle
        float rad = degrees * DEG_TO_RAD
        float r_max


    columns = <float>0.5 * (w - <float>1.0)
    rows    = <float>0.5 * (h - <float>1.0)
    r_max   = <float>sqrt(columns * columns + rows * rows)

    with nogil:
        for j in prange(h, schedule=SCHEDULE, num_threads=THREADS):
            dj = <float> j - rows
            for i in range(w):
                di = <float>i - columns

                r = <float>sqrt(di * di + dj * dj)
                angle = <float>(rad * r/r_max)

                c1 = <float>cos(angle)
                c2 = <float>sin(angle)
                diffx = <int>(di * c1 - dj * c2 + columns)
                diffy = <int>(di * c2 + dj * c1 + rows)

                if (diffx >-1) and (diffx < w) and \
                   (diffy >-1) and (diffy < h):
                    rgb_array_[i, j, 0], rgb_array_[i, j, 1],\
                        rgb_array_[i, j, 2] = array_[diffx, diffy, 0], \
                                              array_[diffx, diffy, 1], array_[diffx, diffy, 2]

                else:
                    rgb_array_[ i, j, 0 ] = 0
                    rgb_array_[ i, j, 1 ] = 0
                    rgb_array_[ i, j, 2 ] = 0



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cdef inline void plasma_inplace_c(
        unsigned char [:, :, :] rgb_array_,
        int frame,
        float hue_=<float>1.0/<float>6.0,
        float sat_=<float>1.0/<float>6.0,
        float value_=<float>1.0/<float>8.0,
        float a_=<float>ONE_255,
        float b_=<float>ONE_TWELVE,
        float c_=<float>ONE_TWELVE):
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
        for x in prange(width, schedule=SCHEDULE, num_threads=THREADS):
            xx = <float> x * <float> 0.5
            for y in range(height):
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


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cdef inline void plasma_c(surface_, float frame, unsigned int [::1] palette_):

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

    cdef unsigned char [:, :, :] rgb_array_
    try:
        rgb_array_ = surface_.get_view('3')

    except Exception as e:
        raise ValueError("Cannot reference source pixels into a 3d array.\n %s " % e)

    cdef:
        int x, y, ii,c

        unsigned char *rr
        unsigned char *gg
        unsigned char *bb
        float color_
        float w2 = <float>width * <float>HALF
        float h2 = <float>height * <float>HALF
        Py_ssize_t length = len(palette_)

    with nogil:

        for y in prange(height, schedule=SCHEDULE, num_threads=THREADS):
            for x in range(width):

                rr = &rgb_array_[x, y, 0]
                gg = &rgb_array_[x, y, 1]
                bb = &rgb_array_[x, y, 2]

                color_ = \
                    <float>128.0 + \
                    <float>128.0 * <float>sin(x * <float>ONE_255 + frame) \
                    + <float>128.0 + \
                    <float>128.0 * <float>sin(frame * <float>ONE_32 + y * <float>ONE_64) \
                    + <float>128.0 \
                    + <float>128.0 * <float>sin(
                        sqrt((x - w2) * (x - w2) + (y - h2) * (y - h2)) * <float>ONE_255) \
                    + <float>128.0 + <float>128.0 * <float>sin(
                    <float>sqrt(x * x + y * y + frame) * <float>ONE_64)

                c = min(<int>(color_ / <float>8.0), <int>length)

                ii = palette_[c]

                rr[0] = (ii >> 16) & 255
                gg[0] = (ii >> 8) & 255
                bb[0] = ii & 255


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
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
        for r in prange(0, 256, schedule=SCHEDULE, num_threads=THREADS):
            for g in range(0, 256):
                for b in range(0, 256):
                    hsl_ = struct_rgb_to_hsl(
                        r * <float>ONE_255, g * <float>ONE_255, b * <float>ONE_255)
                    rgb_to_hsl[r, g, b, 0] = min(<float>(hsl_.h * <float>255.0), <float>255.0)
                    rgb_to_hsl[r, g, b, 1] = min(<float>(hsl_.s * <float>255.0), <float>255.0)
                    rgb_to_hsl[r, g, b, 2] = min(<float>(hsl_.l * <float>255.0), <float>255.0)

    return asarray(rgb_to_hsl, dtype=float32)


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
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
        for h in prange(0, 256, schedule=SCHEDULE, num_threads=THREADS):
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


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cdef inline void hsl_inplace_c(unsigned char [:, :, :] rgb_array, float shift_):

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
        for j in prange(h, schedule=SCHEDULE, num_threads=THREADS):
            for i in range(w):
                r = &rgb_array[i, j, 0]
                g = &rgb_array[i, j, 1]
                b = &rgb_array[i, j, 2]
                hsl_ = struct_rgb_to_hsl(
                    r[0] * <float>ONE_255, g[0] *
                    <float>ONE_255, b[0] * <float>ONE_255)

                #h_ = min((hsl_.h + shift_), <float>1.0)
                #h_ = max(h_, <float>0.0)
                h_ = hsl_.h + shift_
                rgb_ = struct_hsl_to_rgb(h_, hsl_.s, hsl_.l)

                r[0] = <unsigned char>(rgb_.r * <float>255.0)
                g[0] = <unsigned char>(rgb_.g * <float>255.0)
                b[0] = <unsigned char>(rgb_.b * <float>255.0)


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cdef inline void hsv_inplace_c(unsigned char [:, :, :] rgb_array, float shift_):

    """
    ROTATE THE HUE OF THE GAME DISPLAY OR GIVEN TEXTURE

    The Array (rgb_array) must be a numpy array shape (w, h, 3) containing RGB pixels,
    please refer to pygame function pixels3d or array3d to convert an image into a
    3d array (library surfarray)

    e.g:
    hsv_effect(surface, 0.2)

    :param rgb_array    : numpy.ndarray of shape(w, h, 3) of unsigned char, rgb values
    :param shift_       : float; Hue value in range [-1.0 ... 1.0]
    :return             : void
    """

    cdef Py_ssize_t w, h
    w, h = rgb_array.shape[:2]

    cdef:
        int i=0, j=0
        hsv hsv_
        rgb rgb_
        float h_
        unsigned char *r
        unsigned char *g
        unsigned char *b

    with nogil:
        for j in prange(h, schedule=SCHEDULE, num_threads=THREADS):
            for i in range(w):

                r = &rgb_array[i, j, 0]
                g = &rgb_array[i, j, 1]
                b = &rgb_array[i, j, 2]

                hsv_ = struct_rgb_to_hsv(
                    r[0] * <float>ONE_255, g[0] *
                    <float>ONE_255, b[0] * <float>ONE_255)

                # h_ = min((hsv_.h + shift_), <float>1.0)
                # h_ = max(h_, <float>0.0)
                h_ = hsv_.h + shift_

                rgb_ = struct_hsv_to_rgb(h_, hsv_.s, hsv_.v)

                r[0] = <unsigned char>(rgb_.r * <float>255.0)
                g[0] = <unsigned char>(rgb_.g * <float>255.0)
                b[0] = <unsigned char>(rgb_.b * <float>255.0)



# todo check the method below

@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cdef inline void hsl_fast_inplace_c(
        unsigned char [:, :, :] rgb_array,
        float shift_,
        float [:, :, :, ::1] rgb_to_hsl_,
        unsigned char [:, :, :, ::1] hsl_to_rgb_):

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
        for j in prange(h, schedule=SCHEDULE, num_threads=THREADS):

            for i in range(w):

                r = &rgb_array[i, j, 0]
                g = &rgb_array[i, j, 1]
                b = &rgb_array[i, j, 2]

                h_ = rgb_to_hsl_[r[0], g[0], b[0], 0]

                h__ = <unsigned char> (<float>min((h_ * ONE_255 + shift_), <float>1.0) * <float>255.0)

                s__ = <unsigned char> rgb_to_hsl_[r[0], g[0], b[0], 1]
                l__ = <unsigned char> rgb_to_hsl_[r[0], g[0], b[0], 2]

                r[0] = (&hsl_to_rgb_[h__, s__, l__, 0])[0]
                g[0] = (&hsl_to_rgb_[h__, s__, l__, 1])[0]
                b[0] = (&hsl_to_rgb_[h__, s__, l__, 2])[0]



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cdef inline void brightness_inplace_c(
        unsigned char [:, :, :] rgb_array_, float shift_=0):
    """
    SHADER BRIGHTNESS (INPLACE)

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
        float l
        hsl hsl_
        rgb rgb_


    with nogil:
        for j in prange(height, schedule=SCHEDULE, num_threads=THREADS):
            for i in range(width):

                r, g, b = &rgb_array_[i, j, 0], &rgb_array_[i, j, 1], &rgb_array_[i, j, 2]

                hsl_ = struct_rgb_to_hsl(
                    r[0] * <float>ONE_255,
                    g[0] * <float>ONE_255,
                    b[0] * <float>ONE_255
                )# struct_rgb_to_hsl returns floats, range 0.0 ... 1.0

                # l = min((hsl_.l + shift_), <float>1.0)
                # l = max(l, <float>0.0)

                # compensate hsl_.l
                l = hsl_.l + shift_

                # force white pixel, we do not need to run
                # struct_hsl_to_rgb to convert hsl to rgb as we know that
                # the color will be white
                if (hsl_.l + shift_) > 1:
                    r[ 0 ] = <unsigned char> 255
                    g[ 0 ] = <unsigned char> 255
                    b[ 0 ] = <unsigned char> 255
                    continue

                # force black pixel, we do not need to run
                # struct_hsl_to_rgb to convert hsl to rgb as we know that
                # the color will be black
                if l < 0:
                    r[ 0 ] = <unsigned char> 0
                    g[ 0 ] = <unsigned char> 0
                    b[ 0 ] = <unsigned char> 0
                    continue

                rgb_ = struct_hsl_to_rgb(hsl_.h, hsl_.s, l)

                r[0] = <unsigned char> (rgb_.r * <float>255.0)
                g[0] = <unsigned char> (rgb_.g * <float>255.0)
                b[0] = <unsigned char> (rgb_.b * <float>255.0)



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cdef inline object brightness_c(
        unsigned char [:, :, :] rgb_array_, float shift_=0):
    """
    BRIGHTNESS

    This shader control the pygame display brightness level
    It uses two external functions coded in C, struct_rgb_to_hsl & struct_hsl_to_rgb

    The Array (rgb_array) must be a numpy array shape (w, h, 3) containing RGB pixels,
    please refer to pygame function pixels3d or array3d to convert an image into
    a 3d array (library surfarray)

    e.g:
    surface = brightness_c(pixels3d(surface), 0.2)

    :param rgb_array_: numpy ndarray shape (w, h, 3) containing RGB pixels values
    :param shift_    : float; values in range [-1.0 ... 1.0], 0 no change,
    -1 lowest brightness, +1 highest brightness
    :return          : pygame surface
    """

    cdef Py_ssize_t width, height
    width, height = rgb_array_.shape[:2]

    cdef:
        int i=0, j=0
        unsigned char *r
        unsigned char *g
        unsigned char *b
        float l
        hsl hsl_
        rgb rgb_
        unsigned char [:, :, ::1] array_tmp = \
            numpy.empty((height, width, 3), dtype=numpy.uint8, order='C')

    with nogil:
        for j in prange(height, schedule=SCHEDULE, num_threads=THREADS):
            for i in range(width):

                r, g, b = \
                    &rgb_array_[ i, j, 0 ],\
                    &rgb_array_[ i, j, 1 ],\
                    &rgb_array_[ i, j, 2 ]

                hsl_ = struct_rgb_to_hsl(
                    r[ 0 ] * <float> ONE_255,
                    g[ 0 ] * <float> ONE_255,
                    b[ 0 ] * <float> ONE_255
                )  # struct_rgb_to_hsl returns floats, range 0.0 ... 1.0

                l = min((hsl_.l + shift_), <float>1.0)
                l = max(l, <float>0.0)

                # Below does not works with dirst_lens

                # # compensate hsl_.l
                # l = hsl_.l + shift_
                #
                # # force white pixel, we do not need to run
                # # struct_hsl_to_rgb to convert hsl to rgb as we know that
                # # the color will be white
                # if (hsl_.l + shift_) > 1:
                #     r[ 0 ] = <unsigned char> 255
                #     g[ 0 ] = <unsigned char> 255
                #     b[ 0 ] = <unsigned char> 255
                #     continue
                #
                # # force black pixel, we do not need to run
                # # struct_hsl_to_rgb to convert hsl to rgb as we know that
                # # the color will be black
                # if l < 0:
                #     r[ 0 ] = <unsigned char> 0
                #     g[ 0 ] = <unsigned char> 0
                #     b[ 0 ] = <unsigned char> 0
                #     continue

                rgb_ = struct_hsl_to_rgb(hsl_.h, hsl_.s, l)

                array_tmp[j, i, 0] = <unsigned char> (rgb_.r * <float>255.0)
                array_tmp[j, i, 1] = <unsigned char> (rgb_.g * <float>255.0)
                array_tmp[j, i, 2] = <unsigned char> (rgb_.b * <float>255.0)

    return frombuffer(array_tmp, (width, height), "RGB")


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cdef inline void brightness_exclude_inplace_c(
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
        unsigned char * r
        unsigned char * g
        unsigned char * b
        float l, h, s
        hsl hsl_
        rgb rgb_
        unsigned char rr=color_[0], gg=color_[1], bb=color_[2]

    with nogil:
        for j in prange(height, schedule=SCHEDULE, num_threads=THREADS):
            for i in range(width):

                r, g, b =\
                    &rgb_array_[i, j, 0], \
                    &rgb_array_[i, j, 1], \
                    &rgb_array_[i, j, 2]

                if not ((r[0]==rr) and (g[0]==gg) and (b[0]==bb)):

                    hsl_ = struct_rgb_to_hsl(
                        r[0] * <float>ONE_255,
                        g[0] * <float>ONE_255,
                        b[0] * <float>ONE_255
                    )  # struct_rgb_to_hsl returns floats, range 0.0 ... 1.0

                    # l = min((hsl_.l + shift_), <float>1.0)
                    # l = max(l, <float>0.0)

                    # compensate hsl_.l
                    l = hsl_.l + shift_

                    # force white pixel, we do not need to run
                    # struct_hsl_to_rgb to convert hsl to rgb as we know that
                    # the color will be white
                    if (hsl_.l + shift_) > 1:
                        r[ 0 ] = <unsigned char> 255
                        g[ 0 ] = <unsigned char> 255
                        b[ 0 ] = <unsigned char> 255
                        continue

                    # force black pixel, we do not need to run
                    # struct_hsl_to_rgb to convert hsl to rgb as we know that
                    # the color will be black
                    if l < 0:
                        r[ 0 ] = <unsigned char> 0
                        g[ 0 ] = <unsigned char> 0
                        b[ 0 ] = <unsigned char> 0
                        continue

                    rgb_ = struct_hsl_to_rgb(hsl_.h, hsl_.s, l)

                    rgb_array_[i, j, 0] = <unsigned char> (rgb_.r * <float>255.0)
                    rgb_array_[i, j, 1] = <unsigned char> (rgb_.g * <float>255.0)
                    rgb_array_[i, j, 2] = <unsigned char> (rgb_.b * <float>255.0)



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cdef inline void brightness_bpf_c(
        unsigned char [:, :, :] rgb_array_,
        float shift_=0.0,
        unsigned char bpf_threshold_=64):
    """
    
    :param rgb_array_: numpy ndarray shape (w, h, 3) containing RGB pixels values
    :param shift_    : float; values in range [-1.0 ... 1.0], 0 no change,
    -1 lowest brightness, +1 highest brightness
    :param bpf_threshold_ : integer; Bright pass filter threshold value 
    :return          : void
    """


    cdef Py_ssize_t w, h
    w, h = rgb_array_.shape[:2]

    cdef:
        int i=0, j=0
        unsigned char * r
        unsigned char * g
        unsigned char * b
        float l
        hsl hsl_
        rgb rgb_

    with nogil:
        for j in prange(h, schedule=SCHEDULE, num_threads=THREADS):
            for i in range(w):

                r, g, b = \
                    &rgb_array_[i, j, 0],\
                    &rgb_array_[i, j, 1],\
                    &rgb_array_[i, j, 2]

                if r[0] + g[0] + b[0] < bpf_threshold_:
                    continue

                # divide RGB by 255.0
                hsl_ = struct_rgb_to_hsl(
                    r[0] * <float>ONE_255,
                    g[0] * <float>ONE_255,
                    b[0] * <float>ONE_255
                ) # struct_rgb_to_hsl returns floats, range 0.0 ... 1.0

                # compensate hsl_.l
                l = hsl_.l + shift_

                # force white pixel, we do not need to run
                # struct_hsl_to_rgb to convert hsl to rgb as we know that
                # the color will be white
                if (hsl_.l + shift_) > 1:
                    r[0] = <unsigned char> 255
                    g[0] = <unsigned char> 255
                    b[0] = <unsigned char> 255
                    continue

                # force black pixel, we do not need to run
                # struct_hsl_to_rgb to convert hsl to rgb as we know that
                # the color will be black
                if l < 0:
                    r[0] = <unsigned char>0
                    g[0] = <unsigned char>0
                    b[0] = <unsigned char>0
                    continue

                rgb_ = struct_hsl_to_rgb(hsl_.h, hsl_.s, l)

                rgb_array_[i, j, 0] = <unsigned char> (rgb_.r * <float>255.0 )
                rgb_array_[i, j, 1] = <unsigned char> (rgb_.g * <float>255.0 )
                rgb_array_[i, j, 2] = <unsigned char> (rgb_.b * <float>255.0 )



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cdef inline void brightness_inplace1_c(
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

        for i in prange(width, schedule=SCHEDULE, num_threads=THREADS):
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




@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cdef inline void saturation_inplace_c(
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
        for j in prange(height, schedule=SCHEDULE, num_threads=THREADS):
            for i in range(width):

                r, g, b = \
                    &rgb_array_[i, j, 0], \
                    &rgb_array_[i, j, 1], \
                    &rgb_array_[i, j, 2]

                hsl_ = struct_rgb_to_hsl(
                    <float>r[0] * <float>ONE_255,
                    <float>g[0] * <float>ONE_255,
                    <float>b[0] * <float>ONE_255
                )

                s = min((hsl_.s + shift_), <float>1.0)
                s = max(s, <float>0.0)

                rgb_ = struct_hsl_to_rgb(hsl_.h, s, hsl_.l)

                r[0] = <unsigned char>(rgb_.r * <float>255.0)
                g[0] = <unsigned char>(rgb_.g * <float>255.0)
                b[0] = <unsigned char>(rgb_.b * <float>255.0)





@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
# e.g
# heatwave_vertical(
#         surface_, numpy.full((w, h), 255, dtype=numpy.uint8),
#         b*random.uniform(55.0, 100), 0, sigma_=random.uniform(0.4, 1), mu_=b*2)
cdef inline void heatwave24_vertical_inplace_c(
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

        for x in prange(w, schedule=SCHEDULE, num_threads=THREADS):

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


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(False)
@cython.profile(False)
@cython.initializedcheck(False)
# e.g horizontal_glitch(surface, 0.5, 0.08, frame % 20)
cdef inline void horizontal_glitch_inplace_c(
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
        float rad = <float>(<float>3.14/<float>180.0)
        float angle = <float>0.0
        float angle1 = <float>0.0
        # unsigned char [:, :, :] rgb_array_copy = \
        #     numpy.array(rgb_array_, copy=True, dtype=
        unsigned char [::1, :, :] rgb_array_copy = \
            numpy.array(rgb_array_, copy=False, dtype=uint8, order='F')
        int ii=0

    with nogil:

        for j in range(h):

            for i in range(w):

                ii = (i + <int>(<float>cos(angle) * amplitude_))
                if ii > w - 1:
                    ii = w - 1
                if ii < 0:
                    ii = 0

                rgb_array_[i, j, 0] = rgb_array_copy[ii, j, 0]
                rgb_array_[i, j, 1] = rgb_array_copy[ii, j, 1]
                rgb_array_[i, j, 2] = rgb_array_copy[ii, j, 2]

            angle1 = angle1 + frequency_ * rad
            angle = angle + (rad1_ * rad + rand() % angle1 - rand() % angle1)

@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(False)
@cython.profile(False)
@cython.initializedcheck(False)
# e.g horizontal_glitch(surface, 0.5, 0.08, frame % 20)
cdef inline void horizontal_glitch_static_inplace_c(
        unsigned char [:, :, :] rgb_array_,
        unsigned char [:, :, :] array_,
        float rad1_,
        float frequency_,
        float amplitude_):

    """
    SHADER GLITCH EFFECT ON STATIC IMAGE/BACKGROUND (INPLACE)

    Deform the pygame display to create a glitch appearance

    The Array (rgb_array) must be a numpy array shape (w, h, 3) containing RGB pixels,
    please refer to pygame function pixels3d or array3d to convert an image into a
    3d array (library surfarray)

    :param rgb_array_: numpy.ndarray shape (w, h, 3) uint8 containing RGB pixels
    :param array_    : numpy.ndarray shape (w, h, 3) copy
    :param rad1_     : float; Angle in radians, this value control the angle variation over the time
    :param frequency_:  float; signal frequency, factor that amplify the angle variation
    :param amplitude_: float; cos amplitude value
    :return:
    """
    cdef Py_ssize_t w, h, ww, hh
    w, h = rgb_array_.shape[:2]
    ww, hh = array_.shape[:2]

    if w!=ww and h!=hh:
        raise ValueError("\nBoth surface and array_ must have the same sizes/dimensions")

    cdef:
        int i=0, j=0
        float rad = <float>(<float>3.14/<float>180.0)
        float angle = <float>0.0
        float angle1 = <float>0.0
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
                rgb_array_[i, j, 2] = array_[ii, j, 0],\
                    array_[ii, j, 1], array_[ii, j, 2]

            angle1 = angle1 + frequency_ * rad
            angle = angle + (rad1_ * rad + rand() % angle1 - rand() % angle1)




@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cdef inline void bpf24_inplace_c(
        unsigned char [:, :, :] rgb_array_, int threshold = 128):
    """
    SHADER BRIGHT PASS FILTER (INPLACE)

    Conserve only the brightest pixels in an array

    The Array (rgb_array) must be a numpy array shape (w, h, 3) containing RGB pixels,
     please refer to pygame function pixels3d or array3d to convert an image into a
     3d array (library surfarray)

    :param rgb_array_: numpy.ndarray shape (w, h, 3) uint8 containing RGB pixels
    :param threshold : integer; Bright pass threshold default 128
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
        for j in prange(0, h, schedule=SCHEDULE, num_threads=THREADS):
            for i in range(0, w):

                # ITU-R BT.601 luma coefficients
                r = &rgb_array_[i, j, 0]
                g = &rgb_array_[i, j, 1]
                b = &rgb_array_[i, j, 2]

                lum = r[0] * <float>0.299 + g[0] * <float>0.587 + b[0] * <float>0.114
                if lum < threshold:
                    r[ 0 ], g[ 0 ], b[ 0 ] = 0, 0, 0
                    continue

                c = (lum - threshold) / lum
                r[0] = <unsigned char>(r[0] * c)
                g[0] = <unsigned char>(g[0] * c)
                b[0] = <unsigned char>(b[0] * c)




@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cdef inline bpf24_c(
        unsigned char [:, :, :] input_array_,
        int threshold = 128,
        ):
    """
    SHADER BRIGHT PASS FILTER

    Conserve only the brightest pixels in an array

    The Array (rgb_array) must be a numpy array shape (w, h, 3) containing RGB pixels,
    please refer to pygame function pixels3d or array3d to convert an image into a
    3d array (library surfarray)

    :param input_array_: numpy.ndarray shape (w, h, 3) uint8 containing RGB pixels
    :param threshold   : float Bright pass threshold default 128
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
        for j in prange(0, h, schedule=SCHEDULE, num_threads=THREADS):
            for i in range(0, w):

                # ITU-R BT.601 luma coefficients
                r = &input_array_[i, j, 0]
                g = &input_array_[i, j, 1]
                b = &input_array_[i, j, 2]

                lum = r[0] * <float>0.299 + g[0] * <float>0.587 + b[0] * <float>0.114

                if lum < threshold:
                    continue

                c = (lum - threshold) / lum
                output_array_[j, i, 0] = <unsigned char>(r[0] * c)
                output_array_[j, i, 1] = <unsigned char>(g[0] * c)
                output_array_[j, i, 2] = <unsigned char>(b[0] * c)

    return frombuffer(output_array_, (w, h), 'RGB')





@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
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
        rgb_ = surface_.get_view('3')
    except (ValueError, pygame.error):
        raise ValueError('Incompatible surface.')

    cdef:
        unsigned char [:, :, :] rgb = rgb_.transpose(1, 0, 2)
        unsigned char [:, :, ::1] rgb1 = numpy.empty((h, w, 3), numpy.uint8)
        float [:, :] mask = numpy.asarray(mask_, numpy.float32)
        int i, j

    with nogil:

        for i in prange(0, w, schedule=SCHEDULE, num_threads=THREADS):
            for j in range(h):
                rgb1[j, i, 0] = <unsigned char>(rgb[j, i, 0] * mask[i, j])
                rgb1[j, i, 1] = <unsigned char>(rgb[j, i, 1] * mask[i, j])
                rgb1[j, i, 2] = <unsigned char>(rgb[j, i, 2] * mask[i, j])

    return frombuffer(rgb1, (w, h), 'RGB')



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cpdef inline void filtering_inplace_c(object surface_, mask_):
    """
    MULTIPLY MASK VALUES WITH AN ARRAY REPRESENTING
    THE SURFACE PIXELS (COMPATIBLE 24 BIT ONLY).

    Transformation is applied inplace

    Mask values are floats in range (0 ... 1.0)

    :param surface_: pygame.Surface compatible 24-bit
    :param mask_: 2d array (MemoryViewSlice) containing alpha values (float).
        The mask_ output image is monochromatic (values range [0 ... 1.0] and R=B=G.
    :return: void

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
        rgb_ = surface_.get_view('3')

    except (ValueError, pygame.error):
        raise ValueError('Incompatible surface.')

    cdef:
        unsigned char [:, :, :] rgb = rgb_
        float [:, :] mask = numpy.asarray(mask_, numpy.float32)
        int i, j

    with nogil:

        for i in prange(0, w, schedule=SCHEDULE, num_threads=THREADS):

            for j in range(h):

                rgb[i, j, 0] = <unsigned char>(rgb[i, j, 0] * mask[i, j])
                rgb[i, j, 1] = <unsigned char>(rgb[i, j, 1] * mask[i, j])
                rgb[i, j, 2] = <unsigned char>(rgb[i, j, 2] * mask[i, j])






@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cdef inline void bloom_array24_c(
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
        blur_array_inplace_c(s2_array)
        # b2_blurred = frombuffer(numpy.array(s2_array.transpose(1, 0, 2),
        # order='C', copy=False), (w2, h2), 'RGB')
        b2_blurred = make_surface(s2_array)
        s2 = smoothscale(b2_blurred, (w, h))
        surface_.blit(s2, (0, 0), special_flags=BLEND_RGB_ADD)

        # OTHER TECHNIQUE

        # Rescale the image (create a new surface)
        # s2_surf = scale(surface_, (w2, h2))
        # Create array referencing the pixels
        # s2_array = pixels3d(s2_surf)
        # Bright pass filter inplace
        # bpf24_inplace_c(s2_array, threshold=threshold_)
        # Bloom inplace
        # blur_array_inplace_c(s2_array)
        # create surface from array
        # array_to_surface(s2_surf, s2_array)
        # Re-scale the surface to original dim (w, h),
        # use bilinear filtering with smoothscale
        # s2 = smoothscale(s2_surf, (w, h))
        # Blend original image with bloom effect
        # blend_add_surface(surface_, s2)

    # SECOND SUBSURFACE DOWNSCALE x4
    # THIS IS THE SECOND MOST EXPENSIVE IN TERM OF PROCESSING TIME
    if x4:
        s4 = scale(surface_, (w4, h4))
        s4 = bpf24_c(pixels3d(s4), threshold=threshold_)
        s4_array = numpy.array(s4.get_view('3'), dtype=numpy.uint8)
        blur_array_inplace_c(s4_array)
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
        blur_array_inplace_c(s8_array)
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
        blur_array_inplace_c(s16_array)
        # b16_blurred = frombuffer(numpy.array(s16_array.transpose(1, 0, 2),
        # order='C', copy=False), (w16, h16), 'RGB')
        b16_blurred = make_surface(s16_array)
        s16 = smoothscale(b16_blurred, (w, h))
        surface_.blit(s16, (0, 0), special_flags=BLEND_RGB_ADD)

    if mask_ is not None:
        filtering_inplace_c(surface_, mask_)






@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
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

    cp = surface_.copy()


    surface_ = \
        smoothscale(
            surface_,
            (surface_.get_width() >> factor_,
             surface_.get_height() >> factor_)
        )

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
        blur_array_inplace_c(s2_array)
        b2_blurred = make_surface(s2_array)
        s2 = smoothscale(b2_blurred, (w, h))


    # SECOND SUBSURFACE DOWNSCALE x4
    # THIS IS THE SECOND MOST EXPENSIVE IN TERM OF PROCESSING TIME
    if x4:
        s4 = scale(surface_, (w4, h4))
        s4 = bpf24_c(pixels3d(s4), threshold=threshold_)
        s4_array = numpy.array(s4.get_view('3'), dtype=numpy.uint8)
        blur_array_inplace_c(s4_array)
        b4_blurred = make_surface(s4_array)
        s4 = smoothscale(b4_blurred, (w, h))


    # THIRD SUBSURFACE DOWNSCALE x8
    if x8:
        s8 = scale(surface_, (w8, h8))
        s8 = bpf24_c(pixels3d(s8), threshold=threshold_)
        s8_array = numpy.array(s8.get_view('3'), dtype=numpy.uint8)
        blur_array_inplace_c(s8_array)
        # order='C', copy=False), (w8, h8), 'RGB')
        b8_blurred = make_surface(s8_array)
        s8 = smoothscale(b8_blurred, (w, h))


    # FOURTH SUBSURFACE DOWNSCALE x16
    # LESS SIGNIFICANT IN TERMS OF RENDERING AND PROCESSING TIME
    if x16:
        s16 = scale(surface_, (w16, h16))
        s16 = bpf24_c(pixels3d(s16), threshold=threshold_)
        s16_array = numpy.array(s16.get_view('3'), dtype=numpy.uint8)
        blur_array_inplace_c(s16_array)
        blur_array_inplace_c(s16_array)
        b16_blurred = make_surface(s16_array)
        s16 = smoothscale(b16_blurred, (w, h))


    if fast_:
        s16 = smoothscale(s16, (w << factor_, h << factor_))
        cp.blit(s16, (0, 0), special_flags=BLEND_RGB_ADD)
    else:
        s2.blit(s4, (0, 0), special_flags=BLEND_RGB_ADD)
        s2.blit(s8, (0, 0), special_flags=BLEND_RGB_ADD)
        s2.blit(s16, (0, 0), special_flags=BLEND_RGB_ADD)
        s2 = smoothscale(s2, (w << factor_, h << factor_))
        cp.blit(s2, (0, 0), special_flags=BLEND_RGB_ADD)

    return cp



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
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
        int  w, h
        unsigned int bit_size
        unsigned int  w16, h16
        unsigned int r

    assert isinstance(surface_, pygame.Surface), \
        "Argument surface_ must be a pygame.Surface got %s " % type(surface_)
    if flag_ < 0:
        raise ValueError("Argument flag_ cannot be < 0")
    if smooth_ < 0:
        raise ValueError("Argument smooth_ cannot be < 0")

    threshold_ %= <unsigned char>255

    w, h = surface_.get_size()
    bit_size = surface_.get_bitsize()

    w16, h16 = w >> 4, h >> 4

    if w16 == 0 or h16 == 0:
        raise ValueError(
            "\nImage too small and cannot be processed.\n"
                "Try to increase the size of the image")


    s2 = smoothscale(surface_, (w16, h16))
    s2.blit(s2, (0, 0), special_flags=BLEND_RGB_ADD)
    # blend_add_surface_c(s2, s2)

    cdef unsigned char [ :, :, : ] s2_array = s2.get_view('3')
    bpf24_inplace_c(s2_array, threshold=threshold_)

    for r in range(smooth_):
        blur_array_inplace_c(s2_array)
        if saturation_ : saturation_inplace_c(s2_array, <float>0.3)

    pygame.surfarray.array_to_surface(s2, asarray(s2_array, dtype=uint8))
    s2 = smoothscale(s2, (w, h))

    if flag_ is not None and flag_!=0:
        surface_.blit(s2, (0, 0), special_flags=flag_)
    else:
        surface_.blit(s2, (0, 0))



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cdef inline fisheye_footprint_c(
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
    :return        : Return a numpy.ndarray type (w, h, 2) representing the
    fisheye model (coordinates of all surface pixels passing through the fisheye lens model)
    """

    assert w > 0, "Argument w must be > 0"
    assert h > 0, "Argument h must be > 0"

    cdef:
        unsigned int [:, :, :] image_fisheye_model = numpy.zeros((w, h, 3), numpy.uint32)
        int y, x
        float ny, ny2, nx, nx2, r, theta, nr
        float c1 = <float>2.0/w
        float c2 = <float>2.0/h
        float w2 = centre_x
        float h2 = centre_y

    with nogil:

        for x in prange(w, schedule=SCHEDULE, num_threads=THREADS):
            nx = x * c1 - <float>1.0
            nx2 = nx * nx

            for y in range(h):
                ny = y * c2 - <float>1.0
                ny2 = ny * ny

                r = <float>sqrt(nx2 + ny2)

                if r >1:
                    image_fisheye_model[x, y, 0] = <unsigned int>0
                    image_fisheye_model[x, y, 1] = <unsigned int>0
                    continue

                nr = (r + <float>1.0 - <float>sqrt(
                <float>1.0 - (nx2 + ny2))) * <float>0.5

                theta = <float>atan2(ny, nx)
                image_fisheye_model[x, y, 0] = <unsigned int> (nr * <float>cos(theta) * w2 + w2)
                image_fisheye_model[x, y, 1] = <unsigned int> (nr * <float>sin(theta) * h2 + h2)
    return image_fisheye_model

@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cdef inline void fisheye_inplace_c(
        unsigned char [:, :, :] rgb_array_,
        unsigned int [:, :, ::1] fisheye_model
):
    """
    THIS SHADER CAN BE USE TO DISPLAY THE GAME THROUGH A LENS EFFECT

    Display a fisheye effect in real time given a numpy ndarray referencing the
    pixels RGB of a Pygame.Surface. In order to accomplish a real time
    calculation, this algorithm is using a pre-calculated transformation stored
    in the array fisheye_model.
    The function fisheye_footprint_c has to be called prior
    fisheye_inplace_c in order to store the transformation values.

    This shader can be applied directly to the pygame display

    The Array (rgb_array) must be a numpy array shape (w, h, 3) containing RGB pixels,
     please refer to pygame
    function pixels3d or array3d to convert an image into a 3d array (library surfarray)

    :param rgb_array_       : numpy.ndarray shape (width, height, 3) containing RGB pixels
    :param fisheye_model    : numpy.ndarray shape (width, height, 2) int32, fisheye model
        containing the pixels coordinates after the fisheye transformation
    :return                 : void
    """

    cdef:
        Py_ssize_t w, h
    w, h = rgb_array_.shape[:2]

    cdef:
        int x, y
        # unsigned char [:, :, :] rgb_array_copy = \
        #     numpy.array(rgb_array_, copy=False, order='C')
        unsigned char [::1, :, :] rgb_array_copy = \
            numpy.array(rgb_array_, copy=False, order='F')

        unsigned int x2
        unsigned int y2
        unsigned int w_1 = w - 1
        unsigned int h_1 = h - 1

    with nogil:
        for x in prange(w, schedule=SCHEDULE, num_threads=THREADS):
            for y in range(h):

                x2 = min(fisheye_model[x, y, 0], w_1)
                y2 = min(fisheye_model[x, y, 1], h_1)

                if x2==0 and y2==0:
                    continue

                rgb_array_[x, y, 0] = rgb_array_copy[x2, y2, 0]
                rgb_array_[x, y, 1] = rgb_array_copy[x2, y2, 1]
                rgb_array_[x, y, 2] = rgb_array_copy[x2, y2, 2]


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cpdef inline flip_array_vertically(unsigned int [:, :, :] rgb_array_):

    cdef:
        Py_ssize_t w, h

    w, h = rgb_array_.shape[:2]

    cdef:
        int x, y
        unsigned int [:, :, :] new_array = numpy.empty((w, h, 3), dtype=numpy.uint32)
        unsigned int yy
    with nogil:
        for x in prange(w, schedule=SCHEDULE, num_threads=THREADS):
            for y in range(h):
                yy = h - 1 - y
                new_array[x, yy, 0] = rgb_array_[x, y, 0]
                new_array[x, yy, 1] = rgb_array_[x, y, 1]
                new_array[x, yy, 2] = rgb_array_[x, y, 2]

    return numpy.asarray(new_array)



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cdef inline void tv_scanline_inplace_c(
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
        for y in prange(0, h, frame_, schedule=SCHEDULE, num_threads=THREADS):
            for x in range(w):
                for j in range(frame_2):
                    if y + j < h - 1:
                        r = &rgb_array_[x, y + j, <unsigned short int>0]
                        g = &rgb_array_[x, y + j, <unsigned short int>1]
                        b = &rgb_array_[x, y + j, <unsigned short int>2]
                    else:
                        r = &rgb_array_[x, y, <unsigned short int>0]
                        g = &rgb_array_[x, y, <unsigned short int>1]
                        b = &rgb_array_[x, y, <unsigned short int>2]
                    r[0] = <unsigned char> (r[0] * <float>0.65)
                    g[0] = <unsigned char> (g[0] * <float>0.65)
                    b[0] = <unsigned char> (b[0] * <float>0.65)


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cdef inline void rgb_split_inplace_c(object surface_, int offset_):
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
        int z = <int>h * <int>w * <unsigned short int>3
        unsigned char [:] rgb   = numpy.frombuffer(surface_.get_buffer(), dtype=numpy.uint8)
        unsigned char [::1] red   = numpy.empty(z, uint8, order='C')
        unsigned char [::1] green = numpy.empty(z, uint8, order='C')
        unsigned char [::1] blue  = numpy.empty(z, uint8, order='C')


    with nogil:

        for i in prange(0, w * h * 4, 4,
                        schedule=SCHEDULE, num_threads=THREADS):#, chunksize=8):
            j = (i >> 2) * <unsigned short int>3
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
@cython.profile(False)
@cython.initializedcheck(False)
cdef rgb_split_c(object surface_, int offset_):
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
        int z = <int>h * <int>w * <unsigned short int>3
        unsigned char [::1] rgb   = numpy.frombuffer(surface_.get_buffer(), dtype=numpy.uint8)
        unsigned char [::1] red   = numpy.empty(z, uint8, order='C')
        unsigned char [::1] green = numpy.empty(z, uint8, order='C')
        unsigned char [::1] blue  = numpy.empty(z, uint8, order='C')

    # Create a new surface (sizes - offset)
    new_surface = Surface((w-offset_, h-offset_))


    with nogil:

        for i in prange(0, w * h * <unsigned short int>4, <unsigned short int>4,
                        schedule=SCHEDULE, num_threads=THREADS, chunksize=8):
            j = (i >> <unsigned short int>2) * <unsigned short int>3
            red[j]     = rgb[i + <unsigned short int>2]
            green[j + <unsigned short int>1] = rgb[i + <unsigned short int>1]
            blue[j  + <unsigned short int>2]  = rgb[i    ]

    del rgb

    new_surface.blit(fromstring(
        bytes(red), (w, h), 'RGB'), (-offset_, -offset_), special_flags=0)
    new_surface.blit(fromstring(
        bytes(green), (w, h), 'RGB'), (0, 0), special_flags=BLEND_RGB_ADD)
    new_surface.blit(fromstring(
        bytes(blue), (w, h), 'RGB'), (offset_, offset_), special_flags=BLEND_RGB_ADD)
    return new_surface



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
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

    for i in prange(0, rows_, schedule=SCHEDULE, num_threads=THREADS):
        for j in range(0, cols_):
            r = &array_[i, j, <unsigned short int>0]
            g = &array_[i, j, <unsigned short int>1]
            b = &array_[i, j, <unsigned short int>2]
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
@cython.profile(False)
@cython.initializedcheck(False)
cdef inline tuple ripple_c(
       Py_ssize_t rows_, Py_ssize_t cols_,
       float [:, ::1] previous,
       float [:, ::1] current,
       unsigned char [:, :, :] array,
       float dispersion_ = 0.008
       ):
    """
    THIS SHADER CREATE A WATER EFFECT ON A PYGAME SURFACE
    This version does not include any background deformation to keep a reasonable fps rate

    Check demo_ripple.py in the Demo folder 

    :param rows_        : integer; Array width
    :param cols_        : integer; Array height
    :param previous     : numpy.ndarray type (w, h) type float; array use for the transformation
    :param current      : numpy.ndarray type (w, h) type float; array use for the transformation
    :param array        : numpy.ndarray type (w, h, 3) type unsigned char
    :paran dispersion   : float; ripple dampening factor, higher values decrease the ripple effect radius
    :return             : Return a tuple containing 3 arrays
    """

    cdef:
        int i, j, a, b
        float data
        float *c
        float *d
        unsigned char *e
        float r
        unsigned int row_1 = rows_ - 1
        unsigned int col_1 = cols_ - 1

    with nogil:

        for j in prange(0, cols_, schedule=SCHEDULE, num_threads=THREADS):
            for i in range(0, rows_):

                data = (previous[i + 1 if i <row_1 else 0, j]
                        + previous[i - 1 if i > 1 else 0, j] +
                              previous[i, j - 1 if j > 1 else 0] +
                        previous[i, j + 1 if j <col_1 else 0]) * <float>0.5

                c = &current[i, j]
                data = data - <float>c[0]
                c[0] = data - (data * dispersion_)
                d = &previous[i,j]
                e = &array[i, j, <unsigned short int>0]
                e[0] = <unsigned char> d[0] if d[0] > 0 else 0
                array[i, j, <unsigned short int>1] = e[0]
                array[i, j, <unsigned short int>2] = e[0]

    return current, previous



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cdef inline ripple_seabed_c(
           int rows_, int cols_,
           float [:, ::1] previous,                       # type numpy.float32 (w, h)
           float [:, ::1] current,                        # type numpy.float32 (w, h)
           unsigned char [:, :, ::1] texture_array,       # type numpy.ndarray (w, h, 3)
           unsigned char [:, :, :] background_array,    # type numpy.ndarray (w, h, 3)
           float dispersion_ = 0.008
           ):

    cdef:
        float cols2 = cols_ >> 1
        float rows2 = rows_ >> 1
        int i, j
        unsigned int a, b
        unsigned int cols_1 = cols_ - 1
        unsigned int rows_1 = rows_ - 1
        float data

    # from 1 to w - 1 to avoid python wraparound error
    # same for j (1 to h - 1)
    with nogil:
        for j in prange(1, cols_1, schedule=SCHEDULE, num_threads=THREADS):
            for i in range(1, rows_1):

                # data = (previous[i + 1, j] + previous[i - 1, j] +
                #         previous[i, j - 1] + previous[i, j + 1]) * <float>0.5

                data = (previous[i + 1 if i < rows_1 else 0, j]
                        + previous[i - 1 if i > 1 else 0, j] +
                        previous[i, j - 1 if j > 1 else 0] +
                        previous[i, j + 1 if j < rows_1 else 0]) * <float> 0.5


                data -= current[i, j]
                data -= data * dispersion_
                current[i, j] = data
                data = <float>1.0 - data * <float>ONE_1024
                a = max(<int>(((i - rows2) * data) + rows2) % rows_, 0)
                b = max(<int>(((j - cols2) * data) + cols2) % cols_, 0)
                background_array[i, j, 0], background_array[i, j, 1], background_array[i, j, 2] = \
                    texture_array[a, b, 0], texture_array[a, b, 1], texture_array[a, b, 2]

    return current, previous, asarray(background_array)


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
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
@cython.profile(False)
@cython.initializedcheck(False)
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
@cython.profile(False)
@cython.initializedcheck(False)
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
            for i in prange(0, size, bytesize, schedule=SCHEDULE, num_threads=THREADS):

                r = &rgb_array[i]
                g = &rgb_array[i + <unsigned short int>1]
                b = &rgb_array[i + <unsigned short int>2]

                s = r[0] + g[0] + b[0]
                index = <int>(s * f_map)

                r[0] = <unsigned char>heatmap_array[index, <unsigned short int>0]
                g[0] = <unsigned char>heatmap_array[index, <unsigned short int>1]
                b[0] = <unsigned char>heatmap_array[index, <unsigned short int>2]
        # BGR
        else:
            for i in prange(0, size, bytesize, schedule=SCHEDULE, num_threads=THREADS):

                r = &rgb_array[i]
                g = &rgb_array[i + <unsigned short int>1]
                b = &rgb_array[i + <unsigned short int>2]

                s = r[0] + g[0] + b[0]
                index = <int>(s * f_map)

                r[0] = <unsigned char>heatmap_array[index, <unsigned short int>2]
                g[0] = <unsigned char>heatmap_array[index, <unsigned short int>1]
                b[0] = <unsigned char>heatmap_array[index, <unsigned short int>0]



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
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
@cython.profile(False)
@cython.initializedcheck(False)
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
        short int bytesize = surface_.get_bytesize()

    with nogil:
        for i in prange(0, size, bytesize, schedule=SCHEDULE, num_threads=THREADS):

            r = &rgb_array[i]
            g = &rgb_array[i + <unsigned short int>1]
            b = &rgb_array[i + <unsigned short int>2]

            s = r[0] + g[0] + b[0]
            index = <unsigned int>(s * f_bluemap)

            r[0] = <unsigned char>bluemap_array[index, <unsigned short int>2]
            g[0] = <unsigned char>bluemap_array[index, <unsigned short int>1]
            b[0] = <unsigned char>bluemap_array[index, <unsigned short int>0]



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
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
@cython.profile(False)
@cython.initializedcheck(False)
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
        short int bytesize = surface_.get_bytesize()

    with nogil:
        for i in prange(0, size, bytesize, schedule=SCHEDULE, num_threads=THREADS):

            r = &rgb_array[i]
            g = &rgb_array[i + <unsigned short int>1]
            b = &rgb_array[i + <unsigned short int>2]

            s = r[0] + g[0] + b[0]
            index = <unsigned int>(s * f_redmap)

            r[0] = <unsigned char>redmap_array[index, <unsigned short int>2]
            g[0] = <unsigned char>redmap_array[index, <unsigned short int>1]
            b[0] = <unsigned char>redmap_array[index, <unsigned short int>0]



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cdef inline void blood_inplace_c(
        unsigned char [:, :, :] rgb_array_, float [:, :] mask_, float perc_):

    """
    SHADER HURT EFFECT (INPLACE)
    
    THE MASK DETERMINE THE CONTOUR USED FOR THE BLOOD EFFECT.

    The Array (rgb_array) must be a numpy array shape (w, h, 3)
    containing RGB pixels, please refer to pygame
    function pixels3d or array3d to convert an image into a
    3d array (library surfarray)
    
    e.g
    # Outside the main loop 
    blood_surface = pygame.image.load("../Assets/redvignette.png").convert_alpha()
    blood_surface = pygame.transform.smoothscale(blood_surface, (WIDTH, HEIGHT))
    BLOOD_MASK = numpy.asarray(
    pygame.surfarray.pixels_alpha(blood_surface) / 255.0, numpy.float32)
    
    # In the main loop (percentage must change overtime)
    blood(BCK, BLOOD_MASK, percentage)

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
        for j in prange(0, h, schedule=SCHEDULE, num_threads=THREADS):
            for i in range(0, w):

                r = &rgb_array_[i, j, <unsigned short int>0]

                index = <int>(r[0] * f_redmap)
                theta = <float>(mask_[i, j] * perc_)

                # ALTERNATIVE WITH BEST PERFORMANCES
                r[0] = <unsigned char> (
                    min(r[0] + <float> redmap_array[
                        index, <unsigned short int>0] * theta, <unsigned char>255))



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
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
    return <unsigned int>65536 * red + <unsigned int>256 * green + blue


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cdef inline unsigned int rgb_to_int_c(
        unsigned int red,
        unsigned int green,
        unsigned int blue
)nogil:
    """
    CONVERT RGB MODEL INTO A PYTHON INTEGER EQUIVALENT TO THE FUNCTION PYGAME MAP_RGB()

    Cython cpdef function, this function can be called directly and do not require a
    hook function.

    :param red   : Red color value,  must be in range [0..255]
    :param green : Green color value, must be in range [0..255]
    :param blue  : Blue color, must be in range [0.255]
    :return      : returns a positive python integer representing the RGB values(int32)
    """
    return <unsigned int>65536 * red + <unsigned int>256 * green + blue

@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
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

    rgb_.r = <float>((n >> <unsigned short int>16) & <unsigned char>255)
    rgb_.g = <float>((n >> <unsigned short int>8) & <unsigned char>255)
    rgb_.b = <float>(n & <unsigned char>255)
    return rgb_



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cdef inline rgb int_to_rgb_c(unsigned int n)nogil:
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

    rgb_.r = <float>((n >> <unsigned short int>16) & <unsigned char>255)
    rgb_.g = <float>((n >> <unsigned short int>8) & <unsigned char>255)
    rgb_.b = <float>(n & <unsigned char>255)
    return rgb_



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cdef inline unsigned int [::1] make_palette_c(int width, float fh, float fs, float fl):
    """

    CREATE A PALETTE OF MAPPED RGB COLORS VALUES 
    FROM HSL VALUES (HUE, SATURATION, LIGHTNESS)
    
    
    h, s, l = color[i] * fh,  min(fs, 255.0), min(color[i] * fl, 255.0)
    e.g:
        # below: palette of 256 colors (256 colors).
        # hue * 6, saturation = 255.0, lightness * 2.0
        palette, surf = make_palette(256, 6, 255, 2)
        palette, surf = make_palette(256, 4, 255, 2)

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
        for x in prange(width, schedule=SCHEDULE, num_threads=THREADS):
            h, s, l = <float>x * fh,  min(fs, <float>255.0), min(<float>x * fl, <float>255.0)
            rgb_ = struct_hsl_to_rgb(h * <float>ONE_360, s * <float>ONE_255, l * <float>ONE_255)
            # build the palette (1d buffer int values)
            palette[x] = rgb_to_int_c(<unsigned int>(rgb_.r * <float>255.0),
                                    <unsigned int>(rgb_.g * <float>255.0),
                                    <unsigned int>(rgb_.b * <float>255.0 * <float>0.5))

    return palette

@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cdef fire_surface24_c(
        int width,
        int height,
        float factor,
        unsigned int [::1] palette,
        float [:, ::1] fire,
        unsigned int intensity = 0,
        unsigned int low       = 0,
        unsigned int high      = 0,
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
        max_ = randRange(middle + <unsigned short int>1, high)

    else:
        middle = width >> 1
        min_ = randRange(0, middle)
        max_ = randRange(middle + <unsigned short int>1, width)


    with nogil:
        # POPULATE TO THE BASE OF THE FIRE (THIS WILL CONFIGURE THE FLAME ASPECT)
        for x in prange(min_, max_, schedule=SCHEDULE, num_threads=THREADS):
                fire[height-1, x] = <float>randRange(intensity, <unsigned int>260)


        # DILUTE THE FLAME EFFECT (DECREASE THE MAXIMUM INT VALUE) WHEN THE FLAME TRAVEL UPWARD
        for y in prange(1, height-1, schedule=SCHEDULE, num_threads=THREADS):

            c1 = (y + <unsigned short int> 1) % height
            for x in range(0, width):

                    c2 = x % width
                    d = (fire[c1, (x - <unsigned short int>1 + width) % width]
                       + fire[c1, c2]
                       + fire[c1, (x + <unsigned short int>1) % width]
                       + fire[(y + <unsigned short int>2) % height, c2]) * factor

                    d = d - <float>(<float>rand() * <float>0.0001)

                    # Cap the values
                    if d < <unsigned short int>0:
                        d = <float>0.0

                    # CAP THE VALUE TO 255
                    if d > <unsigned char>255:
                        d = <float>255.0
                    fire[y, x] = d

                    ii = palette[<unsigned int>d % width]

                    out[x, y, 0] = (ii >> <unsigned short int>16) & <unsigned char>255
                    out[x, y, 1] = (ii >> <unsigned short int>8) & <unsigned char>255
                    out[x, y, 2] = ii & <unsigned char>255

    return asarray(out)



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(False)
@cython.profile(False)
@cython.initializedcheck(False)
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
        # for x in prange(min_, max_, schedule=SCHEDULE, num_threads=THREADS
        #         fire[height - 1, x] = randRange(intensity, 260)

        # FIRE ARRAY IS [HEIGHT, WIDTH]
        for x in prange(min_, max_, schedule=SCHEDULE, num_threads=THREADS):
                fire[x % height, (height - 1) % width] = <float>randRange(intensity, <int>260)


        # DILUTE THE FLAME EFFECT (DECREASE THE MAXIMUM INT VALUE) WHEN THE FLAME TRAVEL UPWARD
        for y in prange(1, height - 1, schedule=SCHEDULE, num_threads=THREADS):
            c1 = (y + 1) % height
            for x in range(0, width):

                    c2 = x % width
                    d = (fire[c1, (x - 1 + width) % width]
                       + fire[c1, c2]
                       + fire[c1, (x + 1) % width]
                       + fire[(y + 2) % height, c2]) * factor

                    d = d - <float>(rand() * <float>0.0001)

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




@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cdef inline fire_effect_c(
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
    <brightness_exclude_inplace_c> to increase the brightness of the effect / texture
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
        h4 = <int>(<float>150.0 * height_/width_)
        low_ = <int>(low_ * low_/width_)
        high_ = <int>(high_ * <float>150.0/width_)
        reduce_factor_ = 0

    # TEXTURE 100 x 100 * ratio
    elif reduce_factor_ == 4:
        w4 = 100
        h4 = <int> (<float>100.0 * height_ / width_)
        low_ = <int> (low_ * low_ / width_)
        high_ = <int> (high_ * <float>100.0 / width_)
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

        # brightness_exclude_inplace_c(rgb_array_=rgb_array_,
        #                                      shift_=brightness_intensity_, color_=(0, 0, 0))
        brightness_bpf_c(rgb_array_, brightness_intensity_, 16)

    if blur_:
        blur_array_inplace_c(rgb_array_)

    if transpose_:
        rgb_array_ = rgb_array_.transpose(1, 0, 2)
        fire_surface_smallest = rotate(fire_surface_smallest, 90)


    # CONVERT THE ARRAY INTO A PYGAME SURFACE
    array_to_surface(fire_surface_smallest, rgb_array_)


    # BLOOM SHADER EFFECT
    if bloom_:
        assert 0 <= bpf_threshold_ < 256, \
            "Argument bpf_threshold_ must be in range [0 ... 256] got %s " % bpf_threshold_
        # bloom_array24_c(fire_surface_smallest, bpf_threshold_, fast_=fast_bloom_)
        try:
            # fire_surface_smallest = shader_bloom_fast(
            #     fire_surface_smallest, bpf_threshold_, fast_=fast_bloom_, factor_=1)

            shader_bloom_fast1(
                fire_surface_smallest,
                threshold_ = bpf_threshold_,
                smooth_    = 0,
                saturation_= True
            )

        except ValueError:
            raise ValueError(
                "The surface is too small and cannot be bloomed with shader_bloom_fast1.\n"
                "Increase the size of the image")

    # RESCALE THE SURFACE TO THE FULL SIZE
    if smooth_:
        return smoothscale(fire_surface_smallest, (width_, height_))

    else:
        return scale(fire_surface_smallest, (width_, height_))






@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(False)
@cython.profile(False)
@cython.initializedcheck(False)
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
        for x in prange(min_, max_, schedule=SCHEDULE, num_threads=THREADS):
                cloud_[(new_height - 1) % height, x % width] = randRange(intensity, 260)


        # DILUTE THE FLAME EFFECT (DECREASE THE MAXIMUM INT VALUE) WHEN THE FLAME TRAVEL UPWARD
        for y in prange(0, new_height - 1, schedule=SCHEDULE, num_threads=THREADS):
            c1 = (y + 1) % height
            for x in range(0, width):

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


# TODO MASK ? TO MOVE CLOUD ?

@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(False)
@cython.profile(False)
@cython.initializedcheck(False)
cdef inline cloud_effect_c(
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
    <brightness_exclude_inplace_c> to increase the brightness of the effect / texture
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
        brightness_exclude_inplace_c(rgb_array_=rgb_array_,
                                              shift_=brightness_intensity_, color_=(0, 0, 0))

    if blur_:
        blur_array_inplace_c(rgb_array_)

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
        bloom_array24_c(cloud_surface_smallest, bpf_threshold_, fast_=fast_bloom_)

    # RESCALE THE SURFACE TO THE FULL SIZE
    if smooth_:
        cloud_effect = smoothscale(cloud_surface_smallest, (width_, height_))
    else:
        cloud_effect = scale(cloud_surface_smallest, (width_, height_))

    return cloud_effect



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
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
        for x in prange(w, schedule=SCHEDULE, num_threads=THREADS):
            x2 = x >> <unsigned short int> 1
            x3 = <int> w - x2 - <unsigned short int> 1
            for y in range(h):

                r = &rgb_array_[x, y, <unsigned short int>0]
                g = &rgb_array_[x, y, <unsigned short int>1]
                b = &rgb_array_[x, y, <unsigned short int>2]

                new_array[x2, y, <unsigned short int>0] = r[<unsigned short int>0]
                new_array[x2, y, <unsigned short int>1] = g[<unsigned short int>0]
                new_array[x2, y, <unsigned short int>2] = b[<unsigned short int>0]

                new_array[x3, y, <unsigned short int>0] = r[<unsigned short int>0]
                new_array[x3, y, <unsigned short int>1] = g[<unsigned short int>0]
                new_array[x3, y, <unsigned short int>2] = b[<unsigned short int>0]

    return asarray(new_array)



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
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
        # unsigned char [:, :, :] rgb_array_copy = numpy.array(rgb_array_, copy=True)
        unsigned char [::1, :, :] rgb_array_copy = numpy.array(rgb_array_, copy=False, order='F')
        unsigned char *r
        unsigned char *g
        unsigned char *b

    with nogil:
        for x in prange(w, schedule=SCHEDULE, num_threads=THREADS):
            x2 = x >> <unsigned short int> 1
            x3 = <int> w - x2 - <unsigned short int> 1
            for y in range(h):

                r = &rgb_array_copy[x, y, <unsigned short int>0]
                g = &rgb_array_copy[x, y, <unsigned short int>1]
                b = &rgb_array_copy[x, y, <unsigned short int>2]

                rgb_array_[x2, y, <unsigned short int>0] = r[<unsigned short int>0]
                rgb_array_[x2, y, <unsigned short int>1] = g[<unsigned short int>0]
                rgb_array_[x2, y, <unsigned short int>2] = b[<unsigned short int>0]

                rgb_array_[x3, y, <unsigned short int>0] = r[<unsigned short int>0]
                rgb_array_[x3, y, <unsigned short int>1] = g[<unsigned short int>0]
                rgb_array_[x3, y, <unsigned short int>2] = b[<unsigned short int>0]


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
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

    assert freq_ > <float>0.0, "Argument freq_ must be > 0"
    assert duration_ > <float>0.0, "Argument duration_ must be > 0"

    cdef float t = damped_oscillation(<float>((<float>frame_ / freq_) % duration_))
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


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(False)
@cython.profile(False)
@cython.initializedcheck(False)
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

    cdef float t = damped_oscillation(<float>((<float>frame_ / freq_) % duration_)) * amplitude_
    return t

# --------------------------------------------------------------------------------------------------------
# KERNEL DEFINITION FOR SHARPEN ALGORITHM
cdef float [:, ::1] SHARPEN_KERNEL = numpy.array(([0, -1, 0],
                      [-1, 5, -1],
                      [0, -1, 0]), order='C').astype(dtype=float32)
cdef int HALF_KERNEL = <int>len(SHARPEN_KERNEL) >> 1

@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cdef inline void sharpen_inplace_c(unsigned char [:, :, :] rgb_array_):
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

            unsigned char [:, :, :] rgb_array_1 = \
                numpy.empty((w, h, 3), uint8)
            int x, y, xx, yy
            short kernel_offset_y, kernel_offset_x
            float r, g, b
            float * k
            unsigned char *rr
            unsigned char *gg
            unsigned char *bb
            int w_1 = <int>w - 1
            int h_1 = <int>h - 1

        with nogil:

            for y in prange(h, schedule=SCHEDULE, num_threads=THREADS):

                for x in range(w):

                    rr = &rgb_array_1[x, y, 0]
                    gg = &rgb_array_1[x, y, 1]
                    bb = &rgb_array_1[x, y, 2]

                    r, g, b = <unsigned char>0, <unsigned char>0, <unsigned char>0

                    for kernel_offset_y in range(
                            -HALF_KERNEL, HALF_KERNEL + 1):
                        yy = y + kernel_offset_y
                        if yy < 0:
                            yy = <unsigned short int> 0
                        if yy > h_1:
                            yy = h_1

                        for kernel_offset_x in range(
                                -HALF_KERNEL, HALF_KERNEL + 1):
                            xx = x + kernel_offset_x
                            if xx < 0:
                                xx = <unsigned short int>0
                            if xx > w_1:
                                xx = w_1

                            k = &SHARPEN_KERNEL[kernel_offset_y + HALF_KERNEL,
                                               kernel_offset_x + HALF_KERNEL]

                            r = r + rgb_array_[xx, yy, 0] * k[0]
                            g = g + rgb_array_[xx, yy, 1] * k[0]
                            b = b + rgb_array_[xx, yy, 2] * k[0]

                    if r < 0:
                        r = <float>0
                    if g < 0:
                        g = <float>0
                    if b < 0:
                        b = <float>0
                    if r > 255:
                        r= <float>255
                    if g > 255:
                        g = <float>255
                    if b > 255:
                        b = <float>255

                    rr[0] = <unsigned char>r
                    gg[0] = <unsigned char>g
                    bb[0] = <unsigned char>b

            for y in prange(h, schedule=SCHEDULE, num_threads=THREADS):

                for x in range(w):
                    rgb_array_[x, y, 0] =\
                        rgb_array_1[x, y, 0]
                    rgb_array_[x, y, 1] =\
                        rgb_array_1[x, y, 1]
                    rgb_array_[x, y, 2] =\
                        rgb_array_1[x, y, 2]


# Added to version 1.0.1
@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cdef cartoon_c(
        object surface_,
        unsigned int sobel_threshold_,
        unsigned int median_kernel_,
        unsigned int color_,
        unsigned int flag_):
    """

    :param surface_: pygame.Surface compatible 24 - 32 bit 
    :param sobel_threshold_: integer sobel threshold
    :param median_kernel_  : integer median kernel  
    :param color_          : integer; color reduction value (max color)
    :param flag_           : integer; Blend flag e.g (BLEND_RGB_ADD, BLEND_RGB_SUB, 
                             BLEND_RGB_MULT, BLEND_RGB_MAX, BLEND_RGB_MIN  
    :return               : Return a pygame Surface with the cartoon effect 
    """

    surface_branch_1 = surface_.copy()

    try:
        array_ = surface_branch_1.get_view('3')

    except Exception as e:
        raise ValueError(
        "Cannot reference source pixels into a 3d array.\n %s " % e)

    sobel_inplace_c(array_, sobel_threshold_)
    median_fast(surface_, kernel_size_=median_kernel_, reduce_factor_=1)
    pygame.surfarray.array_to_surface(surface_branch_1, array_)
    del array_
    surface_.blit(surface_branch_1, (0, 0), special_flags=flag_)
    color_reduction(surface_, color_)
    return surface_


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cdef object blending(object source_, unsigned char[:, :, :] destination_, float percentage_):
    """
    BLEND A SOURCE TEXTURE TOWARD A DESTINATION TEXTURE (TRANSITION EFFECT)

    * Video system must be initialised 
    * source_ & destination_ Textures must be same sizes
    * Compatible with 24 - 32 bit surface
    * Output create a new surface


    :param source_     : pygame.Surface (Source)
    :param destination_: pygame.Surface (Destination)
    :param percentage_ : float; Percentage value between [0.0 ... 100.0]
    :return: return    : Return a 24 bit pygame.Surface and blended with a percentage
                         of the destination texture.
    """

    cdef:
            unsigned char [:, :, :] source_array

    try:
        source_array      = source_.get_view('3')

    except Exception as e:
        raise ValueError("\nCannot reference source pixels into a 3d array.\n %s " % e)

    cdef:

        unsigned char c1, c2, c3
        int i=0, j=0
        Py_ssize_t w = source_array.shape[0]
        Py_ssize_t h = source_array.shape[1]
        unsigned char[:, :, ::1] final_array = \
            numpy.ascontiguousarray(empty((h, w, 3), dtype=uint8))
        float c4 = percentage_ * <float>0.01
        float tmp = <float> 1.0 - c4

    with nogil:
        for j in prange(h, schedule=SCHEDULE, num_threads=THREADS):
            for i in range(w):

                c1 = min(<unsigned char> (<float> destination_[i, j, 0] * c4 +
                                source_array[i, j, 0] * tmp),
                         <unsigned char>255)
                c2 = min(<unsigned char> (<float> destination_[i, j, 1] * c4 +
                                source_array[i, j, 1] * tmp),
                         <unsigned char>255)
                c3 = min(<unsigned char> (<float> destination_[i, j, 2] * c4 +
                                source_array[i, j, 2] * tmp),
                         <unsigned char>255)

                final_array[j, i, 0] = c1 # if c1>0 else 0
                final_array[j, i, 1] = c2 # if c2>0 else 0
                final_array[j, i, 2] = c3 # if c3>0 else 0

    return frombuffer(final_array, (w, h), 'RGB')


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cdef inline void blend_inplace_c(
        object source_,
        unsigned char[:, :, :] destination_,
        float percentage_
       ):
    """
    BLEND A SOURCE TEXTURE TOWARD A DESTINATION TEXTURE (TRANSITION EFFECT)

    * Video system must be initialised 
    * source_ & destination_ Textures must be same sizes
    * Compatible with 24 - 32 bit surface
    * Output create a new surface

    :param source_     : pygame.Surface (Source)
    :param destination_: 3d array, numpy.ndarray  (Destination)
    :param percentage_ : float; Percentage value between [0.0 ... 100.0]
    :return: return    : Return a 24 bit pygame.Surface and blended with a percentage
                         of the destination texture.
    """

    cdef:
            unsigned char [:, :, :] source_array
    try:
        source_array  =  source_.get_view('3')

    except Exception as e:
        raise ValueError("\nCannot reference source pixels into a 3d array.\n %s " % e)

    cdef:

        unsigned char c1, c2, c3
        int i=0, j=0
        Py_ssize_t w = source_array.shape[0]
        Py_ssize_t h = source_array.shape[1]
        float c4 = percentage_ * <float> 0.01
        float tmp = <float> 1.0 - c4

    with nogil:

        for j in prange(h, schedule=SCHEDULE, num_threads=THREADS):
            for i in range(w):

                c1 = min(<unsigned char> (<float> destination_[i, j, 0] * c4 +
                                source_array[i, j, 0] * tmp), <unsigned char>255)
                c2 = min(<unsigned char> (<float> destination_[i, j, 1] * c4 +
                                source_array[i, j, 1] * tmp), <unsigned char>255)
                c3 = min(<unsigned char> (<float> destination_[i, j, 2] * c4 +
                                source_array[i, j, 2] * tmp), <unsigned char>255)

                source_array[ i, j, 0 ] = c1 # if c1 > 0 else 0
                source_array[ i, j, 1 ] = c2 # if c2 > 0 else 0
                source_array[ i, j, 2 ] = c3 # if c3 > 0 else 0



# new version 1.0.5
@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cpdef object alpha_blending(object source_, object destination_):
    """
    ALPHA BLENDING 

    * Video system must be initialised 
    * source_ & destination_ Textures must be same sizes
    * Compatible with 32 bit surfaces only
    * Output create a new surface

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
        source_array      = \
            (numpy.frombuffer(source_.get_view('0').raw,
             dtype=numpy.uint8) / <float>255.0).astype(dtype=numpy.float32)

    except Exception as e:
        raise ValueError("\nCannot reference source pixels into a 3d array.\n %s " % e)

    try:
        destination_array =\
            (numpy.frombuffer(destination_.get_view('0').raw,
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
        for i in prange(0, l, 4, schedule=SCHEDULE, num_threads=THREADS):

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
                final_array[i]   = <unsigned char>min(rr * <float>255.0, <unsigned char>255)
                final_array[i+1] = <unsigned char>min(gg * <float>255.0, <unsigned char>255)
                final_array[i+2] = <unsigned char>min(bb * <float>255.0, <unsigned char>255)
                final_array[i+3] = <unsigned char>min(alpha * <float>255.0, <unsigned char>255)

    return pygame.image.frombuffer(numpy.asarray(
        final_array).reshape(w, h, 4), (w, h), 'RGBA')

# new version 1.0.5
@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cpdef void alpha_blending_inplace(object source_, object destination_):
    """
    ALPHA BLENDING INPLACE

    * Video system must be initialised 
    * source_ & destination_ Textures must be same sizes
    * Compatible with 32 bit surfaces only
    * Output create a new surface


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
        dest_rgb = destination_.get_view('3')
        dest_alpha = pixels_alpha(destination_)
        destination_array = (numpy.dstack(
            (numpy.asarray(dest_rgb), numpy.asarray(dest_alpha)))/<float>255.0
                             ).astype(dtype=numpy.float32)


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
        for j in prange(h, schedule=SCHEDULE, num_threads=THREADS):
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
                dest_rgb[i, j, 0] = <unsigned char>min(rr * <float>255.0, <unsigned char>255)
                dest_rgb[i, j, 1] = <unsigned char>min(gg * <float>255.0, <unsigned char>255)
                dest_rgb[i, j, 2] = <unsigned char>min(bb * <float>255.0, <unsigned char>255)
                dest_alpha[i, j] = <unsigned char>min(alpha * <float>255.0, <unsigned char>255)




@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cdef inline void dirt_lens_c(
        object source_,
        object lens_model_,
        int flag_,
        float light_=0.0):

    if light_!=0.0:

        lens_model_ = brightness_c(lens_model_.get_view('3'), light_)

    source_.blit(lens_model_, (0, 0), special_flags=flag_)



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cdef object dithering_c(float [:, :, :] rgb_array_):

    """
    Dithering is used in computer graphics to create the illusion of "color depth" in images with
    a limited color palette - a technique also known as color quantization. In a dithered image,
    colors that are not available in the palette are approximated by a diffusion of colored pixels
    from within the available palette. The human eye perceives the diffusion as a mixture of 
    the colors within it (see color vision). Dithered images, particularly those with relatively
    few colors, can often be distinguished by a characteristic graininess or speckled appearance.
    
    :param rgb_array_: pygame.Surface compatible 24-32 bit
    
    A value of 2 means a total of 8 colors
    :return: pygame surface 24-32 bit    
    """

    cdef Py_ssize_t w, h
    w = <object> rgb_array_.shape[ 0 ]
    h = <object> rgb_array_.shape[ 1 ]

    cdef:
        int x=0, y=0
        float new_red, new_green, new_blue
        float quantization_error_red, quantization_error_green, quantization_error_blue
        float oldr, oldg, oldb
        float * r1
        float * g1
        float * b1

    with nogil:

        # prange cause the data to overlap between threads, this is why I am defining a chunksize of
        # h*3 to separate the threads with an interval of h*3. However the algorithm will run faster
        # without prange on that occasion
        for y in range(0, h): #, schedule=SCHEDULE, num_threads=THREADS , chunksize=h*3):

            for x in range(0, w):

                oldr = rgb_array_[x, y, 0]
                oldg = rgb_array_[x, y, 1]
                oldb = rgb_array_[x, y, 2]

                new_red   = <float>round_c(oldr)
                new_green = <float>round_c(oldg)
                new_blue  = <float>round_c(oldb)

                rgb_array_[x, y, 0] = new_red
                rgb_array_[x, y, 1] = new_green
                rgb_array_[x, y, 2] = new_blue

                quantization_error_red   = <float>(oldr - new_red)
                quantization_error_green = <float>(oldg - new_green)
                quantization_error_blue  = <float>(oldb - new_blue)

                if x < w - 1:
                    r1 = &rgb_array_[x + 1, y, 0]
                    r1[0] = r1[0] + quantization_error_red * C1
                    g1 = &rgb_array_[x + 1, y, 1]
                    g1[0] = g1[0] + quantization_error_green * C1
                    b1 = &rgb_array_[x + 1, y, 2]
                    b1[0] = b1[0] + quantization_error_blue * C1

                if y < h - 1:

                    if x > 0:
                        r1 = &rgb_array_[x - 1, y + 1, 0]
                        r1[0] = r1[0] + quantization_error_red * C2
                        g1 = &rgb_array_[x - 1, y + 1, 1]
                        g1[0] = g1[0] + quantization_error_green * C2
                        b1 = &rgb_array_[x - 1, y + 1, 2]
                        b1[0] = b1[0] + quantization_error_blue * C2

                    r1 = &rgb_array_[x, y + 1, 0]
                    r1[0] = r1[0] + quantization_error_red * C3
                    g1 = &rgb_array_[x, y + 1, 1]
                    g1[0] = g1[0] + quantization_error_green * C3
                    b1 = &rgb_array_[x, y + 1, 2]
                    b1[0] = b1[0] + quantization_error_blue * C3

                    if x < w - 1:
                        r1 = &rgb_array_[x + 1, y + 1, 0]
                        r1[0] = r1[0] + quantization_error_red * C4
                        g1 = &rgb_array_[x + 1, y + 1, 1]
                        g1[0] =  g1[0] + quantization_error_green * C4
                        b1 = &rgb_array_[x + 1, y + 1, 2]
                        b1[0] = b1[0] + quantization_error_blue * C4

    # return make_surface((asarray(rgb_array_) * <float>255.0).astype(dtype=numpy.uint8))
    arr = (asarray(rgb_array_).transpose(1, 0, 2) * <float> 255.0).astype(dtype=numpy.uint8, order='C')
    return frombuffer(arr, (w, h), "RGB")




@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cdef inline void dithering_inplace_c(float [:, :, :] rgb_array_, unsigned char[:, :, :] tmp):


    cdef Py_ssize_t w, h
    w = <object> rgb_array_.shape[ 0 ]
    h = <object> rgb_array_.shape[ 1 ]

    cdef:
        int x=0, y=0
        float new_red, new_green, new_blue
        float quantization_error_red, quantization_error_green, quantization_error_blue
        float oldr, oldg, oldb
        float * r1
        float * g1
        float * b1

    with nogil:

        for y in prange(0, h, schedule=SCHEDULE, num_threads=THREADS, chunksize=h):

            for x in range(0, w):

                oldr = rgb_array_[x, y, <unsigned short int>0]
                oldg = rgb_array_[x, y, <unsigned short int>1]
                oldb = rgb_array_[x, y, <unsigned short int>2]

                new_red   = round_c(oldr)
                new_green = round_c(oldg)
                new_blue  = round_c(oldb)

                rgb_array_[x, y, <unsigned short int>0] = new_red
                rgb_array_[x, y, <unsigned short int>1] = new_green
                rgb_array_[x, y, <unsigned short int>2] = new_blue

                quantization_error_red   = <float>(oldr - new_red)
                quantization_error_green = <float>(oldg - new_green)
                quantization_error_blue  = <float>(oldb - new_blue)

                if x < w - 1:
                    r1 = &rgb_array_[x + 1, y, 0]
                    r1[0] = r1[0] + quantization_error_red * C1
                    g1 = &rgb_array_[x + 1, y, 1]
                    g1[0] = g1[0] + quantization_error_green * C1
                    b1 = &rgb_array_[x + 1, y, 2]
                    b1[0] = b1[0] + quantization_error_blue * C1

                if y < h - 1:

                    if x > 0:
                        r1 = &rgb_array_[x - 1, y + 1, 0]
                        r1[0] = r1[0] + quantization_error_red * C2
                        g1 = &rgb_array_[x - 1, y + 1, 1]
                        g1[0] = g1[0] + quantization_error_green * C2
                        b1 = &rgb_array_[x - 1, y + 1, 2]
                        b1[0] = b1[0] + quantization_error_blue * C2

                    r1 = &rgb_array_[x, y + 1, 0]
                    r1[0] = r1[0] + quantization_error_red * C3
                    g1 = &rgb_array_[x, y + 1, 1]
                    g1[0] = g1[0] + quantization_error_green * C3
                    b1 = &rgb_array_[x, y + 1, 2]
                    b1[0] = b1[0] + quantization_error_blue * C3

                    if x < w -1:
                        r1 = &rgb_array_[x + 1, y + 1, 0]
                        r1[0] = r1[0] + quantization_error_red * C4
                        g1 = &rgb_array_[x + 1, y + 1, 1]
                        g1[0] = g1[0] + quantization_error_green * C4
                        b1 = &rgb_array_[x + 1, y + 1, 2]
                        b1[0] = b1[0] + quantization_error_blue * C4

        for y in prange(1, h, schedule=SCHEDULE, num_threads=THREADS):
            for x in range(1, w):
                tmp[x, y, 0] = <unsigned char>(rgb_array_[x, y, 0] * <float>255.0)
                tmp[x, y, 1] = <unsigned char>(rgb_array_[x, y, 1] * <float>255.0)
                tmp[x, y, 2] = <unsigned char>(rgb_array_[x, y, 2] * <float>255.0)



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cdef object dithering_atkinson_c(float [:, :, :] rgb_array_):

    """
    Dithering is used in computer graphics to create the illusion of "color depth" in images with
    a limited color palette - a technique also known as color quantization. In a dithered image,
    colors that are not available in the palette are approximated by a diffusion of colored pixels
    from within the available palette. The human eye perceives the diffusion as a mixture of 
    the colors within it (see color vision). Dithered images, particularly those with relatively
    few colors, can often be distinguished by a characteristic graininess or speckled appearance.

    :param rgb_array_: pygame.Surface compatible 24-32 bit

    A value of 2 means a total of 8 colors
    :return: pygame surface 24-32 bit    
    """

    cdef Py_ssize_t w, h
    w = <object> rgb_array_.shape[ 0 ]
    h = <object> rgb_array_.shape[ 1 ]

    cdef:
        int x=0, y=0
        float new_red, new_green, new_blue
        float quantization_error_red, quantization_error_green, quantization_error_blue
        float oldr, oldg, oldb
        float * r1
        float * g1
        float * b1

    with nogil:

        # prange cause the data to overlap between threads, this is why I am defining a chunksize of
        # h*3 to separate the threads with an interval of h*3. However the algorithm will run faster
        # without prange on that occasion
        for y in range(0, h): #, schedule=SCHEDULE, num_threads=THREADS):# , chunksize=h*3):

            for x in range(0, w):

                oldr = rgb_array_[x, y, 0]
                oldg = rgb_array_[x, y, 1]
                oldb = rgb_array_[x, y, 2]

                new_red   = round_c(oldr)
                new_green = round_c(oldg)
                new_blue  = round_c(oldb)

                rgb_array_[x, y, 0] = new_red
                rgb_array_[x, y, 1] = new_green
                rgb_array_[x, y, 2] = new_blue

                quantization_error_red   = <float>(oldr - new_red) * <float>0.125
                quantization_error_green = <float>(oldg - new_green) * <float>0.125
                quantization_error_blue  = <float>(oldb - new_blue) * <float>0.125

                if x < w - 1:
                    r1 = &rgb_array_[x + 1, y, 0]
                    r1[0] += quantization_error_red
                    g1 = &rgb_array_[x + 1, y, 1]
                    g1[0] += quantization_error_green
                    b1 = &rgb_array_[x + 1, y, 2]
                    b1[0] += quantization_error_blue

                if x < w - 2:
                    r1 = &rgb_array_[x + 2, y, 0]
                    r1[0] += quantization_error_red
                    g1 = &rgb_array_[x + 2, y, 1]
                    g1[0] += quantization_error_green
                    b1 = &rgb_array_[x + 2, y, 2]
                    b1[0] += quantization_error_blue

                if y < h - 1:
                        if x > 0:
                            r1 = &rgb_array_[x - 1, y + 1, 0]
                            r1[0] += quantization_error_red
                            g1 = &rgb_array_[x - 1, y + 1, 1]
                            g1[0] +=  quantization_error_green
                            b1 = &rgb_array_[x - 1, y + 1, 2]
                            b1[0] += quantization_error_blue

                        r1 = &rgb_array_[x, y + 1, 0]
                        r1[0] += quantization_error_red
                        g1 = &rgb_array_[x, y + 1, 1]
                        g1[0] += quantization_error_green
                        b1 = &rgb_array_[x, y + 1, 2]
                        b1[0] += quantization_error_blue

                        if x < w - 1:
                            r1 = &rgb_array_[x + 1, y + 1, 0]
                            r1[0] += quantization_error_red
                            g1 = &rgb_array_[x + 1, y + 1, 1]
                            g1[0] += quantization_error_green
                            b1 = &rgb_array_[x + 1, y + 1, 2]
                            b1[0] += quantization_error_blue

                if y < h - 2:
                    r1 = &rgb_array_[x, y + 2, 0]
                    r1[0] += quantization_error_red
                    g1 = &rgb_array_[x, y + 2, 1]
                    g1[0] += quantization_error_green
                    b1 = &rgb_array_[x, y + 2, 2]
                    b1[0] += quantization_error_blue


    arr = (asarray(rgb_array_).transpose(1,0,2) * <float>255.0).astype(dtype=numpy.uint8, order='C')
    return frombuffer(arr, (w, h), "RGB")




@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
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


    with nogil:
        for y in prange(h, schedule=SCHEDULE, num_threads=THREADS):
            for x in range(0, w):

                r = &rgb_array[x, y, 0]
                g = &rgb_array[x, y, 1]
                b = &rgb_array[x, y, 2]

                r[ 0 ] = <unsigned char>(round_c(c1 * <float> r[ 0 ] ) * f)
                g[ 0 ] = <unsigned char>(round_c(c1 * <float> g[ 0 ] ) * f)
                b[ 0 ] = <unsigned char>(round_c(c1 * <float> b[ 0 ] ) * f)


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
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
        for i in prange(380, 750, schedule=SCHEDULE, num_threads=THREADS):
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
@cython.profile(False)
@cython.initializedcheck(False)
cdef inline void palette_change_c(
        unsigned char [:, :, :] rgb_array_,
        float [:, :] palette_,
        float [:, ::1] tmp_v_
        ):
    """
    CHANGE AN IMAGE COLOR USING A GIVEN PALETTE

    The palette contains RGB values (float in range 0.0 - 255.0)

    :param rgb_array_: numpy.ndarray containing RGB pixel values, type (w, h, 3)
        dtype uint8 range (0 .. 255)
        
    :param palette_: numpy.ndarray containing the palette colors to use for
        substituting the image colors, array format (w, 3) of type float range (0.0 ... 255.0)
        e.g 
        from PygameShader import IRIDESCENTCRYSTAL
        
    :param tmp_v_ : numpy.ndarray (contiguous array) shape 
        (rgb_array_.shape[0] * rgb_array_.shape[1], len(palette_.shape[0])) of type float32
        Temporary array to increase performance (the array does not have to be redeclared every
        frames. 
        e.g 
        tmp_v = numpy.ascontiguousarray(numpy.ndarray(
            (SURFACE.get_width()*SURFACE.get_height(),
            IRIDESCENTCRYSTAL.shape[0]), dtype=float32
        ))
    """

    cdef:
        int i, j, k
        int w = <object>rgb_array_.shape[0]
        int h = <object>rgb_array_.shape[1]
        Py_ssize_t p_length = <object>palette_.shape[0]
        # float * tmp_v = <float *> malloc(p_length * sizeof(float))
        # float [:, ::1] tmp_v = ascontiguousarray(empty((w*h, p_length), dtype=numpy.float32))
        # unsigned int s_min = 0
        unsigned char * r
        unsigned char * g
        unsigned char * b
        float min_v
        unsigned int index, ji


    with nogil:
        for j in prange(h, schedule=SCHEDULE, num_threads=THREADS): #, chunksize=h):
            for i in range(w):
                ji = j*i
                # Get the RGB values of the current pixel
                r = &rgb_array_[i, j, 0]
                g = &rgb_array_[i, j, 1]
                b = &rgb_array_[i, j, 2]

                # Get the Distance for all palette colors using the current pixels and
                # place the distance values into a buffer 1d C array
                for k in range(0, p_length):
                    tmp_v_[ji, k ] = <float>(
                       (<float>r[0] - <float>palette_[ k, 0 ]) ** 2 + \
                       (<float>g[0] - <float>palette_[ k, 1 ]) ** 2 + \
                       (<float>b[0] - <float>palette_[ k, 2 ]) ** 2)

                # Use an external C function to find the smallest distance from
                # the current pixel. The smallest difference will be the closest
                # color from the palette and the original RGB pixel colors will be
                # substitute with the equivalent palette color.
                # !! This algorithm does not take into account other solutions!!
                # More than one palette colors might have the same distance to the original
                # RGB pixel but only the first found color will be returned.

                # s_min = <unsigned int>min_index(&tmp_v[0], p_length)

                # Below same function min_index but allow, multi-processing without
                # chunking (for j in prange(h, schedule=SCHEDULE, num_threads=THREADS)
                # index = 0
                index = 0
                min_v = tmp_v_[ji, 0 ]
                for k in range(0, p_length):
                    if tmp_v_[ji, k ] < min_v:
                        min_v = tmp_v_[ji, k ]
                        index = k;

                # Substitute the current pixel with the equivalent palette colors
                r[0] = <unsigned char>palette_[ index, 0 ]
                g[0] = <unsigned char>palette_[ index, 1 ]
                b[0] = <unsigned char>palette_[ index, 2 ]

    # free(tmp_v)



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cdef inline float distance_ (float x, float y)nogil:
  return <float>sqrt(x*x + y*y)

@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cdef inline float gaussian_ (float v, float sigma2)nogil:
  # return (<float>1.0 / (<float>M_PI * sigma2)) * <float>exp(-(v * v ) / sigma2)

  # sigma2 is inversed
  return (INV_M_PI * sigma2)* <float>exp(-(v * v) * sigma2)


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cdef bilateral_c(
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
        unsigned char * rr
        unsigned char * gg
        unsigned char * bb
        float sigma_i2 = <float>1.0/(<float>2.0 * sigma_i_ * sigma_i_)
        float sigma_s2 = <float>1.0/(<float>2.0 * sigma_s_ * sigma_s_)

    with nogil:

        for y in prange(0, h, schedule=SCHEDULE, num_threads=THREADS):
            for x in range(0, w):

                ir, ig, ib = <float>0.0, <float>0.0, <float>0.0
                wpr, wpg, wpb = <float>0.0, <float>0.0, <float>0.0

                rr = &rgb_array_[x, y, 0]
                gg = &rgb_array_[x, y, 0]
                bb = &rgb_array_[x, y, 0]

                for ky in range(-k, k + 1):
                    yy = y + ky

                    if yy < 0:
                        continue
                    elif yy >= h:
                        continue

                    for kx in range(-k, k + 1):
                        xx = x + kx

                        if xx < 0:
                            continue
                        elif xx >= w:
                            continue

                        gs = gaussian_(distance_(<float>kx, <float>ky), sigma_s2)

                        r = &rgb_array_[xx, yy, 0]
                        g = &rgb_array_[xx, yy, 1]
                        b = &rgb_array_[xx, yy, 2]

                        wr = gaussian_(<float>r[0] - <float>rr[0], sigma_i2) * gs
                        wg = gaussian_(<float>g[0] - <float>gg[0], sigma_i2) * gs
                        wb = gaussian_(<float>b[0] - <float>bb[0], sigma_i2) * gs

                        ir = ir + r[0] * wr
                        ig = ig + g[0] * wg
                        ib = ib + b[0] * wb

                        wpr = wpr + wr
                        wpg = wpg + wg
                        wpb = wpb + wb

                ir = ir / wpr
                ig = ig / wpg
                ib = ib / wpb

                bilateral[y, x, 0] = <unsigned int>ir
                bilateral[y, x, 1] = <unsigned int>ig
                bilateral[y, x, 2] = <unsigned int>ib

    return frombuffer(bilateral, (w, h), 'RGB')




EMBOSS_KERNEL5x5= \
    numpy.array((
        [-1.0, -1.0, -1.0, -1.0, 0.0],
        [-1.0, -1.0, -1.0, 0.0,  1.0],
        [-1.0, -1.0,  0.0, 1.0,  1.0],
        [-1.0,  0.0,  1.0, 1.0,  1.0],
        [ 0.0,  1.0,  1.0, 1.0,  1.0])).astype(dtype=numpy.float32, order='C')

EMBOSS_KERNEL5x5 = \
    numpy.array((
        [-2.0, -1.0, 0.0],
        [-1.0,  1.0, 1.0],
        [ 0.0,  1.0, 2.0],
    )).astype(dtype=numpy.float32, order='C')

cdef float EMBOSS_KERNEL_WEIGHT = numpy.sum(EMBOSS_KERNEL5x5)

@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cdef object emboss5x5_c(unsigned char [:, :, :] rgb_array_):

    cdef:
        float k_weight = EMBOSS_KERNEL_WEIGHT
        unsigned int k_length = <unsigned int>len(EMBOSS_KERNEL5x5)
        unsigned int half_kernel = <unsigned int>len(EMBOSS_KERNEL5x5) >> 1

    # texture sizes
    cdef int w, h
    w = <object>rgb_array_.shape[0]
    h = <object>rgb_array_.shape[1]

    cdef:
        float [:, ::1] kernel = EMBOSS_KERNEL5x5
        float kernel_weight = k_weight
        short kernel_half = half_kernel
        unsigned char [:, :, ::1] emboss = empty((h, w, 3), order='C', dtype=uint8)
        int kernel_length = k_length
        int x, y
        unsigned int xx, yy
        unsigned short red, green, blue,
        short kernel_offset_y, kernel_offset_x
        float r, g, b, k

    with nogil:

        for y in prange(0, h, schedule=SCHEDULE, num_threads=THREADS):

            for x in range(0, w):

                r, g, b = <float>0.0, <float>0.0, <float>0.0

                for kernel_offset_y in range(-kernel_half, kernel_half + 1):

                    yy = y + kernel_offset_y
                    if yy < 0:
                        continue
                    elif yy > h - 1:
                        continue

                    for kernel_offset_x in range(-kernel_half, kernel_half + 1):

                        xx = x + kernel_offset_x

                        if xx < 0:
                            continue
                        elif xx > w - 1:
                            continue

                        red, green, blue = \
                            rgb_array_[xx, yy, 0], \
                            rgb_array_[xx, yy, 1],\
                            rgb_array_[xx, yy, 2]

                        k = kernel[kernel_offset_y + kernel_half, kernel_offset_x + kernel_half]
                        r = r + red * k
                        g = g + green * k
                        b = b + blue * k

                if r < 0:
                    r = <float>0.0
                if g < 0:
                    g = <float>0.0
                if b < 0:
                    b = <float>0.0
                if r > 255:
                    r= <float>255.0
                if g > 255:
                    g = <float>255.0
                if b > 255:
                    b = <float>255.0

                emboss[y, x, 0], emboss[y, x, 1], \
                emboss[y, x, 2] = <unsigned char>r, <unsigned char>g, <unsigned char>b

    return frombuffer(emboss, (w, h), 'RGB')





@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cdef object bilinear_c(
        unsigned char [:, :, :] rgb_array_,
        tuple size_,
        fx=None, fy=None):

    cdef Py_ssize_t w, h
    w, h = rgb_array_.shape[:2]

    cdef:
        float rowscale, colscale
        float original_x, original_y
        unsigned int bl, br, tl, tr,\
            modxiplusonelim, modyiplusonelim
        unsigned int modxi, modyi
        int x, y
        float modxf, modyf, b, t, xf, yf
        int new_width = size_[0]
        int new_height = size_[1]

    rowscale = <float>w / <float>new_width
    colscale = <float>h / <float>new_height

    if fx is not None:
        new_width = <unsigned int> (w * fx)
    if fy is not None:
       new_height = <unsigned int>(h * fy)

    cdef unsigned char [: , :, ::1] new_rgb = \
        numpy.empty((new_height, new_width, 3), dtype=numpy.uint8)

    with nogil:
        for y in prange(0, new_height, schedule=SCHEDULE, num_threads=THREADS):
            original_y = <float> y * colscale
            modyi = <unsigned int> original_y
            modyf = original_y - modyi
            modyiplusonelim = min(modyi + 1, h - 1)
            yf = <float> 1.0 - modyf

            for x in prange(new_width):
                original_x = <float> x * rowscale
                modxi = <unsigned int> original_x
                modxf = original_x - modxi
                modxiplusonelim = min(modxi + 1, w - 1)
                xf = <float> 1.0 - modxf

                bl = rgb_array_[modxi, modyi, 0]
                br = rgb_array_[modxiplusonelim, modyi, 0]
                tl = rgb_array_[modxi, modyiplusonelim, 0]
                tr = rgb_array_[modxiplusonelim, modyiplusonelim, 0]

                # Calculate interpolation
                b = modxf * br + xf * bl
                t = modxf * tr + xf * tl

                new_rgb[y, x, 0] = \
                    <unsigned int> (modyf * t + yf * b + <float> 0.5)

                bl = rgb_array_[modxi, modyi, 1]
                br = rgb_array_[modxiplusonelim, modyi, 1]
                tl = rgb_array_[modxi, modyiplusonelim, 1]
                tr = rgb_array_[modxiplusonelim, modyiplusonelim, 1]

                # Calculate interpolation
                b = modxf * br + xf * bl
                t = modxf * tr + xf * tl

                new_rgb[y, x, 1] =\
                    <unsigned int> (modyf * t + yf * b + <float> 0.5)

                bl = rgb_array_[modxi, modyi, 2]
                br = rgb_array_[modxiplusonelim, modyi, 2]
                tl = rgb_array_[modxi, modyiplusonelim, 2]
                tr = rgb_array_[modxiplusonelim, modyiplusonelim, 2]

                # Calculate interpolation
                b = modxf * br + xf * bl
                t = modxf * tr + xf * tl

                new_rgb[y, x, 2] = \
                    <unsigned int> (modyf * t + yf * b + <float> 0.5)

    return frombuffer(new_rgb, (new_width, new_height), 'RGB')





@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cpdef unsigned char[::1] stack_buffer_c(rgb_array_, alpha_, int w, int h, bint transpose=False):
    """

    STACK RGB & ALPHA MEMORYVIEWSLICE C-BUFFERS STRUCTURES TOGETHER.

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

        for i in prange(0, b_length, 3, schedule=SCHEDULE, num_threads=THREADS):
                ii = i // 3
                jj = ii * 4
                new_buffer[jj]   = rgb_array[i]
                new_buffer[jj+1] = rgb_array[i+1]
                new_buffer[jj+2] = rgb_array[i+2]
                new_buffer[jj+3] = alpha[ii]

        if transpose:
            for i in prange(0, w4, 4, schedule=SCHEDULE, num_threads=THREADS):
                for j in range(0, h):
                    index = i + (w4 * j)
                    k = (j * 4) + (i * h)
                    flipped_array[k    ] = new_buffer[index    ]
                    flipped_array[k + 1] = new_buffer[index + 1]
                    flipped_array[k + 2] = new_buffer[index + 2]
                    flipped_array[k + 3] = new_buffer[index + 3]
            return flipped_array

    return new_buffer



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
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
        for i in prange(w, schedule=SCHEDULE, num_threads=THREADS):
            for j in range(h):
                array_f[i, j] = <float>(array[i, j] * <float>ONE_255)
    return array_f


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cpdef tuple area24_c(int x, int y, np.ndarray[np.uint8_t, ndim=3] background_rgb,
              np.ndarray[np.uint8_t, ndim=2] mask_alpha, float intensity=1.0,
              float [:] color=numpy.array([128.0, 128.0, 128.0], dtype=numpy.float32, copy=False),
              bint smooth=False, bint saturation=False, float sat_value=0.2, bint bloom=False,
              unsigned int threshold=110, bint heat=False, float frequency=1):
    """

    CREATE A REALISTIC LIGHT EFFECT ON A PYGAME.SURFACE OR TEXTURE.

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
    :param threshold: unsigned int, default 110
    :param sat_value: float; Set the saturation value
    :param saturation: boolean; Saturation effect
    :param smooth: boolean; Blur effect
    :param frequency: float; frequency must be incremental
    :param heat: boolean; Allow heat wave effect
    :return: Return a pygame surface 24 bit without per-pixel information,
    surface with same size as the light texture. Represent the lit surface.
    """

    assert intensity >= <float>0.0, '\nIntensity value cannot be > 0.0'


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
    if (x < <unsigned short int>0) or (x > w - <unsigned short int>1) or \
            (y < <unsigned short int>0) or (y > h - <unsigned short int>1):
        return Surface((ax, ay), SRCALPHA), ax, ay

    # return an empty Surface when intensity = 0.0
    if intensity == <float>0.0:
        return Surface((ax, ay), SRCALPHA), ax, ay

    lx = ax >> 1
    ly = ay >> 1

    cdef:
        np.ndarray[np.uint8_t, ndim=3] rgb = empty((ax, ay, <unsigned short int>3), uint8)
        np.ndarray[np.uint8_t, ndim=2] alpha = empty((ax, ay), uint8)
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
        for i in prange(ax, schedule=SCHEDULE, num_threads=THREADS):
            for j in range(ay):
                f = alpha[i, j] * <float>ONE_255 * intensity
                new_array[j, i, <unsigned short int>0] =\
                    <unsigned char>fmin(rgb[i, j, <unsigned short int>0] * f *
                                        color[<unsigned short int>0], <float>255.0)
                new_array[j, i, <unsigned short int>1] =\
                    <unsigned char>fmin(rgb[i, j, <unsigned short int>1] * f *
                                        color[<unsigned short int>1], <float>255.0)
                new_array[j, i, <unsigned short int>2] =\
                    <unsigned char>fmin(rgb[i, j, <unsigned short int>2] * f *
                                        color[<unsigned short int>2], <float>255.0)

    ay, ax = new_array.shape[:2]

    if smooth:
        blur_array_inplace_c(new_array, mask=None, t=<unsigned short int>1)

    if saturation:
        saturation_inplace_c(new_array, sat_value)

    if heat:
        new_array = heatwave_array24_horiz_c(numpy.asarray(new_array).transpose(
            <unsigned short int>1, <unsigned short int>0, <unsigned short int>2),
            alpha, frequency, (frequency % <unsigned short int>8) / <float>1000.0,
            attenuation=<unsigned int>100, threshold=<unsigned short int>10)

    surface = pygame.image.frombuffer(new_array, (ax, ay), "RGB")

    if bloom:
        mask = array2d_normalized_c(alpha)
        bloom_array24_c(surface, threshold_=threshold, fast_=False, mask_=mask)

    return surface, ax, ay




@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cpdef inline object bpf24_c2(image, int threshold = 128, bint transpose=False):
    """
    BRIGHT PASS FILTER COMPATIBLE 24-BIT

    Bright pass filter for 24bit image (method using 3d array data structure)
    Calculate the luminance of every pixels and applied an attenuation c = lum2 / lum
    with lum2 = max(lum - threshold, 0) and
    lum = rgb[i, j, 0] * 0.299 + rgb[i, j, 1] * 0.587 + rgb[i, j, 2] * 0.114
    The output image will keep only bright areas. You can adjust the threshold value
    default 128 in order to get the desire changes.

    :param transpose: Transpose the final array (width and height are transpose if True)
    :param image: pygame.Surface 24 bit format (RGB)  without per-pixel information
    :param threshold: integer; Threshold to consider for filtering pixels luminance values,
    default is 128 range [0..255] unsigned char (python integer)
    :return: Return a Pygame Surface and a 3d numpy.ndarray format (w, h, 3)
    (only bright area of the image remains).
    """

    # Fallback to default threshold value if argument
    # threshold value is incorrect
    if 0 > threshold > 255:
        printf("\nArgument threshold must be in range [0...255], fallback to default value 128.")
        threshold = 128

    assert isinstance(image, pygame.Surface), \
           "\nExpecting pygame surface for argument image, got %s " % type(image)

    try:
        rgb_array = image.get_view('3')
    except (pygame.error, ValueError):
        raise ValueError('\nInvalid surface.')

    cdef:
        int w, h
    w, h = rgb_array.shape[:2]

    # check sizes
    assert w>0 and h>0,\
        'Incorrect surface dimensions should be (w>0, h>0) got (w:%s, h:%s)' % (w, h)


    cdef:
        unsigned char [:, :, :] rgb = rgb_array
        unsigned char [:, :, ::1] out_rgb= numpy.empty((w, h, 3), numpy.uint8)
        unsigned char [:, :, ::1] out_rgb_transposed = numpy.empty((h, w, 3), numpy.uint8)
        int i = 0, j = 0
        float lum, c
        unsigned char *r
        unsigned char *g
        unsigned char *b

    if transpose is not None and transpose==True:
        with nogil:
            for j in prange(0, h, schedule=SCHEDULE, num_threads=THREADS):
                for i in range(0, w):
                    # ITU-R BT.601 luma coefficients
                    r = &rgb[i, j, 0]
                    g = &rgb[i, j, 1]
                    b = &rgb[i, j, 2]

                    lum = r[0] * <float>0.299 + g[0] * <float>0.587 + b[0] * <float>0.114

                    if lum > threshold:
                        c = (lum - threshold) / lum
                        out_rgb_transposed[j, i, 0] = <unsigned char>(r[0] * c)
                        out_rgb_transposed[j, i, 1] = <unsigned char>(g[0] * c)
                        out_rgb_transposed[j, i, 2] = <unsigned char>(b[0] * c)
                    else:
                        out_rgb_transposed[j, i, 0] = 0
                        out_rgb_transposed[j, i, 1] = 0
                        out_rgb_transposed[j, i, 2] = 0

        return frombuffer(out_rgb_transposed, (w, h), 'RGB'), out_rgb_transposed
    else:
        with nogil:
            for j in prange(0, h, schedule=SCHEDULE, num_threads=THREADS):
                for i in range(0, w):
                    r = &rgb[i, j, 0]
                    g = &rgb[i, j, 1]
                    b = &rgb[i, j, 2]
                    # ITU-R BT.601 luma coefficients
                    lum = r[0] * <float>0.299 + g[0] * <float>0.587 + b[0] * <float>0.114
                    if lum > threshold:
                        c = (lum - threshold) / lum
                        out_rgb[i, j, 0] = <unsigned char>(r[0] * c)
                        out_rgb[i, j, 1] = <unsigned char>(g[0] * c)
                        out_rgb[i, j, 2] = <unsigned char>(b[0] * c)
                    else:
                        out_rgb[i, j, 0], out_rgb[i, j, 1], out_rgb[i, j, 2] = 0, 0, 0

        return frombuffer(out_rgb, (w, h), 'RGB'), out_rgb



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cpdef inline object bloom_effect_array24_c2(
        object surface_, unsigned char threshold_,
        int smooth_=1, mask_=None, bint fast_ = False):
    """

    CREATE A BLOOM EFFECT ON A PYGAME.SURFACE (COMPATIBLE 24 BIT SURFACE)

    This method is using array structure.

    definition:
        Bloom is a computer graphics effect used in video games, demos,
        and high dynamic range rendering to reproduce an imaging artifact of real-world cameras.

    1)First apply a bright pass filter to the pygame surface(SDL surface) using methods
      bpf24_b_c  (adjust the threshold value to get the best filter effect).
    2)Downside the newly created bpf image by factor x2, x4, x8, x16 using the pygame scale method (no need to
      use smoothscale (bilinear filtering method).
    3)Apply a Gaussian blur 5x5 effect on each of the downsized bpf images (if smooth_ is > 1, then the Gaussian
      filter 5x5 will by applied more than once. Note, this have little effect on the final image quality.
    4)Re-scale all the bpf images using a bilinear filter (width and height of original image).
      Using an un-filtered rescaling method will pixelate the final output image.
      For best performances sets smoothscale acceleration.
      A value of 'GENERIC' turns off acceleration. 'MMX' uses MMX instructions only.
      'SSE' allows SSE extensions as well.
    5)Blit all the bpf images on the original surface, use pygame additive blend mode for
      a smooth and brighter effect.

    Notes:
    The downscaling process of all sub-images could be done in a single process to increase performance.

    :param fast_: bool; True | False. Speed up the bloom process using only the x16 surface and using
    an optimized bright pass filter (texture size downscale x4 prior processing)

    :param mask_:
    :param surface_: pygame.Surface 24-bit format surface
    :param threshold_: integer; Threshold value used by the bright pass algorithm (default 128)
    :param smooth_: Number of Gaussian blur 5x5 to apply to downside images.
    :return : Returns a pygame.Surface with a bloom effect (24 bit surface)


    """
    # todo mask_ doc

    surface_cp = surface_.copy()

    assert smooth_ > 0, \
        "Argument smooth_ must be > 0, got %s " % smooth_
    assert -1 < threshold_ < 256, \
        "Argument threshold_ must be in range [0...255] got %s " % threshold_

    cdef:
        int w, h, bit_size
        int w2, h2, w4, h4, w8, h8, w16, h16
        bint x2, x4, x8, x16 = False

    w, h = surface_.get_size()
    bit_size = surface_.get_bitsize()

    with nogil:
        w2, h2 = w >> 1, h >> 1
        w4, h4 = w2 >> 1, h2 >> 1
        w8, h8 = w4 >> 1, h4 >> 1
        w16, h16 = w8 >> 1, h8 >> 1

    with nogil:
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

    # check if the first reduction is > 0
    # if not we cannot blur that surface (too small)
    if not x2:
        return pygame.Surface((1, 1))

    if fast_:

        x2, x4, x8 = False, False, False

        s4 = scale(surface_, (w >> 2, h >> 2))
        bpf_surface, bpf_array = bpf24_c2(s4, threshold=threshold_, transpose=True)
        bpf_surface = scale(bpf_surface, (w, h))

    else:
        # BRIGHT PASS FILTER
        bpf_surface, bpf_array = bpf24_c2(surface_, threshold=threshold_, transpose=True)

    if x2:
        s2 = scale(bpf_surface, (w2, h2))
        s2_array = numpy.array(s2.get_view('3'), dtype=numpy.uint8).transpose(1, 0, 2)
        if smooth_ > 1:
            for r in range(smooth_):
                s2_array = blur5x5_array24_c2(s2_array)
        else:
            s2_array = blur5x5_array24_c2(s2_array)
        b2_blurred = frombuffer(s2_array, (w2, h2), 'RGB')
        s2 = smoothscale(b2_blurred, (w, h))
        surface_cp.blit(s2, (0, 0), special_flags=BLEND_RGB_ADD)

    if x4:
        s4 = scale(bpf_surface, (w4, h4))
        s4_array = numpy.array(s4.get_view('3'), dtype=numpy.uint8).transpose(1, 0, 2)
        if smooth_ > 1:
            for r in range(smooth_):
                s4_array = blur5x5_array24_c2(s4_array)
        else:
            s4_array = blur5x5_array24_c2(s4_array)
        b4_blurred = frombuffer(s4_array, (w4, h4), 'RGB')
        s4 = smoothscale(b4_blurred, (w, h))
        surface_cp.blit(s4, (0, 0), special_flags=BLEND_RGB_ADD)

    if x8:
        s8 = scale(bpf_surface, (w8, h8))
        s8_array = numpy.array(s8.get_view('3'), dtype=numpy.uint8).transpose(1, 0, 2)
        if smooth_ > 1:
            for r in range(smooth_):
                s8_array = blur5x5_array24_c2(s8_array)
        else:
            s8_array = blur5x5_array24_c2(s8_array)
        b8_blurred = frombuffer(s8_array, (w8, h8), 'RGB')
        s8 = smoothscale(b8_blurred, (w, h))
        surface_cp.blit(s8, (0, 0), special_flags=BLEND_RGB_ADD)

    if x16:
        s16 = scale(bpf_surface, (w16, h16))
        s16_array = numpy.array(s16.get_view('3'), dtype=numpy.uint8).transpose(1, 0, 2)
        if smooth_ > 1:
            for r in range(smooth_):
                s16_array = blur5x5_array24_c2(s16_array)
        else:
            s16_array = blur5x5_array24_c2(s16_array)
        b16_blurred = frombuffer(s16_array, (w16, h16), 'RGB')
        s16 = smoothscale(b16_blurred, (w, h))
        surface_cp.blit(s16, (0, 0), special_flags=BLEND_RGB_ADD)

    if mask_ is not None:
        # Multiply mask surface pixels with mask values.
        # RGB pixels = 0 when mask value = 0.0, otherwise
        # modify RGB amplitude
        surface_cp = filtering24_c(surface_cp, mask_)

    return surface_cp



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cpdef area24_cc(
        int x, int y,
        np.ndarray[np.uint8_t, ndim=3] background_rgb,
        np.ndarray[np.uint8_t, ndim=2] mask_alpha, float intensity=1.0,
        float [::1] color=numpy.array([128.0, 128.0, 128.0], dtype=numpy.float32, copy=False),
        bint smooth=False, bint saturation=False, float sat_value=0.2, bint bloom=False,
        unsigned char bloom_threshold=64
):
    """
    CREATE A REALISTIC LIGHT EFFECT ON A PYGAME.SURFACE OR TEXTURE.

    Modes definition
    ================
    SMOOTH : Apply a Gaussian blur with kernel 5x5 over the output texture, 
    the light effect will be slightly blurred.
   
    SATURATION : Create a saturation effect (increase of the texture lightness using 
    HSL color conversion algorithm. saturation threshold value should be included 
    in range [-1.0, 1.0] default is 0.2
    Saturation above 0.5 will deteriorate the output coloration. Threshold value below zero will
    greyscale output texture.
    
    BLOOM: Create a bloom effect to the output texture (using blend method)

    intensity:
    Intensity is a float value defining how bright will be the light effect.
    If intensity is zero, a new pygame.Surface is returned with RLEACCEL flag (empty surface)

    Color allows you to change the light coloration, if omitted, the light color by default is
    R = 128.0, G = 128.0 and B = 128.0

    :param x: integer, light x coordinates (must be in range [0..max screen.size x]
    :param y: integer, light y coordinates (must be in range [0..max screen size y]
    :param background_rgb: numpy.ndarray (w, h, 3) uint8. 3d array shape containing all RGB values
    of the background surface (display background).
    :param mask_alpha: numpy.ndarray (w, h) uint8, 2d array with light texture alpha values.
    For better appearances, choose a texture with a radial mask shape (maximum light 
    intensity in the center)
    :param color: numpy.array; Light color (RGB float), default
    array([128.0 / 255.0, 128.0 / 255.0, 128.0 / 255.0], float32, copy=False)
    :param intensity: float; Light intensity range [0.0 ... 20.0]
    :param bloom: boolean; Bloom effect, default False
    :param bloom_threshold:unsigned char;
    :param sat_value: float; Set the saturation value
    :param saturation: boolean; Saturation effect
    :param smooth: boolean; Blur effect

    :return: Return a pygame surface 24 bit without per-pixel information,
    surface with same size as the light texture. Represent the lit surface.
    """

    if intensity < 0.0:
        raise ValueError('\nIntensity value cannot be < 0.0')

    cdef int w, h, lx, ly, ax, ay
    try:
        w, h = background_rgb.shape[:2]
    except (ValueError, pygame.error) as e:
        raise ValueError('\nArray shape not understood.')

    try:
        ax, ay = mask_alpha.shape[:2]
    except (ValueError, pygame.error) as e:
        raise ValueError('\nArray shape not understood.')

    # return an empty Surface when intensity = 0.0
    if intensity == 0:
        return Surface((ax, ay), SRCALPHA), ax, ay

    lx = ax >> 1
    ly = ay >> 1

    cdef:
        int i=0, j=0
        float f
        unsigned int w_low = lx
        unsigned int w_high = lx
        unsigned int h_low = ly
        unsigned int h_high = ly
        unsigned int x1, x2, y1, y2

    with nogil:
        if x < lx:
            w_low = x
        elif x > w - lx:
            w_high = w - x

        if y < ly:
            h_low = y
        elif y >  h - ly:
            h_high = h - y

        x1 = x - w_low
        x2 = x + w_high
        y1 = y - h_low
        y2 = y + h_high
        x1 = max(x1, <unsigned short int>0)
        x1 = min(x1, w)
        x2 = max(x2, <unsigned short int>0)
        x2 = min(x2, w)
        y1 = max(y1, <unsigned short int>0)
        y1 = min(y1, h)
        y2 = max(y2, <unsigned short int>0)
        y2 = min(y2, h)

    # RGB block and ALPHA
    cdef:
        unsigned char [:, : , :] rgb = background_rgb[x1:x2, y1:y2, :]
        unsigned char [:, :] alpha = mask_alpha[lx - w_low:lx + w_high, ly - h_low:ly + h_high]

    # RGB ARRAY IS TRANSPOSED IN THE LOOP
    ax, ay = rgb.shape[:2]
    cdef:
        unsigned char [:, :, ::1] new_array = \
            numpy.ascontiguousarray(empty((ay, ax, 3), numpy.uint8))
        float c1 = <float>ONE_255 * intensity
        float red   = color[0]
        float green = color[1]
        float blue  = color[2]

    with nogil:

        for j in prange(ay, schedule=SCHEDULE, num_threads=THREADS):
            for i in range(ax):
                f = alpha[i, j] * c1
                new_array[j, i, 0] = <unsigned char>min(rgb[i, j, 0] * f * red, 255)
                new_array[j, i, 1] = <unsigned char>min(rgb[i, j, 1] * f * green, 255)
                new_array[j, i, 2] = <unsigned char>min(rgb[i, j, 2] * f * blue, 255)

    # As the array is transposed we
    # we need to adjust ax and ay (swapped).
    ay, ax = new_array.shape[:2]

    # Return an empty surface if the x or y are not within the normal range.
    if ax <1 or ay < 1:
        return Surface((ax, ay), SRCALPHA), ax if ax > 0 else 0, ay if ay > 0 else 0

    if saturation:
        saturation_array_mask_inplace(
            new_array,
            sat_value,
            alpha,
            swap_row_column=True
        )

    cdef unsigned char [:, :, :] n_cp =\
        numpy.array(new_array, copy=False, dtype=uint8)

    if bloom:
        # surf = bpf24_c(new_array, threshold = bloom_threshold)
        # blend_add_array_c(new_array, surf.get_view('3'))

        bpf24_inplace_c(new_array, threshold=bloom_threshold)
        blend_add_array_c(new_array, n_cp)


    if smooth:
        blur_array_inplace_c(new_array, mask=None, t=1)

    surface = frombuffer(new_array, (ax, ay), 'RGB')
    surface.set_colorkey((0, 0, 0, 0), pygame.RLEACCEL)

    return surface, ax, ay



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
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
        rgb_array = surface_.get_view('3')

    except Exception as e:
        raise ValueError("\nCannot reference source pixels into a 3d array.\n %s " % e)

    cdef:
        unsigned char [:, :, ::1] new_array = \
            numpy.ascontiguousarray(empty((h, w, 3), dtype=numpy.uint8))
        int i = 0, j = 0
        float dw = <float>delta_y / <float>w
        float dh = <float>delta_x / <float>h
        float nx, ny, theta_rad, dist, new_dist, cos_, sin_
        unsigned int new_j, new_i

    with nogil:
        for j in prange(0, h, schedule=SCHEDULE, num_threads=THREADS):
            ny = <float> (<float> j / <float> w) - dw
            for i in range(w):
                nx = <float>(<float>i / <float>h) - dh

                theta_rad = <float>atan2 (ny,nx)
                cos_ = <float>cos(<float>theta_rad)
                sin_ = <float>sin(<float>theta_rad)

                dist = <float>sqrt(nx * nx + ny * ny)
                new_dist = dist * (zoom - fx)

                new_j = <unsigned int>((sin_ * new_dist + dw) * <float>w)
                new_i = <unsigned int>((cos_ * new_dist + dh) * <float>h)
                new_array[j, i, 0] = rgb_array[new_i, new_j, 0]

                new_dist = dist * (zoom  - fx * <float>2.0)

                new_i = <unsigned int>((cos_ * new_dist + dh) * <float>h)
                new_j = <unsigned int> ((sin_ * new_dist + dw) * <float> w)
                new_array[j, i, 1] = rgb_array[new_i, new_j, 1]

                new_dist = dist * (zoom  - fx * <float>3.0)

                new_j = <unsigned int>((sin_ * new_dist + dw) * <float>w)
                new_i = <unsigned int>((cos_ * new_dist + dh) * <float>h)

                new_array[j, i, 2] = rgb_array[new_i, new_j, 2]

    return frombuffer(new_array, (w, h), 'RGB')



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cpdef object chromatic_inplace(
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
        # rgb_array = numpy.array(pixels3d(surface_), copy=True, dtype=uint8)
        rgb_array = numpy.array(surface_.get_view('3'), copy=False, dtype=uint8, order='F')
    except Exception as e:
        raise ValueError("\nCannot reference source pixels into a 3d array.\n %s " % e)

    cdef:
        unsigned char [:, :, :] new_array = surface_.get_view('3')
        int i = 0, j = 0
        float dw = <float>delta_y / <float>w
        float dh = <float>delta_x / <float>h
        float nx, ny, theta_rad, dist, new_dist, cos_, sin_
        unsigned int new_j, new_i

    with nogil:
        for j in prange(0, h, schedule=SCHEDULE, num_threads=THREADS):
            ny = <float> (<float> j / <float> w) - dw
            for i in range(w):
                nx = <float>(<float>i / <float>h) - dh

                theta_rad = <float>atan2 (ny,nx)
                cos_ = <float>cos(<float>theta_rad)
                sin_ = <float>sin(<float>theta_rad)

                dist = <float>sqrt(nx * nx + ny * ny)
                new_dist = dist * (zoom - fx)

                new_j = <unsigned int>((sin_ * new_dist + dw) * <float>w)
                new_i = <unsigned int>((cos_ * new_dist + dh) * <float>h)
                new_array[i, j, 0] = rgb_array[new_i, new_j, 0]

                new_dist = dist * (zoom  - fx * <float>2.0)

                new_i = <unsigned int>((cos_ * new_dist + dh) * <float>h)
                new_j = <unsigned int>((sin_ * new_dist + dw) * <float> w)
                new_array[i, j, 1] = rgb_array[new_i, new_j, 1]

                new_dist = dist * (zoom  - fx * <float>3.0)

                new_j = <unsigned int>((sin_ * new_dist + dw) * <float>w)
                new_i = <unsigned int>((cos_ * new_dist + dh) * <float>h)

                new_array[i, j, 2] = rgb_array[new_i, new_j, 2]





@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
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
        rgb_array = surface_.get_view('3')

    except Exception as e:
        raise ValueError("\nCannot reference source pixels into a 3d array.\n %s " % e)

    cdef:
        # unsigned char [:, :, ::1] new_array = \
        #     numpy.ascontiguousarray(empty((h, w, 3), dtype=numpy.uint8))
        unsigned char [:, :, ::1] new_array = \
                    numpy.ascontiguousarray(empty((h, w, 3), dtype=numpy.uint8))
        int i = 0, j = 0
        float dw = delta_y / <float>w
        float dh = delta_x / <float>h
        float nx, ny, theta, nx2, ny2, dist, new_dist
        unsigned int new_j, new_i, r, g, b

    with nogil:
        for j in prange(0, h, schedule=SCHEDULE, num_threads=THREADS):
            ny = <float> (<float> j / <float> w) - dw
            for i in range(w):
                nx = <float>(<float>i / <float>h) - dh

                theta = <float>atan2 (ny,nx)

                new_dist = <float>sqrt(nx * nx + ny * ny) * zx

                new_j = <unsigned int>((<float>sin(<float>theta) * new_dist + dw) * <float>w)
                new_i = <unsigned int>((<float>cos(<float>theta) * new_dist + dh) * <float>h)

                new_array[j, i, 0] = rgb_array[new_i, new_j, 0]
                new_array[j, i, 1] = rgb_array[new_i, new_j, 1]
                new_array[j, i, 2] = rgb_array[new_i, new_j, 2]


    return frombuffer(new_array, (w, h), 'RGB')



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cpdef void Luma_GreyScale(object surface_):
    """
    CONVERT IMAGE INTO GREYSCALE USING YIQ (LUMA INFORMATION)

    :param surface_: pygame.Surface;
    :return: void
    """
    cdef unsigned char [:,:,:] arr = surface_.get_view('3')
    Luma_GreyScale_c(arr)



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cpdef void RGB_TO_YIQ_Q0_inplace(object surface_):
    """
    CONVERT IMAGE INTO YIQ MODEL (REPRESENT IN-PHASE VALUE)
    Final YIQ model without the quadrature value Q = 0

    In YIQ the Y component represents the luma information,
    I and Q represent the chrominance information.
    I stands for in-phase, while Q stands for quadrature, referring to the components
    used in quadrature amplitude modulation.

    :param surface_: pygame.Surface;
    :return: void
    """
    cdef unsigned char [:,:,:] arr = surface_.get_view('3')
    RGB_TO_YIQ_Q0_inplace_c(arr)


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cpdef void RGB_TO_YIQ_I0_inplace(object surface_):
    """
    CONVERT IMAGE INTO YIQ MODEL
    Final YIQ model without the in phase value I = 0

    :param surface_: pygame.Surface;
    :return: void
    """
    cdef unsigned char [:,:,:] arr = surface_.get_view('3')
    RGB_TO_YIQ_I0_inplace_c(arr)

@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cpdef void RGB_TO_YIQ_Y0_inplace(object surface_):
    """
    CONVERT IMAGE INTO YIQ MODEL
    Final YIQ model without the luma value Y = 0

    :param surface_: pygame.Surface;
    :return: void
    """
    cdef unsigned char [:,:,:] arr = surface_.get_view('3')
    RGB_TO_YIQ_Y0_inplace_c(arr)

# -------------------------------------------------------------------------------------------------------------------




@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
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
        for x in prange(0, w, schedule=SCHEDULE, num_threads=THREADS):

            for y in range(h):
                distortion = sin(x * attenuation + frequency) * amplitude * mask_array[x, y]

                xx = <int>(x  + distortion + rand() * <float>0.0002)
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



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cdef inline saturation_array_mask(
        unsigned char [:, :, :] array_,
        float shift_,
        unsigned char [:, :] mask_array,
        bint swap_row_column
):
    """
    CHANGE THE SATURATION LEVEL OF A PYGAME.SURFACE (COMPATIBLE WITH 24BIT ONLY).

    Transform RGB model into HSL model and <shift_> saturation value.
    Optional mask_array to determine area to be modified.
    The mask should be a 2d array filled with float values

    :param array_: 3d numpy.ndarray shapes (w, h, 3) representing a 24bit format pygame.Surface.
    :param shift_: Value must be in range [-1.0 ... 1.0],
                   between [-1.0 ... 0.0] decrease saturation.
                   between [0.0  ... 1.0] increase saturation.
    :param mask_array: unsigned char numpy.ndarray shape (width, height) 
    :param swap_row_column: swap row and column values (only apply to array_) 
    :return: a pygame.Surface 24-bit without per-pixel information 

    """

    assert -1.0 <= shift_ <= 1.0, 'Argument shift_ must be in range [-1.0 .. 1.0].'

    cdef int width, height
    try:
        if swap_row_column:
            height, width = array_.shape[:2]
        else:
            width, height = array_.shape[:2]
    except (ValueError, pygame.error):
        raise ValueError(
            '\nArray type not compatible, expecting MemoryViewSlice got %s ' % type(array_))

    cdef:
        unsigned char *r
        unsigned char *g
        unsigned char *b
        float s
        hsl hsl_
        rgb rgb_
        int i, j

    with nogil:
        for i in prange(width, schedule=SCHEDULE, num_threads=THREADS):
            for j in range(height):

                if mask_array[i, j] == 0:
                    continue

                # load pixel RGB values
                r = &array_[j, i, 0]
                g = &array_[j, i, 1]
                b = &array_[j, i, 2]

                hsl_ = struct_rgb_to_hsl(
                    r[0] * <float>ONE_255,
                    g[0] * <float>ONE_255,
                    b[0] * <float>ONE_255
                )
                s = min((hsl_.s + shift_), <float>1.0)
                s = max(s, <float>0.0)

                rgb_ = struct_hsl_to_rgb(hsl_.h, s, hsl_.l)

                r[0] = <unsigned char> (rgb_.r * <float>255.0)
                g[0] = <unsigned char> (rgb_.g * <float>255.0)
                b[0] = <unsigned char> (rgb_.b * <float>255.0)

    return frombuffer(array_, (width, height), 'RGB')



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cdef inline void saturation_array_mask_inplace(
        unsigned char [:, :, :] array_,
        float shift_,
        unsigned char [:, :] mask_array,
        bint swap_row_column
):
    """
    CHANGE THE SATURATION LEVEL OF A PYGAME.SURFACE (COMPATIBLE WITH 24BIT ONLY).

    Transform RGB model into HSL model and <shift_> saturation value.
    Optional mask_array to determine area to be modified.
    The mask should be a 2d array filled with float values

    :param array_: 3d numpy.ndarray shapes (w, h, 3) representing a 24bit format pygame.Surface.
    :param shift_: Value must be in range [-1.0 ... 1.0],
                   between [-1.0 ... 0.0] decrease saturation.
                   between [0.0  ... 1.0] increase saturation.
    :param mask_array: unsigned char numpy.ndarray shape (width, height) 
    :param swap_row_column: swap row and column values (only apply to array_) 
    :return: a pygame.Surface 24-bit without per-pixel information 

    """

    assert -1.0 <= shift_ <= 1.0, 'Argument shift_ must be in range [-1.0 .. 1.0].'

    cdef int w, h
    try:
        if swap_row_column:
            h, w = array_.shape[:2]
        else:
            w, h = array_.shape[:2]
    except (ValueError, pygame.error):
        raise ValueError(
            '\nArray type not compatible, '
            'expecting MemoryViewSlice got %s ' % type(array_))

    cdef:
        unsigned char *r
        unsigned char *g
        unsigned char *b
        float s
        hsl hsl_
        rgb rgb_
        int i, j

    with nogil:
        for i in prange(w, schedule=SCHEDULE, num_threads=THREADS):
            for j in range(h):

                if mask_array[i, j] == 0:
                     continue

                r, g, b = \
                    &array_[ j, i, 0 ], \
                    &array_[ j, i, 1 ], \
                    &array_[ j, i, 2 ]

                hsl_ = struct_rgb_to_hsl(
                    <float> r[ 0 ] * <float> ONE_255,
                    <float> g[ 0 ] * <float> ONE_255,
                    <float> b[ 0 ] * <float> ONE_255
                )

                s = min((hsl_.s + shift_), <float> 1.0)
                s = max(s, <float> 0.0)

                rgb_ = struct_hsl_to_rgb(hsl_.h, s, hsl_.l)

                r[ 0 ] = <unsigned char> (rgb_.r * <float> 255.0)
                g[ 0 ] = <unsigned char> (rgb_.g * <float> 255.0)
                b[ 0 ] = <unsigned char> (rgb_.b * <float> 255.0)


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cdef inline unsigned char [:, :, ::1] blur5x5_array24_c2(
        unsigned char [:, :, :] rgb_array_, mask=None):
    """
    # Gaussian kernel 5x5
        # |1   4   6   4  1|
        # |4  16  24  16  4|
        # |6  24  36  24  6|  x 1/256
        # |4  16  24  16  4|
        # |1  4    6   4  1|

    This method is using convolution property and process the image in two passes,
    first the horizontal convolution and last the vertical convolution
    pixels convoluted outside image edges will be set to adjacent edge value

    :param mask: default None
    :param rgb_array_: numpy.ndarray type (w, h, 3) uint8 
    :return: Return 24-bit a numpy.ndarray type (w, h, 3) uint8
    """


    cdef int w, h, dim
    try:
        w, h, dim = (<object>rgb_array_).shape[:3]

    except (ValueError, pygame.error) as e:
        raise ValueError('\nArray shape not understood.')

    cdef:
        # float [::1] kernel = kernel_
        # float[5] kernel = [1.0/16.0, 4.0/16.0, 6.0/16.0, 4.0/16.0, 1.0/16.0]
        short int kernel_half = 2
        unsigned char [:, :, ::1] convolve = numpy.empty((w, h, 3), dtype=uint8)
        unsigned char [:, :, ::1] convolved = numpy.empty((w, h, 3), dtype=uint8)
        short int kernel_length = <int>len(GAUSS_KERNEL)
        int x, y, xx, yy
        float k, r, g, b, s
        char kernel_offset
        unsigned char red, green, blue

    with nogil:
        # horizontal convolution
        for y in prange(0, h, schedule=SCHEDULE, num_threads=THREADS):  # range [0..h-1)

            for x in range(0, w):  # range [0..w-1]

                r, g, b = 0, 0, 0

                for kernel_offset in range(-kernel_half, kernel_half + 1):

                    k = GAUSS_KERNEL[kernel_offset + kernel_half]

                    xx = x + kernel_offset

                    # check boundaries.
                    # Fetch the edge pixel for the convolution
                    if xx < 0:
                        red, green, blue = rgb_array_[0, y, 0],\
                        rgb_array_[0, y, 1], rgb_array_[0, y, 2]
                    elif xx > (w - 1):
                        red, green, blue = rgb_array_[w-1, y, 0],\
                        rgb_array_[w-1, y, 1], rgb_array_[w-1, y, 2]
                    else:
                        red, green, blue = rgb_array_[xx, y, 0],\
                            rgb_array_[xx, y, 1], rgb_array_[xx, y, 2]

                    r = r + red * k
                    g = g + green * k
                    b = b + blue * k

                convolve[x, y, 0], convolve[x, y, 1], convolve[x, y, 2] = <unsigned char>r,\
                    <unsigned char>g, <unsigned char>b

        # Vertical convolution
        for x in prange(0,  w, schedule=SCHEDULE, num_threads=THREADS):

            for y in range(0, h):
                r, g, b = 0, 0, 0

                for kernel_offset in range(-kernel_half, kernel_half + 1):

                    k = GAUSS_KERNEL[kernel_offset + kernel_half]
                    yy = y + kernel_offset

                    if yy < 0:
                        red, green, blue = convolve[x, 0, 0],\
                        convolve[x, 0, 1], convolve[x, 0, 2]
                    elif yy > (h -1):
                        red, green, blue = convolve[x, h-1, 0],\
                        convolve[x, h-1, 1], convolve[x, h-1, 2]
                    else:
                        red, green, blue = convolve[x, yy, 0],\
                            convolve[x, yy, 1], convolve[x, yy, 2]

                    r = r + red * k
                    g = g + green * k
                    b = b + blue * k

                convolved[x, y, 0], convolved[x, y, 1], convolved[x, y, 2] = \
                    <unsigned char>r, <unsigned char>g, <unsigned char>b

    return convolved






@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cdef inline void Luma_GreyScale_c(unsigned char [:, :, :] rgb_array):

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
        for j in prange(h, schedule=SCHEDULE, num_threads=THREADS):
            for i in range(w):
                r = &rgb_array[i, j, 0]
                g = &rgb_array[i, j, 1]
                b = &rgb_array[i, j, 2]
                yiq_ = rgb_to_yiq(r[0] * <float>ONE_255, g[0] * <float>ONE_255, b[0] * <float>ONE_255)

                r[0] = <unsigned char>min(<unsigned char>(yiq_.y * <float>255.0), <unsigned char>255)
                g[0] = <unsigned char>min(<unsigned char>(yiq_.y * <float>255.0), <unsigned char>255)
                b[0] = <unsigned char>min(<unsigned char>(yiq_.y * <float>255.0), <unsigned char>255)


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cdef inline void RGB_TO_YIQ_Q0_inplace_c(unsigned char [:, :, :] rgb_array):

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
        for j in prange(h, schedule=SCHEDULE, num_threads=THREADS):
            for i in range(w):
                r = &rgb_array[i, j, 0]
                g = &rgb_array[i, j, 1]
                b = &rgb_array[i, j, 2]
                yiq_ = rgb_to_yiq(r[0] * <float>ONE_255, g[0] * <float>ONE_255, b[0] * <float>ONE_255)
                rgb_ = yiq_to_rgb(yiq_.y, yiq_.i, 0)
                r[0] = <unsigned char>(rgb_.r * <float>255.0)
                g[0] = <unsigned char>(rgb_.g * <float>255.0)
                b[0] = <unsigned char>(rgb_.b * <float>255.0)



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cdef inline void RGB_TO_YIQ_I0_inplace_c(unsigned char [:, :, :] rgb_array):

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
        for j in prange(h, schedule=SCHEDULE, num_threads=THREADS):
            for i in range(w):
                r = &rgb_array[i, j, 0]
                g = &rgb_array[i, j, 1]
                b = &rgb_array[i, j, 2]
                yiq_ = rgb_to_yiq(r[0] * <float>ONE_255, g[0] * <float>ONE_255, b[0] * <float>ONE_255)
                rgb_ = yiq_to_rgb(yiq_.y, 0, yiq_.q)
                r[0] = <unsigned char>(rgb_.r * <float>255.0)
                g[0] = <unsigned char>(rgb_.g * <float>255.0)
                b[0] = <unsigned char>(rgb_.b * <float>255.0)



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cdef inline void RGB_TO_YIQ_Y0_inplace_c(unsigned char [:, :, :] rgb_array):

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
        for j in prange(h, schedule=SCHEDULE, num_threads=THREADS):
            for i in range(w):
                r = &rgb_array[i, j, 0]
                g = &rgb_array[i, j, 1]
                b = &rgb_array[i, j, 2]
                yiq_ = rgb_to_yiq(r[0] * <float>ONE_255, g[0] * <float>ONE_255, b[0] * <float>ONE_255)
                rgb_ = yiq_to_rgb(0, yiq_.i, yiq_.q)
                r[0] = <unsigned char>(rgb_.r * <float>255.0)
                g[0] = <unsigned char>(rgb_.g * <float>255.0)
                b[0] = <unsigned char>(rgb_.b * <float>255.0)

