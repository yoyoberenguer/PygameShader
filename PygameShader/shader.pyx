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
"""

__VERSION__ = "1.0.2"

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


from PygameShader.gaussianBlur5x5 import canny_blur5x5_surface24_c

from libc.stdlib cimport rand, malloc
from libc.math cimport sqrt, atan2, sin, cos, exp, round, pow, floor
from libc.stdio cimport printf

cdef float M_PI = 3.14159265358979323846
cdef float M_PI2 =3.14159265358979323846/2.0
cdef float M_2PI =2 * 3.14159265358979323846
cdef float RAD_TO_DEG=<float>(180.0/M_PI)
cdef float DEG_TO_RAD=<float>(M_PI/180.0)

cdef int THREADS = 4

# TODO PIXELATE
# TODO FUNCTION TESTING IMAGE


cpdef inline void shader_rgb_to_bgr_inplace(object surface_):
    """  
    SHADER RGB to BGR
  
    Convert your game display from RGB to BGR format
    This algorithm can also be used to transform pygame texture in the equivalent bgr format
    
    * The changes are automatically applied inplace to the surface you do not need to create a 
    new surface.

    e.g:
    shader_rgb_to_bgr_inplace(surface)

    :param surface_    : Pygame surface or display surface compatible (image 24-32 bit with or 
                         without per-pixel transparency / alpha channel)
    :return             : void
    """
    assert isinstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)
    shader_rgb_to_bgr_inplace_c(pixels3d(surface_))


cpdef inline void shader_rgb_to_brg_inplace(object surface_):
    """
    SHADER RGB TO BRG

    Convert your game display from RGB to BRG format.
    This algorithm can also be used to transform pygame texture in the equivalent BRG format
    
    * The changes are automatically applied inplace to the surface you do not need to create a 
      new surface.
    
    e.g:
    shader_rgb_to_brg_inplace(surface)

    :param surface_: Pygame surface or display surface compatible (image 24-32 bit with or without 
                     per-pixel transparency / alpha channel)
    :return: void 
    """
    assert isinstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)
    shader_rgb_to_brg_inplace_c(pixels3d(surface_))

cpdef inline void shader_greyscale_luminosity24_inplace(object surface_):
    """
    SHADER GRAYSCALE (CONSERVE LUMINOSITY)

    This shader transform the game display on a grayscale video game effect
    
    * This shader can also be applied to pygame textures/surface to transform them into
      an equivalent grayscale model
    * The changes are automatically applied inplace to the surface you do not need to create a 
      new surface.

    e.g:
    shader_greyscale_luminosity24_inplace(surface)

    :param surface_  : Pygame surface or display surface compatible (image 24-32 bit with 
                       or without per-pixel transparency / alpha channel)
    :return          : void
    
    """
    assert isinstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)
    shader_greyscale_luminosity24_inplace_c(pixels3d(surface_))


cpdef inline void shader_sepia24_inplace(object surface_):
    """
    SHADER SEPIA MODEL

    Transform your video game into an equivalent sepia model
    
    * The changes are automatically applied inplace to the surface you do not need to create a 
      new surface.

    e.g:
    shader_sepia24_inplace(surface)


    :param surface_  : Pygame surface or display surface compatible (image 24-32 bit with 
                       or without per-pixel transparency / alpha channel)
    :return:         : void
    """
    assert isinstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)
    shader_sepia24_inplace_c(pixels3d(surface_))


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

    surface_cp = surface_.copy()
    cdef:
        int w, h
    w, h = surface_cp.get_size()

    surface_cp = smoothscale(surface_cp, (w >> reduce_factor_, h >> reduce_factor_))

    cdef:
        unsigned char [:, :, :] surface_cp_arr = pixels3d(surface_cp)
        int i, j
        unsigned char[:, :, :] org_surface = pixels3d(surface_)

    shader_median_filter24_inplace_c(surface_cp_arr, kernel_size_)
    surface_cp_arr = scale_array24_c(surface_cp_arr, w, h)

    with nogil:
        for i in prange(w, schedule='static', num_threads=THREADS):
            for j in range(h):
                org_surface[i, j, 0] = surface_cp_arr[i, j, 0]
                org_surface[i, j, 1] = surface_cp_arr[i, j, 1]
                org_surface[i, j, 2] = surface_cp_arr[i, j, 2]


cpdef inline void shader_median_filter24_inplace(
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
    assert isinstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)
    if kernel_size_ <= 0:
        raise ValueError('\nArgument kernel_size_ cannot be <= 0')
    if not 0 < reduce_factor_ < 9:
        raise ValueError('\nArgument reduce_factor_ must be in range [1 ... 8] ')

    if fast_:
        median_fast(surface_, kernel_size_, reduce_factor_)
    else:
        shader_median_filter24_inplace_c(pixels3d(surface_), kernel_size_)


cpdef inline void shader_median_grayscale_filter24_inplace(
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
    assert isinstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)

    assert kernel_size_ > 0, "\nArgument kernel_size_ cannot be <= 0"

    shader_median_grayscale_filter24_inplace_c(pixels3d(surface_), kernel_size_)


cpdef inline void shader_median_filter24_avg_inplace(
        object surface_,
        int kernel_size_=2
):
    """
    SHADER MEDIAN FILTER (AVERAGE)
    
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

    assert isinstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)
    assert kernel_size_ > 0, "\nArgument kernel_size_ cannot be <= 0"

    shader_median_filter24_avg_inplace_c(pixels3d(surface_), kernel_size_)



cpdef inline void shader_color_reduction24_inplace(
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
    shader_color_reduction24_inplace(surface, 8)

    :param surface_: pygame.Surface; compatible 24 - 32 bit 
    :param color_: integer must be > 1 default 8
    :return:
    """
    assert isinstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)
    assert color_ > 1, "Argument color_number must be > 1"

    shader_color_reduction24_inplace_c(pixels3d(surface_), color_)


cpdef inline void shader_sobel24_inplace(
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
    shader_sobel24_inplace(surface, 64)

    :param surface_: pygame.Surface; compatible 24 - 32 bit 
    :param threshold_: integer; Value for detecting the edges default 64
    :return:
    """
    assert isinstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)
    assert -1 < threshold_ < 256, "\nArgument threshold must be an integer in range [0 ... 255]"

    shader_sobel24_inplace_c(pixels3d(surface_), threshold_)


cpdef inline void shader_sobel24_fast_inplace(
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
    shader_sobel24_fast_inplace(surface, 64, factor_=1)

    :param surface_: pygame.surface compatible 24-32 bit 
    :param threshold_: integer; default value is 24 
    :param factor_: integer; default value is 1 (div by 2)
    :return:
    """
    assert isinstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)
    assert -1 < threshold_ < 256, "\nArgument threshold must be an integer in range [0 ... 255]"
    assert 0 < factor_ < 9, "\nArgument factor_ must be in range [1 ... 8]"

    shader_sobel24_fast_inplace_c(surface_, threshold_, factor_)




cpdef inline void shader_invert_surface_24bit_inplace(object surface_):
    """
    SHADER INVERT PIXELS
    
    Invert all pixels of the display or a given texture
    
    Compatible 24 - 32 bit with or without alpha layer
    
    * The changes are automatically applied inplace to the surface, you do not need to create a 
      new surface.  
    
    e.g:
    shader_invert_surface_24bit_inplace(surface)
    
    :param surface_: pygame.surface; compatible 24 - 32 bit surfaces
    :return: void
    """
    assert isinstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)
    shader_invert_surface_24bit_inplace_c(pixels3d(surface_))



cpdef inline void shader_hsl_surface24bit_inplace(object surface_, float shift_):
    """
    ROTATE THE HUE OF THE GAME DISPLAY OR GIVEN TEXTURE
    
    Compatible 24 - 32 bit with or without alpha layer
    
    * The changes are automatically applied inplace to the surface, you do not need to create a 
      new surface.  
      
    e.g:
    shader_hsl_surface24bit_inplace(surface, 0.2)
    
    :param surface_: pygame.Surface; Compatible 24 - 32 bit surfaces
    :param shift_: float; float value in range [-1.0 ... 1.0]
    :return: void 
    """
    assert isinstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)
    assert -1.0 <= shift_ <= 1.0, \
        "Argument shift must be in range[-1.0 ... 1.0]"

    shader_hsl_surface24bit_inplace_c(pixels3d(surface_), shift_)



cpdef inline void shader_hsl_surface24bit_fast_inplace(
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
    shader_hsl_surface24bit_fast_inplace(
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
    assert isinstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)
    assert -1.0 <= shift_ <= 1.0, \
        "Argument shift must be in range[-1.0 ... 1.0]"
    assert isinstance(hsl_model_, (numpy.ndarray, cython.view.memoryview)), \
        "\nArgument hsl_model_ must be a numpy.ndarray or memoryview type, got %s " % type(
            hsl_model_)
    assert isinstance(rgb_model_, (numpy.ndarray, cython.view.memoryview)), \
        "\nArgument rgb_model_ must be a numpy.ndarray or memoryview type, got %s " % type(
            rgb_model_)

    shader_hsl_surface24bit_fast_inplace_c(pixels3d(surface_), shift_, hsl_model_, rgb_model_)


cpdef inline void shader_blur5x5_array24_inplace(object surface_):
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
    
    Compatible 24 - 32 bit with or without alpha layer
    
    * The changes are automatically applied inplace to the surface, you do not need to create a 
      new surface.  

    :param surface_: pygame.Surface; compatible 24 - 32 bit surfaces
    :return: void 
    """
    assert isinstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)

    shader_blur5x5_array24_inplace_c(pixels3d(surface_))


cpdef inline void shader_wave24bit_inplace(object surface_, float rad, int size):
    """
    CREATE A WAVE EFFECT TO THE GAME DISPLAY OR TO A GIVEN SURFACE

    Compatible 24 - 32 bit with or without alpha layer
    
    * The changes are automatically applied inplace to the surface, you do not need to create a 
      new surface.  

    e.g:
    shader_wave24bit_inplace(surface, 8 * math.pi/180.0 + frame_number, 5)
    
    :param surface_: pygame.Surface; pygame surface compatible 24 - 32 bit  
    :param rad     : float; angle in rad to rotate over time
    :param size    : int; Number of sub-surfaces
    :return        : void
    """
    assert isinstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)
    assert size > 0, "Argument size must be > 0"

    shader_wave24bit_inplace_c(pixels3d(surface_), rad, size)


cpdef inline void shader_swirl24bit_inplace(object surface_, float degrees):
    """
    SWIRL AN IMAGE (ANGLE APPROXIMATION METHOD)

    This algorithm uses a table of cos and sin.
    
    Compatible 24 - 32 bit with or without alpha layer
    
    * The changes are automatically applied inplace to the surface, you do not need to create a 
      new surface.  

    e.g:
    shader_swirl24bit_inplace(surface, frame_number / 1000.0)
    
    :param surface_: pygame.Surface, compatible 24 - 32 bit 
    :param degrees : float; angle in degrees 
    :return        : void 
    """
    assert isinstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)

    shader_swirl24bit_inplace_c(pixels3d(surface_), degrees)



cpdef inline void shader_swirl24bit_inplace1(object surface_, float degrees):
    """
    SWIRL AN IMAGE WITHOUT ANGLE APPROXIMATION

    Compatible 24 - 32 bit with or without alpha layer
    
    * The changes are automatically applied inplace to the surface, you do not need to create a 
      new surface.  
      
    e.g:
    shader_swirl24bit_inplace(surface_, frame_number / 1000)
    
    :param surface_: pygame.Surface, compatible 24 - 32 bit 
    :param degrees : float; angle in degrees
    :return        : void 
    """
    assert isinstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)

    shader_swirl24bit_inplace_c1(pixels3d(surface_), degrees)



cpdef inline void shader_plasma24bit_inplace(
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
    shader_plasma24bit_inplace(surface, frame_number)

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
    assert isinstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)

    shader_plasma24bit_inplace_c(pixels3d(surface_), frame, hue_, sat_, value_, a_, b_, c_)


cpdef inline void shader_plasma(surface_, float frame, unsigned int [::1] palette_):
    """
    CREATE A PLASMA EFFECT INPLACE

    e.g:
    shader_plasma(surface, frame_number, palette_)
    
    :param surface_: pygame.Surface; compatible 24 - 32 bit 
    :param frame   : float; frame number
    :param palette_: 1d array containing colors
    :return:
    """
    assert isinstance(surface_, pygame.Surface), \
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



cpdef inline void shader_brightness24_inplace(object surface_, float shift_):
    """
    SHADER BRIGHTNESS

    This shader control the pygame display brightness level
    It uses two external functions coded in C, struct_rgb_to_hsl & struct_hsl_to_rgb
    
    Compatible 24 - 32 bit with or without alpha layer
    
    * The changes are automatically applied inplace to the surface, you do not need to create a 
      new surface.  
      
    e.g:
    shader_brightness24_inplace(surface, 0.2)
    
    :param surface_ : pygame.surface; 
    :param shift_   : float must be in range [ -1.0 ... 1.0 ]
    :return         : void
    """
    assert isinstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)
    assert -1.0 <= shift_ <= 1.0, "\nArgument shift_ must be in range [-1.0 ... 1.0]"

    shader_brightness24_inplace_c(pixels3d(surface_), shift_)


# TODO TEST
cpdef inline void shader_brightness24_exclude_inplace(
        object surface_,
        float shift_,
        color_=(0, 0, 0)
):
    """

    :param surface_:
    :param shift_:
    :param color_:
    :return:
    """
    assert isinstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)
    shader_brightness24_exclude_inplace_c(pixels3d(surface_), shift_, color_)




cpdef inline void shader_brightness_24_inplace1(
        object surface_,
        float shift_,
        float [:, :, :, :] rgb_to_hsl_model
):
    """
    
    SHADER BRIGHTNESS (EXCLUDE A SPECIFIC COLOR FROM THE PROCESS, DEFAULT BLACK COLOR)

    This shader control the pygame display brightness level
    It uses two external functions coded in C, struct_rgb_to_hsl & struct_hsl_to_rgb

    e.g:
    shader_brightness24_exclude_inplace(surface, 0.2)
    
    :param surface_ : pygame.surface; compatible 24 - 32 bit 
    :param shift_   : float in range [-1.0 ... 1.0 ]
    :param rgb_to_hsl_model : numpy.ndarray shape (256, 256, 256, 3)
    :return : void 
    """
    assert isinstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)

    warnings.warn("Deprecated version, use shader_brightness_24_inplace (fastest version)",
                  DeprecationWarning)
    assert -1.0 <= shift_ <= 1.0, "\nArgument shift_ must be in range [-1.0 ... 1.0]"

    shader_brightness_24_inplace1_c(pixels3d(surface_), shift_, rgb_to_hsl_model)




cpdef inline void shader_saturation_array24_inplace(object surface_, float shift_):
    """
    SHADER SATURATION

    This shader control the saturation level of the pygame display or surface/texture

    e.g:
    shader_saturation_array24_inplace(surface, 0.2)
    
    
    :param surface_: pygame.Surface; compatible 24 - 32 bit
    :param shift_  : float must be in range [ -1.0 ... 1.0] 
    :return:
    """
    assert isinstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)
    assert -1.0 <= shift_ <= 1.0, "\nArgument shift_ must be in range [-1.0 ... 1.0]"

    shader_saturation_array24_inplace_c(pixels3d(surface_), shift_)



cpdef inline void shader_heatwave24_vertical_inplace(
        object surface_,
        unsigned char [:, :] mask,
        float factor_,
        float center_,
        float sigma_,
        float mu_):
    """

    APPLY A GAUSSIAN TRANSFORMATION TO A SURFACE

    This effect can be use for simulate air turbulence or heat flow/convection

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
    assert isinstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)
    assert isinstance(mask, (numpy.ndarray, cython.view.memoryview)), \
        "\nArgument mask must be a numpy.array or memoryview type, got %s " % type(mask)

    shader_heatwave24_vertical_inplace_c(pixels3d(surface_), mask, factor_, center_, sigma_, mu_)



cpdef inline void shader_horizontal_glitch24_inplace(
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
    assert isinstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)

    shader_horizontal_glitch24_inplace_c(pixels3d(surface_), rad1_, frequency_, amplitude_)



cpdef inline void shader_bpf24_inplace(object surface_, int threshold = 128):
    """
    
    SHADER BRIGHT PASS FILTER (INPLACE)

    Conserve only the brightest pixels in a surface

    :param surface_ : pygame.Surface; compatible 24 - 32 bit 
    :param threshold: integer; Bright pass threshold default 128
    :return: void 
    """
    assert isinstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)

    shader_bpf24_inplace_c(pixels3d(surface_), threshold)



cpdef inline void shader_bloom_effect_array24(object surface_, int threshold_, bint fast_=False):
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
    :return             : void
    
    """
    assert isinstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)

    shader_bloom_effect_array24_c(surface_, threshold_, fast_)



cpdef inline shader_fisheye24_footprint_inplace(int w, int h):
    """

    :param w:
    :param h:
    :return:
    """
    return shader_fisheye24_footprint_c(w, h)




cpdef inline void shader_fisheye24_inplace(
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
    assert isinstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)

    assert isinstance(fisheye_model, (cython.view.memoryview, numpy.ndarray)), \
        "\nArgument fisheye_model must be a numpy.ndarray or a cython.view.memoryview  type, " \
        "got %s " % type(fisheye_model)

    shader_fisheye24_inplace_c(pixels3d(surface_), fisheye_model)



# TODO TESTING
cpdef inline tuple shader_rain_footprint_inplace(int w, int h):
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




# TODO TESTING
cpdef inline void shader_rain_fisheye24_inplace(
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
    assert isinstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)

    assert isinstance(surface_, (numpy.ndarray, cython.view.memoryview)), \
        "\nArgument rain_fisheye_model must be a " \
        "numpy.ndarray or a cython memoryview type, got %s " % type(rain_fisheye_model)

    shader_rain_fisheye24_inplace_c(pixels3d(surface_), rain_fisheye_model)




cpdef inline void shader_tv_scanline_inplace(surface_, int space=5):
    """
    
    SHADER CREATING A TV SCANLINE EFFECT ON PYGAME SURFACE

    The space between each scanline can by adjusted with the space value.
    The scanline intensity/colors is lower that the original image

    :param surface_     : pygame.Surface compatible 24-32 bit 
    :param space        : integer; space between the lines
    :return             : void
    
    """
    # TODO SCANLINE VERTICAL | HORIZONTAL

    assert isinstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)

    assert space > 0, "Argument space cannot be <=0"

    shader_tv_scanline_inplace_c(pixels3d(surface_), space)




cpdef inline void shader_rgb_split_inplace(object surface_, int offset_=10):
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
    assert isinstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)

    shader_rgb_split_inplace_c(surface_, offset_)




cpdef inline tuple shader_ripple(
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
    assert isinstance(previous_, (numpy.ndarray, cython.view.memoryview)), \
        "\nArgument previous must be a numpy.ndarray type got %s " % type(previous_)

    assert isinstance(current_, (numpy.ndarray, cython.view.memoryview)), \
        "\nArgument current must be a numpy.ndarray type got %s " % type(current_)

    assert isinstance(array_, (numpy.ndarray, cython.view.memoryview)), \
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




cpdef inline void heatmap_surface24_conv_inplace(object surface_, bint rgb_=True):
    """
    TRANSFORM AN IMAGE INTO A HEATMAP EQUIVALENT

    :param surface_ : pygame.Surface
    :param rgb_     : boolean; True transformed the image into a RGB heatmap model of False (BGR)
    :return         : void
    """
    assert isinstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)

    heatmap_surface24_conv_inplace_c(surface_, rgb_)





cpdef inline predator_vision_mode(
        object surface_,
        unsigned int sobel_threshold=12,
        unsigned int bpf_threshold=50,
        unsigned int bloom_threshold=50,
        inv_colormap=False,
        fast=False
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
    :return                : Return a pygame surface  
    """

    surface_copy = surface_.copy()

    if fast:
        shader_sobel24_fast_inplace(surface_copy, sobel_threshold, factor_=1)
    else:
        shader_sobel24_inplace(surface_copy, sobel_threshold)

    shader_bpf24_inplace(surface_, bpf_threshold)
    shader_bloom_effect_array24_c(surface_, bloom_threshold, fast_=True)
    heatmap_surface24_conv_inplace_c(surface_, inv_colormap)
    surface_.blit(surface_copy, (0, 0), special_flags=pygame.BLEND_RGB_ADD)

    return surface_.convert()




cpdef inline shader_blood_inplace(object surface_, float [:, :] mask_, float perc_):
    """
    SHADER 2D GAME "HURT EFFECT"
    
    This effect is used in 2D game when the player is being hurt
    THE MASK DETERMINE THE CONTOUR USED FOR THE BLOOD EFFECT.

    :param surface_ : pygame.Surface; compatible surface 24 - 32 bit
    :param mask_    : numpy.ndarray shape (w, h) of float values in range [0.0...1.0]
    :param perc_    : Percentage value in range [0.0 ... 1.0] with 1.0 being 100%
    :return         : void
    
    """
    assert isinstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)
    assert isinstance(mask_, (numpy.ndarray, cython.view.memoryview)), \
        "\nArgument mask_ must be a numpy.ndarray or cython memoryview types got %s " % type(mask_)

    cdef Py_ssize_t w, h
    w, h = surface_.get_size()

    cdef Py_ssize_t mask_w, mask_h
    mask_w, mask_h = mask_.shape[:2]

    assert w == mask_w and h == mask_h, "\nSurface size and mask size mismatch"

    shader_blood_inplace_c(pixels3d(surface_), mask_, perc_)



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


# todo develop DOC
cpdef inline shader_fire_surface24(
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




cpdef inline shader_fire_effect(
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



cpdef inline shader_cloud_effect(
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



cpdef inline mirroring(object surface_):
    """
    
    SHADER MIRRORING

    This method create a mirror image 
    
    Compatible 24 - 32 bit image / surface
    
    :param surface_ : pygame.Surface; compatible 24 - 32 bit 
    :return         : returns a numpy ndarray shape (w, h, 3) 
    
    """
    assert isinstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)

    return mirroring_c(pixels3d(surface_))




cpdef inline void mirroring_inplace(object surface_):
    """
    SHADER MIRRORING (INPLACE)

    This method create a mirror image 

    :param surface_ : pygame.Surface; compatible 24 - 32 bit 
    :return : void
    """
    assert isinstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)

    mirroring_inplace_c(pixels3d(surface_))

# cpdef inline transpose_inplace(object surface_):
#     return tranpose_c(pixels3d(surface_))




cpdef inline void shader_sharpen_filter_inplace(object surface_):
    """
    
    SHARPEN IMAGE APPLYING THE BELOW 3 X 3 KERNEL OVER EVERY PIXELS.

    :param surface_ : pygame.Surface; compatible 24 - 32 bit 
    :return         : void 
    """
    assert isinstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)

    shader_sharpen_filter_inplace_c(pixels3d(surface_))

# *******************************************************************


cpdef inline zoom_in_effect_inplace(object surface_):
    """

    :param surface_ : pygame.Surface; compatible 24 - 32 bit 
    :return:
    """
    assert isinstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)
    raise NotImplementedError



cpdef inline electric_effect_inplace(object surface_):
    """

    :param surface_ : pygame.Surface; compatible 24 - 32 bit 
    :return:
    """
    assert isinstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)
    raise NotImplementedError



cpdef inline filmstrips(object surface_):
    """

    :param surface_ : pygame.Surface; compatible 24 - 32 bit 
    :return:
    """
    assert isinstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)
    raise NotImplementedError



cpdef cubism(object surface_):
    """

    :param surface_ : pygame.Surface; compatible 24 - 32 bit 
    :return:
    """
    assert isinstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)
    raise NotImplementedError



cpdef code_listing(object surface_, size_):
    """

    :param surface_: pygame.Surface; compatible 24 - 32 bit 
    :param size_:
    :return:
    """
    assert isinstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)
    raise NotImplementedError

cpdef transition(object surface_):
    """
    
    :param surface_: 
    :return: 
    """
    raise NotImplementedError


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



cpdef explode(object surface_):
    """

    :param surface_: 
    :return: 
    """
    raise NotImplementedError


# ******************************************************************

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


cdef float C1 = <float>1.0 / <float>sqrt(M_2PI)

cdef inline float gauss(float x, float c, float sigma=1.0, float mu=0.0)nogil:
    """
    
    :param x: 
    :param c: 
    :param sigma: 
    :param mu: 
    :return: 
    """
    x -= c
    return (1.0 / sigma * C1) * exp(-0.5 * ((x - mu) * (x - mu)) / (sigma * sigma))


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
    shader_rgb_to_bgr_inplace(surface)

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
    shader_rgb_to_brg_inplace(surface)

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
    shader_greyscale_luminosity24_inplace(surface)

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
    shader_sepia24_inplace(surface)

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
cdef inline int partition(unsigned char [::1] nums, int low, int high)nogil:
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
        split_index = partition(items, low, high)
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

                rgb_array_[i, j, 0] = tmpr[k + 1]
                rgb_array_[i, j, 1] = tmpg[k + 1]
                rgb_array_[i, j, 2] = tmpb[k + 1]



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline void shader_median_filter24_avg_inplace_c(
        unsigned char [:, :, :] rgb_array_, int kernel_size_=2):

    """
    SHADER MEDIAN FILTER

    This shader cannot be used for real time rendering as the performance of the algorithm are not
    satisfactory. The code would have to be changed and improved with C or assembler in order to
    be adapted for a real time application. Another version can also be written with a surface
    downscale prior processing /rescale method that would boost the fps performance.

    In the state, this shader can be used for texture/surface transformation offline

    The Array (rgb_array) must be a numpy array shape (w, h, 3) containing RGB pixels,
    please refer to pygame function pixels3d or array3d to convert an image into a
    3d array (library surfarray)

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

        int i=0, j=0
        int ky, kx
        Py_ssize_t ii=0, jj=0
        int k = kernel_size_ >> 1
        int k_size = kernel_size_ * kernel_size_

        Py_ssize_t w_1 = w - 1, h_1 = h - 1
        int red, green, blue
        unsigned char max_r, min_r, max_g, min_g, max_b, min_b
        unsigned char *r
        unsigned char *g
        unsigned char *b

    with nogil:
        for i in prange(w, schedule='static', num_threads=THREADS, chunksize=2048):
            for j in range(h):

                max_r = 0
                min_r = 0
                max_g = 0
                min_g = 0
                max_b = 0
                min_b = 0

                for kx in range(-k, k):
                    for ky in range(-k, k):
                        ii = (i + kx)
                        jj = (j + ky)

                        if ii < 0:
                            ii = 0
                        elif ii > w_1:
                            ii = w_1

                        if jj < 0:
                            jj = 0
                        elif jj > h_1:
                            jj = h_1

                        r = &rgb_array_[ii, jj, 0]
                        g = &rgb_array_[ii, jj, 1]
                        b = &rgb_array_[ii, jj, 2]

                        if r[0] > max_r:
                            max_r = r[0]
                        if g[0] > max_g:
                            max_g = g[0]
                        if b[0] > max_b:
                            max_b = b[0]

                        if r[0] < min_r:
                            min_r = r[0]
                        if g[0] < min_g:
                            min_g = g[0]
                        if b[0] < min_b:
                            min_b = b[0]

                rgb_array_[i, j, 0] = <unsigned char>((max_r - min_r) *0.5)
                rgb_array_[i, j, 1] = <unsigned char>((max_g - min_g) *0.5)
                rgb_array_[i, j, 2] = <unsigned char>((max_b - min_b) *0.5)



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
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

    cdef:
        unsigned char [:, :, ::1] rgb_array_copy = \
            ascontiguousarray(numpy.array(rgb_array_, copy=True))

        int i=0, j=0, ky, kx
        Py_ssize_t ii=0, jj=0

        int k = kernel_size_ >> 1
        int k_size = kernel_size_ * kernel_size_
        # int [64] tmp_  = empty(64, numpy.int16, order='C')
        int *tmp_   = <int *> malloc(k_size * sizeof(int))
        int *tmp
        int index = 0
        unsigned char *v


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
    shader_color_reduction24_inplace(surface, 8)

    :param rgb_array    : numpy.ndarray shape(w, h, 3) uint8 (unsigned char 0...255) containing the
    pygame display pixels format RGB
    :param color_number : integer; color number color_number^2
    :return             : void
    """


    cdef Py_ssize_t w, h
    w, h = rgb_array.shape[:2]

    # cdef float color_number = <float>pow(2, factor)

    cdef:
        int x=0, y=0
        float f = <float>255.0 / <float>color_number
        unsigned char *r
        unsigned char *g
        unsigned char *b
        float c1

    with nogil:
        for y in prange(h, schedule='static', num_threads=THREADS):
            for x in range(0, w):

                r = &rgb_array[x, y, 0]
                g = &rgb_array[x, y, 1]
                b = &rgb_array[x, y, 2]

                c1 = <float>color_number * <float>ONE_255

                r[0] = <unsigned char>(<int>(c1 * <float>r[0]) * f)
                g[0] = <unsigned char>(<int>(c1 * <float>g[0]) * f)
                b[0] = <unsigned char>(<int>(c1 * <float>b[0]) * f)



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
    shader_sobel24_inplace(surface, 64)

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

                        if kernel_offset_x != 0:

                            r_gx = r_gx + <float> gray[0] * \
                                   <float> GX[kernel_offset_x + KERNEL_HALF,
                                              kernel_offset_y + KERNEL_HALF]

                        if kernel_offset_y != 0:

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
        unsigned char [:, :, ::1] new_array = numpy.zeros((w2, h2, 3), numpy.uint8)
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
    shader_sobel24_fast_inplace(surface, 64, factor_=1)

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
    shader_invert_surface_24bit_inplace(surface)

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


cdef float[5] GAUSS_KERNEL = [1.0/16.0, 4.0/16.0, 6.0/16.0, 4.0/16.0, 1.0/16.0]

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline void shader_blur5x5_array24_inplace_c(
        unsigned char [:, :, :] rgb_array_, mask=None):
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
    :return             : Return 24-bit a numpy.ndarray type (w, h, 3) uint8
    """

    cdef Py_ssize_t w, h
    w, h = rgb_array_.shape[:2]

    # kernel 5x5 separable
    cdef:

        short int kernel_half = 2
        unsigned char [:, :, ::1] convolve = numpy.empty((w, h, 3), dtype=uint8)
        unsigned char [:, :, ::1] convolved = numpy.empty((w, h, 3), dtype=uint8)
        Py_ssize_t kernel_length = len(GAUSS_KERNEL)
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
    shader_wave24bit_inplace(surface, 8 * math.pi/180.0 + frame_number, 5)

    :param rgb_array_   : numpy.ndarray shape (w, h, 3) containing all the RGB values
    :param rad          : float; angle in rad to rotate over time
    :param size         : int; Number of sub-surfaces
    :return             : void
    """



    cdef Py_ssize_t w, h
    w, h = rgb_array_.shape[:2]

    cdef:
        unsigned char [:, :, :] rgb = numpy.array(rgb_array_, copy=True)
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
    shader_swirl24bit_inplace(surface, frame_number / 1000.0)

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


    columns = <float>0.5 * (<float>w - <float>1.0)
    rows    = <float>0.5 * (<float>h - <float>1.0)

    with nogil:
        for j in prange(h, schedule='static', num_threads=THREADS):
            for i in range(w):

                di = <float>i - columns
                dj = <float>j - rows

                r = <float>sqrt(di * di + dj * dj)
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
    shader_swirl24bit_inplace(surface_, frame_number / 1000)

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


    columns = <float>0.5 * (w - <float>1.0)
    rows    = <float>0.5 * (h - <float>1.0)

    with nogil:
        for i in prange(w, schedule='static', num_threads=THREADS):
            for j in range(h):

                di = <float>i - columns
                dj = <float>j - rows

                r = <float>sqrt(di * di + dj * dj)
                angle = rad * r

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
    shader_plasma24bit_inplace(surface, frame_number)

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
                    r, g, b =  v, t, p
                if i == 1:
                     r, g, b = q, v, p
                if i == 2:
                     r, g, b = p, v, t
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
    shader_plasma(surface, frame_number)

    :param surface_: pygame Surface
    :param frame   : float; frame number
    :param palette_: color palette
    :return        : void
    """
    cdef Py_ssize_t width, height
    width, height = surface_.get_size()

    cdef:
        int x, y, ii,c

        unsigned char [:, :, :] rgb_array_ = pixels3d(surface_)

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
    shader_hsl_surface24bit_inplace(surface, 0.2)

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
    shader_hsl_surface24bit_inplace(surface, 0.2, hsl_model, rgb_model)

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

                h__ = <unsigned char> (<float>min((h_ * ONE_255 + shift_), <float>1.0) * \
                    <float>255.0)

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
    shader_brightness24_inplace(surface, 0.2)

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
    shader_brightness24_exclude_inplace(surface, 0.2)

    :param rgb_array_: numpy ndarray shape (w, h, 3) containing RGB pixels values
    :param shift_    : float; values in range [-1.0 ... 1.0], 0 no change,
    -1 lowest brightness, +1 highest brightness
    :param color_    : tuple; Color to excude from the brightness process
    :return          : void
    """
    assert -1.0 <= shift_ <= 1.0, \
        "Argument shift must be in range[-1.0 ... 1.0]"

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
    shader_brightness_24_inplace1(surface, 0.2 rgb_to_hsl_model)

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
    shader_saturation_array24_inplace(surface, 0.2)

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
# shader_heatwave24_vertical_inplace(
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
# e.g shader_horizontal_glitch24_inplace(surface, 0.5, 0.08, frame % 20)
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
cdef inline void shader_bloom_effect_array24_c(
        surface_,
        int threshold_,
        bint fast_ = False):
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

    with nogil:
        w2, h2   = <int>w >> 1, <int>h >> 1
        w4, h4   = w2 >> 1, h2 >> 1
        w8, h8   = w4 >> 1, h4 >> 1
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

    # SUBSURFACE DOWNSCALE CANNOT
    # BE PERFORMED AND WILL RAISE AN EXCEPTION
    if not x2:
        return

    if fast_:
        x2, x4, x8 = False, False, False

    surface_cp = bpf24_c(pixels3d(surface_), threshold=threshold_)


    # FIRST SUBSURFACE DOWNSCALE x2
    # THIS IS THE MOST EXPENSIVE IN TERM OF PROCESSING TIME
    if x2:
        s2 = scale(surface_cp, (w2, h2))
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
        s4 = scale(surface_cp, (w4, h4))
        s4_array = numpy.array(s4.get_view('3'), dtype=numpy.uint8)
        shader_blur5x5_array24_inplace_c(s4_array)
        # b4_blurred = frombuffer(numpy.array(s4_array.transpose(1, 0, 2),
        # order='C', copy=False), (w4, h4), 'RGB')
        b4_blurred = make_surface(s4_array)
        s4 = smoothscale(b4_blurred, (w, h))
        surface_.blit(s4, (0, 0), special_flags=BLEND_RGB_ADD)

    # THIRD SUBSURFACE DOWNSCALE x8
    if x8:
        s8 = scale(surface_cp, (w8, h8))
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
        s16 = scale(surface_cp, (w16, h16))
        s16_array = numpy.array(s16.get_view('3'), dtype=numpy.uint8)
        shader_blur5x5_array24_inplace_c(s16_array)
        # b16_blurred = frombuffer(numpy.array(s16_array.transpose(1, 0, 2),
        # order='C', copy=False), (w16, h16), 'RGB')
        b16_blurred = make_surface(s16_array)
        s16 = smoothscale(b16_blurred, (w, h))
        surface_.blit(s16, (0, 0), special_flags=BLEND_RGB_ADD)


    # if mask_ is not None:
    #     # Multiply mask surface pixels with mask values.
    #     # RGB pixels = 0 when mask value = 0.0, otherwise
    #     # modify RGB amplitude
    #     surface_cp = filtering24_c(surface_cp, mask_)


cdef unsigned int [:, :, ::1] IMAGE_FISHEYE_MODEL = numpy.zeros((800, 1024, 2), uint32)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline shader_fisheye24_footprint_c(Py_ssize_t w, Py_ssize_t h):

    """
    CREATE A FISHEYE MODEL TO HOLD THE PIXEL COORDINATES OF A SURFACE/ GAME DISPLAY

    * The surface and the model must have the same dimensions.

    Store the fisheye model into an external array image_fisheye_model shape (width, height, 2)

    IMAGE_FISHEYE_MODEL contains the fisheye transformation coordinate (x2 & y2) that reference
    the final image pixel position (fisheye model)
    This method has to be call once before the main loop in order to calculate
    the projected position for each pixels.

    :param w    : integer; width of the model
    :param h    : integer; height of the model
    :return     : Return a numpy.ndarray type (w, h, 2) representing the fisheye model (coordinates
    of all surface pixels passing through the fisheye lens model)
    """

    assert w > 0, "Argument w must be > 0"
    assert h > 0, "Argument h must be > 0"

    cdef:
        unsigned int [:, :, :] image_fisheye_model = numpy.zeros((w, h, 2), uint32)
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

    return asarray(ascontiguousarray(image_fisheye_model))

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

                rgb_array_[x, y, 0] = rgb_array_copy[x2[0], y2[0], 0]
                rgb_array_[x, y, 1] = rgb_array_copy[x2[0], y2[0], 1]
                rgb_array_[x, y, 2] = rgb_array_copy[x2[0], y2[0], 2]



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
        unsigned int [:, :, ::1] rain_fisheye_model = numpy.zeros((w, h, 3), uint32)
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
                    nr = (r + <float>1.0 - <float>sqrt(1.0 - (nx2 + ny2))) * <float>0.5
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
        unsigned char [:] red   = numpy.empty(z, uint8, order='C')
        unsigned char [:] green = numpy.empty(z, uint8, order='C')
        unsigned char [:] blue  = numpy.empty(z, uint8, order='C')

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
            angles[i]    = <int>round(<float>s_width * atan2(y - screen_height,
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

    return pygame.image.frombuffer(dest_array, (screen_width, screen_height), "RGBA")




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
cpdef tuple heatmap(int wavelength, float gamma=1.0):
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
HEATMAP = [heatmap(i, 1.0) for i in range(380, 750)]

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
cpdef inline void heatmap_surface24_conv_inplace_c(object surface_, bint rgb_=True):
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
cpdef tuple bluemap(int wavelength, float gamma=1.0):
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
BLUEMAP = [bluemap(i, 1.0) for i in range(450, 495)]

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
cpdef inline void bluemap_surface24_inplace_c(object surface_):
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
cpdef tuple redmap(int wavelength, float gamma=1.0):
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
REDMAP = [redmap(i, 1.0) for i in range(620, 750)]

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
cpdef inline void redmap_surface24_inplace_c(object surface_):
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
                fire[height-1, x] = randRange(intensity, 260)


        # DILUTE THE FLAME EFFECT (DECREASE THE MAXIMUM INT VALUE) WHEN THE FLAME TRAVEL UPWARD
        for y in prange(1, height-1, schedule='static', num_threads=THREADS):

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
# surface_ = shader_fire_effect(
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
        if isinstance(surface_, pygame.Surface):
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
        shader_bloom_effect_array24_c(fire_surface_smallest, bpf_threshold_, fast_=fast_bloom_)

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

    * Do not call that function directly, this function is a sub function of shader_cloud_effect

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
# surface_ = shader_cloud_effect(
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
        if isinstance(surface_, pygame.Surface):
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
cpdef tuple dampening_effect(
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
    surf, xx, yy = dampening_effect(BCK, frame, width, height,
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
cpdef inline float lateral_dampening_effect(int frame_, float amplitude_=50.0,
                                            int duration_=30, float freq_=20.0):
    """
    DAMPENING EFFECT

    * This method return the lateral displacement (x)

    e.g:
    tm = lateral_dampening_effect(frame, amplitude_=50.0, duration_=35, freq_=5.0)
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
    :return          : Return a float coresponding to the lateral displacement (x)
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
        shader_sharpen_filter_inplace(surface_)

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
    array_ = canny_blur5x5_surface24_c(surface_branch_1)
    shader_greyscale_luminosity24_inplace_c(array_)
    shader_sobel24_inplace_c(array_, sobel_threshold_)

    # Second branch
    arr = pixels3d(surface_)
    shader_median_filter24_inplace_c(arr, median_kernel_)

    # median_fast(
    #     surface_,
    #     kernel_size_=median_kernel_,
    #     reduce_factor_=2)

    # Color reduction
    shader_color_reduction24_inplace_c(arr, color_)

    del arr

    # Blend both branch
    surface_.blit(make_surface(array_), (0, 0), special_flags=flag_)

    return surface_

# ------------------------------------------------------------------------------------------------