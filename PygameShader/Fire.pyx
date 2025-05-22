# cython: binding=False, boundscheck=False, wraparound=False, nonecheck=False, cdivision=True,
# profile=Fal5se, initializedcheck=False, exceptval(check=False)
# cython: optimize.use_switch=True
# cython: warn.maybe_uninitialized=False
# cython: warn.unused=False
# cython: warn.unused_result=False
# cython: warn.unused_arg=False
# cython: language_level=3
# cython: write_stub_file = True
# encoding: utf-8

"""
                 GNU GENERAL PUBLIC LICENSE
                       Version 3, 29 June 2007

 Copyright (C) 2007 Free Software Foundation, Inc. <https://fsf.org/>
 Everyone is permitted to copy and distribute verbatim copies
 of this license document, but changing it is not allowed.

Copyright Yoann Berenguer
"""

# Cython Function Declarations

# def - Basically, it’s Python

# def is used for code that will be:

#     Called directly from Python code with Python objects as arguments.
#     Returns a Python object.

# The generated code treats every operation as if it was dealing with Python
# objects with Python consequences so it incurs a high overhead. def is safe
# to use with no gotchas. Declaring the types of arguments and local types
# (thus return values) can allow Cython to generate optimised code which speeds
# up the execution. If the types are declared then a TypeError will be raised if
# the function is passed the wrong types.

# cdef - Basically, it’s C

# cdef is used for Cython functions that are intended to be pure ‘C’ functions.
# All types must be declared. Cython aggressively optimises the the code and there
# are a number of gotchas. The generated code is about as fast as you can get though.

# cdef declared functions are not visible to Python code that imports the module.

# Take some care with cdef declared functions; it looks like you are writing
# Python but actually you are writing C.

# cpdef - It’s Both
# cpdef functions combine both def and cdef by creating two functions;
# a cdef for C types and a def fr Python types. This exploits early binding
# so that cpdef functions may be as fast as possible when using C fundamental
# types (by using cdef). cpdef functions use dynamic binding when passed Python
# objects and this might much slower, perhaps as slow as def declared functions.

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=ImportWarning)

import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

try:
    cimport cython
    from cython.parallel cimport prange, parallel

except ImportError:
    raise ImportError("\n<cython> library is missing on your system."
          "\nTry: \n   C:\\pip install cython on a window command prompt.")

try:
    import numpy
    from numpy import empty, uint8, int16, float32, asarray, \
        ascontiguousarray, zeros, uint16, uint32, int32, int8
except ImportError:
    raise ImportError("\n<numpy> library is missing on your system."
          "\nTry: \n   C:\\pip install numpy on a window command prompt.")

try:
    import pygame
    from pygame import Surface
    from pygame.surfarray import pixels3d, array_alpha, pixels_alpha, array3d, \
        make_surface, blit_array, array_to_surface
    from pygame.transform import smoothscale, rotate, scale

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


from libc.stdlib cimport rand

from PygameShader.config import OPENMP, THREAD_NUMBER, __VERSION__
from PygameShader.misc cimport randRange
from PygameShader.Palette cimport make_palette_c
from PygameShader.shader cimport brightness_bpf_c, bloom_c, brightness_ex_c
from PygameShader.shader import shader_bloom_fast1
from PygameShader.gaussianBlur5x5 cimport blur3d_c

cdef int THREADS = 1

if OPENMP:
     THREADS = THREAD_NUMBER

DEF SCHEDULE = 'static'


# This can be improved removing % modulo
@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(False)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef fire_surface24_c_border(
        const int width,
        const int height,
        const float factor,
        const unsigned int [::1] palette,
        float [:, ::1] fire,
        int intensity = 0,
        int low = 0,
        int high = 0,
):
    """
    Generates a fire effect on the borders of an image (border effect). This function is 
    a critical part of the `fire_effect_c` function and performs the main computation 
    to simulate the fire on the image's borders.

    The resulting array is in 24-bit RGB format with shape (width, height, 3), 
    where each pixel represents the fire effect.

    This function cannot be called directly from a Python interpreter.

    :param width    : int
        The width (max horizontal dimension) of the effect.

    :param height   : int
        The height (max vertical dimension) of the effect.

    :param factor   : float
        A factor to scale down the flame effect's intensity. A higher value will reduce the effect's size.

    :param palette  : ndarray (unsigned int)
        A 1D NumPy array representing the color palette used for the fire effect. Each color is stored as an unsigned int.

    :param fire     : ndarray (float)
        A 2D NumPy array containing float values representing the intensity of the fire at each position. 
        The shape of the array is (x, y), where x and y correspond to pixel coordinates.

    :param intensity: int, optional (default=0)
        Controls the intensity of the fire effect. Higher values result in a more intense flame. 
        Valid range: [0, 32].

    :param low      : int, optional (default=0)
        The lowest x-coordinate (starting point) for the fire effect's horizontal range. 
        Must satisfy: 0 <= low < high.

    :param high     : int, optional (default=0)
        The highest x-coordinate for the fire effect's horizontal range. 
        Must satisfy: low < high <= width.

    :return         : ndarray (uint8)
        A 3D NumPy array of shape (width, height, 3), containing the generated fire effect in RGB format. 
        This is a 24-bit color image.
    """


    cdef:
        # flame opacity palette
        unsigned char [:, :, ::1] out = zeros((width, height, 3), dtype=uint8)
        int x = 0, y = 0
        float d
        unsigned int ii=0, dd 
        int c1 = 0, c2 = 0

    cdef int min_, max_, middle


    if low != 0 or high != 0:
        assert 0 <= low < high, "Argument low_ must be < high_"
        assert high <= width, "Argument high must be <= w"

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
        #         fire[h - 1, x] = randRange(intensity, 260)

        # FIRE ARRAY IS [HEIGHT, WIDTH]
        for x in prange(min_, max_, schedule=SCHEDULE, num_threads=THREADS):
            # fire[x % height, (height - 1) % width] = <float>randRange(intensity, <int>260)
            fire [x if x < height else x - height, (height - 1) if (height - 1) < width else (height - 1) - width] = \
                <float>randRange(intensity, <int>260)

        # DILUTE THE FLAME EFFECT (DECREASE THE MAXIMUM INT VALUE) WHEN THE FLAME TRAVEL UPWARD
        for y in prange(1, height - 1, schedule=SCHEDULE, num_threads=THREADS):

            # c1 = (y + 1) % height
            
            c1 = (y + 1) if (y + 1) < height else (y + 1) - height

            for x in range(0, width):

                    # c2 = x % width
                    
                    c2 = x if x < width else x - width
 
                    d = (fire[c1, (x - 1) if (x - 1) < width else (x - 1) - width ] # (x - 1) % width ] 
                       + fire[c1, c2 ] # x % width ]
                       + fire[c1, (x + 1) if (x + 1) < width else (x + 1) - width ] #(x + 1) % width]
                       + fire[(y + 2) % height, c2 ]) * factor # x % width]) * factor
                 
                    d = d - <float>(rand() * <float>0.0001)

                    # Cap the values
                    if d < 0:
                        d = <float>0.0

                    if d>255.0:
                        d = <float>255.0

                    fire[x if x < height else x - height ,  y if y < width else y - width] = d
                    dd = <unsigned int>d 

                    ii = palette[dd if dd < width else dd - width]

                    out[x, y, 0] = (ii >> <unsigned char>16) & <unsigned char>255
                    out[x, y, 1] = (ii >> <unsigned char>8) & <unsigned char>255
                    out[x, y, 2] = ii & <unsigned char>255

    return asarray(out)




@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef inline fire_sub(
        const int width,
        const int height,
        const float factor,
        const unsigned int [::1] palette,
        float [:, ::1] fire
):
    """

    Create a fire effect

    :param width    : 
        integer; max w of the effect
        
    :param height   : 
        integer; max h of the effect
        
    :param factor   : 
        float; factor to reduce the flame effect
        
    :param palette  : 
        ndarray; Color palette 1d numpy array (colors buffer unsigned int values)
        
    :param fire     : 
        ndarray; 2d array (x, y) (contiguous) containing float values
        
    :return         : 
        Return a numpy array containing the fire effect array shape (w, h, 3) of RGB pixels

    """

    return fire_surface24_c(width, height, factor, palette, fire)



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef fire_surface24_c(
        const int width,
        const int height,
        const float factor,
        const unsigned int [::1] palette,
        float [:, ::1] fire,
        unsigned int intensity = 0,
        unsigned int low       = 0,
        unsigned int high      = 0,
):
    """

    Create a fire effect

    :param width    : 
        integer; max width of the effect
        
    :param height   : 
        integer; max height of the effect
        
    :param factor   : 
        float; factor to reduce the flame effect
        
    :param palette  : 
        ndarray; Color palette 1d numpy array (colors buffer unsigned int values)
        
    :param fire     : 
        ndarray; 2d array (x, y) (contiguous) containing float values
        
    :param intensity: 
        integer; Control the flame intensity default 0 (low intensity), range [0...32]
        
    :param low      : 
        integer; The x lowest position of the effect, x must be >=0 and < high
        
    :param high     : 
        integer; The x highest position of the effect, x must be > low and <= high
        
    :return         : 
        Return a numpy array containing the fire effect array shape (w, h, 3) of RGB pixels
        
    """
    # todo array out can be passed as an argument and the cdef function can
    #  changed to nogil
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
        assert high <= width, "Argument high must be <= w"

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
                fire[height - 1, x] = <float>randRange(intensity, <unsigned int>260)


        # DILUTE THE FLAME EFFECT (DECREASE THE MAXIMUM INT VALUE) WHEN THE FLAME TRAVEL UPWARD
        for y in prange(1, height - 1, schedule=SCHEDULE, num_threads=THREADS):

            # c1 = (y + 1) % height
            c1 = (y + 1) if (y + 1) < height else (y + 1) - height

            for x in range(0, width):

                    # c2 = x % width
                    c2 = x if x < width else x - width

                    d = (fire[c1, (x - 1) if (x - 1) < width else (x - 1) - width ] # (x - 1) % width]
                       + fire[c1, c2]
                       + fire[c1, (x + 1) if (x + 1) < width else (x + 1) - width ] # (x + 1) % width]
                       + fire[(y + 2) % height, c2]) * factor

                    d = d - <float>(<float>rand() * <float>0.0001)

                    # Cap the values
                    if d < 0:
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
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
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
    Apply a fire shader effect to a given surface, simulating fire behavior with customizable 
    visual aspects such as intensity, bloom, and color adjustments.

    This effect can be used in games or simulations to create dynamic fire effects, allowing 
    for control over visual details like fire intensity, bloom effects, color palettes, and 
    orientation of the flame.

    ### Parameters:
    - **width_** (int): Width of the surface or display in pixels.
    - **height_** (int): Height of the surface or display in pixels.
    - **factor_** (float): Controls the fire height; should be in the range [3.95, 4.2].  
      A value of 3.95 results in the highest flame effect.
    - **palette_** (numpy.ndarray): A palette containing RGB color mappings (uint8 values).
    - **fire_** (numpy.ndarray): 2D array representing fire intensity values. 
      For optimal performance, the array size should match the texture after applying the `reduce_factor_`.
    - **reduce_factor_** (int, optional): Controls texture resolution (range [0, 4]). 
      Smaller values improve performance but may degrade fire appearance.
    - **fire_intensity_** (int, optional): Intensity of the fire at its base (range [0, 32]),
     where 32 represents maximum intensity.
    - **smooth_** (bool, optional): Whether to use smooth scaling (True) or a faster pixelated effect (False).
    - **bloom_** (bool, optional): Enables bloom effect to enhance light around the fire.
    - **fast_bloom_** (bool, optional): A faster but less accurate bloom effect.
    - **bpf_threshold_** (int, optional): Brightness pass filter threshold (range [0, 255]), controlling bloom intensity.
    - **low_** (int, optional): X-coordinate for the fire's starting position.
    - **high_** (int, optional): X-coordinate for the fire's ending position.
    - **brightness_** (bool, optional): Whether to apply brightness adjustment to the fire effect.
    - **brightness_intensity_** (float, optional): Intensity of the brightness effect, in the range [-1.0, 1.0].
    - **surface_** (pygame.Surface, optional): An optional surface to reuse for performance improvements, 
    instead of creating a new one each time.
    - **adjust_palette_** (bool, optional): Whether to adjust the color palette using the HSL model.
    - **hsl_** (tuple, optional): Tuple (hue, saturation, lightness) to modify the palette colors.
    - **transpose_** (bool, optional): If True, transposes the fire effect (swaps width and height).
    - **border_** (bool, optional): If True, the fire effect will burn along the border of the texture.
    - **blur_** (bool, optional): If True, applies a blur effect to smooth out jagged edges.

    ### Returns:
    - **pygame.Surface**: A surface containing the fire effect, ready to be blitted onto the game display.

    ### Example Usage:
    ```python
    palette = make_palette(256, 0.1, 350, 1.2)
    fire_array = numpy.zeros((height, width), dtype=numpy.float32)
    fire_surface = fire_effect(
        width_=800, height_=1024, factor_=3.95, palette_=palette, fire_=fire_array, 
        low_=30, high_=770, reduce_factor_=3
    )
    SCREEN.blit(fire_surface, (0, 0))
    ```

    ### Notes:
    - **Texture Size & Performance**: Adjust the `reduce_factor_` to balance between performance and visual quality.
    - **Fire Behavior**: Adjust the `factor_`, `fire_intensity_`, and `reduce_factor_` 
        for different fire behavior effects.
    - **Bloom & Brightness**: The bloom effect can be controlled by `bloom_` and `bpf_threshold_`. 
        Adjust brightness with `brightness_` and `brightness_intensity_`.
    - **Palette Adjustments**: Use `adjust_palette_` and `hsl_` to customize the flame colors.
    """

    # todo reduce_factor=0 and border = True crash

    assert reduce_factor_ in (0, 1, 2, 3, 4), \
        "Argument reduce factor must be in range 0 ... 4 " \
        "\n reduce_factor_ = 1 correspond to dividing the image size by 2" \
        "\n reduce_factor_ = 2 correspond to dividing the image size by 4"
    assert 0 <= fire_intensity_ < 33, \
        "Argument fire_intensity_ must be in range [0 ... 32] got %s" % fire_intensity_

    assert width_ > 0 and height_ > 0, "Argument w or h cannot be null or < 0"
    assert factor_ > 0, "Argument amplitude cannot be null or < 0"

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
@cython.exceptval(check=False)
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
    Apply a fire shader effect to a given surface, simulating fire behavior with customizable 
    visual aspects such as intensity, bloom, and color adjustments.

    This effect can be used in games or simulations to create dynamic fire effects, allowing 
    for control over visual details like fire intensity, bloom effects, color palettes, and 
    orientation of the flame.

    ### Parameters:
    - **width_** (int): Width of the surface or display in pixels.
    - **height_** (int): Height of the surface or display in pixels.
    - **factor_** (float): Controls the fire height; should be in the range [3.95, 4.2].  
      A value of 3.95 results in the highest flame effect.
    - **palette_** (numpy.ndarray): A palette containing RGB color mappings (uint8 values).
    - **fire_** (numpy.ndarray): 2D array representing fire intensity values. 
      For optimal performance, the array size should match the texture after applying the `reduce_factor_`.
    - **reduce_factor_** (int, optional): Controls texture resolution (range [0, 4]). 
      Smaller values improve performance but may degrade fire appearance.
    - **fire_intensity_** (int, optional): Intensity of the fire at its base (range [0, 32]),
     where 32 represents maximum intensity.
    - **smooth_** (bool, optional): Whether to use smooth scaling (True) or a faster pixelated effect (False).
    - **bloom_** (bool, optional): Enables bloom effect to enhance light around the fire.
    - **fast_bloom_** (bool, optional): A faster but less accurate bloom effect.
    - **bpf_threshold_** (int, optional): Brightness pass filter threshold (range [0, 255]), controlling bloom intensity.
    - **low_** (int, optional): X-coordinate for the fire's starting position.
    - **high_** (int, optional): X-coordinate for the fire's ending position.
    - **brightness_** (bool, optional): Whether to apply brightness adjustment to the fire effect.
    - **brightness_intensity_** (float, optional): Intensity of the brightness effect, in the range [-1.0, 1.0].
    - **surface_** (pygame.Surface, optional): An optional surface to reuse for performance improvements, 
    instead of creating a new one each time.
    - **adjust_palette_** (bool, optional): Whether to adjust the color palette using the HSL model.
    - **hsl_** (tuple, optional): Tuple (hue, saturation, lightness) to modify the palette colors.
    - **transpose_** (bool, optional): If True, transposes the fire effect (swaps width and height).
    - **border_** (bool, optional): If True, the fire effect will burn along the border of the texture.
    - **blur_** (bool, optional): If True, applies a blur effect to smooth out jagged edges.

    ### Returns:
    - **pygame.Surface**: A surface containing the fire effect, ready to be blitted onto the game display.

    ### Example Usage:
    ```python
    palette = make_palette(256, 0.1, 350, 1.2)
    fire_array = numpy.zeros((height, width), dtype=numpy.float32)
    fire_surface = fire_effect(
        width_=800, height_=1024, factor_=3.95, palette_=palette, fire_=fire_array, 
        low_=30, high_=770, reduce_factor_=3
    )
    SCREEN.blit(fire_surface, (0, 0))
    ```

    ### Notes:
    - **Texture Size & Performance**: Adjust the `reduce_factor_` to balance between performance and visual quality.
    - **Fire Behavior**: Adjust the `factor_`, `fire_intensity_`, and `reduce_factor_` 
        for different fire behavior effects.
    - **Bloom & Brightness**: The bloom effect can be controlled by `bloom_` and `bpf_threshold_`. 
        Adjust brightness with `brightness_` and `brightness_intensity_`.
    - **Palette Adjustments**: Use `adjust_palette_` and `hsl_` to customize the flame colors.
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

        # brightness_ex_c(bgr_array=bgr_array,
        #                                      shift=brightness_intensity_, color=(0, 0, 0))
        brightness_bpf_c(rgb_array_, brightness_intensity_, 16)

    if blur_:
        blur3d_c(rgb_array_)

    if transpose_:
        rgb_array_ = rgb_array_.transpose(1, 0, 2)
        fire_surface_smallest = rotate(fire_surface_smallest, 90)


    # CONVERT THE ARRAY INTO A PYGAME SURFACE
    array_to_surface(fire_surface_smallest, rgb_array_)


    # BLOOM SHADER EFFECT
    if bloom_:
        assert 0 <= bpf_threshold_ < 256, \
            "Argument bpf_threshold_ must be in range [0 ... 256] got %s " % bpf_threshold_
        try:
            # fire_surface_smallest = shader_bloom_fast(
            #     fire_surface_smallest, bpf_threshold_, fast=fast_bloom_, amplitude=1)

            shader_bloom_fast1(
                fire_surface_smallest,
                threshold_ = bpf_threshold_,
                smooth_    = 0,
                saturation_= True
            )
            # bloom_c(
            #     fire_surface_smallest,
            #     threshold = bpf_threshold_,
            #     fast = fast_bloom_
            # )

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
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef inline cloud_effect(
        int width_,
        int height_,
        float factor_,
        unsigned int [::1] palette_,
        float [:, ::1] cloud_,

        # OPTIONAL PARAMETERS
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
    Generate a cloud or smoke effect for a game display.

    This function creates a procedural cloud or smoke effect using a combination of noise textures,
    color palettes, and optional post-processing effects like bloom, brightness adjustment, and blur.

    Parameters:
    -----------
    width_ : int
        Width of the cloud effect texture. Must be greater than 0.
    height_ : int
        Height of the cloud effect texture. Must be greater than 0.
    factor_ : float
        Controls the vertical size of the cloud effect. 
        - Values > 3.95 keep the effect within the display height.
        - Values < 3.95 allow the effect to extend beyond the display height.
    palette_ : numpy.ndarray or cython memoryview
        Color palette for the cloud effect, represented as an array of RGB values.
    cloud_ : numpy.ndarray
        Cloud intensity array, where values range from [0, 32]. For optimal performance,
        the size of this array should match the reduced texture dimensions.

    Optional Parameters:
    -------------------
    reduce_factor_ : int, optional (default=2)
        Controls the resolution of the cloud texture for performance optimization.
        - Values: 0 (no reduction), 1 (reduce by 2), 2 (reduce by 4), etc.
        - Recommended: 2 or 3 for a balance of performance and visual quality.
    cloud_intensity_ : int, optional (default=16)
        Intensity of the cloud effect at its source. Range: [0, 32].
    smooth_ : bool, optional (default=True)
        If True, applies smooth scaling (bi-linear filtering) for a smoother appearance.
        If False, uses basic scaling for a pixelated look.
    bloom_ : bool, optional (default=False)
        If True, applies a bloom effect to areas of high intensity.
    fast_bloom_ : bool, optional (default=True)
        If True, uses a faster but less accurate bloom effect for better performance.
    bpf_threshold_ : int, optional (default=128)
        Brightness threshold for the bloom effect. Range: [0, 255].
    low_ : int, optional (default=0)
        Starting horizontal position of the cloud effect. Range: [0, width_].
    high_ : int, optional (default=0)
        Ending horizontal position of the cloud effect. Range: [0, width_].
    brightness_ : bool, optional (default=False)
        If True, adjusts the brightness of the cloud effect.
    brightness_intensity_ : float, optional (default=0.0)
        Controls the brightness adjustment intensity. Range: [-1.0, 1.0].
        - Negative values decrease brightness.
        - Positive values increase brightness.
        - Values below -0.4 make the effect translucent.
    surface_ : pygame.Surface, optional (default=None)
        Optional surface to reuse for rendering, improving performance.
        Must match the reduced texture dimensions.
    transpose_ : bool, optional (default=False)
        If True, transposes the cloud texture, swapping width and height.
    blur_ : bool, optional (default=True)
        If True, applies a blur effect to soften the cloud edges.

    Returns:
    --------
    pygame.Surface
        A surface containing the generated cloud effect, ready to be blitted to the game display.

    Raises:
    -------
    AssertionError
        - If `reduce_factor_` is not in the range [0, 4].
        - If `width_` or `height_` is less than or equal to 0.
        - If `factor_` is less than or equal to 0.
    """

    # Validate input parameters
    assert reduce_factor_ in (0, 1, 2, 3, 4), \
        "Argument reduce_factor_ must be in range 0 ... 4. " \
        "\n reduce_factor_ = 1 corresponds to dividing the image size by 2" \
        "\n reduce_factor_ = 2 corresponds to dividing the image size by 4"

    assert width_ > 0 and height_ > 0, "Arguments width_ and height_ must be greater than 0."
    assert factor_ > 0, "Argument factor_ must be greater than 0."

    # Call the core cloud effect function with all parameters
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
@cython.cdivision(False)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
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
    Generate a cloud/smoke effect as a 24-bit RGB pixel array.

    This function is a low-level utility for creating a cloud or smoke effect. It is not intended
    to be called directly; instead, it is used internally by the `cloud_effect` function.

    Parameters:
    -----------
    width : int
        Width of the surface or display in pixels. Must be greater than 0.
    height : int
        Height of the surface or display in pixels. Must be greater than 0.
    factor : float
        Controls the vertical size of the cloud effect. 
        - Values in the range [3.95, 4.2] are recommended.
        - A value of 3.95 fills the entire display with the cloud effect.
        - Values above 3.95 shrink the cloud effect vertically.
    palette : numpy.ndarray or cython memoryview
        A buffer containing mapped RGB colors as unsigned integers.
    cloud_ : numpy.ndarray
        A 2D array of shape (width, height) containing float values representing cloud intensity.
    intensity : int, optional (default=0)
        Determines the guaranteed amount of smoke generated at the base of the effect.
        - Values must be in the range [0, 260].
        - If 0, a random value between 0 and 260 is assigned.
        - If 250, a random value between 250 and 260 is assigned.
    low : int, optional (default=0)
        The starting X position of the cloud effect on the display.
        - Example: A value of 100 means the effect starts at the 100th pixel (the first 100 pixels
          are unaffected).
    high : int, optional (default=0)
        The ending X position of the cloud effect on the display.
        - Example: A value of 800 means the effect ends at the 800th pixel (the remaining pixels
          are unaffected).

    Returns:
    --------
    numpy.ndarray
        A 3D array of shape (width, height, 3) containing RGB pixel values for the cloud/smoke effect.

    Notes:
    ------
    - This function is optimized for internal use and should not be called directly.
    - Ensure that `cloud_` and `palette` are properly initialized to avoid unexpected behavior.
    - The `factor` parameter significantly impacts the visual appearance of the cloud effect.
    """

    cdef:
        int new_height = height
        unsigned char [:, :, ::1] out = empty((width, new_height, 3), dtype=uint8)
        int x = 0, y = 0
        float d
        unsigned int ii=0
        unsigned c1 = 0, c2 = 0
        int p_length = (<object>palette).size
        int height_1

    cdef int min_, max_, middle

    if low != 0 or high != 0:
        assert 0 <= low < high, "Argument low_ must be < high_"
        assert high <= width,   "Argument high must be <= w"

        middle = low + ((high - low) >> 1)
        min_ = randRange(low, middle)
        max_ = randRange(middle + 1, high)
    else:
        middle = width >> 1
        min_ = randRange(0, middle)
        max_ = randRange(middle +1, width)

    height_1 = new_height - 1
    with nogil:
        # POPULATE TO THE BASE OF THE FIRE (THIS WILL CONFIGURE THE FLAME ASPECT)
        for x in prange(min_, max_, schedule=SCHEDULE, num_threads=THREADS):
                # cloud_[(new_height - 1) % height, x % width] = randRange(intensity, 260)
                cloud_[height_1 - height if height_1 >= height else height_1, x - width if x>width else x ] = \
                    randRange(intensity, 260)

        # DILUTE THE FLAME EFFECT (DECREASE THE MAXIMUM INT VALUE) WHEN THE FLAME TRAVEL UPWARD
        for y in prange(0, new_height - 1, schedule=SCHEDULE, num_threads=THREADS):
            
            # c1 = (y + 1) % height
            c1 = ( y + 1 ) - height if ( y + 1 ) > height else ( y + 1 )

            for x in range(0, width):

                    # c2 = x % width
                    c2 = x - width if x > width else x 

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
@cython.exceptval(check=False)
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
    Generate a cloud or smoke effect for a game display.

    This function creates a procedural cloud or smoke effect using noise textures, color palettes,
    and optional post-processing effects like bloom, brightness adjustment, and blur. It is optimized
    for performance and flexibility, allowing fine-grained control over the appearance and behavior
    of the effect.

    Parameters:
    -----------
    width_ : int
        Width of the cloud texture in pixels. Must be greater than 0.
    height_ : int
        Height of the cloud texture in pixels. Must be greater than 0.
    factor_ : float
        Controls the vertical size of the cloud effect.
        - Values > 3.95 contain the effect within the display height.
        - Values < 3.95 extend the effect beyond the display height.
        Recommended value: 3.95 for optimal results.
    palette_ : numpy.ndarray or cython memoryview
        A buffer containing mapped RGB colors as unsigned integers.
    cloud_ : numpy.ndarray
        A 2D array of shape (width, height) containing float values representing cloud intensity.
        For best performance, the array size should match the reduced texture dimensions.

    Optional Parameters:
    -------------------
    reduce_factor_ : int, optional (default=2)
        Controls the resolution of the cloud texture for performance optimization.
        - Values: 0 (no reduction), 1 (reduce by 2), 2 (reduce by 4), etc.
        - Recommended: 2 or 3 for a balance of performance and visual quality.
    cloud_intensity_ : int, optional (default=16)
        Intensity of the cloud effect at its source. Range: [0, 32].
    smooth_ : bool, optional (default=True)
        If True, applies smooth scaling (bi-linear filtering) for a smoother appearance.
        If False, uses basic scaling for a pixelated look.
    bloom_ : bool, optional (default=False)
        If True, applies a bloom effect to areas of high intensity.
    fast_bloom_ : bool, optional (default=True)
        If True, uses a faster but less accurate bloom effect for better performance.
    bpf_threshold_ : int, optional (default=128)
        Brightness threshold for the bloom effect. Range: [0, 255].
    low_ : int, optional (default=0)
        Starting horizontal position of the cloud effect. Range: [0, width_].
    high_ : int, optional (default=0)
        Ending horizontal position of the cloud effect. Range: [0, width_].
    brightness_ : bool, optional (default=False)
        If True, adjusts the brightness of the cloud effect.
    brightness_intensity_ : float, optional (default=0.0)
        Controls the brightness adjustment intensity. Range: [-1.0, 1.0].
        - Negative values decrease brightness.
        - Positive values increase brightness.
        - Values below -0.4 make the effect translucent.
    surface_ : pygame.Surface, optional (default=None)
        Optional surface to reuse for rendering, improving performance.
        Must match the reduced texture dimensions.
    transpose_ : bool, optional (default=False)
        If True, transposes the cloud texture, swapping width and height.
    blur_ : bool, optional (default=True)
        If True, applies a blur effect to soften the cloud edges.

    Returns:
    --------
    pygame.Surface
        A surface containing the generated cloud effect, ready to be blitted to the game display.

    Notes:
    ------
    - This function is optimized for internal use and should not be called directly.
    - Ensure that `cloud_` and `palette_` are properly initialized to avoid unexpected behavior.
    - The `factor_` parameter significantly impacts the visual appearance of the cloud effect.
    - For best performance, use `reduce_factor_` values of 2 or 3 and ensure `cloud_` matches the
      reduced texture dimensions.
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
        brightness_ex_c(rgb_array_=rgb_array_,
                                              shift_=brightness_intensity_, color_=(0, 0, 0))

    if blur_:
        blur3d_c(rgb_array_)

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
        bloom_c(cloud_surface_smallest, bpf_threshold_, fast_=fast_bloom_)

    # RESCALE THE SURFACE TO THE FULL SIZE
    if smooth_:
        cloud_effect = smoothscale(cloud_surface_smallest, (width_, height_))
    else:
        cloud_effect = scale(cloud_surface_smallest, (width_, height_))

    return cloud_effect
