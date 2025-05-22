# cython: binding=False, boundscheck=False, wraparound=False, nonecheck=False, cdivision=True,
# profile=False, initializedcheck=False, exceptval(check=False)
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

"""
# Python Library Summary

The provided code appears to be part of a Python module for image manipulation and processing, 
specifically with an emphasis on **gradient generation**, **image channel manipulation**, 
**color space calculations**, and **sorting algorithms**. Let's break down the key functionality
 and purpose of the different functions and methods in the module:

## 1. **Gradient Generation Functions**
These functions create various types of gradients, which are often used in graphical applications 
for creating smooth transitions between colors.

- **`create_line_gradient_rgb`**: Generates a linear gradient of RGB values between two specified
     colors (`start_rgb` and `end_rgb`).
- **`create_line_gradient_rgba`**: Similar to the above, but for RGBA colors (adds an alpha channel for transparency).
- **`horizontal_rgb_gradient`**: Creates a horizontal gradient in RGB color space.
- **`horizontal_rgba_gradient`**: Creates a horizontal gradient in RGBA color space.
- **`create_radial_gradient`**: Creates a radial gradient that radiates outward from a point, 
    with customizable start and end colors and offsets. A precomputed gradient can be provided to optimize performance.
- **`create_radial_gradient_alpha`**: Similar to `create_radial_gradient`, but specifically for handling RGBA with transparency.
- **`create_quarter_radial_gradient`** and **`create_quarter_radial_gradient_alpha`**: 
These create gradients for a quarter of a circle (sector-based gradients), useful for specialized visual effects.

## 2. **Color Space and Color Matching Functions**
These functions are used to work with different color spaces (HSV, HSL, RGB) and find closest color
 matches, which is useful for color-based image processing or palette generation.

- **`color_dist_hsv`**: Calculates the distance between two colors in the HSV color space.
- **`color_dist_hsl`**: Calculates the distance between two colors in the HSL color space.
- **`closest_hsv_color`** and **`closest_hsl_color`**: Find the closest color in a given palette 
    to a target color, using the HSV or HSL color space respectively.
- **`close_color`**: Finds a color in a palette that is close to a given color (presumably using RGB space).
- **`use_palette`**: Similar to `close_color`, but it likely selects colors from a palette based 
    on a defined selection criterion.

## 3. **Image Manipulation Functions**
These functions focus on manipulating images or buffers, particularly for scrolling, blending, 
and combining channels.

- **`swap_image_channels`**: Swaps color channels (such as RGB) in an image surface, useful when 
    working with different color formats or performing transformations.
- **`scroll_surface_24bit`**, **`scroll24_inplace`**, **`scroll_rgb_array_inplace`**, 
    **`scroll_alpha_inplace`**: These functions allow for scrolling of an image or image channels in 
    the x and y directions, which can be used for animations or visual effects. The `c` versions are 
    likely optimized implementations written in C for performance.
- **`combine_rgba_buffers`**: Combines a BGR image buffer and an alpha channel into a single RGBA image buffer.
     This is useful for working with images that have separate color and alpha channels.
- **`normalize_2d_array`**: Normalizes a 2D array, possibly for color or pixel intensity normalization.

## 4. **Utility Functions**
These functions perform a variety of utility tasks, including sorting and random number generation.

- **`rgb_to_int`** and **`int_to_rgb`**: Convert between RGB values and their integer representation
     (often used for packing/unpacking colors).
- **`_randf`**, **`_randi`**: Generate random float and integer values within a specified range, 
    likely for random color or pixel generation.
- **`bubble_sort`**, **`insertion_sort`**, **`quick_sort`**, **`heap_sort`**: These are different 
    sorting algorithms implemented for sorting arrays of pixel or image data. `heap_sort` and `quick_sort`
     are especially useful for sorting large datasets or pixel values efficiently.
- **`partition_cython`**, **`_quick_sort`**, **`heapify`**: These are helper functions for sorting 
    algorithms like Quick Sort and Heap Sort, optimized for performance in Cython.

## 5. **Image Format and Type Checking**
These functions help determine the type of image data or buffers.

- **`get_image_format`**: Likely checks the format of an image (whether it is in RGBA, RGB, etc.).
- **`is_type_memoryview`**, **`is_uint8`**, **`is_float64`**, **`is_int32`**: These functions check 
the type and datatype of a given image array or buffer, useful for validating input and ensuring the correct data format.

## Overall Purpose of the Library:
The library is designed for **image processing and graphical operations**, with a strong emphasis 
on **gradient creation**, **image channel manipulation**, **color transformations**, **sorting algorithms**,
 and **optimization**. It provides functions to create smooth color transitions (gradients), 
 manipulate pixel data, check image formats, and optimize computationally expensive operations using 
 algorithms like QuickSort and HeapSort, as well as providing support for custom gradient creation 
 (both radial and linear). The functions are highly optimized with `cpdef` and `cdef` to allow seamless
  integration between Python and C for performance-critical tasks.

## Use Cases:
- **Image Processing**: Manipulating and transforming images, handling RGBA/BGR channels, 
    scrolling image pixels, and applying color transformations.
- **Graphics Rendering**: Generating gradients (linear, radial, quarter radial) for background effects 
    or visual transitions.
- **Color Matching**: Finding the closest matching colors in a given color palette based on various 
    color spaces (HSV, HSL).
- **Performance Optimization**: Utilizing sorting and color manipulation algorithms to efficiently
    process large images or datasets, with performance improvements using Cython (`cpdef`/`cdef`).

This library would be useful in applications like **image editors**, **graphic design tools**, 
**data visualization**, or **real-time graphical applications**.

"""

try:
    cimport cython
    from cython.parallel cimport prange

except ImportError:
    raise ImportError("\n<cython> library is missing on your system."
          "\nTry: \n   C:\\pip install cython on a window command prompt.")


try:
    cimport cython
    from cython.parallel cimport prange
    from cpython cimport PyObject_CallFunctionObjArgs, PyObject, \
        PyList_SetSlice, PyObject_HasAttr, PyObject_IsInstance, \
        PyObject_CallMethod, PyObject_CallObject, PyTuple_GET_ITEM, PyTuple_Size, \
        PyTuple_Check


except ImportError:
    raise ImportError("\n<cython> library is missing on your system."
          "\nTry: \n   C:\\pip install cython on a window command prompt.")


import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

try:
    import pygame
    from pygame import Rect
    from pygame.math import Vector2
    from pygame import Rect, BLEND_RGB_ADD, HWACCEL
    from pygame import Surface, SRCALPHA, mask, RLEACCEL
    from pygame.transform import rotate, scale, smoothscale
    from pygame.surfarray import array3d, pixels3d, array_alpha, pixels_alpha, \
        make_surface
    from pygame.image import frombuffer

except ImportError:
    raise ImportError("\n<Pygame> library is missing on your system."
          "\nTry: \n   C:\\pip install pygame on a window command prompt.")


# NUMPY IS REQUIRED
try:
    import numpy
    from numpy import ndarray, zeros, empty, uint8, int32, float64, \
        float32, dstack, full, ones, asarray, ascontiguousarray, full_like,\
        add, putmask, int16, arange, repeat, newaxis
except ImportError:
    raise ImportError("\n<numpy> library is missing on your system."
          "\nTry: \n   C:\\pip install numpy on a window command prompt.")

from PygameShader.shader cimport hsv, rgb, struct_rgb_to_hsv
from libc.math cimport roundf as round_c
from libc.math cimport floorf  as floor_c, sqrtf as sqrt
from libc.math cimport fabsf as abs_c
from libc.stdlib cimport malloc, free
from libc.stdio cimport printf
from libc.string cimport memcpy


cimport numpy as np

DEF SCHEDULE = 'static'
from PygameShader.config import OPENMP, THREAD_NUMBER, __VERSION__

cdef int THREADS = 1

if OPENMP:
     THREADS = THREAD_NUMBER


# todo tidy up unused DEF
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




@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef swap_image_channels(image_surface, channel_order):
    """
    Swap or nullify the channels of an image based on the specified channel order string.

    This function allows you to swap the color channels (RGB) of an image represented as a 
    pygame.Surface. You can specify the new channel order using a string where each 
    letter represents a color channel (R, G, B) or '0' to nullify a channel (remove it).

    The channel order string should contain exactly 3 characters, representing the desired order of the 
    channels. For example:
    - 'RGB' will keep the original channel order.
    - 'R0B' will swap the red and blue channels while removing the green channel.
    - 'BGR' will swap the channels to blue, green, red order.

    :param image_surface: pygame.Surface
        The image (pygame Surface) whose channels are to be swapped.

    :param channel_order: str
        A string representing the desired channel order. It must contain exactly 3 characters,
        where 'R', 'G', and 'B' represent the respective color channels, and '0' nullifies a channel.
        Example values: 'RGB', 'RBG', 'GRB', 'BGR', 'BRG', 'R0B', etc.

    :return: pygame.Surface
        A new 24-bit pygame surface with the swapped channels.

    :raises ValueError:
        If the channel order string is not valid or if the surface format is incompatible.
    """

    # Assert that the input is a valid pygame.Surface object
    assert PyObject_IsInstance(image_surface, Surface), \
        f"Expecting Surface for argument image_surface got {type(image_surface)}"

    # Validate that channel order string is exactly 3 characters long
    if len(channel_order) != 3:
        raise ValueError("Channel order string must have exactly 3 characters. "
                         "Choose from: 'RGB', 'RBG', 'GRB', 'GBR', 'BRG', 'BGR', etc.")

    # Validate that all characters in the channel order string are valid ('R', 'G', 'B', '0')
    for c in channel_order:
        if c not in ('R', 'G', 'B', '0'):
            raise ValueError(f"Invalid channel order string: '{channel_order}'. "
                "Channel order string must be composed of 'R', 'G', 'B' and '0' (e.g., 'RGB', 'R0B').")

    # Extract the channel order from the string
    red_channel, green_channel, blue_channel = list(channel_order)
    channel_index_map = {'R': 0, 'G': 1, 'B': 2, '0': -1}

    # Get the width and height of the surface
    cdef Py_ssize_t width, height
    width, height = image_surface.get_size()

    # Attempt to get the pixel data from the surface
    try:
        rgb_view = image_surface.get_view('3')  # Try accessing the surface pixel data as a view

    except (pygame.error, ValueError) as e:
        # More specific error handling for incompatible pixel formats
        raise ValueError(f"Unable to retrieve pixel data from the surface: {e}")

    # Create a new array to store the channel-swapped image
    cdef:
        unsigned char [:, :, :] rgb_array = rgb_view
        unsigned char [:, :, ::1] swapped_pixel_array = empty((height, width, 3), dtype=uint8)
        int i = 0, j = 0
        short int red_index, green_index, blue_index

    # Get the respective channel indices for red, green, blue
    red_index = channel_index_map[red_channel]
    green_index = channel_index_map[green_channel]
    blue_index = channel_index_map[blue_channel]

    # Perform the channel swapping operation in parallel using nogil for better performance
    with nogil:
        for j in prange(height, schedule=SCHEDULE, num_threads=THREADS):
            for i in range(width):
                # Set the red channel (or null if '0')
                if red_index == -1:
                    swapped_pixel_array[j, i, 0] = 0
                else:
                    swapped_pixel_array[j, i, 0] = rgb_array[i, j, red_index]

                # Set the green channel (or null if '0')
                if green_index == -1:
                    swapped_pixel_array[j, i, 1] = 0
                else:
                    swapped_pixel_array[j, i, 1] = rgb_array[i, j, green_index]

                # Set the blue channel (or null if '0')
                if blue_index == -1:
                    swapped_pixel_array[j, i, 2] = 0
                else:
                    swapped_pixel_array[j, i, 2] = rgb_array[i, j, blue_index]

    # Convert the new array back to a pygame.Surface and return it
    return frombuffer(swapped_pixel_array, (width, height), 'RGB')






@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef np.ndarray[np.uint8_t, ndim=2] create_line_gradient_rgb(
        int num_pixels,
        tuple start_rgb = (255, 0, 0),
        tuple end_rgb   = (0, 255, 0)
):
    """
    Generate a 2D horizontal gradient of RGB colors from a start color to an end color.

    This function creates an array of RGB values representing a smooth horizontal gradient 
    transitioning from the specified `start_rgb` to the `end_rgb`. The gradient will 
    consist of `num_pixels` number of pixels, with the color gradually changing from the start color 
    to the end color across the 1D array.

    :param num_pixels : int; The number of pixels in the gradient (must be > 0).
                        Determines the length of the gradient in the horizontal direction.

    :param start_rgb  : tuple; A tuple representing the RGB values of the starting color 
                        (default is red: (255, 0, 0)).

    :param end_rgb    : tuple; A tuple representing the RGB values of the ending color 
                        (default is green: (0, 255, 0)).

    :return           : numpy.ndarray; A 2D numpy array of shape (num_pixels, 3), where each row 
                        represents an RGB color. The array contains the pixel values of the gradient, 
                        with the color transitioning from the `start_rgb` to the `end_rgb`.

    :raises ValueError: 
        If the `num_pixels` is less than or equal to 0.
        If `start_rgb` or `end_rgb` are not valid RGB tuples of length 3, or if any of the 
        color values are out of the valid RGB range [0, 255].
        If `num_pixels` is not a positive integer.

    :raises TypeError:
        If `start_rgb` or `end_rgb` are not tuples, or if they contain non-integer values.

    Example:
        gradient = create_line_gradient_between_colors(256, start_rgb=(255, 0, 0), end_rgb=(0, 0, 255))
        # Generates a horizontal gradient from red to blue with 256 pixels.
    """

    # Check if the num_pixels is a positive integer
    if num_pixels <= 0:
        raise ValueError("Argument num_pixels must be a positive integer greater than 0")

    # Validate the start_rgb and end_rgb tuples
    if not isinstance(start_rgb, tuple) and len(start_rgb) != 3:
        raise ValueError(f"start_rgb must be a tuple of 3, got {start_rgb}")

    for c in start_rgb:
        if not(0 <= c <= 255):
            raise ValueError(f"start_rgb must be in range [0, 255], got {start_rgb}")

    if not isinstance(end_rgb, tuple) and len(end_rgb) != 3:
        raise ValueError(f"end_rgb must be a tuple of 3, got {end_rgb}")
    for c in end_rgb:
        if not (0 <= c <= 255):
            raise ValueError(f"end_rgb must be in range [0, 255], got {end_rgb}")


    cdef:
        float [:] diff_ =  \
            numpy.array(end_rgb, dtype=float32) - \
            numpy.array(start_rgb, dtype=float32)
        float [::1] row = numpy.arange(num_pixels, dtype=float32) / (num_pixels - 1.0)
        unsigned char [:, ::1] rgb_gradient = empty((num_pixels, 3), dtype=uint8)
        float [3] start = numpy.array(start_rgb, dtype=float32)
        int i=0
        float * row_

    with nogil:
        for i in prange(num_pixels, schedule=SCHEDULE, num_threads=THREADS):
           row_ = &row[i]
           rgb_gradient[i, 0] = <unsigned char>(start[0] + row_[0] * diff_[0])
           rgb_gradient[i, 1] = <unsigned char>(start[1] + row_[0] * diff_[1])
           rgb_gradient[i, 2] = <unsigned char>(start[2] + row_[0] * diff_[2])

    return asarray(rgb_gradient)




@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef np.ndarray[np.uint8_t, ndim=2] create_line_gradient_rgba(
        int num_pixels,
        tuple start_rgba = (255, 0, 0, 255),
        tuple end_rgba   = (0, 255, 0, 0)
):
    """
    Generate a 2D horizontal gradient of RGB(A) colors from a start color to an end color.

    This function creates an array of RGB(A) values representing a smooth horizontal gradient 
    transitioning from the specified `start_rgba` to the `end_rgba`. The gradient will 
    consist of `num_pixels` number of pixels, with the color gradually changing from the start color 
    to the end color across the 1D array.
    
    :param num_pixels : int; The number of pixels in the gradient (must be > 0).
                        Determines the length of the gradient in the horizontal direction.

    :param start_rgba  : tuple; A tuple representing the RGB(A) values of the starting color 
                        (default is red: (255, 0, 0, 255)).

    :param end_rgba    : tuple; A tuple representing the RGB(A) values of the ending color 
                        (default is green: (0, 255, 0, 0)).

    :return           : numpy.ndarray; A 2D numpy array of shape (num_pixels, 3), where each row 
                        represents an RGB(A) color. The array contains the pixel values of the gradient, 
                        with the color transitioning from the `start_rgba` to the `end_rgba`.

    :raises ValueError: 
        If the `num_pixels` is less than or equal to 0.
        If `start_rgba` or `end_rgba` are not valid RGBA tuples of length 3, or if any of the 
        color values are out of the valid RGBA range [0, 255].
        If `num_pixels` is not a positive integer.

    :raises TypeError:
        If `start_rgba` or `end_rgba` are not tuples, or if they contain non-integer values.

    Example:
        gradient = create_line_gradient_between_colors(256, start_rgba=(255, 0, 0), end_rgba=(0, 0, 255))
        # Generates a horizontal gradient from red to blue with 256 pixels.
    """

    # Check if the num_pixels is a positive integer
    if num_pixels <= 0:
        raise ValueError("Argument num_pixels must be a positive integer greater than 0")

    # Validate the start_rgba and end_rgba tuples
    if not isinstance(start_rgba, tuple) and len(start_rgba) != 4:
        raise ValueError(f"start_rgba must be a tuple of 4, got {start_rgba}")

    for c in start_rgba:
        if not(0 <= c <= 255):
            raise ValueError(f"start_rgba must be in range [0, 255], got {start_rgba}")

    if not isinstance(end_rgba, tuple) and len(end_rgba) != 4:
        raise ValueError(f"end_rgba must be a tuple of 4, got {end_rgba}")
    for c in end_rgba:
        if not (0 <= c <= 255):
            raise ValueError(f"end_rgba must be in range [0, 255], got {end_rgba}")


    cdef:
        float [:] diff_ =  \
            numpy.array(end_rgba, dtype=float32) - \
            numpy.array(start_rgba, dtype=float32)
        float [::1] row = numpy.arange(num_pixels, dtype=float32) / (num_pixels - <float>1.0)
        unsigned char [:, ::1] rgba_gradient = empty((num_pixels, 4), dtype=uint8)
        float [4] start = numpy.array(start_rgba, dtype=float32)
        int i=0
        float * row_

    with nogil:
        for i in prange(num_pixels, schedule=SCHEDULE, num_threads=THREADS):
           row_ = &row[i]
           rgba_gradient[i, 0] = <unsigned char>(start[0] + row_[0] * diff_[0])
           rgba_gradient[i, 1] = <unsigned char>(start[1] + row_[0] * diff_[1])
           rgba_gradient[i, 2] = <unsigned char>(start[2] + row_[0] * diff_[2])
           rgba_gradient[i, 3] = <unsigned char>(start[3] + row_[0] * diff_[3])

    return asarray(rgba_gradient)



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef object horizontal_rgb_gradient(
        int w,
        int h,
        tuple color_start=(255, 0, 0),
        tuple color_end=(0, 255, 0)
):
    """
    Generates a 24-bit horizontal gradient between two RGB colors.

    This function creates a smooth transition from `color_start` to `color_end`
    across a surface of the specified dimensions.

    **Parameters:**
    :param w: int
        The width of the gradient surface in pixels (must be > 1).
        
    :param h: int
        The height of the gradient surface in pixels (must be > 0).
        
    :param color_start: tuple (R, G, B)
        The starting color as an (R, G, B) tuple, where each value is 0-255.
        
    :param color_end: tuple (R, G, B)
        The ending color as an (R, G, B) tuple, where each value is 0-255.

    **Raises:**
    :raises ValueError:
        If `w` or `h` is not valid.
        If `color_start` or `color_end` is not a valid RGB tuple.

    **Returns:**
    :return: pygame.Surface
        A 24-bit RGB pygame.Surface object of size (`w`, `h`) with the gradient applied.

    **Example Usage:**
    ```python

    # Create a gradient from red to green
    gradient = horizontal_rgb_gradient(500, 300, (255, 0, 0), (0, 255, 0))

    ```

    **Credits:** Function improved with collaboration from ChatGPT.
    """

    cdef int ii

    # Validate width and height
    if w <= 1:
        raise ValueError("w (width) must be greater than 1.")
    if h <= 0:
        raise ValueError("h (height) must be a positive integer.")

    # Validate color tuples (Cython-friendly way)
    if not PyTuple_Check(color_start) or PyTuple_Size(color_start) != 3:
        raise ValueError("color_start must be a tuple of three integers (0-255).")

    if not PyTuple_Check(color_end) or PyTuple_Size(color_end) != 3:
        raise ValueError("color_end must be a tuple of three integers (0-255).")

    # Ensure each color component is within the valid range (0-255)
    for ii in range(3):
        if not (0 <= color_start[ii] <= 255):
            raise ValueError("color_start values must be in the range 0-255.")

        if not (0 <= color_end[ii] <= 255):
            raise ValueError("color_end values must be in the range 0-255.")

    # Cython-optimized gradient generation
    cdef:
        # Compute the difference between start and end colors for interpolation
        float [:] diff_ =  \
            numpy.array(color_end, dtype=float32) - \
            numpy.array(color_start, dtype=float32)

        # Create a row gradient factor (ranging from 0 to 1) across the width
        float [::1] row = numpy.arange(w, dtype=float32) / (w - <float>1.0)

        # Allocate memory for the RGB gradient array (unsigned 8-bit integers)
        unsigned char [:, :, ::1] rgb_gradient = empty((h, w, 3), dtype=uint8)

        # Convert color_start to a float array for fast computation
        float [3] start = numpy.array(color_start, dtype=float32)

        int j = 0  # Loop variable for height
        int i = 0
        float * row_  # Pointer to row data

    # Use Cython's nogil for parallel processing (threaded execution)
    with nogil:
        for j in prange(h, schedule=SCHEDULE, num_threads=THREADS):
            for i in range(w):
                row_ = &row[i]  # Pointer to the current interpolation factor

                # Compute the gradient for each color channel (R, G, B)
                rgb_gradient[j, i, 0] = <unsigned char>(start[0] + row_[0] * diff_[0])  # Red
                rgb_gradient[j, i, 1] = <unsigned char>(start[1] + row_[0] * diff_[1])  # Green
                rgb_gradient[j, i, 2] = <unsigned char>(start[2] + row_[0] * diff_[2])  # Blue

    # Convert the generated gradient array into a pygame.Surface and return it
    return frombuffer(rgb_gradient, (w, h), "RGB")



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef object horizontal_rgba_gradient(
    int w,
    int h,
    tuple color_start=(255, 0, 0, 255),
    tuple color_end=(0, 255, 0, 0)
):
    """
    Generates a 32-bit horizontal gradient with transparency between two RGBA colors.

    This function creates a smooth horizontal transition from `color_start` to `color_end`
    across the given width while maintaining the specified height.

    **Parameters:**
    :param w: int
        The width of the gradient surface in pixels (must be > 1).

    :param h: int
        The height of the gradient surface in pixels (must be > 0).

    :param color_start: tuple[int, int, int, int]
        The starting color in RGBA format, where each component (R, G, B, A) is in the range 0-255.
        Default: (255, 0, 0, 255) (opaque red).

    :param color_end: tuple[int, int, int, int]
        The ending color in RGBA format, where each component (R, G, B, A) is in the range 0-255.
        Default: (0, 255, 0, 0) (transparent green).
        
    **Returns:**
    :return: pygame.Surface
        A 32-bit per pixel `pygame.Surface` with dimensions (w, h),
        containing the generated gradient with an alpha channel.

    **Raises:**
    :raises ValueError:
        - If `w` or `h` is not a valid positive integer.
        - If `color_start` or `color_end` is not a tuple of four integers (0-255).
        
    **Example Usage:**
    ```python

    # Create a gradient from red to green
    gradient = horizontal_rgba_gradient(500, 300, (255, 0, 0, 0), (0, 255, 0, 255))

    ```
    **Credits:** Function improved with collaboration from ChatGPT.
    
    """

    # Ensure width and height are valid
    if w <= 1:
        raise ValueError("Width (w) must be greater than 1.")
    if h <= 0:
        raise ValueError("Height (h) must be a positive integer.")

    # Validate color tuples
    if not isinstance(color_start, tuple) or len(color_start) != 4:
        raise ValueError("color_start must be a tuple of four integers (R, G, B, A) in the range 0-255.")
    if not isinstance(color_end, tuple) or len(color_end) != 4:
        raise ValueError("color_end must be a tuple of four integers (R, G, B, A) in the range 0-255.")

    cdef int channel
    # Ensure each color component is within the valid range (0-255)
    for channel in (*color_start, *color_end):
        if not isinstance(channel, int) or not (0 <= channel <= 255):
            raise ValueError("All RGBA values in color_start and color_end must be integers between 0 and 255.")

    cdef:
        # Compute the color difference between start and end colors
        float [:] diff_ = numpy.array(color_end, dtype=float32) - numpy.array(color_start, dtype=float32)

        # Create a linear interpolation array for blending colors across the width
        float [::1] row = numpy.arange(w, dtype=float32) / (w - <float>1.0)

        # Allocate memory for the gradient (h x w x 4 channels for RGBA)
        unsigned char [:, :, ::1] rgb_gradient = empty((h, w, 4), dtype=uint8)

        # Convert start color to a float array for computation
        float [4] start = numpy.array(color_start, dtype=float32)

        int i = 0, j = 0
        float * row_

    # Perform gradient calculation in parallel (multithreading enabled)
    with nogil:
        for j in prange(h, schedule=SCHEDULE, num_threads=THREADS):  # Loop through height
            for i in range(w):  # Loop through width
                row_ = &row[i]

                # Compute interpolated color values for each channel
                rgb_gradient[j, i, 0] = <unsigned char>(start[0] + row_[0] * diff_[0])  # Red
                rgb_gradient[j, i, 1] = <unsigned char>(start[1] + row_[0] * diff_[1])  # Green
                rgb_gradient[j, i, 2] = <unsigned char>(start[2] + row_[0] * diff_[2])  # Blue
                rgb_gradient[j, i, 3] = <unsigned char>(start[3] + row_[0] * diff_[3])  # Alpha (Fixed incorrect index)

    # Convert the generated gradient buffer to a Pygame surface with alpha support
    return frombuffer(rgb_gradient, (w, h), "RGBA").convert_alpha()



DEF r_max = 1.0 / 0.707106781 #inverse sqrt(0.5) or 1.0/cos45



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef create_radial_gradient(
    const int w,
    const int h,
    float offset_x              = 0.5,
    float offset_y              = 0.5,
    tuple color_start           = (255, 0, 0),
    tuple color_end             = (0, 0, 0),
    object precomputed_gradient = None,  # Renamed from gradient_array_
    float scale_factor          = r_max,  # Renamed from factor_
    unsigned short int threads  = 8
):
    """
    Generates a radial gradient (24-bit RGB) with a smooth transition between two colors, centered 
    within a rectangular surface of given width and height. The gradient is calculated using an optional 
    precomputed gradient or generated dynamically, with adjustable scale and thread control.

    The gradient radiates from the center of the surface, and the transition between colors is based 
    on the distance from the center, allowing for a variety of gradient effects (such as circular or radial).

    :param w: int
        The width of the output gradient surface in pixels (must be > 0).

    :param h: int
        The height of the output gradient surface in pixels (must be > 0).

    :param offset_x: float, optional
        The horizontal offset of the gradient center, where 0.5 represents the center of the surface 
        (default is 0.5). Adjust this value to shift the gradient horizontally.

    :param offset_y: float, optional
        The vertical offset of the gradient center, where 0.5 represents the center of the surface 
        (default is 0.5). Adjust this value to shift the gradient vertically.

    :param color_start: tuple (R, G, B), optional
        A tuple representing the starting color of the gradient in RGB format (0-255). Default is 
        (255, 0, 0), which corresponds to opaque red.

    :param color_end: tuple (R, G, B), optional
        A tuple representing the ending color of the gradient in RGB format (0-255). Default is 
        (0, 0, 0), which corresponds to black.

    :param precomputed_gradient: numpy.array, optional
        A precomputed gradient 2d array (shape: (n, 3), containing RGB values). If not provided, a 
        new gradient will be computed based on the specified color_start and color_end.

    :param scale_factor: float, optional
        A scaling factor that adjusts the radius of the gradient. A value greater than 1 will 
        increase the size of the gradient, and values smaller than 1 will reduce its radius. 
        Default is 1.4. Must be > 0.

    :param threads: int, optional
        The number of concurrent threads to use for gradient computation. The default is 8. 
        More threads may improve performance on large surfaces.

    :return: pygame.Surface
        A `pygame.Surface` object with the generated radial gradient, centered at (w/2, h/2), 
        in 24-bit RGB format.

    :raises ValueError: If `color_start` or `color_end` is not a tuple of 3 integers.
    :raises ValueError: If any RGB value in `color_start` or `color_end` is out of range (0-255).
    :raises ValueError: If `scale_factor` is None or <= 0.
    :raises ValueError: If `w` or `h` is <= 1.
    :raises TypeError: If `precomputed_gradient` is not a contiguous array.
    :raises ValueError: If `precomputed_gradient` is not a 2D array, not of type uint8, has length <= 1,
        or is not in RGB format.
        
        
    **Example Usage:**
    ```python

    precomputed_gradient = create_line_gradient_rgb(
        math.sqrt(800 ** 2 + 800 ** 2),
        start_rgb = (255, 0, 0),
        end_rgb = (0, 0, 0))
    
    surface = create_radial_gradient(800, 800,
        offset_x              = 0.5,
        offset_y              = 0.5,
        color_start           = (255, 0, 0),
        color_end             = (0, 0, 0),
        precomputed_gradient  = precomputed_gradient  
    )

    ```
    **Credits:** Function improved with collaboration from ChatGPT.  
    
    """

    # Validate color tuples
    if not isinstance(color_start, tuple) or len(color_start) != 3:
        raise ValueError("color_start must be a tuple of four integers (R, G, B) in the range 0-255.")
    if not isinstance(color_end, tuple) or len(color_end) != 3:
        raise ValueError("color_end must be a tuple of four integers (R, G, B) in the range 0-255.")

    cdef int channel
    # Ensure each color component is within the valid range (0-255)
    for channel in (*color_start, *color_end):
        if not isinstance(channel, int) or not (0 <= channel <= 255):
            raise ValueError("All RGB values in color_start and color_end must be integers between 0 and 255.")

    # Assertions to ensure valid inputs
    # Ensure valid inputs by raising ValueError with appropriate messages
    if scale_factor is None or scale_factor <= 0:
        raise ValueError("scale_factor cannot be None and  must be > 0.")

    if w <= 1:
        raise ValueError("Width (w) must be greater than 1.")

    if h <= 1:
        raise ValueError("Height (h) must be greater than 1.")

    if precomputed_gradient is not None:
        if not is_type_memoryview(precomputed_gradient):
            if isinstance(precomputed_gradient, numpy.ndarray) and not precomputed_gradient.flags[ 'C_CONTIGUOUS' ]:
                    raise TypeError("Argument 'precomputed_gradient' must be contiguous.")

            if precomputed_gradient.ndim != 2:
                raise ValueError(
                    f"Argument 'precomputed_gradient' must be a 2D array. Received: "
                    f"{(<object> precomputed_gradient).ndim} dimensions.")


            # Ensure precomputed_gradient dtype is either uint8 or float32
            if (<object>precomputed_gradient).dtype != numpy.uint8:
                raise ValueError(f"Input array precomputed_gradient must "
                                 f"be of type uint8, got {precomputed_gradient.dtype}.")

            if (<object>precomputed_gradient).shape[0] <= 1:
                raise ValueError(f"Input array precomputed_gradient array length must "
                                 f"be >1, got {precomputed_gradient.shape[0]}.")

    cdef:
        unsigned char [:, :, ::1] rgb_array = \
            empty((h, w, 3), dtype=uint8)
        float nx, ny
        float r0 = <float>sqrt(
            (<float>w * <float>scale_factor) ** 2 + (<float>h * <float>scale_factor) ** 2)

        int i, j
        unsigned int x
        float n1 = <float>1.0 / w
        float n2 = <float>1.0 / h

    cdef unsigned short int THREADS = threads

    cdef unsigned char [:, ::1] gradient_array

    # If no precomputed gradient is passed, generate a new one
    if precomputed_gradient is None:

        precomputed_gradient = create_line_gradient_rgb(
            <int>sqrt(w ** 2 + h ** 2),
            start_rgb = color_start,
            end_rgb   = color_end)

    gradient_array = precomputed_gradient

    cdef Py_ssize_t l = gradient_array.shape[0] - 1

    with nogil:
        for j in prange(h, schedule = SCHEDULE, num_threads = THREADS):

            ny = (<float> j * n2) - <float> offset_y

            for i in range(w):

                nx = (<float> i * n1) - <float> offset_x

                # position in the gradient
                x = <int> ((<float> sqrt(nx * nx + ny * ny) * r0) * <float>r_max)

                # check if the radius is greater than the size of the gradient,
                # in which case, the color is black
                if x > l:
                    rgb_array[ j, i, 0 ] = 0
                    rgb_array[ j, i, 1 ] = 0
                    rgb_array[ j, i, 2 ] = 0
                    continue
                # assign the gradient

                rgb_array[ j, i, 0 ] = <unsigned char> gradient_array[ x, 0 ]
                rgb_array[ j, i, 1 ] = <unsigned char> gradient_array[ x, 1 ]
                rgb_array[ j, i, 2 ] = <unsigned char> gradient_array[ x, 2 ]

    return frombuffer(rgb_array, (w, h), "RGB")


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef create_radial_gradient_alpha(
        const int w,
        const int h,
        float offset_x              = 0.5,
        float offset_y              = 0.5,
        tuple color_start           = (255, 0, 0, 255),
        tuple color_end             = (0, 0, 0, 0),
        object precomputed_gradient = None,
        float scale_factor          = r_max,
        unsigned short int threads  = 8
):
    """
    Creates a 32-bit radial gradient with transparency.

    This function generates a radial gradient centered on the given surface,
    blending smoothly between `color_start` and `color_end`. It supports optional
    precomputed gradient arrays for performance optimization.

    Uses `pygame.Surface.convert_alpha()` for optimal rendering.

    :param w: 
        int - Width of the surface in pixels.
    
    :param h: 
        int - Height of the surface in pixels.
    
    :param offset_x: 
        float - X-coordinate of the gradient center (relative, default: 0.5).
    
    :param offset_y: 
        float - Y-coordinate of the gradient center (relative, default: 0.5).
    
    :param color_start: 
        tuple - RGBA start color (default: (255, 0, 0, 255)).
    
    :param color_end: 
        tuple - RGBA end color (default: (0, 0, 0, 0)).
    
    :param precomputed_gradient: 
        numpy.ndarray or None - Precomputed gradient array (w, 4) in uint8 format.
    
    :param scale_factor: 
        float - Scaling factor (>0, default: `r_max`).
    
    :param threads: 
        int - Number of concurrent threads for computation (default: 8).
    
    :return: 
        pygame.Surface - Generated radial gradient surface with per-pixel transparency.
    
    :raises ValueError: If `color_start` or `color_end` is not a tuple of four integers.
    :raises ValueError: If any RGBA value in `color_start` or `color_end` is out of range (0-255).
    :raises ValueError: If `scale_factor` is None or <= 0.
    :raises ValueError: If `w` or `h` is <= 1.
    :raises TypeError: If `precomputed_gradient` is not a contiguous array.
    :raises ValueError: If `precomputed_gradient` is not a 2D array, not of type uint8, 
        has length <= 1, or is not in RGBA format.

    
    **Example Usage:**
    ```python
    precomputed_gradient = create_line_gradient_rgba(
        int(math.sqrt(800 ** 2 + 800 ** 2)),
        start_rgba=(255, 0, 0, 255),
        end_rgba=(0, 0, 0, 0)
    )

    surface = create_radial_gradient_alpha(
        800, 800,
        precomputed_gradient=precomputed_gradient
    ).convert_alpha()
    ```
    """

    # Validate color tuples
    if not isinstance(color_start, tuple) or len(color_start) != 4:
        raise ValueError("color_start must be a tuple of four integers (R, G, B, A) in the range 0-255.")
    if not isinstance(color_end, tuple) or len(color_end) != 4:
        raise ValueError("color_end must be a tuple of four integers (R, G, B, A) in the range 0-255.")

    cdef int channel
    # Ensure each color component is within the valid range (0-255)
    for channel in (*color_start, *color_end):
        if not isinstance(channel, int) or not (0 <= channel <= 255):
            raise ValueError("All RGBA values in color_start and color_end must be integers between 0 and 255.")

    # Validate input parameters
    if scale_factor is None or scale_factor <= 0:
        raise ValueError("scale_factor must be greater than 0.")
    if w <= 1:
        raise ValueError("Width (w) must be greater than 1.")
    if h <= 1:
        raise ValueError("Height (h) must be greater than 1.")

    # Validate precomputed gradient if provided
    if precomputed_gradient is not None:
        if not is_type_memoryview(precomputed_gradient):
            if isinstance(precomputed_gradient, numpy.ndarray) and not precomputed_gradient.flags['C_CONTIGUOUS']:
                raise TypeError("precomputed_gradient must be a contiguous array.")
            if precomputed_gradient.ndim != 2:
                raise ValueError(f"precomputed_gradient must be a 2D array. Received: "
                                 f"{(<object> precomputed_gradient).ndim} dimensions.")
            if (<object> precomputed_gradient).dtype != numpy.uint8:
                raise ValueError(f"precomputed_gradient must be of type uint8, got {precomputed_gradient.dtype}.")
            if (<object> precomputed_gradient).shape[0] <= 1:
                raise ValueError(f"precomputed_gradient length must be >1, got {precomputed_gradient.shape[0]}.")
            if (<object> precomputed_gradient).shape[1] != 4:
                raise ValueError(f"precomputed_gradient must be in RGBA format, got {precomputed_gradient.shape[1]}.")

    # Initialize the output image array (RGBA format)
    cdef unsigned char[:, :, ::1] rgb_array = empty((h, w, 4), dtype=uint8)
    cdef float nx, ny
    cdef float w2 = <float>w * <float>scale_factor
    cdef float h2 = <float>h * <float>scale_factor
    cdef float r0 = <float>sqrt(w2 * w2 + h2 * h2)  # Maximum radius for the gradient

    cdef int i, j
    cdef unsigned int x
    cdef float n1 = <float>1.0 / w
    cdef float n2 = <float>1.0 / h
    cdef unsigned short int THREADS = threads

    # Generate the gradient array if not precomputed
    cdef unsigned char[:, ::1] gradient_array = (
        create_line_gradient_rgba(
            <int>sqrt(w ** 2 + h ** 2),
            start_rgba=color_start,
            end_rgba=color_end
        ) if precomputed_gradient is None else precomputed_gradient
    )

    cdef unsigned int l = <object> gradient_array.shape[0] - 1  # Maximum gradient index

    # Parallel processing for performance optimization
    with nogil:
        for j in prange(h, schedule=SCHEDULE, num_threads=THREADS):
            ny = (<float>j * n2) - <float>offset_y
            for i in range(w):
                nx = (<float>i * n1) - <float>offset_x
                x = <int>((<float>sqrt(nx * nx + ny * ny) * r0) * r_max)  # Compute radial distance
                if x > l:
                    # Out of bounds, assign fully transparent black
                    rgb_array[j, i, 0] = 0
                    rgb_array[j, i, 1] = 0
                    rgb_array[j, i, 2] = 0
                    rgb_array[j, i, 3] = 0
                else:
                    # Assign computed gradient color
                    rgb_array[j, i, 0] = <unsigned char>gradient_array[x, 0]
                    rgb_array[j, i, 1] = <unsigned char>gradient_array[x, 1]
                    rgb_array[j, i, 2] = <unsigned char>gradient_array[x, 2]
                    rgb_array[j, i, 3] = <unsigned char>gradient_array[x, 3]

    return frombuffer(rgb_array, (w, h), "RGBA")



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef create_quarter_radial_gradient(
        int width_,
        int height_,
        tuple start_color_           = (255, 0, 0),
        tuple end_color_             = (0, 0, 0),
        object gradient_array_       = None,
        float factor_                = 1.4,
        unsigned short int threads_  = 8
):
    """
    Generates a quarter radial gradient (24-bit opaque).

    This function creates a **radial gradient** by iterating over the **northwest (NW) quarter** 
    of the surface (w/2, h/2) and mirroring the computed pixels to the remaining three 
    quadrants (**northeast (NE), southeast (SE), and southwest (SW)**).  

    The gradient transitions from `start_color_` at the center to `end_color_` at the edges.  

    Parameters:
    ----------
    width_ : int  
        The width of the surface in pixels.  
    height_ : int  
        The height of the surface in pixels.  
    start_color_ : tuple (R, G, B), optional  
        The starting color (center of the gradient). Default is **(255, 0, 0) [Red]**.  
    end_color_ : tuple (R, G, B), optional  
        The ending color (outer edge of the gradient). Default is **(0, 0, 0) [Black]**.  
    gradient_array_ : numpy.ndarray, optional  
        A NumPy array of shape `(width, 3)`, containing precomputed RGB color values (`uint8`).  
        If `None`, the gradient is computed dynamically.  
    factor_ : float, optional  
        Controls the intensity and spread of the gradient. **Must be > 0**. Default is **1.4**.  
    threads_ : int, optional  
        Number of concurrent threads to use for computation. Default is **8**.  

    Returns:
    -------
    pygame.Surface  
        A `pygame.Surface` object containing the radial gradient, **centered at (width/2, height/2)**.  
    """


    assert r_max != 0, "Constant r_max cannot be null"
    if factor_ <=0:
        raise ValueError("Argument amplitude cannot be <= 0.0 default is 1.4")
    assert width_ > 0, "Argument w cannot be <=0"
    assert height_ > 0, "Argument h cannot be <=0"

    cdef:

        float nx, ny
        float w2 = <float>width_ * <float>factor_
        float h2 = <float>height_ * <float>factor_
        float r0 = <float>sqrt(w2 * w2 + h2 * h2)
        unsigned char [:, :, ::1] rgb_array = empty((height_, width_, 3), dtype=uint8)

        int i, j
        unsigned int x
        float n1 = <float>1.0 / width_
        float n2 = <float>1.0 / height_
        unsigned char * g1
        unsigned char * g2
        unsigned char * g3
        unsigned int width_1 = width_ - 1
        unsigned int height_1 = height_ - 1
        unsigned w_i
        unsigned h_j

    cdef unsigned short int THREADS = threads_

    cdef unsigned char [:, ::1] gradient_array__

    if gradient_array_ is None:

        gradient_array_ = create_line_gradient_rgb(
            <int>sqrt(width_ * width_ + (height_ * <float>0.5) * (height_ * <float>0.5)),
            start_rgb=start_color_,
            end_rgb=end_color_
        )


    gradient_array__ = gradient_array_
    cdef unsigned int l = <object> gradient_array__.shape[0] - 1


    with nogil:
        for j in prange(height_ >> 1, schedule=SCHEDULE, num_threads=THREADS):
            ny = (<float> j * n2) - <float> 0.5

            for i in range(width_ >> 1):
                nx = (<float>i * n1) - <float> 0.5

                # position in the gradient
                x = <int>((<float>sqrt(nx * nx + ny * ny) * r0) * r_max)

                g1 = &gradient_array__[x, 0]
                g2 = &gradient_array__[x, 1]
                g3 = &gradient_array__[x, 2]

                w_i = width_1 - i
                h_j = height_1 - j

                if x > l:
                    # NW
                    rgb_array[j, i, 0] = 0
                    rgb_array[j, i, 1] = 0
                    rgb_array[j, i, 2] = 0
                    # NE
                    rgb_array[j, w_i, 0] = 0
                    rgb_array[j, w_i, 1] = 0
                    rgb_array[j, w_i, 2] = 0

                    # SE
                    rgb_array[h_j, w_i, 0] = 0
                    rgb_array[h_j, w_i, 1] = 0
                    rgb_array[h_j, w_i, 2] = 0

                    # SW
                    rgb_array[h_j, i, 0] = 0
                    rgb_array[h_j, i, 1] = 0
                    rgb_array[h_j, i, 2] = 0

                else:
                    # NW
                    rgb_array[j, i, 0] = g1[0]
                    rgb_array[j, i, 1] = g2[0]
                    rgb_array[j, i, 2] = g3[0]
                    # NE
                    rgb_array[j, w_i, 0] = g1[0]
                    rgb_array[j, w_i, 1] = g2[0]
                    rgb_array[j, w_i, 2] = g3[0]
                    # SE
                    rgb_array[h_j, w_i, 0] = g1[0]
                    rgb_array[h_j, w_i, 1] = g2[0]
                    rgb_array[h_j, w_i, 2] = g3[0]
                    # SW
                    rgb_array[h_j, i, 0] = g1[0]
                    rgb_array[h_j, i, 1] = g2[0]
                    rgb_array[h_j, i, 2] = g3[0]

    return frombuffer(rgb_array, (width_, height_), "RGB")




@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef create_quarter_radial_gradient_alpha(
        int width_,
        int height_,
        tuple start_color_            = (255, 0, 0, 255),
        tuple end_color_              = (0, 0, 0, 0),
        object gradient_array_        = None,
        float factor_                 = 1.4,
        unsigned short int threads_   = 8
):
    """
    Generates a quarter radial gradient with alpha transparency (32-bit).

    This function creates a **radial gradient with transparency** by iterating over the 
    **northwest (NW) quarter** of the surface (width/2, height/2) and mirroring the computed 
    pixels to the remaining three quadrants (**northeast (NE), southeast (SE), and southwest (SW)**).  

    The gradient smoothly transitions from `start_color_` at the center to `end_color_` at the edges, 
    incorporating **alpha blending for transparency**.

    Parameters:
    ----------
    width_ : int  
        The width of the surface in pixels.  
    height_ : int  
        The height of the surface in pixels.  
    start_color_ : tuple (R, G, B, A), optional  
        The color at the **center** of the gradient (RGBA format).  
        Default is **(255, 0, 0, 255) [Opaque Red]**.  
    end_color_ : tuple (R, G, B, A), optional  
        The color at the **outer edge** of the gradient (RGBA format).  
        Default is **(0, 0, 0, 0) [Fully Transparent Black]**.  
    gradient_array_ : numpy.ndarray, optional  
        A NumPy array of shape `(width, 4)`, containing precomputed RGBA color values (`uint8`).  
        If `None`, the gradient is computed dynamically.  
    factor_ : float, optional  
        Controls the intensity and spread of the gradient. **Must be > 0**. Default is **1.4**.  
    threads_ : int, optional  
        Number of concurrent threads to use for computation. Default is **8**.  

    Returns:
    -------
    pygame.Surface  
        A `pygame.Surface` object containing the radial gradient, **centered at (width/2, height/2)**, 
        with **alpha transparency for smooth blending**.  
    """


    assert r_max != 0, "Constant r_max cannot be null"
    if factor_ <=0:
        raise ValueError("Argument amplitude cannot be <= 0.0 default is 1.4")
    assert width_ > 0, "Argument w cannot be <=0"
    assert height_ > 0, "Argument h cannot be <=0"

    cdef:

        float nx, ny
        float w2 = <float>width_ * <float>factor_
        float h2 = <float>height_ * <float>factor_
        float r0 = <float>sqrt(w2 * w2 + h2 * h2)
        unsigned char [:, :, ::1] rgb_array = empty((height_, width_, 4), dtype=uint8)

        int i, j
        unsigned int x
        float n1 = <float>1.0 / width_
        float n2 = <float>1.0 / height_
        unsigned char * g1
        unsigned char * g2
        unsigned char * g3
        unsigned char * g4
        unsigned int width_1  = width_ - 1
        unsigned int height_1 = height_ - 1
        unsigned w_i
        unsigned h_j

    cdef unsigned short int THREADS = threads_

    cdef unsigned char [:, ::1] gradient_array__

    if gradient_array_ is None:

        gradient_array_ = create_line_gradient_rgba(
            <int>sqrt(width_ * width_ + (height_ * <float>0.5) * (height_ * <float>0.5)),
            start_rgba=start_color_,
            end_rgba=end_color_
        )

    gradient_array__ = gradient_array_
    cdef unsigned int l = <object> gradient_array__.shape[0] - 1


    with nogil:
        for j in prange(height_ >> 1, schedule=SCHEDULE, num_threads=THREADS):
            ny = (<float> j * n2) - <float> 0.5

            for i in range(width_ >> 1):
                nx = (<float>i * n1) - <float> 0.5

                # position in the gradient
                x = <int>((<float>sqrt(nx * nx + ny * ny) * r0) * r_max)

                g1 = &gradient_array__[x, 0]
                g2 = &gradient_array__[x, 1]
                g3 = &gradient_array__[x, 2]
                g4 = &gradient_array__[x, 3]

                w_i = width_1 - i
                h_j = height_1 - j

                if x > l:
                    # NW
                    rgb_array[j, i, 0] = 0
                    rgb_array[j, i, 1] = 0
                    rgb_array[j, i, 2] = 0
                    rgb_array[j, i, 3] = 0
                    # NE
                    rgb_array[j, w_i, 0] = 0
                    rgb_array[j, w_i, 1] = 0
                    rgb_array[j, w_i, 2] = 0
                    rgb_array[j, w_i, 3] = 0

                    # SE
                    rgb_array[h_j, w_i, 0] = 0
                    rgb_array[h_j, w_i, 1] = 0
                    rgb_array[h_j, w_i, 2] = 0
                    rgb_array[h_j, w_i, 3] = 0

                    # SW
                    rgb_array[h_j, i, 0] = 0
                    rgb_array[h_j, i, 1] = 0
                    rgb_array[h_j, i, 2] = 0
                    rgb_array[h_j, i, 3] = 0

                else:
                    # NW
                    rgb_array[j, i, 0] = g1[0]
                    rgb_array[j, i, 1] = g2[0]
                    rgb_array[j, i, 2] = g3[0]
                    rgb_array[j, i, 3] = g4[0]
                    # NE
                    rgb_array[j, w_i, 0] = g1[0]
                    rgb_array[j, w_i, 1] = g2[0]
                    rgb_array[j, w_i, 2] = g3[0]
                    rgb_array[j, w_i, 3] = g4[0]
                    # SE
                    rgb_array[h_j, w_i, 0] = g1[0]
                    rgb_array[h_j, w_i, 1] = g2[0]
                    rgb_array[h_j, w_i, 2] = g3[0]
                    rgb_array[h_j, w_i, 3] = g4[0]
                    # SW
                    rgb_array[h_j, i, 0] = g1[0]
                    rgb_array[h_j, i, 1] = g2[0]
                    rgb_array[h_j, i, 2] = g3[0]
                    rgb_array[h_j, i, 3] = g4[0]

    return frombuffer(rgb_array, (width_, height_), "RGBA").convert_alpha()


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef float _test_color_dist_hsv(list rgb1, list rgb2):
    """
    TEST ONLY
    """
    cdef hsv hsv_1, hsv_2
    hsv_1 = struct_rgb_to_hsv(<float>rgb1[0]/<float>255.0, <float>rgb1[1]/<float>255.0, <float>rgb1[2]/<float>255.0)
    hsv_2 = struct_rgb_to_hsv(<float>rgb2[0]/<float>255.0, <float>rgb2[1]/<float>255.0, <float>rgb2[2]/<float>255.0)
    return color_dist_hsv(hsv_1, hsv_2)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cdef float color_dist_hsv(const hsv hsv_1, const hsv hsv_2)nogil:

    """
    Computes the squared Euclidean distance between two colors in the HSV color space.

    This function calculates the perceptual difference between two colors 
    represented in the HSV (Hue, Saturation, Value) model using the squared 
    Euclidean distance formula:

        d^2 = (h1 - h2)^2 + (s1 - s2)^2 + (v1 - v2)^2

    Since this function returns the squared distance, it avoids the computational 
    cost of computing the square root. If the actual distance is needed, the 
    square root can be taken externally.

    Note:
        - The hue component (h) is assumed to be on a linear scale (not wrapped around 360°).
        - This function is marked `nogil` for efficient multi-threaded execution.

    :param hsv_1: normalised values; struct hsv (h, s, v) - First HSV color.
    :param hsv_2: normalised values; struct hsv (h, s, v) - Second HSV color.
    
    :return: float - Squared Euclidean distance between the two colors.
    
    """
    return (hsv_1.h - hsv_2.h) ** 2 + (hsv_1.s - hsv_2.s) ** 2 + (hsv_1.v - hsv_2.v) ** 2


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef float _test_color_dist_hsl(list rgb1, list rgb2):
    """
    TEST ONLY
    """
    cdef hsl hsl_1, hsl_2
    hsl_1 = struct_rgb_to_hsl(<float>rgb1[0]/255.0, <float>rgb1[1]/255.0, <float>rgb1[2]/255.0)
    hsl_2 = struct_rgb_to_hsl(<float>rgb2[0]/255.0, <float>rgb2[1]/255.0, <float>rgb2[2]/255.0)
    return color_dist_hsl(hsl_1, hsl_2)


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef float color_dist_hsl(const hsl hsl_1, const hsl hsl_2)nogil:
    """
    Computes the squared Euclidean distance between two colors in the HSL color space.
    
    This function calculates the perceptual difference between two colors 
    represented in the HSL (Hue, Saturation, Luma) model using the squared 
    Euclidean distance formula:
    
       d^2 = (h1 - h2)^2 + (s1 - s2)^2 + (l1 - l2)^2
    
    Since this function returns the squared distance, it avoids the computational 
    cost of computing the square root. If the actual distance is needed, the 
    square root can be taken externally.
    
    Note:
       - The hue component (h) is assumed to be on a linear scale (not wrapped around 360°).
       - This function is marked `nogil` for efficient multi-threaded execution.
    
    :param hsl_1: hsl - normalised values; struct (h, s, l) First HSL color.
    :param hsl_2: hsl - normalised values; struct (h, s, l) Second HSL color.
    
    :return: float - Squared Euclidean distance between the two colors.
    """
    return (hsl_1.h - hsl_2.h) ** 2 + (hsl_1.s - hsl_2.s) ** 2 + (hsl_1.l - hsl_2.l) ** 2



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef (unsigned char, unsigned char, unsigned char) test_color_diff_hsv(
        list rgb__,
        palette_
):
    """
    TEST ONLY
    A helper function for testing purposes that finds the closest color match in HSV space.

    :param rgb__: A list of three integers representing an RGB color in the range [0, 255].
    :param palette_: A color palette, expected to be a list or NumPy array of RGB colors.

    :return: A tuple of three unsigned char values representing the closest matching RGB color.

    :raises ValueError: If `palette_` is None.
    """

    # Define a struct to store the normalized RGB values
    cdef rgb rgb_

    # Normalize RGB values from [0, 255] to [0.0, 1.0]
    rgb_.r = <float> rgb__[ 0 ] / <float> 255.0
    rgb_.g = <float> rgb__[ 1 ] / <float> 255.0
    rgb_.b = <float> rgb__[ 2 ] / <float> 255.0

    # Define a variable to store the number of colors in the palette
    cdef Py_ssize_t l

    if palette_ is not None:
        # Get the length of the palette (number of colors)
        l = len(palette_)
    else:
        # Raise an error if the palette is None
        raise ValueError(f"Argument palette_ cannot be None")

    # Call the function to find the closest color match in HSV space
    return closest_hsv_color(rgb_, palette_, l)



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef (unsigned char, unsigned char, unsigned char) closest_hsv_color(
        const rgb target_color,
        const float [:, ::1] color_palette,
        const Py_ssize_t palette_size):

    """
    Find the closest color match in HSV color space for a given color from a palette.
    
    This function compares a given color (target_color) with a palette of RGB colors, 
    finding the closest match based on the difference in their HSV values. The comparison 
    is done in the HSV color space to achieve a perceptually accurate distance, which is 
    often more aligned with human color perception than direct RGB comparison.

    :param target_color: 
        A struct RGB (float, float, float) containing the target pixel color, with RGB values normalized 
        in the range [0.0, 1.0].
        
    :param color_palette: 
        A numpy.ndarray with shape (w, 3) and dtype float32, representing a palette of 
        colors in RGB, with each color normalized in the range [0.0, 1.0]. The palette 
        should contain `w` colors, where each color is a 3-element array of RGB values.
        
    :param palette_size: 
        An integer representing the number of colors in the palette (`w`).
        
    :return: 
        A tuple containing the RGB values of the closest matching color from the palette, 
        in the range [0, 255].
    
    :raises ValueError:
        - If `target_color` contains values greater than 1.0.
        - If `palette_size` does not match the length of `color_palette`.
    
    :raises TypeError:
        - If `color_palette` is not a contiguous 2D NumPy array.
        - If `color_palette` does not have exactly 3 color components per row.
        - If `color_palette` has zero length.
         
    **Example**:
    ```python
    palette = numpy.asarray(create_line_gradient_rgb(256, (0, 0, 0), (0, 255, 0)), dtype=numpy.float32)
    palette /= 255.0
    print(test_color_diff_hsv([0, 254, 0], palette))
    ```
    This example creates a palette from black to green, and then finds the closest match to 
    the color [0, 254, 0] (a green shade) in the palette using the `closest_hsv_color` function.
    """

    if target_color.r > 1.0 or target_color.g > 1.0 or target_color.b > 1.0:
        raise ValueError(f"Argument target_color struct has at least "
                         f"one value greater than 1.0, values are: "
                         f"'{target_color.r}, {target_color.g}, {target_color.b}'")

    if color_palette is not None:

        if not is_type_memoryview(color_palette) and isinstance(color_palette, np.ndarray):
            if not color_palette.flags['C_CONTIGUOUS']:
                raise TypeError(f"Argument palette_ must be a contiguous 2d array.")

            if color_palette.ndim != 2 or color_palette.shape[1] != 3:
                raise TypeError(
                    f"argument palette_ must have 2 dimensions (w, 3) "
                    f"containing RGB values, got '{color_palette.ndim} "
                    f"dimensions and '{color_palette.shape[1]} color components.")

            if color_palette.shape[0] == 0:
                raise TypeError(f"Argument palette_ length cannot be zero.")


            if palette_size != color_palette.shape[0]:
                raise ValueError(f"Argument palette_size ({palette_size}) does not match the "
                         f"length of the array palette_ ({color_palette.shape[0]}).")

    cdef:
        int i  # Loop index for iterating through the palette
        hsv hsv1, hsv2  # HSV color structs for target_color and each palette color
        float min_value  # Stores the minimum color difference (HSV)
        float current_value  # Holds the current HSV difference for each palette color
        int min_index = 0 # Index of the color with the minimum difference
        const float * p1  # Pointer to the current palette color in the loop

    with nogil:
        # Convert the RGB value of target_color to HSV (target_color is in RGB normalized)
        hsv1 = struct_rgb_to_hsv(target_color.r, target_color.g, target_color.b)

        # Set initial minimum difference to a large value (max possible RGB distance)
        min_value = <float> (255 ** 2 + 255 ** 2 + 255 ** 2)

        # Loop through the palette to find the closest color in HSV space
        for i in range(palette_size):
            p1 = &color_palette[ i, 0 ]  # Pointer to the current palette color
            # Convert the current palette color from RGB to HSV
            hsv2 = struct_rgb_to_hsv(p1[ 0 ], (p1 + 1)[ 0 ], (p1 + 2)[ 0 ])

            # Calculate the HSV distance between target_color and the current palette color
            current_value = <float> color_dist_hsv(hsv1, hsv2)

            # If the current distance is smaller than the previous smallest, update
            if current_value < min_value:
                min_value = current_value  # Update the minimum distance
                min_index = i  # Update the index of the closest color

        # Return the RGB values of the closest color, scaled to [0..255] and rounded
        return <unsigned char> round_c(color_palette[ min_index, 0 ] * <float> 255.0), \
               <unsigned char> round_c(color_palette[ min_index, 1 ] * <float> 255.0), \
               <unsigned char> round_c(color_palette[ min_index, 2 ] * <float> 255.0)



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef (unsigned char, unsigned char, unsigned char) test_color_diff_hsl(
        list rgb__,
        palette_):
    """
    TEST ONLY
    A helper function for testing purposes that finds the closest color match in HSL space.

    :param rgb__: A list of three integers representing an RGB color in the range [0, 255].
    :param palette_: A color palette, expected to be a list or NumPy array of RGB colors.

    :return: A tuple of three unsigned char values representing the closest matching RGB color.

    :raises ValueError: If `palette_` is None.
    """

    # Define a struct to store the normalized RGB values
    cdef rgb rgb_

    # Normalize RGB values from [0, 255] to [0.0, 1.0]
    rgb_.r = <float> rgb__[ 0 ] / <float> 255.0
    rgb_.g = <float> rgb__[ 1 ] / <float> 255.0
    rgb_.b = <float> rgb__[ 2 ] / <float> 255.0

    # Define a variable to store the size of the palette
    cdef Py_ssize_t l

    if palette_ is not None:
        # Get the number of colors in the palette
        l = len(palette_)
    else:
        # Raise an error if the palette is None
        raise ValueError(f"Argument palette_ cannot be None")

    # Call the function to find the closest color match in HSL space
    return closest_hsl_color(rgb_, palette_, l)


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef (unsigned char, unsigned char, unsigned char) closest_hsl_color(
        const rgb target_color,
        const float [:, :] color_palette,
        const Py_ssize_t palette_size):

    """
    Find the closest color match in HSL color space for a given color from a palette.
    
    This function compares a given color (target_color) with a palette of RGB colors, 
    finding the closest match based on the difference in their HSL values. The comparison 
    is done in the HSL color space to achieve a perceptually accurate distance, which is 
    often more aligned with human color perception than direct RGB comparison.

    :param target_color: 
        A struct RGB (float, float, float) containing the target pixel color, with RGB values normalized 
        in the range [0.0, 1.0].
        
    :param color_palette: 
        A numpy.ndarray with shape (w, 3) and dtype float32, representing a palette of 
        colors in RGB, with each color normalized in the range [0.0, 1.0]. The palette 
        should contain `w` colors, where each color is a 3-element array of RGB values.
        
    :param palette_size: 
        An integer representing the number of colors in the palette (`w`).
        
    :return: 
        A tuple containing the RGB values of the closest matching color from the palette, 
        in the range [0, 255].
    
    :raises ValueError:
        - If `target_color` contains values greater than 1.0.
        - If `palette_size` does not match the length of `color_palette`.
    
    :raises TypeError:
        - If `color_palette` is not a contiguous 2D NumPy array.
        - If `color_palette` does not have exactly 3 color components per row.
        - If `color_palette` has zero length.
        
    **Example**:
    ```python
    palette = numpy.asarray(create_line_gradient_rgb(256, (0, 0, 0), (0, 255, 0)), dtype=numpy.float32)
    palette /= 255.0
    print(test_color_diff_hsl([0, 254, 0], palette))
    ```
    This example creates a palette from black to green, and then finds the closest match to 
    the color [0, 254, 0] (a green shade) in the palette using the `closest_hsl_color` function.
    """

    if target_color.r > 1.0 or target_color.g > 1.0 or target_color.b > 1.0:
        raise ValueError(f"Argument target_color struct has at least "
                         f"one value greater than 1.0, values are: "
                         f"'{target_color.r}, {target_color.g}, {target_color.b}'")

    if color_palette is not None:

        if not is_type_memoryview(color_palette) and isinstance(color_palette, np.ndarray):
            if not color_palette.flags['C_CONTIGUOUS']:
                raise TypeError(f"Argument color_palette must be a contiguous 2d array.")

            if color_palette.ndim != 2 or color_palette.shape[1] != 3:
                raise TypeError(
                    f"argument color_palette must have 2 dimensions (w, 3) "
                    f"containing RGB values, got '{color_palette.ndim} "
                    f"dimensions and '{color_palette.shape[1]} color components.")

            if color_palette.shape[0] == 0:
                raise TypeError(f"Argument color_palette length cannot be zero.")


            if palette_size != color_palette.shape[0]:
                raise ValueError(f"Argument palette_size ({palette_size}) does not match the "
                         f"length of the array color_palette ({color_palette.shape[0]}).")


    cdef:
        hsl hsl1, hsl2
        float min_value, current_value = 0
        int min_index = 0, i

    with nogil:

        # THE RGB TO HSL VALUE NEVER CHANGE INSIDE THE LOOP
        hsl1 = struct_rgb_to_hsl(<float>target_color.r, <float>target_color.g, <float>target_color.b)

        # Set initial minimum difference to a large value (max possible RGB distance)
        min_value = <float> (255 ** 2 + 255 ** 2 + 255 ** 2)

        for i in range(palette_size):

            p1 = &color_palette[ i, 0 ]  # Pointer to the current palette color
            # Convert the current palette color from RGB to HSL
            hsl2 = struct_rgb_to_hsl(p1[ 0 ], (p1 + 1)[ 0 ], (p1 + 2)[ 0 ])

            # Calculate the HSL distance between target_color and the current palette color
            current_value = <float> color_dist_hsl(hsl1, hsl2)

            # If the current distance is smaller than the previous smallest, update
            if current_value < min_value:
                min_value = current_value  # Update the minimum distance
                min_index = i  # Update the index of the closest color

        # Return the RGB values of the closest color, scaled to [0..255] and rounded
        return <unsigned char> round_c(color_palette[ min_index, 0 ] * 255.0), \
               <unsigned char> round_c(color_palette[ min_index, 1 ] * 255.0), \
               <unsigned char> round_c(color_palette[ min_index, 2 ] * 255.0)




@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef _test_close_color(list rgb__, palette_):
    """
    TEST ONLY
    """
    cdef rgb rgb_
    rgb_.r = <float>rgb__[0]/<float>255.0
    rgb_.g = <float>rgb__[1]/<float>255.0
    rgb_.b = <float>rgb__[2]/<float>255.0
    cdef Py_ssize_t l = len(palette_)
    return close_color(rgb_, palette_, l)

# todo check
@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef rgb close_color(
        rgb colors,
        float [:, :] palette_,
        Py_ssize_t w) nogil:
    """
    Finds the closest matching color from a given palette.

    This function iterates over a **2D color palette** containing RGB values 
    (shape: `(w, 3)`, type: `float32`). The palette is **not normalized** and 
    contains values in the range **[0, 255]**.  

    During iteration, the given color is compared with each palette entry.  
    The input color (`colors`) is replaced with the **closest matching RGB value** 
    found in the palette.  

    **Note:**  
    - This algorithm assumes the palette contains **unique RGB values**.  
    - If the palette has **duplicate colors**, the function will select the first 
      match based on the **sum of RGB component differences**.  

    Parameters:
    ----------
    colors : struct RGB  
        A struct representing the RGB color to be matched.  
    palette_ : numpy.ndarray (float32)  
        A 2D array of shape `(w, 3)`, containing RGB color values in the range **[0, 255]**.  
    w : int  
        The number of colors in the palette.  

    Returns:
    -------
    rgb  
        A struct containing the **closest matching RGB values** from the palette.  
    """


    cdef:
        int i
        float * tmp_v = <float *> malloc(w * sizeof(float))
        rgb rgb_
        float v_min,
        unsigned int s1 = 0
        unsigned int s2 = 0


    # Iterate over the palette colors an calculate the
    # distance between both colors (palette and current_ pixels RGB colors),
    # Place the difference into an 1d buffer.
    for i in prange(w, schedule=SCHEDULE, num_threads=THREADS):
        tmp_v[ i ] =\
           (colors.r - <float>palette_[ i, 0 ]) ** 2 + \
           (colors.g - <float>palette_[ i, 1 ]) ** 2 + \
           (colors.b - <float>palette_[ i, 2 ]) ** 2

    # Find the min value from 1D buffer tmp_v,
    # The min value is the closest color to the pixel value
    # !!!! The array is run through from the right to the left !!!!
    # This method does not check for multiple solutions as two values
    # in the array can be identical. The value returned will always be the first
    # minimal value encounter within the array.
    # All values from the array are by default rounded (no significant
    # decimal point values).
    v_min = <float> min_f(tmp_v, w)

    # Run through the 1D buffer to find the color and
    # the index value
    for i in range(w):
        if v_min == tmp_v[i]:
            if s1 == 0:
                s1 = i
            else:
                s2 = i
                break

    cdef:
        hsv color0_hsv
        hsv hsv1
        hsv hsv2

    # at lease two solution
    if s1!= 0 and s2!= 0:

        color0_hsv = struct_rgb_to_hsv(
            colors.r/<float>255.0,
            colors.g/<float>255.0,
            colors.b/<float>255.0)

        hsv1 = struct_rgb_to_hsv(
            palette_[ s1, 0 ]/<float>255.0,
            palette_[ s1, 1 ]/<float>255.0,
            palette_[ s1, 2 ]/<float>255.0)

        hsv2 = struct_rgb_to_hsv(
            palette_[ s2, 0 ]/<float>255.0,
            palette_[ s2, 1 ]/<float>255.0,
            palette_[ s2, 2 ]/<float>255.0)


        if (color0_hsv.h - hsv1.h) ** 2 <  (color0_hsv.h - hsv2.h) ** 2:
            i = s1
        else:
            i = s2
    else:
        i = s1

    rgb_.r = palette_[ i, 0 ]
    rgb_.g = palette_[ i, 1 ]
    rgb_.b = palette_[ i, 2 ]

    free(tmp_v)

    return rgb_




# todo check
@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef rgb use_palette(
        rgb colors,
        float [:, :] palette_,
        Py_ssize_t w) nogil:
    """
    Selects RGB values from a given palette to approximate a target color.
    
    **PAINTING MODE**
    
    Instead of selecting the closest color directly from the palette, this function 
    independently picks the closest **red, green, and blue** values from the palette.  
    The resulting color **may not exist in the palette** but will be composed of 
    individual RGB components that best match the input color.
    
    Parameters:
    ----------
    colors : struct  
        A struct containing the RGB values of the target color, with integer values 
        in the range **[0, 255]**.  
    palette_ : numpy.ndarray (float32)  
        A **2D array** of shape `(w, 3)`, containing RGB color values in the range **[0, 255]**.  
    w : int  
        The number of colors available in the palette.  
    
    Returns:
    -------
    rgb  
        A struct containing the **approximated RGB color**, where each channel is 
        selected independently from the closest matching palette values.
    """

    cdef:
        int i, j, k  # Loop variables
        float * v_red   = <float *> malloc(w * sizeof(float))   # Array to store red differences
        float * v_green = <float *> malloc(w * sizeof(float))   # Array to store green differences
        float * v_blue  = <float *> malloc(w * sizeof(float))   # Array to store blue differences
        float r_min, g_min, b_min  # Minimum difference for each channel
        rgb rgb_  # Struct to store the final RGB values

    # Compute the absolute difference between the input color and each color in the palette
    # for the R, G, and B channels separately.
    for i in prange(w):
        v_red[i]   = abs_c(colors.r - palette_[i, 0])
        v_green[i] = abs_c(colors.g - palette_[i, 1])
        v_blue[i]  = abs_c(colors.b - palette_[i, 2])

    # Find the minimum difference for each channel.
    r_min = <float> min_f(v_red, w)
    g_min = <float> min_f(v_green, w)
    b_min = <float> min_f(v_blue, w)

    # Identify the closest matching red value in the palette.
    for i in range(w):
        if v_red[i] == r_min:
            break  # Stop at the first occurrence

    # Identify the closest matching green value in the palette.
    for j in range(w):
        if v_green[j] == g_min:
            break  # Stop at the first occurrence

    # Identify the closest matching blue value in the palette.
    for k in range(w):
        if v_blue[k] == b_min:
            break  # Stop at the first occurrence

    # Assign the closest R, G, and B values from the palette independently.
    rgb_.r = palette_[i, 0]
    rgb_.g = palette_[j, 1]
    rgb_.b = palette_[k, 2]

    # Free dynamically allocated memory.
    free(v_red)
    free(v_green)
    free(v_blue)

    return rgb_



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef object scroll_surface_24bit(surface, const short int dx, const short int dy):
    """
    Scroll a 24-bit or 32-bit Pygame surface horizontally and/or vertically.

    This function creates and returns a new surface 24-bit 
    with the pixel data shifted according to the specified `dx` and `dy` values. 
    Pixels that are shifted out of bounds will wrap around to the opposite side.

    :param surface: 
        A Pygame `Surface` object in 24-bit or 32-bit format. 

    :param dx: 
        The number of pixels to scroll horizontally. 
        - A positive `dx` shifts the surface to the right.
        - A negative `dx` shifts the surface to the left.

    :param dy: 
        The number of pixels to scroll vertically.
        - A positive `dy` shifts the surface downward.
        - A negative `dy` shifts the surface upward.

    :return: 
        A new 24-bit Pygame `Surface` object with the applied scroll effect.
        
    :raises TypeError: 
        If `surface` is not a pygame.Surface object.
        
    **Example Usage**:
    ```python
    new_surface = scroll_surface_24bit(original_surface, dx=10, dy=-5)
    ```
    This shifts the surface 10 pixels to the right and 5 pixels upward.
    
    """

    # Validate that the input is a pygame Surface
    if not isinstance(surface, pygame.Surface):
        raise TypeError(f"surface must be a pygame.Surface (got {type(surface)})")

    return scroll_surface_24bit_c(surface, dx, dy)


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef scroll_surface_24bit_c(surface, const short int dx, const short int dy):

    """
    Scroll a 24-bit pygame surface horizontally and/or vertically.

    This function creates and returns a new surface by shifting the pixel data of 
    the input surface (`surface`) based on the specified horizontal (`dx`) and 
    vertical (`dy`) displacement. The scrolling is done in a toroidal (wrap-around) 
    manner, meaning pixels that move off one edge reappear on the opposite edge.

    :param surface: 
        A pygame.Surface object in 24-bit or 32-bit format (compatible with pixel access).
        
    :param dx: 
        Horizontal scroll amount. Positive values move the surface to the right, 
        and negative values move it to the left.
        
    :param dy: 
        Vertical scroll amount. Positive values move the surface downward, 
        and negative values move it upward.
        
    :return: 
        A new pygame Surface with the pixels shifted accordingly.
        
    :raises ValueError: 
        If `surface` has an incompatible pixel format.
        
    ** example ** 
    ```python
    
    image_surface = scroll_surface_24bit(image_surface, 0, 1)
    ```
    
    """

    cdef:
        Py_ssize_t w, h, dim  # Surface dimensions
        unsigned char [:, :, :] rgb_array  # Memoryview for pixel access

    try:
        # Get a pixel memory view in 24-bit format ('3' means RGB with 3 bytes per pixel)
        rgb_array = surface.get_view('3')
    except (ValueError, pygame.error):
        raise ValueError("Incompatible pixel format. Expected a 24-bit or 32-bit format.")

    try:
        # Extract width, height, and color depth from the array shape
        w, h, dim = (<object> rgb_array).shape[:3]
    except (ValueError, pygame.error):
        raise ValueError("Array shape not compatible with expected surface dimensions.")

    cdef:
        int i=0, j=0, ii=0, jj=0  # Loop variables
        unsigned char [:, :, ::1] new_array = numpy.empty((h, w, 3), dtype=numpy.uint8)

    # If no scrolling is required, return the original surface
    if dx == 0 and dy == 0:
        return surface

    # Use OpenMP for parallelized pixel shifting
    with nogil:

        # Case 1: Scrolling in both horizontal (dx) and vertical (dy) directions
        if dx != 0 and dy != 0:
            for j in prange(h, schedule=SCHEDULE, num_threads=THREADS):
                for i in range(w):
                    ii = i + dx
                    jj = j + dy

                    # Wrap around horizontally
                    if ii >= w:
                        ii = ii - w
                    elif ii < 0:
                        ii = ii + w

                    # Wrap around vertically
                    if jj >= h:
                        jj = jj - h
                    elif jj < 0:
                        jj = jj + h

                    # Copy pixel data from the old position to the new position
                    new_array[jj, ii, 0] = rgb_array[i, j, 0]
                    new_array[jj, ii, 1] = rgb_array[i, j, 1]
                    new_array[jj, ii, 2] = rgb_array[i, j, 2]

        # Case 2: Scrolling only horizontally
        elif dx != 0:
            for j in prange(h, schedule=SCHEDULE, num_threads=THREADS):
                for i in range(w):
                    ii = i + dx

                    # Wrap around horizontally
                    if ii >= w:
                        ii = ii - w
                    elif ii < 0:
                        ii = ii + w

                    # Copy pixel data for horizontal shift
                    new_array[j, ii, 0] = rgb_array[i, j, 0]
                    new_array[j, ii, 1] = rgb_array[i, j, 1]
                    new_array[j, ii, 2] = rgb_array[i, j, 2]

        # Case 3: Scrolling only vertically
        elif dy != 0:
            for j in prange(h, schedule=SCHEDULE, num_threads=THREADS):
                for i in range(w):
                    jj = j + dy

                    # Wrap around vertically
                    if jj >= h:
                        jj = jj - h
                    elif jj < 0:
                        jj = jj + h

                    # Copy pixel data for vertical shift
                    new_array[jj, i, 0] = rgb_array[i, j, 0]
                    new_array[jj, i, 1] = rgb_array[i, j, 1]
                    new_array[jj, i, 2] = rgb_array[i, j, 2]

    # Return a new pygame Surface with the updated pixel data
    return frombuffer(new_array, (w, h), 'RGB')




@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef void scroll24_inplace(surface, const short int dx, const short int dy):

    """
    Scroll a 24-bit pygame surface horizontally and/or vertically in place. **in-place**.

    This function modifies the given surface directly by shifting its pixel data 
    based on the specified horizontal (`dx`) and vertical (`dy`) displacement. 
    The scrolling wraps around, meaning pixels that move off one edge reappear 
    on the opposite edge.

    :param surface: 
        A pygame.Surface object in 24-bit or 32-bit format (compatible with pixel access).
        
    :param dx: 
        Horizontal scroll amount. Positive values move pixels to the right, 
        and negative values move them to the left.
        
    :param dy: 
        Vertical scroll amount. Positive values move pixels downward, 
        and negative values move them upward.
        
    :return: 
        None (modifies the surface in place).
        
    :raises TypeError: 
        If `surface` is not a pygame.Surface object.
    
    **Example**
    
    ```python
    
    scroll24_inplace(image_surface, 1, 0)
    ```    
    """
    if not isinstance(surface, pygame.Surface):
        raise TypeError(f"Argument surface must be a pygame surface type got '{type(surface)}'")

    scroll24_inplace_c(surface, dx, dy)


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef void scroll24_inplace_c(surface, const short int dx, const short int dy):
    """
    Scroll a 24-bit surface horizontally and/or vertically **in-place**.

    This function modifies the given pygame surface directly, shifting pixel values 
    according to the provided horizontal (`dx`) and vertical (`dy`) offsets. 

    The scrolling wraps around, meaning pixels shifted beyond the boundary of the 
    surface will reappear on the opposite side.

    **Parameters:**
    :param surface: pygame.Surface
        - A pygame Surface object in **24-bit** or **32-bit** format.
        
    :param dx: short int
        - Horizontal shift amount. 
        - Negative `dx` moves the image **left**, positive `dx` moves it **right**.
        
    :param dy: short int
        - Vertical shift amount.
        - Negative `dy` moves the image **up**, positive `dy` moves it **down**.

    **Returns:**
    :return: void (modifies the surface in-place)

    **Notes:**
    - If both `dx` and `dy` are zero, no changes are applied.
    - The function operates without the GIL (`nogil`) to improve performance.
    - Uses a **manual wrap-around technique** instead of modulo (`%`) for efficiency.
    
    **Example**
    ```python
    
    scroll24_inplace_c(image_surface, 1, 1)
    
    ```
    
    """

    cdef int w, h, dim

    cdef:
        unsigned char [:, :, :] rgb_array

    try:
        rgb_array = surface.get_view('3')

    except (ValueError, pygame.error) as e:
            raise ValueError('\nIncompatible pixel format.')

    try:
        w, h, dim = (<object> rgb_array).shape[:3]
    except (ValueError, pygame.error) as e:
        raise ValueError('\nArray shape not compatible.')

    cdef:
        int i=0, j=0, ii=0, jj=0

    with nogil:

        if dx==0 and dy==0:
            return

        if dx > 0:
            for j in prange(h, schedule=SCHEDULE, num_threads=THREADS):
                for i in range(w):

                    ii = i + dx
                    if ii >= w:
                        ii = ii - w
                    elif ii < 0:
                        ii = ii + w

                    rgb_array[i, j, 0] = rgb_array[ii, j, 0]
                    rgb_array[i, j, 1] = rgb_array[ii, j, 1]
                    rgb_array[i, j, 2] = rgb_array[ii, j, 2]

        else:
            for j in prange(h, schedule=SCHEDULE, num_threads=THREADS):
                for i in range(w-1, -1, -1):

                    ii = i + dx
                    if ii >= w:
                        ii = ii - w
                    elif ii < 0:
                        ii = ii + w

                    rgb_array[i, j, 0] = rgb_array[ii, j, 0]
                    rgb_array[i, j, 1] = rgb_array[ii, j, 1]
                    rgb_array[i, j, 2] = rgb_array[ii, j, 2]
        if dy > 0:
            for i in prange(w, schedule=SCHEDULE, num_threads=THREADS):
                for j in range(h):

                    jj = j + dy
                    if jj >= h:
                        jj = jj - h
                    elif jj < 0:
                        jj = jj + h

                    rgb_array[i, j, 0] = rgb_array[i, jj, 0]
                    rgb_array[i, j, 1] = rgb_array[i, jj, 1]
                    rgb_array[i, j, 2] = rgb_array[i, jj, 2]

        else:
            for i in prange(w, schedule=SCHEDULE, num_threads=THREADS):
                for j in range(h-1, -1, -1):

                    jj = j + dy
                    if jj >= h:
                        jj = jj - h
                    elif jj < 0:
                        jj = jj + h

                    rgb_array[i, j, 0] = rgb_array[i, jj, 0]
                    rgb_array[i, j, 1] = rgb_array[i, jj, 1]
                    rgb_array[i, j, 2] = rgb_array[i, jj, 2]


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef void scroll_rgb_array_inplace(
    unsigned char [:, :, :] rgb_array,
    const short int dx,
    const short int dy
    ):
    """
    Scroll a 3D RGB pixel array horizontally and/or vertically **in-place**.

    This function modifies the given `rgb_array` directly by shifting pixel values 
    according to the specified horizontal (`dx`) and vertical (`dy`) offsets. 
    
    The scrolling **wraps around**, meaning pixels shifted beyond the array's boundaries 
    reappear on the opposite side.

    **Parameters:**
    :param rgb_array: numpy.ndarray
        - A **3D NumPy array** containing **RGB pixel values**.
        - The array must reference **all pixels** to ensure modification in-place.
        
    :param dx: short int, optional (default=0)
        - Horizontal shift amount.
        - Negative `dx` moves pixels **left**, positive `dx` moves pixels **right**.
        
    :param dy: short int, optional (default=0)
        - Vertical shift amount.
        - Negative `dy` moves pixels **up**, positive `dy` moves pixels **down**.

    **Returns:**
    :return: void (modifies `rgb_array` in-place)
    
    **Raises:**
    :raises TypeError: If `rgb_array` is not a `numpy.ndarray` or `memoryviewslice`.
    :raises ValueError: If `rgb_array` is not a 3D array with shape `(height, width, 3)`.
    :raises ValueError: If `rgb_array.dtype` is not `numpy.uint8` (expected 8-bit RGB format).
    :raises ValueError: If `rgb_array` is empty (`shape[0] == 0`).

    **Notes:**
    - If both `dx` and `dy` are zero, the function performs **no operation**.
    - This function is optimized for **performance** and operates **without copying data**.
    - The array must be contiguous and properly referenced to avoid unexpected behavior.
    
    """


    # Check that the input array is a valid type
    if not (isinstance(rgb_array, numpy.ndarray) or rgb_array.__class__.__name__ == '_memoryviewslice'):
        raise TypeError("Input rgb_array must be a numpy.ndarray or memoryviewslice.")

    if not is_type_memoryview(rgb_array):
        if <object>rgb_array.dtype != numpy.uint8:
            raise ValueError()

        # Validate that `rgb_array` is a 3D array with the last dimension being size 3 (RGB).
        if <object> rgb_array.ndim != 3 or <object> rgb_array.shape[ 2 ] != 3:
            raise ValueError("Input rgb_array must be a 3D array with shape (w, h, 3).")

        # Ensure the array is not empty
        if <object>rgb_array.shape[ 0 ] == 0:
            raise ValueError("Input `rgb_array` cannot be empty.")


    scroll_rgb_array_inplace_c(rgb_array, dx, dy)


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef void scroll_rgb_array_inplace_c(
    unsigned char [:, :, :] rgb_array,
    const short int dx,
    const short int dy
    ):
    """
    Scroll a 3D RGB pixel array horizontally and/or vertically **in-place**.

    This function modifies the given `rgb_array` directly by shifting pixel values 
    according to the specified horizontal (`dx`) and vertical (`dy`) offsets. 

    The scrolling **wraps around**, meaning pixels shifted beyond the array's boundaries 
    reappear on the opposite side.

    **Parameters:**
    :param rgb_array: numpy.ndarray
        - A **3D NumPy array** containing **RGB pixel values**.
        - The array must reference **all pixels** to ensure modification in-place.

    :param dx: short int, optional (default=0)
        - Horizontal shift amount.
        - Negative `dx` moves pixels **left**, positive `dx` moves pixels **right**.

    :param dy: short int, optional (default=0)
        - Vertical shift amount.
        - Negative `dy` moves pixels **up**, positive `dy` moves pixels **down**.

    **Returns:**
    :return: void (modifies `rgb_array` in-place)
    """


    cdef int w, h, dim

    try:
        w, h, dim = (<object> rgb_array).shape[:3]
    except (ValueError, pygame.error) as e:
        raise ValueError('\nrgb_array shape not compatible.')


    cdef:
        int i=0, j=0, ii=0, jj=0

    with nogil:

        if dx == 0 and dy == 0:
            return

        if dx > 0:
            for j in prange(h, schedule=SCHEDULE, num_threads=THREADS):
                for i in range(w):

                    ii = i + dx

                    if ii >= w:
                        ii = ii - w
                    elif ii < 0:
                        ii = ii + w

                    rgb_array[i, j, 0] = rgb_array[ii, j, 0]
                    rgb_array[i, j, 1] = rgb_array[ii, j, 1]
                    rgb_array[i, j, 2] = rgb_array[ii, j, 2]

        else:
            for j in prange(h, schedule=SCHEDULE, num_threads=THREADS):
                for i in range(w-1, -1, -1):

                    ii = i + dx

                    if ii >= w:
                        ii = ii - w
                    elif ii < 0:
                        ii = ii + w

                    rgb_array[ i, j, 0 ] = rgb_array[ ii, j, 0 ]
                    rgb_array[ i, j, 1 ] = rgb_array[ ii, j, 1 ]
                    rgb_array[ i, j, 2 ] = rgb_array[ ii, j, 2 ]

        if dy > 0:
            for i in prange(w, schedule=SCHEDULE, num_threads=THREADS):
                for j in range(h):

                    jj = j + dy
                    if jj >= h:
                        jj = jj - h
                    elif jj < 0:
                        jj = jj + h

                    rgb_array[i, j, 0] = rgb_array[i, jj, 0]
                    rgb_array[i, j, 1] = rgb_array[i, jj, 1]
                    rgb_array[i, j, 2] = rgb_array[i, jj, 2]

        else:
            for i in prange(w, schedule=SCHEDULE, num_threads=THREADS):
                for j in range(h-1, -1, -1):

                    jj = j + dy
                    if jj >= h:
                        jj = jj - h
                    elif jj < 0:
                        jj = jj + h

                    rgb_array[ i, j, 0 ] = rgb_array[i, jj, 0]
                    rgb_array[ i, j, 1 ] = rgb_array[i, jj, 1]
                    rgb_array[ i, j, 2 ] = rgb_array[i, jj, 2]



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef void scroll_alpha_inplace(
    unsigned char [:, :] alpha_array,
    const short int dx,
    const short int dy
    ):
    """
    Scroll an alpha channel array (2D) horizontally and/or vertically **in-place**.

    This function modifies the provided `alpha_array` directly by shifting the alpha 
    values based on the given horizontal (`dx`) and vertical (`dy`) offsets. The scrolling 
    operation **wraps around**, meaning any pixels shifted out of the array boundaries will 
    reappear on the opposite side.

    **Parameters:**
    :param alpha_array: numpy.ndarray
        - A **2D NumPy array** containing **alpha channel pixel values** 
        - The array must reference **all alpha pixel values** for the in-place transformation to work.
        
    :param dx: short int
        - The horizontal offset to shift the pixels.
        - A negative `dx` moves the pixels **left**, a positive `dx` moves the pixels **right**.
        
    :param dy: short int
        - The vertical offset to shift the pixels.
        - A negative `dy` moves the pixels **up**, a positive `dy` moves the pixels **down**.

    **Returns:**
    :return: void (modifies `alpha_array` in-place)

    **Notes:**
    - If both `dx` and `dy` are zero, no operation will be performed.
    - This function is **optimized for performance** and does not involve copying the data.
    - The array must be **contiguous and properly referenced** to avoid unexpected behavior.
    - Alpha channel values typically represent pixel transparency, where 0 is fully 
        transparent and 255 is fully opaque.
        
    """

    # Check that the input alpha_array is a valid type
    if not (isinstance(alpha_array, numpy.ndarray) or alpha_array.__class__.__name__ == '_memoryviewslice'):
        raise TypeError("Input alpha_array must be a numpy.ndarray or memoryviewslice.")

    if not is_type_memoryview(alpha_array):
        if <object> alpha_array.dtype != numpy.uint8:
            raise ValueError()

        # Validate that `alpha_array` is a 2D array with the last dimension being size 1 (Alpha).
        if <object> alpha_array.ndim != 2:
            raise ValueError("Input alpha_array must be a 2D array with shape (w, h).")

        # Ensure the alpha_array is not empty
        if <object> alpha_array.shape[ 0 ] == 0:
            raise ValueError("Input `alpha_array` cannot be empty.")

    scroll_alpha_inplace_c(alpha_array, dx, dy)


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef void scroll_alpha_inplace_c(
    unsigned char [:, :] alpha_array,
    const short int dx,
    const short int dy
    ):
    """
    Scroll an alpha channel array horizontally and/or vertically **in-place**.

    This function shifts the pixel values of the given `alpha_array` according to the 
    specified horizontal (`dx`) and vertical (`dy`) offsets. The scrolling is performed
    in-place, meaning that the original array is modified directly, and the values are 
    wrapped around when they exceed the array's boundaries.

    **Parameters:**
    :param alpha_array: numpy.ndarray
        - A 2D NumPy array containing **alpha channel values** (i.e., transparency values).
        - The array must reference all the pixels, and the size of the array must be compatible
          for modification in-place.

    :param dx: short int
        - The horizontal offset by which to scroll the array.
        - A negative value shifts pixels **left**, while a positive value shifts them **right**.

    :param dy: short int
        - The vertical offset by which to scroll the array.
        - A negative value shifts pixels **up**, while a positive value shifts them **down**.

    **Returns:**
    :return: void
        - This function does not return any value. It modifies the `alpha_array` in-place.

    **Notes:**
    - If both `dx` and `dy` are zero, the function does nothing (no scroll).
    - The scrolling **wraps around** the array. When a pixel is shifted beyond 
        the array's boundary, it reappears on the opposite side.
    - This function operates **in-place** and does not create a copy of the array, 
        making it efficient for large data.
    - Ensure that the input array (`alpha_array`) is contiguous and properly referenced
        to avoid unexpected behavior.
    - The array is expected to be a 2D array with shape `(width, height)`,
        where the pixel values represent alpha (transparency).

    **Example:**
    >>> alpha_array = np.zeros((10, 10), dtype=np.uint8)
    >>> scroll_alpha_inplace_c(alpha_array, dx=1, dy=0)  # Scroll the alpha array 1 pixel to the right.
    """


    cdef Py_ssize_t w, h, dim

    try:
        w, h = (<object> alpha_array).shape[:2]
    except (ValueError, pygame.error) as e:
        raise ValueError('\rgb_array shape not compatible.')

    cdef:
        int i=0, j=0, ii=0, jj=0

    with nogil:

        if dx==0 and dy==0:
            return

        if dx > 0:
            for j in prange(h, schedule=SCHEDULE, num_threads=THREADS):
                for i in range(w):

                    ii = i + dx
                    if ii >= w:
                        ii = ii - w  # Wrap around when exceeding the width
                    elif ii < 0:
                        ii = ii + w  # Wrap around when going negative

                    alpha_array[i, j] = alpha_array[ii, j]

        else:
            for j in prange(h, schedule=SCHEDULE, num_threads=THREADS):
                for i in range(w-1, -1, -1):

                    ii = i + dx
                    if ii >= w:
                        ii = ii - w  # Wrap around when exceeding the width
                    elif ii < 0:
                        ii = ii + w  # Wrap around when going negative

                    alpha_array[i, j] = alpha_array[ii, j]

        if dy > 0:
            for i in prange(w, schedule=SCHEDULE, num_threads=THREADS):
                for j in range(h):

                    jj = j + dy
                    if jj >= h:
                        jj = jj - h  # Wrap around when exceeding the height
                    elif jj < 0:
                        jj = jj + h  # Wrap around when going negative

                    alpha_array[i, j] = alpha_array[i, jj]

        else:
            for i in prange(w, schedule=SCHEDULE, num_threads=THREADS):
                for j in range(h-1, -1, -1):

                    jj = j + dy
                    if jj >= h:
                        jj = jj - h  # Wrap around when exceeding the height
                    elif jj < 0:
                        jj = jj + h  # Wrap around when going negative

                    alpha_array[i, j] = alpha_array[i, jj]



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef void scroll32_inplace(surface, const short int dx, const short int dy):
    """
    Scroll a 24-bit or 32-bit pygame.Surface in-place.

    This function shifts the pixel data of the given `surface` horizontally (`dx`) 
    and/or vertically (`dy`) **without creating a new surface**. The scrolling **wraps around**, 
    meaning pixels that move beyond the surface boundaries reappear on the opposite side.

    **Parameters:**
    :param surface: pygame.Surface
        - A `pygame.Surface` object with a **32-bit pixel format**.
        - The surface's pixel data will be modified directly.
        
    :param dx: short int, optional 
        - Number of pixels to scroll **horizontally**.
        - Negative values shift **left**, positive values shift **right**.
        
    :param dy: short int, optional 
        - Number of pixels to scroll **vertically**.
        - Negative values shift **up**, positive values shift **down**.

    **Returns:**
    :return: void
        - The function modifies the `surface` **in-place** and does not return anything.

    **Exceptions:**
    - Raises a `TypeError` if `surface` is not a valid `pygame.Surface`.

    **Notes:**
    - If both `dx` and `dy` are `0`, the function **does nothing**.
    - The function ensures **efficient, in-place modification** without copying data.
    - Scrolling is **cyclic (toroidal wrap-around)**, meaning pixels that exit one edge appear on the opposite edge.

    **Example Usage:**
    ```python
    surface = pygame.Surface((200, 200))
    scroll32_inplace(surface, dx=15, dy=-10)  # Moves 15 pixels right and 10 pixels up.
    ```
    """

    if not isinstance(surface, pygame.Surface):
        raise TypeError('surface, a pygame.Surface is required (got type %s)' % type(surface))

    scroll32_inplace_c(surface, dx, dy)


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef void scroll32_inplace_c(surface, const short int dx, const short int dy):
    """
    Scrolls a 32-bit pygame.Surface horizontally and/or vertically in-place.

    This function modifies the pixel data of the given surface by shifting its contents 
    according to `dx` and `dy`, with a **wrap-around effect** to preserve all pixels.

    **Parameters:**
    - **surface** (*pygame.Surface*): 
        A `pygame.Surface` object with a **32-bit pixel format** (RGBA or RGBX).
        
    - **dx** (*short int, optional*): 
        Number of pixels to scroll **horizontally**.
        - Negative values shift **left**.
        - Positive values shift **right**.
        
    - **dy** (*short int, optional*): 
        Number of pixels to scroll **vertically**.
        - Negative values shift **up**.
        - Positive values shift **down**.

    **Behavior:**
    - The scrolling transformation is applied **in-place**, meaning the original surface is modified.
    - If `dx` and `dy` are both `0`, the function **returns immediately** without modifying the surface.
    - The function ensures a **wrap-around effect**, meaning pixels that move out on one side 
      reappear on the opposite side.

    **Raises:**
    - `ValueError`: If the surface format is **not compatible** (i.e., not 32-bit).
    - `TypeError`: If the provided surface is **not a pygame.Surface**.

    **Example Usage:**
    ```python
    import pygame

    # Scroll the surface 10 pixels to the right and 5 pixels down
    scroll32_inplace_c(surface, dx=10, dy=5)

    ```

    **Performance Considerations:**
    - Uses **parallel processing** with `prange` (from OpenMP) for **faster execution**.
    - Optimized to avoid modulo (`%`) operations, improving performance on large surfaces.
    
    """

    # Define variables for width, height, and color depth (channels)
    cdef Py_ssize_t w, h, dim

    # Declare a memory view for the RGB(A) pixel data of the surface
    cdef:
        unsigned char [:, :, :] rgb_array

    # Attempt to obtain a memory view of the surface's pixel data in 3D (RGB or RGBA format)
    try:
        rgb_array = surface.get_view('3')  # '3' ensures a 3D view for color images

    except (ValueError, pygame.error) as e:
        raise ValueError('\nIncompatible pixel format.')  # Raise an error if the format is invalid

    # Get the dimensions of the image (width, height, and color channels)
    try:
        w, h, dim = (<object> rgb_array).shape[:3]
    except (ValueError, pygame.error) as e:
        raise ValueError('\nArray shape not compatible.')  # Raise an error if the shape is invalid

    # Declare loop variables
    cdef:
        int i=0, j=0, ii=0, jj=0

    # If no scrolling is needed, exit early
    if dx == 0 and dy == 0:
        return

    # Begin a "no global interpreter lock" (nogil) block for parallel execution
    with nogil:

        # Handle horizontal scrolling
        if dx > 0:
            # Parallel processing over height for better performance
            for j in prange(h, schedule=SCHEDULE, num_threads=THREADS):
                for i in range(w):
                    # Compute new column index with wrap-around
                    ii = i + dx
                    if ii >= w:
                        ii = ii - w  # Wrap around to the left if it exceeds width
                    elif ii < 0:
                        ii = ii + w  # Wrap around to the right if negative

                    # Assign new pixel values for RGB(A) channels
                    rgb_array[i, j, 0] = rgb_array[ii, j, 0]  # Red
                    rgb_array[i, j, 1] = rgb_array[ii, j, 1]  # Green
                    rgb_array[i, j, 2] = rgb_array[ii, j, 2]  # Blue
                    rgb_array[i, j, 3] = rgb_array[ii, j, 3]  # Alpha (if present)

        else:
            # Scroll left (looping in reverse for correct overwriting)
            for j in prange(h, schedule=SCHEDULE, num_threads=THREADS):
                for i in range(w-1, -1, -1):
                    # Compute new column index with wrap-around
                    ii = i + dx
                    if ii >= w:
                        ii = ii - w  # Wrap around to the left
                    elif ii < 0:
                        ii = ii + w  # Wrap around to the right

                    # Assign new pixel values for RGB(A) channels
                    rgb_array[i, j, 0] = rgb_array[ii, j, 0]
                    rgb_array[i, j, 1] = rgb_array[ii, j, 1]
                    rgb_array[i, j, 2] = rgb_array[ii, j, 2]
                    rgb_array[i, j, 3] = rgb_array[ii, j, 3]

        # Handle vertical scrolling
        if dy > 0:
            # Parallel processing over width for efficiency
            for i in prange(w, schedule=SCHEDULE, num_threads=THREADS):
                for j in range(h):
                    # Compute new row index with wrap-around
                    jj = j + dy
                    if jj >= h:
                        jj = jj - h  # Wrap around to the top if it exceeds height
                    elif jj < 0:
                        jj = jj + h  # Wrap around to the bottom if negative

                    # Assign new pixel values for RGB(A) channels
                    rgb_array[i, j, 0] = rgb_array[i, jj, 0]
                    rgb_array[i, j, 1] = rgb_array[i, jj, 1]
                    rgb_array[i, j, 2] = rgb_array[i, jj, 2]
                    rgb_array[i, j, 3] = rgb_array[i, jj, 3]

        else:
            # Scroll up (looping in reverse for correct overwriting)
            for i in prange(w, schedule=SCHEDULE, num_threads=THREADS):
                for j in range(h-1, -1, -1):
                    # Compute new row index with wrap-around
                    jj = j + dy
                    if jj >= h:
                        jj = jj - h  # Wrap around to the top
                    elif jj < 0:
                        jj = jj + h  # Wrap around to the bottom

                    # Assign new pixel values for RGB(A) channels
                    rgb_array[i, j, 0] = rgb_array[i, jj, 0]
                    rgb_array[i, j, 1] = rgb_array[i, jj, 1]
                    rgb_array[i, j, 2] = rgb_array[i, jj, 2]
                    rgb_array[i, j, 3] = rgb_array[i, jj, 3]



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef inline unsigned int rgb_to_int(
    const unsigned char red,
    const unsigned char green,
    const unsigned char blue)nogil:

    """
    Convert RGB color values into a single 32-bit integer, 
    similar to pygame's map_rgb() function.

    This Cython `cpdef` function allows direct calls without requiring a Python hook function.
    It efficiently encodes red, green, and blue values into a packed integer representation.

    :param red   : 
        Red color component (0-255).
        
    :param green : 
        Green color component (0-255).
        
    :param blue  : 
        Blue color component (0-255).
        
    :return      : 
        A 32-bit unsigned integer representing the combined RGB color.
        
    """
    return (<unsigned int>red << 16) | (<unsigned int>green << 8) | blue




@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef inline unsigned int rgb_to_int_c(
    const unsigned char red,
    const unsigned char green,
    const unsigned char blue
)nogil:
    """
    Convert RGB color values into a single 32-bit integer, 
    similar to pygame's map_rgb() function.

    This Cython `cpdef` function allows direct calls without requiring a Python hook function.
    It efficiently encodes red, green, and blue values into a packed integer representation.

    :param red   : 
        Red color component (0-255).
        
    :param green : 
        Green color component (0-255).
        
    :param blue  : 
        Blue color component (0-255).
        
    :return      : 
        A 32-bit unsigned integer representing the combined RGB color.
        
    """
    return (<unsigned int>red << 16) | (<unsigned int>green << 8) | blue



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef inline (unsigned char, unsigned char, unsigned char) int_to_rgb(const unsigned int n)nogil:
    """
    Convert a 32-bit integer into an RGB color representation.

    This function extracts the red, green, and blue components from a packed
    integer format (similar to pygame's unmap_rgb()) and returns them as tuple RGB 
    containing values in the range [0, 255].

    :param n: A 32-bit unsigned integer representing an RGB color.
    :return : An tuple of unsigned char containing the extracted red, green, and blue values.
    """

    return <unsigned char>((n >> <unsigned short int>16) & <unsigned char>255),\
           <unsigned char> ((n >> <unsigned short int> 8) & <unsigned char> 255),\
           <unsigned char> (n & <unsigned char> 255)



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef inline (unsigned char, unsigned char, unsigned char) int_to_rgb_c(const unsigned int n)nogil:
    """
    Convert a 32-bit integer into an RGB color representation.

    This function extracts the red, green, and blue components from a packed
    integer format (similar to pygame's unmap_rgb()) and returns them as tuple RGB 
    containing values in the range [0, 255].

    :param n: A 32-bit unsigned integer representing an RGB color.
    :return : An tuple of unsigned char containing the extracted red, green, and blue values.
    
    """

    return <unsigned char>((n >> <unsigned short int>16) & <unsigned char>255),\
        <unsigned char>((n >> <unsigned short int>8) & <unsigned char>255), \
        <unsigned char>(n & <unsigned char>255)


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef inline float _randf(const float lower, const float upper) nogil:
    """
    Generate a random floating-point number in the range [lower, upper).

    This function is an optimized alternative to `random.uniform(lower, upper)`, 
    leveraging an external C function (`randRangeFloat`) for improved performance.

    **Note:** This function operates without the Python GIL (`nogil`), 
    making it safe for parallel execution in Cython.

    :param lower: The lower bound of the range (inclusive).
    :param upper: The upper bound of the range (exclusive).
    :return: A random float value in the range [lower, upper).
    """

    # Call an external C function to generate a random float
    return randRangeFloat(lower, upper)



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef inline float _randf_c(const float lower, const float upper) nogil:
    """
    Generate a random floating-point number in the range [lower, upper).

    This function is a highly optimized alternative to `random.uniform(lower, upper)`, 
    utilizing an external C function (`randRangeFloat`) for improved performance.

    **Performance Note:**  
    - This function is declared `inline` for faster execution.
    - It operates without the Python Global Interpreter Lock (`nogil`), making it safe 
      for use in parallelized Cython code.

    **Parameters:**
    :param lower: The lower bound of the range (inclusive).
    :param upper: The upper bound of the range (exclusive).

    **Returns:**
    :return: A random float value in the range [lower, upper).
    """

    # Call an external C function to generate a random float within the given range
    return randRangeFloat(lower, upper)


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef inline int _randi(const int lower, const int upper) nogil:
    """
    Generate a random integer in the range [lower, upper].

    This function is an optimized alternative to `random.randint(lower, upper)`, 
    using an external C function (`randRange`) for improved performance.

    **Performance Notes:**
    - Declared as `inline` for faster execution.
    - Operates without the Python Global Interpreter Lock (`nogil`), making it 
      safe for use in parallelized Cython code.

    **Parameters:**
    :param lower: The lower bound of the range (inclusive).
    :param upper: The upper bound of the range (inclusive).

    **Returns:**
    :return: A random integer in the range [lower, upper].

    **Note:** 
    - The function does not check whether `lower <= upper`, so incorrect usage 
      may result in undefined behavior depending on `randRange` implementation.
    """

    # Call an external C function to generate a random integer within the given range
    return randRange(lower, upper)



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef inline int _randi_c(const int lower, const int upper) nogil:
    """
    Generate a random integer in the range [lower, upper].

    This function is an optimized equivalent to `random.randint(lower, upper)`, 
    utilizing an external C function (`randRange`) for better performance.

    **Performance Notes:**
    - Declared as `inline` for reduced function call overhead.
    - Operates without the Python Global Interpreter Lock (`nogil`), allowing 
      safe execution in multithreaded Cython code.

    **Parameters:**
    :param lower: The lower bound of the range (inclusive).
    :param upper: The upper bound of the range (inclusive).

    **Returns:**
    :return: A random integer in the range [lower, upper].

    **Notes:** 
    - This function directly calls an external C function (`randRange`).
    - No input validation is performed, so passing `lower > upper` may lead 
      to undefined behavior depending on the `randRange` implementation.
    """

    # Call an external C function to generate a random integer within the given range
    return randRange(lower, upper)



# todo this can be simplify using a single loop
#  using a transposition of the index
@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef unsigned char[::1] combine_rgba_buffers(
    const Py_ssize_t w,
    const Py_ssize_t h,
    const unsigned char [::1] bgr_buffer,
    const unsigned char [::1] alpha_channel,
    unsigned char [::1] output_rgba_buffer,
    bint transpose_output=False):

    """
    Combines separate BGR and Alpha memory buffers into a contiguous RGBA buffer.

    This function takes separate BGR (Blue-Green-Red) and Alpha memory buffers, 
    stacks them together into a single RGBA buffer, and optionally transposes the result.
    It is optimized using Cython and parallelized with OpenMP for improved performance.

    If `transpose_output` is True, the output memory view is flipped by swapping 
    rows and columns.

    Parameters:
    -----------
    w : int
        The width of the texture.
        
    h : int
        The height of the texture.
        
    bgr_buffer : memoryview (unsigned char [::1])
        A 1D contiguous memory view containing packed BGR values (uint8). 
        Expected size: `width * height * 3`.
        
    alpha_channel : memoryview (unsigned char [::1])
        A 1D contiguous memory view containing alpha values (uint8).
        Expected size: `width * height`.
        
    output_rgba_buffer : memoryview (unsigned char [::1])
        A pre-allocated 1D memory view to store the resulting RGBA values (uint8).
        Expected size: `width * height * 4`.
        
    transpose_output : bool, optional (default=False)
        If True, the resulting RGBA buffer is transposed (flipped).

    Returns:
    --------
    memoryview (unsigned char [::1])
        A contiguous memory view containing the combined RGBA pixel values (uint8).

    Raises:
    -------
    ValueError:
        - If w or h is non-positive.
        - If `bgr_buffer` size does not match `width * height * 3`.
        - If `alpha_channel` size does not match `width * height`.
        - If `output_rgba_buffer` is not large enough to store `width * height * 4`.

    Notes:
    ------
    - Uses `memcpy` for efficient memory copying.
    - Supports optional transposition of the buffer.
    - Designed for performance-critical applications.
    """

    cdef:
        unsigned short int rgb_bytesize = 3
        unsigned short int rgba_bytesize = 4
        int buffer_size = w * h * rgb_bytesize
        int new_size = w * h * rgba_bytesize

    if w <= 0 or h <= 0:
        raise ValueError("Width and height must be positive integers.")

    if len(bgr_buffer) != buffer_size:
        raise ValueError("bgr_buffer size does not match expected dimensions.")

    if len(alpha_channel) != w * h:
        raise ValueError("alpha_channel size does not match expected dimensions.")

    if len(output_rgba_buffer) != new_size:
        raise ValueError("output_rgba_buffer size does not match expected dimensions.")

    cdef:
        unsigned char [::1] transposed_buffer = numpy.empty(new_size, dtype=uint8)
        int i=0, j=0
        int alpha_index, index
        int rgba_index
        int h4 = h * rgba_bytesize
        int transposed_index

    with nogil:

        for i in prange(0, buffer_size, rgb_bytesize, schedule=SCHEDULE, num_threads=THREADS):

            alpha_index = i // rgb_bytesize
            rgba_index = alpha_index * rgba_bytesize

            memcpy(&output_rgba_buffer[ rgba_index ], &bgr_buffer[ i ], 3)
            output_rgba_buffer[ rgba_index + 3 ] = alpha_channel[ alpha_index ]

        if transpose_output:
            for i in prange(0, h4, 4, schedule=SCHEDULE, num_threads=THREADS):
                for j in range(0, w):
                    index = i + (h4 * j)
                    transposed_index = (j * 4) + (i * w)
                    memcpy(&transposed_buffer[ transposed_index ], &output_rgba_buffer[ index ], 4)

    return transposed_buffer if transpose_output else output_rgba_buffer



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef inline float [:, :] normalize_2d_array(const unsigned char [:, :] array2d):

    """
    Normalize a 2D array of unsigned 8-bit integers (uint8) 
    to floating-point values in the range [0, 1].

    This function takes a 2D array with shape (width, height) containing uint8 values (0-255) and 
    converts it into a MemoryViewSlice (2D array) of floats, where each value is rescaled to the range [0, 1].

    Parameters:
    -----------
    array2d : memoryview (unsigned char[:, :])
        A 2D array of shape (width, height) containing uint8 values representing pixel intensities.

    Returns:
    --------
    memoryview (float[:, :])
        A 2D array of shape (width, height) with float values normalized to the range [0, 1].

    Raises:
    -------
    ValueError:
        - If the input array does not have exactly two dimensions.

    Notes:
    ------
    - Uses `prange` with OpenMP for parallelized operations to improve performance.
    - Performs element-wise normalization using the constant `ONE_255 = 1/255.0`.
    - Designed for fast image processing and machine learning applications.
    """

    cdef:
        Py_ssize_t w, h

    try:
        w, h = <object>array2d.shape[ 0 ], <object>array2d.shape[ 1 ]

    except (ValueError, IndexError) as e:
        raise ValueError('\nArray shape not understood. Only 2d array shape (w, h) are compatible.')

    cdef:
        int i = 0, j = 0
        float [:, :] normalized_array = numpy.empty((w, h), numpy.float32)

    with nogil:
        for i in prange(w, schedule=SCHEDULE, num_threads=THREADS):
            for j in range(h):

                normalized_array[i, j] =array2d[i, j] * ONE_255

    return normalized_array




@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef object generate_spectrum_surface(int width, int height, float gamma=1.0):

    """
    Create a pygame surface displaying the light spectrum 380-750 nm

    Color   Wavelength(nm) Frequency(THz)
    Red     620-750        484-400
    Orange  590-620        508-484
    Yellow  570-590        526-508
    Green   495-570        606-526
    Blue    450-495        668-606
    Violet  380-450        789-668

    :param width: 
        integer; w of the image
    
    :param height: 
        integer; h of the image
    
    :param gamma: 
        float; gamma value 
    
    :return: 
        Return a pygame surface 24-bit (w, h) converted for fast blit 

    """

    return generate_spectrum_surface_c(width, height, gamma)


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef object generate_spectrum_surface_c(int width, int height, float gamma=1.0):
    """
    Generate a Pygame surface displaying the visible light spectrum (380-750 nm).

    This function creates an image representation of the electromagnetic spectrum in the 
    visible range, mapping wavelengths (in nanometers) to their corresponding RGB colors.
    
    ### Visible Light Spectrum:
    | Color   | Wavelength (nm) | Frequency (THz)|
    |---------|----------------|-----------------|
    | Red     | 620-750        | 484-400         |
    | Orange  | 590-620        | 508-484         |
    | Yellow  | 570-590        | 526-508         |
    | Green   | 495-570        | 606-526         |
    | Blue    | 450-495        | 668-606         |
    | Violet  | 380-450        | 789-668         |
    
    ### Parameters:
    - **width** (*int*): Width of the generated image.
    - **height** (*int*): Height of the generated image.
    - **gamma** (*float*, optional): Gamma correction value (default is `1.0`), which 
      adjusts brightness and contrast.
    
    ### Returns:
    - **Pygame Surface**: A 24-bit Pygame surface (`(width, height)`) optimized for fast rendering.
    
    ### Notes:
    - Uses parallel computation (`prange`) with OpenMP for efficient processing.
    - Maps each wavelength to an RGB color using `wavelength_to_rgb()`.
    - The generated surface is rescaled to the requested `(width, height)`.
    """

    cdef:
        int i, k  # Loop variables
        rgb_color_int rgb_c  # Structure to hold RGB color values
        unsigned char [:, :, :] spectrum_array = numpy.empty((370, 1, 3), numpy.uint8)
        # Create a NumPy array to store RGB values for wavelengths (380-750 nm = 370 values)

    with nogil:
        # Iterate over the wavelength range from 380 nm to 750 nm
        for i in prange(380, 750, schedule=SCHEDULE, num_threads=THREADS):
          rgb_c = wavelength_to_rgb(i, gamma)  # Convert wavelength to RGB values
          k = i - 380  # Normalize index to fit within the array range (0-369)

          # Store RGB values in the array (single-column format)
          spectrum_array[k, 0, 0] = rgb_c.r  # Red channel
          spectrum_array[k, 0, 1] = rgb_c.g  # Green channel
          spectrum_array[k, 0, 2] = rgb_c.b  # Blue channel

          # Duplicate the RGB values for an additional row (for future expansion if needed)
          spectrum_array[k, 1, 0] = rgb_c.r
          spectrum_array[k, 1, 1] = rgb_c.g
          spectrum_array[k, 1, 2] = rgb_c.b

    # Convert the NumPy array to a Pygame surface
    surface = make_surface(asarray(spectrum_array))

    # Scale the surface to match the requested width and height
    surface = scale(surface, (width, height)).convert()

    return surface  # Return the final Pygame surface


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef bint get_image_format(surface_):
    """
    Determines whether the given Pygame surface is in a 32-bit (RGBA) format.

    This function checks if the provided `pygame.Surface` has a **bit depth of 32** 
    and a **byte size of 4**, which indicates compatibility with **RGBA (32-bit) pixel format**.

    Parameters:
    ----------
    surface_ : pygame.Surface  
        A Pygame surface, expected to be either **24-bit (RGB) or 32-bit (RGBA)**.

    Returns:
    -------
    bool  
        **True** if the surface is **32-bit (RGBA)**, otherwise **False**.

    Raises:
    ------
    TypeError  
        If the provided argument is **not** a `pygame.Surface` object.
    """

    cdef:
        int bitsize, bytesize  # Variables to store bit depth and byte size of the surface

    try:
        # Retrieve the bit depth and byte size of the surface
        bitsize = surface_.get_bitsize()
        bytesize = surface_.get_bytesize()
    except AttributeError:
        # Raise an error if the provided object is not a pygame.Surface
        raise TypeError(
            f'\nExpecting pygame.Surface type for argument surface_, got {type(surface_)}')

    # A 32-bit surface has 32 bits per pixel and 4 bytes per pixel (RGBA format)
    if bitsize == 32 and bytesize == 4:
        return True

    return False


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef bint is_type_memoryview(object object_):
    """
    Checks if the given object is of the `memoryview` type.

    This function verifies if the provided object is a **memoryview** object, 
    which is a built-in Python type used for accessing memory buffers directly. 
    This is useful for working with large data sets, such as arrays, without 
    creating unnecessary copies.

    Parameters
    ----------
    object_ : object
        The Python object to check for being of type `memoryview`.

    Returns
    -------
    bint
        Returns `1` (True) if the object is of type `memoryview`, otherwise `0` (False).

    Example
    -------
    >>> is_type_memoryview(memoryview(b'abc'))
    True

    >>> is_type_memoryview('not a memoryview')
    False
    """
    
    # Check if the type of object is "_memoryviewslice", which indicates it is a memoryview
    if type(object_).__name__ == "_memoryviewslice":
        return True  # Return True if it's a memoryview type
    
    return False  # Return False if it's not a memoryview type


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef bint is_uint8(rgb_array):
    """
    Checks if the given array has the `uint8` data type.

    This function verifies if the provided `rgb_array` has a data type of 
    `numpy.uint8` (unsigned 8-bit integer), which is commonly used for image 
    data with pixel values in the range of [0, 255].

    Parameters
    ----------
    rgb_array : numpy.ndarray or memoryviewslice
        The array to check. It can be a `numpy` array or a `memoryview` slice, 
        typically representing an image with pixel values (e.g., RGB channels).

    Returns
    -------
    bint
        Returns `1` (True) if the array has the `uint8` data type, otherwise `0` (False).

    Raises
    ------
    TypeError
        If the provided array is `None`, indicating no valid array was passed.

    Example
    -------
    >>> is_uint8(numpy.array([[255, 0, 255]], dtype=numpy.uint8))
    True

    >>> is_uint8(numpy.array([[255, 0, 255]], dtype=numpy.float32))
    False
    """

    # Check if the input array is None and raise an error if so
    if rgb_array is None:
        raise TypeError("\nArray cannot be a NoneType")

    # If the input is a memoryview, convert it to a numpy array and check its dtype
    if is_type_memoryview(rgb_array):
        return True if numpy.asarray(rgb_array).dtype == numpy.uint8 else False

    # If the input is a numpy array, check its dtype directly
    else:
        if hasattr(rgb_array, 'dtype'):
            return True if rgb_array.dtype == numpy.uint8 else False

    # Return False if the array is of an unsupported type
    return False


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef bint is_float32(rgb_array):
    """
    Checks if the given array has the `float32` data type.

    This function verifies if the provided `rgb_array` has a data type of 
    `numpy.float32`, which is commonly used for storing floating-point values 
    with 32-bit precision. This can be used to represent images with higher 
    precision pixel values or scientific data.

    Parameters
    ----------
    rgb_array : numpy.ndarray or memoryviewslice
        The array to check. It can be a `numpy` array or a `memoryview` slice, 
        typically representing image data or other numerical arrays.

    Returns
    -------
    bint
        Returns `1` (True) if the array has the `float32` data type, otherwise `0` (False).

    Raises
    ------
    TypeError
        If the provided array is `None`, indicating no valid array was passed.

    Example
    -------
    >>> is_float32(numpy.array([[0.5, 1.0, 0.0]], dtype=numpy.float32))
    True

    >>> is_float32(numpy.array([[0.5, 1.0, 0.0]], dtype=numpy.uint8))
    False
    """

    # Check if the input array is None and raise an error if so
    if rgb_array is None:
        raise TypeError("\nArray cannot be a NoneType")

    # If the input is a memoryview, convert it to a numpy array and check its dtype
    if is_type_memoryview(rgb_array):
        return True if numpy.asarray(rgb_array).dtype == numpy.float32 else False

    # If the input is a numpy array, check its dtype directly
    else:
        if hasattr(rgb_array, 'dtype'):
            return True if rgb_array.dtype == numpy.float32 else False

    # Return False if the array is of an unsupported type
    return False


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef bint is_float64(rgb_array):
    """
    Checks if the given array has the `float64` data type.

    This function checks whether the provided `rgb_array` has the data type of 
    `numpy.float64`, which is a 64-bit floating-point data type commonly used 
    for applications requiring higher precision than `float32`.

    Parameters
    ----------
    rgb_array : numpy.ndarray or memoryviewslice
        The array to check. It can be either a `numpy` array or a `memoryview` slice, 
        typically representing image data or any numerical array that could 
        potentially use 64-bit floating point numbers.

    Returns
    -------
    bint
        Returns `1` (True) if the array has the `float64` data type, otherwise `0` (False).

    Raises
    ------
    TypeError
        If the provided array is `None`, indicating no valid array was passed.

    Example
    -------
    >>> is_float64(numpy.array([[1.5, 2.0, 3.0]], dtype=numpy.float64))
    True

    >>> is_float64(numpy.array([[1.5, 2.0, 3.0]], dtype=numpy.float32))
    False
    """
    
    # Check if the input array is None and raise an error if so
    if rgb_array is None:
        raise TypeError("\nArray cannot be a NoneType")

    # If the input is a memoryview, convert it to a numpy array and check its dtype
    if is_type_memoryview(rgb_array):
        return True if numpy.asarray(rgb_array).dtype == numpy.float64 else False

    # If the input is a numpy array, check its dtype directly
    else:
        if hasattr(rgb_array, 'dtype'):
            return True if rgb_array.dtype == numpy.float64 else False

    # Return False if the array is of an unsupported type
    return False






@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef bint is_int32(rgb_array):
    """
    Checks if the given array has the `int32` data type.

    This function checks whether the provided `rgb_array` has the data type of 
    `numpy.int32`, which is a 32-bit signed integer data type commonly used 
    for applications that require integer precision.

    Parameters
    ----------
    rgb_array : numpy.ndarray or memoryviewslice
        The array to check. It can be either a `numpy` array or a `memoryview` slice, 
        typically representing image data or numerical arrays that use 32-bit signed integers.

    Returns
    -------
    bint
        Returns `1` (True) if the array has the `int32` data type, otherwise `0` (False).

    Raises
    ------
    TypeError
        If the provided array is `None`, indicating no valid array was passed.

    Example
    -------
    >>> is_int32(numpy.array([[1, 2, 3]], dtype=numpy.int32))
    True

    >>> is_int32(numpy.array([[1, 2, 3]], dtype=numpy.float32))
    False
    """
    
    # Check if the input array is None and raise an error if so
    if rgb_array is None:
        raise TypeError("\nArray cannot be a NoneType")

    # If the input is a memoryview, convert it to a numpy array and check its dtype
    if is_type_memoryview(rgb_array):
        return True if numpy.asarray(rgb_array).dtype == numpy.int32 else False

    # If the input is a numpy array, check its dtype directly
    else:
        if hasattr(rgb_array, 'dtype'):
            return True if rgb_array.dtype == numpy.int32 else False

    # Return False if the array is of an unsupported type
    return False




# ************* SORTING ALGORITHM FOR MEDIAN FILTER


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
@cython.cfunc
cpdef inline void bubble_sort(unsigned char [::1] nums, const unsigned int size) nogil:
    """
    Sorts an array of unsigned chars (e.g., bytes or pixel values) 
    using the bubble sort algorithm.

    The function iteratively compares adjacent elements of the array
     and swaps them if they are in the wrong order,
    continuing this process until the array is fully sorted. 
    The algorithm stops when no swaps are made during a complete pass.

    Parameters
    ----------
    nums : unsigned char [::1]
        A one-dimensional memoryview or array of unsigned char values to be sorted.
        Typically, this array will represent byte values or pixel data (e.g., RGB values).

    size : int
        The number of elements in the `nums` array to sort.

    Returns
    -------
    None
        This function sorts the array in place. No value is returned;
         the input array is modified directly.

    Example
    -------
    >>> nums = bytearray([4, 2, 3, 1, 5])
    >>> bubble_sort(nums, len(nums))
    >>> print(nums)
    bytearray(b'\x01\x02\x03\x04\x05')
    
    Notes
    -----
    This is an in-place sorting algorithm, meaning it modifies the original array.
    Bubble sort is not the most efficient sorting algorithm, 
    but it is simple to implement.

    The algorithm runs in O(n^2) time in the worst and average cases,
     which makes it less efficient for large datasets.
    """

    cdef:
        int i
        int last_swapped
        unsigned char *p
        bint swapped

    # Set swapped to True to start the loop at least once
    swapped = True
    last_swapped = size - 1  # Initially, the last swapped index is the last element in the array

    while swapped:
        swapped = False
        # Limit the comparison range based on the last swapped index
        for i in range(last_swapped):
            p  = &nums[i]
            # If the current element is greater than the next, swap them
            if p[0] > (p + 1)[0]:
                p[0], (p + 1)[0] = (p + 1)[0], p[0]
                swapped = True
                last_swapped = i  # Update the last swapped index



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef inline void insertion_sort(unsigned char [::1] nums, const unsigned int size) nogil:
    """
    Sorts an array of unsigned char elements using the Insertion Sort algorithm.

    Insertion sort iterates through the array and picks each element one by one,
    inserting it into the sorted portion of the array, shifting the other elements
    accordingly. This implementation uses parallel processing for improved performance.

    Example usage:
    nums = bytearray([5, 2, 9, 1, 5, 6])
    insertion_sort(nums, len(nums))

    :param nums:
        A 1D numpy array or memory view of unsigned char (uint8) elements,
        representing the array to be sorted.

    :param size:
        The size of the array `nums`. It specifies the number of elements in the array.

    :return:
        This function does not return any value; it sorts the array in-place.
    """

    cdef:
        int i, j               # Loop variables for iterating through the array
        unsigned char item_to_insert  # The current item being inserted into the sorted portion

    # Iterate through the array starting from the second element (index 1)
    for i in prange(1, size, schedule=SCHEDULE, num_threads=THREADS):
        item_to_insert = nums[i]  # Store the current item to be inserted

        j = i - 1  # Start comparing the item with the previous elements
        # Find the correct position for the item by shifting larger elements
        while j >= 0 and nums[j] > item_to_insert:
            nums[j + 1] = nums[j]  # Shift the larger element to the right
            j = j - 1  # Move to the next element

        # Insert the item into its correct position
        nums[j + 1] = item_to_insert




@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
# There are different ways to do a Quick Sort partition, this implements the
# Hoare partition scheme. Tony Hoare also created the Quick Sort algorithm.
cpdef inline int partition_cython(unsigned char [::1] nums, const unsigned int low, const unsigned int high) nogil:
    """
    Partition function for quicksort algorithm.

    This function selects a pivot element and partitions the input array
    `nums` such that all elements less than the pivot are moved to its left,
    and all elements greater than the pivot are moved to its right. It returns
    the index of the partition point.

    :param nums: 1D memoryview of unsigned char values (input array).
    :param low: The starting index of the subarray to be partitioned.
    :param high: The ending index of the subarray to be partitioned.
    :return: The index of the partition point.
    """
    cdef:
        unsigned int pivot  # Pivot element for partitioning
        int i, j   # Indices used for scanning the array

    pivot = nums[(low + high) >> 1]  # Choose the middle element as pivot
    i = low - 1
    j = high + 1

    while True:
        # Move `i` to the right until an element >= pivot is found
        i += 1
        while nums[i] < pivot:
            i += 1

        # Move `j` to the left until an element <= pivot is found
        j -= 1
        while nums[j] > pivot:
            j -= 1

        # If pointers meet or cross, return partition index
        if i >= j:
            return j

        # Swap elements at i and j to ensure correct partitioning
        nums[i], nums[j] = nums[j], nums[i]



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef inline void _quick_sort(unsigned char [::1] items, unsigned int low, unsigned int high) nogil:
    """
    In-place quicksort implementation for an array of unsigned char values.

    This function recursively sorts the array `items` by selecting a pivot and partitioning
    the array such that elements smaller than the pivot move to the left and larger elements
    move to the right.

    Optimized to reduce recursion depth by sorting the smaller partition first.

    :param items: 1D memoryview of unsigned char values (input array).
    :param low: The starting index of the subarray to be sorted.
    :param high: The ending index of the subarray to be sorted.
    """
    cdef int split_index

    while low < high:
        # Partition the array and get the pivot index
        split_index = partition_cython(items, low, high)

        # Recur on the smaller half to minimize recursion depth
        if split_index - low < high - split_index:
            _quick_sort(items, low, split_index)  # Sort left half first
            low = split_index + 1  # Move to sorting the right half
        else:
            _quick_sort(items, split_index + 1, high)  # Sort right half first
            high = split_index  # Move to sorting the left half


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef inline void heapify(unsigned char [::1] nums, const unsigned int heap_size, unsigned int root_index) nogil:
    """
    Ultra-optimized iterative heapify function for maintaining the max heap property.

    This function “sifts down” the value at the given root index so that the subtree
    rooted at that index becomes a valid max heap. It is assumed that the subtrees 
    below are already valid heaps. For full heapsort, make sure to build the heap by 
    calling this function on all non-leaf nodes in reverse order.

    :param nums: 1D memoryview of unsigned char values representing the heap.
    :param heap_size: The number of elements in the heap.
    :param root_index: The index of the root node of the subtree to heapify.
    """

    cdef int largest, left_child, right_child
    cdef unsigned char temp
    cdef unsigned char* arr = &nums[0]  # Use raw pointer for faster access

    temp = arr[root_index]  # Store the original root value

    while True:
        left_child = (root_index << 1) + 1  # Compute left child index: 2 * root_index + 1
        if left_child >= heap_size:
            break  # No children exist; stop

        # Assume left child is the largest candidate
        largest = left_child
        right_child = left_child + 1  # Right child index

        # More robust: compare right child with the current largest candidate
        if right_child < heap_size and arr[right_child] > arr[largest]:
            largest = right_child

        # If the root's value is larger than or equal to the largest child's value, we're done
        if temp >= arr[largest]:
            break

        # Move the larger child up
        arr[root_index] = arr[largest]
        root_index = largest  # Continue sifting down from the child position

    # Place the original root value into its correct position
    arr[root_index] = temp





@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef inline void heap_sort(unsigned char [::1] nums, const unsigned int n) nogil:
    """
    Performs in-place heap sort on an array of unsigned char values.

    This function first builds a max heap from the input array, ensuring the largest 
    element is at the root. Then, it repeatedly extracts the maximum element 
    (swaps it with the last element) and re-heapifies the reduced heap to maintain 
    heap properties until the entire array is sorted.

    Time Complexity:
        - Heap construction: O(n)
        - Extraction and re-heapification: O(n log n)
        - Overall: O(n log n)

    Space Complexity: O(1) (in-place sorting)

    :param nums: 1D memoryview of unsigned char values to be sorted.
    :param n: Number of elements in the array.
    """
    cdef int i

    # Step 1: Build a max heap by heapifying all non-leaf nodes in reverse order
    for i in range(n >> 1, -1, -1):  # Start from last non-leaf node down to root
        heapify(nums, n, i)

    # Step 2: Extract elements one by one from the heap
    for i in range(n - 1, 0, -1):
        # Swap the root (largest element) with the last element of the heap
        nums[i], nums[0] = nums[0], nums[i]
        
        # Heapify the reduced heap to restore heap property
        heapify(nums, i, 0)

