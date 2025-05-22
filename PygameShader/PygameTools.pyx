# cython: binding=False, boundscheck=False, wraparound=False, nonecheck=False, cdivision=True,
# profile=False, initializedcheck=False, exceptval(check=False)
# cython: optimize.use_switch=True
# cython: warn.maybe_uninitialized=False
# cython: warn.unused=False
# cython: warn.unused_result=False
# cython: warn.unused_arg=False
# cython: language_level=3
# cython: write_stub_file=True
# cython: profile=True
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
        PyObject_CallMethod, PyObject_CallObject


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
from PygameShader.misc cimport is_type_memoryview
from PygameShader.misc cimport get_image_format, is_uint8, is_float64, is_int32
from libc.math cimport roundf as round_c, fmodf
from libc.math cimport floorf  as floor_c, sqrtf as sqrt, powf as pow, roundf as round_f
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

cdef float c_1_255 = <float>1.0/<float>255.0


"""
REF 

http://www.brucelindbloom.com/
http://www.differencebetween.net/technology/difference-between-rgb-and-srgb/
https://en.wikipedia.org/wiki/SRGB#The_reverse_transformation
http://www.brucelindbloom.com/index.html?Eqn_XYZ_to_RGB.html
https://craftofcoding.files.wordpress.com/2018/08/pracn6_whitebal.pdf
https://craftofcoding.files.wordpress.com/2018/08/pracn6_whitebal.pdf
"""


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef inline np.ndarray[np.uint8_t, ndim=3] RemoveAlpha(
        const unsigned char[:, :, :] rgba_array
):
    """
    Convert an RGBA or BGRA array to RGB by removing the alpha channel.
    This function is used to process images with an alpha transparency channel
    and convert them to a format with no alpha, reducing a 32-bit image to a 24-bit image.

    The function accepts a 3D numpy array or memoryview containing RGBA/BGRA data,
    and outputs a new 3D array with the alpha channel removed (RGB).

    ### Example:
    ```python
    rgb_array = RemoveAlpha(rgba_array)
    ```

    :param rgba_array: 
        A numpy.ndarray or memoryviewslice with shape (w, h, 4) and dtype uint8.
        The array represents an image in RGBA or BGRA format (with alpha transparency).

    :return: 
        A new numpy.ndarray with shape (w, h, 3) and dtype uint8, containing the RGB
        values (with no alpha transparency).
        
    :raises ValueError: If the input array does not have the expected shape (w, h, 4).
    :raises TypeError: If the input array is not of type uint8.
    """

    # Check that the input array is a valid type
    if not (isinstance(rgba_array, numpy.ndarray) or rgba_array.__class__.__name__ == '_memoryviewslice'):
        raise TypeError("Input rgb_array must be a numpy.ndarray or memoryviewslice.")

    # Check if the array has 3 dimensions and the last dimension size is 4 (for RGBA or BGRA)
    if <object>rgba_array.ndim != 3 or <object>rgba_array.shape[ 2 ] != 4:
        raise ValueError("Input array must have shape (w, h, 4) for RGBA or BGRA format")

    # Check if the array dtype is uint8
    if not is_type_memoryview(rgba_array):
        if <object>rgba_array.dtype != numpy.uint8:
            raise TypeError("Input array must have dtype uint8")

    # Check if the array is empty
    if <object>rgba_array.size == 0:
        raise ValueError("Input array is empty")

    cdef:
        # Declare variables for image dimensions (width, height) and output array
        int i, j
        # Get the dimensions of the input array (width and height)
        Py_ssize_t w = <object>rgba_array.shape[0]
        Py_ssize_t h = <object>rgba_array.shape[1]
        # Create a new empty numpy array to store the RGB values (no alpha channel)
        unsigned char [:, :, :] rgb_array = numpy.empty((h, w, 3), dtype=numpy.uint8)

    # Begin parallelized operation to process the input array without holding the GIL (Global Interpreter Lock)
    with nogil:

        # Loop over the height (rows) of the image in parallel using prange for multi-threading
        for j in prange(h, schedule=SCHEDULE, num_threads=THREADS):
            # Loop over the width (columns) of the image
            for i in range(w):
                # Copy RGB values from input (rgba_array) to the output (rgb_array),
                # ignoring the alpha channel (which is rgba_array[i, j, 3] in the 4th channel).
                rgb_array[j, i, 0] = rgba_array[i, j, 0]  # Red channel
                rgb_array[j, i, 1] = rgba_array[i, j, 1]  # Green channel
                rgb_array[j, i, 2] = rgba_array[i, j, 2]  # Blue channel

    # Return the new RGB array as a numpy array (with no alpha channel)
    return numpy.array(rgb_array)


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef unsigned char [:, :, ::1] resize_array(
    const unsigned char [:, :, :] rgb_array,
    const Py_ssize_t w2,
    const Py_ssize_t h2
):

    """
    Rescale an array (returning a new array).

    This function rescales a 3D RGB or RGBA array (of type uint8) 
    to a new width and height. The output will have the shape (w2, h2, 3|4),
    preserving the number of channels (3 for RGB or 4 for RGBA).
    
    e.g:
    result = resize_array(rgb_array, 800, 600)
    
    Parameters
    ----------
    rgb_array : 
        numpy.ndarray or memoryviewslice of shape (w, h, 3|4).
        Contains pixels in RGB or RGBA format (uint8).

    w2 : 
        int; The width of the output array.
         
    h2 : 
        int; The height of the output array.

    Returns
    -------
    numpy.ndarray or memoryviewslice:
        A new array with the shape (w2, h2, 3|4) and type uint8.
        The format of the pixel data (RGB or RGBA) is the same as the input.

    Raises
    ------
    ValueError:
        If the input array does not have a valid shape (w, h, 3|4) or the `bit_size` is not supported.
    TypeError:
        If the input array is not of type uint8.
    TypeError:
        If the input is not a numpy array or memoryviewslice, or if the data type is not uint8.
    
    ### ChatGPT Collaboration:
    - Assisted in improving the function's readability, error handling, and type checks
    
    """

    # Check that the input array is a valid type
    if not (isinstance(rgb_array, numpy.ndarray) or rgb_array.__class__.__name__ == '_memoryviewslice'):
        raise TypeError("Input rgb_array must be a numpy.ndarray or memoryviewslice.")

    cdef:
        Py_ssize_t w, h
        Py_ssize_t bit_size = 0 # Number of channels (3 for RGB, 4 for RGBA)

    try:
        # Extract the dimensions of the input array
        w, h, bit_size = (<object>rgb_array).shape[ :3 ]

        # Only RGB or RGBA arrays are supported (bit_size must be 3 or 4)
        if bit_size not in (3, 4):
            raise ValueError(
                "Invalid bit_size: only RGB or RGBA arrays are supported."
            )

    except ValueError as e:
        # Handle invalid array shapes
        # If the input is a memoryview, display its flags
        if is_type_memoryview(rgb_array):
            print(numpy.array(rgb_array).flags)
        # If the input is a numpy array, display its flags
        else:
            print(rgb_array.flags)

        # Re-raise the exception with more context
        raise ValueError(
            f"\n{e}\nExpecting array shape (w, h, n) for RGB or RGBA; "
            f"got ({w if w else 'unknown'}, {h if h else 'unknown'},"
            f" {len(rgb_array[ :3 ]) if hasattr(rgb_array, '__len__') else 'unknown'})"
        )

        # Ensure the input array is of type uint8
        if not is_uint8(rgb_array):
            raise TypeError(
                f"\nExpecting uint8 (unsigned char) data type, but got {rgb_array.dtype}."
            )

    # Call the C-implemented function to perform the scaling
    return resize_array_c(rgb_array, w2, h2)



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef unsigned char [:, :, ::1] resize_array_c(
        const unsigned char [:, :, :] rgb_array,
        const Py_ssize_t w2,
        const Py_ssize_t h2):

    """
    
    Array rescale (return a new array)
    
    This function rescales an image array in RGB or RGBA format (uint8) to a specified width (`w2`) 
    and height (`h2`). The resulting array maintains the pixel format of the input and has a shape 
    of (w2, h2, 3|4).
    
    Example:
        new_array = resize_array_c(rgb_array, 800, 400)
    
    Parameters
    ----------
    rgb_array : numpy.ndarray or memoryviewslice
        A 3D array with a shape of (w, h, 3|4) representing the pixel data of the image.
        The array must be in RGB (3 channels) or RGBA (4 channels) format and have a dtype of uint8.

    w2 : int
        The width of the resized image. Must be a positive integer greater than 0.

    h2 : int
        The height of the resized image. Must be a positive integer greater than 0.

    Returns
    -------
    memoryviewslice : 
        A resized image array with shape (w2, h2, 3|4), dtype uint8, and the same pixel format
        (RGB or RGBA) as the input.

    
    Raises:
    ------
    ValueError:
        If `w2` or `h2` is not greater than 0, or if the shape of `rgb_array` is not understood 
        (must be (w, h, 3|4)).
    
    TypeError:
        If `rgb_array` is not a numpy.ndarray or memoryviewslice, or if its dtype is not uint8.
    
    
    Collaboration:
    --------------
    This code includes improvements and comments suggested by ChatGPT.
    """

    # Check that the input array is a valid type
    if not (isinstance(rgb_array, numpy.ndarray) or rgb_array.__class__.__name__ == '_memoryviewslice'):
        raise TypeError("Input rgb_array must be a numpy.ndarray or memoryviewslice.")

    # Ensure the new dimensions (w2 and h2) are valid
    if not w2 > 0:
        raise ValueError("Argument w2 must be a positive integer greater than 0.")

    if not h2 > 0:
        raise ValueError("Argument h2 must be a positive integer greater than 0.")

    # Initialize variables for the original dimensions and bit size
    cdef Py_ssize_t w1, h1, bit_size

    # Try to extract the dimensions of the input array
    try:

        w1, h1, bit_size = rgb_array.shape[:3]  # w1, h1: original dimensions; bit_size: number of channels (3 or 4)

    except (ValueError, AttributeError) as e:
        # Raise an error if the input array shape is not understood
        raise ValueError("Input rgb_array must have a shape of (w, h, 3|4).")

    # Ensure the array has the correct number of channels (3 for RGB, 4 for RGBA)
    if bit_size not in (3, 4):
        raise ValueError(
            f"Unsupported number of channels in rgb_array. "
            f"Expected 3 (RGB) or 4 (RGBA), but got {bit_size}."
        )



    # Declare variables for the rescaling process
    cdef:
        # Create an empty output array
        unsigned char[:, :, ::1] new_array = numpy.empty((w2, h2, bit_size), numpy.uint8)
        float fx = <float>w1 / <float>w2  # Calculate horizontal scaling factor
        float fy = <float>h1 / <float>h2  # Calculate scaling factor
        int x, y
        Py_ssize_t xx, yy  # Original coordinates corresponding to the scaled coordinates
        unsigned char * index  # Pointer to the current pixel in the new array

    # Perform the rescaling operation without the Python Global Interpreter Lock (GIL) for better performance
    with nogil:
        # Loop over the width of the new array
        for x in prange(w2, schedule=SCHEDULE, num_threads=THREADS):
            xx = <int>(x * fx)  # Map the scaled x-coordinate to the original x-coordinate

            # Loop over the height of the new array
            for y in range(h2):
                yy = <int>(y * fy)  # Map the scaled y-coordinate to the original y-coordinate
                index = &new_array[x, y, 0]  # Pointer to the pixel data in the new array

                # Copy the RGB channels
                index[0] = <unsigned char>rgb_array[xx, yy, 0]
                (index + 1)[0] = <unsigned char>rgb_array[xx, yy, 1]
                (index + 2)[0] = <unsigned char>rgb_array[xx, yy, 2]

                # If the input array is RGBA (4 channels), copy the alpha channel
                if bit_size == 4:
                    (index + 3)[0] = <unsigned char>rgb_array[xx, yy, 3]

    # Return the rescaled array
    return new_array



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef  unsigned char [:, :, :] make_rgba_array(
        const unsigned char [:, :, :] rgb_,
        const unsigned char [:, :] alpha_,
        bint transpose_=False
):
    """
    Function: Create RGBA Array from RGB and Alpha Arrays
    
    This function generates a new RGBA memoryviewslice array by combining an RGB array 
    (shape (w, h, 3)) and an Alpha array (shape (w, h)). The resulting RGBA array has 
    a shape of (w, h, 4) or (h, w, 4) if `transpose_` is set to `True`. The function 
    is useful for creating images with transparency, such as 32-bit PNG images.
    
    Example Usage:
        im = pygame.image.load("../Assets/px.png").convert_alpha()
        w, h = im.get_width(), im.get_height()
        rgb = pygame.surfarray.pixels3d(im)
        alpha = pygame.surfarray.pixels_alpha(im)
        rgba = make_rgba_array(rgb, alpha, transpose_=True)
        surf = pygame.image.frombuffer(rgba, (w, h), "RGBA").convert_alpha()
    
    Parameters:
    ----------
    rgb_ : numpy.ndarray or memoryviewslice
        A 3D array with shape (w, h, 3) containing RGB pixel values (uint8). The dimensions
        must match the `alpha_` array.
    
    alpha_ : numpy.ndarray or memoryviewslice
        A 2D array with shape (w, h) containing alpha (transparency) values (uint8). The 
        dimensions must match the `rgb_` array.
    
    transpose_ : bool
        A flag to indicate if the output array should be transposed. If `True`, the shape 
        of the output will be (h, w, 4) instead of (w, h, 4).
    
    Returns:
    -------
    memoryviewslice:
        A Cython memoryviewslice with shape (w, h, 4) or (h, w, 4) containing RGBA pixel 
        values (uint8). This array can be directly used for image processing or converted 
        to a numpy.ndarray.
    
    Raises:
    ------
    TypeError:
        If `rgb_` or `alpha_` is not a numpy.ndarray or memoryviewslice, or if the data 
        type is not uint8.
    
    ValueError:
        If the dimensions of `rgb_` and `alpha_` do not match.
    
    Notes:
    -----
    - Both input arrays (`rgb_` and `alpha_`) must be of uint8 data type.
    - The function is designed for high-performance processing using memoryviewslice.
    
    Collaboration:
    --------------
    This implementation includes improvements and comments by ChatGPT.
    """

    # Note:
    # When cythonizing, Cython automatically checks variable type limits,
    # as well as array shapes and types, ensuring compliance with defined constraints.

    # Check that the input `rgb_` is either a numpy array or a memoryviewslice.
    if not (isinstance(rgb_, numpy.ndarray) or rgb_.__class__.__name__ == '_memoryviewslice'):
        raise TypeError("Input rgb_ must be a numpy.ndarray or memoryviewslice.")

    # Check that the input `alpha_` is either a numpy array or a memoryviewslice.
    if not (isinstance(alpha_, numpy.ndarray) or alpha_.__class__.__name__ == '_memoryviewslice'):
        raise TypeError("Input alpha_ must be a numpy.ndarray or memoryviewslice.")

    # Validate that `rgb_` is a 3D array with the last dimension being size 3 (RGB).
    if <object> rgb_.ndim != 3 or <object> rgb_.shape[ 2 ] != 3:
        raise ValueError("Input rgb_ must be a 3D array with shape (w, h, 3).")

    # If `rgb_` is not a memoryviewslice, check its data type.
    if not is_type_memoryview(rgb_):
        if rgb_.dtype != numpy.uint8:
            raise TypeError("Input rgb_ must have a dtype of uint8.")

    # Validate that `alpha_` is a 2D array.
    if <object> alpha_.ndim != 2:
        raise ValueError("Input alpha_ must be a 2D array with shape (w, h).")

    # If `alpha_` is not a memoryviewslice, check its data type.
    if not is_type_memoryview(alpha_):
        if <object> alpha_.dtype != numpy.uint8:
            raise TypeError("Input alpha_ must have a dtype of uint8.")

    # Extract the dimensions of the RGB array.
    cdef Py_ssize_t w1, h1, w2, h2
    w1, h1 = (<object> rgb_).shape[ :2 ]

    # Extract the dimensions of the Alpha array.
    w2, h2 = (<object> alpha_).shape

    if (w1 != w2) or (h1 != h2):
        # Ensure that the dimensions of `rgb_` and `alpha_` match.
        raise ValueError(
            f"Input dimensions mismatch: rgb_ ({w1}, {h1}) vs alpha_ ({w2}, {h2})."
        )

    # Call the Cython implementation of the function `make_rgba_array_c` to create the RGBA array.
    return make_rgba_array_c(
        rgb_,
        alpha_,
        transpose_
    )


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef  unsigned char [:, :, :] make_rgba_array_c(
        const unsigned char [:, :, :] rgb_,
        const unsigned char [:, :] alpha_,
        bint transpose_ = False
):
    """
    
    Function: Create RGBA Array from RGB and Alpha Arrays
    
    This function generates a new RGBA memoryviewslice array by combining an RGB array 
    (shape (w, h, 3)) and an Alpha array (shape (w, h)). The resulting RGBA array has 
    a shape of (w, h, 4) or (h, w, 4) if `transpose_` is set to `True`. The function 
    is useful for creating images with transparency, such as 32-bit PNG images.
    
    Example Usage:
        im = pygame.image.load("../Assets/px.png").convert_alpha()
        w, h = im.get_width(), im.get_height()
        rgb = pygame.surfarray.pixels3d(im)
        alpha = pygame.surfarray.pixels_alpha(im)
        rgba = make_rgba_array_c(rgb, alpha, transpose_=True)
        surf = pygame.image.frombuffer(rgba, (w, h), "RGBA").convert_alpha()
    
    Parameters:
    ----------
    rgb_ : numpy.ndarray or memoryviewslice
        A 3D array with shape (w, h, 3) containing RGB pixel values (uint8). The dimensions
        must match the `alpha_` array.
    
    alpha_ : numpy.ndarray or memoryviewslice
        A 2D array with shape (w, h) containing alpha (transparency) values (uint8). The 
        dimensions must match the `rgb_` array.
    
    transpose_ : bool
        A flag to indicate if the output array should be transposed. If `True`, the shape 
        of the output will be (h, w, 4) instead of (w, h, 4).
    
    Returns:
    -------
    memoryviewslice:
        A Cython memoryviewslice with shape (w, h, 4) or (h, w, 4) containing RGBA pixel 
        values (uint8). This array can be directly used for image processing or converted 
        to a numpy.ndarray.
    
    Notes:
    -----
    - Both input arrays (`rgb_` and `alpha_`) must be of uint8 data type.
    - The function is designed for high-performance processing using memoryviewslice.
    
     Collaboration:
    --------------
    This implementation includes improvements and comments by ChatGPT.
    """
    # Note:
    # When cythonizing, Cython automatically checks variable type limits,
    # as well as array shapes and types, ensuring compliance with defined constraints.

    # Define local variables for the width, height, and bit size of the input RGB array.
    cdef:
        Py_ssize_t w, h  # Dimensions of the input array (width and height).
        Py_ssize_t bit_size  # Number of channels in the RGB input array (should be 3 for RGB).

    # Extract the dimensions (width, height, and bit size) from the input `rgb_` array.
    w, h, bit_size = (<object>rgb_).shape[:3]

    # Define additional local variables:
    cdef:
        int i, j  # Loop counters for iterating through the pixels.
        unsigned char[:, :, ::1] rgba_array  # Output array for RGBA data.

    # Check the `transpose_` flag to determine the shape of the output RGBA array.
    if transpose_:
        # If transposing, the width and height are swapped.
        rgba_array = empty((h, w, 4), dtype=numpy.uint8)
    else:
        # If not transposing, keep the same width and height.
        rgba_array = empty((w, h, 4), dtype=numpy.uint8)

    # Enable the `nogil` block for thread-safe operations without the Global Interpreter Lock (GIL).
    with nogil:
        # Check if transposing is required.
        if transpose_:
            # Parallelize the outer loop over the height (h) using OpenMP `prange`.
            for j in prange(h, schedule=SCHEDULE, num_threads=THREADS):
                for i in range(w):  # Inner loop over the width (w).
                    # Copy RGB values from `rgb_` to the transposed `rgba_array`.
                    rgba_array[j, i, 0] = rgb_[i, j, 0]  # Red channel.
                    rgba_array[j, i, 1] = rgb_[i, j, 1]  # Green channel.
                    rgba_array[j, i, 2] = rgb_[i, j, 2]  # Blue channel.
                    rgba_array[j, i, 3] = alpha_[i, j]   # Alpha channel.
        else:
            # If no transposing, directly map RGB and alpha values into `rgba_array`.
            for j in prange(h, schedule=SCHEDULE, num_threads=THREADS):
                for i in range(w):  # Iterate over each pixel.
                    # Copy RGB values and alpha values into the non-transposed `rgba_array`.
                    rgba_array[i, j, 0] = rgb_[i, j, 0]  # Red channel.
                    rgba_array[i, j, 1] = rgb_[i, j, 1]  # Green channel.
                    rgba_array[i, j, 2] = rgb_[i, j, 2]  # Blue channel.
                    rgba_array[i, j, 3] = alpha_[i, j]   # Alpha channel.

    # Return the resulting RGBA array as a Cython memoryviewslice.
    return rgba_array



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef object create_rgba_surface(
        const unsigned char [:, :, :] rgb_,
        const unsigned char [:, :] alpha_,
        tmp_array_ = None
):
    """
    
    Create a 32-bit image/surface from RGB and Alpha arrays.

    This function generates a 32-bit image surface (with alpha transparency) by combining:
      - An RGB array with shape (w, h, 3) and dtype uint8.
      - An Alpha array with shape (w, h) and dtype uint8.

    Both input arrays must have the same width and height (w, h), and must be of type uint8.
    Optionally, a pre-allocated temporary array can be passed to optimize performance.
    
    Example Usage:
        im = pygame.image.load("../Assets/alpha.png").convert_alpha()
        rgb = pygame.surfarray.pixels3d(im)          # Extract the RGB pixel values.
        alpha = pygame.surfarray.pixels_alpha(im)    # Extract the Alpha (transparency) values.
        
        # Create the surface. Use convert_alpha() for optimal performance:
        image = create_rgba_surface(rgb, alpha).convert_alpha()
    
    Parameters:
    ----------
    :param rgb_:
        numpy.ndarray (or memoryviewslice):
        A 3D array with shape (w, h, 3) containing RGB pixel values (dtype=uint8).

    :param alpha_:
        numpy.ndarray (or memoryviewslice):
        A 2D array with shape (w, h) containing alpha (transparency) values (dtype=uint8).

    :param tmp_array_:
        Optional numpy.ndarray (or memoryviewslice):
        A 3D array with shape (w, h, 4) used as a temporary array for creating the RGBA surface.
        Passing this pre-allocated array can speed up the process by avoiding allocation overhead.

    Returns:
    -------
    :return:
        A 32-bit pygame surface with RGBA values.
        Use `convert_alpha()` to enhance performance when rendering the surface.
        Note: `convert_alpha()` requires the video display to be initialized.

    Notes:
    -----
    - The input arrays must have compatible shapes and types.
    - For optimal performance:
        - Use pygame's `convert_alpha()` method.
        - Ensure the video display is initialized before calling `convert_alpha()`.
    
     Collaboration:
    --------------
    This implementation includes improvements and comments by ChatGPT.
    """

    # Note:
    # When cythonizing, Cython automatically checks variable type limits,
    # as well as array shapes and types, ensuring compliance with defined constraints.

    # 1. Validate the `rgb_` input
    if not isinstance(rgb_, numpy.ndarray) and rgb_.__class__.__name__ != '_memoryviewslice':
        raise TypeError(
            f"`rgb_` must be a numpy.ndarray or memoryviewslice. Got {type(rgb_)} instead."
        )
    if <object> rgb_.ndim != 3 or <object> rgb_.shape[ 2 ] != 3:
        raise ValueError(
            f"`rgb_` must be a 3D array with shape (w, h, 3). Got shape {<object> rgb_.shape} instead."
        )
    if not is_type_memoryview(rgb_) and rgb_.dtype != numpy.uint8:
        raise TypeError("`rgb_` must have a dtype of uint8.")

    # 2. Validate the `alpha_` input
    if not isinstance(alpha_, numpy.ndarray) and alpha_.__class__.__name__ != '_memoryviewslice':
        raise TypeError(
            f"`alpha_` must be a numpy.ndarray or memoryviewslice. Got {type(alpha_)} instead."
        )
    if <object> alpha_.ndim != 2:
        raise ValueError(
            f"`alpha_` must be a 2D array with shape (w, h). Got shape {<object> alpha_.shape} instead."
        )
    if not is_type_memoryview(alpha_) and alpha_.dtype != numpy.uint8:
        raise TypeError("`alpha_` must have a dtype of uint8.")

    # 3. Check that the dimensions of `rgb_` and `alpha_` match
    rgb_dims = (<object> rgb_).shape[ :2 ]
    alpha_dims = (<object> alpha_).shape
    if rgb_dims != alpha_dims:
        raise ValueError(
            f"`rgb_` dimensions {rgb_dims} must match `alpha_` dimensions {alpha_dims}."
        )

    # 4. Validate the `tmp_array_` input (if provided)
    if tmp_array_ is not None:
        if not isinstance(tmp_array_, numpy.ndarray) and tmp_array_.__class__.__name__ != '_memoryviewslice':
            raise TypeError(
                f"`tmp_array_` must be a numpy.ndarray or memoryviewslice. Got {type(tmp_array_)} instead."
            )
        if <object> tmp_array_.ndim != 3 or <object> tmp_array_.shape[ 2 ] != 4:
            raise ValueError(
                f"`tmp_array_` must be a 3D array with shape (w, h, 4). Got shape {<object> tmp_array_.shape} instead."
            )
        if not is_type_memoryview(tmp_array_) and tmp_array_.dtype != numpy.uint8:
            raise TypeError("`tmp_array_` must have a dtype of uint8.")
        if <object> tmp_array_.shape[ :2 ] != rgb_dims:
            raise ValueError(
                f"The dimensions of `tmp_array_` ({<object> tmp_array_.shape[ :2 ]}) "
                f"must match the dimensions of `rgb_` ({rgb_dims})."
            )


    # Call the Cython C implementation (`create_rgba_surface`) to handle the heavy lifting.
    return create_rgba_surface_c(rgb_, alpha_, tmp_array_)


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef object create_rgba_surface_c(
        const unsigned char [:, :, :] rgb_,
        const unsigned char [:, :] alpha_,
        tmp_array_ = None
):
    """
    Create a 32-bit image/surface from RGB and Alpha arrays.

    This function generates a 32-bit image surface (with alpha transparency) by combining:
      - An RGB array with shape (w, h, 3) and dtype uint8.
      - An Alpha array with shape (w, h) and dtype uint8.

    Both input arrays must have the same width and height (w, h), and must be of type uint8.
    Optionally, a pre-allocated temporary array can be passed to optimize performance.
    
    Example Usage:
        im = pygame.image.load("../Assets/alpha.png").convert_alpha()
        rgb = pygame.surfarray.pixels3d(im)          # Extract the RGB pixel values.
        alpha = pygame.surfarray.pixels_alpha(im)    # Extract the Alpha (transparency) values.
        
        # Create the surface. Use convert_alpha() for optimal performance:
        image = create_rgba_surface_c(rgb, alpha).convert_alpha()
    
    Parameters:
    ----------
    :param rgb_:
        numpy.ndarray (or memoryviewslice):
        A 3D array with shape (w, h, 3) containing RGB pixel values (dtype=uint8).

    :param alpha_:
        numpy.ndarray (or memoryviewslice):
        A 2D array with shape (w, h) containing alpha (transparency) values (dtype=uint8).

    :param tmp_array_:
        Optional numpy.ndarray (or memoryviewslice):
        A 3D array with shape (w, h, 4) used as a temporary array for creating the RGBA surface.
        Passing this pre-allocated array can speed up the process by avoiding allocation overhead.

    Returns:
    -------
    :return:
        A 32-bit pygame surface with RGBA values.
        Use `convert_alpha()` to enhance performance when rendering the surface.
        Note: `convert_alpha()` requires the video display to be initialized.

    Notes:
    -----
    - The input arrays must have compatible shapes and types.
    - For optimal performance:
        - Use pygame's `convert_alpha()` method.
        - Ensure the video display is initialized before calling `convert_alpha()`.
    
    Collaboration:
    --------------
    This implementation includes improvements and comments by ChatGPT.    
    """
    # Note:
    # When cythonizing, Cython automatically checks variable type limits,
    # as well as array shapes and types, ensuring compliance with defined constraints.

    # Declare variables to hold dimensions of the input array
    cdef Py_ssize_t w, h, dim

    # Try to extract the shape of the `rgb_` array
    # This extracts the width (w), height (h), and the third dimension (dim), which should be 3 (for RGB channels)
    try:
        w, h, dim = (<object> rgb_).shape[:3]
    except (ValueError, pygame.error) as e:
        # Raise an error if the shape of the input array is not compatible (e.g., not 3D or missing channels)
        raise ValueError('\nArray shape not compatible.')

    # Declare additional variables for the nested loops
    cdef:
        int i, j  # Loop indices
        unsigned char[:, :, ::1] rgba_array  # Memoryview for the RGBA array

    # Try to create or reuse a temporary RGBA array
    try:
        # If `tmp_array_` is None, create a new array with the shape (h, w, 4) to store RGBA values
        # Otherwise, use the provided `tmp_array_` for efficiency
        rgba_array = empty((h, w, 4), dtype=numpy.uint8) if tmp_array_ is None else tmp_array_
    except Exception as e:
        # Raise an error if the array cannot be created (e.g., memory allocation issues)
        raise ValueError("Cannot create tmp array shape (h, w, 4).\n %s " % e)

    # Use a `nogil` block for parallelized execution of the pixel manipulation
    # This block ensures that the code runs efficiently without Python's Global Interpreter Lock (GIL)
    with nogil:
        # Outer loop: Iterate over the height of the image
        for j in prange(h, schedule=SCHEDULE, num_threads=THREADS):
            # Inner loop: Iterate over the width of the image
            for i in range(w):
                # Assign RGB values from the input `rgb_` array to the RGBA array
                rgba_array[j, i, 0] = rgb_[i, j, 0]  # Red channel
                rgba_array[j, i, 1] = rgb_[i, j, 1]  # Green channel
                rgba_array[j, i, 2] = rgb_[i, j, 2]  # Blue channel
                rgba_array[j, i, 3] = alpha_[i, j]   # Alpha channel (transparency)

    # Convert the RGBA array to a pygame surface using `frombuffer`
    # The `convert_alpha()` method requires that the video display is initialized in pygame
    # The resulting surface has the dimensions (w, h) and uses the 'RGBA' pixel format
    return frombuffer(rgba_array, (w, h), 'RGBA')




@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef alpha_array_to_surface(const unsigned char[:, :] array):
    """
    Convert a 2D alpha array (shape w x h, type uint8) into a 24-bit Pygame surface (RGB).

    This function takes a 2D alpha array and creates a new 24-bit surface (RGB). 
    Each pixel in the output surface has its R, G, and B channels set to the 
    corresponding alpha value from the input array.

    Example Usage:
        im = pygame.image.load("../Assets/alpha.png").convert_alpha()
        alpha = pygame.surfarray.pixels_alpha(im)  
        image = alpha_array_to_surface(alpha)

    :param array: 
        A 2D numpy array or memoryview with shape (w, h) and dtype uint8.

    :return: 
        A 24-bit Pygame.Surface (RGB).
        For better in-game performance, use `pygame.Surface.convert()` after creation.
    
    Collaboration:
    --------------
    This implementation includes improvements and comments by ChatGPT.     
    
    """

    # Note:
    # When cythonizing, Cython automatically checks variable type limits,
    # as well as array shapes and types, ensuring compliance with defined constraints.

    # Declare variables for array dimensions and loop counters
    cdef:
        int i, j  # Loop counters for width and height
        Py_ssize_t w = array.shape[0]  # Width of the array
        Py_ssize_t h = array.shape[1]  # Height of the array

        # Create a 3D array to store RGB values, initialized as empty
        # Shape is (h, w, 3), and dtype is uint8
        unsigned char[:, :, :] new_array = numpy.empty((h, w, 3), dtype=numpy.uint8)

    # Perform pixel-wise conversion in a multithreaded loop
    with nogil:  # Release the Global Interpreter Lock for better performance
        # Use prange for parallelized looping across rows
        for j in prange(h, schedule=SCHEDULE, num_threads=THREADS):
            for i in range(w):
                # Set R, G, and B channels to the same alpha value
                new_array[j, i, 0] = array[i, j]  # Red channel
                new_array[j, i, 1] = array[i, j]  # Green channel
                new_array[j, i, 2] = array[i, j]  # Blue channel

    # Convert the 3D RGB array into a Pygame surface using frombuffer
    return frombuffer(new_array, (w, h), 'RGB')





@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef inline unsigned int index3d_to_1d(
        const unsigned int x,
        const unsigned int y,
        const unsigned int z,
        const unsigned int width,
        const unsigned short int bytesize
)nogil:

    """
    Index mapping (3d array indexing --> buffer)
    
    This function calculates the equivalent 1D buffer index for a given 3D array index (x, y, z).
    It is useful when working with C-style contiguous memory buffers and converting 3D array indexing 
    into a single 1D index.
    
     Example:
        For a 3D array of shape (w, h, depth), the buffer index for array[x, y, z] is computed as:
        index = y * width * bytesize + x * bytesize + z

    This is especially helpful for transferring pixel data between 3D arrays (e.g., RGB or RGBA) 
    and 1D buffers in C-based libraries.

    Parameters:
    ----------
    x : unsigned int
        The x-coordinate (row index) in the 3D array.

    y : unsigned int
        The y-coordinate (column index) in the 3D array.

    z : unsigned int
        The z-coordinate (depth index) in the 3D array (e.g., R, G, or B channel).

    width : unsigned int
        The width of the 3D array (number of columns per row).

    bytesize : unsigned short int
        The depth of the 3D array, representing the number of bytes per pixel (e.g., 3 for RGB, 4 for RGBA).

    Returns:
    -------
    unsigned int
        The computed 1D buffer index corresponding to the given 3D array index (x, y, z).

    Example:
    --------
    Suppose we have a 3D array `array` of shape (width=4, height=3, depth=3):
    To access the equivalent buffer index of `array[1, 2, 0]`:
        index = index3d_to_1d(1, 2, 0, 4, 3)

    This can then be used to index into a 1D buffer, e.g., `buffer[index]`.
    
    Collaboration:
    --------------
    This implementation includes improvements and comments by ChatGPT. 
    """

    # Note:
    # When cythonizing, Cython automatically checks variable type limits,
    # as well as array shapes and types, ensuring compliance with defined constraints.

    # Compute the 1D buffer index based on the formula:
    # index = y * (width * bytesize) + x * bytesize + z
    # - y * width * bytesize: computes the starting offset for the row
    # - x * bytesize: computes the starting offset for the column within the row
    # - z: selects the specific channel (R, G, B, or A)
    return <unsigned int>(y * width * bytesize + x * bytesize + z)



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef inline (int, int, int) index1d_to_3d(
        const unsigned int index,
        const unsigned int width,
        const unsigned short int bytesize
)nogil:
    """
    Maps a 1D C-Buffer index to 3D array indices (x, y, z).

    This function converts a 1D index from a C-Buffer (flattened representation) 
    into its corresponding 3D array indices (x, y, z) for an array of shape 
    (width, height, bytesize). It facilitates efficient data exchange between 
    a C-Buffer and an equivalent 3D array.

    **Use Case:**
    This function is useful when working with data in both 1D (e.g., for performance reasons) 
    and 3D (e.g., for image processing). For example, it allows mapping a pixel index in 
    a C-Buffer to its equivalent position in a 3D array.

    **Example:**
    ```python
    # Loop through a C-Buffer of size (w * h * bytesize)
    for i in range(w * h * bytesize):
        x, y, z = index1d_to_3d(i, w, bytesize)
        bgr_array[x, y, z] = buffer[i]
    ```

    **Parameters:**
    - `index` (int): The 1D index in the C-Buffer to be converted.
    - `width` (int): The width of the referenced 3D array or image.
    - `bytesize` (int): The depth of the referenced 3D array or image. Typically:
        - `3` for 24-bit images (RGB).
        - `4` for 32-bit images (RGBA).

    **Returns:**
    - `(int, int, int)`: A tuple `(x, y, z)` representing the corresponding 3D array indices:
        - `x`: The column index (horizontal position).
        - `y`: The row index (position).
        - `z`: The depth index (e.g., color channel).

    **Notes:**
    - The provided `index` must be valid within the bounds of the C-Buffer and 3D array.
    - Ensure the width and bytesize match the dimensions of the 3D array and C-Buffer.

    **Performance:**
    - This function avoids using the `%` operator by substituting equivalent arithmetic 
      expressions, which may provide slight performance benefits in specific scenarios.

    **Acknowledgment:**
    - Collaborative improvements and optimizations provided with assistance from OpenAI's ChatGPT.
    """

    # Declare the 3D indices (x, y, z) as integers
    cdef:
        int x, y, z
        unsigned int ix = index // bytesize  # Compute the base 2D index (ignoring depth)

    # Compute the y-coordinate (row index)
    y = <int>(ix // width)

    # Compute the x-coordinate (column index)
    x = <int>(ix - (y * width))

    # Compute the z-coordinate (depth index)
    z = <int>(index - (ix * bytesize))

    # Return the computed 3D indices as a tuple
    return x, y, z


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef inline unsigned int vmap_buffer_c(
        const unsigned int index,
        const unsigned int width,
        const unsigned int height,
        const unsigned short int depth
)nogil:
    """
    
    :param index: 
 
    :param width: 
   
    :param height: 
  
    :param depth: 
  
    :return: 
  
    """

    cdef:
        unsigned int ix
        unsigned int x, y, z

    ix = index // depth
    y = <unsigned int> (ix / width)
    x = ix % width
    z = index % depth
    return <unsigned int> (x * height * depth) + (depth * y) + z



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef unsigned char [::1] bufferize(const unsigned char[:, :, :] rgb_array):

    """
    Create a C-buffer from a 3D numpy array.
    
    This function takes a 3D numpy array representing an RGB or RGBA image and converts it into a 
    1D C-style buffer. The numpy array should have a shape of (width, height, 3|4) with uint8 type.
    
  
    Parameters
    ----------
    rgb_array : 
        numpy.ndarray.
        A numpy array of shape (w, h, 3|4), representing an RGB or RGBA image.
        The last dimension (3 or 4) corresponds to the color channels (RGB or RGBA).
    
    Returns
    -------
        unsigned char [::1]
        A memoryview slice (C buffer) containing the RGB or RGBA pixel values. 
        This is a 1D array that can be accessed directly in C.

    ### ChatGPT Collaboration:
    To write comments 
    """

    # Declare Cython variables for width (w), height (h), and bit_size (3 or 4)
    cdef:
        Py_ssize_t w, h, bit_size
        unsigned int i, j
        unsigned int index
        unsigned int jw3
        unsigned char * p1

    # Get the shape of the input array
    w, h, bit_size = rgb_array.shape[:3]

    # Declare a pointer for the C buffer (temporary memory allocation)
    cdef:
        # unsigned char * tmp_v = <unsigned char *> malloc(w * h * bit_size * sizeof(unsigned char))
        # unsigned char [::1] c_buffer = <unsigned char[:w * h * bit_size]>tmp_v
        unsigned char [::1] c_buffer = numpy.empty(w * h * bit_size, dtype='uint8')

    # The block within 'with nogil' allows multi-threading (without holding the GIL)
    # to speed up the iteration over large images.
    with nogil:
        # Loop over the height of the image using parallel processing
        # prange is used to parallelize the outer loop
        for j in prange(h, schedule=SCHEDULE, num_threads=THREADS):
            # Compute the starting index for the current row (j)
            jw3 = j * w * bit_size
            # Iterate over each pixel in the row
            for i in range(w):
                # Compute the linear index for the 1D C buffer
                index = jw3 + i * bit_size

                p1 = &c_buffer[index]
                # Assign the RGB values from the numpy array to the C buffer
                p1[0]     = rgb_array[i, j, 0]  # Red channel
                (p1 + 1)[0] = rgb_array[i, j, 1]  # Green channel
                (p1 + 2)[0] = rgb_array[i, j, 2]  # Blue channel

                # If the image has an alpha channel (RGBA), copy the alpha value as well
                if bit_size == 4:
                    (p1 + 3)[0] = rgb_array[i, j, 3]  # Alpha channel

    # Free the memory manually if you used malloc
    # free(tmp_v)
    # Return the C buffer as a memoryview, which is a 1D array view of the allocated buffer
    return c_buffer


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef unsigned char [:, :, :] unbufferize(
        const unsigned char[:] c_buffer,
        const Py_ssize_t width,
        const Py_ssize_t height,
        const Py_ssize_t bit_size
):
    """
    Convert a 1D array (buffer) of uint8 data type into a 3D numpy array (copy).

    This function takes a 1D buffer of image data (such as RGB or RGBA values) and reconstructs it into
    a 3D array (numpy array) of shape (width, height, bit_size), where bit_size can be 3 (RGB) or 4 (RGBA).
    
    ### Example:
    ```python
    c_buffer = <some_1d_buffer>
    width = 256
    height = 256
    bit_size = 3  # RGB
    result = Unbufferizing(c_buffer, width, height, bit_size)
    ```

    ### Raises:
    - `ValueError`: If `bit_size == 0`, as it's an invalid value for image data.
    
    ### ChatGPT Collaboration:
    - This code was analyzed and improved with the assistance of **ChatGPT**. 
    The original code logic was maintained, but optimizations such as replacing 
    the modulo operation with integer division were proposed to enhance performance.
    
    :param c_buffer: 
        The input 1D buffer array (as a memoryview) containing pixel values in uint8 format.
    :param width: 
        The width of the resulting 3D array (image width in pixels).
    :param height: 
        The height of the resulting 3D array (image height in pixels).
    :param bit_size: 
        The number of channels in the image. Typically 3 (RGB) or 4 (RGBA).
    :return: 
        A memoryview slice (3D array) of shape (width, height, bit_size), 
        containing the RGB or RGBA pixel values.
    """

    # Ensure bit_size is not 0 (this would be an invalid value for image data).
    if bit_size == 0:
        raise ValueError("\nArgument bit_size cannot be null.")

    # Declare local variables for processing the buffer.
    cdef:
        int length, i  # length: total number of elements in the buffer, i: loop counter.
        unsigned int x, y  # x: column index, y: row index of the 3D array.
        unsigned int wb = width * bit_size  # wb: width * bit_size (total elements in one row).
        float one_bit_size = <float>1.0 / <float>bit_size  # one_bit_size: a factor used to scale the index by bit_size.

    # Get the length of the input buffer (total number of elements in c_buffer).
    length = c_buffer.shape[0]

    # Allocate memory for the 3D array (the output), where the shape is (width, height, bit_size).
    # numpy.empty is used for fast memory allocation without initializing the values.
    cdef:
        unsigned char [:, :, :] array_3d = numpy.empty((width, height, bit_size), dtype=uint8)

    # The block within 'with nogil' allows parallel execution (without holding the GIL).
    # This is important for performance when processing large images.
    with nogil:
        # Loop over the 1D buffer (c_buffer) in steps of bit_size (3 for RGB, 4 for RGBA).
        # prange is used to parallelize the iteration across multiple threads for performance.
        for i in prange(0, length, bit_size, schedule=SCHEDULE, num_threads=THREADS):

            # Compute the y (row) and x (column) indices of the 3D array from the 1D buffer index.
            # 'y' is the row index in the output 3D array.
            # 'x' is the column index in the output 3D array, which is scaled by the bit_size.
            y = <int> (i / wb)  # Integer division to get the row index.
            # x = <int>((i % wb) * one_bit_size)  # Modulo operation to get the column index, scaled by bit_size.
            x = <int> ((i - (i // wb) * wb) * one_bit_size)  # Calculate column index without modulo operation.

            # Assign the values from the buffer (c_buffer) to the corresponding positions in the 3D array.
            # We handle the RGB or RGBA channels based on the bit_size.
            array_3d[ x, y, 0 ] = c_buffer[ i    ]  # Red channel.
            array_3d[ x, y, 1 ] = c_buffer[ i + 1]  # Green channel.
            array_3d[ x, y, 2 ] = c_buffer[ i + 2]  # Blue channel.

            # If the image is RGBA (bit_size == 4), also assign the alpha channel.
            if bit_size == 4:
                array_3d[ x, y, 3 ] = c_buffer[ i + 3 ]  # Alpha channel.

    # Return the 3D array as a memoryview slice.
    return array_3d




#
# cdef inline void flip_bgr_buffer_c(
#     const Py_ssize_t width,
#     const Py_ssize_t height,
#     const unsigned char[::1] buffer,
#     unsigned char[::1] flipped_buffer
# ) nogil:
#     """
#     Perform vertical flipping of a BGR image buffer.
#
#     This function flips a BGR image **vertically** (not transposing) by rearranging
#     the pixel data from the input `buffer` into `flipped_buffer`.
#
#     Example Input/Output:
#     ---------------------
#     Input buffer (4x3 pixels):
#         buffer = [BGR1, BGR2, BGR3, BGR4, BGR5, BGR6, BGR7, BGR8, BGR9, BGR10, BGR11, BGR12]
#         Represented as:
#             [BGR1,  BGR2,  BGR3,  BGR4]
#             [BGR5,  BGR6,  BGR7,  BGR8]
#             [BGR9,  BGR10, BGR11, BGR12]
#
#     After flipping:
#         flipped_buffer = [BGR9, BGR10, BGR11, BGR12, BGR5, BGR6, BGR7, BGR8, BGR1, BGR2, BGR3, BGR4]
#         Represented as:
#             [BGR9,  BGR10, BGR11, BGR12]
#             [BGR5,  BGR6,  BGR7,  BGR8]
#             [BGR1,  BGR2,  BGR3,  BGR4]
#
#     Parameters
#     ----------
#     width : Py_ssize_t
#         The width of the image in pixels.
#
#     height : Py_ssize_t
#         The height of the image in pixels.
#
#     buffer : memoryview (unsigned char[::1])
#         The input 1D buffer containing BGR pixel data. The buffer size must be
#         `width * height * 3` to accommodate all pixels in the image.
#
#     flipped_buffer : memoryview (unsigned char[::1])
#         The output 1D buffer to store the vertically flipped BGR data. The size
#         must also be `width * height * 3`.
#
#     Notes
#     -----
#     - This function performs a **vertical flip** (not a transpose).
#     - The function is **parallelized** using OpenMP for performance.
#     - Ensure `flipped_buffer` is allocated properly before calling this function.
#     """
#
#     cdef:
#         Py_ssize_t row, src_offset, dst_offset
#         const Py_ssize_t row_size = width * 3  # Number of bytes per row
#
#     # Perform vertical flipping using OpenMP parallelization
#     for row in prange(height, schedule="static", nogil=True):
#         src_offset = row * row_size
#         dst_offset = (height - 1 - row) * row_size
#
#         # Use memcpy for fast row copying
#         memcpy(&flipped_buffer[dst_offset], &buffer[src_offset], row_size)



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef np.ndarray[np.uint8_t, ndim=1] bgr_buffer_transpose(
        const Py_ssize_t width,
        const Py_ssize_t height,
        const unsigned char [::1] buffer,
        transposed_buffer = None):

    """
    Transpose rows & columns of a BGR image buffer.

    This function transposes a BGR image by rearranging the pixel data from the 
    input `buffer` into the `transposed_buffer`. It assumes the image is represented as 
    a 1D array of BGR pixels in row-major order.

    This is equivalent to performing a transpose operation on the pixel data: 
    transposing rows and columns of BGR blocks.
    
    Example Input/Output:
    ---------------------
    BGR image's pixels represented as:
            [BGR1, BGR2,  BGR3,  BGR4]
            [BGR5, BGR6,  BGR7,  BGR8]
            [BGR9, BGR10, BGR11, BGR12]
    
    After transpose:
    
    output image's pixels Represented as:
            [BGR1, BGR5, BGR9]
            [BGR2, BGR6, BGR10]
            [BGR3, BGR7, BGR11]
            [BGR4, BGR8, BGR12]
    
    Code example:
    Load the image 
    source = pygame.image.load('../Assets/px.png').convert(24)  
    
    # ----- Transposing using bgr_buffer_transpose
    source = pygame.transform.smoothscale(source, (800, 600))
    arr = numpy.empty(800*600*3, dtype=numpy.uint8) 
    arr = bgr_buffer_transpose(800, 600, source.get_buffer(), arr) 
    
    # ----- transposing using numpy
    buff2 = source.get_buffer()
    buff2 = numpy.frombuffer(buff2, dtype=numpy.uint8).reshape(800, 600, 3)
    buff2 = buff2.transpose(1, 0, 2)
    buff2 = buff2.flatten()
    
    Parameters
    ----------
    width : Py_ssize_t
        The width of the video frame. Must be greater than 0.
        
    height : Py_ssize_t
        The height of the video frame. Must be greater than 0.
        
    buffer : memoryview (unsigned char[::1])
        A 1D memoryview containing the pixel data (typically in BGR format).
        
    transposed_buffer : memoryview (optional)
        An optional 1D memoryview to store the transposed pixel data. If not provided, 
        a new buffer will be created.
    
    Returns
    -------
    numpy.ndarray[np.uint8_t, ndim=1]
        A NumPy array containing the transposed pixel data.
    
    Raises
    ------
    ValueError
        - If `width` or `height` is less than or equal to 0.
        - If `buffer` is None or its size does not match `width * height * 3`.
        - If `transposed_buffer` is provided but its size does not match `width * height * 3`.
    TypeError
        - If `buffer` or `transposed_buffer` is not a memoryview or NumPy array.
        - If `buffer` is numpy.array and not contiguous
    
    Notes
    -----
    - This function assumes the buffer data is in BGR format and performs transpose.
    - The operation is done in-place if a `transposed_buffer` is provided.
    """

    # Validate `width` and `height`
    # Ensure that `width` is greater than 0, as a non-positive width would be invalid.
    if width <= 0:
        raise ValueError(f"Argument 'width' must be greater than 0. Received: {width}")

    # Ensure that `height` is greater than 0, as a non-positive height would be invalid.
    if height <= 0:
        raise ValueError(f"Argument 'height' must be greater than 0. Received: {height}")

    # Validate `buffer`
    # Check if `buffer` is either a NumPy array or a memoryview slice. If not, raise a TypeError.
    if not (isinstance(buffer, numpy.ndarray) or buffer.__class__.__name__ == '_memoryviewslice'):
        raise TypeError("Argument 'buffer' must be a memoryview or a NumPy array.")

    if isinstance(buffer, numpy.ndarray):
        if not buffer.flags['C_CONTIGUOUS']:
            raise TypeError("Argument 'buffer' must be contiguous.")

    # Ensure that `buffer` is a 1D array. If not, raise a ValueError.
    if (<object> buffer).ndim != 1:
        raise ValueError(f"Argument 'buffer' must be a 1D array. Received: {(<object> buffer.ndim)} dimensions.")

    # Check if the size of `buffer` matches the expected size, which is `width * height * 3`.
    # If not, raise a ValueError indicating the mismatch.
    if (<object> buffer).shape[ 0 ] != width * height * 3:
        raise ValueError(
            f"Argument 'buffer' size must match width * height * 3. "
            f"Expected: {width * height * 3}, Received: {(<object> buffer).shape[ 0 ]}"
        )

    cdef unsigned char [::1]transposed_buf
    # Validate `transposed_buffer`
    # If `transposed_buffer` is provided, ensure it meets the requirements.
    if transposed_buffer is not None:
        # Check if `transposed_buffer` is either a NumPy array or a memoryview slice. If not, raise a TypeError.
        if not (isinstance(transposed_buffer, numpy.ndarray)
                or transposed_buffer.__class__.__name__ == '_memoryviewslice'):
            raise TypeError("Argument 'transposed_buffer' must be a memoryview or a NumPy array.")

        # Ensure that `transposed_buffer` is a 1D array. If not, raise a ValueError.
        if (<object> transposed_buffer).ndim != 1:
            raise ValueError(
                f"Argument 'transposed_buffer' must be a 1D array. Received: "
                f"{(<object> transposed_buffer).ndim} dimensions.")

        # Check if the size of `transposed_buffer` matches the expected size, which is `width * height * 3`.
        # If not, raise a ValueError indicating the mismatch.
        if (<object> transposed_buffer).shape[ 0 ] != width * height * 3:
            raise ValueError(
                f"Argument 'transposed_buffer' size must match width * height * 3. "
                f"Expected: {width * height * 3}, Received: {(<object> transposed_buffer).shape[ 0 ]}"
            )
        transposed_buf = transposed_buffer
    else:
        transposed_buf = numpy.empty(width * height * 3, dtype=numpy.uint8)


    # Call the C function to perform the BGR transpose
    # This function processes the provided `buffer` and writes the transposed BGR data to `transposed_buf`.
    bgr_buffer_transpose_c(width, height, buffer, transposed_buf)

    # Convert the `transposed_buf` memoryview into a NumPy array and return it.
    # This ensures compatibility with other Python code that uses NumPy arrays.
    return numpy.asarray(transposed_buf)



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef inline void bgr_buffer_transpose_c(
        const Py_ssize_t width,
        const Py_ssize_t height,
        const unsigned char [::1] buffer,
        unsigned char [::1] transposed_buffer
       )nogil:
    """
    Transposes a BGR image buffer from its original layout to a new layout.

    This function takes a buffer representing a BGR image and transposes it into a new buffer. The function works on 
    image data stored in a linear memory format (1D array), where each pixel is represented by 3 consecutive bytes 
    for Blue, Green, and Red channels. The transposition operation swaps the pixel data into a new format while 
    maintaining the integrity of the individual pixel values.

    The function operates in parallel, processing chunks of the buffer with multiple threads for improved performance.

    Note:
        - The input `buffer` and output `transposed_buffer` must be valid pointers to allocated memory.
        - This function is optimized for performance in Cython, working directly with the memory layout of the array.
        - It assumes the image is in BGR format, with each pixel consisting of 3 consecutive bytes.

    Parameters:
    -----------
    width : Py_ssize_t
        The width (number of columns) of the image in pixels.

    height : Py_ssize_t
        The height (number of rows) of the image in pixels.

    buffer : unsigned char [::1]
        A 1D array (buffer) of type `unsigned char`, representing the image in BGR format. 
        The size of this array should be `width * height * 3`.

    transposed_buffer : unsigned char [::1]
        A 1D array where the transposed BGR image will be written. 
        This buffer must have the same size as the input buffer (`width * height * 3`).

    Returns:
    --------
    None
        The function modifies the `transposed_buffer` in place, writing the transposed BGR image to it.

    Raises:
    -------
    ValueError
        If either the `buffer` or `transposed_buffer` is `None`, a `ValueError` will be raised.

    Example:
    --------
    # Example usage:
    cdef unsigned char[:] buffer = np.zeros((width * height * 3,), dtype=np.uint8)
    cdef unsigned char[:] transposed_buffer = np.zeros_like(buffer)
    bgr_buffer_transpose_c(width, height, buffer, transposed_buffer)
    """

    if buffer is None:
        raise ValueError(f"buffer cannot be None. Received: {buffer} (<object>NoneType)")

    if transposed_buffer is None:
        raise ValueError(f"transposed_buffer cannot be None. Received: {buffer} (<object>NoneType)")

    cdef:
        int i, j, index
        unsigned short bytesize = 3
        int h3 = height * bytesize
        unsigned int iw
        int tmp
        const unsigned char * p1
        unsigned char *p2


    for i in prange(0, h3, bytesize, schedule = METHOD, num_threads = THREADS):

        iw = i * width
        for j in range(0, width):

            index = i + (h3 * j)


            tmp = (j * 3) + iw

            p1 = &buffer[ index ]
            p2 = &transposed_buffer[ tmp ]

            p2[ 0 ] = p1[ 0 ]  # Red
            (p2 + 1)[ 0 ] = (p1 + 1)[ 0 ]  # Green
            (p2 + 2)[ 0 ] = (p1 + 2)[ 0 ]  # Blue
            # memcpy(&transposed_buffer[ tmp ], &buffer[ index ], 3)


# @cython.binding(False)
# @cython.boundscheck(False)
# @cython.wraparound(False)
# @cython.nonecheck(False)
# @cython.cdivision(True)
# @cython.profile(False)
# @cython.initializedcheck(False)
# @cython.exceptval(check=False)
# cdef inline void bgr_buffer_transpose_c(
#         const Py_ssize_t width,
#         const Py_ssize_t height,
#         const unsigned char [::1] buffer,
#         unsigned char [::1] transposed_buffer
#        ) nogil:
#     """
#     Performs an in-memory transpose of a BGR image buffer.
#
#     Transforms an image stored in row-major order from (height, width, 3)
#     to (width, height, 3), effectively swapping the width and height dimensions.
#     """
#
#     cdef:
#         int i, j, index
#         int bytesize = 3
#         unsigned int row_offset, col_offset
#         const unsigned char *p1
#         unsigned char *p2
#
#     # Iterate over height and width
#     for i in prange(height, schedule="static"):  # OpenMP parallelization
#         row_offset = i * width * bytesize
#         for j in range(width):
#             col_offset = j * height * bytesize
#
#             index = row_offset + (j * bytesize)  # Original index
#             p1 = &buffer[index]
#
#             index = col_offset + (i * bytesize)  # Transposed index
#             p2 = &transposed_buffer[index]
#
#             # Copy BGR data
#             p2[0] = p1[0]  # Blue
#             p2[1] = p1[1]  # Green
#             p2[2] = p1[2]  # Red


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef np.ndarray[np.uint8_t, ndim=1] flip_bgra_buffer(
        const Py_ssize_t width,
        const Py_ssize_t height,
        const unsigned char [::1] buffer,
        unsigned char [::1] flipped_buffer,
        ):
    """
    Perform flipping of a BGRA image buffer.

    This core function flips an BGRA image by rearranging the pixel data from the 
    input `buffer` into the `flipped_buffer`. It assumes the image is represented as 
    a 1D array of BGRA pixels in row-major order.

    This is equivalent to performing a transpose operation on the pixel data: 
    flipping rows and columns of BGRA blocks.

    Example Input/Output:
    ---------------------
    Input buffer:
        buffer = [BGRA1, BGRA2, BGRA3, BGRA4, BGRA5, BGRA6, BGRA7, BGRA8, BGRA9]
        Represented as:
            [BGRA1, BGRA2, BGRA3]
            [BGRA4, BGRA5, BGRA6]
            [BGRA7, BGRA8, BGRA9]
    
    After flipping:
        output buffer = [BGRA1, BGRA4, BGRA9, BGRA2, BGRA5, BGRA8, BGRA3, BGRA6, BGRA9]
        Represented as:
            [BGRA1, BGRA4, BGRA9]
            [BGRA2, BGRA5, BGRA8]
            [BGRA3, BGRA6, BGRA9]

    Parameters
    ----------
    width : Py_ssize_t
        The width of the image in pixels.

    height : Py_ssize_t
        The height of the image in pixels.

    buffer : memoryview (unsigned char[::1])
        The input 1D buffer containing BGRA pixel data. The buffer size must be 
        `width * height * 4` to accommodate all pixels in the image.

    flipped_buffer : memoryview (unsigned char[::1])
        The output 1D buffer to store the flipped BGRA data. The size 
        must also be `width * height * 4`.
        
    Returns
    -------
    numpy.ndarray[np.uint8_t, ndim=1]
        A 1D NumPy array containing the flipped BGRA pixel data, stored in row-major order.
        
    Raises
    ------
    ValueError
        - If `width` or `height` is less than or equal to 0.
        - If `buffer` is not 1D or its size does not match `width * height * 4`.
        - If `flipped_buffer` is not 1D or its size does not match `width * height * 4`.
    TypeError
        - If `buffer` or `flipped_buffer` is not a memoryview or NumPy array.
    
    Notes
    -----
    - The flipping operation is performed in parallel using OpenMP's `prange` for better 
      performance on multi-threaded systems.
    - If `flipped_buffer` is not provided, a new writable buffer is allocated and returned.

    ChatGPT Collaboration:
    To write comments

    """

    # Validate `width` and `height`
    # Ensure that `width` is greater than 0, as a non-positive width would be invalid.
    if width <= 0:
        raise ValueError(f"Argument 'width' must be greater than 0. Received: {width}")

    # Ensure that `height` is greater than 0, as a non-positive height would be invalid.
    if height <= 0:
        raise ValueError(f"Argument 'height' must be greater than 0. Received: {height}")

    # Validate `buffer`
    # Check if `buffer` is either a NumPy array or a memoryview slice. If not, raise a TypeError.
    if not (isinstance(buffer, numpy.ndarray) or buffer.__class__.__name__ == '_memoryviewslice'):
        raise TypeError("Argument 'buffer' must be a memoryview or a NumPy array.")

    # Ensure that `buffer` is a 1D array. If not, raise a ValueError.
    if (<object> buffer).ndim != 1:
        raise ValueError(f"Argument 'buffer' must be a 1D array. Received: {(<object> buffer.ndim)} dimensions.")

    # Check if the size of `buffer` matches the expected size, which is `width * height * 4`.
    # If not, raise a ValueError indicating the mismatch.
    if (<object> buffer).shape[ 0 ] != width * height * 4:
        raise ValueError(
            f"Argument 'buffer' size must match width * height * 3. "
            f"Expected: {width * height * 4}, Received: {(<object> buffer).shape[ 0 ]}"
        )

    cdef unsigned char [::1] flip_buf

    # Validate `flipped_buffer`
    # If `flipped_buffer` is provided, ensure it meets the requirements.
    if flipped_buffer is not None:
        # Check if `flipped_buffer` is either a NumPy array or a memoryview slice. If not, raise a TypeError.
        if not (isinstance(flipped_buffer, numpy.ndarray) or
                flipped_buffer.__class__.__name__ == '_memoryviewslice'):
            raise TypeError("Argument 'flipped_buffer' must be a memoryview or a NumPy array.")

        # Ensure that `flipped_buffer` is a 1D array. If not, raise a ValueError.
        if (<object> flipped_buffer).ndim != 1:
            raise ValueError(
                f"Argument 'flipped_buffer' must be a 1D array. Received: "
                f"{(<object> flipped_buffer).ndim} dimensions.")

        # Check if the size of `flipped_buffer` matches the expected size, which is `width * height * 4`.
        # If not, raise a ValueError indicating the mismatch.
        if (<object> flipped_buffer).shape[ 0 ] != width * height * 4:
            raise ValueError(
                f"Argument 'flipped_buffer' size must match width * height * 3. "
                f"Expected: {width * height * 4}, Received: {(<object> flipped_buffer).shape[ 0 ]}"
            )
        flip_buf = flipped_buffer
    else:
        flip_buf = numpy.empty(width * height * 4, dtype=numpy.uint8)


    # Call the C function to perform the BGRA flipping
    # This function processes the provided `buffer` and writes the flipped BGRA data to `flip_buf`.
    flip_bgra_buffer_c(width, height, buffer, flip_buf)

    # Convert the `flip_buf` memoryview into a NumPy array and return it.
    # This ensures compatibility with other Python code that uses NumPy arrays.
    return numpy.asarray(flip_buf)






@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef inline void flip_bgra_buffer_c(
        const Py_ssize_t width,
        const Py_ssize_t height,
        const unsigned char [::1] buffer,
        unsigned char [::1] flipped_buffer)nogil:

    cdef:
        int i, j, index, v
        int h4 = 4 * height
        const unsigned char * tmp_r
        const unsigned char * tmp_g
        const unsigned char * tmp_b
        const unsigned char * tmp_a

    for i in prange(0, h4, 4, schedule=METHOD, num_threads=THREADS):
        for j in range(0, width):
            index = i + (h4 * j)
            v = (j * 4) + (i * width)
            tmp_r = &buffer[index    ]
            tmp_g = &buffer[index + 1]
            tmp_b = &buffer[index + 2]
            tmp_a = &buffer[index + 3]
            flipped_buffer[v    ] = tmp_r[0]
            flipped_buffer[v + 1] = tmp_g[0]
            flipped_buffer[v + 2] = tmp_b[0]
            flipped_buffer[v + 3] = tmp_a[0]






@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef unsigned char [:, :] binary_mask(unsigned char [:, :] mask_alpha_):

    """
    Convert a 2D memoryview or a numpy.array into a black-and-white mask alpha array.

    This function processes a 2D memoryview of type `uint8` (values ranging from 0 to 255) 
    and converts it into a binary black-and-white mask. Pixels with a value of `0` 
    are converted to `0` (black), while pixels with a value greater than `0` are 
    converted to `255` (white). The function returns a memoryview, not a NumPy array.

    Parameters
    ----------
    mask_alpha_ : memoryview (unsigned char[:, :])
        A 2D memoryview of shape (W, H) and type `uint8`. Each element represents 
        an alpha value ranging from 0 (fully transparent) to 255 (fully opaque).

    Returns
    -------
    memoryview (unsigned char[:, :])
        A 2D memoryview of shape (W, H) and type `uint8`, where each element is either:
        - `0` (black): Corresponds to pixels with an original value of `0`.
        - `255` (white): Corresponds to pixels with an original value greater than `0`.

    Raises
    ------
    TypeError
        If the input `mask_alpha_` is not a `numpy.ndarray` or a memoryview slice.

    ValueError
        If:
        - `mask_alpha_` is not of type `uint8`.
        - `mask_alpha_` does not have exactly 2 dimensions.
    
    Example
    -------

    # Load an image with alpha channel
    im = pygame.image.load("../Assets/alpha.png").convert_alpha()
    # Get the alpha channel as a NumPy array
    alpha = pygame.surfarray.pixels_alpha(im)
    # Convert the alpha channel to a black-and-white solid transparency mask
    solid_transparency = binary_mask(alpha)

     Notes
    -----
    - This function modifies the input memoryview `mask_alpha_` in place for performance reasons.
    - It operates on Cython memoryviews directly, which are more efficient than NumPy arrays 
      for this type of low-level operation.
    - The operation is performed in parallel using OpenMP's `prange` for improved performance.
    - Ensure that `mask_alpha_` is writable before calling this function.
    
    """

    # Check if the input is either a numpy.ndarray or a memoryview slice
    if not (isinstance(mask_alpha_, numpy.ndarray) or mask_alpha_.__class__.__name__ == '_memoryviewslice'):
        raise TypeError("Input mask_alpha_ must be a numpy.ndarray or memoryviewslice.")

    # If the input is not a memoryview, check its dtype
    if not is_type_memoryview(mask_alpha_):
        if (<object> mask_alpha_).dtype != numpy.uint8:
            raise ValueError(f"mask_alpha_ must be of type uint8, but got {(<object> mask_alpha_).dtype}.")

    # Ensure the input array has exactly two dimensions
    if (<object> mask_alpha_).ndim != 2:
        raise ValueError(f"mask_alpha_ must have shape (w, h). Received: {(<object> mask_alpha_).shape})")

    cdef:
        int i, j  # Loop counters for iterating through the array
        Py_ssize_t w = (<object> mask_alpha_).shape[ 0 ]  # Width of the array (number of rows)
        Py_ssize_t h = (<object> mask_alpha_).shape[ 1 ]  # Height of the array (number of columns)
        unsigned char * a  # Pointer to the current pixel value

    # Perform the conversion to a black-and-white mask in parallel
    with nogil:
        for j in prange(h, schedule = SCHEDULE, num_threads = THREADS):  # Iterate over columns (parallelized)
            for i in range(w):  # Iterate over rows
                a = &mask_alpha_[ i, j ]  # Get a pointer to the current pixel
                if a[ 0 ] > 0:  # If the pixel value is greater than 0
                    a[ 0 ] = 255  # Set it to 255 (white)
                else:  # Otherwise
                    a[ 0 ] = 0  # Set it to 0 (black)

    # Return the modified memoryview
    return mask_alpha_





@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef object mask32(
        unsigned char [:, :, :] rgb_,
        unsigned char [:, :] alpha_,
        unsigned char [:, :] mask_alpha_,
):
    """
    CREATE A 32-BIT MASK ALPHA 
    
    Apply a mask to an image split into its respective RGB & alpha values
    This method will create a new alpha channel from the original alpha transparency 
    and from the mask alpha.  
    
    Compatible only with array type uint8
    All arrays must have the same sizes (w, h)
    No need to use the pygame convert_alpha method, the new surface is already formatted 
    for fast blit.
    
    e.g
    >>>new_image = mask32(rgb, alpha, mask_alpha).convert_alpha()
    
    Parameters
    ----------
    rgb_ : 3d numpy.array shape (w, h, 3) type uint8  
    alpha_ : 2d numpy.array shape (w, h) type uint8 
    mask_alpha_ : 2d numpy.array shape (w, h) type uint8

    Returns
    -------
    32-bit PYGAME Surface (with a new alpha channel) 

    """

    cdef:
        int i, j
        Py_ssize_t w = rgb_.shape[0]
        Py_ssize_t h = rgb_.shape[1]
        unsigned char [:, :] new_mask = numpy.full((w, h), 0, dtype=numpy.uint8)
        unsigned char [:, :, :] new_array = numpy.full((w, h, 3), 0, dtype=numpy.uint8)

        unsigned char * s_a
        unsigned char * d_a
        unsigned char * new_a
        unsigned char * r
        unsigned char * g
        unsigned char * b



    with nogil:

        for j in prange(h, schedule=SCHEDULE, num_threads=THREADS):
            for i in range(w):

                s_a = &alpha_[i, j]
                d_a = &mask_alpha_[i, j]
                r = &rgb_[i, j, 0]
                g = &rgb_[i, j, 1]
                b = &rgb_[i, j, 2]

                # if both, mask alpha and image alpha values are zero (100% transparency) then
                # Red, green and blue pixels values will no longer be visible. In that
                # case the pixels values are irrelevant
                if d_a[0] == 0 and s_a[0] == 0:
                    pass
                else:
                    # mask alpha and image alpha values are not zero (partially transparent)
                    # if rgb values are not null
                    if r[0] and g[0] and b[0] != 0:
                        new_array[i, j, 0] = r[0]
                        new_array[i, j, 1] = g[0]
                        new_array[i, j, 2] = b[0]
                    # Transparency is set by defining a specific color as alpha layer,
                    # usually black pixels defined the intensity of the transparency,
                    # fully black pixel gives 100% of transparency while solid white pixel
                    # give 0% transparency.

                    # NULL image rgb values (are most likely to be pixels attached
                    # to the original transparent layer). In that case, the alpha
                    # value should be zero to hide the pixel).
                    else:
                        mask_alpha_[i, j] = 0

    return create_rgba_surface(new_array, mask_alpha_).convert_alpha()


# @cython.binding(False)
# @cython.boundscheck(False)
# @cython.wraparound(False)
# @cython.nonecheck(False)
# @cython.cdivision(True)
# @cython.profile(False)
# @cython.initializedcheck(False)
# cpdef object mask32(
#         object surface_,
#         unsigned char [:, :] mask_alpha_,
# ):
#     raise NotImplemented


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef object channels_to_surface(
        unsigned char[:, :] red_channel,       # Input array for the red channel (shape: (w, h), type: uint8)
        unsigned char[:, :] green_channel,     # Input array for the green channel (shape: (w, h), type: uint8)
        unsigned char[:, :] blue_channel,      # Input array for the blue channel (shape: (w, h), type: uint8)
        unsigned char[:, :] alpha_channel,     # Input array for the alpha channel (shape: (w, h), type: uint8)
        output_rgba_buffer=None                # Optional temporary array (empty, same shape as input arrays)
):
    """
    Create a 32-bit pygame surface from respective red, green, blue, and alpha channels.

    This function combines four separate 2D arrays (representing red, green, blue, and alpha channels) 
    into a single 32-bit RGBA pygame surface. The result is a surface suitable for rendering in Pygame, 
    with transparency supported.

    Use `pygame.convert_alpha()` after creating the surface for optimal performance and to ensure 
    proper alpha transparency handling. Note that `convert_alpha()` requires the pygame display to 
    be initialized beforehand.

    Compatible only with uint8 array types, and all input arrays must have the same shape.

    Example:
        import pygame
        im = pygame.image.load("../Assets/rgba_image.png").convert_alpha()
        red = pygame.surfarray.pixels_red(im)
        green = pygame.surfarray.pixels_green(im)
        blue = pygame.surfarray.pixels_blue(im)
        alpha = pygame.surfarray.pixels_alpha(im)

        surface = channels_to_surface(red, green, blue, alpha)
        surface = surface.convert_alpha()  # Optimize the surface for rendering

    Parameters:
    -----------
    red_channel : numpy.ndarray or memoryview
        A 2D array or memoryview of shape (w, h) and type uint8 containing pixel values for the red channel.
        Each value ranges from 0 to 255.

    green_channel : numpy.ndarray or memoryview
        A 2D array or memoryview of shape (w, h) and type uint8 containing pixel values for the green channel.
        Each value ranges from 0 to 255.

    blue_channel : numpy.ndarray or memoryview
        A 2D array or memoryview of shape (w, h) and type uint8 containing pixel values for the blue channel.
        Each value ranges from 0 to 255.

    alpha_channel : numpy.ndarray or memoryview
        A 2D array or memoryview of shape (w, h) and type uint8 containing pixel values for the alpha channel.
        Each value ranges from 0 to 255.

    output_rgba_buffer : numpy.ndarray, optional
        A temporary 3D array or memoryview of shape (h, w, 4) and type uint8 used to store intermediate RGBA data.
        If provided, it can speed up processing by reusing memory. If `None`, a new array will be allocated.

    Returns:
    --------
    pygame.Surface
        A 32-bit Pygame surface with RGBA pixel format, created from the input channel arrays or memoryviews.
        If alpha transparency is required, use `pygame.Surface.convert_alpha()` on the returned surface.

    Raises:
    -------
    ValueError
        If the input channel arrays or memoryviews do not have the 
        same shape or if `output_rgba_buffer` has an incorrect shape.
    TypeError
        If any of the input arrays or memoryviews is not a 2D NumPy array of type uint8.

    Notes:
    ------
    - This function supports both NumPy arrays and memoryviews as inputs, allowing for efficient processing of data.
    - This function is optimized for performance using Cython and can process large images efficiently.
    - To ensure the correct handling of transparency, call `convert_alpha()` on the resulting surface.
    """

    # Ensure the input arrays or memoryviews are valid and have the same shape
    return channels_to_surface_c(red_channel, green_channel, blue_channel, alpha_channel, output_rgba_buffer)



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef object channels_to_surface_c(
        unsigned char [:, :] red_channel,
        unsigned char [:, :] green_channel,
        unsigned char [:, :] blue_channel,
        unsigned char [:, :] alpha_channel,
        output_rgba_buffer = None
):
    """
    Converts separate RGBA channel arrays or memoryviews into a single Pygame 32-bit Surface.

    This function takes separate red (R), green (G), blue (B), and alpha (A) channel arrays 
    (or memoryviews) and combines them into a single RGBA surface. The resulting surface can then be used 
    in Pygame for rendering. If the alpha channel is required, make sure to call 
    `convert_alpha()` on the returned surface for proper transparency handling.

    Example:
        surface = channels_to_surface(
            pygame.surfarray.array_red(source),
            pygame.surfarray.array_green(source),
            pygame.surfarray.array_blue(source),
            pygame.surfarray.array_alpha(source),
            output_rgba_buffer = None
        ).convert_alpha()

    Parameters:
    -----------
    red_channel : numpy.ndarray or memoryview
        A 2D array or memoryview of shape (w, h) and type uint8 containing pixel values for the red channel.
        Each value ranges from 0 to 255.

    green_channel : numpy.ndarray or memoryview
        A 2D array or memoryview of shape (w, h) and type uint8 containing pixel values for the green channel.
        Each value ranges from 0 to 255.

    blue_channel : numpy.ndarray or memoryview
        A 2D array or memoryview of shape (w, h) and type uint8 containing pixel values for the blue channel.
        Each value ranges from 0 to 255.

    alpha_channel : numpy.ndarray or memoryview
        A 2D array or memoryview of shape (w, h) and type uint8 containing pixel values for the alpha channel.
        Each value ranges from 0 to 255.

    output_rgba_buffer : numpy.ndarray, optional
        A temporary 3D array or memoryview of shape (h, w, 4) and type uint8 used to store intermediate RGBA data.
        If provided, it can speed up processing by reusing memory. If `None`, a new array will be allocated.

    Returns:
    --------
    pygame.Surface
        A 32-bit Pygame surface with RGBA pixel format, created from the input channel arrays or memoryviews.
        If alpha transparency is required, use `convert_alpha()` on the returned surface.

    Raises:
    -------
    ValueError
        If the input channel arrays or memoryviews do not have the same shape or 
        if `output_rgba_buffer` has an incorrect shape.
    TypeError
        If any of the input arrays or memoryviews is not a 2D NumPy array of type uint8.

    Notes:
    ------
    - This function supports both NumPy arrays and memoryviews as inputs, allowing for efficient processing of data.
    - This function is optimized for performance using Cython and can process large images efficiently.
    - To ensure the correct handling of transparency, call `convert_alpha()` on the resulting surface.
    """

    # Validate input dimensions and types
    cdef int w, h
    try:
        w, h = (<object>red_channel).shape  # Get dimensions from the red channel
        if (<object>green_channel).shape != (w, h) or \
                (<object>blue_channel).shape != (w, h) or (<object>alpha_channel).shape != (w, h):
            raise ValueError("All input channels (R, G, B, A) must have the same dimensions.")
    except AttributeError:
        raise TypeError("All input arrays or memoryviews must be 2D NumPy arrays of type uint8.")

    if isinstance(output_rgba_buffer, numpy.ndarray):
        if not output_rgba_buffer.flags['C_CONTIGUOUS']:
            raise TypeError("Argument 'output_rgba_buffer' is not contiguous.")

    cdef:
        int i, j  # Loop variables

    # Ensure output_rgba_buffer is correctly sized or allocate a new one
    cdef unsigned char[:, :, ::1] rgba_array
    try:
        rgba_array = (
            empty((h, w, 4), dtype=numpy.uint8) if output_rgba_buffer is None else output_rgba_buffer
        )
        if (<object>rgba_array).shape != (h, w, 4):
            raise ValueError("output_rgba_buffer must have the shape (h, w, 4).")
    except Exception as e:
        raise ValueError(f"Failed to initialize the temporary RGBA array: {e}")

    # Begin processing the data without the Global Interpreter Lock (GIL) for parallel execution
    with nogil:
        # Iterate over the rows (height)
        for j in prange(h, schedule=SCHEDULE, num_threads=THREADS):
            # Iterate over the columns (width)
            for i in range(w):
                # Assign red, green, blue, and alpha values to the RGBA array
                rgba_array[j, i, 0] = red_channel[i, j]  # Red channel
                rgba_array[j, i, 1] = green_channel[i, j]  # Green channel
                rgba_array[j, i, 2] = blue_channel[i, j]  # Blue channel
                rgba_array[j, i, 3] = alpha_channel[i, j]  # Alpha channel

    # Return a Pygame Surface from the RGBA buffer
    # Note: Ensure that convert_alpha() is called if alpha transparency is needed
    return frombuffer(rgba_array, (w, h), 'RGBA')




@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef object compare_png24bit(surface1, surface2):

    """
    COMPARE 24-BIT PYGAME SURFACES AND 
     OUTPUT THE DIFFERENCE INTO A NEW SURFACE
    
    Incompatible with JPG images (JPEG is a lossy algorithm modifying pixel values)
    PNG is lossless and will preserves the pixel values.
    Surface1 & Surface2 must be same size and format (PNG)
    You can use PYGAME convert_alpha to convert the output image for fast blit
    
    :param surface1: pygame.Surface; Surface 1 (24-bit image) 
    :param surface2: pygame.Surface; Surface 2 (24--bit image) 
    :return : New 24-bit pygame surface (difference between surface1 and surface2) 
    
    """

    cdef:
        unsigned char [:, :, :] rgb_array1
        unsigned char [:, :, :] rgb_array2

    try:
        rgb_array1 = pixels3d(surface1) # .get_view('3')
    except (ValueError, pygame.error) as e:
        raise ValueError('\nImage/Surface not compatible.')

    try:
        rgb_array2 = pixels3d(surface2) # .get_view('3')
    except (ValueError, pygame.error) as e:
        raise ValueError('\nImage/Surface not compatible.')
    cdef int w, h
    w, h = surface1.get_width(), surface1.get_height()

    cdef:
        int i, j
        unsigned char [:, :, :] diffs = zeros((h, w, 3), dtype=numpy.uint8)
        unsigned char * r
        unsigned char * g
        unsigned char * b
        unsigned char * rr
        unsigned char * gg
        unsigned char * bb

    with nogil:

        for j in prange(h , schedule=SCHEDULE, num_threads=THREADS):
            for i in range(w):
                r = &rgb_array2[i, j, 0]
                g = &rgb_array2[i, j, 1]
                b = &rgb_array2[i, j, 2]
                rr = &rgb_array1[i, j, 0]
                gg = &rgb_array1[i, j, 1]
                bb = &rgb_array1[i, j, 2]
                if rr[0] != r[0] or gg[0] != g[0] or bb[0] != b[0]:
                            diffs[j, i, 0] = r[0]
                            diffs[j, i, 1] = g[0]
                            diffs[j, i, 2] = b[0]

    return frombuffer(diffs, (w, h), 'RGB')



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef object compare_png32bit(surface1, surface2):
    """
    COMPARE 32-BIT PYGAME SURFACES AND OUTPUT THE 
      DIFFERENCES INTO A NEW SURFACE 
    
    Incompatible with JPG images (JPEG is a lossy algorithm). 
    Surface1 & Surface2 must be same size and format (PNG)
    You can use PYGAME convert_alpha to convert the output image for fast blit
  
    :param surface1: pygame.Surface; Surface 1 (32-bit image) 
    :param surface2: pygame.Surface; Surface 2 (32-bit image) 
    :return : New 32-bit pygame surface (difference between surface1 and surface2) 
    
    """

    cdef:
        unsigned char [:, :, :] rgb_array1
        unsigned char [:, :, :] rgb_array2

    try:
        rgb_array1 = pixels3d(surface1) # .get_view('3')
    except (ValueError, pygame.error) as e:
        raise ValueError('\nImage/Surface not compatible.')

    try:
        rgb_array2 = pixels3d(surface2) # .get_view('3')
    except (ValueError, pygame.error) as e:
        raise ValueError('\nImage/Surface not compatible.')
    cdef int w, h
    w, h = surface1.get_width(), surface1.get_height()

    cdef:
        int i, j
        unsigned char [:, :, :] diffs = zeros((h, w, 4), dtype=numpy.uint8)
        unsigned char * r
        unsigned char * g
        unsigned char * b
        unsigned char * a

        unsigned char * rr
        unsigned char * gg
        unsigned char * bb
        unsigned char * aa

    with nogil:

        for j in prange(h , schedule=SCHEDULE, num_threads=THREADS):
            for i in range(w):
                r = &rgb_array2[i, j, 0]
                g = &rgb_array2[i, j, 1]
                b = &rgb_array2[i, j, 2]
                a = &rgb_array2[i, j, 3]
                rr = &rgb_array1[i, j, 0]
                gg = &rgb_array1[i, j, 1]
                bb = &rgb_array1[i, j, 2]
                aa = &rgb_array1[i, j, 3]
                if rr[0] != r[0] or gg[0] != g[0] or bb[0] != b[0] or aa[0]!=a[0]:
                            diffs[j, i, 0] = r[0]
                            diffs[j, i, 1] = g[0]
                            diffs[j, i, 2] = b[0]
                            diffs[j, i, 3] = a[0]

    return frombuffer(diffs, (w, h), 'RGBA')


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef np.ndarray[np.uint8_t, ndim=3] unmapping_array(
        const int[:, :] indexed_array_,
        tmp_array_ = None):

    """
    Convert a Pygame 2D array of shape (w, h) containing 
    int32 values into a 3D array of RGB values (uint8).

    This function transforms a 2D array where each element is a 32-bit integer
    encoding RGB pixel data into a 3D array where each pixel's red, green, 
    and blue channels are stored as separate uint8 values.

    **Purpose**:
    This is the reverse operation of Pygame's `pygame.surfarray.map_array()` function, which 
    converts a 3D RGB array into a 2D int32 array. It is particularly useful 
    for reconstructing RGB channels from a pixel map.

    **Use Cases**:
    - Pixel manipulation for rendering or processing.
    - Reverting encoded 2D pixel data into individual RGB channels for analysis.

    **Parameters**:
    - `indexed_array_` (numpy.ndarray):
        A 2D array of shape `(w, h)` with int32 values, where each element
        represents an RGB pixel value encoded as a single integer.

    - `tmp_array_` (numpy.ndarray, optional):
        An optional preallocated 3D array of shape `(w, h, 3)` and type `uint8`.
        If provided, this array will be filled with the decomposed RGB values.
        Providing this array can improve performance by avoiding additional
        memory allocation.

        - Shape: `(w, h, 3)`
        - Data type: `uint8`

    **Returns**:
    - A 3D numpy array of shape `(w, h, 3)` where:
        - `[..., 0]` contains the red (R) channel values.
        - `[..., 1]` contains the green (G) channel values.
        - `[..., 2]` contains the blue (B) channel values.

      Each channel's values are uint8 integers in the range 0-255.

    **Error Handling**:
    - Raises `TypeError` if the inputs are not valid numpy arrays or memory views.
    - Raises `ValueError` if the input shapes are incompatible or the data types
      are incorrect.

    **Performance Note**:
    - If `tmp_array_` is not provided, the function allocates a new array.
      Supplying a preallocated array is recommended for large datasets to 
      improve performance.

    **Example**:
    ```python
    # Assuming `indexed_array` is a 2D array with int32 encoded pixel data.
    indexed_array = pygame.surfarray.map_array(source, pixels3d(source))
    rgb_array = unmapping_array(indexed_array)  
    surface = pygame.Surface((800, 600))
    pygame.pixelcopy.array_to_surface(surface, rgb_array)

    # Alternatively, using a preallocated array:
    preallocated_array = numpy.empty((w, h, 3), dtype=numpy.uint8)
    rgb_array = unmapping_array(indexed_array, preallocated_array)
    ```
    
    ChatGPT Collaboration:
    To write comments  
    
    """

    # Check that the input array is a valid type
    if not (isinstance(indexed_array_, numpy.ndarray) or indexed_array_.__class__.__name__ == '_memoryviewslice'):
        raise TypeError("Input rgb_array must be a numpy.ndarray or memoryviewslice.")

    # If a temporary array is provided, validate its type
    if tmp_array_ is not None:
        if not (isinstance(tmp_array_, numpy.ndarray) or tmp_array_.__class__.__name__ == '_memoryviewslice'):
            raise TypeError("Input rgb_array must be a numpy.ndarray or memoryviewslice.")

    # Validate the data type of the input array
    if not is_type_memoryview(indexed_array_):
        # Check data types
        if (object>indexed_array_).dtype != numpy.int32:
            raise ValueError(f"indexed_array_ must be of type int32, but got {(<object>indexed_array_).dtype}.")

    # Validate the data type of the temporary array
    if not is_type_memoryview(tmp_array_):
        if tmp_array_ is not None and (<object>tmp_array_).dtype != numpy.uint8:
            raise ValueError(f"tmp_array_ must be of type uint8, but got {(<object>tmp_array_).dtype}.")

    # Ensure the input array is 2D
    if len((<object> indexed_array_).shape) != 2:
        raise ValueError(f"indexed_array_ must be a 2D array, but got shape {(<object> indexed_array_).shape}.")

    cdef:
        Py_ssize_t w, h  # Dimensions of the input array
        unsigned char[:, :, ::1] rgb_array  # Memoryview for the output array

    # Extract the width and height of the input array
    w, h = (<object>indexed_array_).shape[:2]

    # Validate tmp_array_ if provided
    if tmp_array_ is not None:
        # Ensure it is a 3D array
        if len((<object> tmp_array_).shape) != 3:
            raise ValueError(f"tmp_array_ must be a 3D array, but got shape {(<object> tmp_array_).shape}.")

        # Ensure the first two dimensions match the indexed array
        if (<object> tmp_array_).shape[ :2 ] != (w, h):
            raise ValueError(
                f"tmp_array_ must match the shape of indexed_array_ in the first two dimensions "
                f"(expected {(w, h)}, got {(<object> tmp_array_).shape[ :2 ]})."
            )

        # Ensure the last dimension is 3 (for RGB channels)
        if (<object> tmp_array_).shape[ 2 ] != 3:
            raise ValueError(f"tmp_array_ must have a depth of 3 (for RGB channels), "
                             f"but got {(<object> tmp_array_).shape[ 2 ]}.")

    try:
        # Allocate a new array if `tmp_array_` is not provided; otherwise, use it.
        rgb_array = empty((w, h, 3), dtype=numpy.uint8) if tmp_array_ is None else tmp_array_
    except Exception as e:
        # Handle memory allocation errors and provide detailed error messages
        raise ValueError(f"Cannot create tmp array shape (w, h, 3). Error: {e}")


    # Perform the unmapping operation (handled in the helper function)
    unmapping_array_c(w, h, indexed_array_, rgb_array)

    # Convert the result back to a NumPy array and return
    return numpy.asarray(rgb_array)


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef inline void unmapping_array_c(
        const Py_ssize_t w,
        const Py_ssize_t h,
        const int [:, :] indexed_array_,
        unsigned char [:, :, ::1] rgb_array)nogil:

    """
    
    This function performs the core operation of converting a 2D array of 
    int32 values (`indexed_array_`) into a 3D array of RGB uint8 values.
    
    The function is designed for performance, utilizing the "nogil" directive 
    and parallel loops to process pixel data efficiently.
    
    **Example**:
    ```python
    indexed_array = pygame.surfarray.map_array(source, pixels3d(source))
    rgb_array = numpy.empty((w, h, 3), dtype=numpy.uint8)
    rgb_array = unmapping_array_c(indexed_array, rgb_array)  
    surface = pygame.Surface((800, 600))
    pygame.pixelcopy.array_to_surface(surface, rgb_array)

    ```

    **Parameters**:
    - `indexed_array_` (memoryview slice of numpy.ndarray):
    A 2D array of shape `(w, h)` containing RGB pixel values encoded as 
    int32 integers. This array is processed as a memoryview for efficient
    element access during "nogil" execution.
    
    - `rgb_array` (memoryview slice of numpy.ndarray):
    A preallocated 3D array of shape `(w, h, 3)` and type `uint8`. This array
    will be modified in place to store the decomposed RGB values.

    **Returns**:
    - void: The function does not return a value. The results are written 
      directly into the `rgb_array` parameter.
    
    **ChatGPT Collaboration**:
    To write comments  
    """

    if rgb_array is None:
        raise ValueError('Argument rgb_array cannot be None')

    cdef:
        int i, j  # Loop indices for iterating through the array
        unsigned int n  # Temporary variable to store the int32 pixel value
        unsigned char * r  # Pointer to the red channel of the current pixel
        unsigned char * g  # Pointer to the green channel of the current pixel
        unsigned char * b  # Pointer to the blue channel of the current pixel

    # Use parallel processing for the outer loop (over rows) to enhance performance.
    # The `prange` function (from Cython) enables multi-threaded execution.
    # The schedule and number of threads are controlled by the `SCHEDULE` and `THREADS` constants.
    for j in prange(h, schedule=SCHEDULE, num_threads=THREADS):
        for i in range(w):
            # Get the encoded int32 pixel value from the input array
            n = indexed_array_[i, j]

            # Extract pointers to the RGB channels in the output array
            r = &rgb_array[i, j, 0]
            g = &rgb_array[i, j, 1]
            b = &rgb_array[i, j, 2]

            # Decompose the int32 value into separate R, G, and B channels
            r[0] = ((n >> 16) & 255)  # Extract the red channel (bits 16-23)
            g[0] = ((n >> 8) & 255)   # Extract the green channel (bits 8-15)
            b[0] = (n & 255)          # Extract the blue channel (bits 0-7)




@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef np.ndarray[np.int32_t, ndim=2] mapping_array(
        const unsigned char [:, :, :] rgb_array,
        tmp_array_ = None):

    """
    Convert a 3D RGB array (shape: (W, H, 3)) to a mapped 2D array of integers.

    This function maps a 3D array of RGB values (unsigned 8-bit integers) to a 2D array 
    of mapped integer values. It is an optimized equivalent of `pygame.map_array(Surface, array3d) -> array2d`, 
    designed to process NumPy arrays of shape (W, H, 3). The mapping assigns an integer representation 
    to each RGB triplet.

    Parameters
    ----------
    rgb_array : numpy.ndarray
        A 3D NumPy array of shape (W, H, 3) containing the source RGB pixels, with data type `uint8`.
        This is the input array that will be converted to a 2D array of mapped values.
    
    tmp_array_ : numpy.ndarray, optional
        A pre-allocated empty 2D NumPy array of shape (W, H) with data type `int32`.
        If provided, this array will be used to store the mapped integer values.
        If not provided, a new array will be created internally.

    Returns
    -------
    numpy.ndarray
        A 2D NumPy array of shape (W, H) with data type `int32`, containing the mapped integer 
        values corresponding to the RGB triplets in the input array.

    Raises
    ------
    ValueError
        - If `rgb_array` is `None`.
        - If `rgb_array` does not have the required shape (W, H, 3).
        - If `rgb_array` has an invalid data type (not `uint8`).
        - If `tmp_array_` has the wrong shape, data type, or other inconsistencies.

    TypeError
        - If `rgb_array` or `tmp_array_` is not a NumPy array or memoryview slice.

    Notes
    -----
    - The input `rgb_array` must be compatible with `pygame.surfarray.array3d` output 
      (an array of unsigned 8-bit integers).
    - The function directly modifies the `tmp_array_` if provided.
    - If no `tmp_array_` is provided, a new array is created internally for the output.

    """

    # Validate the input `rgb_array`
    if rgb_array is None:
        # Check if the input array is None
        raise ValueError(f"rgb_array cannot be None. Received: {rgb_array} (<object>NoneType)")

    if not (isinstance(rgb_array, numpy.ndarray) or rgb_array.__class__.__name__ == '_memoryviewslice'):
        # Ensure the input is a numpy array or memoryview slice
        raise TypeError("Input rgb_array must be a numpy.ndarray or memoryviewslice.")

    if not is_type_memoryview(rgb_array):
        # Validate the data type of `rgb_array`
        if (<object>rgb_array).dtype != numpy.int32:
            raise ValueError(f"rgb_array must be of type int32, but got {(<object>rgb_array).dtype}.")

    if (<object>rgb_array).ndim != 3 or (<object>rgb_array).shape[2] != 3:
        # Ensure `rgb_array` has 3 dimensions and the last dimension has a size of 3
        raise ValueError(
            f"rgb_array must have shape (w, h, 3). Received: {(<object>rgb_array).shape})")

    cdef:
        Py_ssize_t w, h  # Dimensions of the input array
        int[:, ::1] indexed_array  # 2D array to store the mapped values

    w, h = rgb_array.shape[:2]  # Extract the width and height of the array

    try:
        # Validate or create `tmp_array_`
        if tmp_array_ is None:
            # If no temporary array is provided, create an empty array of the correct size and type
            indexed_array = empty((w, h), dtype=numpy.int32)

        else:
            # Validate the provided `tmp_array_`
            if not (isinstance(tmp_array_, numpy.ndarray) or tmp_array_.__class__.__name__ == '_memoryviewslice'):
                raise TypeError("Input tmp_array_ must be a numpy.ndarray or memoryviewslice.")

            if not is_type_memoryview(tmp_array_):
                # Check if `tmp_array_` has the correct data type
                if (<object>tmp_array_).dtype != numpy.int32:
                    raise TypeError(
                        f"tmp_array_ must have dtype int32. Received: {(<object>tmp_array_).dtype})")

                # Ensure `tmp_array_` has the correct shape
                if (<object>tmp_array_).shape != (w, h):
                    raise ValueError(
                        f"tmp_array_ must have shape (w, h). Received: {(<object>tmp_array_).shape})")

            # Use the provided `tmp_array_` for mapping
            indexed_array = tmp_array_

    except Exception as e:
        # Catch any exceptions during validation or creation of `tmp_array_` and raise a descriptive error
        raise ValueError(
            f"Error while validating or creating tmp_array_: {e} "
            f"(<object>{type(tmp_array_).__name__ if tmp_array_ is not None else 'NoneType'})")

    # Perform the mapping operation using the mapping_array_c function
    mapping_array_c(w, h, rgb_array, indexed_array)

    # Return the mapped array as a numpy array
    return numpy.asarray(indexed_array)



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef inline void mapping_array_c(
        const Py_ssize_t w,
        const Py_ssize_t h,
        const unsigned char [:, :, :] rgb_array,
        int [:, ::1] indexed_array)nogil:

    """
    Perform RGB to integer mapping for a 3D array (shape: (W, H, 3)) into a 2D array (shape: (W, H)).
    
    This function is a low-level implementation for mapping RGB triplets in a 3D array to integer 
    values in a 2D array. It processes each pixel of the input array `rgb_array` (of shape (W, H, 3)) 
    and stores the resulting integer values in the `indexed_array` (of shape (W, H)).
    
    Parameters
    ----------
    w : Py_ssize_t
       The width (number of columns) of the input `rgb_array`.
    
    h : Py_ssize_t
       The height (number of rows) of the input `rgb_array`.
    
    rgb_array : unsigned char[:, :, :]
       A 3D memoryview representing the source RGB pixel data. Each pixel is a triplet of unsigned 
       8-bit integers (values in the range 0-255) representing red, green, and blue channels.
    
    indexed_array : int[:, ::1]
       A 2D memoryview (pre-allocated) to store the mapped integer values. Each pixel in the 
       `indexed_array` corresponds to a packed 24-bit RGB value derived from the input `rgb_array`.
    
    Returns
    -------
    None
       The function modifies the `indexed_array` in place.
    
    Notes
    -----
    - The function operates in `nogil` mode for multithreaded performance. 
    - It uses OpenMP's `prange` to parallelize operations along the width (`w`), with threading 
     parameters controlled by `SCHEDULE` and `THREADS`.
    - The mapping formula for each pixel is:
       `indexed_array[i, j] = (r << 16) + (g << 8) + b`
     where `r`, `g`, and `b` are the red, green, and blue values of the corresponding pixel.
    
    Performance Considerations
    --------------------------
    - Ensure that both `rgb_array` and `indexed_array` are properly aligned and contiguous in memory 
     to avoid performance degradation.
    - Pre-allocate `indexed_array` to the correct dimensions (W, H) and data type (`int`) before calling this function.
    """

    cdef:
        int i, j  # Loop counters for rows and columns
        const unsigned char *r  # Pointer to the red channel of a pixel
        const unsigned char *g  # Pointer to the green channel of a pixel
        const unsigned char *b  # Pointer to the blue channel of a pixel

        # Parallel loop over rows of the array, utilizing OpenMP for multithreading
    for i in prange(w, schedule = SCHEDULE, num_threads = THREADS):
        # Loop over columns of the array
        for j in range(h):
            # Get pointers to the red, green, and blue values of the current pixel
            r = &rgb_array[ i, j, 0 ]  # Pointer to red channel value
            g = &rgb_array[ i, j, 1 ]  # Pointer to green channel value
            b = &rgb_array[ i, j, 2 ]  # Pointer to blue channel value

            # Map the RGB values to an integer and store in the indexed array
            # The mapping formula is: (R << 16) + (G << 8) + B
            # This packs the RGB triplet into a single 24-bit integer
            indexed_array[ i, j ] = (r[ 0 ] << 16) + (g[ 0 ] << 8) + b[ 0 ]





@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef inline void blend_pixel_mapped_arrays(
    unsigned int [:, :] target_pixels,
    const unsigned int [:, :] blend_pixels,
    unsigned char special_flags = 0
):
    """
    Blend two 2D mapped arrays together (compatible with Pygame special flags).

    This function blends `target_pixels` and `blend_pixels` using the specified `special_flags`. 
    The blending operation is performed **in-place**, modifying `target_pixels` directly.

    Parameters
    ----------
    target_pixels : memoryview (unsigned int[:, :])
        A **2D Cython memoryview** (or a NumPy array converted via `numpy.asarray()`), where each element 
        represents a pixel with mapped RGB values. The blend will be applied directly to this array 
        (modifying it in-place).

    blend_pixels : memoryview (unsigned int[:, :])
        A **2D Cython memoryview** (or a NumPy array converted via `numpy.asarray()`), containing the 
        destination array pixels with mapped RGB values. This array provides the pixel data to blend 
        into `target_pixels`.

    special_flags : unsigned char, optional (default=0)
        A blending mode flag that determines how `target_pixels` and `blend_pixels` are combined.
        It supports values between `0-5` and is compatible with **Pygame BLEND flags**, such as:
            - `BLEND_RGB_ADD`  (1) → Additive blending
            - `BLEND_RGB_SUB`  (2) → Subtractive blending
            - `BLEND_RGB_MULT` (3) → Multiplicative blending
            - `BLEND_RGB_MIN`  (4) → Minimum value blending
            - `BLEND_RGB_MAX`  (5) → Maximum value blending
        If `special_flags = 0`, no blending is applied.

    Returns
    -------
    void
        The function modifies `target_pixels` in-place, so no value is returned.

    Raises
    ------
    TypeError
        - If `target_pixels` or `blend_pixels` is not a **Cython memoryview** or **NumPy array**.
    ValueError
        - If `target_pixels` and `blend_pixels` have different shapes.
        - If `special_flags` is outside the valid range `[0-5]`.

    Notes
    -----
    - This function is optimized for performance using Cython (`cpdef inline`).
    - The blending operation is applied directly to `target_pixels`, so ensure you pass a copy if you 
      need to retain the original data.
    - Compatible with Pygame's surface blending operations.

    Contributions
    -------------
    - **Enhancements & Documentation Improvements**: ChatGPT (OpenAI)
      - Added explicit input validation for memoryviews and NumPy arrays.
      - Improved error handling for mismatched shapes and invalid `special_flags` values.
      - Refactored docstring for clarity, readability, and consistency.
    """

    # Ensure inputs are memoryviews or NumPy arrays
    if not (is_type_memoryview(target_pixels) or isinstance(target_pixels, numpy.ndarray)) or \
            not (is_type_memoryview(blend_pixels) or isinstance(blend_pixels, numpy.ndarray)):
        raise TypeError("Both target_pixels and blend_pixels must be memoryviews or NumPy arrays.")

    # Ensure both arrays have the same shape
    if (<object>target_pixels).shape != (<object>blend_pixels).shape:
        raise ValueError(f"target_pixels shape {(<object>target_pixels).shape} "
                         f"does not match blend_pixels shape {(<object>blend_pixels).shape}.")

    # Ensure special_flags is a valid unsigned char (0-5)
    if not (0 <= special_flags <= 5):
        raise ValueError(f"Invalid special_flags value: {special_flags}. Must be between 0 and 5.")

    # Call the Cython function to perform the blending
    blend_pixel_mapped_arrays_c(target_pixels, blend_pixels, special_flags)


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef inline void blend_pixel_mapped_arrays_c(
    unsigned int [:, :] target_pixels,
    const unsigned int [:, :] blend_pixels,
    unsigned char special_flags = 0
    )nogil:


    # No exception checks, this is the core function and all the exception checks
    #  are made upon calling this function. If you are calling this function
    #  directly from cython code make sure to provides memoryviewslice and both array
    #  must be unsigned int with same shapes
    cdef:

        int c1, c2, c3, r1, g1, b1
        unsigned char r, g, b
        int i=0, j=0
        Py_ssize_t w = target_pixels.shape[0]
        Py_ssize_t h = target_pixels.shape[1]
        unsigned int * ind1
        const unsigned int * ind2

    with nogil:
        # BLEND_RGB_ADD
        if special_flags == 1:
            for i in prange(w, schedule=SCHEDULE, num_threads=THREADS):
                for j in range(h):
                    ind1 = &target_pixels[i, j]
                    ind2 = &blend_pixels[i, j]
                    c1 = ((ind1[0] >> 16) & 255) + ((ind2[0] >> 16) & 255)
                    c2 = ((ind1[0] >> 8) & 255) + ((ind2[0] >> 8) & 255)
                    c3 = (ind1[0] & 255) + (ind2[0] & 255)
                    r = c1 if c1 < 255 else 255
                    g = c2 if c2 < 255 else 255
                    b = c3 if c3 < 255 else 255
                    ind1[0] = (r << 16) + (g << 8) + b
        # BLEND_RGB_SUB
        elif special_flags == 2:
            for i in prange(w, schedule=SCHEDULE, num_threads=THREADS):
                for j in range(h):
                    ind1 = &target_pixels[i, j]
                    ind2 = &blend_pixels[i, j]
                    c1 = ((ind1[0] >> 16) & 255) - ((ind2[0] >> 16) & 255)
                    c2 = ((ind1[0] >> 8) & 255) - ((ind2[0] >> 8) & 255)
                    c3 = (ind1[0] & 255) - (ind2[0] & 255)
                    r = c1 if c1 > 0 else 0
                    g = c2 if c2 > 0 else 0
                    b = c3 if c3 > 0 else 0
                    ind1[0] = (r << 16) + (g << 8) + b
        # BLEND_RGB_MULT
        elif special_flags == 3:
            for i in prange(w, schedule=SCHEDULE, num_threads=THREADS):
                for j in range(h):
                    ind1 = &target_pixels[i, j]
                    ind2 = &blend_pixels[i, j]
                    c1 = (((ind1[0] >> 16) & 255) * ((ind2[0] >> 16) & 255)) >> 8
                    c2 = (((ind1[0] >> 8) & 255) * ((ind2[0] >> 8) & 255)) >> 8
                    c3 = ((ind1[0] & 255) * (ind2[0] & 255)) >> 8
                    r = c1 if c1 < 255 else 255
                    g = c2 if c2 < 255 else 255
                    b = c3 if c3 < 255 else 255
                    ind1[0] = (r << 16) + (g << 8) + b

        # BLEND_RGB_MIN
        elif special_flags == 4:
            for i in prange(w, schedule=SCHEDULE, num_threads=THREADS):
                for j in range(h):
                    ind1 = &target_pixels[i, j]
                    ind2 = &blend_pixels[i, j]
                    r1 = ((ind1[0] >> 16) & 255)
                    c1 = ((ind2[0] >> 16) & 255)
                    g1 = ((ind1[0] >> 8) & 255)
                    c2 = ((ind2[0] >> 8) & 255)
                    b1 = (ind1[0] & 255)
                    c3 = (ind2[0] & 255)
                    r1 = r1 if r1 < c1 else c1
                    g1 = g1 if g1 < c2 else c2
                    b1 = b1 if b1 < c3 else c3
                    ind1[0] = (r1 << 16) + (g1 << 8) + b1
        # BLEND_RGB_MAX
        elif special_flags == 5:
            for i in prange(w, schedule=SCHEDULE, num_threads=THREADS):
                for j in range(h):
                    ind1 = &target_pixels[i, j]
                    ind2 = &blend_pixels[i, j]
                    r1 = ((ind1[0] >> 16) & 255)
                    c1 = ((ind2[0] >> 16) & 255)
                    g1 = ((ind1[0] >> 8) & 255)
                    c2 = ((ind2[0] >> 8) & 255)
                    b1 = (ind1[0] & 255)
                    c3 = (ind2[0] & 255)
                    r1 = r1 if r1 > c1 else c1
                    g1 = g1 if g1 > c2 else c2
                    b1 = b1 if b1 > c3 else c3
                    ind1[0] = (r1 << 16) + (g1 << 8) + b1




@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
@cython.profile(True)
cpdef tuple get_rgb_channel_means(object array3d):
    """
    Compute the mean values for each channel of an RGB 3D array.

    This function calculates the mean for the red, green, and blue channels 
    for each pixel in an RGB image represented by a 3D array with shape (W, H, 3).
    The function is compatible with arrays of type uint8 or float32.

    :param array3d: A NumPy 3D array of shape (W, H, 3), representing an RGB image.
                   The dtype must be either uint8 or float32.
    
    :return: A tuple (mean_red, mean_green, mean_blue, pixel_count), where:
             - `mean_red`: Mean value of the red channel.
             - `mean_green`: Mean value of the green channel.
             - `mean_blue`: Mean value of the blue channel.
             - `pixel_count`: Total number of pixels (W * H).
             
    :raises ValueError: If the input array does not meet the expected shape or dtype.
    """

    # Ensure the input is a numpy array
    if not isinstance(array3d, np.ndarray):
        raise ValueError("Input must be a numpy ndarray.")

    # Check for the expected shape: (W, H, 3)
    if array3d.ndim != 3 or array3d.shape[ 2 ] != 3:
        raise ValueError(f"Input array must have shape (W, H, 3), got shape {array3d.shape}.")

    # Ensure dtype is either uint8 or float32
    if array3d.dtype not in [ numpy.uint8, numpy.float32 ]:
        raise ValueError(f"Input array must be of type uint8 or float32, got {array3d.dtype}.")

    return get_rgb_channel_means_c(array3d)


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
@cython.profile(True)
cdef tuple get_rgb_channel_means_c(object array3d):
    """
    Compute the mean values for each RGB(A) channel in a 3D image array.

    This Cython function calculates the mean of the red, green, and blue channels 
    for each pixel in an RGB(A) image represented by a 3D array with shape (W, H, 3) or (W, H, 4). 
    The function is optimized for performance using multi-threading, and it supports 
    both uint8 and float32 data types for the input array.

    This function should be called from Cython code. For Python usage, 
    use the `get_rgb_channel_means` function.

    :param array3d: A 3D numpy array of shape (W, H, 3) or (W, H, 4), where `W` is the width,
                     `H` is the height, and the last dimension represents the RGB(A) channels. 
                     The array must have dtype `uint8` or `float32`.
                     
    :return: A tuple containing:
             - The mean value of the red channel (float).
             - The mean value of the green channel (float).
             - The mean value of the blue channel (float).
             - The mean value of the alpha channel (float) if 32-bit data 
             - The total number of pixels in the image (unsigned int).
             
    :raises ValueError: If the input array’s dtype is not `uint8` or `float32`, or if the shape is invalid.
    
    """

    # Declare Cython variables with static types for performance optimization
    cdef:
        Py_ssize_t w, h, bytesize  # Variables to store the width and height of the image

    # Extract the width (w) and height (h) from the shape of the input 3D array (array3d)
    w, h, bytesize = (<object>array3d).shape[:3]

    # Declare additional Cython variables
    cdef:
        int i, j  # Loop counters for iterating over the image pixels
        unsigned int c = 0  # Variable to store the total number of pixels (width * height)

        volatile float r = 0  # Accumulator for the sum of red channel values
        volatile float g = 0  # Accumulator for the sum of green channel values
        volatile float b = 0  # Accumulator for the sum of blue channel values
        volatile float a = 0

        # Memory views for efficient access to the array data
        const unsigned char [:, :, :] u_array = \
            array3d if (<object>array3d).dtype == 'uint8' else None  # For 8-bit unsigned integer arrays

        const float [:, :, :] f_array = array3d \
            if (<object>array3d).dtype == 'float32' else None  # For 32-bit float arrays


    # Calculate the total number of pixels in the image
    c = w * h

    # Check the data type of the input array to determine how to process it
    if (<object>array3d).dtype == 'float32':
        # If the array is of type float32, use the f_array memory view
        with nogil:  # Release the GIL (Global Interpreter Lock) for parallel execution
            for j in prange(h, schedule='dynamic', num_threads=THREADS):  # Parallel loop over height
                for i in range(w):  # Loop over width
                    r += f_array[i, j, 0]  # Accumulate red channel values
                    g += f_array[i, j, 1]  # Accumulate green channel values
                    b += f_array[i, j, 2]  # Accumulate blue channel values
                    if bytesize == 4:
                        a += f_array[i, j, 3]
    else:
        # If the array is of type uint8, use the u_array memory view
        with nogil:  # Release the GIL for parallel execution
            for j in prange(h, schedule='dynamic', num_threads=THREADS):  # Parallel loop over height
                for i in range(w):  # Loop over width
                    r += u_array[i, j, 0]  # Accumulate red channel values
                    g += u_array[i, j, 1]  # Accumulate green channel values
                    b += u_array[i, j, 2]  # Accumulate blue channel values
                    if bytesize == 4:
                        a += u_array[i, j, 3]

    # Compute and return the mean values for each channel by dividing
    # the accumulated sums by the total number of pixels
    cdef float inv_c = <float>1.0 / c

    if bytesize == 4:
        return r * inv_c, g * inv_c, b * inv_c, a * inv_c, c
    else:
        return r * inv_c, g * inv_c, b * inv_c, c



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef inline im_stats_with_alpha analyze_image_channels (object image_array):
    """
    Calculate the mean and standard deviation for each RGB channel in an image array.

    This function computes the mean and standard deviation for each of the RGB channels 
    (Red, Green, and Blue) from the input 3D image array. The input array can be of type 
    `uint8` or `float32`, and it should have the shape (w, h, 3), where w is the width 
    and h is the height of the image. The function returns an `im_stats_with_alpha` structure that 
    contains the mean and standard deviation for each channel, which is similar to a Python 
    dictionary.

    The function operates in an efficient manner by utilizing Cython and multi-threading 
    (with `nogil`) for large image arrays.

    Example usage:
        image_stats = array3d_stats(image_array)
        print(f"Red Channel Mean: {image_stats.red_mean}")
        print(f"Green Channel Mean: {image_stats.green_mean}")
        print(f"Blue Channel Mean: {image_stats.blue_mean}")

    :param image_array: numpy.ndarray; a 3D array with shape (w, h, 3), of type uint8 or float32.
                    It contains RGB pixel values, where each pixel consists of red, green, 
                    and blue channel values (ranging from 0 to 255 for uint8, or 0 to 1 for float32).

    :return: im_stats_with_alpha; a structure containing normalized values:
                - `red_mean`: Mean value of the red channel.
                - `green_mean`: Mean value of the green channel.
                - `blue_mean`: Mean value of the blue channel.
                - `red_std_dev`: Standard deviation of the red channel.
                - `green_std_dev`: Standard deviation of the green channel.
                - `blue_std_dev`: Standard deviation of the blue channel.

    :raises TypeError: If `image_array` is not a numpy ndarray or is of an unsupported data type.
    
    :raises ValueError: 
        - If `image_array` does not have exactly 3 dimensions, or the third dimension is not 3 (RGB channels).
        - If `image_array` is empty (width or height is 0).
        - If `image_array` has an invalid data type (not `uint8` or `float32`).

    Notes:
        - The function assumes the input image array has three channels (RGB).
        - If `image_array` has zero pixels (e.g., an empty array), the function raises a `ValueError`.
        - The `im_stats_with_alpha` structure is similar to a Python dictionary but designed for use in Cython.
    """

    # Check if the input is a numpy ndarray
    if not hasattr(image_array, 'dtype') or not hasattr(image_array, 'shape'):
        raise TypeError("Input must be a numpy ndarray.")

    # Check if the array has 3 dimensions and shape is (w, h, 3)
    if len(image_array.shape) != 3 or image_array.shape[2] not in (3, 4):
        raise ValueError(f"Input array must have 3 dimensions (w, h, 3 | 4). Got shape {image_array.shape}.")

    # Check if the array is empty (width or height is 0)
    if image_array.shape[0] == 0 or image_array.shape[1] == 0:
        raise ValueError("Input array cannot be empty. Width and height must be non-zero.")

    # Check if the array is of a valid type (uint8 or float32)
    array_type = image_array.dtype
    if array_type not in ['uint8', 'float32']:
        raise TypeError(f"Invalid array type: {array_type}. Supported types are 'uint8' and 'float32'.")

    # Define necessary variables and structures
    cdef:
        Py_ssize_t w, h
        int i, j
        im_stats_with_alpha rgba_stats          # Structure to store the statistics (mean and std dev)

    # Extract width and height from the array shape
    w, h, num_channels = image_array.shape[:3]


    # Variables to store the computed means and standard deviations
    cdef:
        float mean_r, mean_g, mean_b, mean_a
        volatile float dev_r = <float>0.0
        volatile float dev_g = <float>0.0
        volatile float dev_b = <float>0.0
        volatile float dev_a = <float>0.0
        # Memory views for uint8 (most common) and float32 arrays
        unsigned char[:, :, :] uint8_array = image_array if array_type == 'uint8' else None
        float[:, :, :] float32_array = image_array if array_type == 'float32' else None

    # Get the mean values for each RGB channel (from a helper function)
    if num_channels == 3:
        mean_r, mean_g, mean_b, c = get_rgb_channel_means_c(image_array)
    else:
        mean_r, mean_g, mean_b, mean_a, c = get_rgb_channel_means_c(image_array)

    # Compute the standard deviation for each channel (looping over each pixel)
    # Loop over all pixels in the array (using the appropriate array type)
    if array_type == 'uint8':
        with nogil:
            for i in range(w):
                for j in range(h):
                    # Calculate squared deviation for the red channel
                    dev_r += <float>pow(uint8_array[i, j, 0] - mean_r, <float>2.0)
                    # Calculate squared deviation for the green channel
                    dev_g += <float>pow(uint8_array[i, j, 1] - mean_g, <float>2.0)
                    # Calculate squared deviation for the blue channel
                    dev_b += <float>pow(uint8_array[i, j, 2] - mean_b, <float>2.0)
                    if num_channels == 4:
                        # Calculate squared deviation for the alpha channel
                        dev_a += <float> pow(uint8_array[ i, j, 3 ] - mean_a, <float> 2.0)

    else:
        with nogil:
            for i in range(w):
                for j in range(h):
                    # Calculate squared deviation for the red channel
                    dev_r += <float>pow(float32_array[i, j, 0] - mean_r, <float>2.0)
                    # Calculate squared deviation for the green channel
                    dev_g += <float>pow(float32_array[i, j, 1] - mean_g, <float>2.0)
                    # Calculate squared deviation for the blue channel
                    dev_b += <float>pow(float32_array[i, j, 2] - mean_b, <float>2.0)
                    if num_channels == 4:
                        # Calculate squared deviation for the alpha channel
                        dev_a += <float> pow(float32_array[ i, j, 3 ] - mean_a, <float> 2.0)

    # Calculate standard deviation from squared deviations
    cdef float std_dev_r, std_dev_g, std_dev_b, std_dev_a

    if c==0:
        raise ValueError("Internal variable c cannot be null")

    std_dev_r = sqrt(dev_r / c)  # Red channel standard deviation
    std_dev_g = sqrt(dev_g / c)  # Green channel standard deviation
    std_dev_b = sqrt(dev_b / c)  # Blue channel standard deviation
    if num_channels == 4:
        std_dev_a = sqrt(dev_a / c)  # Alpha channel standard deviation

    # Store the results in the stats structure (multiply by scale factor)
    rgba_stats.red_mean = mean_r * c_1_255
    rgba_stats.red_std_dev = std_dev_r * c_1_255
    rgba_stats.green_mean = mean_g * c_1_255
    rgba_stats.green_std_dev = std_dev_g * c_1_255
    rgba_stats.blue_mean = mean_b * c_1_255
    rgba_stats.blue_std_dev = std_dev_b * c_1_255
    if num_channels == 4:
        rgba_stats.alpha_mean = mean_a * c_1_255
        rgba_stats.alpha_std_dev = std_dev_a * c_1_255
    else:
        rgba_stats.alpha_mean = <float>0.0
        rgba_stats.alpha_std_dev = <float>0.0
        # Return the image statistics structure containing mean and std dev for each channel
    return rgba_stats


