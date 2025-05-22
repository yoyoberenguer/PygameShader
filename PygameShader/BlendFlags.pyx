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

import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

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


cimport numpy as np

from libc.math cimport sqrtf as sqrt, atan2f as atan2, sinf as sin,\
    cosf as cos, nearbyintf as nearbyint, expf as exp, powf as pow, floorf as floor
from libc.stdlib cimport malloc, rand, free
from libc.math cimport roundf as round_c, fminf as fmin, fmaxf as fmax
from libc.stdio cimport printf
from libc.string cimport memcpy
from PygameShader.config import OPENMP, THREAD_NUMBER, __VERSION__

cdef int THREADS = 1

if OPENMP:
     THREADS = THREAD_NUMBER

DEF SCHEDULE = 'static'


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef inline void blit_s(destination, image, tuple position=(0, 0)):

    """
    Blends a sprite (image) onto a destination surface.

    This function overlays `image` onto `destination` at the specified `position`.
    It is similar to the `pygame.Surface.blit()` function but performs the operation
    manually at the pixel level.

    :param destination: pygame.Surface
        The target surface onto which the image will be blended.
        
    :param image: pygame.Surface
        The image to overlay onto the destination.
        
    :param position: tuple (x, y), optional
        The (x, y) coordinates where the `image` should be placed. Default is (0, 0).

    :raises TypeError: If `destination` or `image` is not a `pygame.Surface`.
    :raises ValueError: If the pixel formats of `destination` and `image` do not match.
    :raises ValueError: If `position` contains negative coordinates.
    :raises ValueError: If the pixel data cannot be accessed as a 3D array.
    """

    # Ensure `destination` is a pygame.Surface
    if not isinstance(destination, pygame.Surface):
        raise TypeError(f'Argument `destination` must be a pygame.Surface, got {type(destination)}')

    # Ensure `image` is a pygame.Surface
    if not isinstance(image, pygame.Surface):
        raise TypeError(f'Argument `image` must be a pygame.Surface, got {type(image)}')

    cdef:
        unsigned char[:, :, :] destination_rgb_array
        unsigned char[:, :, :] image_rgb_array
        Py_ssize_t w, h, byte_size, w1, h1, byte_size1

    # Get pixel byte sizes for both surfaces
    byte_size = destination.get_bytesize()
    byte_size1 = image.get_bytesize()

    # Ensure both surfaces have the same pixel format
    if byte_size != byte_size1:
        raise ValueError(
            f"Both surfaces must have the same pixel format. "
            f"Got destination ({destination.get_bitsize()}-bit), "
            f"image ({image.get_bitsize()}-bit)."
        )

    # Try to get a 3D pixel array view of the destination surface
    try:
        destination_rgb_array = destination.get_view('3')

    except Exception as e:
        raise ValueError(f"Cannot reference destination pixels as a 3D array.\n{e}")

    # Try to get a 3D pixel array view of the image surface
    try:
        image_rgb_array = image.get_view('3')

    except Exception as e:
        raise ValueError(f"Cannot reference image pixels as a 3D array.\n{e}")

    # Get dimensions of both surfaces
    w, h   = (<object>destination_rgb_array).shape[:2]
    w1, h1 = (<object>image_rgb_array).shape[:2]

    cdef:
        int w_min = position[0]
        int h_min = position[1]

    # Ensure the position is within valid bounds
    if w_min < 0 or h_min < 0:
        raise ValueError(f"Argument `position` must have non-negative values, got {w_min, h_min}")

    cdef:
        # Determine the effective width and height of the blending region
        int w_max = min(w, w_min + w1)
        int h_max = min(h, h_min + h1)
        unsigned int x0, y0
        int i, j

    # Perform the blending operation in parallel
    with nogil, parallel():

        for j in prange(h_min, h_max):
            y0 = j - h_min # Offset into the image surface

            for i in range(w_min, w_max):
                x0 = i - w_min  # Offset into the image surface

                if byte_size == 3:

                    destination_rgb_array[ i, j, 0 ] = image_rgb_array[x0, y0, 0]
                    destination_rgb_array[ i, j, 1 ] = image_rgb_array[x0, y0, 1]
                    destination_rgb_array[ i, j, 2 ] = image_rgb_array[x0, y0, 2]

                else:
                    destination_rgb_array[ i, j, 0 ] = image_rgb_array[x0, y0, 0]
                    destination_rgb_array[ i, j, 1 ] = image_rgb_array[x0, y0, 1]
                    destination_rgb_array[ i, j, 2 ] = image_rgb_array[x0, y0, 2]
                    destination_rgb_array[ i, j, 3 ] = image_rgb_array[x0, y0, 3]
                    


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef inline void blend_add_surface(image1, image2):

    """
    Python hook method

    Blend-adds two surfaces, equivalent to `pygame.BLEND_RGB_ADD`.

    Both surfaces must have the same size and be compatible with 24-bit or 32-bit formats.
    This function adds the pixel values of `image2` onto `image1`, modifying `image1` in place.

    Parameters
    ----------
    image1 : pygame.Surface
        The destination surface (modified in place).

    image2 : pygame.Surface
        The source surface to be added.

    Returns
    -------
    None
        This function modifies `image1` in place.

    Examples
    --------
    Basic usage:

    >>> blend_add_surface(image1, image2)

    Equivalent to:

    >>> image1.blit(image2, (0, 0), special_flags=pygame.BLEND_RGB_ADD)

    Performance Comparison:

    >>> import timeit
    >>> t = timeit.timeit('blend_add_surface(image1, image2)',
    ...                   'from __main__ import blend_add_surface, image1, image2', number=1000)
    >>> print(t / 1000)  # Example output: 0.00038

    >>> t = timeit.timeit('image1.blit(im1, (0, 0), special_flags=pygame.BLEND_RGB_ADD)',
    ...                   'from __main__ import image1, image2, pygame', number=1000)
    >>> print(t / 1000)  # Example output: 0.00121
    """

    blend_add_surface_c(image1, image2)


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef inline void blend_add_surface_c(image1, image2):

    """
    Blend-adds two surfaces, equivalent to `pygame.BLEND_RGB_ADD`.

    Both surfaces must have the same size and be compatible with 24-bit or 32-bit formats.
    This function adds the pixel values of `image2` onto `image1`, modifying `image1` in place.

    Parameters
    ----------
    image1 : pygame.Surface
        The destination surface (modified in place).

    image2 : pygame.Surface
        The source surface to be added.

    Returns
    -------
    None
        This function modifies `image1` in place.

    Examples
    --------
    Basic usage:

    >>> blend_add_surface(image1, image2)

    Equivalent to:

    >>> image1.blit(image2, (0, 0), special_flags=pygame.BLEND_RGB_ADD)

    Performance Comparison:

    >>> import timeit
    >>> t = timeit.timeit('blend_add_surface(image1, image2)',
    ...                   'from __main__ import blend_add_surface, image1, image2', number=1000)
    >>> print(t / 1000)  # Example output: 0.00038

    >>> t = timeit.timeit('image1.blit(image1, (0, 0), special_flags=pygame.BLEND_RGB_ADD)',
    ...                   'from __main__ import image1, image2, pygame', number=1000)
    >>> print(t / 1000)  # Example output: 0.00121
    """

    cdef:
        unsigned char[::1] image1_buffer
        unsigned char[::1] image2_buffer
        Py_ssize_t bytesize1
        Py_ssize_t bytesize2

    bytesize1 = image1.get_bytesize()
    bytesize2 = image2.get_bytesize()

    if bytesize1 != bytesize2:
        raise ValueError(f"Expecting same image format but got '\
        f' image1 {bytesize1}-bit, image2 {bytesize2}-bit")

    try:
        image1_buffer = image1.get_buffer()

    except Exception as e:
        raise ValueError("\nCannot reference image1 pixels into a C buffer; %s " % e)

    try:
        image2_buffer = image2.get_buffer()

    except Exception as e:
        raise ValueError("\nCannot reference image2 pixels into a C buffer; %s " % e)

    cdef:
        int i=0
        unsigned char * p1
        unsigned char * p2

    if len(image1_buffer) != len(image2_buffer):
        raise ValueError(f'\nBoth images must have the dimensions'\
            ' image1 {len(image1_buffer)}, image2 {len(image2_buffer)}.')

    with nogil:

        for i in prange(0, len(image1_buffer) - 1, 3, schedule=SCHEDULE, num_threads=THREADS):

            p1 = &image1_buffer[i]
            p2 = &image2_buffer[i]

            p1[0] = (p1[0] + p2[0]) if (p1[0] + p2[0]) < 255 else 255
            p1[1] = (p1[1] + p2[1]) if (p1[1] + p2[1]) < 255 else 255
            p1[2] = (p1[2] + p2[2]) if (p1[2] + p2[2]) < 255 else 255

            if bytesize1 == 4:
                p1[3] = (p1[3] + p2[3]) if (p1[3] + p2[3]) < 255 else 255



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef inline void blend_add_array(
    unsigned char [:, :, :] rgb_array1,
    const unsigned char [:, :, :] rgb_array2):

    """
    Performs additive blending of two RGB arrays in place.

    Parameters
    ----------
    rgb_array1 : unsigned char[:, :, :]
        The first RGB array. The result is stored in this array.

    rgb_array2 : unsigned char[:, :, :]
        The second RGB array, which is added to `rgb_array1`.

    Notes
    -----
    The operation clamps values to the range [0, 255] to prevent overflow.
    """

    blend_add_array_c(rgb_array1, rgb_array2)




@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef inline void blend_add_array_c(
    unsigned char [:, :, :] rgb_array1,
    const unsigned char [:, :, :] rgb_array2):

    """
    Performs additive blending of two RGB arrays in place.

    Parameters
    ----------
    rgb_array1 : unsigned char[:, :, :]
        The first RGB array. The result is stored in this array.

    rgb_array2 : unsigned char[:, :, :]
        The second RGB array, which is added to `rgb_array1`.

    Notes
    -----
    The operation clamps values to the range [0, 255] to prevent overflow.
    """

    cdef:

        unsigned int c1, c2, c3
        int i=0, j=0
        Py_ssize_t w = rgb_array1.shape[0]
        Py_ssize_t h = rgb_array1.shape[1]
        unsigned char * r
        unsigned char * g
        unsigned char * b


    with nogil:

        for j in prange(h, schedule=SCHEDULE, num_threads=THREADS):

            for i in range(w):

                r = &rgb_array1[i, j, 0]
                g = &rgb_array1[i, j, 1]
                b = &rgb_array1[i, j, 2]

                c1 = r[0] + rgb_array2[i, j, 0]
                c2 = g[0] + rgb_array2[i, j, 1]
                c3 = b[0] + rgb_array2[i, j, 2]

                r[0] = c1 if c1 < 255 else 255
                g[0] = c2 if c2 < 255 else 255
                b[0] = c3 if c3 < 255 else 255



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef inline void blend_add_bgra_inplace(
    unsigned char [::1] bgra_array1,
    const unsigned char [::1] bgra_array2,
    bint format_32 = True
    ):
    """
    Blend-adds two arrays, equivalent to `pygame.BLEND_RGB_ADD`.

    The operation is performed in-place on `bgra_array1`.
    Both arrays must have the same length and pixel format (24-bit or 32-bit).

    Parameters
    ----------
    bgra_array1 : numpy.ndarray
        A 1D array of unsigned 8-bit integers representing BGRA or BGR pixel values.
        This array serves as the destination and is modified in place.

    bgra_array2 : numpy.ndarray
        A 1D array of unsigned 8-bit integers representing BGRA or BGR pixel values.
        This array is added to `bgra_array1`.

    format_32 : bool, optional
        If `True`, the arrays are treated as 32-bit (BGRA).
        If `False`, they are treated as 24-bit (BGR).
        Default is `True`.

    Returns
    -------
    None
        This function modifies `bgra_array1` in place.

    Notes
    -----
    - This function is optimized for performance using Cython memory views.
    - Both arrays must have the same length and format.
    - The addition is performed per-channel, clamping values at 255.

    Examples
    --------
    >>> import numpy as np
    >>> bgra1 = np.array([100, 150, 200, 255, 50, 75, 100, 255], dtype=np.uint8)  # Example 32-bit pixels
    >>> bgra2 = np.array([50, 50, 50, 255, 50, 50, 50, 255], dtype=np.uint8)
    >>> blend_add_bgra_inplace_c(bgra1, bgra2)
    >>> print(bgra1)  # Values are added and clamped at 255

    """

    blend_add_bgra_inplace_c(bgra_array1, bgra_array2)



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef inline void blend_add_bgra_inplace_c(
    unsigned char [::1] bgra_array1,
    const unsigned char [::1] bgra_array2,
    bint format_32 = True
    ):

    """
    Blend-adds two arrays, equivalent to `pygame.BLEND_RGB_ADD`.

    The operation is performed in-place on `bgra_array1`.
    Both arrays must have the same length and pixel format (24-bit or 32-bit).

    Parameters
    ----------
    bgra_array1 : numpy.ndarray
        A 1D array of unsigned 8-bit integers representing BGRA or BGR pixel values.
        This array serves as the destination and is modified in place.

    bgra_array2 : numpy.ndarray
        A 1D array of unsigned 8-bit integers representing BGRA or BGR pixel values.
        This array is added to `bgra_array1`.

    format_32 : bool, optional
        If `True`, the arrays are treated as 32-bit (BGRA).
        If `False`, they are treated as 24-bit (BGR).
        Default is `True`.

    Returns
    -------
    None
        This function modifies `bgra_array1` in place.

    Notes
    -----
    - This function is optimized for performance using Cython memory views.
    - Both arrays must have the same length and format.
    - The addition is performed per-channel, clamping values at 255.

    Examples
    --------
    >>> import numpy as np
    >>> bgra1 = np.array([100, 150, 200, 255, 50, 75, 100, 255], dtype=np.uint8)  # Example 32-bit pixels
    >>> bgra2 = np.array([50, 50, 50, 255, 50, 50, 50, 255], dtype=np.uint8)
    >>> blend_add_bgra_inplace_c(bgra1, bgra2)
    >>> print(bgra1)  # Values are added and clamped at 255

    """


    cdef:
        int i=0
        unsigned char * p1
        const unsigned char * p2

    if len(<object>bgra_array1) != len(<object>bgra_array2):
        raise ValueError(
            f"Both input arrays must have the same length,"\
            f" got bgra_array1 {len(<object>bgra_array1)} and "\
            f"bgra_array2 {len(<object>bgra_array2)}.")

    with nogil:

        for i in prange(0, len(bgra_array1) - 1, 4 if format_32 else 3,
            schedule=SCHEDULE, num_threads=THREADS):

            p1 = &bgra_array1[i]
            p2 = &bgra_array2[i]

            p1[0] = (p1[0] + p2[0]) if (p1[0] + p2[0]) < 255 else 255
            p1[1] = (p1[1] + p2[1]) if (p1[1] + p2[1]) < 255 else 255
            p1[2] = (p1[2] + p2[2]) if (p1[2] + p2[2]) < 255 else 255

            if format_32:
                p1[3] = (p1[3] + p2[3]) if (p1[3] + p2[3]) < 255 else 255





@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef inline void blend_add_alpha(
    unsigned char [:, :] alpha_array1,
    const unsigned char [:, :] alpha_array2
):
    """

    Python wrapper for additive blending of two alpha images.

    This function blends two 2D arrays, typically representing alpha channels,
    by adding their pixel values element-wise. The result is stored in `alpha_array1`.

    Parameters
    ----------
    alpha_array1 : unsigned char[:, :]
        The first image (2D array) representing alpha values. This array is modified in place.

    alpha_array2 : unsigned char[:, :]
        The second image (2D array) added to `alpha_array1`.

    Notes
    -----
    - Both arrays must have the same dimensions.
    - Values are clamped to the range [0, 255] to prevent overflow.
    - Calls `blend_add_alpha_c` for optimized performance.
    """

    blend_add_alpha_c(alpha_array1, alpha_array2)




@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef inline void blend_add_alpha_c(
    unsigned char [:, :] alpha_array1,
    const unsigned char [:, :] alpha_array2
):
    """
    Python wrapper for additive blending of two alpha images.

    This function blends two 2D arrays, typically representing alpha channels,
    by adding their pixel values element-wise. The result is stored in `alpha_array1`.

    Parameters
    ----------
    alpha_array1 : unsigned char[:, :]
        The first image (2D array) representing alpha values. This array is modified in place.

    alpha_array2 : unsigned char[:, :]
        The second image (2D array) added to `alpha_array1`.

    Raises
    ------
    ValueError
        If `alpha_array1` and `alpha_array2` have different dimensions.

    Notes
    -----
    - Both arrays must have the same dimensions.
    - Values are clamped to the range [0, 255] to prevent overflow.
    - This function runs in parallel using OpenMP for performance optimization.
    - The operation is performed in a no-GIL context for efficiency.
    """

    cdef:
        int i = 0, j = 0  # Loop indices
        Py_ssize_t w = alpha_array1.shape[0]  # Width of the first array
        Py_ssize_t h = alpha_array1.shape[1]  # Height of the first array
        Py_ssize_t w1 = alpha_array2.shape[0]  # Width of the second array
        Py_ssize_t h1 = alpha_array2.shape[1]  # Height of the second array
        Py_ssize_t alpha  # Temporary variable to store blended value

    # Ensure both arrays have the same dimensions
    if w != w1 or h != h1:
        raise ValueError(
            f"Both 2D arrays must have the same dimensions, "
            f"but got ({w}, {h}) and ({w1}, {h1})."
        )

    # Execute in a no-GIL (Global Interpreter Lock) context for performance
    with nogil:

        # Parallelize across image height for better performance
        for j in prange(h, schedule=SCHEDULE, num_threads=THREADS):

            for i in range(w):

                # Compute additive blending
                alpha = alpha_array1[i, j] + alpha_array2[i, j]

                # Clamp values to the valid range [0, 255] to prevent overflow
                alpha_array1[i, j] = alpha if alpha < 255 else 255





@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef inline void blend_sub_surface(image1, image2):

    """
    Perform per-pixel subtractive blending between two surfaces.

    This function subtracts the pixel values of `image2` from `image1`,
    modifying `image1` in place. The operation is performed element-wise
    on the RGB channels, with values clamped to the range [0, 255].

    Parameters
    ----------
    image1 : pygame.Surface
        The first image (destination surface), modified in place.

    image2 : pygame.Surface
        The second image (source surface) to be subtracted from `image1`.

    Raises
    ------
    ValueError
        If the surfaces cannot be referenced as 3D arrays.
        If the surfaces have mismatched dimensions.
        If the surface have different pixel format.

    Notes
    -----
    - Both surfaces must have the same dimensions.
    - Only RGB channels are processed; alpha channels are ignored.
    - Uses OpenMP for parallel execution to improve performance.

    """

    if image1.get_bytesize() != image2.get_bytesize():
        raise ValueError(f"Both image must be same format, "\
            f"got image1 {image1.get_bitsize()}-bit, image2 {image2.get_bitsize()}-bit.")

    cdef:
        unsigned char[:, :, :] rgb_array1  # View of the first image's RGB pixel data
        unsigned char[:, :, :] rgb_array2  # View of the second image's RGB pixel data

    # Attempt to obtain 3D array views of the surfaces' pixel data
    try:
        rgb_array1 = image1.get_view('3')

    except Exception as e:
        raise ValueError(f"Cannot reference image1 pixels into a 3D array: {e}")

    try:
        rgb_array2 = image2.get_view('3')

    except Exception as e:
        raise ValueError(f"Cannot reference image2 pixels into a 3D array: {e}")

    cdef:
        Py_ssize_t w, h, w1, h1  # Image dimensions

    # Get the dimensions of both images
    w, h = (<object>rgb_array1).shape[:2]
    w1, h1 = (<object>rgb_array2).shape[:2]

    # Ensure both images have the same dimensions
    if w != w1 or h != h1:
        raise ValueError(
            f"Both arrays must have the same dimensions, "
            f"got rgb_array1({w}, {h}) and rgb_array2({w1}, {h1})."
        )

    blend_sub_array_c(w, h, rgb_array1, rgb_array2)



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef inline void blend_sub_array(
    const Py_ssize_t w,
    const Py_ssize_t h,
    unsigned char [:, :, :] rgb_array1,
    const unsigned char [:, :, :] rgb_array2):

    """
    Perform per-pixel subtractive blending between two RGB arrays.

    This function subtracts the pixel values of `rgb_array2` from `rgb_array1`, modifying
    `rgb_array1` in place. The operation is performed element-wise on the RGB channels,
    clamping values to the range [0, 255] to prevent underflow.

    Parameters
    ----------
    w : Py_ssize_t
        The width of the arrays.

    h : Py_ssize_t
        The height of the arrays.

    rgb_array1 : unsigned char[:, :, :]
        The first RGB image array (destination). This array is modified in place.

    rgb_array2 : unsigned char[:, :, :]
        The second RGB image array (source) to be subtracted from `rgb_array1`.

    Notes
    -----
    - Both arrays must have the same dimensions `(w, h, 3)`, where the last dimension
      represents the RGB channels.
    - This function operates in parallel using OpenMP (`prange`) to improve performance.
    - Only RGB channels are processed; alpha channels, if present, are ignored.
    - This function must be called within a context where `nogil` is used.
    """

    blend_sub_array_c(w, h, rgb_array1, rgb_array2)



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef inline void blend_sub_array_c(
    const Py_ssize_t w,
    const Py_ssize_t h,
    unsigned char [:, :, :] rgb_array1,
    const unsigned char [:, :, :] rgb_array2):

    """
    Perform per-pixel subtractive blending between two RGB arrays.

    This function subtracts the pixel values of `rgb_array2` from `rgb_array1`, modifying
    `rgb_array1` in place. The operation is performed element-wise on the RGB channels,
    clamping values to the range [0, 255] to prevent underflow.

    Parameters
    ----------
    w : Py_ssize_t
        The width of the arrays.

    h : Py_ssize_t
        The height of the arrays.

    rgb_array1 : unsigned char[:, :, :]
        The first RGB image array (destination). This array is modified in place.

    rgb_array2 : unsigned char[:, :, :]
        The second RGB image array (source) to be subtracted from `rgb_array1`.

    Notes
    -----
    - Both arrays must have the same dimensions `(w, h, 3)`, where the last dimension
      represents the RGB channels.
    - This function operates in parallel using OpenMP (`prange`) to improve performance.
    - Only RGB channels are processed; alpha channels, if present, are ignored.
    - This function must be called within a context where `nogil` is used.
    """

    cdef:

        int c1, c2, c3
        int i=0, j=0
        unsigned char * r
        unsigned char * g
        unsigned char * b

    with nogil:

        for j in prange(h, schedule=SCHEDULE, num_threads=THREADS):

            for i in range(w):

                # Get pointers to the current pixel's RGB values in image1
                r = &rgb_array1[i, j, 0]
                g = &rgb_array1[i, j, 1]
                b = &rgb_array1[i, j, 2]

                # Perform subtraction and clamp values to [0, 255]
                c1 = r[0] - rgb_array2[i, j, 0]
                c2 = g[0] - rgb_array2[i, j, 1]
                c3 = b[0] - rgb_array2[i, j, 2]

                r[0] = c1 if c1 > 0 else 0
                g[0] = c2 if c2 > 0 else 0
                b[0] = c3 if c3 > 0 else 0



cpdef inline void blend_sub_bgra_inplace(
    unsigned char [::1] bgra_array1,
    const unsigned char [::1] bgra_array2,
    bint format_32 = True
    ):

    """
    Python hook function for in-place per-pixel subtraction blending on two BGRA arrays.

    This function subtracts pixel values of `bgra_array2` from `bgra_array1`, clamping
    the results to the range [0, 255]. It acts as a wrapper for the C-level function
    `blend_sub_bgra_inplace_c`.

    Parameters
    ----------
    bgra_array1 : unsigned char[::1]
        The first image array (destination). This array is modified in place.

    bgra_array2 : unsigned char[::1]
        The second image array (source), subtracted from `bgra_array1`.

    format_32 : bool, default=True
        If `True`, assumes 32-bit BGRA format (4 channels per pixel).
        If `False`, assumes 24-bit BGR format (3 channels per pixel).

    Notes
    -----
    - This function serves as the Python hook for `blend_sub_bgra_inplace_c`.
    - Both arrays must have the same length.
    - Calls `blend_sub_bgra_inplace_c` for optimized C-level processing.
    """

    blend_sub_bgra_inplace_c(bgra_array1, bgra_array2, format_32)


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef inline void blend_sub_bgra_inplace_c(
    unsigned char [::1] bgra_array1,
    const unsigned char [::1] bgra_array2,
    bint format_32 = True
    ):

    """
    Perform per-pixel subtraction blending on two BGRA arrays in place.

    This function modifies `bgra_array1` by subtracting the corresponding pixel values
    from `bgra_array2`, clamping the results to the range [0, 255]. The operation is
    performed in parallel using OpenMP.

    Parameters
    ----------
    bgra_array1 : unsigned char[::1]
        The first image array (destination). This array is modified in place.

    bgra_array2 : unsigned char[::1]
        The second image array (source), subtracted from `bgra_array1`.

    format_32 : bool, default=True
        If `True`, assumes 32-bit BGRA format (4 channels per pixel).
        If `False`, assumes 24-bit BGR format (3 channels per pixel).

    Raises
    ------
    ValueError
        If the input arrays do not have the same length.

    Notes
    -----
    - Both arrays must have the same length.
    - Pixels are processed in steps of 4 (BGRA) or 3 (BGR) based on `format_32`.
    - The subtraction is performed element-wise for the blue, green, and red channels.
    - If `format_32` is `True`, the alpha channel is also processed.
    - The function operates in parallel using OpenMP for better performance.
    """


    cdef:
        int i=0
        unsigned char * p1
        const unsigned char * p2
        int c1, c2, c3

    if len(<object>bgra_array1) != len(<object>bgra_array2):
        raise ValueError(
            f"Both input arrays must have the same length,"\
            f" got bgra_array1 {len(<object>bgra_array1)} and "\
            f"bgra_array2 {len(<object>bgra_array2)}.")

    with nogil:

        for i in prange(0, len(bgra_array1) - 1, 4 if format_32 else 3,
            schedule=SCHEDULE, num_threads=THREADS):

            p1 = &bgra_array1[i]
            p2 = &bgra_array2[i]
            c1 = p1[0] - p2[0]
            c2 = p1[1] - p2[1]
            c3 = p1[2] - p2[2]
            p1[0] = c1 if c1 > 0 else 0
            p1[1] = c2 if c2 > 0 else 0
            p1[2] = c3 if c3 > 0 else 0

            if format_32:
                p1[3] = (p1[3] - p2[3]) if (p1[3] - p2[3]) > 0 else 0




@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef inline void blend_min_surface(image1, image2):

    """
    Perform per-pixel minimum blending between two pygame surfaces.

    This function blends two surfaces by setting each pixel's RGB value in `image1`
    to the minimum of the corresponding values in `image1` and `image2`. The operation
    is performed in place on `image1`.

    Parameters
    ----------
    image1 : pygame.Surface
        The first surface (destination). This surface is modified in place.

    image2 : pygame.Surface
        The second surface (source), compared to `image1` for the minimum value.

    Raises
    ------
    ValueError
        - If the surfaces have different bit depths.
        - If the surfaces cannot be referenced as 3D arrays.
        - If the surfaces have different dimensions.

    Notes
    -----
    - Both surfaces must have the same format (bit depth).
    - Both surfaces must have the same dimensions.
    - This function internally calls `blend_min_array_c` for optimized performance.
    """


    if image1.get_bytesize() != image2.get_bytesize():
        raise ValueError(f"Both image must be same format, "\
            f"got image1 {image1.get_bitsize()}-bit, image2 {image2.get_bitsize()}-bit.")

    cdef:
        unsigned char[:, :, :] rgb_array1
        unsigned char[:, :, :] rgb_array2

    try:
        rgb_array1 = image1.get_view('3')

    except Exception as e:
        raise ValueError("\nCannot reference image1 pixels into a 3d array.\n %s " % e)

    try:
        rgb_array2 = image2.get_view('3')

    except Exception as e:
        raise ValueError("\nCannot reference image2 pixels into a 3d array.\n %s " % e)

    cdef:
        Py_ssize_t w, h, w1, h1

    # Get the dimensions of both images
    w, h = (<object>rgb_array1).shape[:2]
    w1, h1 = (<object>rgb_array2).shape[:2]

    # Ensure both images have the same dimensions
    if w != w1 or h != h1:
        raise ValueError(
            f"Both arrays must have the same dimensions, "
            f"got rgb_array1({w}, {h}) and rgb_array2({w1}, {h1})."
        )

    blend_min_array_c(w, h, rgb_array1, rgb_array2)




@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef inline void blend_min_array(
    const Py_ssize_t w,
    const Py_ssize_t h,
    unsigned char [:, :, :] rgb_array1,
    const unsigned char [:, :, :] rgb_array2
    ):

    """
    Perform per-pixel minimum blending between two RGB arrays.

    This function modifies `rgb_array1` by setting each pixel's RGB value to the minimum
    of the corresponding values in `rgb_array1` and `rgb_array2`. The operation is
    performed element-wise and in parallel for better performance.

    Parameters
    ----------
    w : Py_ssize_t
        The width of the arrays.

    h : Py_ssize_t
        The height of the arrays.

    rgb_array1 : unsigned char[:, :, :]
        The first RGB image array (destination). This array is modified in place.

    rgb_array2 : unsigned char[:, :, :]
        The second RGB image array (source), compared to `rgb_array1` for the minimum value.

    Notes
    -----
    - Both arrays must have the same dimensions `(w, h, 3)`, where the last dimension
      represents the RGB channels.
    - This function operates in parallel using OpenMP (`prange`) to improve performance.
    - Only RGB channels are processed; alpha channels, if present, are ignored.
    - This function must be called within a context where `nogil` is used.
    """

    blend_min_array_c(w, h, rgb_array1, rgb_array2)


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef inline void blend_min_array_c(
    const Py_ssize_t w,
    const Py_ssize_t h,
    unsigned char[:, :, :] rgb_array1,
    const unsigned char[:, :, :] rgb_array2
):
    """
    Perform per-pixel minimum blending between two RGB arrays.

    This function modifies `rgb_array1` by setting each pixel's RGB value to the minimum
    of the corresponding values in `rgb_array1` and `rgb_array2`. The operation is
    performed element-wise and in parallel for better performance.

    Parameters
    ----------
    w : Py_ssize_t
        The width of the arrays.

    h : Py_ssize_t
        The height of the arrays.

    rgb_array1 : unsigned char[:, :, :]
        The first RGB image array (destination). This array is modified in place.

    rgb_array2 : unsigned char[:, :, :]
        The second RGB image array (source), compared to `rgb_array1` for the minimum value.

    Notes
    -----
    - Both arrays must have the same dimensions `(w, h, 3)`, where the last dimension
      represents the RGB channels.
    - This function operates in parallel using OpenMP (`prange`) to improve performance.
    - Only RGB channels are processed; alpha channels, if present, are ignored.
    - This function must be called within a context where `nogil` is used.
    """

    cdef:
        int i = 0, j = 0  # Loop variables

        # Pointers to pixel values in rgb_array1 (destination)
        unsigned char *r
        unsigned char *g
        unsigned char *b

        # Pointers to pixel values in rgb_array2 (source)
        const unsigned char *rr
        const unsigned char *gg
        const unsigned char *bb

    with nogil:
        # Parallel loop over image height (rows)
        for j in prange(h, schedule=SCHEDULE, num_threads=THREADS):
            # Iterate over image width (columns)
            for i in range(w):
                # Get pointers to the current pixel's RGB values in image1
                r = &rgb_array1[i, j, 0]
                g = &rgb_array1[i, j, 1]
                b = &rgb_array1[i, j, 2]

                # Get pointers to the corresponding pixel's RGB values in image2
                rr = &rgb_array2[i, j, 0]
                gg = &rgb_array2[i, j, 1]
                bb = &rgb_array2[i, j, 2]

                # Set each channel to the minimum value between the two images
                r[0] = r[0] if r[0] < rr[0] else rr[0]
                g[0] = g[0] if g[0] < gg[0] else gg[0]
                b[0] = b[0] if b[0] < bb[0] else bb[0]






cpdef inline void blend_min_bgra_inplace(
    unsigned char [::1] bgra_array1,
    const unsigned char [::1] bgra_array2,
    bint format_32 = True
):

    """
    Python hook function for in-place per-pixel minimum blending on two BGRA arrays.

    This function serves as a Python interface to the optimized C-level function
    `blend_min_bgra_inplace_c`. It modifies the first BGRA array (`bgra_array1`)
    by taking the per-channel minimum between corresponding pixels in `bgra_array1`
    and `bgra_array2`.

    The operation is performed in-place and is optimized for both 24-bit and 32-bit
    BGRA formats.

    Parameters
    ----------
    bgra_array1 : unsigned char[::1]
        The first image array (destination). This array is modified in place.

    bgra_array2 : unsigned char[::1]
        The second image array (source), compared with `bgra_array1` to determine
        the minimum pixel values.

    format_32 : bool, default=True
        If `True`, assumes 32-bit BGRA format (4 channels per pixel).
        If `False`, assumes 24-bit BGR format (3 channels per pixel).

    Notes
    -----
    - This function acts as a Python wrapper for `blend_min_bgra_inplace_c`.
    - Both input arrays must have the same length.
    - The function operates in parallel using OpenMP for better performance.
    - The blending operation is performed in-place on `bgra_array1`.
    """

    blend_min_bgra_inplace_c(bgra_array1, bgra_array2, format_32)



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef inline void blend_min_bgra_inplace_c(
    unsigned char [::1] bgra_array1,
    const unsigned char [::1] bgra_array2,
    bint format_32 = True
    ):

    """
    Perform in-place per-pixel minimum blending on two BGRA arrays.

    This function modifies `bgra_array1` by taking the per-channel minimum
    between corresponding pixels in `bgra_array1` and `bgra_array2`. The
    operation is performed in parallel using OpenMP.

    Parameters
    ----------
    bgra_array1 : unsigned char[::1]
        The first image array (destination). This array is modified in place.

    bgra_array2 : unsigned char[::1]
        The second image array (source), compared with `bgra_array1` to determine
        the minimum pixel values.

    format_32 : bool, default=True
        If `True`, assumes 32-bit BGRA format (4 channels per pixel).
        If `False`, assumes 24-bit BGR format (3 channels per pixel).

    Raises
    ------
    ValueError
        If the input arrays do not have the same length.

    Notes
    -----
    - Both arrays must have the same length.
    - Pixels are processed in steps of 4 (BGRA) or 3 (BGR) based on `format_32`.
    - The function operates in parallel using OpenMP for better performance.
    - Calls `blend_min_bgra_inplace_c` for optimized C-level processing.
    """


    cdef:
        int i=0
        unsigned char * p1
        const unsigned char * p2

    if len(<object>bgra_array1) != len(<object>bgra_array2):
        raise ValueError(
            f"Both input arrays must have the same length,"\
            f" got bgra_array1 {len(<object>bgra_array1)} and "\
            f"bgra_array2 {len(<object>bgra_array2)}.")

    with nogil:

        for i in prange(0, len(bgra_array1) - 1, 4 if format_32 else 3,
            schedule=SCHEDULE, num_threads=THREADS):

            p1 = &bgra_array1[i]
            p2 = &bgra_array2[i]

            p1[0] = p1[0] if p1[0] < p2[0] else p2[0]
            p1[1] = p1[1] if p1[1] < p2[1] else p2[1]
            p1[2] = p1[2] if p1[2] < p2[2] else p2[2]

            if format_32:
                p1[3] = p1[3] if p1[3] < p2[3] else p2[3]





@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef inline void blend_max_surface(image1, image2):

    """
    Perform in-place blending of two surfaces by taking the per-channel maximum
    between corresponding pixels.

    This function compares the pixel values of two surfaces and updates the pixels
    of the first image (`image1`) with the maximum value for each channel (RGB).

    The operation is performed in-place, and both images must have the same dimensions
    and format. This function uses an optimized C-level function `blend_max_array_c`
    for the blending operation.

    Parameters
    ----------
    image1 : pygame.Surface
        The first image surface, which will be modified in place.

    image2 : pygame.Surface
        The second image surface, whose pixel values are compared against `image1`
        for the blending operation.

    Raises
    ------
    ValueError
        If the images do not have the same format (byte size) or dimensions.

    """

    cdef:
        unsigned char[:, :, :] rgb_array1
        unsigned char[:, :, :] rgb_array2
        Py_ssize_t bytesize1 = image1.get_bytesize()
        Py_ssize_t bytesize2 = image2.get_bytesize()

    # Check that both images have the same format (bytesize)
    if bytesize1 != bytesize2:
        raise ValueError(
            f"Expecting same image format but got "
            f"image1 {bytesize1}-bit, image2 {bytesize2}-bit"
        )

    try:
        # Get the RGB pixel data for image1
        rgb_array1 = image1.get_view('3')

    except Exception as e:
        raise ValueError("\nCannot reference image1 pixels into a 3d array.\n %s " % e)

    try:
        # Get the RGB pixel data for image2
        rgb_array2 = image2.get_view('3')
    except Exception as e:
        raise ValueError("\nCannot reference image2 pixels into a 3d array.\n %s " % e)

    cdef:
        Py_ssize_t w, h, w1, h1

    # Get the dimensions of both images
    w, h = (<object>rgb_array1).shape[:2]
    w1, h1 = (<object>rgb_array2).shape[:2]

    # Ensure both images have the same dimensions
    if w != w1 or h != h1:
        raise ValueError(
            f"Both arrays must have the same dimensions, "
            f"got rgb_array1({w}, {h}) and rgb_array2({w1}, {h1})."
        )

    # Perform the blending using the optimized C function
    blend_max_array_c(w, h, rgb_array1, rgb_array2)



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef inline void blend_max_array(
    const Py_ssize_t w,
    const Py_ssize_t h,
    unsigned char [:, :, :] rgb_array1,
    const unsigned char [:, :, :] rgb_array2
):

    """
    Perform per-channel maximum blending of two RGB arrays
    and store the result in `rgb_array1`.

    This function compares the pixel values of two RGB arrays (`rgb_array1` and `rgb_array2`)
    and updates `rgb_array1` in-place with the maximum value of the corresponding channels (RGB).

    The operation is carried out for all pixels in the arrays, where each channel (Red, Green, Blue)
    of `rgb_array1` is replaced with the maximum of the corresponding channels from both arrays.

    Parameters
    ----------
    w : int
        The width of the images (arrays).

    h : int
        The height of the images (arrays).

    rgb_array1 : unsigned char[:, :, :]
        The first RGB image array (destination). This array will be modified in place.

    rgb_array2 : unsigned char[:, :, :]
        The second RGB image array (source). Its pixel values will be compared with `rgb_array1`
        to determine the per-channel maximum.

    Notes
    -----
    - The dimensions of `rgb_array1` and `rgb_array2` must be the same.
    - This function calls the optimized C function `blend_max_array_c` for efficient blending.

    See Also
    --------
    blend_max_array_c : C function that performs the actual blending.

    """

    blend_max_array_c(w, h, rgb_array1, rgb_array2)




@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef inline void blend_max_array_c(
    const Py_ssize_t w,
    const Py_ssize_t h,
    unsigned char [:, :, :] rgb_array1,
    const unsigned char [:, :, :] rgb_array2
):

    """
    Perform per-channel maximum blending of two RGB arrays
     and update the first array in place.

    This function compares the pixel values of two RGB arrays (`rgb_array1` and `rgb_array2`)
    and updates `rgb_array1` with the maximum value between the corresponding channels (RGB)
    from the two arrays.

    The blending is performed for each pixel in the arrays, modifying the RGB channels of `rgb_array1`
    to hold the maximum value of each corresponding channel from both arrays.

    Parameters
    ----------
    w : int
        The width of the image arrays (number of pixels along the horizontal axis).

    h : int
        The height of the image arrays (number of pixels along the vertical axis).

    rgb_array1 : unsigned char[:, :, :]
        The first RGB image array (destination). This array will be modified in place with the maximum
        channel values.

    rgb_array2 : unsigned char[:, :, :]
        The second RGB image array (source). The channel values will be compared with the values from
        `rgb_array1` to determine the per-channel maximum.

    Notes
    -----
    - Both arrays must have the same dimensions.
    - This function operates in a multi-threaded context for improved performance, processing pixel
      data concurrently across threads.

    Example
    -------
    Given two images represented as RGB arrays, `rgb_array1` and `rgb_array2`, the function will
    update `rgb_array1` such that each pixel’s RGB value in `rgb_array1` becomes the maximum of
    the corresponding pixel's RGB values from both arrays.

    See Also
    --------
    blend_max_array : Python wrapper function that invokes this C function.

    """

    cdef:
        int i = 0, j = 0
        unsigned char * r
        unsigned char * g
        unsigned char * b
        const unsigned char * rr
        const unsigned char * gg
        const unsigned char * bb

    with nogil:
        # Loop through the pixels of both arrays to blend the channels.
        for j in prange(h, schedule=SCHEDULE, num_threads=THREADS):
            for i in range(w):

                # Get pointers to the current pixel's RGB values in both images.
                r = &rgb_array1[i, j, 0]
                g = &rgb_array1[i, j, 1]
                b = &rgb_array1[i, j, 2]

                rr = &rgb_array2[i, j, 0]
                gg = &rgb_array2[i, j, 1]
                bb = &rgb_array2[i, j, 2]

                # Update the RGB channels with the maximum value from both arrays.
                r[0] = r[0] if r[0] > rr[0] else rr[0]
                g[0] = g[0] if g[0] > gg[0] else gg[0]
                b[0] = b[0] if b[0] > bb[0] else bb[0]




cpdef inline void blend_max_bgra_inplace(
    unsigned char [::1] bgra_array1,
    const unsigned char [::1] bgra_array2,
    bint format_32 = True
):

    """
    Blends two BGRA (or BGR) arrays in-place by taking the maximum value
    for each channel at each pixel.

    Parameters
    ----------
    bgra_array1 : unsigned char[::1]
        The first BGRA/BGR array, modified in place.

    bgra_array2 : const unsigned char[::1]
        The second BGRA/BGR array, used for comparison.

    format_32 : bool, optional (default=True)
        If `True`, the arrays are assumed to be in BGRA format (4 channels per pixel).
        If `False`, the arrays are assumed to be in BGR format (3 channels per pixel).

    Raises
    ------
    ValueError
        If `bgra_array1` and `bgra_array2` have different lengths.

    Notes
    -----
    - This function runs in parallel using OpenMP with `prange`, and
      modifications occur within a `nogil` block for performance.
    - The operation performed is:

      .. code-block:: python

          for each pixel:
              bgra_array1[channel] = max(bgra_array1[channel], bgra_array2[channel])
    """

    blend_max_bgra_inplace_c(bgra_array1, bgra_array2, format_32)



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef inline void blend_max_bgra_inplace_c(
    unsigned char [::1] bgra_array1,
    const unsigned char [::1] bgra_array2,
    bint format_32 = True
    ):


    """
    Blends two BGRA (or BGR) arrays in-place by taking the maximum value
    for each channel at each pixel.

    Parameters
    ----------
    bgra_array1 : unsigned char[::1]
        The first BGRA/BGR array, modified in place.

    bgra_array2 : const unsigned char[::1]
        The second BGRA/BGR array, used for comparison.

    format_32 : bool, optional (default=True)
        If `True`, the arrays are assumed to be in BGRA format (4 channels per pixel).
        If `False`, the arrays are assumed to be in BGR format (3 channels per pixel).

    Raises
    ------
    ValueError
        If `bgra_array1` and `bgra_array2` have different lengths.

    Notes
    -----
    - This function runs in parallel using OpenMP with `prange`, and
      modifications occur within a `nogil` block for performance.
    - The operation performed is:

      .. code-block:: python

          for each pixel:
              bgra_array1[channel] = max(bgra_array1[channel], bgra_array2[channel])
    """

    cdef:
        int i=0
        unsigned char * p1
        const unsigned char * p2

    if len(<object>bgra_array1) != len(<object>bgra_array2):
        raise ValueError(
            f"Both input arrays must have the same length,"\
            f" got bgra_array1 {len(<object>bgra_array1)} and "\
            f"bgra_array2 {len(<object>bgra_array2)}.")

    with nogil:

        for i in prange(0, len(bgra_array1) - 1, 4 if format_32 else 3,
            schedule=SCHEDULE, num_threads=THREADS):

            p1 = &bgra_array1[i]
            p2 = &bgra_array2[i]

            p1[0] = p1[0] if p1[0] > p2[0] else p2[0]
            p1[1] = p1[1] if p1[1] > p2[1] else p2[1]
            p1[2] = p1[2] if p1[2] > p2[2] else p2[2]

            if format_32:
                p1[3] = p1[3] if p1[3] > p2[3] else p2[3]







@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef inline void blend_max_alpha(
    unsigned char [:, :] alpha1_array,
    unsigned char [:, :] alpha2_array
    ):

    """
    Blends two alpha arrays in-place by taking the maximum alpha value
    at each pixel.

    Parameters
    ----------
    alpha1_array : unsigned char[:, :]
        The first alpha array, which will be modified in place.

    alpha2_array : unsigned char[:, :]
        The second alpha array, used as the source for comparison.

    Raises
    ------
    ValueError
        If `alpha1_array` and `alpha2_array` have different dimensions.

    Notes
    -----
    This function runs in parallel using OpenMP with `prange`, and
    modifications occur within a `nogil` block for performance.

    The operation performed is:

    .. code-block:: python

        alpha1_array[i, j] = max(alpha1_array[i, j], alpha2_array[i, j])
    """

    blend_max_alpha_c(alpha1_array, alpha2_array)


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef inline void blend_max_alpha_c(
    unsigned char [:, :] alpha1_array,
    unsigned char [:, :] alpha2_array
    ):

    """
    Blends two alpha arrays in-place by taking the maximum alpha value
    at each pixel.

    Parameters
    ----------
    alpha1_array : unsigned char[:, :]
        The first alpha array, which will be modified in place.

    alpha2_array : unsigned char[:, :]
        The second alpha array, used as the source for comparison.

    Raises
    ------
    ValueError
        If `alpha1_array` and `alpha2_array` have different dimensions.

    Notes
    -----
    This function runs in parallel using OpenMP with `prange`, and
    modifications occur within a `nogil` block for performance.

    The operation performed is:

    .. code-block:: python

        alpha1_array[i, j] = max(alpha1_array[i, j], alpha2_array[i, j])
    """

    cdef:
        Py_ssize_t w, h, w1, h1

    w, h = alpha1_array.shape[:2]
    w1, h1 = alpha2_array.shape[:2]

    if w != w1 or h != h1:
        raise ValueError(f"both alpha arrays must have the same "\
        " dimensions alpha1 {w, h}, alpha2 {w1, h1}")

    cdef:
        int i=0, j=0
        unsigned char * p1
        unsigned char * p2

    with nogil:

        for j in prange(h, schedule=SCHEDULE, num_threads=THREADS):

            for i in range(w):

                p1 = &alpha1_array[i, j]
                p2 = &alpha2_array[i, j]
                p1[0] = p1[0] if p1[0] > p2[0] else p2[0]




@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef inline void blend_min_alpha(
    unsigned char [:, :] alpha1_array,
    unsigned char [:, :] alpha2_array
    ):

    """
    Blends two alpha arrays in-place by taking the minimum alpha value
    at each pixel.

    Parameters
    ----------
    alpha1_array : unsigned char[:, :]
        The first alpha array, which will be modified in place.

    alpha2_array : unsigned char[:, :]
        The second alpha array, used as the source for comparison.

    Raises
    ------
    ValueError
        If `alpha1_array` and `alpha2_array` have different dimensions.

    Notes
    -----
    This function runs in parallel using OpenMP with `prange`, and
    modifications occur within a `nogil` block for performance.

    The operation performed is:

    .. code-block:: python

        alpha1_array[i, j] = min(alpha1_array[i, j], alpha2_array[i, j])
    """

    blend_min_alpha_c(alpha1_array, alpha2_array)


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef inline void blend_min_alpha_c(
    unsigned char [:, :] alpha1_array,
    unsigned char [:, :] alpha2_array
    ):

    """
    Blends two alpha arrays in-place by taking the minimum alpha value
    at each pixel.

    Parameters
    ----------
    alpha1_array : unsigned char[:, :]
        The first alpha array, which will be modified in place.

    alpha2_array : unsigned char[:, :]
        The second alpha array, used as the source for comparison.

    Raises
    ------
    ValueError
        If `alpha1_array` and `alpha2_array` have different dimensions.

    Notes
    -----
    This function runs in parallel using OpenMP with `prange`, and
    modifications occur within a `nogil` block for performance.

    The operation performed is:

    .. code-block:: python

        alpha1_array[i, j] = min(alpha1_array[i, j], alpha2_array[i, j])
    """


    cdef:
        Py_ssize_t w, h, w1, h1

    w, h = alpha1_array.shape[:2]
    w1, h1 = alpha2_array.shape[:2]

    if w != w1 or h != h1:
        raise ValueError(f"both alpha arrays must have the same "\
        " dimensions alpha1 {w, h}, alpha2 {w1, h1}")

    cdef:
        int i=0, j=0
        unsigned char * p1
        unsigned char * p2

    with nogil:

        for j in prange(h, schedule=SCHEDULE, num_threads=THREADS):

            for i in range(w):

                p1 = &alpha1_array[i, j]
                p2 = &alpha2_array[i, j]
                p1[0] = p1[0] if p1[0] < p2[0] else p2[0]




