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


import numpy
cimport numpy as np

DEF SCHEDULE = 'static'
from PygameShader.config import OPENMP, THREAD_NUMBER, __VERSION__

cdef int THREADS = 1

if OPENMP:
     THREADS = THREAD_NUMBER



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cpdef swap_channels24_c(surface_, model):
    """
    THIS PLUGIN ALLOW YOU TO SWAP CHANNEL OF AN IMAGE 
    
    :param surface_: pygame.Surface
    :param model: python string; String representing the channel order e.g
    RGB, RBG, GRB, GBR, BRG, BGR etc. letters can also be replaced by the digit 0
    to null the entire channel. e.g : 'R0B' -> no green channel

    """

    assert PyObject_IsInstance(surface_, Surface), \
           'Expecting Surface for argument surface_ got %s ' % type(surface_)

    if len(model) != 3:
        print("\nArgument model is invalid.")
        raise ValueError("Choose between RGB, RBG, GRB, GBR, BRG, BGR")

    rr, gg, bb = list(model)
    order = {'R' : 0, 'G' : 1, 'B' : 2, '0': -1}

    cdef Py_ssize_t width, height
    width, height = surface_.get_size()

    try:
        rgb_ = surface_.get_view('3')
    except (pygame.error, ValueError):
        try:
            rgb_ = array3d(surface_)
        except(pygame.error, ValueError):
            raise ValueError('\nIncompatible pixel format.')

    cdef:
        unsigned char [:, :, :] rgb_array = rgb_
        unsigned char [:, :, ::1] new_array = empty((height, width, 3), dtype=uint8)
        int i=0, j=0
        short int ri, gi, bi

    ri = order[rr]
    gi = order[gg]
    bi = order[bb]

    with nogil:
        for i in prange(width, schedule=SCHEDULE, num_threads=THREADS):
            for j in range(height):
                if ri == -1:
                    new_array[j, i, 0] = 0
                else:
                    new_array[j, i, 0] = rgb_array[i, j, ri]

                if gi == -1:
                    new_array[j, i, 1] = 0
                else:
                    new_array[j, i, 1] = rgb_array[i, j, gi]

                if bi == -1:
                    new_array[j, i, 2] = 0
                else:
                    new_array[j, i, 2] = rgb_array[i, j, bi]

    return frombuffer(new_array, (width, height), 'RGB')




@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cpdef create_horizontal_gradient_1d(
        int value,
        tuple start_color=(255, 0, 0),
        tuple end_color=(0, 255, 0)
):
    """
    CREATE AN ARRAY FILLED WITH A GRADIENT COLOR (RGB) 
    
    :param value      : integer; Size of the gradient 1d width
    :param start_color: tuple; Tuple containing the starting RGB color 
    :param end_color  : tuple; Tuple containing the RGB values of the final color
    :return           : numpy.ndarray 2D array shape (w, 3) of type uint8 (unsigned char) 
     containing all the pixels 
    """

    if value <= 0:
        raise ValueError("Argument value cannot be <= 1")

    cdef:
        float [:] diff_ =  \
            numpy.array(end_color, dtype=float32) - \
            numpy.array(start_color, dtype=float32)
        float [::1] row = numpy.arange(value, dtype=float32) / (value - 1.0)
        unsigned char [:, ::1] rgb_gradient = empty((value, 3), dtype=uint8)
        float [3] start = numpy.array(start_color, dtype=float32)
        int i=0
        float * row_

    with nogil:
        for i in prange(value, schedule=SCHEDULE, num_threads=THREADS):
           row_ = &row[i]
           rgb_gradient[i, 0] = <unsigned char>(start[0] + row_[0] * diff_[0])
           rgb_gradient[i, 1] = <unsigned char>(start[1] + row_[0] * diff_[1])
           rgb_gradient[i, 2] = <unsigned char>(start[2] + row_[0] * diff_[2])

    return asarray(rgb_gradient)



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cpdef create_horizontal_gradient_1d_alpha(
        int value,
        tuple start_color=(255, 0, 0, 255),
        tuple end_color=(0, 255, 0, 0)
):
    """
    CREATE AN ARRAY FILLED WITH A GRADIENT COLOR (RGBA)

    :param value      : integer; Size of the gradient 1d width
    :param start_color: tuple; Tuple containing the starting RGB color 
    :param end_color  : tuple; Tuple containing the RGB values of the final color
    :return           : numpy.ndarray 2D array shape (w, 3) of type uint8 (unsigned char) 
     containing all the pixels 
    """

    if value <= 0:
        raise ValueError("Argument value cannot be <= 1")

    cdef:
        float [:] diff_ =  \
            numpy.array(end_color, dtype=float32) - \
            numpy.array(start_color, dtype=float32)
        float [::1] row = numpy.arange(value, dtype=float32) / (value - 1.0)
        unsigned char [:, ::1] rgba_gradient = empty((value, 4), dtype=uint8)
        float [4] start = numpy.array(start_color, dtype=float32)
        int i=0
        float * row_

    with nogil:
        for i in prange(value, schedule=SCHEDULE, num_threads=THREADS):
           row_ = &row[i]
           rgba_gradient[i, 0] = <unsigned char>(start[0] + row_[0] * diff_[0])
           rgba_gradient[i, 1] = <unsigned char>(start[1] + row_[0] * diff_[1])
           rgba_gradient[i, 2] = <unsigned char>(start[2] + row_[0] * diff_[2])
           rgba_gradient[i, 3] = <unsigned char>(start[3] + row_[0] * diff_[3])

    return asarray(rgba_gradient)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cpdef object horizontal_grad3d(
        int width,
        int height,
        tuple start_color=(255, 0, 0),
        tuple end_color=(0, 255, 0)
):
    """
    CREATE A 24-BIT GRADIENT FROM TWO COLORS

    :param width      : integer; width of the new surface in pixels
    :param height     : integer; height of the new surface in pixels
    :param start_color: tuple; Value RGB, Starting color
    :param end_color  : tuple; value RGB, ending color or final color
    :return           : Surface; pygame.Surface format 24 bit 
    size width x height 
    """
    cdef:
        float [:] diff_ =  \
            numpy.array(end_color, dtype=float32) - \
            numpy.array(start_color, dtype=float32)
        float [::1] row = numpy.arange(width, dtype=float32) / (width - 1.0)
        unsigned char [:, :, ::1] rgb_gradient = empty((height, width, 3), dtype=uint8)
        float [3] start = numpy.array(start_color, dtype=float32)
        int i=0, j=0
        float * row_

    with nogil:
        for j in prange(height, schedule=SCHEDULE, num_threads=THREADS):
            for i in range(width):
                row_ = &row[i]
                rgb_gradient[j, i, 0] = <unsigned char>(start[0] + row_[0] * diff_[0])
                rgb_gradient[j, i, 1] = <unsigned char>(start[1] + row_[0] * diff_[1])
                rgb_gradient[j, i, 2] = <unsigned char>(start[2] + row_[0] * diff_[2])

    return frombuffer(rgb_gradient, (width, height), "RGB")



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cpdef object horizontal_grad3d_alpha(
        int width,
        int height,
        tuple start_color=(255, 0, 0, 255),
        tuple end_color=(0, 255, 0, 0)
):
    """
    CREATE A 32-BIT GRADIENT WITH TRANSPARENCY FROM TWO COLORS (RGBA)

    :param width      : integer; width of the new surface in pixels
    :param height     : integer; height of the new surface in pixels
    :param start_color: tuple; Value RGB, Starting color
    :param end_color  : tuple; value RGB, ending color or final color
    :return           : Surface; pygame.Surface format 32 bit 
    size width x height 
    """
    cdef:
        float [:] diff_ =  \
            numpy.array(end_color, dtype=float32) - \
            numpy.array(start_color, dtype=float32)
        float [::1] row = numpy.arange(width, dtype=float32) / (width - 1.0)
        unsigned char [:, :, ::1] rgb_gradient = empty((height, width, 4), dtype=uint8)
        float [4] start = numpy.array(start_color, dtype=float32)
        int i=0, j=0
        float * row_

    with nogil:
        for j in prange(height, schedule=SCHEDULE, num_threads=THREADS):
            for i in range(width):
                row_ = &row[i]
                rgb_gradient[j, i, 0] = <unsigned char>(start[0] + row_[0] * diff_[0])
                rgb_gradient[j, i, 1] = <unsigned char>(start[1] + row_[0] * diff_[1])
                rgb_gradient[j, i, 2] = <unsigned char>(start[2] + row_[0] * diff_[2])
                rgb_gradient[j, i, 3] = <unsigned char> (start[2] + row_[0] * diff_[3])

    return frombuffer(rgb_gradient, (width, height), "RGBA").convert_alpha()


DEF r_max = 1.0 / 0.707106781 #inverse sqrt(0.5) or 1.0/cos45



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cpdef create_radial_gradient(
        int width_,
        int height_,
        float offset_x              = 0.5,
        float offset_y              = 0.5,
        tuple start_color_          = (255, 0, 0),
        tuple end_color_            = (0, 0, 0),
        object gradient_array_      = None,
        float factor_               = 1.4,
        unsigned short int threads_ = 8
):
    """
    CREATE A RADIAL GRADIENT (24-BIT) OPAQUE 
    
    :param width_         : integer; surface width in pixels 
    :param height_        : integer; surface height in pixels
    :param offset_x       : float; Centre of the gradient within the surface default (0.5 centre)
    :param offset_y       : float; Centre of the gradient within the surface default (0.5 centre)
    :param end_color_     : tuple; Contains the start color of the radian (RGB values), default (255, 0, 0)
    :param start_color_   : tuple; Contains the end color of the radian (RGB values), default (0, 0, 0)
    :param gradient_array_: numpy.array; gradient array shape (w, 3) containing RGB colors (uint8) 
    :param factor_        : float; Value must be > 0. Default is 1.4 
    :param threads_       : integer; concurrent threads default 8
    :return               : pygame.Surface; Return a radial gradient centre from the surface 
        origin C(Width/2, height/2), surface is 24 bit 
    """
    assert r_max != 0, "Constant r_max cannot be null"
    if factor_ <=0:
        raise ValueError("Argument factor_ cannot be <= 0.0 default is 1.4")
    assert width_ > 0, "Argument width cannot be <=0"
    assert height_ > 0, "Argument height cannot be <=0"

    cdef:
        unsigned char [:, :, ::1] rgb_array = empty((height_, width_, 3), dtype=uint8)
        float nx, ny
        float w2 = <float>width_ * <float>factor_
        float h2 = <float>height_ * <float>factor_
        float r0 = <float>sqrt(w2 * w2 + h2 * h2)

        int i, j
        unsigned int x
        float n1 = <float>1.0 / width_
        float n2 = <float>1.0 / height_
        unsigned char * r
        unsigned char * g
        unsigned char * b

    cdef unsigned short int THREADS = threads_

    cdef unsigned char [:, ::1] gradient_array__

    if gradient_array_ is None:

        gradient_array_ = create_horizontal_gradient_1d_alpha(
            <int>sqrt(width_ * width_ + (height_ * <float>0.5) * (height_ * <float>0.5)),
            start_color = start_color_,
            end_color   = end_color_
        )


    gradient_array__ = gradient_array_
    cdef unsigned int l = <object> gradient_array__.shape[0] - 1

    with nogil:
        for j in prange(height_, schedule=SCHEDULE, num_threads=THREADS):
            ny = (<float> j * n2) - <float> offset_y
            for i in range(width_):
                nx = (<float>i * n1) - <float> offset_x

                r = &rgb_array[j, i, 0]
                g = &rgb_array[j, i, 1]
                b = &rgb_array[j, i, 2]
                # position in the gradient
                x = <int>((<float>sqrt(nx * nx + ny * ny) * r0) * r_max)

                # check if the radius is greater than the size of the gradient,
                # in which case, the color is black
                if x > l:
                    r[0] = 0
                    g[0] = 0
                    b[0] = 0
                # assign the gradient
                else:
                    r[0] = <unsigned char>gradient_array__[x, 0]
                    g[0] = <unsigned char>gradient_array__[x, 1]
                    b[0] = <unsigned char>gradient_array__[x, 2]

    return frombuffer(rgb_array, (width_, height_), "RGB")



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cpdef create_radial_gradient_alpha(
        int width_,
        int height_,
        float offset_x              = 0.5,
        float offset_y              = 0.5,
        tuple start_color_          = (255, 0, 0, 255),
        tuple end_color_            = (0, 0, 0, 0),
        object gradient_array_      = None,
        float factor_               = 1.4,
        unsigned short int threads_ = 8
):
    """
    CREATE A RADIAL GRADIENT (32-bit WITH TRANSPARENCY) 

    
    :param width_         : integer; surface width in pixels 
    :param height_        : integer; surface height in pixels
    :param offset_x       : float; Centre of the gradient within the surface default (0.5 centre)
    :param offset_y       : float; Centre of the gradient within the surface default (0.5 centre)
    :param end_color_     : tuple; Contains the start color of the radian (RGBA values), default (255, 0, 0, 255)
    :param start_color_   : tuple; Contains the end color of the radian (RGBA values), default (0, 0, 0, 0)
    :param gradient_array_: numpy.array; gradient array shape (w, 3) containing RGB colors (uint8) 
    :param factor_        : float; Value must be > 0. Default is 1.4 
    :param threads_       : integer; concurrent threads default 8
    :return               : pygame.Surface; Return a radial gradient centre from the surface 
        origin C(Width/2, height/2, surface is 32 bit containing per-pixel transparency
    """

    assert r_max != 0, "Constant r_max cannot be null"
    if factor_ <=0:
        raise ValueError("Argument factor_ cannot be <= 0.0 default is 1.4")
    assert width_ > 0, "Argument width cannot be <=0"
    assert height_ > 0, "Argument height cannot be <=0"

    cdef:
        unsigned char [:, :, ::1] rgb_array = empty((height_, width_, 4), dtype=uint8)
        float nx, ny
        float w2 = <float>width_ * <float>factor_
        float h2 = <float>height_ * <float>factor_
        float r0 = <float>sqrt(w2 * w2 + h2 * h2)

        int i, j
        unsigned int x

        float n1 = <float>1.0 / width_
        float n2 = <float>1.0 / height_
        unsigned char * r
        unsigned char * g
        unsigned char * b
        unsigned char * a

    cdef unsigned short int THREADS = threads_

    cdef unsigned char [:, ::1] gradient_array__

    if gradient_array_ is None:

        gradient_array_ = create_horizontal_gradient_1d_alpha(
            <int>sqrt(width_ * width_ + (height_ * <float>0.5) * (height_ * <float>0.5)),
            start_color = start_color_,
            end_color   = end_color_
        )


    gradient_array__ = gradient_array_
    cdef unsigned int l = <object> gradient_array__.shape[0] - 1

    with nogil:
        for j in prange(height_, schedule=SCHEDULE, num_threads=THREADS):
            ny = (<float> j * n2) - <float> offset_y

            for i in range(width_):
                nx = (<float>i * n1) - <float> offset_x

                r = &rgb_array[j, i, 0]
                g = &rgb_array[j, i, 1]
                b = &rgb_array[j, i, 2]
                a = &rgb_array[j, i, 3]

                # position in the gradient
                x = <int>((<float>sqrt(nx * nx + ny * ny) * r0) * r_max)

                # check if the radius is greater than the size of the gradient,
                # in which case, the color is black
                if x > l:
                    r[0] = 0
                    g[0] = 0
                    b[0] = 0
                    a[0] = 0
                # assign the gradient
                else:
                    r[0] = <unsigned char>gradient_array__[x, 0]
                    g[0] = <unsigned char>gradient_array__[x, 1]
                    b[0] = <unsigned char>gradient_array__[x, 2]
                    a[0] = <unsigned char>gradient_array__[x, 3]

    return frombuffer(rgb_array, (width_, height_), "RGBA").convert_alpha()


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
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
    CREATE A RADIAL GRADIENT (24-bit OPAQUE)

    Iterate over width/2 and height/2 (NW quarter of the surface) and mirror the 
    pixels for the other blocks  (NE, SE, SW)
    
    :param width_         : integer; surface width in pixels 
    :param height_        : integer; surface height in pixels
    :param start_color_   : tuple; Contains the end color of the radian (RGB values), default (0, 0, 0)
    :param end_color_     : tuple; Contains the start color of the radian (RGB values), default (255, 0, 0)
    :param gradient_array_: numpy.array; gradient array shape (w, 3) containing RGB colors (uint8) 
    :param factor_        : float; Value must be > 0. Default is 1.4 
    :param threads_       : integer; concurrent threads default 8
    :return               : pygame.Surface; Return a radial gradient centre from the surface 
        origin C(Width/2, height/2)
    """

    assert r_max != 0, "Constant r_max cannot be null"
    if factor_ <=0:
        raise ValueError("Argument factor_ cannot be <= 0.0 default is 1.4")
    assert width_ > 0, "Argument width cannot be <=0"
    assert height_ > 0, "Argument height cannot be <=0"

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

        gradient_array_ = create_horizontal_gradient_1d(
            <int>sqrt(width_ * width_ + (height_ * <float>0.5) * (height_ * <float>0.5)),
            start_color=start_color_,
            end_color=end_color_
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




@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
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
    CREATE A RADIAL GRADIENT (32-BIT WITH TRANSPARENCY)

    Iterate over width/2 and height/2 (NW quarter of the surface) and mirror the 
    pixels for the other blocks  (NE, SE, SW)

    :param width_         : integer; surface width in pixels 
    :param height_        : integer; surface height in pixels
    :param end_color_     : tuple; Contains the start color of the radian (RGBA values), default (255, 0, 0, 255)
    :param start_color_   : tuple; Contains the end color of the radian (RGBA values), default (0, 0, 0, 0)
    :param gradient_array_: numpy.array; gradient array shape (w, 3) containing RGB colors (uint8) 
    :param factor_        : float; Value must be > 0. Default is 1.4 
    :param threads_       : integer; concurrent threads default 8
    :return               : pygame.Surface; Return a radial gradient centre from the surface 
        origin C(Width/2, height/2)
    """

    assert r_max != 0, "Constant r_max cannot be null"
    if factor_ <=0:
        raise ValueError("Argument factor_ cannot be <= 0.0 default is 1.4")
    assert width_ > 0, "Argument width cannot be <=0"
    assert height_ > 0, "Argument height cannot be <=0"

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

        gradient_array_ = create_horizontal_gradient_1d_alpha(
            <int>sqrt(width_ * width_ + (height_ * <float>0.5) * (height_ * <float>0.5)),
            start_color=start_color_,
            end_color=end_color_
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


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cpdef float _test_color_dist_hsv(list rgb1, list rgb2):
    """
    TEST ONLY
    """
    cdef hsv hsv_1, hsv_2
    hsv_1 = struct_rgb_to_hsv(<float>rgb1[0]/255.0, <float>rgb1[1]/255.0, <float>rgb1[2]/255.0)
    hsv_2 = struct_rgb_to_hsv(<float>rgb2[0]/255.0, <float>rgb2[1]/255.0, <float>rgb2[2]/255.0)
    return color_dist_hsv(hsv_1, hsv_2)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cdef float color_dist_hsv(hsv hsv_1, hsv hsv_2)nogil:
    """
    DETERMINE THE DISTANCE BETWEEN TWO COLORS IN DOMAIN HSV
    """
    return (hsv_1.h - hsv_2.h) ** 2 + (hsv_1.s - hsv_2.s) ** 2 + (hsv_1.v - hsv_2.v) ** 2


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cpdef float _test_color_dist_hsl(list rgb1, list rgb2):
    """
    TEST ONLY
    """
    cdef hsl hsl_1, hsl_2
    hsl_1 = struct_rgb_to_hsl(<float>rgb1[0]/255.0, <float>rgb1[1]/255.0, <float>rgb1[2]/255.0)
    hsl_2 = struct_rgb_to_hsl(<float>rgb2[0]/255.0, <float>rgb2[1]/255.0, <float>rgb2[2]/255.0)
    return color_dist_hsl(hsl_1, hsl_2)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cdef float color_dist_hsl(hsl hsl_1, hsl hsl_2)nogil:
    """
    DETERMINE THE DISTANCE BETWEEN TWO COLORS IN DOMAIN HSV
    """
    return (hsl_1.h - hsl_2.h) ** 2 + (hsl_1.s - hsl_2.s) ** 2 + (hsl_1.l - hsl_2.l) ** 2



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cpdef _test_color_diff_hsv(list rgb__, palette_):
    """
    TEST ONLY
    """
    cdef rgb rgb_
    rgb_.r = <float>rgb__[0]/<float>255.0
    rgb_.g = <float>rgb__[1]/<float>255.0
    rgb_.b = <float>rgb__[2]/<float>255.0
    cdef Py_ssize_t l = len(palette_)
    return color_diff_hsv(rgb_, palette_, l)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cdef rgb color_diff_hsv(
        rgb color0,
        float [:, :] palette_,
        Py_ssize_t p_length)nogil:

    """
    FIND THE CLOSEST MATCH FOR A GIVEN COLOR (color0) FROM 
    A COLOR PALETTE (palette_). 
    
    This method is using hsv space to find the closest color from the palette.
    Each colors from the palette are compare with the HSV value of the given color
    and the minimal difference between HSV values represent the closest match. 
    
    :param color0  : struct RGB containing the pixel color (normalized)
    :param palette_: numpy.ndarray shape (w, 3) type float32 containing the palette colors RGB normalized
    :param p_length: integer; size of the given palette or number of colors
    :return: struct RGB containing the new pixel values RGB range [0..255]
    """

    cdef:

        float * hsv_values = <float *> malloc(p_length * sizeof(float))
        int i
        rgb color1
        hsv hsv1, hsv2
        float minimum


    # THE RGB TO HSV VALUE NEVER CHANGE INSIDE THE LOOP
    hsv1 = struct_rgb_to_hsv(<float>color0.r,<float>color0.g,<float>color0.b)

    for i in range(p_length):
        hsv2 = struct_rgb_to_hsv(<float>palette_[ i, 0 ],<float>palette_[ i, 1 ],<float>palette_[ i, 2 ])
        hsv_values[i] = <float>color_dist_hsv(hsv1, hsv2)

    minimum = <float>min_f(hsv_values, p_length)

    cdef bint found = False
    for i in range(p_length):
        if minimum == hsv_values[i]:
            found = True
            break

    if found:
        color1.r = palette_[ i, 0 ]
        color1.g = palette_[ i, 1 ]
        color1.b = palette_[ i, 2 ]
    else:
        color1.r = <float>0.0
        color1.g = <float>0.0
        color1.b = <float>0.0

    free(hsv_values)

    return color1



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cpdef _test_color_diff_hsl(list rgb__, palette_):
    """
    TEST ONLY
    """
    cdef rgb rgb_
    rgb_.r = <float>rgb__[0]/<float>255.0
    rgb_.g = <float>rgb__[1]/<float>255.0
    rgb_.b = <float>rgb__[2]/<float>255.0
    cdef Py_ssize_t l = len(palette_)
    return color_diff_hsl(rgb_, palette_, l)


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cdef rgb color_diff_hsl(
        rgb color0,
        float [:, :] palette_,
        Py_ssize_t p_length)nogil:

    """
    FIND THE CLOSEST RGB COLOR (COLOR0) FROM A GIVEN PALETTE (PALETTE_)

    This method is using HSL space to find the closest color from the palette.
    Each colors from the palette are compare with the HSL value of the given color
    and the minimal difference between HSL values represent the closest match.

    :param color0  : struct RGB containing the pixel color (normalized colors)
    :param palette_: numpy.ndarray shape (w, 3) type float32 containing the normalized palette colors
    :param p_length: integer; size of the given palette or number of colors
    :return: struct RGB containing the new pixel values RGB range [0..255]
    """

    cdef:

        float * hsl_values = <float *> malloc(p_length * sizeof(float))
        int i
        rgb color1
        hsl hsl1, hsl2
        float minimum


    # THE RGB TO HSL VALUE NEVER CHANGE INSIDE THE LOOP
    hsl1 = struct_rgb_to_hsl(<float>color0.r, <float>color0.g, <float>color0.b)

    for i in range(p_length):
        hsl2 = struct_rgb_to_hsl(<float>palette_[ i, 0 ], <float>palette_[ i, 1 ], <float>palette_[ i, 2 ])
        hsl_values[i] = <float>color_dist_hsl(hsl1, hsl2)

    minimum = <float>min_f(hsl_values, p_length)

    cdef bint found = False

    for i in range(p_length):

        if minimum == hsl_values[i]:
            found = True
            break

    if found:
        color1.r = palette_[ i, 0 ]
        color1.g = palette_[ i, 1 ]
        color1.b = palette_[ i, 2 ]
    else:
        color1.r = <float>0.0
        color1.g = <float>0.0
        color1.b = <float>0.0

    free(hsl_values)

    return color1




@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
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


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cdef rgb close_color(
        rgb colors,
        float [:, :] palette_,
        Py_ssize_t w)nogil:

    """
    FIND THE NEAREST COLOR MATCHING A GIVEN COLOR 
    
    Iterate over a 2d color palette filled with RGB values (array shape, 
    w, 3) type float32. The palette is not normalised and contains values 
    in range [0...255] 
    During the iteration, the given color is compared with the palette values
    The original color 'colors' will be replaced with a new color
    from the palette (new RGB components). 
    This algorithm will work if the color palette contains unique RGB components.
    For palettes with duplicate colors, the algorithm will select the first color
    corresponding to the sum of the RGB (difference between each components)
    
    :param colors  : struct RGB; float; Colors define with a struct type of data  
    :param palette_: numpy.array; Palette containing the RGB color values range [0...255]  
    :param w       : integer; size of the palette 
    :return        : struct rgb containing the new RGB values close match from 
    the given color  
    """

    cdef:
        int i
        float * tmp_v = <float *> malloc(w * sizeof(float))
        rgb rgb_
        float v_min,
        unsigned int s1 = 0
        unsigned int s2 = 0


    # Iterate over the palette colors an calculate the
    # distance between both colors (palette and current pixels RGB colors),
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





@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cdef rgb use_palette(
        rgb colors,
        float [:, :] palette_,
        Py_ssize_t w)nogil:

    """
    PICKED RGB VALUES FROM A GIVEN PALETTE TO MATCH A GIVEN COLOR
    ** PAINTING MODE **

    Picked r, g, b values from a palette by going through every
    elements and picking the closest match for each channels.
    The final color will be not necessary match a color from the palette
    but will have instead, a palette values from each channels.


    :param colors  : struct; Contains RGB values integer values in range [0..255]
    :param palette_: numpy.array shape (w, 3) type float32 containing RGB colors values
    :param w       : Size of the palette or number of colors available
    :return        : Return a color RGB containing colors from the palette.
    """

    cdef:
        int i, j, k
        float * v_red   = <float *> malloc(w * sizeof(float))
        float * v_green = <float *> malloc(w * sizeof(float))
        float * v_blue  = <float *> malloc(w * sizeof(float))
        float r_min, g_min, b_min
        rgb rgb_

    # GO THROUGH EACH CHANNELS OF THE PALETTE AND
    # SUBTRACT VALUES.
    for i in prange(w):
        v_red[i]   = abs_c(colors.r - palette_[i, 0] )
        v_green[i] = abs_c(colors.g - palette_[i, 1] )
        v_blue[i]  = abs_c(colors.b - palette_[i, 2] )

    # THE FINAL R, G, B VALUES WILL BE A PICKED
    # COLOR FROM EACH CHANNELS CLOSE TO THE ORIGINAL PIXEL
    # VALUE
    r_min = <float> min_f(v_red, w)
    g_min = <float> min_f(v_green, w)
    b_min = <float> min_f(v_blue, w)

    # FIND A RED COLOR FROM THE PALETTE
    # CLOSE TO THE ORIGINAL RED VALUE
    for i in range(w):
        if v_red[i] == r_min:
            break

    # FIND A GREEN COLOR FROM THE PALETTE
    # CLOSE TO THE ORIGINAL GREEN VALUE
    for j in range(w):
        if v_green[j] == g_min:
            break

    # FIND A BLUE COLOR FROM THE PALETTE
    # CLOSE TO THE ORIGINAL BLUE VALUE
    for k in range(w):
        if v_blue[k] == b_min:
            break

    rgb_.r = palette_[i, 0]
    rgb_.g = palette_[j, 1]
    rgb_.b = palette_[k, 2]

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
cpdef object scroll24(surface, short int dx=0, short int dy=0):
    """
    SCROLL SURFACE HORIZONTALLY

    create and return a new surface

    :param surface: pygame Surface 24, 32-bit format compatible.
    :param dx: scroll the array horizontally (-dx left, +dx right)
    :param dy: scroll the array vertically (-dy up, +dy down)
    :return: void
    """
    return scroll24_c(surface, dx, dy)


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cdef scroll24_c(surface, short int dx, short int dy):
    """
    SCROLL SURFACE HORIZONTALLY

    create and return a new surface

    :param surface: pygame Surface 24, 32-bit format compatible.
    :param dx: scroll the array horizontally (-dx left, +dx right)
    :param dy: scroll the array vertically (-dy up, +dy down)
    :return: void
    """

    if not isinstance(dx, int):
        raise TypeError('dx, an integer is required (got type %s)' % type(dx))

    if not isinstance(dy, int):
        raise TypeError('dy, an integer is required (got type %s)' % type(dy))

    if not isinstance(surface, pygame.Surface):
        raise TypeError('surface, a pygame.Surface is required (got type %s)' % type(surface))

    cdef:
        int w, h, dim
        unsigned char [:, :, :] rgb_array

    try:
        # array = pixels3d(surface)
        rgb_array = surface.get_view('3')

    except (ValueError, pygame.error) as e:
            raise ValueError('\nIncompatible pixel format.')

    try:
        w, h, dim = (<object> rgb_array).shape[:3]
    except (ValueError, pygame.error) as e:
        raise ValueError('\nArray shape not compatible.')

    cdef:
        int i=0, j=0, ii=0, jj=0
        unsigned char [:, :, ::1] new_array = numpy.empty((h, w, 3), dtype=numpy.uint8)

    if dx==0 and dy==0:
        return surface

    with nogil:

        if dx !=0 and dy != 0:

            for j in prange(h, schedule=SCHEDULE, num_threads=THREADS):
                    for i in range(w):

                        ii = (i + dx) % w
                        jj = (j + dy) % h
                        if ii < 0: ii = ii + w
                        if jj < 0: jj = jj + h
                        new_array[jj, ii, 0] = rgb_array[i, j, 0]
                        new_array[jj, ii, 1] = rgb_array[i, j, 1]
                        new_array[jj, ii, 2] = rgb_array[i, j, 2]

        elif dx != 0:

            for j in prange(h, schedule=SCHEDULE, num_threads=THREADS):
                for i in range(w):

                    ii = (i + dx) % w
                    if ii < 0: ii = ii + w

                    new_array[j, ii, 0] = rgb_array[i, j, 0]
                    new_array[j, ii, 1] = rgb_array[i, j, 1]
                    new_array[j, ii, 2] = rgb_array[i, j, 2]

        elif dy != 0:

            for j in prange(h, schedule=SCHEDULE, num_threads=THREADS):
                for i in range(w):

                    jj = (j + dy) % h
                    if jj < 0:
                        jj = jj + h

                    new_array[jj, i, 0] = rgb_array[i, j, 0]
                    new_array[jj, i, 1] = rgb_array[i, j, 1]
                    new_array[jj, i, 2] = rgb_array[i, j, 2]

    return frombuffer(new_array, (w, h), 'RGB')



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cpdef void scroll24_inplace(surface, short int dx=0, short int dy=0):
    """
    SCROLL SURFACE HORIZONTALLY

    Transformation is applied inplace

    :param surface: pygame Surface 24, 32-bit format compatible.
    :param dx: scroll the array horizontally (-dx left, +dx right)
    :param dy: scroll the array vertically (-dy up, +dy down)
    :return: void
    """
    scroll24_inplace_c(surface, dx, dy)

@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cdef void scroll24_inplace_c(surface, short int dx, short int dy):
    """
    SCROLL SURFACE HORIZONTALLY

    Transformation is applied inplace

    :param surface: pygame Surface 24, 32-bit format compatible.
    :param dx: scroll the array horizontally (-dx left, +dx right)
    :param dy: scroll the array vertically (-dy up, +dy down)
    :return: void
    """

    if not isinstance(dx, int):
        raise TypeError('dx, an short integer is required (got type %s)' % type(dx))

    if not isinstance(dy, int):
        raise TypeError('dy, an short integer is required (got type %s)' % type(dy))

    if not isinstance(surface, pygame.Surface):
        raise TypeError('surface, a pygame.Surface is required (got type %s)' % type(surface))

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
        unsigned char * r
        unsigned char * g
        unsigned char * b


    with nogil:

        if dx==0 and dy==0:
            return

        if dx > 0:
            for j in prange(h, schedule=SCHEDULE, num_threads=THREADS):
                for i in range(w):

                    ii = (i + dx) % w
                    if ii < 0: ii = ii + w

                    r = &rgb_array[i, j, 0]
                    g = &rgb_array[i, j, 1]
                    b = &rgb_array[i, j, 2]

                    r[0] = rgb_array[ii, j, 0]
                    g[0] = rgb_array[ii, j, 1]
                    b[0] = rgb_array[ii, j, 2]

        else:
            for j in prange(h, schedule=SCHEDULE, num_threads=THREADS):
                for i in range(w-1, -1, -1):

                    ii = (i + dx) % w
                    if ii < 0: ii = ii + w

                    r = &rgb_array[i, j, 0]
                    g = &rgb_array[i, j, 1]
                    b = &rgb_array[i, j, 2]

                    r[0] = rgb_array[ii, j, 0]
                    g[0] = rgb_array[ii, j, 1]
                    b[0] = rgb_array[ii, j, 2]
        if dy > 0:
            for i in prange(w, schedule=SCHEDULE, num_threads=THREADS):
                for j in range(h):

                    jj = (j + dy) % h
                    if jj < 0: jj = jj + h

                    r = &rgb_array[i, j, 0]
                    g = &rgb_array[i, j, 1]
                    b = &rgb_array[i, j, 2]

                    r[0] = rgb_array[i, jj, 0]
                    g[0] = rgb_array[i, jj, 1]
                    b[0] = rgb_array[i, jj, 2]

        else:
            for i in prange(w, schedule=SCHEDULE, num_threads=THREADS):
                for j in range(h-1, -1, -1):

                    jj = (j + dy) % h
                    if jj < 0: jj = jj + h

                    r = &rgb_array[i, j, 0]
                    g = &rgb_array[i, j, 1]
                    b = &rgb_array[i, j, 2]

                    r[0] = rgb_array[i, jj, 0]
                    g[0] = rgb_array[i, jj, 1]
                    b[0] = rgb_array[i, jj, 2]

@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cpdef void scroll24_arr_inplace(
    unsigned char [:, :, :] rgb_array,
    short int dx=0,
    short int dy=0
    ):
    """
    SCROLL ARRAY/SURFACE HORIZONTALLY

    Transformation is applied inplace

    :param rgb_array: numpy.ndarray containing RGB pixels (the array must reference all the pixels)
    :param dx: short int; scroll the array horizontally (-dx left, +dx right)
    :param dy: short int; scroll the array vertically (-dy up, +dy down)
    :return: void
    """
    scroll24_arr_inplace_c(rgb_array, dx, dy)


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cdef void scroll24_arr_inplace_c(
    unsigned char [:, :, :] rgb_array,
    short int dx,
    short int dy
    ):
    """
    SCROLL ARRAY/SURFACE HORIZONTALLY

    Transformation is applied inplace

    :param rgb_array: numpy.ndarray containing RGB pixels (the array must reference all the pixels)
    :param dx: short int; scroll the array horizontally (-dx left, +dx right)
    :param dy: short int; scroll the array vertically (-dy up, +dy down)
    :return: void
    """

    if not isinstance(dx, int):
        raise TypeError('dx, an integer is required (got type %s)' % type(dx))

    if not isinstance(dy, int):
        raise TypeError('dy, an integer is required (got type %s)' % type(dy))

    cdef int w, h, dim

    try:
        w, h, dim = (<object> rgb_array).shape[:3]
    except (ValueError, pygame.error) as e:
        raise ValueError('\rgb_array shape not compatible.')


    cdef:
        int i=0, j=0, ii=0, jj=0
        unsigned char * r
        unsigned char * g
        unsigned char * b


    with nogil:

        if dx==0 and dy==0:
            return

        if dx > 0:
            for j in prange(h, schedule=SCHEDULE, num_threads=THREADS):
                for i in range(w):

                    ii = (i + dx) % w
                    if ii < 0: ii = ii + w

                    r = &rgb_array[i, j, 0]
                    g = &rgb_array[i, j, 1]
                    b = &rgb_array[i, j, 2]

                    r[0] = rgb_array[ii, j, 0]
                    g[0] = rgb_array[ii, j, 1]
                    b[0] = rgb_array[ii, j, 2]

        else:
            for j in prange(h, schedule=SCHEDULE, num_threads=THREADS):
                for i in range(w-1, -1, -1):

                    ii = (i + dx) % w
                    if ii < 0: ii = ii + w

                    r = &rgb_array[i, j, 0]
                    g = &rgb_array[i, j, 1]
                    b = &rgb_array[i, j, 2]

                    r[0] = rgb_array[ii, j, 0]
                    g[0] = rgb_array[ii, j, 1]
                    b[0] = rgb_array[ii, j, 2]

        if dy > 0:
            for i in prange(w, schedule=SCHEDULE, num_threads=THREADS):
                for j in range(h):

                    jj = (j + dy) % h
                    if jj < 0: jj = jj + h

                    r = &rgb_array[i, j, 0]
                    g = &rgb_array[i, j, 1]
                    b = &rgb_array[i, j, 2]

                    r[0] = rgb_array[i, jj, 0]
                    g[0] = rgb_array[i, jj, 1]
                    b[0] = rgb_array[i, jj, 2]

        else:
            for i in prange(w, schedule=SCHEDULE, num_threads=THREADS):
                for j in range(h-1, -1, -1):

                    jj = (j + dy) % h
                    if jj < 0: jj = jj + h

                    r = &rgb_array[i, j, 0]
                    g = &rgb_array[i, j, 1]
                    b = &rgb_array[i, j, 2]

                    r[0] = rgb_array[i, jj, 0]
                    g[0] = rgb_array[i, jj, 1]
                    b[0] = rgb_array[i, jj, 2]



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cpdef object surface_copy(object surface_):
    """
    HOOK METHOD FOR SURFACE COPY
    """
    return surface_copy_c(surface_)

@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cdef inline object surface_copy_c(object surface_):
    """
    EQUIVALENT TO PYGAME SURFACE COPY 
    """

    cdef unsigned char [:, :, :] array_

    try:
        array_ = surface_.get_view('3')

    except Exception as e:
        raise ValueError(
        "Cannot reference source pixels into a 3d array.\n %s " % e)

    cdef:
        int i, j
        int w = <object>(array_.shape[0])
        int h = <object>(array_.shape[1])
        unsigned char [:, :, ::1] array_copy =\
            numpy.empty((h, w, 3), dtype=uint8)

    with nogil:
        for j in prange (h, schedule=SCHEDULE, num_threads=THREADS):
            for i in range(w):
                array_copy[j, i, 0] = array_[i, j, 0]
                array_copy[j, i, 1] = array_[i, j, 1]
                array_copy[j, i, 2] = array_[i, j, 2]

    return frombuffer(array_copy, (w, h), "RGB")



