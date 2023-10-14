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


cimport numpy as np

from libc.math cimport sqrtf as sqrt, atan2f as atan2, sinf as sin,\
    cosf as cos, nearbyintf as nearbyint, expf as exp, powf as pow, floorf as floor
from libc.stdlib cimport malloc, rand, free
from libc.math cimport roundf as round_c, fminf as fmin, fmaxf as fmax
from libc.stdio cimport printf
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
cpdef inline void blit_s(image_, spr_, tuple xy_=(0, 0)):
    """
    BLEND SPRITE TO IMAGE/SURFACE

    :param image_   : pygame.Surface (Source)
    :param spr_     : pygame.Surface (image to add)
    :param xy_      : tuple xy_ (x, y)
    """

    cdef:
            unsigned char[:, :, :] array_
            unsigned char[:, :, :] array_spr

    try:

        array_ = image_.get_view('3')
        array_spr = spr_.get_view('3')

    except Exception as e:
        raise ValueError("\nCannot reference source pixels into a 3d array.\n %s " % e)

    cdef:
        int x_min = xy_[0]
        int y_min = xy_[1]
        int x_max = x_min + array_spr.shape[0]
        int y_max = y_min + array_spr.shape[1]
        unsigned int x0, y0
        int i, j
        unsigned char * r
        unsigned char * g
        unsigned char * b


    with nogil, parallel():

        for j in prange(y_min, y_max):
            y0 = j - y_min

            for i in prange(x_min, x_max):
                x0 = i - x_min
                r = &array_[ i, j, 0 ]
                g = &array_[ i, j, 1 ]
                b = &array_[ i, j, 2 ]
                r[0] = array_spr[ x0, y0, 0 ]
                g[0] = array_spr[ x0, y0, 1 ]
                b[0] = array_spr[ x0, y0, 2 ]



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cpdef inline void blend_add_surface(source_, destination_):
    blend_add_surface_c(source_, destination_)

@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cdef inline void blend_add_surface_c(source_, destination_):
    """
    BLEND ADD 2 SURFACES EQUIVALENT TO BLEND_RGB_ADD
    
    both surface must have the same sizes
    Use the source surface to visualized the effect

    :param source_     : pygame.Surface (Source)
    :param destination_: pygame.Surface (destination)

    """

    cdef:
        unsigned char[:, :, :] source_array_
        unsigned char[:, :, :] destination_array_

    try:

        source_array_ = source_.get_view('3')
        destination_array_ = destination_.get_view('3')

    except Exception as e:
        raise ValueError("\nCannot reference source pixels into a 3d array.\n %s " % e)

    cdef:

        unsigned int c1, c2, c3
        int i=0, j=0
        Py_ssize_t w = source_array_.shape[0]
        Py_ssize_t h = source_array_.shape[1]
        unsigned char * r
        unsigned char * g
        unsigned char * b


    with nogil:

        for j in prange(h, schedule=SCHEDULE, num_threads=THREADS):

            for i in range(w):

                r = &source_array_[i, j, 0]
                g = &source_array_[i, j, 1]
                b = &source_array_[i, j, 2]

                c1 = r[0] + destination_array_[i, j, 0]
                c2 = g[0] + destination_array_[i, j, 1]
                c3 = b[0] + destination_array_[i, j, 2]

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
cpdef inline void blend_add_array(
    unsigned char [:, :, :] source_array_,
    unsigned char [:, :, :] destination_array_
    ):
    blend_add_array_c(source_array_, destination_array_)

@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cdef inline void blend_add_array_c(
    unsigned char [:, :, :] source_array_,
    unsigned char [:, :, :] destination_array_
    ):

    """
    BLEND, ADD TWO ARRAYS EQU TO BLEND_RGB_ADD
    
    Both arrays must have the same size, shape (w, h, 3) of type unsigned char 
    Use the source array to visualized the effect
    
    :param source_array_     : numpy.ndarray type (w, h, 3) containing 
        RGB pixel values (SOURCE)
    :param destination_array_: numpy.ndarray type (w, h, 3) containing 
        RGB pixel values (DESTINATION)
    """

    cdef:

        unsigned int c1, c2, c3
        int i=0, j=0
        Py_ssize_t w = source_array_.shape[0]
        Py_ssize_t h = source_array_.shape[1]
        unsigned char * r
        unsigned char * g
        unsigned char * b


    with nogil:

        for j in prange(h, schedule=SCHEDULE, num_threads=THREADS):

            for i in range(w):

                r = &source_array_[i, j, 0]
                g = &source_array_[i, j, 1]
                b = &source_array_[i, j, 2]

                c1 = r[0] + destination_array_[i, j, 0]
                c2 = g[0] + destination_array_[i, j, 1]
                c3 = b[0] + destination_array_[i, j, 2]

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
cpdef inline void blend_add_alpha(
    unsigned char [:, :] source_array_,
    unsigned char [:, :] destination_array_
    ):
    blend_add_alpha_c(source_array_,destination_array_)

@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cdef inline void blend_add_alpha_c(
    unsigned char [:, :] source_array_,
    unsigned char [:, :] destination_array_
    ):

    """
    BLEND ADD 2 ARRAYS EQU TO BLEND_RGB_ADD (FOR 2D ARRAY)

    This method is useful for blending alpha channels (2D arrays)
    Both arrays must have the same size of type unsigned char
    Use the source array to visualized the effect

    :param source_array_     : numpy.ndarray type (w, h) containing 
        RGB pixel values (SOURCE)
    :param destination_array_: numpy.ndarray type (w, h) containing 
        RGB pixel values (DESTINATION)

    """

    cdef:

        unsigned char c1, c2, c3
        int i=0, j=0
        Py_ssize_t w = source_array_.shape[0]
        Py_ssize_t h = source_array_.shape[1]
        unsigned char * r
        unsigned char * g
        unsigned char * b


    with nogil:

        for j in prange(h, schedule=SCHEDULE, num_threads=THREADS):

            for i in range(w):

                r = &source_array_[i, j]
                g = &source_array_[i, j]
                b = &source_array_[i, j]

                c1 = r[0] + destination_array_[i, j]
                c2 = g[0] + destination_array_[i, j]
                c3 = b[0] + destination_array_[i, j]

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
cpdef inline void blend_sub_surface(source_, destination_):
    blend_sub_surface_c(source_, destination_)

@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cdef inline void blend_sub_surface_c(source_, destination_):
    """
    BLEND SUB 2 SURFACES EQU TO BLEND_RGB_SUB
    
    Both surfaces must have the same sizes, array shape (w, h, 3) of type unsigned char
    Use the source surface to visualized the effect
    
    :param source_     : pygame.Surface (Source)
    :param destination_: pygame.Surface (destination)

    """

    cdef:
            unsigned char[:, :, :] source_array_
            unsigned char[:, :, :] destination_array_

    try:

        source_array_ = source_.get_view('3')
        destination_array_ = destination_.get_view('3')

    except Exception as e:
        raise ValueError("\nCannot reference source pixels into a 3d array.\n %s " % e)

    cdef:

        int c1, c2, c3
        int i=0, j=0
        Py_ssize_t w = source_array_.shape[0]
        Py_ssize_t h = source_array_.shape[1]
        unsigned char * r
        unsigned char * g
        unsigned char * b


    with nogil:

        for j in prange(h, schedule=SCHEDULE, num_threads=THREADS):

            for i in range(w):

                r = &source_array_[i, j, 0]
                g = &source_array_[i, j, 1]
                b = &source_array_[i, j, 2]

                c1 = r[0] - destination_array_[i, j, 0]
                c2 = g[0] - destination_array_[i, j, 1]
                c3 = b[0] - destination_array_[i, j, 2]

                r[0] = c1 if c1 > 0 else 0
                g[0] = c2 if c2 > 0 else 0
                b[0] = c3 if c3 > 0 else 0


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cpdef inline void blend_sub_array(
    unsigned char [:, :, :] source_array_,
    unsigned char [:, :, :] destination_array_
    ):
    blend_sub_array(source_array_, destination_array_)

@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cdef inline void blend_sub_array_c(
    unsigned char [:, :, :] source_array_,
    unsigned char [:, :, :] destination_array_
    ):

    """
    BLEND SUB 2 ARRAYS 
    
    Both arrays must have the same sizes, array shape(w, h, 3) of type unsigned char
    Use the source array to visualized the effect

    :param source_array_     : numpy.ndarray type (w, h, 3) containing 
        RGB pixel values (SOURCE)
    :param destination_array_: numpy.ndarray type (w, h, 3) containing
        RGB pixel values (DESTINATION)

    """

    cdef:

        int c1, c2, c3
        int i=0, j=0
        Py_ssize_t w = source_array_.shape[0]
        Py_ssize_t h = source_array_.shape[1]
        unsigned char * r
        unsigned char * g
        unsigned char * b


    with nogil:

        for j in prange(h, schedule=SCHEDULE, num_threads=THREADS):

            for i in range(w):

                r = &source_array_[i, j, 0]
                g = &source_array_[i, j, 1]
                b = &source_array_[i, j, 2]

                c1 = r[0] - destination_array_[i, j, 0]
                c2 = g[0] - destination_array_[i, j, 1]
                c3 = b[0] - destination_array_[i, j, 2]

                r[0] = c1 if c1 > 0 else 0
                g[0] = c2 if c2 > 0 else 0
                b[0] = c3 if c3 > 0 else 0


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cpdef inline void blend_min_surface(source_, destination_):
    blend_min_surface_c(source_, destination_)

@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cdef inline void blend_min_surface_c(source_, destination_):
    """
    BLEND ADD MIN 2 SURFACES EQU TO BLEND_ADD_MIN

    Both surfaces must have the same sizes
    Use the source surface to visualized the effect
    
    :param source_     : pygame.Surface (Source)
    :param destination_: pygame.Surface (destination)

    """

    cdef:
            unsigned char[:, :, :] source_array_
            unsigned char[:, :, :] destination_array_

    try:

        source_array_ = source_.get_view('3')
        destination_array_ = destination_.get_view('3')

    except Exception as e:
        raise ValueError("\nCannot reference source pixels into a 3d array.\n %s " % e)

    cdef:


        int i=0, j=0
        Py_ssize_t w = source_array_.shape[0]
        Py_ssize_t h = source_array_.shape[1]
        unsigned char * r
        unsigned char * g
        unsigned char * b
        unsigned char * rr
        unsigned char * gg
        unsigned char * bb

    with nogil:

        for j in prange(h, schedule=SCHEDULE, num_threads=THREADS):

            for i in range(w):

                r = &source_array_[i, j, 0]
                g = &source_array_[i, j, 1]
                b = &source_array_[i, j, 2]

                rr = &destination_array_[i, j, 0]
                gg = &destination_array_[i, j, 1]
                bb = &destination_array_[i, j, 2]

                r[0] = r[0] if r[0] < rr[0] else rr[0]
                g[0] = g[0] if g[0] < gg[0] else gg[0]
                b[0] = b[0] if b[0] < bb[0] else bb[0]



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cpdef inline void blend_min_array(
    unsigned char [:, :, :] source_array_,
    unsigned char [:, :, :] destination_array_
    ):
    blend_min_array_c(source_array_, destination_array_)


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cdef inline void blend_min_array_c(
    unsigned char [:, :, :] source_array_,
    unsigned char [:, :, :] destination_array_
    ):

    """
    BLEND ADD MIN 2 ARRAYS EQU TO BLEND_ADD_MIN

    Both arrays must have the same sizes, array shape(w, h, 3) of type unsigned char
    Use the source array to visualized the effect
    
    :param source_array_     : numpy.ndarray type (w, h, 3) containing RGB 
        pixel values (SOURCE)
    :param destination_array_: numpy.ndarray type (w, h, 3) containing RGB 
        pixel values (DESTINATION)

    """

    cdef:


        int i=0, j=0
        Py_ssize_t w = source_array_.shape[0]
        Py_ssize_t h = source_array_.shape[1]
        unsigned char * r
        unsigned char * g
        unsigned char * b
        unsigned char * rr
        unsigned char * gg
        unsigned char * bb


    with nogil:

        for j in prange(h, schedule=SCHEDULE, num_threads=THREADS):

            for i in range(w):

                r = &source_array_[i, j, 0]
                g = &source_array_[i, j, 1]
                b = &source_array_[i, j, 2]

                rr = &destination_array_[i, j, 0]
                gg = &destination_array_[i, j, 1]
                bb = &destination_array_[i, j, 2]

                r[0] = r[0] if r[0] < rr[0] else rr[0]
                g[0] = g[0] if g[0] < gg[0] else gg[0]
                b[0] = b[0] if b[0] < bb[0] else bb[0]


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cpdef inline void blend_max_surface(source_, destination_):
    blend_max_surface_c(source_, destination_)


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cdef inline void blend_max_surface_c(source_, destination_):
    """
    BLEND ADD MAX 2 SURFACES EQU TO BLEND_ADD_MAX

    Both surfaces must have the same sizes
    Use the source surface to visualized the effect

    :param source_     : pygame.Surface (Source)
    :param destination_: pygame.Surface (destination)

    """

    cdef:
            unsigned char[:, :, :] source_array_
            unsigned char[:, :, :] destination_array_

    try:

        source_array_ = source_.get_view('3')
        destination_array_ = destination_.get_view('3')

    except Exception as e:
        raise ValueError("\nCannot reference source pixels into a 3d array.\n %s " % e)

    cdef:


        int i=0, j=0
        Py_ssize_t w = source_array_.shape[0]
        Py_ssize_t h = source_array_.shape[1]
        unsigned char * r
        unsigned char * g
        unsigned char * b
        unsigned char * rr
        unsigned char * gg
        unsigned char * bb

    with nogil:

        for j in prange(h, schedule=SCHEDULE, num_threads=THREADS):

            for i in range(w):

                r = &source_array_[i, j, 0]
                g = &source_array_[i, j, 1]
                b = &source_array_[i, j, 2]

                rr = &destination_array_[i, j, 0]
                gg = &destination_array_[i, j, 1]
                bb = &destination_array_[i, j, 2]

                r[0] = r[0] if r[0] > rr[0] else rr[0]
                g[0] = g[0] if g[0] > gg[0] else gg[0]
                b[0] = b[0] if b[0] > bb[0] else bb[0]



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cpdef inline void blend_max_array(
    unsigned char [:, :, :] source_array_,
    unsigned char [:, :, :] destination_array_
    ):
    blend_max_array_c(source_array_, destination_array_)

@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cdef inline void blend_max_array_c(
    unsigned char [:, :, :] source_array_,
    unsigned char [:, :, :] destination_array_
    ):

    """
    BLEND ADD MAX 2 ARRAYS EQU TO BLEND_ADD_MAX

    Both arrays must have the same sizes, array shape(w, h, 3) of type unsigned char
    Use the source array to visualized the effect

    :param source_array_     : numpy.ndarray type (w, h, 3) containing 
        RGB pixel values (SOURCE)
    :param destination_array_: numpy.ndarray type (w, h, 3) containing 
        RGB pixel values (DESTINATION)

    """

    cdef:


        int i=0, j=0
        Py_ssize_t w = source_array_.shape[0]
        Py_ssize_t h = source_array_.shape[1]
        unsigned char * r
        unsigned char * g
        unsigned char * b
        unsigned char * rr
        unsigned char * gg
        unsigned char * bb


    with nogil:

        for j in prange(h, schedule=SCHEDULE, num_threads=THREADS):

            for i in range(w):

                r = &source_array_[i, j, 0]
                g = &source_array_[i, j, 1]
                b = &source_array_[i, j, 2]

                rr = &destination_array_[i, j, 0]
                gg = &destination_array_[i, j, 1]
                bb = &destination_array_[i, j, 2]

                r[0] = r[0] if r[0] > rr[0] else rr[0]
                g[0] = g[0] if g[0] > gg[0] else gg[0]
                b[0] = b[0] if b[0] > bb[0] else bb[0]

