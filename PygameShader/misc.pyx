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
    from pygame.surfarray import array3d, pixels3d, array_alpha, pixels_alpha
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


DEF SCHEDULE = 'static'

DEF OPENMP = True
# num_threads – The num_threads argument indicates how many threads the team should consist of.
# If not given, OpenMP will decide how many threads to use.
# Typically this is the number of cores available on the machine. However,
# this may be controlled through the omp_set_num_threads() function,
# or through the OMP_NUM_THREADS environment variable.
DEF THREAD_NUMBER = 1
if OPENMP is True:
    DEF THREAD_NUMBER = 8


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef swap_channels24_c(surface_, model):
    """
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
        rgb_ = pixels3d(surface_)
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
        for i in prange(width, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
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
cpdef void rgb_to_bgr_inplace(unsigned char [:, :, :] rgb_array):
    """

    :param rgb_array:
    :return:
    """

    cdef Py_ssize_t w, h
    w, h = (<object>rgb_array).shape[:2]

    cdef:
        int i=0, j=0
        unsigned char tmp

    with nogil:

        for i in prange(w, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            for j in range(h):

                tmp = rgb_array[i, j, 0]  # keep the blue color
                rgb_array[i, j, 0]  = rgb_array[i, j, 2]
                rgb_array[i, j, 2]  = tmp


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef create_horizontal_gradient_1d(
        int value,
        tuple start_color=(255, 0, 0),
        tuple end_color=(0, 255, 0)
):
    cdef:
        float [:] diff_ =  numpy.array(end_color, dtype=float32) - \
                            numpy.array(start_color, dtype=float32)
        float [::1] row = numpy.arange(value, dtype=float32) / (value - 1.0)
        unsigned char [:, ::1] rgb_gradient = empty((value, 3), dtype=uint8)
        float [3] start = numpy.array(start_color, dtype=float32)
        int i=0

    with nogil:
        for i in prange(value, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
               rgb_gradient[i, 0] = <unsigned char>(start[0] + row[i] * diff_[0])
               rgb_gradient[i, 1] = <unsigned char>(start[1] + row[i] * diff_[1])
               rgb_gradient[i, 2] = <unsigned char>(start[2] + row[i] * diff_[2])

    return asarray(rgb_gradient)
