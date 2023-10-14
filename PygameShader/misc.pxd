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

cdef extern from 'Include/Shaderlib.c':

    struct hsv:
        float h;
        float s;
        float v;

    struct hsl:
        float h
        float s
        float l

    struct rgb:
        float r
        float g
        float b

    struct rgb_color_int:
        int r;
        int g;
        int b;

    struct s_min:
        float value;
        unsigned int index;

    hsl struct_rgb_to_hsl(float r, float g, float b)nogil;
    rgb struct_hsl_to_rgb(float h, float s, float l)nogil;

    rgb struct_hsv_to_rgb(float h, float s, float v)nogil;
    hsv struct_rgb_to_hsv(float r, float g, float b)nogil;

    float min_f(float arr[], unsigned int n)nogil;
    s_min minf_struct(float arr[], unsigned int n)nogil;
    unsigned int min_index(float arr[], unsigned int n)nogil;

cimport numpy as np

ctypedef rgb rgb_

cpdef object swap_channels24_c(object surface_, object model)


cpdef create_horizontal_gradient_1d(
        int value,
        tuple start_color=*,
        tuple end_color=*
)
cpdef create_horizontal_gradient_1d_alpha(
        int value,
        tuple start_color=*,
        tuple end_color=*
)


cpdef object horizontal_grad3d(
        int width,
        int height,
        tuple start_color=*,
        tuple end_color=*
)
cpdef object horizontal_grad3d_alpha(
        int width,
        int height,
        tuple start_color=*,
        tuple end_color=*
)

cpdef create_radial_gradient(
        int width_,
        int height_,
        float offset_x              = *,
        float offset_y              = *,
        tuple start_color_          = *,
        tuple end_color_            = *,
        object gradient_array_      = *,
        float factor_               = *,
        unsigned short int threads_ = *
)
cpdef create_radial_gradient_alpha(
        int width_,
        int height_,
        float offset_x              = *,
        float offset_y              = *,
        tuple start_color_          = *,
        tuple end_color_            = *,
        object gradient_array_      = *,
        float factor_               = *,
        unsigned short int threads_ = *
)

cpdef create_quarter_radial_gradient(
        int width_,
        int height_,
        tuple start_color_            = *,
        tuple end_color_              = *,
        object gradient_array_        = *,
        float factor_                 = *,
        unsigned short int threads_   = *
)

cpdef create_quarter_radial_gradient_alpha(
        int width_,
        int height_,
        tuple start_color_            = *,
        tuple end_color_              = *,
        object gradient_array_        = *,
        float factor_                 = *,
        unsigned short int threads_   = *
)

cdef float color_dist_hsv(hsv hsv_1, hsv hsv_2)nogil
cdef float color_dist_hsl(hsl hsl_1, hsl hsl_2)nogil

cdef rgb color_diff_hsv(
        rgb color0,
        float [:, :] palette_,
        Py_ssize_t p_length
)nogil


cdef rgb color_diff_hsl(
        rgb color0,
        float [:, :] palette_,
        Py_ssize_t p_length
)nogil


cdef rgb close_color(
        rgb colors,
        float [:, :] palette_,
        Py_ssize_t w
)nogil




cdef rgb use_palette(
        rgb colors,
        float [:, :] palette_,
        Py_ssize_t w
)nogil


cpdef object scroll24(surface, short int dx=*, short int dy=*)
cdef scroll24_c(surface, short int dx, short int dy)

cpdef void scroll24_inplace(surface, short int dx=*, short int dy=*)
cdef void scroll24_inplace_c(surface, short int dx, short int dy)

cpdef void scroll24_arr_inplace(
    unsigned char [:, :, :] rgb_array, short int dx=*, short int dy=*)
cdef void scroll24_arr_inplace_c(
    unsigned char [:, :, :] rgb_array, short int dx, short int dy)


cpdef object surface_copy(object surface_)
cdef object surface_copy_c(object surface_)