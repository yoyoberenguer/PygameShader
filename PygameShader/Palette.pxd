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

"""
Library Overview
The functions appear to be part of a library designed for image manipulation and palette management, 
for applications such as graphical rendering, image processing, or creating custom color palettes.
The library may be used in graphics software or visual applications where images need to be rendered 
with specific color palettes, or where images need to be modified dynamically based on palette data.

Image Surface: refers to an object representing an image, where individual pixels can be accessed
and manipulated. Palette: Refers to a collection of colors that can be applied to images, often used in
visual design, digital art, or simulations.Temporary Buffer: This could be used to store intermediate 
results during computations, reducing the need to perform expensive operations multiple times.
Overall, the purpose of the library is to provide tools for manipulating images at the color level, 
either by changing palettes or creating new ones from scratch, optimizing image handling for graphical applications.
"""

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=ImportWarning)

cdef extern from 'Include/Shaderlib.c':

    packed struct rgb:
        float r
        float g
        float b

    rgb struct_hsl_to_rgb(const float h, const float s, const float l)nogil;


cpdef void palette_change(
    object image_surface,
    object color_palette,
    object temp_buffer
)

cdef void palette_change_c(
    const Py_ssize_t w,
    const Py_ssize_t h,
    const Py_ssize_t p_length,
    unsigned char [:, :, :] image_surface,
    float [:, :] color_palette,
    float [:, ::1] temp_buffer
)nogil


cpdef object make_palette(
    const int width,
    const float fh,
    const float fs,
    const float fl)

cdef unsigned int [::1] make_palette_c(
    const int width,
    const float fh,
    const float fs,
    const float fl)

cpdef object create_surface_from_palette(unsigned int [::1] palette_c)


