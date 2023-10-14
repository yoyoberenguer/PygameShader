# cython: binding=False, boundscheck=False, wraparound=False, nonecheck=False, cdivision=True,
# profile=False, initializedcheck=False
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


import warnings
cimport numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=ImportWarning)

cpdef void blit_s(image_, spr_, tuple xy_=*)

# EQU TO BLEND_RGB_ADD
cpdef void blend_add_surface(source_, destination_)
cdef void blend_add_surface_c(source_, destination_)

cpdef void blend_add_array(
    unsigned char [:, :, :] source_,
    unsigned char [:, :, :] destination_
    )
cdef void blend_add_array_c(
    unsigned char [:, :, :] source_array_,
    unsigned char [:, :, :] destination_array_
    )

# BLEND_RGB_ADD FOR ALPHA
cpdef void blend_add_alpha(
    unsigned char [:, :] source_array_,
    unsigned char [:, :] destination_array_
    )
cdef void blend_add_alpha_c(
    unsigned char [:, :] source_array_,
    unsigned char [:, :] destination_array_
    )


# EQU BLEND_RGB_SUB
cpdef void blend_sub_surface(source_, destination_)
cdef void blend_sub_surface_c(source_, destination_)

cpdef void blend_sub_array(
    unsigned char [:, :, :] source_array_,
    unsigned char [:, :, :] destination_array_
    )
cdef void blend_sub_array_c(
    unsigned char [:, :, :] source_array_,
    unsigned char [:, :, :] destination_array_
    )

# BLEND_RGB_MIN
cpdef void blend_min_surface(source_, destination_)
cdef void blend_min_surface_c(source_, destination_)

cpdef void blend_min_array(
    unsigned char [:, :, :] source_array_,
    unsigned char [:, :, :] destination_array_
    )
cdef void blend_min_array_c(
    unsigned char [:, :, :] source_array_,
    unsigned char [:, :, :] destination_array_
    )

# BLEND_RGB_MAX
cpdef void blend_max_surface(source_, destination_)
cdef void blend_max_surface_c(source_, destination_)

cpdef void blend_max_array(
    unsigned char [:, :, :] source_array_,
    unsigned char [:, :, :] destination_array_
    )
cdef void blend_max_array_c(
    unsigned char [:, :, :] source_array_,
    unsigned char [:, :, :] destination_array_
    )

