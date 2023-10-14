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


cdef unsigned char [:, :, ::1] pixel_block_rgb_c(
        unsigned char [:, :, :] array_, int start_x, int start_y,
        int w, int h, unsigned char [:, :, ::1] block) nogil


cdef surface_split_c(surface_, int size_, int rows_, int columns_)

cdef void burst_c(image_, list vertex_array_, int block_size_, int rows_,
          int columns_, int x_, int y_, int max_angle_=*)

cdef void display_burst_c(object screen_, list vertex_array_, unsigned char blend_=*)

cdef void rebuild_from_frame_c(
        object screen_,
        unsigned int current_frame_,
        unsigned int start_frame,
        list vertex_array_,
        unsigned char blend_ = *
)

cdef void burst_into_memory_c(
        unsigned int n_,
        object sg_,
        object screenrect,
        bint warn_ = *,
        bint auto_n_ = *
)

cdef void rebuild_from_memory_c(object screen_, list vertex_array_, unsigned char blend_=*)

"""
<< ---------------------------------- EXPERIMENTAL ------------------------------------------ >>
Using pygame _sdl library 
"""

cdef void burst_experimental_c(
        object render_,
        object image_,
        object group_,
        int block_size_,
        int rows_,
        int columns_,
        int x_, int y_,
        int
        max_angle_=*)


cdef void db_experimental_c(
        object screenrect_,
        object render_,
        object group_,
        bint clean_ = *,
        unsigned char blend_=*)


cdef void rff_experimental_c(
        object render_,
        object screenrect_,
        unsigned int current_frame_,
        unsigned int start_frame,
        object group_,
        unsigned char blend_ = *
)

cdef void rfm_experimental_c(
        object screenrect_,
        object render_,
        object group_,
        unsigned char blend_=*
)

cpdef void build_surface_inplace(
        object surface_,
        object group_,
        unsigned int block_width,
        unsigned int block_height
)


cpdef void build_surface_inplace_fast(
        object surface_,
        object group_,
        unsigned int block_width,
        unsigned int block_height
)