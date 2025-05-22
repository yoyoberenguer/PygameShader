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
.. _cython_graphics_lib:

Cython Graphics Library
=======================

Key Features
------------
This library provides advanced tools for image manipulation and rendering in graphical applications.
Built on pygame and other low-level rendering techniques, it enables efficient handling of surfaces,
sprites, and pixel-based effects. The key features include:

- Splitting images into smaller blocks for sprite sheets and tile maps.
- Applying dynamic effects like bursts and controlled rebuilding of images.
- Storing and manipulating graphical elements in memory for optimized rendering.
- Enhancing performance in real-time applications through optimized functions.
- Supporting experimental features for advanced graphical effects and debugging.

Library Functional Overview
---------------------------
The library consists of a set of functions that facilitate various graphical transformations
and effects. It focuses on the efficient handling of pixel-based surfaces, allowing developers
to create complex visual effects such as explosions, dynamic surface reconstruction, and optimized
sprite manipulations.

Target Applications
-------------------
This library is particularly useful for:

- Game development: Enabling real-time effects like explosions, animations, and sprite transformations.
- Graphics simulations: Creating dynamic visual effects that require image manipulation.
- Image processing: Splitting, reconstructing, and modifying images for artistic or technical purposes.
- Research and experimentation: Testing new graphical rendering techniques and optimization strategies.

Summary
-------
This library is designed to enhance the capabilities of graphical applications, particularly
in game development and advanced image manipulation. By offering optimized functions for handling
surfaces, splitting images, applying burst effects, and rebuilding images, it provides a flexible
and efficient toolset for developers. Experimental functions add further possibilities for exploring
novel rendering techniques. The library is a valuable resource for those looking to implement complex
graphical transformations efficiently.

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