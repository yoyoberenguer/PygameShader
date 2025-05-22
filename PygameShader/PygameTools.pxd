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
Purpose of This Library
=======================

This library is designed for **high-performance image processing** with a focus on **efficiency
and low-level memory management**.
It leverages **Cython and NumPy** to provide fast image operations, including **resizing,
 format conversions, pixel manipulations, and color channel analysis**.
By using memory views and Cython optimizations, the library ensures minimal overhead when
processing images in **RGB and RGBA formats**.

Key Features
============

- **Efficient Image Manipulation**: Functions to resize, transform, and reformat images with minimal processing overhead.
- **Memory-Efficient Buffer Handling**: Converts images to and from compact buffer representations for efficient storage and processing.
- **Pixel-Level Operations**: Supports pixel indexing, blending, and mapping/unmapping between color representations.
- **Alpha Channel Handling**: Enables operations such as alpha removal, binary masking, and blending with transparency.
- **Performance Optimizations**: Uses Cython’s `nogil` operations for multi-threading and fast execution.
- **Comparison and Analysis Tools**: Functions for comparing PNG images and analyzing color channels.

Library Functional Overview
===========================

This library provides several core functionalities for working with images:

Image Resizing & Buffer Manipulation
------------------------------------

- ``resize_array()``: Resize an RGB image efficiently.
- ``bufferize()`` / ``unbufferize()``: Convert images to and from compact buffer formats.
- ``flip_bgra_buffer()`` / ``bgr_buffer_transpose()``: Perform buffer-based transformations.

Pixel Mapping & Indexing
------------------------

- ``mapping_array()`` / ``unmapping_array()``: Convert RGB images to indexed formats and vice versa.
- ``index3d_to_1d()`` / ``index1d_to_3d()``: Handle pixel coordinate transformations.

Alpha Channel Processing
------------------------

- ``RemoveAlpha()``: Strip the alpha channel from an RGBA image.
- ``make_rgba_array()``: Combine an RGB image with an alpha mask.
- ``binary_mask()``: Generate binary masks from alpha values.

Image Blending & Comparison
---------------------------

- ``blend_pixel_mapped_arrays()``: Blend images using mapped pixel values.
- ``compare_png24bit()`` / ``compare_png32bit()``: Compare images for similarity.

Image Analysis
--------------

- ``get_rgb_channel_means()``: Compute average color values per channel.
- ``analyze_image_channels()``: Extract statistical insights from an image.

Target Applications
===================

- **Game Development**: Handling and processing game textures efficiently.
- **Computer Vision**: Preprocessing images for machine learning models.
- **Graphics Software**: Performing transformations, blending, and format conversions.
- **Embedded Systems**: Optimized image handling in memory-constrained environments.
- **Scientific Image Analysis**: Extracting statistics and performing pixel-level computations.

Summary
=======

This **Cython-based image processing library** provides **high-performance image transformations, buffer manipulation, and alpha channel handling**.
With optimized **resizing, mapping, blending, and analysis functions**, it is designed for applications needing **fast, memory-efficient image processing**.
Its **low-level optimizations** make it particularly well-suited for **real-time graphics, computer vision, and game development**.

"""


cimport numpy as np

cdef extern from 'Include/Shaderlib.c':

    packed struct im_stats:
        float red_mean;
        float red_std_dev;
        float green_mean;
        float green_std_dev;
        float blue_mean;
        float blue_std_dev;

    packed struct im_stats_with_alpha:
        float red_mean;
        float red_std_dev;
        float green_mean;
        float green_std_dev;
        float blue_mean;
        float blue_std_dev;
        float alpha_mean;
        float alpha_std_dev;

    packed struct lab:
        float l;
        float a;
        float b;

    packed struct xyz:
        float x;
        float y;
        float z;

    packed struct rgb:
        float r
        float g
        float b


# TODO CHECK NOGIL ? WHY SOME CDEF FUNC DOESN'T HAVE NOGIL


# New version 1.0.11
cpdef np.ndarray[np.uint8_t, ndim=3] RemoveAlpha(
    const unsigned char[:, :, :] rgba_array)


cpdef unsigned char [:, :, ::1] resize_array(
    const unsigned char [:, :, :] rgb_array,
    const Py_ssize_t w2,
    const Py_ssize_t h2)

cdef unsigned char [:, :, ::1] resize_array_c(
    const unsigned char [:, :, :] rgb_array,
    const Py_ssize_t w2,
    const Py_ssize_t h2)


cpdef  unsigned char [:, :, :] make_rgba_array(
    const unsigned char [:, :, :] rgb_,
    const unsigned char [:, :] alpha_,
    bint transpose_ = *
)
cdef  unsigned char [:, :, :] make_rgba_array_c(
    const unsigned char [:, :, :] rgb_,
    const unsigned char [:, :] alpha_,
    bint transpose_ = *
)


cpdef alpha_array_to_surface(const unsigned char[:, :] array)

cdef unsigned int index3d_to_1d(
    const unsigned int x,
    const unsigned int y,
    const unsigned int z,
    const unsigned int width,
    const unsigned short int bytesize
)nogil



cdef (int, int, int) index1d_to_3d(
    const unsigned int index,
    const unsigned int width,
    const unsigned short int bytesize
)nogil

cdef unsigned int vmap_buffer_c(
    const unsigned int index,
    const unsigned int width,
    const unsigned int height,
    const unsigned short int depth
)nogil





# New version 1.0.11
cpdef unsigned char [::1] bufferize(const unsigned char[:, :, :] rgb_array)

# New version 1.0.11
cpdef unsigned char [:, :, :] unbufferize(
    const unsigned char[:] c_buffer,
    const Py_ssize_t width,
    const Py_ssize_t height,
    const Py_ssize_t bit_size
)
# New version 1.0.11


cpdef np.ndarray[np.uint8_t, ndim=1] bgr_buffer_transpose(
    const Py_ssize_t width,
    const Py_ssize_t height,
    const unsigned char [::1] buffer,
    transposed_buffer = *)

# New version 1.0.11
cdef void bgr_buffer_transpose_c(
    const Py_ssize_t width,
    const Py_ssize_t height,
    const unsigned char [::1] buffer,
    unsigned char [::1] transposed_buffer
        )nogil
# New version 1.0.11


cpdef np.ndarray[np.uint8_t, ndim=1] flip_bgra_buffer(
    const Py_ssize_t width,
    const Py_ssize_t height,
    const unsigned char [::1] buffer,
    unsigned char [::1] flipped_buffer)

cdef void flip_bgra_buffer_c(
    const Py_ssize_t width,
    const Py_ssize_t height,
    const unsigned char [::1] buffer,
    unsigned char [::1] flipped_buffer)nogil

cpdef unsigned char [:, :] binary_mask(unsigned char [:, :] mask_alpha_)

cpdef object mask32(
    unsigned char [:, :, :] rgb_,
    unsigned char [:, :] alpha_,
    unsigned char [:, :] mask_alpha_,
)



cpdef object create_rgba_surface(
    const unsigned char [:, :, :] rgb_,
    const unsigned char [:, :] alpha_,
    tmp_array_ = *
)
cdef object create_rgba_surface_c(
    const unsigned char [:, :, :] rgb_,
    const unsigned char [:, :] alpha_,
    tmp_array_ = *
)



cdef object channels_to_surface_c(
    unsigned char [:, :] red_channel,
    unsigned char [:, :] green_channel,
    unsigned char [:, :] blue_channel,
    unsigned char [:, :] alpha_channel,
    output_rgba_buffer = *
)
cdef object channels_to_surface_c(
    unsigned char [:, :] red_channel,
    unsigned char [:, :] green_channel,
    unsigned char [:, :] blue_channel,
    unsigned char [:, :] alpha_channel,
    output_rgba_buffer = *
)

cpdef object compare_png24bit(surface1, surface2)

cpdef object compare_png32bit(surface1, surface2)


cpdef np.ndarray[np.uint8_t, ndim=3] unmapping_array(
    const int [:, :] indexed_array,
    tmp_array_ = *)

cdef void unmapping_array_c(
    const Py_ssize_t w,
    const Py_ssize_t h,
    const int [:, :] indexed_array_,
    unsigned char [:, :, ::1] rgb_array)nogil


cpdef np.ndarray[np.int32_t, ndim=2] mapping_array(
    const unsigned char [:, :, :] rgb_array,
    tmp_array_ = *)

cdef void mapping_array_c(
    const Py_ssize_t w,
    const Py_ssize_t h,
    const unsigned char [:, :, :] rgb_array,
    int [:, ::1] indexed_array)nogil


cpdef void blend_pixel_mapped_arrays(
    unsigned int [:, :] target_pixels,
    const unsigned int [:, :] blend_pixels,
    unsigned char special_flags = *)

cdef void blend_pixel_mapped_arrays_c(
    unsigned int [:, :] target_pixels,
    const unsigned int [:, :] blend_pixels,
    unsigned char special_flags = *
    )nogil

cpdef tuple get_rgb_channel_means(object array3d)
cdef tuple get_rgb_channel_means_c(object array3d)

cpdef im_stats_with_alpha analyze_image_channels (object image_array)


