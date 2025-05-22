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
Summary of the Library
This Python library provides fast and efficient image processing functions, 
focusing on blurring, Gaussian filtering, and edge detection. It is implemented 
in Cython for high performance, making it ideal for applications in computer vision,
 graphics, and image preprocessing.

Purpose
The library is designed to perform fast blurring and edge detection, which are essential for:

Noise reduction (smoothing images).
Edge detection (for object recognition and feature extraction).
Graphics effects (motion blur, bloom effects).
Preprocessing for computer vision tasks (image segmentation, filtering).
Main Features
1. Blurring Functions
These functions apply various types of blurring to smooth images:

blur(surface_, npass): Blurs an image surface, where npass controls intensity.
blur3d(rgb_array, npass): Blurs a 3D (RGB) image array.
blur1d(bgr_array, width, height, npass, format_32): Optimized 1D blurring for efficiency.
blur1d_cp(bgr_array, width, height, npass, format_32): Returns a new blurred array instead of modifying in-place.
blur4bloom_c(surface_, npass, tmp_array): Likely used for bloom effects, enhancing bright areas.
Several internal Cython-optimized versions (blur3d_c, blur1d_c, etc.) run without the Global
 Interpreter Lock (GIL) for multi-threading support.

2. Gaussian Filtering
gauss_filter24(surface_): Applies Gaussian blur to a 24-bit image.
gauss_filter32(surface_): Applies Gaussian blur to a 32-bit image.
📌 Why Gaussian Blur?
It helps smooth images before edge detection, reducing false edges caused by noise.

3. Edge Detection (Canny Algorithm)
The Canny algorithm is widely used in computer vision to detect object boundaries.

canny3d(rgb_array, threshold, low, high): Detects edges in an RGB image.
canny1d(bgr_array, width, height, format_32, threshold): Detects edges in a linear 1D image representation for efficiency.
There are also Cython-optimized versions (canny3d_c, canny1d_c) that improve performance using multi-threading.

Optimization Features
Cython-based (cdef, cpdef, nogil) → Direct C-level performance.
In-place operations (blur1d, blur3d) → Saves memory.
Multi-threading (nogil) → Faster execution on multi-core processors.
Optimized memory handling ([::1] memory views) → Reduces Python overhead.
Use Cases
Computer Vision → Object recognition, feature extraction.
Graphics & Gaming → Motion blur, bloom effects.
Image Processing Pipelines → Preprocessing before machine learning models.
Medical Imaging → Enhancing and detecting features in scans.
Conclusion
This library is highly optimized for fast blurring, Gaussian filtering, and edge detection,
making it a great choice for computer vision, graphics, and machine learning applications where speed is critical.

"""



cimport numpy as np

cdef extern from 'Include/Shaderlib.c':
    cdef struct xyz:
        int x;
        int y;
        int z;

cpdef void blur(object surface_, unsigned int npass=*)
cpdef void blur3d(unsigned char [:, :, :] rgb_array, unsigned int npass=*)
# New version 1.0.11
cpdef void blur1d(
        unsigned char [::1] bgr_array,
        const unsigned int width,
        const unsigned int height,
        unsigned int npass=*,
        bint format_32=*
)
# New version 1.0.11
cpdef unsigned char [::1] blur1d_cp(
        const unsigned char [::1] bgr_array,
        const unsigned int width,
        const unsigned int height,
        unsigned int npass=*,
        bint format_32=*)
# New version 1.0.11

cdef void blur4bloom_c(
        object surface_,
        unsigned int npass=*,
        tmp_array=*)
# New version 1.0.11

cdef void blur3d_c(unsigned char [:, :, :] rgb_array, unsigned int npass=*)
cdef void blur1d_c(
        unsigned char [::1] bgr_array,
        const unsigned int width,
        const unsigned int height,
        int npass=*,
        bint format_32=*
)
# New version 1.0.11
cdef unsigned char [::1] blur1d_cp_c(
        const unsigned char [::1] bgr_array,
        const unsigned int width,
        const unsigned int height,
        unsigned int npass=*,
        bint format_32=*)
# New version 1.0.11


cpdef blur3d_cp(unsigned char [:, :, :]rgb_array)
cdef unsigned char [:, :, :] blur3d_cp_c(unsigned char [:, :, :] rgb_array)

# Gauss filtering for Canny
cpdef gauss_filter24(surface_)
cpdef gauss_filter32(surface_)

cpdef canny3d(unsigned char [:, :, :] rgb_array,
              unsigned char threshold = *,
              unsigned char low = *,
              unsigned char high = *
              )
cdef canny3d_c(unsigned char [:, :, :] rgb_array,
               unsigned char threshold = *,
               unsigned char low = *,
               unsigned char high = *
               )

cpdef canny1d(
        unsigned char [::1] bgr_array,
        unsigned int width,
        unsigned int height,
        bint format_32=*,
        unsigned char threshold = *
)

cdef canny1d_c(
        unsigned char [::1] bgr_array,
        unsigned int width,
        unsigned int height,
        bint format_32=*,
        unsigned char threshold = *
)