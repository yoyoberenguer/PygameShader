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

"""
This Python library provides efficient color space conversion functions, 
primarily implemented using Cython for performance. The functions allow
 conversions between RGB and other color models like YIQ, HSL, and HSV.

Purpose of the Library:
The main goal is to facilitate fast and efficient color space conversions, 
which are commonly used in image processing, computer vision, and color correction.

Key Functions and Their Roles:
1. YIQ ↔ RGB Conversions
yiq_2_rgb(y, i, q) → (r, g, b):
Converts a pixel from YIQ (luma, in-phase, quadrature) to RGB color space.
rgb_2_yiq(r, g, b) → (y, i, q):
Converts a pixel from RGB to YIQ color space.
RGB_TO_YIQ_inplace(image_surface, include_y, include_i, include_q):
Converts an entire image from RGB to YIQ in-place, modifying the given image.
RGB_TO_YIQ_inplace_c(w, h, rgb_array, luma, in_phase, quadrature):
Cython-optimized, low-level function for in-place RGB-to-YIQ conversion without Python overhead.
✅ Why YIQ?
YIQ is mainly used in NTSC television broadcasting and for image processing 
applications where luma (brightness) and chroma (color) separation is beneficial.

2. HSL ↔ RGB Conversions
hsl_to_rgb_pixel(h, s, l) → (r, g, b):
Converts a pixel from HSL (Hue, Saturation, Lightness) to RGB.
rgb_pixel_to_hsl(r, g, b) → (h, s, l):
Converts a pixel from RGB to HSL.
✅ Why HSL?
HSL is useful for color adjustments, such as tint, shade, and saturation adjustments in graphics applications.

3. HSV ↔ RGB Conversions
hsv_to_rgb_pixel(h, s, v) → (r, g, b):
Converts a pixel from HSV (Hue, Saturation, Value) to RGB.
rgb_pixel_to_hsv(r, g, b) → (h, s, v):
Converts a pixel from RGB to HSV.
✅ Why HSV?
HSV is widely used in color selection tools and image segmentation because it 
separates chromatic content from brightness.

Optimization Features:
Cython-based (cdef, cpdef, nogil): Improves performance by compiling critical functions into C for speed.
In-place operations (RGB_TO_YIQ_inplace): Reduces memory overhead by modifying arrays directly.
No GIL (nogil): Enables multi-threading in Cython for parallel execution.
Use Cases:
Image processing: Converting images to different color spaces for filtering, thresholding, and analysis.
Computer vision: Color-based object detection (e.g., using HSV).
Graphics applications: Adjusting colors, creating effects, and improving contrast.
Broadcasting & Video Processing: Converting between RGB and YIQ for NTSC signals

"""


import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=ImportWarning)

cdef extern from 'Include/Shaderlib.c':

    packed struct yiq:
        float y;
        float i;
        float q;

    packed struct rgb:
        float r
        float g
        float b

    packed struct hsv:
        float h;
        float s;
        float v;

    packed struct hsl:
        float h
        float s
        float l


    yiq rgb_to_yiq(const float r, const float g, const float b)nogil;
    rgb yiq_to_rgb(const float y, const float i, const float q)nogil;

    hsl struct_rgb_to_hsl(const float r, const float g, const float b)nogil;
    rgb struct_hsl_to_rgb(const float h, const float s, const float l)nogil;

    rgb struct_hsv_to_rgb(const float h, const float s, const float v)nogil;
    hsv struct_rgb_to_hsv(const float r, const float g, const float b)nogil;


cpdef (unsigned char, unsigned char, unsigned char) yiq_2_rgb(const float y, const float i, const float q)
cpdef (float, float, float) rgb_2_yiq(unsigned char r, unsigned char g, unsigned char b)

cpdef void RGB_TO_YIQ_inplace(object image_surface,
    bint include_y = *,
    bint include_i = *,
    bint include_q = *
    )

cdef void RGB_TO_YIQ_inplace_c(
    const Py_ssize_t w,
    const Py_ssize_t h,
    unsigned char [:, :, :] rgb_array,
    bint luma = *,
    bint in_phase = *,
    bint quadrature = *
) nogil


cpdef (unsigned char, unsigned char, unsigned char) hsl_to_rgb_pixel(
        const float h,
        const float s,
        const float l
)nogil

cpdef (float, float, float) rgb_pixel_to_hsl(
        const unsigned char r,
        const unsigned char g,
        const unsigned char b
)nogil

cpdef (unsigned char, unsigned char, unsigned char) hsv_to_rgb_pixel(
        const float h,
        const float s,
        const float v
)nogil

cpdef (float, float, float) rgb_pixel_to_hsv(
        const unsigned char r,
        const unsigned char g,
        const unsigned char b
)nogil