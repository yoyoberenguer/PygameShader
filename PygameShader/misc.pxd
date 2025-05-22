# cython: binding=False, boundscheck=False, wraparound=False, nonecheck=False, cdivision=True,
# profile=False, initializedcheck=False, exceptval(check=False)
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
# Python Library Summary

The provided code appears to be part of a Python module for image manipulation and processing, 
specifically with an emphasis on **gradient generation**, **image channel manipulation**, 
**color space calculations**, and **sorting algorithms**. Let's break down the key functionality
 and purpose of the different functions and methods in the module:

## 1. **Gradient Generation Functions**
These functions create various types of gradients, which are often used in graphical applications 
for creating smooth transitions between colors.

- **`create_line_gradient_rgb`**: Generates a linear gradient of RGB values between two specified
     colors (`start_rgb` and `end_rgb`).
- **`create_line_gradient_rgba`**: Similar to the above, but for RGBA colors (adds an alpha channel for transparency).
- **`horizontal_rgb_gradient`**: Creates a horizontal gradient in RGB color space.
- **`horizontal_rgba_gradient`**: Creates a horizontal gradient in RGBA color space.
- **`create_radial_gradient`**: Creates a radial gradient that radiates outward from a point, 
    with customizable start and end colors and offsets. A precomputed gradient can be provided to optimize performance.
- **`create_radial_gradient_alpha`**: Similar to `create_radial_gradient`, but specifically for handling RGBA with transparency.
- **`create_quarter_radial_gradient`** and **`create_quarter_radial_gradient_alpha`**: 
These create gradients for a quarter of a circle (sector-based gradients), useful for specialized visual effects.

## 2. **Color Space and Color Matching Functions**
These functions are used to work with different color spaces (HSV, HSL, RGB) and find closest color
 matches, which is useful for color-based image processing or palette generation.

- **`color_dist_hsv`**: Calculates the distance between two colors in the HSV color space.
- **`color_dist_hsl`**: Calculates the distance between two colors in the HSL color space.
- **`closest_hsv_color`** and **`closest_hsl_color`**: Find the closest color in a given palette 
    to a target color, using the HSV or HSL color space respectively.
- **`close_color`**: Finds a color in a palette that is close to a given color (presumably using RGB space).
- **`use_palette`**: Similar to `close_color`, but it likely selects colors from a palette based 
    on a defined selection criterion.

## 3. **Image Manipulation Functions**
These functions focus on manipulating images or buffers, particularly for scrolling, blending, 
and combining channels.

- **`swap_image_channels`**: Swaps color channels (such as RGB) in an image surface, useful when 
    working with different color formats or performing transformations.
- **`scroll_surface_24bit`**, **`scroll24_inplace`**, **`scroll_rgb_array_inplace`**, 
    **`scroll_alpha_inplace`**: These functions allow for scrolling of an image or image channels in 
    the x and y directions, which can be used for animations or visual effects. The `c` versions are 
    likely optimized implementations written in C for performance.
- **`combine_rgba_buffers`**: Combines a BGR image buffer and an alpha channel into a single RGBA image buffer.
     This is useful for working with images that have separate color and alpha channels.
- **`normalize_2d_array`**: Normalizes a 2D array, possibly for color or pixel intensity normalization.

## 4. **Utility Functions**
These functions perform a variety of utility tasks, including sorting and random number generation.

- **`rgb_to_int`** and **`int_to_rgb`**: Convert between RGB values and their integer representation
     (often used for packing/unpacking colors).
- **`_randf`**, **`_randi`**: Generate random float and integer values within a specified range, 
    likely for random color or pixel generation.
- **`bubble_sort`**, **`insertion_sort`**, **`quick_sort`**, **`heap_sort`**: These are different 
    sorting algorithms implemented for sorting arrays of pixel or image data. `heap_sort` and `quick_sort`
     are especially useful for sorting large datasets or pixel values efficiently.
- **`partition_cython`**, **`_quick_sort`**, **`heapify`**: These are helper functions for sorting 
    algorithms like Quick Sort and Heap Sort, optimized for performance in Cython.

## 5. **Image Format and Type Checking**
These functions help determine the type of image data or buffers.

- **`get_image_format`**: Likely checks the format of an image (whether it is in RGBA, RGB, etc.).
- **`is_type_memoryview`**, **`is_uint8`**, **`is_float64`**, **`is_int32`**: These functions check 
the type and datatype of a given image array or buffer, useful for validating input and ensuring the correct data format.

## Overall Purpose of the Library:
The library is designed for **image processing and graphical operations**, with a strong emphasis 
on **gradient creation**, **image channel manipulation**, **color transformations**, **sorting algorithms**,
 and **optimization**. It provides functions to create smooth color transitions (gradients), 
 manipulate pixel data, check image formats, and optimize computationally expensive operations using 
 algorithms like QuickSort and HeapSort, as well as providing support for custom gradient creation 
 (both radial and linear). The functions are highly optimized with `cpdef` and `cdef` to allow seamless
  integration between Python and C for performance-critical tasks.

## Use Cases:
- **Image Processing**: Manipulating and transforming images, handling RGBA/BGR channels, 
    scrolling image pixels, and applying color transformations.
- **Graphics Rendering**: Generating gradients (linear, radial, quarter radial) for background effects 
    or visual transitions.
- **Color Matching**: Finding the closest matching colors in a given color palette based on various 
    color spaces (HSV, HSL).
- **Performance Optimization**: Utilizing sorting and color manipulation algorithms to efficiently
    process large images or datasets, with performance improvements using Cython (`cpdef`/`cdef`).

This library would be useful in applications like **image editors**, **graphic design tools**, 
**data visualization**, or **real-time graphical applications**.

"""

cdef extern from 'Include/Shaderlib.c':

    packed struct hsv:
        float h;
        float s;
        float v;

    packed struct hsl:
        float h
        float s
        float l

    packed struct rgb:
        float r
        float g
        float b

    packed struct rgb_color_int:
        int r;
        int g;
        int b;

    packed struct s_min:
        float value;
        unsigned int index;

    hsl struct_rgb_to_hsl(const float r, const float g, const float b)nogil;
    rgb struct_hsl_to_rgb(const float h, const float s, const float l)nogil;

    rgb struct_hsv_to_rgb(const float h, const float s, const float v)nogil;
    hsv struct_rgb_to_hsv(const float r, const float g, const float b)nogil;

    rgb_color_int wavelength_to_rgb(int wavelength, float gamma)nogil;
    rgb_color_int wavelength_to_rgb_custom(int wavelength, int arr[ ], float gamma)nogil;

    float randRangeFloat(float lower, float upper)nogil;
    int randRange(int lower, int upper)nogil;
    float min_f(float arr[], unsigned int n)nogil;
    s_min minf_struct(float arr[], unsigned int n)nogil;
    unsigned int min_index(float arr[], unsigned int n)nogil;

cimport numpy as np

ctypedef rgb rgb_


cpdef swap_image_channels(image_surface, channel_order)


cpdef np.ndarray[np.uint8_t, ndim=2] create_line_gradient_rgb(
        int num_pixels,
        tuple start_rgb = *,
        tuple end_rgb   = *
)

cpdef np.ndarray[np.uint8_t, ndim=2] create_line_gradient_rgba(
        int num_pixels,
        tuple start_rgba = *,
        tuple end_rgba   = *
)


cpdef object horizontal_rgb_gradient(
        int w,
        int h,
        tuple color_start = *,
        tuple color_end   = *
)
cpdef object horizontal_rgba_gradient (
        int w,
        int h,
        tuple color_start = *,
        tuple color_end   = *
)

cpdef create_radial_gradient(
        const int w,
        const int h,
        float offset_x              = *,
        float offset_y              = *,
        tuple color_start           = *,
        tuple color_end             = *,
        object precomputed_gradient = *,  # Renamed from gradient_array_
        float scale_factor          = *,  # Renamed from factor_
        unsigned short int threads  = *
)


cpdef create_radial_gradient_alpha(
        const int width_,
        const int height_,
        float offset_x              = *,
        float offset_y              = *,
        tuple color_start           = *,
        tuple color_end             = *,
        object precomputed_gradient = *,
        float scale_factor          = *,
        unsigned short int threads  = *
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

cdef float color_dist_hsv(const hsv hsv_1, const hsv hsv_2)nogil
cdef float color_dist_hsl(const hsl hsl_1, const hsl hsl_2)nogil

cdef (unsigned char, unsigned char, unsigned char) closest_hsv_color(
        const rgb target_color,
        const float [:, ::1] color_palette,
        const Py_ssize_t palette_size
)

cdef (unsigned char, unsigned char, unsigned char) closest_hsl_color (
        const rgb target_color,
        const float [:, :] color_palette,
        const Py_ssize_t palette_size
)


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


cpdef object scroll_surface_24bit(surface, const short int dx, const short int dy)
cdef scroll_surface_24bit_c(surface, const short int dx, const short int dy)

cpdef void scroll24_inplace(surface, const short int dx, const short int dy)
cdef void scroll24_inplace_c(surface, const short int dx, const short int dy)

cpdef void scroll_rgb_array_inplace(
    unsigned char [:, :, :] rgb_array, const short int dx, const short int dy)
cdef void scroll_rgb_array_inplace_c(
    unsigned char [:, :, :] rgb_array, const short int dx, const short int dy)

cpdef void scroll_alpha_inplace(
    unsigned char [:, :] alpha_array,
    const short int dx,
    const short int dy
    )
cdef void scroll_alpha_inplace_c(
    unsigned char [:, :] alpha_array,
    const short int dx,
    const short int dy
    )

# ----- 32 BIT TEXTURE
cpdef void scroll32_inplace(surface, const short int dx, const short int dy)
cdef void scroll32_inplace_c(surface, const short int dx, const short int dy)



cpdef unsigned int rgb_to_int(
    const unsigned char red,
    const unsigned char green,
    const unsigned char blue)nogil

cdef unsigned int rgb_to_int_c(
    const unsigned char red,
    const unsigned char green,
    const unsigned char blue
)nogil

cpdef (unsigned char, unsigned char, unsigned char) int_to_rgb(const unsigned int n)nogil
cdef (unsigned char, unsigned char, unsigned char) int_to_rgb_c(const unsigned int n)nogil

cpdef float _randf(const float lower, const float upper)nogil
cdef float _randf_c(const float lower, const float upper)nogil

cpdef int _randi(const int lower, const int upper)nogil
cdef int _randi_c(const int lower, const int upper)nogil


cpdef unsigned char[::1] combine_rgba_buffers(
    const Py_ssize_t w,
    const Py_ssize_t h,
    const unsigned char [::1] bgr_buffer,
    const unsigned char [::1] alpha_channel,
    unsigned char [::1] output_rgba_buffer,
    bint transpose_output=*)

cpdef float [:, :] normalize_2d_array(const unsigned char [:, :] array)

cpdef object generate_spectrum_surface(int width, int height, float gamma=*)
cdef object generate_spectrum_surface_c(int width, int height, float gamma=*)


cdef bint get_image_format(surface_)
cdef bint is_type_memoryview(object object_)
cdef bint is_uint8(rgb_array)
cdef bint is_float64(rgb_array)
cdef bint is_int32(rgb_array)

cpdef void bubble_sort(unsigned char [::1] nums, const unsigned int size)nogil

cpdef void insertion_sort(unsigned char [::1] nums, const unsigned int size)nogil

cpdef int partition_cython(unsigned char [::1] nums, const unsigned int low, const unsigned int high) nogil

cpdef void _quick_sort(unsigned char [::1] items, unsigned int low, unsigned int high)nogil

cpdef void heapify(unsigned char [::1] nums, const unsigned int heap_size, unsigned int root_index) nogil

cpdef void heap_sort(unsigned char [::1] nums, const unsigned int n)nogil
