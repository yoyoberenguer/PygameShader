# cython: binding=False, boundscheck=False, wraparound=False, nonecheck=False, cdivision=True,
# profile=False, initializedcheck=False, exceptval(check=False)
# cython: optimize.use_switch=True
# cython: warn.maybe_uninitialized=False
# cython: warn.unused=False
# cython: warn.unused_result=False
# cython: warn.unused_arg=False
# cython: language_level=3
# cython: write_stub_file = True
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
# **Image Processing Library in Cython**

## **Purpose**
This Cython library is designed for **high-performance image processing and manipulation**. 
It provides a wide range of functions to transform, filter, and enhance images, leveraging Cython's 
ability to combine Python's ease of use with C-like performance. The library is optimized for
real-time or computationally intensive tasks, making it suitable for applications in game development,
computer vision, artistic effects, and scientific visualization.

---

## **Key Functionalities**

### **1. Color Manipulation**
- **Color Space Conversions**:
  - `bgr`, `brg`: Convert images between BGR and BRG color formats.
  - `grey`: Convert images to grayscale.
  - `sepia`: Apply a sepia tone effect.
  - `hsl_effect`, `hsv_effect`: Transform images using HSL (Hue, Saturation, Lightness) and 
        HSV (Hue, Saturation, Value) color spaces.
- **Brightness and Saturation**:
  - `brightness`: Adjust the brightness of an image.
  - `saturation`: Adjust the saturation of an image.
  - `brightness_exclude`, `brightness_bpf`: Apply brightness adjustments with exclusions 
        or based on specific thresholds.
- **Inversion**:
  - `invert`: Invert the colors of an image.

---

### **2. Visual Effects**
- **Filters**:
  - `median`: Apply median filtering for noise reduction.
  - `sobel`: Perform edge detection using the Sobel operator.
  - `bloom`: Add a bloom effect to highlight bright areas.
  - `posterize_surface`: Reduce the number of colors for a posterized effect.
- **Distortions**:
  - `wave`, `swirl`, `fisheye`: Apply wave, swirl, and fisheye distortions.
  - `horizontal_glitch`, `horizontal_sglitch`: Simulate horizontal glitch effects.
- **Artistic Effects**:
  - `painting`, `cartoon`: Apply artistic effects to mimic painting or cartoon styles.
  - `dithering`, `dithering_atkinson`: Apply dithering algorithms to reduce color 
        depth while preserving visual quality.

---

### **3. Advanced Image Processing**
- **Edge Detection and Sharpening**:
  - `sobel`, `sobel_fast`: Detect edges using the Sobel operator.
  - `sharpen`, `sharpen32`: Sharpen images to enhance details.
- **Blending and Compositing**:
  - `blend`, `alpha_blending`: Blend two images with adjustable opacity or blending modes.
- **Heatmap and Predator Vision**:
  - `heatmap`: Generate heatmap visualizations.
  - `predator_vision`: Simulate a predator-like vision effect with edge detection and color mapping.

---

### **4. Physical Simulations**
- **Ripple Effects**:
  - `ripple`, `ripple_seabed`: Simulate ripple effects on water surfaces.
- **Heat Convection**:
  - `heatconvection`: Simulate heat convection effects on images.
- **Plasma Effects**:
  - `plasma`, `plasma_config`: Generate dynamic plasma effects.

---

### **5. Utility Functions**
- **Image Transformations**:
  - `mirroring`: Mirror images horizontally or vertically.
  - `pixelation`: Pixelate images by reducing resolution.
  - `bilinear`: Perform bilinear interpolation for resizing images.
- **Color Mapping**:
  - `wavelength2rgb`, `custom_map`: Convert wavelengths to RGB colors or apply custom color mappings.
- **Performance-Oriented Functions**:
  - Many functions (e.g., `bgr_1d`, `grey_1d`) are optimized for 1D or 3D pixel arrays, 
        enabling efficient processing of large datasets.

---

### **6. Special Effects**
- **TV Scanlines**:
  - `tv_scan`: Simulate old TV scanline effects.
- **Blood Effects**:
  - `blood`: Apply blood-like effects to images.
- **Dirt and Lens Effects**:
  - `dirt_lens`: Simulate dirt or lens distortion effects.

---

### **7. Performance and Optimization**
- **Memory Efficiency**:
  - Functions like `bgr_1d`, `grey_1d`, and `invert1d` are designed to work with 1D or 
        3D arrays, ensuring efficient memory usage.
- **Parallel Processing**:
  - Many functions use `nogil` to release the Global Interpreter Lock (GIL), 
        enabling multi-threaded execution for faster processing.

---

## **Target Use Cases**
This library is ideal for:
- **Game Development**: Real-time image effects (e.g., distortions, filters, blending).
- **Computer Vision**: Preprocessing images for machine learning or analysis.
- **Artistic Applications**: Applying creative effects to images or videos.
- **Scientific Visualization**: Simulating physical phenomena (e.g., heat convection, ripples).

---

## **Summary**
The library is a **powerful and versatile toolkit** for image processing, 
combining **performance optimization** with a **wide range of visual effects and transformations**. 
It is well-suited for applications requiring real-time or high-performance image manipulation, 
such as games, simulations, and computer vision tasks.

"""

"""

## 1. Color Space Conversions

### **BGR/RGB Conversions**
- `bgr`, `bgr_copy`, `bgr_3d`, `bgr_1d`, `bgr_1d_cp`
- `brg`, `brg_copy`, `brg_3d`, `brg_1d`, `brg_1d_cp`

### **Grayscale Conversions**
- `grey`, `grey_copy`, `grey_2d`, `grey_3d`, `grey_1d`, `grey_1d_cp`

### **Sepia Effect**
- `sepia`, `sepia_copy`, `sepia_3d`, `sepia_1d`, `sepia_1d_cp`

### **HSL/HSV Conversions**
- `hsl_effect`, `hsl3d`, `hsl1d`, `hsl1d_cp`
- `hsv_effect`, `hsv3d`, `hsv1d`, `hsv1d_cp`

### **Brightness and Saturation**
- `brightness`, `brightness3d`, `brightness1d`, `brightness1d_copy`, `brightness_copy`, `brightness_exclude`, `brightness_bpf`
- `saturation`, `saturation3d`, `saturation1d`, `saturation1d_cp`

### **Other Color Effects**
- `convert_27`, `Luma_GreyScale`

---

## 2. Image Filtering and Effects

### **Median Filter**
- `median`, `median_fast`, `median_grayscale`

### **Posterization**
- `posterize_surface`

### **Edge Detection (Sobel)**
- `sobel`, `sobel_1d`, `sobel_fast`

### **Inversion**
- `invert`, `invert_copy`, `invert3d`, `invert1d`, `invert1d_cp`

### **Wave and Swirl Effects**
- `wave`, `wave32`, `wave_static`
- `swirl`, `swirl32`, `swirlf`

### **Plasma Effect**
- `plasma_config`, `plasma`

### **Heat Convection**
- `heatconvection`

### **Glitch Effects**
- `horizontal_glitch`, `horizontal_sglitch`

### **Bloom Effect**
- `bloom`

### **Dirt Lens Effect**
- `dirt_lens`

### **Sharpen**
- `sharpen`, `sharpen_1d`, `sharpen_1d_cp`, `sharpen32`

### **Emboss**
- `emboss`, `emboss_inplace`, `emboss1d`, `emboss_gray`

### **Bilateral Filter**
- `bilateral`

### **Cartoon Effect**
- `cartoon`

### **Chromatic Aberration**
- `chromatic`, `chromatic_inplace`

### **Zoom**
- `zoom`, `zoom_inplace`

### **Heatmap**
- `heatmap`

### **Predator Vision**
- `predator_vision`

### **Blood Effect**
- `blood`

### **Mirroring**
- `mirroring_array`, `mirroring`

### **Dithering**
- `dithering`, `dithering_inplace`, `dithering1d`, `dithering1d_cp`, `dithering_atkinson`, `dithering_atkinson1d`

### **Pixelation**
- `pixelation`

### **Blending**
- `blend`, `blend1d`, `blend_inplace`, `alpha_blending`, `alpha_blending_inplace`

---

## 3. Image Manipulation and Transformation

### **Bilinear Transformation**
- `bilinear`

### **Tunnel Effects**
- `tunnel_modeling24`, `tunnel_render24`, `tunnel_modeling32`, `tunnel_render32`

### **Split Channels**
- `split_channels`, `split_channels_inplace`

### **Ripple Effects**
- `ripple`, `ripple_seabed`

### **Dampening Effects**
- `dampening`, `lateral_dampening`

### **Fisheye Effect**
- `fisheye_footprint`, `fisheye_footprint_param`, `fisheye`

### **TV Scan Effect**
- `tv_scan`

### **Render Light Effects**
- `render_light_effect24`, `area24_cc`

### **Heatwave Effect**
- `heatwave_array24_horiz_c`

---

## 4. Utility and Helper Functions

### **Color Mapping**
- `wavelength2rgb`, `custom_map`, `blue_map`, `bluescale`, `red_map`, `redscale`

### **Shader Effects**
- `shader_bloom_fast`, `shader_bloom_fast1`

---

## 5. Special Effects and Shaders

### **Plasma**
- `plasma_config`, `plasma`

### **Heat Convection**
- `heatconvection`

### **Glitch Effects**
- `horizontal_glitch`, `horizontal_sglitch`

### **Bloom**
- `bloom`

### **Chromatic Aberration**
- `chromatic`, `chromatic_inplace`

### **Zoom**
- `zoom`, `zoom_inplace`

### **Heatmap**
- `heatmap`

### **Predator Vision**
- `predator_vision`

### **Blood Effect**
- `blood`

### **Mirroring**
- `mirroring_array`, `mirroring`

### **Dithering**
- `dithering`, `dithering_inplace`, `dithering1d`, `dithering1d_cp`, `dithering_atkinson`, `dithering_atkinson1d`

### **Pixelation**
- `pixelation`

### **Blending**
- `blend`, `blend1d`, `blend_inplace`, `alpha_blending`, `alpha_blending_inplace`

"""




"""
# VERSION 1.0.11
"""



import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=ImportWarning)

try:
    import numpy
    from numpy import empty, uint8, int16, float32, asarray, linspace, \
        ascontiguousarray, zeros, uint16, uint32, int32, int8
except ImportError:
    raise ImportError("\n<numpy> library is missing on your system."
          "\nTry: \n   C:\\pip install numpy on a window command prompt.")

cimport numpy as np

try:
    cimport cython
    from cython.parallel cimport prange, parallel

except ImportError:
    raise ImportError("\n<cython> library is missing on your system."
          "\nTry: \n   C:\\pip install cython on a window command prompt.")

# PYGAME IS REQUIRED

import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

try:
    import pygame
    from pygame import Color, Surface, SRCALPHA, RLEACCEL, BufferProxy, HWACCEL, HWSURFACE, \
    QUIT, K_SPACE, BLEND_RGB_ADD, Rect, BLEND_RGB_MAX, BLEND_RGB_MIN
    from pygame.surfarray import pixels3d, array_alpha, pixels_alpha, array3d, \
        make_surface, blit_array, pixels_red, \
    pixels_green, pixels_blue
    from pygame.image import frombuffer, fromstring 
    from pygame.math import Vector2
    from pygame.transform import scale, smoothscale, rotate, scale2x
    from pygame.pixelcopy import array_to_surface

except ImportError:
    raise ImportError("\n<Pygame> library is missing on your system."
          "\nTry: \n   C:\\pip install pygame on a window command prompt.")

try:
    # python > 2.1.3
    from pygame.image import tobytes 
except:
    from pygame.image import tostring as tobytes

try:
    cimport cython
    from cython.parallel cimport prange
    from cpython cimport PyObject_CallFunctionObjArgs, PyObject, \
        PyList_SetSlice, PyObject_HasAttr, PyObject_IsInstance, \
        PyObject_CallMethod, PyObject_CallObject
    from cpython.dict cimport PyDict_DelItem, PyDict_Clear, PyDict_GetItem, PyDict_SetItem, \
        PyDict_Values, PyDict_Keys, PyDict_Items
    from cpython.list cimport PyList_Append, PyList_GetItem, PyList_Size, PyList_SetItem
    from cpython.object cimport PyObject_SetAttr

except ImportError:
    raise ImportError("\n<cython> library is missing on your system."
          "\nTry: \n   C:\\pip install cython on a window command prompt.")

from PygameShader.misc cimport get_image_format, is_uint8, is_float64, is_int32
from PygameShader.misc cimport is_type_memoryview
from PygameShader.BlendFlags import blend_add_array
from PygameShader.BlendFlags cimport blend_add_surface_c, blend_add_array_c
from PygameShader.PygameTools cimport resize_array_c, make_rgba_array_c
from PygameShader.gaussianBlur5x5 cimport blur3d_c, blur3d_cp_c, blur, blur4bloom_c, blur1d_cp_c, blur1d_c

cimport numpy as np

from libc.math cimport sqrtf as sqrt, atan2f as atan2, sinf as sin,\
    cosf as cos, nearbyintf as nearbyint, expf as exp, powf as pow, floorf as floor, \
roundf as round_c, fminf as fmin, fmaxf as fmax, rintf, fmodf

from libc.stdlib cimport malloc, rand, free
from libc.string cimport memcpy



from libc.stdio cimport printf
from PygameShader.config import OPENMP, THREAD_NUMBER, __VERSION__

cdef int THREADS = 1

if OPENMP:
     THREADS = THREAD_NUMBER

DEF SCHEDULE = 'static'

cdef float M_PI = 3.14159265358979323846
cdef float INV_M_PI = 1.0 / M_PI
cdef float M_PI2 =3.14159265358979323846/2.0
cdef float M_2PI =2 * 3.14159265358979323846

cdef float RAD_TO_DEG=<float>(180.0/M_PI)
cdef float DEG_TO_RAD=<float>(M_PI/180.0)

cdef float C1 = <float>7.0/<float>16.0
cdef float C2 = <float>3.0/<float>16.0
cdef float C3 = <float>5.0/<float>16.0
cdef float C4 = <float>1.0/<float>16.0

cdef float C1_ = <float>1.0 / <float>sqrt(M_2PI)


DEF HALF         = 1.0/2.0
DEF ONE_THIRD    = 1.0/3.0
DEF ONE_FOURTH   = 1.0/4.0
DEF ONE_FIFTH    = 1.0/5.0
DEF ONE_SIXTH    = 1.0/6.0
DEF ONE_SEVENTH  = 1.0/7.0
DEF ONE_HEIGHT   = 1.0/8.0
DEF ONE_NINTH    = 1.0/9.0
DEF ONE_TENTH    = 1.0/10.0
DEF ONE_ELEVENTH = 1.0/11.0
DEF ONE_TWELVE   = 1.0/12.0
DEF ONE_32       = 1.0/32.0
DEF ONE_64       = 1.0/64.0
DEF ONE_128      = 1.0/128.0
DEF ONE_255      = 1.0/255.0
DEF ONE_360      = 1.0/360.0
DEF TWO_THIRD    = 2.0/3.0
DEF ONE_1024     = 1.0/1024
DEF ONE_65025    = 1.0/65025

cdef float[360] COS_TABLE
cdef float[360] SIN_TABLE
cdef int ANGLE

with nogil:
    for ANGLE in prange(0, 360):
        COS_TABLE[ANGLE] = <float>cos(<float>(ANGLE * DEG_TO_RAD))
        SIN_TABLE[ANGLE] = <float>sin(<float>(ANGLE * DEG_TO_RAD))

COLORS_CPC64 = numpy.array(
    [[0, 0, 0],
    [0, 0, 128],
    [0, 0, 255],
    [128, 0, 0],
    [128, 0, 128],
    [128, 0, 255],
    [255, 0, 0],
    [255, 0, 128],
    [255, 0, 255],
    [0, 128, 0],
    [0, 128, 128],
    [0, 128, 255],
    [128, 128, 0],
    [128, 128, 128],
    [128, 128, 255],
    [255, 128, 0],
    [255, 128, 128],
    [255, 128, 255],
    [0, 255, 0],
    [0, 255, 128],
    [0, 255, 255],
    [128, 255, 0],
    [128, 255, 128],
    [128, 255, 255],
    [255, 255, 0],
    [255, 255, 128],
    [255, 255, 255]], dtype=numpy.uint8)


print("PygameShader version %s " % __VERSION__)
print("OPENMP %s " % OPENMP)
print("THREADS %s " % THREADS)
print("SCHEDULE %s " % SCHEDULE)

cdef:
    float [:, :] COLORS_CPC64_C = numpy.divide(COLORS_CPC64, <float>255.0).astype(dtype=float32)





@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef inline void bgr(object surface_):
    """  
    Convert an image from RGB(A) to BGR(A) format (in-place).
  
    Converts the game display, image, or surface from RGB(A) to BGR(A) format. 
    The alpha channel will be ignored in the process, but it is retained in case 
    of a 32-bit surface.

    RGB is commonly used in image editing and display applications, where the 
    order is red, green, and blue. On the other hand, BGR is often used in image 
    processing applications, where the order is blue, green, and red.

    **Note**: This function operates in-place, meaning it modifies the original surface.

    **Example usage**:
        bgr(surface)

    :param surface_: 
        Pygame surface (either display or image) with a compatible format 
        (24-bit or 32-bit, with or without transparency/alpha channel).
        
    :return: 
        void; modifies the surface in-place.
    """
    # Assert that the input is a valid Pygame Surface
    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)

    cdef unsigned char [:, :, :] rgb_array
    try:

        # Retrieve the pixel data view from the surface in RGB format.
        # This assumes the surface is either 24-bit (RGB) or 32-bit (RGBA).
        rgb_array = surface_.get_view('3')


    except Exception as e:
        # Handle potential errors with providing the image format.
        print("Surface is format 32-bit? %s" % get_image_format(surface_))
        raise ValueError(
            "\nCannot reference source pixels into a 3d array.\n %s " % e)
    # Call the 'bgr_c' function to perform the actual RGB to BGR conversion.
    bgr_c(rgb_array)



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef bgr_copy(object surface_):
    """  
    Convert an image format from RGB(A) to BGR(A) and return a new surface (copy).

    This function converts the pixel data of an input image from RGB(A) to BGR(A) format.
    The Alpha channel (if present) will be retained, but the order of the color channels 
    is switched from RGB to BGR. This is useful when working with image processing libraries 
    that expect the BGR format (such as OpenCV), while the RGB format is commonly used in 
    display and image editing applications.

    e.g:
    new_surface = bgr_copy(surface)

    :param surface_: 
        A Pygame Surface object representing the image. The surface can be in 24-bit or 
        32-bit format (with or without an alpha channel). The function assumes the image 
        is in RGB(A) format and will convert it to BGR(A).

    :return: 
        Returns a new Pygame Surface object with the converted BGR(A) pixel format.

    """
    # Ensure the input is a valid Pygame surface
    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)

    # Declare variables to store surface dimensions and bit size
    cdef:
        Py_ssize_t w, h       # Image width and height
        int bit_size          # Number of bytes per pixel (bit depth)

    # Get the width and height of the surface
    w, h = surface_.get_size()

    # Get the bit size of the surface (either 3 for RGB or 4 for RGBA)
    bit_size = surface_.get_bytesize()

    # Try to obtain the pixel buffer from the surface
    cdef unsigned char [:] rgb_array
    try:
        rgb_array = surface_.get_buffer()  # Get the underlying pixel data as a memory view

    except Exception as e:
        # If an error occurs during buffer extraction, print the error message and raise a ValueError
        print("Surface is format 32-bit? %s" % get_image_format(surface_))
        raise ValueError(
            "\nCannot reference source pixels into a 1d memoryview slice.\n %s " % e)

    # Create and return a new surface with the pixel data in BGR(A) format
    # If bit_size is 4 (RGBA), use "BGRA" format; otherwise, use "BGR" format
    return pygame.image.frombuffer(rgb_array, (w, h), "BGRA" if bit_size == 4 else "BGR")


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef inline void bgr_3d(unsigned char [:, :, :] rgb_array):
    """
    Convert an RGB(A) array (shape: w, h, n) with uint8 data type to BGR(A) format (inplace).

    This function directly processes a 3D array (such as an image or surface) from RGB(A) format 
    to BGR(A) format. It assumes that the input array is in RGB or RGBA format, and it switches 
    the red and blue channels to convert it to BGR or BGRA format respectively. The operation 
    is done in place, modifying the original array.

    RGB is the common color order used in display and image editing applications, where the color 
    channels are arranged as red, green, blue. On the other hand, BGR is often used in image 
    processing applications, where the color channels are arranged as blue, green, red.

    e.g:
    bgr_3d(array)

    Parameters
    ----------
    rgb_array : numpy.ndarray
        A 3D numpy array of shape (w, h, n), where `w` is the width, `h` is the height, 
        and `n` is the number of channels (3 for RGB, 4 for RGBA). The array should have 
        uint8 data type (values between 0 and 255), and contain pixel data in RGB(A) format.

    Returns
    -------
    void
        This function modifies the input array in place and does not return any value.

    Raises
    ------
    ValueError
        If the input array shape is not compatible with RGB(A) (i.e., it does not have 
        the shape (w, h, 3) or (w, h, 4)).

    TypeError
        If the input array does not have the uint8 (unsigned byte) data type.
    """

    cdef:
        Py_ssize_t w, h  # Width and height of the image
        Py_ssize_t bit_size = 0  # Number of color channels (3 or 4)

    try:
        # Extract shape dimensions (width, height, and number of channels)
        w, h, bit_size = rgb_array.shape[:3]

        # Check if the array has 3 or 4 channels (RGB or RGBA)
        if bit_size not in (3, 4):
            raise ValueError('\nIncorrect bit_size, only RGB(A) is supported.')

    except ValueError as e:
        # Handle ValueError and print array flags for debugging
        if is_type_memoryview(rgb_array):
            print(numpy.array(rgb_array).flags)
        else:
            print(rgb_array.flags)

        raise ValueError(
            "\n%s\nExpecting array shape (w, h, n), RGB(A), got (%s, %s, %s)" % (e, w, h, len(rgb_array[:3])))

    # Ensure the array is of uint8 data type (unsigned bytes)
    if not is_uint8(rgb_array):
        raise TypeError("\nExpecting uint8 (unsigned char) data type, got %s" % rgb_array.dtype)

    # Perform the BGR conversion in place by calling the helper function
    bgr_c(rgb_array)



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef inline void bgr_1d(unsigned char [::1] rgb_array, bint format_32=False):
    """
    Convert a 1D array of uint8 data type from RGB(A) to BGR(A) format (inplace).

    This function processes a 1D array directly, converting the color channels from 
    the RGB(A) format to the BGR(A) format. The conversion is done in place, modifying 
    the original array. The `format_32` flag determines whether the input is in RGB 
    (24-bit) or RGBA (32-bit) format.

    RGB is the standard color order used in many image editing and display applications, 
    where the order of the color channels is red, green, and blue. In contrast, BGR is 
    often used in image processing, where the color channels are arranged as blue, 
    green, and red.

    Example usage:
    bgr_1d(array)

    Parameters
    ----------
    rgb_array : numpy.ndarray or bytearray
        A 1D array (or buffer) of pixel data in RGB(A) format, with uint8 data type 
        (values between 0 and 255). The array length should be a multiple of 3 (for RGB) 
        or 4 (for RGBA). The array contains the pixel color values that will be converted 
        from RGB(A) to BGR(A) format.

    format_32 : bool, optional
        A boolean flag indicating the format of the input array. 
        - `True` indicates the array is in 'RGB' (24-bit, 3 channels).
        - `False` (default) indicates the array is in 'RGBA' (32-bit, 4 channels).

    Returns
    -------
    void
        The function modifies the input array in place and does not return any value.

    Raises
    ------
    TypeError
        If the input array does not have a `uint8` data type.
    """

    # Ensure the array is of uint8 data type (values between 0 and 255)
    if not is_uint8(rgb_array):
        raise TypeError("\nExpecting uint8 (unsigned char) data type, got %s" % rgb_array.dtype)

    # Call the underlying C function to perform the BGR conversion in place
    bgr_1d_c(rgb_array, format_32)


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef inline np.ndarray[np.uint8_t, ndim=1] bgr_1d_cp(
        unsigned char [::1] rgb_array, bint format_32=False):
    """
    Convert a 1D array from RGB(A) to BGR(A) format and return a new copy.

    This function takes a 1D array (or memoryview slice) that contains pixel data in 
    RGB(A) order and produces a new 1D array with the color channels reordered to 
    BGR(A). This conversion is useful when interfacing with libraries or routines 
    that expect pixels in BGR(A) format instead of the more common RGB(A) order.

    The function assumes that the input array is of type uint8, and its length should 
    be a multiple of 3 (for RGB data) or 4 (for RGBA data). The `format_32` flag indicates 
    whether the input contains 32-bit pixels (True for RGBA, False for RGB).

    **Example:**
    ```python
    new_bgr_array = bgr_1d_cp(rgb_array)
    ```

    ### Parameters:
    - **bgr_array** (*numpy.ndarray* or *memoryview slice*, shape `(w,)`, dtype `uint8`):
      A 1D array or buffer containing pixel data in RGB(A) order.  
      (For instance, if the image is RGB, the array length should be 3 * number_of_pixels.)

    - **format_32** (*bool*, optional):
      A flag indicating the pixel format:
        - `False` (default): the input is assumed to be 24-bit (RGB, 3 channels).
        - `True`: the input is assumed to be 32-bit (RGBA, 4 channels).

    ### Returns:
    - **numpy.ndarray**:  
      A new 1D array (uint8) with the pixel data converted to BGR(A) order.
    """

    # Only uint8 data is compatible
    if not is_uint8(rgb_array):
        raise TypeError("\nExpecting uint8 (unsigned char) data type got %s" % rgb_array.dtype)

    return bgr_1d_cp_c(rgb_array, format_32)




# ------------ RGB TO BRG

@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef inline void brg(object surface_):
    """
    Convert a Pygame surface from RGB(A) to BRG(A) format in-place.

    This function modifies the given surface by swapping the red and green color channels,
    converting an image from RGB(A) order to BRG(A) order. The alpha channel, if present,
    is preserved but ignored during the conversion process.

    **Example Usage:**
        brg(surface)

    ### Parameters:
    - **surface_** (*pygame.Surface*):  
      A Pygame surface or display surface compatible with 24-bit (RGB) or 32-bit (RGBA) pixel formats.
      The function operates directly on the provided surface.

    ### Returns:
    - **None**:  
      The function modifies the input surface in-place and does not return a new surface.
      
    """
    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)

    cdef unsigned char [:, :, :] rgb_array
    try:
        '3 returns a (surface-w, surface-h, 3) array of RGB color components. ' \
        'Each of the red, green, and blue components are unsigned bytes. Only 24-bit' \
        ' and 32-bit surfaces are supported. The color components must be in either RGB' \
        ' or BGR order within the pixel.'

        rgb_array = surface_.get_view('3')

    except Exception as e:

        print("Surface is format 32-bit? %s" % get_image_format(surface_))
        raise ValueError(
            "\nCannot reference source pixels into a 3d array.\n %s " % e)

    # Not needed since the brg_c process ignore the channel alpha
    # cdef image_format = get_image_format(surface_)

    brg_c(rgb_array)



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef brg_copy(object surface_):
    """
    Convert an image from RGB(A) to BRG(A) format and return a new copy.

    This function swaps the red and green channels of an image while preserving 
    the blue channel. It creates and returns a new surface with the modified 
    color format, leaving the original surface unchanged.

    ### Example Usage:
        brg_surface = brg_copy(surface)

    ### Parameters:
    - **surface_** (*pygame.Surface*):  
      A Pygame surface or display surface in 24-bit (RGB) or 32-bit (RGBA) format.  
      The alpha channel (if present) will be ignored.

    ### Returns:
    - **pygame.Surface**:  
      A new surface with the color channels converted to BRG format.
    """

    # Ensure the input `surface_` is a valid Pygame Surface
    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)

    # Declare Cython variables for efficiency
    cdef:
        unsigned char [::1] bgr_array  # 1D memoryview slice for pixel data
        Py_ssize_t w, h, bit_size      # Image dimensions and bit depth

    # Get width and height of the Pygame Surface
    w, h = surface_.get_size()

    # Get the bit depth (bytes per pixel)
    bit_size = surface_.get_bytesize()

    try:
        # Attempt to retrieve the raw pixel buffer of the Pygame Surface
        bgr_array = surface_.get_buffer()

    except Exception as e:
        # If getting the buffer fails, check the surface format and raise an error
        print("Surface is format 32-bit? %s" % get_image_format(surface_))
        raise ValueError(
            "\nCannot reference source pixels into a 1D memoryview slice array.\n %s " % e
        )

    # Convert the processed BGR array back into a Pygame image Surface
    # - `brg_1d_cp_c()` processes the buffer
    # - If bit depth is 4 bytes per pixel, assume an "RGBA" format; otherwise, assume "RGB".
    return pygame.image.frombuffer(
        brg_1d_cp_c(bgr_array, format_32=True if bit_size == 4 else False),
        (w, h), 
        "RGB" if bit_size == 3 else "RGBA"
)






@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef inline void brg_3d(unsigned char [:, :, :] rgb_array):
    """
    Convert a 3D array from RGB(A) to BRG(A) format (inplace).
    
    This function swaps the red and green channels while preserving the blue 
    and alpha channels (if present). It modifies the input array directly 
    without creating a copy.
    
    ### Example Usage:
        brg_3d(rgb_array)
    
    ### Parameters:
    - **rgb_array** (*numpy.ndarray*, shape *(w, h, n)*, dtype *uint8*):  
      A 3D array representing an image, where pixel values range from 0 to 255.  
      The last dimension *(n)* must be 3 (RGB) or 4 (RGBA).
    
    ### Returns:
    - **None**:  
      The function operates inplace and does not return a new array.
    """

    # Declare variables using Cython's cdef for performance optimization
    cdef:
        Py_ssize_t w, h            # Width and height of the image array
        Py_ssize_t bit_size = 0    # Bit size (number of channels, e.g., 3 for RGB, 4 for RGBA)

    try:
        # Extract the width, height, and bit size (number of channels) from the RGB array's shape
        w, h, bit_size = rgb_array.shape[:3]

        # Check if the image has either 3 (RGB) or 4 (RGBA) channels, raise error if not
        if bit_size not in (3, 4):
            raise ValueError('\nIncorrect bit_size, support only RGB(A)')

    except ValueError as e:
        # If a ValueError occurs, check whether the array is a memoryview or a numpy array
        if is_type_memoryview(rgb_array):
            # For memoryviews, print flags of the numpy array (viewable properties like contiguous in memory)
            print(numpy.array(rgb_array).flags)
        else:
            # For regular numpy arrays, print flags for debugging
            print(rgb_array.flags)

        # Raise a more descriptive ValueError with the dimensions of the array
        raise ValueError(
            "\n%s\nExpecting array shape (w, h, n), "
            "RGB(A) got (%s, %s, %s)" % (e, w, h, len(rgb_array[:3])))

    # Check if the data type is uint8 (unsigned 8-bit integer), which is required for compatibility
    if not is_uint8(rgb_array):
        raise TypeError("\nExpecting uint8 (unsigned char) data type got %s" % rgb_array.dtype)

    # Process the RGB array (likely converting from RGB to BGR or applying GPU operations)
    brg_c(rgb_array)




@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef void brg_1d(unsigned char [::1] rgb_array, bint format_32=False):
    """
    Converts a 1D array of RGB(A) pixel data to BRG(A) format in-place.

    This function swaps the red and blue channels of an input array or buffer 
    representing RGB or RGBA pixel data. It works on both 24-bit (RGB) and 
    32-bit (RGBA) formats.

    Parameters
    ----------
    rgb_array : numpy.ndarray or bytearray
        A 1D array of shape (w,) containing uint8 pixel data (values 0-255).
        Can be either a NumPy array or a raw byte buffer.

    format_32 : bool, optional (default: False)
        - True: Treats the input as an RGB (24-bit) buffer.
        - False: Treats the input as an RGBA (32-bit) buffer.

    Returns
    -------
    None
        The operation is performed in-place, modifying `rgb_array` directly.

    Example
    -------
    ```python
    brg_1d(rgb_array)  # Converts an RGB(A) buffer to BRG(A)
    ```
    """

    # Only uint8 data is compatible
    if not is_uint8(rgb_array):
        raise TypeError("\nExpecting uint8 (unsigned char) data type got %s" % rgb_array.dtype)

    brg_1d_c(rgb_array, format_32)


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef inline np.ndarray[np.uint8_t, ndim=1] brg_1d_cp(
        const unsigned char [::1] bgr_array, bint format_32=False):
    """
    Converts a 1D array of uint8 BGR(A) pixel data to BRG(A) format and returns a new array.

    Unlike the in-place version (`brg_1d`), this function creates and returns a 
    new array with the red and blue channels swapped.

    BRG stands for Blue, Red, Green.

    Example
    -------
    ```python
    bgr_array = brg_1d_cp(bgr_array)  # Converts an BGR(A) buffer to BRG(A)
    ```

    Parameters
    ----------
    bgr_array : numpy.ndarray or bytearray
        A 1D array of shape (w,) containing uint8 pixel data (values 0-255).
        Can be either a NumPy array or a raw byte buffer.

    format_32 : bool, optional (default: False)
        - True: Treats the input as an BGR (24-bit) buffer.
        - False: Treats the input as an BGRA (32-bit) buffer.

    Returns
    -------
    numpy.ndarray
        A new array of shape (w,) with the BRG(A) pixel format (copied).
        
    """

    # Only uint8 data is compatible
    if not is_uint8(bgr_array):
        raise TypeError("\nExpecting uint8 (unsigned char) data type got %s" % bgr_array.dtype)

    return brg_1d_cp_c(bgr_array, format_32)


# ------------ GRAYSCALE

@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef inline void grey(object surface_):
    """
    Convert an image to grayscale while preserving luminosity (in-place).

    A grayscale image has a single channel representing pixel intensity or brightness,
    where pixel values range from 0 (black) to 255 (white). This function computes the
    grayscale values based on luminosity, preserving perceived brightness from the original color image.

    Example Usage:
    --------------
    grey(surface)

    Parameters
    ----------
    surface_ : 
        pygame.Surface
        A Pygame surface or display surface compatible object, with an image in 
        24-bit or 32-bit format. The surface may include transparency or an alpha 
        channel.

    Returns
    -------
    None
        The function modifies the input surface in place and does not return a new surface.
    """

    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)

    cdef unsigned char [:, :, :] rgb_array
    try:
        # '3' returns a (surface-w, surface-h, 3) array of RGB color components.
        # Each of the red, green, and blue components are unsigned bytes.
        # Only 24-bit and 32-bit surfaces are supported.
        # The color components must be in either RGB or BGR order within the pixel.
        rgb_array = surface_.get_view('3')

    except Exception as e:
        raise ValueError("\nCannot reference source pixels into a 3d array.\n %s " % e)

    grey_c(rgb_array)



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef grey_copy(object surface_):
    """
    Convert an image to grayscale while preserving luminosity and return a new surface.

    A grayscale image has a single channel representing pixel intensity or brightness, 
    with pixel values typically ranging from 0 (black) to 255 (white). This function 
    converts the original image to grayscale based on luminosity, preserving perceived 
    brightness from the color image. The alpha channel is preserved in images with 
    transparency (RGBA format), but it is not altered.

    Example Usage:
    --------------
    im = grey_copy(surface)

    Parameters
    ----------
    surface_ : pygame.Surface
        A Pygame surface or display surface compatible object, with an image in 
        24-bit or 32-bit format. The surface may include transparency (alpha channel).

    Returns
    -------
    pygame.Surface
        A new surface object containing the grayscale image. The original surface remains unchanged.
    """

    # Ensure the input `surface_` is a valid Pygame Surface object
    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)

    # Declare Cython variables for optimized memory usage
    cdef:
        unsigned char [::1] bgr_buffer  # 1D memory view for pixel buffer (BGR format)
        Py_ssize_t w, h                 # Image dimensions: width (w) and height (h)
        int bit_size                    # Image bit depth (number of channels)

    # Retrieve the width and height of the Pygame Surface
    w, h = surface_.get_size()

    # Get the bit size of the surface (bytes per pixel)
    bit_size = surface_.get_bytesize()

    try:
        # Try to get the raw pixel buffer from the Pygame Surface (this is the pixel data)
        bgr_buffer = surface_.get_buffer()

    except Exception as e:
        # If an error occurs, raise a more descriptive exception
        raise ValueError("\nCannot reference source pixels into a 1d memoryviewslice array.\n %s " % e)

    # Process the raw buffer using `grey_1d_cp_c` function (likely for a grayscale conversion or format change)
    # - The function processes the buffer, with `format_32=True` if the bit size is 4 (RGBA), else False for RGB
    # Convert the processed buffer back into a Pygame image using `pygame.image.frombuffer`
    # - If bit size is 4, assume it's "RGBA", otherwise "RGB"
    return pygame.image.frombuffer(
        grey_1d_cp_c(bgr_buffer, format_32=True if bit_size == 4 else False),
        (w, h), 
        "RGBA" if bit_size == 4 else "RGB"
    )





@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef np.ndarray[np.uint8_t, ndim=2] grey_2d(surface_):
    """
    Convert an image into a 2D grayscale array.

    A grayscale image has a single channel representing pixel intensity or brightness,
    with pixel values typically ranging from 0 (black) to 255 (white). This function 
    converts the input image to grayscale based on luminosity, preserving the intensity 
    from the original color image. The alpha channel, if present, will be ignored in the 
    output.

    Example Usage:
    --------------
    gray = grey_2d(surface)

    Parameters
    ----------
    surface_ : pygame.Surface
        A Pygame surface or display surface-compatible object, with an image in 
        24-bit or 32-bit format. The surface may include transparency (alpha channel),
        which will be ignored during the conversion.

    Returns
    -------
    numpy.ndarray
        A 2D NumPy array containing the grayscale image data. The array will have 
        shape (w, h) and dtype uint8, where each value represents pixel intensity.
    """

    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)

    cdef unsigned char [:, :, :] rgb_array
    try:
        # '3' returns a (surface-w, surface-h, 3) array of RGB color components.
        # Each of the red, green, and blue components are unsigned bytes.
        # Only 24-bit and 32-bit surfaces are supported.
        # The color components must be in either RGB or BGR order within the pixel.
        rgb_array = surface_.get_view('3')

    except Exception as e:
        raise ValueError("\nCannot reference source pixels into a 3d array.\n %s " % e)

    return grey_2d_c(rgb_array)



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef inline void grey_3d(unsigned char [:, :, :] rgb_array):
    """
    Convert a 3D array (w, h, n) in RGB(A) format to grayscale (with alpha) in place.

    This function processes a 3D array directly, converting it to grayscale based on 
    luminosity while preserving the alpha channel (if present). The pixel values in 
    the resulting grayscale array represent intensity or brightness, ranging from 
    0 (black) to 255 (white).

    Example Usage:
    --------------
    # For a 24-bit image/surface
    grey_3d(pixels3d(im))

    # For a 32-bit image with alpha channel
    grey_3d(numpy.asarray(im.get_view('0'), dtype=uint8).reshape(w, h, 4))

    Parameters
    ----------
    rgb_array : numpy.ndarray
        A 3D NumPy array of shape (w, h, n) with dtype uint8, containing pixel data 
        in RGB(A) format. The values in the array range from 0 to 255. For 32-bit arrays 
        (RGBA), the alpha channel will be preserved but ignored in the grayscale conversion.

    Returns
    -------
    None
        This function modifies the input array in place and does not return a new array.
    """


    # Declare Cython variables for optimized memory usage
    cdef:
        Py_ssize_t w, h             # Width (w) and Height (h) of the image
        Py_ssize_t bit_size = 0     # Bit depth (number of color channels) of the image, initially set to 0

    try:
        # Try to unpack the shape of the input `rgb_array` into width, height, and bit size (number of channels)
        w, h, bit_size = rgb_array.shape[:3]  # Extract first 3 dimensions (w, h, bit_size)

        # Check if the image has valid bit sizes (either 3 for RGB or 4 for RGBA)
        if bit_size not in (3, 4):
            raise ValueError('\nIncorrect bit_size, support only RGB(A)')

    except ValueError as e:
        # If the array doesn't have the expected shape, handle errors:
        
        # Check if the input is a memory view (memoryviewslice)
        if is_type_memoryview(rgb_array):
            print(numpy.array(rgb_array).flags)  # Print memory flags for memoryview input
        else:
            # Print array flags for numpy array input
            print(rgb_array.flags)

        # Raise an error if the array shape is incorrect (not in the expected (w, h, 3) or (w, h, 4) format)
        raise ValueError(
            "\n%s\nExpecting array shape (w, h, n), "
            "RGB(A) got (%s, %s, %s)" % (e, w, h, len(rgb_array[:3])))

    # Check if the array data type is uint8 (unsigned 8-bit integer), which is compatible
    # RGB(A) images are expected to be in uint8 format (values 0-255 for each color channel)
    if not is_uint8(rgb_array):
        raise TypeError(
            "\nExpecting uint8 (unsigned char) data type got %s" % rgb_array.dtype)

    # Call the `grey_c` function (likely for grayscale conversion or another image processing function)
    grey_c(rgb_array)




@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef inline void grey_1d(unsigned char [:] rgb_array, bint format_32=False):
    """
    Convert a 1D array of uint8 data (RGB(A)) to grayscale (with alpha) in place.

    A grayscale image has a single channel representing pixel intensity or brightness, 
    with pixel values typically ranging from 0 (black) to 255 (white). This function 
    converts the input RGB(A) array to grayscale while preserving the alpha channel 
    (if present). The conversion is performed in place, modifying the original array.

    Example Usage:
    --------------
    # For a 24-bit buffer (RGB)
    grey_1d(image.get_buffer(), format_32=True)
    grey_1d(im.get_view('0'), format_32=True)
    grey_1d(numpy.frombuffer(im.get_view('1'), dtype=uint8), format_32=True)

    # For a 32-bit buffer (RGBA)
    import PIL
    from PIL import Image
    im = Image.open("../Assets/px.png")
    w, h = im.size
    arr = numpy.frombuffer(numpy.asarray(im.getdata(), dtype=uint8), dtype=uint8)
    grey_1d(arr, format_32=True)
    image = Image.frombytes('RGBA', (w, h), arr)
    image.show()

    Set `format_32` to `True` if the array is a 32-bit buffer containing RGBA values.

    Parameters
    ----------
    rgb_array : numpy.ndarray or bytearray
        A 1D array or buffer containing pixel data in RGB(A) format, with dtype uint8 
        (unsigned char values ranging from 0 to 255).

    format_32 : bool, optional (default=False)
        If `True`, the function assumes the input is a 32-bit buffer (RGBA).
        If `False`, the function assumes a 24-bit buffer (RGB).

    Returns
    -------
    None
        The function modifies the input array in place and does not return a new array.
    """

    # Only uint8 data is compatible
    if not is_uint8(rgb_array):
        raise TypeError("\nExpecting uint8 (unsigned char) data type got %s" % rgb_array.dtype)

    grey_1d_c(rgb_array, format_32)




@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef inline np.ndarray[np.uint8_t, ndim=1]\
        grey_1d_cp(const unsigned char [::1] bgr_array, bint format_32=False):

    """
    Convert a 1D array of uint8 BGR(A) data to grayscale (with alpha channel) and return a copy.

    This function checks that the input array has the expected uint8 data type, 
    and then converts the pixel data from BGR or BGRA format to grayscale. The conversion 
    preserves the alpha channel (if present). It returns a new 1D NumPy array in grayscale format.

    The grayscale conversion is based on the luminosity formula, which calculates the 
    brightness based on the red, green, and blue channels.

    Parameters
    ----------
    bgr_array : numpy.ndarray
        A 1D array containing pixel data in BGR(A) format, with dtype uint8. The pixel values 
        should range from 0 to 255. If the array represents a BGRA image, the alpha channel 
        will be preserved.

    format_32 : bool, optional (default=False)
        If `True`, the input array is assumed to be in BGRA (32-bit) format. 
        If `False`, the array is assumed to be in BGR (24-bit) format.

    Returns
    -------
    numpy.ndarray
        A new 1D NumPy array of shape (w,) with dtype uint8, containing the grayscale 
        pixel data. If the input was in BGRA format, the alpha channel is preserved in the output.

    Raises
    ------
    TypeError
        If the input array does not have dtype uint8, a `TypeError` will be raised.
    """

    # Only uint8 data is compatible
    if not is_uint8(bgr_array):
        raise TypeError(
            "\nExpecting uint8 (unsigned char) data type got %s" % bgr_array.dtype)

    return grey_1d_cp_c(bgr_array, format_32)


# -------------------- SEPIA


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef inline void sepia(object surface_):
    """
    Apply a sepia tone filter to an image, surface, or video game graphics (inplace).

    Sepia toning is a technique used in photography and imaging where the image is given 
    a reddish-brown tint, simulating the warm tones of old photographs. It creates a 
    softer, dreamier aesthetic compared to standard grayscale, adding depth and a vintage look.

    This function transforms the provided surface into an equivalent sepia-toned model, 
    adjusting the pixel colors accordingly while preserving the original structure. The 
    transformation is applied directly to the surface (inplace), and no new surface is returned.

    Example:
    --------
    sepia(surface)

    Parameters
    ----------
    surface_ : pygame.Surface
        A Pygame surface or display surface, which can be 24- or 32-bit with or without 
        per-pixel transparency (alpha channel). The surface will be modified directly.

    Returns
    -------
    None
        The transformation is applied inplace, and no new object is returned.
    """

    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)
    cdef unsigned char [:, :, :] rgb_array
    try:
        rgb_array = surface_.get_view('3')

    except Exception as e:
        raise ValueError("\nCannot reference source pixels into a 3d array.\n %s " % e)

    sepia_c(rgb_array)



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef sepia_copy(object surface_):
    """
    Sepia model (New Surface)

    Transform your video game/ image or surface into an equivalent sepia model
    
    While traditional black-and-white photographs use a standard grayscale to create
    highlights and shadows, sepia-toned photos use a reddish-brown tone to create that spectrum.
    “Sepia is a softer manipulation of light,”. This gives them a softer, dreamier aesthetic.
    
    e.g:
    im = sepia_copy(surface)

    :param surface_  : 
        Pygame surface or display surface compatible (image 24-32 bit with 
        or without per-pixel transparency / alpha channel)
        
    :return:         : Return a new surface 
    """

    # Ensure the surface_ argument is of type pygame.Surface
    # If it's not a pygame.Surface type, raise an assertion error with the provided type
    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)

    # Declare Cython variables for optimized memory usage
    cdef:
        unsigned char [::1] bgr_array  # Array to hold the pixel data of the surface
        Py_ssize_t w, h                # Width (w) and Height (h) of the surface
        Py_ssize_t bit_size            # Bit size (depth) of the surface's pixels

    # Retrieve the dimensions (width and height) of the surface
    w, h = surface_.get_size()

    # Get the number of bytes per pixel (bit depth), e.g., 3 for RGB, 4 for RGBA
    bit_size = surface_.get_bytesize()

    # Declare additional Cython variables to create a new array to hold the processed data
    cdef:
        Py_ssize_t l = w * h * bit_size  # Calculate the total number of bytes in the surface
        unsigned char [::1] destination_array = numpy.empty(l, dtype=numpy.uint8)  # Allocate a new array for destination

    # Attempt to get the buffer of the surface (raw pixel data)
    try:
        bgr_array = surface_.get_buffer()  # This gives access to the underlying pixel data

    except Exception as e:
        # If unable to access the buffer, raise a ValueError with a detailed error message
        raise ValueError("\nCannot reference source pixels into a 1d array.\n %s " % e)

    # Return a new pygame image created from the processed data.
    # Call sepia_1d_cp_c to apply a sepia filter to the image's buffer (raw pixel data)
    # `format_32=True` if bit_size is 4 (RGBA), otherwise format is RGB
    return pygame.image.frombuffer(
        sepia_1d_cp_c(
            bgr_array, destination_array, format_32=True if bit_size == 4 else False
        ),
        (w, h),  # Provide the width and height of the surface
        "RGBA" if bit_size == 4 else "RGB"  # Specify color format (RGBA for 4 channels, RGB for 3 channels)
    )




@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef inline void sepia_3d(unsigned char [:, :, : ] rgb_array):
    """
    Apply a sepia tone filter to a 3D RGB(A) image array (inplace).

    This function processes a 3D NumPy array representing pixel data in RGB(A) format 
    and applies a sepia filter. The sepia effect is achieved by adjusting the red, 
    green, and blue channels according to predefined coefficients, creating a warm-toned, 
    vintage effect. The function modifies the input array directly and does not return anything.

    Only arrays with shapes (w, h, 3) for RGB or (w, h, 4) for RGBA are supported.

    Parameters
    ----------
    rgb_array : numpy.ndarray
        A 3D array with shape (w, h, 3) for RGB or (w, h, 4) for RGBA pixel data, where
        `w` is the width, `h` is the height, and `3` or `4` corresponds to the RGB or RGBA channels.
        The array must have dtype uint8 (unsigned char) with pixel values ranging from 0 to 255.

    Raises
    ------
    ValueError
        If the input array does not have the expected shape (w, h, 3) or (w, h, 4).

    TypeError
        If the input array does not have dtype uint8.

    Notes
    -----
    - This function modifies the input `rgb_array` directly (inplace).
    - If the input is in RGBA format, the alpha channel is preserved.
    - The sepia effect is applied by adjusting the luminosity of the RGB channels using a set of coefficients.

    Example
    -------
    # Assuming rgb_array is a 3D NumPy array with shape (w, h, 3) for RGB or (w, h, 4) for RGBA:
    sepia_3d(rgb_array)
    """

    cdef:
        Py_ssize_t w, h
        Py_ssize_t bit_size = 0

    try:

        w, h, bit_size = rgb_array.shape[ :3 ]

        # Only RGB(A) array supported (w, h, 3|4)
        if bit_size not in (3, 4):
            raise ValueError('\nIncorrect bit_size, support only RGB(A)')

    except ValueError as e:
        # _memoryviewslice
        if is_type_memoryview(rgb_array):
            print(numpy.array(rgb_array).flags)
        # numpy.array
        else:
            print(rgb_array.flags)

        raise ValueError(
            "\n%s\nExpecting array shape (w, h, n), "
            "RGB(A) got (%s, %s, %s)" % (e, w, h, len(rgb_array[ :3 ])))

        # Only uint8 data is compatible
    if not is_uint8(rgb_array):
        raise TypeError(
            "\nExpecting uint8 (unsigned char) data type got %s" % rgb_array.dtype)

    sepia_c(rgb_array)


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef inline void sepia_1d(unsigned char [:] rgb_array, bint format_32=False):
    """
    Convert 1d array uint8 data type, RGB(A) to sepia equivalent (inplace)
    
    While traditional black-and-white photographs use a standard grayscale to create
    highlights and shadows, sepia-toned photos use a reddish-brown tone to create that spectrum.
    “Sepia is a softer manipulation of light,”. This gives them a softer, dreamier aesthetic.
    
    e.g 
    # image 24-bit  
    im = pygame.image.load("../Assets/px.png")
    w, h = im.get_width(), im.get_height()
    c = numpy.ndarray(shape=(w*h*3), buffer=im.get_view('0'), dtype=uint8)
    sepia_1d(c, False)   

    
    # image 32-bit 
    import pygame
    im = pygame.image.load("../Assets/px.png")
    w, h = im.get_width(), im.get_height()
    sepia_1d(im.get_view('0'), True)
    
    or 
    
    im = pygame.image.load("../Assets/px.png")
    w, h = im.get_width(), im.get_height()
    sepia_1d(numpy.ndarray(shape=(w*h*4), buffer=im.get_view('1'), dtype=uint8), True)
    
    
    Parameters
    ----------
    rgb_array : 
        numpy.ndarray shape(w,) uint8 data type, RGB(A) 
        (unsigned char 0...255) containing pixels or bytearray buffer
        
    format_32: 
        bool True | for 'RGB' buffer type (24-bit) or False 'RGBA' (32-bit)
 
    Returns
    -------
    Void
    
    """

        # Only uint8 data is compatible
    if not is_uint8(rgb_array):
        raise TypeError("\nExpecting uint8 (unsigned char) data type got %s" % rgb_array.dtype)

    sepia_1d_c(rgb_array, format_32)


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef np.ndarray[np.uint8_t, ndim=1, mode='c'] sepia_1d_cp(
        const unsigned char [::1] bgr_array, bint format_32=False):
    """
    Apply sepia tone transformation to a 1D array of BGR (or BGRA) pixels.
    
    This function processes a 1D NumPy array containing BGR (or BGRA) pixel data 
    and applies a sepia tone filter to each pixel. It returns a new 1D array of 
    the same shape with the sepia-toned pixel values. If the input is in BGRA format, 
    the alpha channel is preserved.
    
    The sepia effect is achieved by transforming the red, green, and blue components 
    of each pixel using a specific mathematical formula that produces a reddish-brown tone.
    
    Parameters
    ----------
    bgr_array : numpy.ndarray
       A 1D NumPy array with dtype uint8, containing pixel data in BGR (or BGRA) format. 
       The pixel values must be in the range 0 to 255. The array is expected to have 
       shape (w, 3) for BGR or (w, 4) for BGRA, where `w` is the number of pixels.
    
    format_32 : bool, optional (default=False)
       If `True`, the input array is assumed to be in BGRA (32-bit) format, 
       where each pixel has 4 channels (Blue, Green, Red, Alpha). 
       If `False`, the input is assumed to be in BGR (24-bit) format, with 3 channels.
    
    Returns
    -------
    numpy.ndarray
       A new 1D NumPy array of the same shape as `bgr_array`, containing the sepia-toned pixel data. 
       The returned array will have dtype uint8 and will apply the sepia filter while preserving the 
       alpha channel if the input is in BGRA format.
    
    Raises
    ------
    TypeError
       If `bgr_array` does not have dtype uint8, an error is raised indicating the expected data type.
    
    ValueError
       If the length of `bgr_array` is undefined or there is a problem with the array shape.
    """

    # Declare a variable to store the length of the input array
    cdef:
        Py_ssize_t length = 0  

    try:
        # Attempt to get the length of the input bgr_array
        length = len(bgr_array)

    except Exception as e:
        # Raise an error if the length cannot be determined
        raise ValueError(
            "\nArray length is 'undefined'.\n%s " % e)

    # Declare the destination array for processing
    cdef:
        # Create an empty array of the same length as the input array
        # This array will store the processed sepia-toned pixel data
        unsigned char [::1] destination_array = numpy.empty(length, dtype=numpy.uint8)

    # Ensure that the input array contains only uint8 (unsigned 8-bit integer) data
    if not is_uint8(bgr_array):
        raise TypeError(
            "\nExpecting uint8 (unsigned char) data type got %s" % bgr_array.dtype)

    # Apply the sepia filter to the input array and store the result in the destination array
    return sepia_1d_cp_c(bgr_array, destination_array, format_32)




# ------------------- MEDIAN


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef inline void median(
        object surface_,
        unsigned short int kernel_size_=2,
        bint fast_=True,
        unsigned short int reduce_factor_=1
) except *:

    """
    Apply median filter to a surface (inplace).

    The median filter is a non-linear image filtering technique that is often used 
    to remove noise from an image or signal. It works by replacing each pixel 
    value with the median of the pixel values in a neighborhood defined by the 
    kernel size. This technique is widely used for noise reduction in digital 
    image processing, especially for preserving edges while removing noise.

    The strength of the effect is controlled by the `kernel_size` parameter, 
    with larger kernel sizes producing stronger filtering effects. However, 
    larger kernel sizes may also slow down the process significantly. 

    Note: This filter is not suitable for real-time rendering in games or animations 
    due to its computational cost.

    Example usage:
    # For 24-bit image
    im = pygame.image.load("../Assets/background.jpg")
    im = scale(im, (800, 600))
    median(im, fast=True)
    
    # For 32-bit image
    im = pygame.image.load("../Assets/px.png").convert_alpha()
    im = scale(im, (800, 600))
    median(im, fast=False)
    
    Parameters
    ----------
    surface_ : pygame.Surface
        A Pygame surface object (24-bit or 32-bit) containing the image to be processed.
        It can have or lack an alpha channel for transparency.

    kernel_size_ : unsigned short int, optional
        The size of the kernel or neighborhood of pixels used for the median calculation. 
        Default is 2. Increasing the kernel size improves the filter effect but decreases performance.

    fast_ : bool, optional
        A flag to enable fast calculation (default is `True`). If `True`, the filter 
        will use a smaller texture size to improve performance, and may reduce the quality 
        based on the `reduce_factor_` argument.

    reduce_factor_ : unsigned short int, optional
        A factor to reduce the size of the surface before processing. A value of 1 divides 
        the surface by 2, and a value of 2 reduces the surface by 4. Values larger than 2 
        may degrade the image quality. Has no effect if `fast_` is `False`. Default is 1.

    Returns
    -------
    void
        This function modifies the surface directly (in-place) and does not return a value.

    Raises
    ------
    ValueError
        If `surface_` is not a valid `pygame.Surface` or if the `kernel_size_` 
        or `reduce_factor_` are out of valid ranges.
    """

    # Ensure the provided surface is a valid Pygame Surface object
    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)

    # Ensure the kernel size is greater than 1
    if kernel_size_ <= 1:
        raise ValueError('\nArgument kernel_size_ must be > 1')

    cdef unsigned char [:, :, :] rgb_array

    # If 'fast_' flag is set to True, use a faster approach with surface reduction
    if fast_:
        # Ensure reduce_factor_ is within a valid range (1 < reduce_factor_ < 8)
        if not 0 < reduce_factor_ < 9:
            raise ValueError('\nArgument reduce_factor_ must be > 1 and < 8 ')

        # Apply fast median filter with the given kernel size and reduction factor
        median_fast(surface_, kernel_size_, reduce_factor_)

    else:
        try:
            # Get a reference to the pixel data as a 3D array (RGB values)
            rgb_array = surface_.get_view('3')

        except Exception as e:
            raise ValueError("\nCannot reference source pixels into a 3D array.\n %s " % e)

        # Apply standard (slower) median filter inplace to the RGB array
        median_inplace_c(rgb_array, kernel_size_)


# ------------------- PAINTING

@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef inline void painting(object surface_) except *:
    """   
    Apply a painting effect (in-place) to a Pygame surface.

    This function transforms an image to resemble a hand-painted artistic style.
    It achieves this by using a fast median filter algorithm (`median_fast`), 
    which smooths pixel values while maintaining edge details, giving the image
    a brushstroke-like appearance.

    Note:
    -----
    - This method **modifies the input surface in place**.
    - It **is not optimized for real-time rendering** and is intended for offline 
      processing of images.

    Example Usage:
    --------------
    ```python
    # Load a 24-bit or 32-bit image
    im = pygame.image.load("../Assets/background.jpg").convert(24)

    # Scale the image to a preferred size
    im = scale(im, (800, 600))

    # Apply the painting effect
    painting(im)
    ```

    Parameters:
    -----------
    surface_ : pygame.Surface
        A Pygame-compatible surface (24-bit or 32-bit), 
        with or without per-pixel transparency (alpha channel).

    Returns:
    --------
    void
        The function modifies the input surface directly.
    """

    # Ensure the input is a valid Pygame surface
    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument 'surface_' must be a pygame.Surface type, got %s " % type(surface_)

    # Apply the median fast filter to create the painting effect
    median_inplace_c(pixels3d(surface_), 8)

# ------------------- PIXELATION

@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef inline void pixels(object surface_) except *:
    """
    Apply a pixelation effect to an image (INPLACE).

    Pixelation is a visual effect where an image is displayed at a low resolution, 
    making individual pixels (small, single-colored square elements) clearly visible. 
    This technique is often used in digital graphics to create artistic styles, 
    censor parts of an image, or simulate retro, low-resolution video game graphics.

    This function modifies the input surface in-place, reducing fine details and 
    emphasizing blocky pixel structures.

    Example Usage:
    --------------
    # Works with both 24-bit and 32-bit images
    import pygame
    im = pygame.image.load("../Assets/background.jpg").convert()
    im = pygame.transform.scale(im, (800, 600))  # Rescale image
    pixels(im)  # Apply pixelation effect

    Parameters:
    -----------
    surface_ : pygame.Surface
        A Pygame-compatible surface (24-bit or 32-bit image) with or without 
        per-pixel transparency (alpha channel).

    Returns:
    --------
    None
        The input surface is modified directly (in-place).
    """

    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)

    median_fast(surface_, 2, 4)


# -------------------

@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef inline void median_fast(
        object surface_,
        unsigned short int kernel_size_   = 2,
        unsigned short int reduce_factor_ = 1
):
    """
    Apply a fast median filter effect (INPLACE).

    This function is a lower-level implementation used internally by the `median`, 
    `painting`, and `pixels` algorithms. It cannot be called directly from 
    Python's interactive shell (IDLE). Instead, it is optimized for high-performance 
    image processing in compiled Cython code.

    The **median filter** is a non-linear digital filtering technique commonly 
    used to remove noise from images while preserving edges. It is widely used 
    in digital image processing as a pre-processing step for tasks such as edge 
    detection and segmentation.

    The strength of the median effect is controlled by the `kernel_size_` parameter. 
    However, this filter is not optimized for real-time display rendering.

    Example Usage:
    --------------
    # 24-bit image
    im = pygame.image.load("../Assets/background.jpg")
    median_fast(im, kernel_size_=3, reduce_factor_=1)

    # 32-bit image (with alpha channel)
    im = pygame.image.load("../Assets/px.png").convert_alpha()
    median_fast(im, kernel_size_=5, reduce_factor_=2)

    Parameters:
    -----------
    surface_ : pygame.Surface
        A 24-bit or 32-bit Pygame-compatible surface.

    kernel_size_ : int, optional (default=2)
        The size of the kernel (neighborhood of pixels considered for filtering).
        Increasing the kernel size enhances the filtering effect but significantly 
        impacts performance.

    reduce_factor_ : int, optional (default=1)
        Determines the reduction factor applied to the image before processing.
        - `1` reduces the image size by half.
        - `2` reduces the image size by a factor of four.
        Values higher than 2 may result in excessive blurring and pixelation.

    Returns:
    --------
    None
        The function modifies the input surface directly (in-place).
    """

    try:
        # Attempt to retrieve a 3D view (RGB) of the pixel data from the input surface
        rgb_array = surface_.get_view('3')

    except Exception as e:
        # Raise an error if the surface pixel data cannot be accessed as a 3D array
        raise ValueError("\nCannot reference source pixels into a 3D array.\n %s " % e)

    # Create a copy of the input surface
    surface_cp = surface_.copy()

    cdef:
        int w, h  # Declare width and height variables

    # Retrieve the dimensions (width and height) of the copied surface
    w, h = surface_cp.get_size()

    # Downscale the copied surface by a factor of 2^reduce_factor_ using smooth scaling
    surface_cp = smoothscale(surface_cp, (w >> reduce_factor_, h >> reduce_factor_))

    try:
        # Retrieve a 3D view of the downscaled surface
        cp_array = surface_cp.get_view('3')

    except Exception as e:
        # Raise an error if the downscaled surface pixel data cannot be accessed as a 3D array
        raise ValueError("\nCannot reference source pixels into a 3D array.\n %s " % e)

    cdef:
        int i, j  # Loop variables for iterating through pixels
        unsigned char[:, :, :] org_surface = rgb_array  # Original surface pixel data
        unsigned char[:, :, :] surface_cp_arr = cp_array  # Downscaled surface pixel data

    # Apply an in-place median filter to the downscaled surface using a given kernel size
    median_inplace_c(surface_cp_arr, kernel_size_)

    # Resize the filtered array back to the original dimensions of the input surface
    surface_cp_arr = resize_array_c(surface_cp_arr, w, h)

    # Perform parallel processing (using OpenMP) to update the original surface with the processed pixels
    with nogil:
        for j in prange(h, schedule=SCHEDULE, num_threads=THREADS):  # Parallel loop over height
            for i in range(w):  # Loop over width
                # Copy the processed pixel values back to the original surface
                org_surface[i, j, 0] = surface_cp_arr[i, j, 0]  # Red channel
                org_surface[i, j, 1] = surface_cp_arr[i, j, 1]  # Green channel
                org_surface[i, j, 2] = surface_cp_arr[i, j, 2]  # Blue channel




# ------------------- MEDIAN GRAYSCALE

@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef inline void median_grayscale(object surface_, int kernel_size_=2):
    """
    Apply a median filter to a grayscale version of the image (INPLACE).
    
    The **median filter** is a non-linear digital filtering technique commonly 
    used to remove noise from images while preserving edges. It is widely used 
    in digital image processing as a pre-processing step for tasks such as edge 
    detection and segmentation.
    
    This function converts the input surface to grayscale and applies a median 
    filter effect. The strength of the filtering effect is controlled by the 
    `kernel_size_` parameter.

    ⚠ **Note**:  
    - This method **modifies the surface in place**.
    - It is **not suitable for real-time rendering**.
    - Compatible with **24-bit and 32-bit surfaces**, with or without an alpha channel.

    Example Usage:
    --------------
    # Apply median filter to a 24-bit image
    im = pygame.image.load("../Assets/background.jpg")
    median_grayscale(im)

    # Apply median filter to a 32-bit image (with alpha channel)
    im = pygame.image.load("../Assets/px.png").convert_alpha()
    median_grayscale(im)

    Parameters:
    -----------
    surface_ : pygame.Surface
        A Pygame-compatible surface (24-bit or 32-bit, with or without alpha).

    kernel_size_ : int, optional (default=2)
        The size of the kernel (neighborhood of pixels considered for filtering).
        Must be greater than 0. Increasing the kernel size enhances the filtering 
        effect but significantly impacts performance.

    Returns:
    --------
    None
        The function modifies the input surface directly (in-place).
    """


    # Ensure that the input surface_ is an instance of pygame.Surface
    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)

    # Declare a 3D memoryview to store the RGB pixel data
    cdef unsigned char [:, :, :] rgb_array

    # Ensure that the kernel size for the median filter is greater than 1
    assert kernel_size_ > 1, "\nArgument kernel_size_ must be > 1"

    try:
        # Retrieve a 3D view (RGB) of the pixel data from the input surface
        rgb_array = surface_.get_view('3')

    except Exception as e:
        # Raise an error if the surface pixel data cannot be accessed as a 3D array
        raise ValueError("\nCannot reference source pixels into a 3D array.\n %s " % e)

    # Apply an in-place median filter to the grayscale version of the RGB array
    median_grayscale_inplace_c(rgb_array, kernel_size_)



# --------------- COLOR REDUCTION

@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef inline void posterize_surface(
        object surface_,
        int color_ = 8
):
    """
    Reduce the number of colors in an image (INPLACE).

    This function decreases the number of unique colors in the given surface, 
    effectively creating a posterization effect. Reducing colors can be useful 
    for artistic effects, image compression, or preprocessing for stylized 
    graphics.

    ⚠ **Note**:  
    - **Modifies the surface in place**.  
    - **Works with 24-bit and 32-bit surfaces**, with or without an alpha channel.  
    - If the surface has a **32-bit per-pixel alpha channel**, the alpha layer will 
      be disregarded, meaning the effect is applied only to RGB values.

    Example Usage:
    --------------
    # Reduce the number of colors in a Pygame surface to 8
    color_reduction(surface, 8)

    Parameters:
    -----------
    surface_ : pygame.Surface
        A Pygame-compatible surface (24-bit or 32-bit, with or without alpha).

    color_ : int, optional (default=8)
        The number of colors to reduce the image to.
        Must be greater than 0. Lower values produce a more dramatic effect.

    Returns:
    --------
    None
        The function modifies the input surface directly (in-place).
    """


    # Ensure that the input surface_ is an instance of pygame.Surface
    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)

    # Ensure that the color depth (number of colors) is greater than 0
    assert color_ > 0, "Argument color_number must be > 0"

    # Declare a 3D memoryview to store the RGB pixel data
    cdef unsigned char [:, :, :] rgb_array

    try:
        # Retrieve a 3D view (RGB) of the pixel data from the input surface
        rgb_array = surface_.get_view('3')

    except Exception as e:
        # Raise an error if the surface pixel data cannot be accessed as a 3D array
        raise ValueError("\nCannot reference source pixels into a 3D array.\n %s " % e)

    # Apply a posterization effect to the surface using the specified number of colors
    posterize_surface_c(rgb_array, color_)



# ----------------------- SOBEL


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef inline void sobel(
        object surface_,
        int threshold_ = 64
):
    """
    Apply Sobel Edge Detection (Inplace)
    
    The Sobel operator, also known as the Sobel-Feldman operator, is commonly used in image processing 
    for edge detection. It highlights edges in an image by emphasizing areas with high intensity 
    gradients.
    
    This function applies the Sobel edge detection to a Pygame surface, transforming the surface to emphasize 
    its edges based on the gradient of pixel intensities.
    
    The input surface must be in grayscale (24 - 32 bit) for best results. If the surface is not in grayscale, 
    only the red channel will be used for the edge detection.
    
    Example usage:
        sobel(surface, 64)  # Apply Sobel edge detection with a threshold of 64.
    
    :param surface_:
        A Pygame.Surface object (24 - 32 bit depth), which may or may not have an alpha channel. 
        The surface should be greyscaled, although non-greyscale images will use the red channel 
        for edge detection.
    
    :param threshold_:
        An integer (default is 64), representing the threshold for detecting edges. 
        The threshold determines the sensitivity of edge detection, with higher values requiring 
        stronger gradients to be considered an edge.
    
    :return:
        None. The function modifies the input surface in place, updating it with the Sobel edge detection result.
    """


    # Ensure that the input surface_ is an instance of pygame.Surface
    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)

    # Ensure that the threshold value is within the valid range [0, 255]
    assert -1 < threshold_ < 256, "\nArgument threshold must be an integer in range [0 ... 255]"

    # Declare a 3D memoryview to store the RGB pixel data
    cdef unsigned char [:, :, :] rgb_array

    try:
        # Retrieve a 3D view (RGB) of the pixel data from the input surface
        rgb_array = surface_.get_view('3')

    except Exception as e:
        # Raise an error if the surface pixel data cannot be accessed as a 3D array
        raise ValueError("\nCannot reference source pixels into a 3D array.\n %s " % e)

    # Apply the Sobel edge detection filter to the RGB image in-place
    # The threshold_ value determines the edge intensity sensitivity
    sobel_inplace_c(rgb_array, <float>threshold_)




@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef inline void sobel_1d(
        Py_ssize_t w,
        Py_ssize_t h,
        unsigned char [::1] bgr_array,
        tmp_array = None,
        int threshold = 64,
        bint format_32 = False,
        bint greyscale = False
        ):

    """
    Apply 1D Sobel Edge Detection (Inplace)

    This function applies the 1D Sobel operator to a given image (or surface), emphasizing 
    the edges in the horizontal or vertical direction based on the gradient of pixel intensities. 
    It modifies the input buffer(s) in place.

    Example usage:
    # For 24-bit image
    image = pygame.image.load('../Assets/px.png').convert(24)
    image = pygame.transform.smoothscale(image, (800, 600))
    grey(image)
    image_copy = image.copy()
    sobel_1d(800, 600, image.get_buffer(), image_copy.get_buffer(), threshold=25)

    # For 32-bit image (with alpha)
    image = pygame.image.load('../Assets/px.png').convert_alpha()
    image = pygame.transform.smoothscale(image, (800, 600))
    image_copy = image.copy()
    sobel_1d(800, 600, image.get_buffer(), image_copy.get_buffer(), threshold=25, format_32=True)

    Parameters
    ----------
    w : int
        Width of the Pygame surface that the source array (`bgr_array`) is referencing.

    h : int
        Height of the Pygame surface that the source array (`bgr_array`) is referencing.

    bgr_array : numpy.ndarray
        A 1D C-buffer of type uint8 containing pixel data in BGR format. If the image is not 
        greyscale, the algorithm processes all three RGB channels. If greyscale is enabled, only 
        the blue channel is used for edge detection.

    tmp_array : numpy.ndarray, optional
        A 1D C-buffer of type uint8 containing pixel data in BGR format. It is a copy of the 
        source `bgr_array`. Both `bgr_array` and `tmp_array` must have the same size and 
        data format. This is used as a temporary buffer during processing.

    threshold : int, optional, default=64
        The threshold for edge detection. Pixels with gradient values above this threshold will 
        be considered edges.

    format_32 : bool, optional, default=False
        If `True`, the input array is assumed to be in 32-bit BGRA format. If `False`, the input 
        array is assumed to be in 24-bit BGR format.

    greyscale : bool, optional, default=False
        If `True`, the algorithm processes only the blue channel for edge detection, which can 
        simplify the computation for greyscale images. If `False`, all three RGB channels are 
        used in the Sobel operator.

    Returns
    -------
    None
        The function modifies the input buffers (`bgr_array` and `tmp_array`) in place.
    """


    assert PyObject_IsInstance(bgr_array, (numpy.ndarray, cython.view.memoryview)), \
        "\nArgument bgr_array must be a numpy ndarray type, got %s " % type(bgr_array)

    # Only uint8 data is compatible
    if not is_uint8(bgr_array):
        raise TypeError("\nExpecting uint8 (unsigned char) data type got %s" % bgr_array.dtype)

    if tmp_array is not None:
        assert PyObject_IsInstance(tmp_array, (numpy.ndarray, pygame.bufferproxy.BufferProxy)), \
            "\nArgument tmp_array must be a numpy ndarray type, got %s " % type(tmp_array)

        # Cannot check data type for BufferProxy yet
        if not PyObject_IsInstance(tmp_array,  pygame.bufferproxy.BufferProxy):
            # Only uint8 data is compatible
            if not is_uint8(tmp_array):
                raise TypeError("\nExpecting uint8 (unsigned char) data type got %s" % tmp_array.dtype)

    assert -1 < threshold < 256, "\nArgument threshold must be an integer in range [0 ... 255]"

    cdef:
        cdef Py_ssize_t l = bgr_array.shape[ 0 ]
        # below create a copy False of the array and do not reference the pixels.
        # The real time transformation of the identical copy of the array will not be functional as all the pixels
        # undergo constant transformations. It is then necessary to load the pixels from a copy of the source array
        # to implement the inplace transformation. Such as below
        unsigned char [::1] bgr_array_cp = numpy.ndarray(shape=l, buffer=bgr_array, dtype=uint8).copy() if\
            tmp_array is None else tmp_array

    sobel_1d_c(w, l, bgr_array, bgr_array_cp, <float>threshold, format_32)



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef inline void sobel_fast(
        object surface_,
        int threshold_ = 64,
        unsigned short factor_ = 1
):
    """
    Fast sobel (inplace)
    
    The Sobel operator, sometimes called the Sobel–Feldman operator or Sobel filter,
    is used in image processing and computer vision, particularly within edge detection
    algorithms where it creates an image emphasising edges.

    Transform the game display or a pygame surface into a sobel equivalent model
    This version is slightly fastest than sobel_inplace_c as
    it down-scale the array containing all the pixels and apply the sobel algorithm to a smaller
    sample. When the processing is done, the array is re-scale to its original dimensions.
    If this method is in theory faster than sobel_inplace_c, down-scaling and up-scaling
    an array does have a side effect of decreasing the overall image definition
    (jagged lines non-antialiasing)
    
    Compatible 24 - 32 bit with or without alpha layer. 
    The surface must be greyscale, but non greyscale image will also work. However only
    the red channel will be used to code the sobel effect
      
    e.g:
     sobel_fast(surface, 64, amplitude=1)

    :param surface_: 
        pygame.surface compatible 24-32 bit
         
    :param threshold_: 
        integer; default value is 24
         
    :param factor_: 
        integer; default value is 1 (div by 2)
        
    :return: 
        void
        
    """

    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)

    assert -1 < threshold_ < 256, "\nArgument threshold must be an integer in range [0 ... 255]"
    assert 0 < factor_ < 9, "\nArgument amplitude must be in range [1 ... 8]"

    sobel_fast_inplace_c(surface_, threshold_, factor_)


#-------------------- INVERT

@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef inline void invert(object surface_):
    """
    Invert pixels (inplace)
    
    Invert all pixels of the display or a given texture
    
    Inverting an image means inverting the pixel values.
    Images are represented using RGB or Red Green Blue values. 
    Each can take up an integer value between 0 and 255 (both included). 
    For example, a red color is represent using (255, 0, 0), white with (255, 255, 255), 
    black with (0, 0, 0) and so on. Inverting an image means reversing the colors on the image.
    For example, the inverted color for red color will be (0, 255, 255). Note that 0 
    became 255 and 255 became 0. This means that inverting an image is essentially subtracting 
    the old RGB values from 255. 
    
    Compatible 24 - 32 bit with or without alpha layer
      
    e.g:
     invert(surface)
    
    :param surface_: 
        pygame.surface; compatible 24 - 32 bit surfaces
        
    :return: 
        void
        
    """

    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)

    cdef unsigned char [:, :, :] rgb_array

    try:
        rgb_array = surface_.get_view('3')

    except Exception as e:
        raise ValueError("\nCannot reference source pixels into a 3d array.\n %s " % e)

    invert_inplace_c(rgb_array)



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef invert_copy(object surface_):
    """
    Invert pixels and return a copy

    Invert all pixels of the display or a given texture
    
    Inverting an image means inverting the pixel values.
    Images are represented using RGB or Red Green Blue values. 
    Each can take up an integer value between 0 and 255 (both included). 
    For example, a red color is represent using (255, 0, 0), white with (255, 255, 255), 
    black with (0, 0, 0) and so on. Inverting an image means reversing the colors on the image.
    For example, the inverted color for red color will be (0, 255, 255). Note that 0 
    became 255 and 255 became 0. This means that inverting an image is essentially subtracting 
    the old RGB values from 255. 

    Compatible 24 - 32 bit with or without alpha layer


    e.g:
     inv = invert_copy(surface)

    :param surface_: 
        pygame.surface; compatible 24 - 32 bit surfaces
        
    :return: 
        return a copy with inverted pixels
        
    """

    # Ensure that the input surface_ is an instance of pygame.Surface
    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)

    # Declare necessary variables
    cdef:
        Py_ssize_t w, h  # Image width and height
        int bit_size      # Number of bytes per pixel (3 for RGB, 4 for RGBA)
        unsigned char [:] rgb_array  # 1D memory view of the image buffer

    # Retrieve the dimensions (width and height) of the input surface
    w, h = surface_.get_size()

    # Get the bit depth (bytes per pixel) of the surface (3 for RGB, 4 for RGBA)
    bit_size = surface_.get_bytesize()

    try:
        # Retrieve a direct buffer view of the surface pixel data
        rgb_array = surface_.get_buffer()

    except Exception as e:
        # Raise an error if the buffer cannot be accessed
        raise ValueError(
            "\nCannot reference source pixels into a 1D memoryviewslice array.\n %s " % e)

    # Process the image by inverting its colors using the invert1d_cp_c function
    # The function modifies the pixel values, flipping them to create a negative effect
    # The format_32 flag is set to True if the image has 4 bytes per pixel (RGBA), otherwise False
    return pygame.image.frombuffer(
        invert1d_cp_c(rgb_array, format_32=True if bit_size == 4 else False),
        (w, h), 
        "RGB" if bit_size == 3 else "RGBA"
)




@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef inline void invert3d(unsigned char [:, :, :] rgb_array):

    """
    Invert 3d array pixels (inplace)
    
    Invert a 3d array shape (w, h, n) uint8 data type
    
    Inverting an image means inverting the pixel values.
    Images are represented using RGB or Red Green Blue values. 
    Each can take up an integer value between 0 and 255 (both included). 
    For example, a red color is represent using (255, 0, 0), white with (255, 255, 255), 
    black with (0, 0, 0) and so on. Inverting an image means reversing the colors on the image.
    For example, the inverted color for red color will be (0, 255, 255). Note that 0 
    became 255 and 255 became 0. This means that inverting an image is essentially subtracting 
    the old RGB values from 255. 
    
    e.g
    # 24 bit
    image = pygame.image.load('../Assets/px.png').convert(24)
    invert3d(array3d)
    
    # 32 bit
    image = pygame.image.load('../Assets/px.png').convert_alpha()
    invert3d(array3d)
    
    Parameters
    ----------
    rgb_array : 
        numpy.ndarray shape (w, h, n) containing RGB(A) pixel format and 
        works with any other formats such as BGR, BGRA

    Returns
    -------
    void
    
    """

    cdef:
        Py_ssize_t w, h
        Py_ssize_t bit_size = 0

    try:

        w, h, bit_size = rgb_array.shape[ :3 ]

        # Only RGB(A) array supported (w, h, 3|4)
        if bit_size not in (3, 4):
            raise ValueError('\nIncorrect bit_size, support only RGB(A)')

    except ValueError as e:
        # _memoryviewslice
        if is_type_memoryview(rgb_array):
            print(numpy.array(rgb_array).flags)
        # numpy.array
        else:
            print(rgb_array.flags)

        raise ValueError(
            "\n%s\nExpecting array shape (w, h, n), "
            "RGB(A) got (%s, %s, %s)" % (e, w, h, len(rgb_array[ :3 ])))

        # Only uint8 data is compatible
    if not is_uint8(rgb_array):
        raise TypeError("\nExpecting uint8 (unsigned char) data type got %s" % rgb_array.dtype)

    invert3d_c(rgb_array)




@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef void invert1d(unsigned char [:] rgb_array, bint format_32=False):
    """
    Invert directly a C-buffer pixel values 
    
    Invert a C-buffer uint8 data types RGB(A) format
    
    This method will works with other buffer format such as BGR, BGRA
    
    Inverting an image means inverting the pixel values.
    Images are represented using RGB or Red Green Blue values. 
    Each can take up an integer value between 0 and 255 (both included). 
    For example, a red color is represent using (255, 0, 0), white with (255, 255, 255), 
    black with (0, 0, 0) and so on. Inverting an image means reversing the colors on the image.
    For example, the inverted color for red color will be (0, 255, 255). Note that 0 
    became 255 and 255 became 0. This means that inverting an image is essentially subtracting 
    the old RGB values from 255. 
    
    e.g
    # 24 bit
    image = pygame.image.load('../Assets/px.png').convert(24)
    invert1d(image.get_buffer(), False)
    
    # 32 bit
    image = pygame.image.load('../Assets/px.png').convert_alpha()
    invert1d(image.get_buffer(), True)
    
    Parameters
    ----------
    rgb_array : 
        numpy.ndarray 1d array, memoryviewslice uint8 data type containing 
        RGB(A) pixel format, works also with other format pixel (BGR, BGRA etc)
         
    format_32 : 
        bool; True | 'RGB' for 24-bit array (RGB) or False | 'RGBA' for 32-bit array (RGBA)
         

    Returns
    -------
    void
    
    """

    # Only uint8 data is compatible
    if not is_uint8(rgb_array):
        raise TypeError("\nExpecting uint8 (unsigned char) data type got %s" % rgb_array.dtype)

    invert1d_c(rgb_array, format_32)





@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef np.ndarray[np.uint8_t, ndim=1] invert1d_cp(const unsigned char [:] rgb_array, bint format_32=False):
    """
    Invert directly a C-buffer pixel values (return a copy)
    
    Invert C buffer uint8 data types RGB(A) format
    
    Inverting an image means inverting the pixel values.
    Images are represented using RGB or Red Green Blue values. 
    Each can take up an integer value between 0 and 255 (both included). 
    For example, a red color is represent using (255, 0, 0), white with (255, 255, 255), 
    black with (0, 0, 0) and so on. Inverting an image means reversing the colors on the image.
    For example, the inverted color for red color will be (0, 255, 255). Note that 0 
    became 255 and 255 became 0. This means that inverting an image is essentially subtracting 
    the old RGB values from 255. 
    
    This method will works with other buffer format such as BGR, BGRA
    
    e.g:
    # 24-bit
    image = pygame.image.load('../Assets/px.png').convert(24)
    arr3d = invert1d_cp(image.get_buffer(), False)
    image = pygame.image.frombuffer(arr3d, (WIDTH, HEIGHT), "BGR")
    
    # 32-bit
    image = pygame.image.load('../Assets/px.png').convert_alpha()
    arr3d = invert1d_cp(image.get_buffer(), True)
    image = pygame.image.frombuffer(arr3d, (WIDTH, HEIGHT), "BGRA")
    
    Parameters
    ----------
    rgb_array : 
        numpy.ndarray 1d array, memoryviewslice uint8 data type containing 
        RGB(A) pixel format, works also with other format pixel (BGR, BGRA etc)
         
    format_32 : 
        bool; True | 'RGB' for 24-bit array (RGB) or False | 'RGBA' for 32-bit array (RGBA)
         

    Returns
    -------
    numpy.ndarray 1d array uint8 data type. Copy of the input buffer
    
    """

    # Only uint8 data is compatible
    if not is_uint8(rgb_array):
        raise TypeError("\nExpecting uint8 (unsigned char) data type got %s" % rgb_array.dtype)

    return invert1d_cp_c(rgb_array, format_32)




# -------------------- HSL



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef inline void hsl_effect(object surface_, const float shift):
    """
    Apply Hue Rotation to an Image (HSL Color Space)
    
    This function directly modifies the hue of a Pygame surface using the HSL (Hue, Saturation, Lightness) 
    color model. Hue rotation shifts the colors of the surface in a way that corresponds to a rotation 
    on the color wheel, allowing you to alter the overall color tone.
    
    The surface must be compatible with 24-bit or 32-bit color depth, with or without an alpha layer. 
    If the `shift` value is 0.0, the surface remains unchanged.
    
    The hue shift value must be within the range [0.0, 1.0], where 0.0 represents no rotation, and 1.0 
    represents a 360-degree rotation.
    
    Example usage:
        hsl_effect(surface, 0.2)  # Apply a 72-degree hue shift to the surface.
    
    Parameters
    ----------
    surface_ : pygame.Surface
        A Pygame surface, compatible with 24-bit or 32-bit color formats (with or without alpha).
    
    shift : float
        A float value in the range [0.0, 1.0], where 0.0 corresponds to no hue shift, 
        and 1.0 corresponds to a full 360-degree rotation of the hue.
    
    Returns
    -------
    None
        The function modifies the input `surface_` in place, applying the hue shift effect.
    """

    # Ensure that the input surface_ is an instance of pygame.Surface
    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)

    # Validate that the shift value is within the allowed range [0.0, 1.0]
    assert 0.0 <= shift <= 1.0, \
        "\nArgument shift must be in range [0.0 ... 1.0]"

    # If shift is 0, there is no need to modify the image, so return immediately
    if shift == 0:
        return

    # Declare a 3D unsigned char array to hold the pixel data
    cdef unsigned char [:, :, :] rgb_array

    try:
        # Attempt to get a direct view of the pixel data in RGB format
        rgb_array = surface_.get_view('3')

        # Alternative approach (commented out) - converting the pixel data to a NumPy array
        # bgr_array = numpy.array(surface_.get_view('3'), copy=False, order='F')

    except Exception as e:
        # Raise an error if unable to retrieve the pixel buffer
        raise ValueError("\nCannot reference source pixels into a 3D array.\n %s " % e)

    # Apply HSL transformation with the given shift value
    # This function modifies the image's color properties by shifting the hue
    hsl_c(rgb_array, shift)





@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef void hsl3d(unsigned char [:, :, :] rgb_array, const float shift):
    """
    Apply Hue Rotation to a 3D Array (HSL Color Space)

    This function applies hue rotation to a 3D numpy array representing an image in the HSL (Hue, Saturation, 
    Lightness) color space. The hue shift is applied directly to the array, modifying the color tone of the image.

    The array must be in the shape (w, h, n), where `w` is the width, `h` is the height, and `n` is the number 
    of color channels (3 for RGB or 4 for RGBA, etc.). The data type should be uint8.

    The `shift` value must be within the range [0.0, 1.0], where 0.0 represents no hue rotation, and 1.0 represents 
    a full 360-degree hue rotation.

    Example usage:
        # For 24-bit RGB image
        image = pygame.image.load('../Assets/px.png').convert(24)
        array3d = pygame.surfarray.pixels3d(image)
        hsl3d(array3d, 0.2)  # Apply a 72-degree hue shift

        # For 32-bit RGBA image
        image = pygame.image.load('../Assets/px.png').convert_alpha()
        array3d = pygame.surfarray.pixels3d(image)
        hsl3d(array3d, 0.2)  # Apply a 72-degree hue shift

    Parameters
    ----------
    rgb_array : numpy.ndarray
        A 3D numpy array with shape (w, h, n) containing RGB or RGBA pixel data. 
        The array can also represent other formats such as BGR, BGRA.

    shift : float
        A float value in the range [0.0, 1.0], representing the hue rotation. 
        A value of 0.0 means no rotation, while 1.0 corresponds to a full 360-degree hue shift.

    Returns
    -------
    None
        The function modifies the input `rgb_array` in place, applying the hue shift.
    """

    # If the shift value is 0, no transformation is needed, so return immediately
    if shift == 0:
        return

    # Validate that the shift value is within the allowed range [0.0, 1.0]
    assert 0.0 <= shift <= 1.0, \
        "\nArgument shift must be in range [0.0 ... 1.0]"

    # Declare variables for image dimensions and bit depth
    cdef:
        Py_ssize_t w, h      # Width and height of the image
        Py_ssize_t bit_size = 0  # Number of color channels (3 for RGB, 4 for RGBA)

    try:
        # Extract the shape of the rgb_array (should be in the format (w, h, 3) or (w, h, 4))
        w, h, bit_size = rgb_array.shape[:3]

        # Ensure that the image has either 3 (RGB) or 4 (RGBA) channels
        if bit_size not in (3, 4):
            raise ValueError('\nIncorrect bit_size, support only RGB(A)')

    except ValueError as e:
        # Handle cases where the input is a memoryview or NumPy array
        if is_type_memoryview(rgb_array):
            # Print memory layout details for debugging if it's a memoryview
            print(numpy.array(rgb_array).flags)
        else:
            # Print NumPy array flags if it's a NumPy array
            print(rgb_array.flags)

        # Raise an error with a detailed message about the expected format
        raise ValueError(
            "\n%s\nExpecting array shape (w, h, n), "
            "RGB(A) got (%s, %s, %s)" % (e, w, h, len(rgb_array[:3])))

    # Ensure that the input pixel data is of type uint8 (unsigned char)
    if not is_uint8(rgb_array):
        raise TypeError(
            "\nExpecting uint8 (unsigned char) data type, got %s" % rgb_array.dtype)

    # Apply HSL transformation with the given shift value
    # This modifies the color properties of the image by shifting the hue
    hsl_c(rgb_array, shift)




@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef void hsl1d(unsigned char [::1] bgr_array, const float shift, bint format_32=False):

    """
    Apply Hue Rotation to a C-buffer (HSL Color Space)

    This function performs hue rotation on a C-buffer (1D array) containing pixel data in RGB(A) format 
    using the HSL (Hue, Saturation, Lightness) color model. It modifies the array in place.

    The function supports pixel formats like RGB, BGR, RGBA, and BGRA, adjusting the hue of each pixel 
    according to the specified `shift`. The `shift` value should be in the range [0.0, 1.0], which 
    corresponds to a rotation of 0.0 to 360.0 degrees on the hue color wheel.

    Example usage:
        # For 24-bit RGB image
        image = pygame.image.load('../Assets/px.png').convert(24)
        hsl1d(image.get_buffer(), 0.2)

        # For 32-bit RGBA image
        image = pygame.image.load('../Assets/px.png').convert_alpha()
        hsl1d(image.get_buffer(), 0.2, format_32=True)

    Parameters
    ----------
    bgr_array : numpy.ndarray
        A 1D numpy array (C-buffer) of type uint8 containing pixel data in RGB(A) format. 
        The array can also represent other formats such as BGR, BGRA.

    shift : float
        A float value in the range [0.0, 1.0] representing the hue rotation. 
        A value of 0.0 means no rotation, while 1.0 corresponds to a full 360-degree rotation.

    format_32 : bool, optional, default=False
        If `True`, the input array is assumed to be in 32-bit RGBA format. 
        If `False`, the array is assumed to be in 24-bit RGB format.

    Returns
    -------
    None
        The function modifies the input `bgr_array` in place by applying the hue shift.
    """

    if shift == 0:
        return

    assert 0.0 <= shift <= 1.0, \
        "\nArgument shift must be in range[0.0 ... 1.0]"

    # Only uint8 data is compatible
    if not is_uint8(bgr_array):
        raise TypeError("\nExpecting uint8 (unsigned char) data type got %s" % bgr_array.dtype)

    hsl1d_c(bgr_array, shift, format_32)



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef np.ndarray[np.uint8_t, ndim=1] hsl1d_cp(
        const unsigned char [::1] bgr_array, const float shift, bint format_32=False):
    """
    Rotate hue (HSL) directly to a C-buffer (return a copy)

    HSL (C buffer) uint8 data types RGB(A) format 
    
    This method will works with other buffer format such as BGR, BGRA
    
    HSL (Hue, Saturation, Lightness) is another color representation 
    model used in digital imaging and graphics. It defines colors in 
    terms of their hue, saturation, and lightness, offering an intuitive 
    way to describe and manipulate colors based on human perception.
    
    e.g:
    # 24-bit
    image = pygame.image.load('../Assets/px.png').convert(24)
    arr = hsl1d_cp(image.get_buffer(), 0.2, format_32 = False)
    image = pygame.image.frombuffer(arr, (WIDTH, HEIGHT), "BGR")
    
    # 32-bit
    image = pygame.image.load('../Assets/px.png').convert_alpha()
    arr = hsl1d_cp(image.get_buffer(), 0.2, format_32 = True)
    image = pygame.image.frombuffer(arr, (WIDTH, HEIGHT), "BGRA")
    
    Hue value (shift) must be in range [0.0 ...1.0] corresponding to 0.0 - 360.0 degrees rotation
    
    Parameters
    ----------
    bgr_array : 
        numpy.ndarray 1d array, memoryviewslice uint8 data type containing 
       RGB(A) pixel format, works also with other format pixel (BGR, BGRA etc)
        
    shift: 
        float; float value in range [0.0 ... 1.0] corresponding to 0.0 - 360.0 degrees rotation
        
    format_32: 
        bool True | for 'RGB' buffer type (24-bit) or False 'RGBA' (32-bit)
         
    
    Returns
    -------
    numpy.ndarray 1d array type uint8 new array containing pixels with rotated hue
    
    """

    if shift == 0:
        return numpy.asarray(bgr_array)

    assert 0.0 <= shift <= 1.0, \
        "\nArgument shift must be in range[0.0 ... 1.0]"

        # Only uint8 data is compatible
    if not is_uint8(bgr_array):
        raise TypeError("\nExpecting uint8 (unsigned char) data type got %s" % bgr_array.dtype)

    return hsl1d_cp_c(bgr_array, shift, format_32)


# -------------------- HSV

@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef inline void hsv_effect(object surface_, const float shift):
    """
    Apply Hue Rotation to a Surface (HSV Color Space)

    This function applies a hue rotation to a Pygame surface using the HSV (Hue, Saturation, Value) color model. 
    It modifies the surface in place, rotating the hue of the colors on the surface based on the specified shift.

    The surface must be compatible with 24-bit or 32-bit color formats, with or without an alpha channel. 
    The hue shift is specified as a float value in the range [0.0, 1.0], where 0.0 corresponds to no hue change, 
    and 1.0 represents a full 360-degree hue rotation.

    Example usage:
        surface = pygame.image.load('../Assets/px.png').convert_alpha()
        hsv_effect(surface, 0.2)  # Rotate the hue by 72 degrees (0.2 * 360)

    Parameters
    ----------
    surface_ : pygame.Surface
        A Pygame surface, compatible with 24-bit or 32-bit color formats (with or without alpha).

    shift : float
        A float value in the range [0.0, 1.0], specifying the hue rotation. 
        A value of 0.0 means no rotation, while 1.0 corresponds to a full 360-degree rotation.

    Returns
    -------
    None
        The function modifies the input surface in place by applying the hue shift.
    """

    # Ensure that the input argument `surface_` is a valid Pygame Surface object
    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)

    # If shift is 0, no transformation is needed, so return early
    if shift == 0:
        return

    # Validate that the shift value is within the allowed range (0.0, 1.0]
    assert 0.0 < shift <= 1.0, \
        "Argument shift must be in range [0.0 ... 1.0]"

    # Declare a Cython memoryview for the RGB array
    cdef unsigned char [:, :, :] rgb_array

    try:
        # Attempt to obtain a 3D pixel view (RGB format) from the surface
        rgb_array = surface_.get_view('3')

    except Exception as e:
        # If accessing the pixel data fails, raise an error with details
        raise ValueError("\nCannot reference source pixels into a 3D array.\n %s " % e)

    # Apply an HSV transformation with the given shift value
    # This function modifies the pixel colors in HSV space
    hsv3d_c(rgb_array, shift)





@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef inline void hsv3d(unsigned char [:, :, :] rgb_array, const float shift):
    """
    Rotate hue 3d array 
    
    Rotate the hue (HSV conversion method), directly from a 3d array  
    
    Compatible 24 - 32 bit with or without alpha layer
    
    HSV (Hue, Saturation, Value) is a color model similar to HSL (Hue, Saturation, Lightness)
    but with some differences in how it represents and manipulates colors. 
    It’s often used in graphics software and computer vision applications for its 
    simplicity in specifying and adjusting color attributes.
    
    New Shift value. Must be between [0.0 ... 1.0] corresponding to 0.0 - 360.0 degrees 
     (e.g 0.5 = 180 degrees)
     
    e.g:
    # Rotate the hue 72 degrees
    array3d = pygame.surfarray.pixels3d(image)
    hsv3d(array3d, 0.2)
    
    Parameters
    ----------
    rgb_array : 
        numpy.ndarray shape (w, h, n) containing RGB(A) pixel format and 
        works with any other formats such as BGR, BGRA
        
    shift     : 
        float; float value in range [0.0 ... 1.0] corresponding to 0 - 360 degrees. New hue value. 

    Returns
    -------
    void 

    """

    # If shift is 0, no transformation is needed, so return early
    if shift == 0:
        return

    # Validate that the shift value is within the allowed range [0.0, 1.0]
    assert 0.0 <= shift <= 1.0, \
        "\nArgument shift must be in range [0.0 ... 1.0]"

    # Declare Cython variables for image dimensions and bit depth
    cdef:
        Py_ssize_t w, h  # Image width and height
        Py_ssize_t bit_size = 0  # Number of color channels (3 for RGB, 4 for RGBA)

    try:
        # Retrieve image dimensions and bit depth from the given RGB array
        w, h, bit_size = rgb_array.shape[:3]

        # Ensure that the image format is either RGB (3 channels) or RGBA (4 channels)
        if bit_size not in (3, 4):
            raise ValueError('\nIncorrect bit_size, only RGB(A) images are supported')

    except ValueError as e:
        # Check if `rgb_array` is a memoryview slice and print its flags for debugging
        if is_type_memoryview(rgb_array):
            print(numpy.array(rgb_array).flags)
        # If it's a NumPy array, print its memory layout flags
        else:
            print(rgb_array.flags)

        # Raise an error indicating an incorrect image shape
        raise ValueError(
            "\n%s\nExpecting array shape (w, h, n), "
            "RGB(A) got (%s, %s, %s)" % (e, w, h, len(rgb_array[:3]))
        )

    # Ensure that the pixel data type is uint8 (8-bit unsigned integer)
    if not is_uint8(rgb_array):
        raise TypeError(
            "\nExpecting uint8 (unsigned char) data type, got %s" % rgb_array.dtype
        )

    # Apply an HSV transformation to the image with the given shift value
    hsv3d_c(rgb_array, shift)



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef void hsv1d(unsigned char [::1] bgr_array, const float shift, bint format_32=False):
    """
    Rotate hue 1d array
    
    Rotate the hue directly from a C-buffer (1d array uint8 data types RGB(A) format)
     Changes apply inplace
    
    This method will works with other buffer format such as BGR, BGRA

    HSV (Hue, Saturation, Value) is a color model similar to HSL (Hue, Saturation, Lightness)
    but with some differences in how it represents and manipulates colors. 
    It’s often used in graphics software and computer vision applications for its 
    simplicity in specifying and adjusting color attributes.
    
    e.g 
    #compatible with 32 bits images 
    image = pygame.image.load('../Assets/px.png').convert_alpha()
    hsv1d(image.get_buffer(), angle/36.0, format_32=True)
    
    #compatible with 24 bits images 
    image = pygame.image.load('../Assets/px.png').convert(24)
    hsv1d(image.get_buffer(), angle/36.0, format_32=False) 
    
    Parameters
    ----------
    bgr_array : 
        numpy.ndarray 1d array, memoryviewslice uint8 data type containing 
        BGR(A) pixel format, works also with other format pixel (BGR, BGRA etc)
         
    shift     : 
        float; float value in range [0.0 ... 1.0] corresponding to 0 - 360 degrees. New hue value.
         
    format_32 : 
        bool True | for 'BGR' buffer type (24-bit) or False 'BGRA' (32-bit)
         

    Returns
    -------
    void

    """

    if shift == 0:
        return

    assert 0.0 <= shift <= 1.0, \
        "\nArgument shift must be in range[0.0 ... 1.0]"

    # Only uint8 data is compatible
    if not is_uint8(bgr_array):
        raise TypeError("\nExpecting uint8 (unsigned char) data type got %s" % bgr_array.dtype)

    hsv1d_c(bgr_array, shift, format_32)



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef np.ndarray[np.uint8_t, ndim=1] hsv1d_cp(
        const unsigned char [::1] bgr_array, const float shift, bint format_32=False):
    """
    Rotate the hue 1d array (return a copy)
    
    HSV 1d array (C buffer) uint8 data types RGB(A) format 

    This method will works with other buffer format such as BGR, BGRA
    
    HSV (Hue, Saturation, Value) is a color model similar to HSL (Hue, Saturation, Lightness)
    but with some differences in how it represents and manipulates colors. 
    It’s often used in graphics software and computer vision applications for its 
    simplicity in specifying and adjusting color attributes.
    
    e.g:
    # 32-bit image
    image = pygame.image.load('../Assets/px.png').convert_alpha()
    arr = hsv1d_cp(image.get_buffer(), angle/360.0, format_32=True) 
    image = pygame.image.frombuffer(arr, (WIDTH, HEIGHT), "BGRA")
    
    # 24-bit image 
    image = pygame.image.load('../Assets/px.png').convert(24)
    arr = hsv1d_cp(image.get_buffer(), angle/360.0, format_32=False)
    
    Parameters
    ----------
    bgr_array : 
        numpy.ndarray 1d array, memoryviewslice uint8 data type containing 
       RGB(A) pixel format, works also with other format pixel (BGR, BGRA etc)
        
    shift     : 
        float; float value in range [0.0 ... 1.0] corresponding to 0 - 360 degrees. New hue value.
         
    format_32 : 
        bool True | for 'BGR' buffer type (24-bit) or False 'BGRA' (32-bit)
         

    Returns
    -------
    numpy.ndarray 1d array type uint8 new array containing pixels with rotated hue

    """

    if shift == 0:
        return numpy.array(bgr_array)

    assert 0.0 <= shift <= 1.0, \
        "\nArgument shift must be in range[0.0 ... 1.0]"

    # Only uint8 data is compatible
    if not is_uint8(bgr_array):
        raise TypeError("\nExpecting uint8 (unsigned char) data type got %s" % bgr_array.dtype)

    return hsv1d_cp_c(bgr_array, shift, format_32)



# --------------- WAVE EFFECT

@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef inline void wave(object surface_, const float rad, int size=5):
    """
    Apply Wave Effect to a Surface (Inplace)

    This function applies a wave effect to a Pygame surface, modifying it in place. 
    The effect is applied to the surface based on an angle (in radians) and the 
    number of sub-surfaces. It is compatible with 24-bit surfaces.

    The wave effect creates a dynamic, wave-like distortion, often used for water 
    or other fluid-like visual effects in games.

    Example usage:
        wave(surface, 8 * math.pi / 180.0 + frame_number, 5)  # Animate with a changing angle
        wave(surface, x * math.pi / 180.0, 5)  # Apply wave with a fixed angle

    Parameters
    ----------
    surface_ : pygame.Surface
        A Pygame surface, compatible with 24-bit color depth.

    rad : float
        The angle in radians for the wave effect. This value controls the wave's 
        motion over time. 

    size : int, optional, default=5
        The number of sub-surfaces used to create the wave effect. A higher number 
        results in a more complex wave.

    Returns
    -------
    None
        The function modifies the input surface in place, applying the wave effect.
    """


    # Ensure the surface is a valid pygame.Surface object
    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)

    # Validate that the size parameter is positive
    assert size > 0, "Argument size must be > 0"

    # Declare a Cython memoryview variable for the 3D RGB array (height x width x channels)
    cdef unsigned char [:, :, :] rgb_array

    try:
        # Attempt to retrieve a 3D view of the surface's pixel data
        # '3' indicates we're working with a 3D array (height, width, 3 channels - RGB)
        rgb_array = surface_.get_view('3')

    except Exception as e:
        # Raise an error if the reference to the surface's pixel data fails
        raise ValueError("\nCannot reference source pixels into a 3d array.\n %s " % e)

    # Apply a wave distortion effect on the image's pixel data in-place
    # This function modifies the rgb_array based on the given `rad` and `size` parameters
    wave_inplace_c(rgb_array, rad, size)





@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef void wave32(object surface_, const float rad, int size=5):
    """
    Apply Wave Effect to a 32-bit Surface (Inplace)

    This function applies a wave effect to a 32-bit Pygame surface, modifying it in place. 
    The effect is applied to both the RGB and alpha channels, meaning the wave will 
    also displace the alpha layer (transparency) of the surface. It is fully compatible 
    with 32-bit SDL surfaces, including those with an alpha channel.

    The wave effect creates a dynamic distortion that simulates the motion of waves, 
    often used for effects like water or fluid movement in games.

    Example usage:
        wave32(surface, x * math.pi / 180.0, 5)  # Apply wave effect with a rotating angle

    Parameters
    ----------
    surface_ : pygame.Surface
        A Pygame surface, compatible with 32-bit color depth and alpha channel (RGBA).

    rad : float
        The angle in radians for the wave effect, controlling its rotation over time. 

    size : int, optional, default=5
        The number of sub-surfaces used to create the wave effect. A higher value results 
        in a more detailed wave.

    Returns
    -------
    None
        The function modifies the input surface in place, applying the wave effect to both 
        the color and alpha channels.
    """

    # Ensure that the surface_ is a valid pygame.Surface object
    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)

    # Validate that the size argument is greater than 0
    assert size > 0, "Argument size must be > 0"

    # Declare variables to hold the width, height, and bit size of the surface
    cdef:
        Py_ssize_t w, h  # Width and height of the surface
        int bit_size     # Number of bytes per pixel (bit size)

    # Retrieve the width and height of the surface using get_size()
    w, h = surface_.get_size()

    # Get the number of bytes per pixel (bit size) of the surface
    bit_size = surface_.get_bytesize()

    # Convert the surface's pixel data into a contiguous numpy array and reshape it to 3D (height, width, bit_size)
    # 'get_view('0')' retrieves a raw memory view of the pixel data as a 1D array
    # We reshape it into a 3D array with dimensions (height, width, bit_size) for easier manipulation
    cdef unsigned char [:, :, ::1] rgba_array = \
        numpy.ascontiguousarray(surface_.get_view('0'), dtype = numpy.uint8).reshape(h, w, bit_size)

    # Apply the wave distortion effect to the RGBA array
    # 'wave32_c' modifies the image data in place based on the provided `rad` and `size` parameters
    wave32_c(rgba_array, rad, size)




@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef inline void wave_static(object surface_, array_, const float rad, int size=5):
    """
    Wave effect for static background (inplace)
     
    This function is different to the wave method as a copy of the 
    static background or game display is passed to the function as an argument `array_` to 
    improve the overall performances 

    Compatible 24-bit

    e.g:
    background = pygame.image.load('../Assets/px.png').convert(24)
    background = pygame.transform.smoothscale(background, (800, 600))
    background_cp = background.copy()
    wave_static_c(pixels3d(background), pixels3d(background_cp), FRAME * math.pi/180 , 5)
    SCREEN.blit(background, (0, 0))

    :param surface_: 
        Pygame.Surface compatible 24-bit
         
    :param array_: 
        numpy.ndarray shape (w, h, 3) type uint8 copy of the game display or image 
        to be modified (copy of the game display)
         
    :param rad: 
        float; angle in rad to rotate over time
         
    :param size: 
        int; Number of sub-surfaces, default is 5
         
    :return:
        void
        
    """

  # Ensure that the surface_ is a valid pygame.Surface object
    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)

    # Ensure that the array_ is a valid numpy ndarray
    assert PyObject_IsInstance(array_, numpy.ndarray), \
        "\nArgument surface_ must be a numpy ndarray type, got %s " % type(array_)

    # Validate that the size argument is greater than 0
    assert size > 0, "Argument size must be > 0"

    # Declare a variable to hold the RGB pixel data of the surface as a 3D array
    cdef unsigned char [:, :, :] rgb_array

    # Attempt to get a 3D view of the surface pixels (3D: height, width, bit depth)
    try:
        rgb_array = surface_.get_view('3')  # '3' represents a 3D memory view (height, width, bit depth)

    # Handle exceptions if surface_.get_view fails to get a 3D array
    except Exception as e:
        raise ValueError("\nCannot reference source pixels into a 3d array.\n %s " % e)

    # Apply a static wave effect on the image pixels using the wave_static_c function
    # It modifies the rgb_array in place based on the data from array_, with parameters rad and size
    wave_static_c(rgb_array, array_, rad, size)



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef inline void swirl(object surface_, rgb_array_cp, float degrees):
    """
    Swirl an image (inplace)
    
    The swirl effect is a visual distortion that creates
    a spiraling appearance in an image or graphic. 
    This effect can draw attention to specific areas of a design
    and add a sense of movement or dynamism. It can be used creatively
    in various contexts, from social media graphics to advertising
    and digital art.
    
    Works with 24 - 32 bit image format but not compatible 
    with 32-bit format due to the layer alpha.
    
    If the image is 32-bit with alpha channel, the layer alpha will be unchanged
     during the transformation. This will cause the layer alpha to bleed on the effect.
    If you do not which to see that undesirable effect, convert the image to 24-bit instead. 
    
    For 32-bit image with layer alpha, prefer the method swirl32 (designed for 32-bit).
     
    This algorithm uses a table of cos and sin
    
    e.g:
    background = pygame.image.load("../Assets/background.jpg").convert(24)
    background = pygame.transform.smoothscale(background, (WIDTH, HEIGHT))
    background_cp = background.copy()
    
    # in the game loop
    swirl(background_cp, pixels3d(background), angle)
    SCREEN.blit(background_cp, (0, 0))
    
    :param surface_: 
        pygame.Surface, works with 24 - 32 bit but not compatible with 32-bit due 
        to the alpha channel.
        
    :param rgb_array_cp:
        3d numpy.ndarray shape (w, h, 3) containing RGB pixel format. 
        Copy of the image to swirl. Both array must have same shapes and types
           
    :param degrees: 
        float; angle in degrees
         
    :return:
        void 
        
    """

    # Ensure that surface_ is a valid pygame.Surface object
    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)

    # Declare a variable to hold the RGB pixel data from the surface as a 3D array (height, width, color channels)
    cdef unsigned char [:, :, :] rgb_array

    # Try to get a 3D view (height, width, color channels) of the surface pixels
    try:
        rgb_array = surface_.get_view('3')  # '3' indicates a 3D memory view (height, width, bit-depth)

    # If there is an error accessing the pixel data, raise a ValueError
    except Exception as e:
        raise ValueError("\nCannot reference source pixels into a 3d array.\n %s " % e)

    # Declare variables for the width (w) and height (h) of the image
    cdef Py_ssize_t w, h

    # Get the width and height from the shape of the 3D rgb_array
    w, h = rgb_array.shape[:2]  # Extract the first two dimensions (height, width)

    # Apply a swirl effect to the image using the 'swirl_c' function, passing the image data and other parameters
    # 'rgb_array' is the original image, 'rgb_array_cp' is likely a copy for manipulation, and 'degrees' is the swirl angle
    swirl_c(w, h, rgb_array, rgb_array_cp, degrees)





@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef inline void swirl32(object surface_, float degrees):
    """
    Swirl an image (inplace)
    
    Compatible with both 24, 32-bit format (with or without alpha layer).
    
    The swirl effect is a visual distortion that creates
    a spiraling appearance in an image or graphic. 
    This effect can draw attention to specific areas of a design
    and add a sense of movement or dynamism. It can be used creatively
    in various contexts, from social media graphics to advertising
    and digital art.
    
    This algorithm uses a table of cos and sin for angle approximation
    
    Unlike the method `swirl`, this algorithm will take into account the layer alpha 
    during the transformation if the image is 32-bit with per pixel transparency.  

    e.g:
     swirl32(image, angle)

    :param surface_: 
        pygame.Surface, compatible 24 - 32 bit
         
    :param degrees: 
        float; angle in degrees
         
    :return: 
        void 
        
    """

    # Ensure that surface_ is a valid pygame.Surface object
    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)

    # Declare variables for image width (w), height (h), and bit depth (bit_size)
    cdef:
        Py_ssize_t w, h  # Width and height of the surface
        unsigned int bit_size  # Bit size of each pixel (for example, 32-bit for RGBA)

    # Get the size (width and height) of the surface and its byte size per pixel
    w, h = surface_.get_size()  # Get the width and height of the surface
    bit_size = surface_.get_bytesize()  # Get the number of bytes per pixel

    # Declare a 3D array to hold the RGBA pixel data from the surface
    cdef unsigned char [:, :, ::1] rgba_array

    # Attempt to create a contiguous array with shape (h, w, bit_size) from the surface pixel data
    try:
        # Use numpy to get a contiguous array of uint8 data, reshaped to (height, width, bit_size)
        rgba_array = numpy.ascontiguousarray(surface_.get_view('0'), dtype=uint8).reshape(h, w, bit_size)

    # If there's an error while trying to reference the pixel data, raise a ValueError with an appropriate message
    except Exception as e:
        raise ValueError("\nCannot reference source pixels into a 3d array.\n %s " % e)

    # Create a copy of the RGBA array as a 3D numpy array for further processing (presumably for the swirl effect)
    cdef const unsigned char[:, :, :] rgb = numpy.array(rgba_array, copy=True)

    # Apply the swirl effect to the image using the 'swirl32_c' function, passing width, height, 
    # the original RGBA array, the copied RGB array, and the swirl degree parameter.
    swirl32_c(w, h, rgba_array, rgb, degrees)








@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef inline void swirlf(object surface_, float degrees):
    """
    Swirl an image (inplace) floating point accuracy 
    
    This algorithm DO NOT use COS and SIN tables, it determines the angles with 
    floating point accuracy instead.
    
    compatible with 24-bit image format only
    
    The swirl effect is a visual distortion that creates
    a spiraling appearance in an image or graphic. 
    This effect can draw attention to specific areas of a design
    and add a sense of movement or dynamism. It can be used creatively
    in various contexts, from social media graphics to advertising
    and digital art.
      
    e.g:
     swirlf(surface_, angle)
    
    :param surface_: 
        pygame.Surface, compatible 24-bit
         
    :param degrees : 
        float; angle in degrees
        
    :return        : 
    void
     
    """

    # Ensure that the input surface_ is a valid pygame.Surface object
    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)

    # Declare a 3D array to hold the RGB pixel data from the surface
    cdef unsigned char [:, :, :] rgb_array

    # Attempt to retrieve the pixel data of the surface as a 3D array
    try:
        rgb_array = surface_.get_view('3')  # '3' indicates accessing the surface as a 3D array (height x width x color channels)
    except Exception as e:
        # Raise a ValueError if there is an issue referencing the source pixels
        raise ValueError("\nCannot reference source pixels into a 3d array.\n %s " % e)

    # Create a Fortran-contiguous array from the RGB data (Fortran order means column-major)
    cdef unsigned char [::1, :, :] rgb = numpy.asarray(rgb_array, order='F')

    # Declare variables for the image dimensions (width and height)
    cdef Py_ssize_t w, h
    w, h = rgb_array.shape[:2]  # Extract the width and height from the shape of the rgb_array

    # Apply the swirl effect to the image using the 'swirlf_c' function
    # Pass the width (w), height (h), the original rgb_array, 
    # array (rgb), and the degrees of swirl
    swirlf_c(w, h, rgb_array, rgb, degrees)





@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef inline void plasma_config(
        object surface_,
        int frame,
        float hue_=<float>1.0/<float>6.0,
        float sat_=<float>1.0/<float>6.0,
        float value_=<float>1.0/<float>8.0,
        float a_=<float>1.0/<float>255.0,
        float b_=<float>1.0/<float>12.0,
        float c_=<float>1.0/<float>12.0
):
    """
    
    
    CREATE A BASIC PLASMA EFFECT ON THE TOP OF A PYGAME SURFACE (INPLACE)

    Compatible 24 - 32 bit with or without alpha layer
    
    e.g:
     plasma_config(surface, frame_number)
    
    :param surface_:
        pygame.surface; compatible 24 - 32 bit

    :param frame: 
        integer; Variable that need to change over time

    :param hue_: 
        float; (Optional), default value 1.0/6.0 hue value factor
 
    :param sat_: 
        float; (Optional), default value 1.0/6.0 saturation value

    :param value_: 
        float; (Optional), default value 1.0/8.0 value factor

    :param a_: 
        float; (Optional), default value 1.0/255.0 control the plasma equation

    :param b_: 
        float; (Optional), default value 1.0/12.0 control the plasma equation

    :param c_: 
        float; (Optional), default value 1.0/12.0 control the plasma equation 

    :return: 
        void
    """

    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)
    cdef unsigned char [:, :, :] rgb_array
    try:
        rgb_array = surface_.get_view('3')

    except Exception as e:
        raise ValueError("\nCannot reference source pixels into a 3d array.\n %s " % e)

    plasma_inplace_c(rgb_array, frame, hue_, sat_, value_, a_, b_, c_)


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef inline void plasma(surface_, float frame, unsigned int [::1] palette_):
    """
    Apply Plasma Effect Inplace to a Surface

    This function generates a dynamic plasma effect on a pygame surface. The effect is created 
    using a palette of colors and evolves over time based on the `frame` value, producing 
    a fluid, glowing visual pattern. The plasma effect is applied directly to the surface, 
    modifying its pixels in place.

    The function works with both 24-bit and 32-bit surfaces (with or without an alpha channel).

    Example usage:
        plasma(surface, frame_number, palette)

    Parameters
    ----------
    surface_ : pygame.Surface
        A pygame surface compatible with 24-bit or 32-bit formats. The plasma effect is 
        applied directly to this surface, modifying its pixels in place.

    frame : float
        The current frame number, which drives the evolution of the plasma effect. This value 
        determines the shifting patterns and animation in the plasma effect.

    palette_ : numpy.ndarray (1D)
        A 1D array containing a palette of colors (as unsigned integers) used to generate 
        the plasma effect. The colors are applied cyclically to create the visual effect.

    Returns
    -------
    None
        This function modifies the `surface_` in place, applying the plasma effect to the surface.
    """


    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)

    plasma_c(surface_, frame, palette_)



# ---------------------------- BRIGHTNESS

@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef inline void brightness(object surface_, float shift_):
    """
    Brightness (inplace)

    This method control the pygame display or SDL surface brightness level
    
    Parameter shift is a float value in range [ -1.0 ... 1.0]. with +1.0 for the 
    maximum brightness. A value of 0.0 will not perform any changes to the original 
    surface
    
    Compatible 24, 32-bit (with or without alpha layer).
      
    e.g:
     brightness(surface, 0.2)
    
    :param surface_: 
        pygame.surface; Compatible 24 - 32 bit with or without alpha layer
        
    :param shift_: 
        float must be in range [ -1.0 ... 1.0 ]
        
    :return: 
        void
        
    """

    # Ensure that the input surface_ is a valid pygame.Surface object
    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)

    # If the shift is 0.0, no operation is needed, so return early
    if shift_ == 0.0:
        return

    # Ensure that the shift value is within the valid range [-1.0, 1.0]
    assert -1.0 <= shift_ <= 1.0, "\nArgument shift must be in range [-1.0 ... 1.0]"

    # Declare a 3D array to hold the RGB pixel data from the surface
    cdef unsigned char [:,:,:] rgb_array

    # Attempt to retrieve the pixel data of the surface as a 3D array
    try:
        rgb_array = surface_.get_view('3')  # '3' indicates accessing the surface as a 3D array (height x width x color channels)
    except Exception as e:
        # Raise a ValueError if there is an issue referencing the source pixels
        raise ValueError("\nCannot reference source pixels into a 3d array.\n %s " % e)

    # Apply the brightness shift effect to the image using the 'brightness_c' function
    # Pass the 3D RGB array and the shift value
    brightness_c(rgb_array, shift_)




@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef inline void brightness3d(unsigned char [:, :, :] rgb_array, float shift):
    """
    Control the brightness of an image given its array shape (w, h, n) uint8 
    data type, RGB(A) to BGR(A) (inplace)

    Allow to process 3d array directly 

    e.g:
    brightness3d(rgb_array)

    Parameters
    ----------
    :param rgb_array:  
        numpy.ndarray shape(w, h, n) uint8 data type, RGB(A) 
        (unsigned char 0...255) containing pixels
        
    :param shift:
        float must be in range [ -1.0 ... 1.0 ]

    Returns
    -------
    void

    """

    # If shift is 0, no operation is needed, so return early
    if shift == 0:
        return

    # Declare variables for width (w), height (h), and bit_size (bit_size)
    cdef:
        Py_ssize_t w, h
        Py_ssize_t bit_size = 0

    # Try to extract the shape of the rgb_array to get width, height, and bit size
    try:
        w, h, bit_size = rgb_array.shape[:3]  # Extract the shape dimensions (height, width, and bit_size)

        # Ensure that the bit size is either 3 (RGB) or 4 (RGBA)
        if bit_size not in (3, 4):
            raise ValueError('\nIncorrect bit_size, support only RGB(A)')

    # Handle the ValueError exception in case of invalid shape
    except ValueError as e:
        # If it's a memory view, print the flags of the numpy array
        if is_type_memoryview(rgb_array):
            print(numpy.array(rgb_array).flags)
        # Otherwise, print the flags of the numpy array directly
        else:
            print(rgb_array.flags)

        # Raise a new error with the details about the expected and actual array shapes
        raise ValueError(
            "\n%s\nExpecting array shape (w, h, n), "
            "RGB(A) got (%s, %s, %s)" % (e, w, h, len(rgb_array[:3])))

    # Ensure the data type is uint8 (unsigned char), as it is the expected type for RGB values
    if not is_uint8(rgb_array):
        raise TypeError("\nExpecting uint8 (unsigned char) data type got %s" % rgb_array.dtype)

    # Call the brightness function to apply the brightness shift on the RGB array
    brightness_c(rgb_array, shift)



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef void brightness1d(unsigned char [:] bgr_array, const float shift, bint format_32=False):
    """
    Control brightness of an image given its C buffer, 1d array shape (w, )
     BGR(A) or RGB(A) (inplace)

    e.g:
    # for 24-bit  
    array_bck = brightness1d_copy(background.get_buffer(), 0.1, False)
    background = pygame.image.frombuffer(array_bck, (800, 600), 'BGR')
    
    # for 32-bit 
    array_bck = brightness1d_copy(background.get_buffer(), 0.1, True)
    background = pygame.image.frombuffer(array_bck, (800, 600), 'BGRA')

    :param bgr_array:  
        numpy.ndarray shape(w,) uint8 data type, (unsigned char 0...255) containing pixels or
        bytearray buffer
        
    :param shift: 
        float; must be in range [ -1.0 ... 1.0 ]
        
    :param format_32: 
        bool; True for 'BGRA' buffer type (32-bit) or False 'BGR' (24-bit) 
        
    :return: void 
    """

    if shift == 0:
        return

    # Only uint8 data is compatible
    if not is_uint8(bgr_array):
        raise TypeError("\nExpecting uint8 (unsigned char) data type got %s" % bgr_array.dtype)

    brightness1d_c(bgr_array, shift, format_32)




@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef np.ndarray[np.uint8_t, ndim=1] brightness1d_copy(
        unsigned char [:] bgr_array,
        const float shift,
        bint format_32=False
):
    """
    Brightness control (return a copy)
    
    Control brightness of an image given its C buffer, 1d array shape (w, )
    The bgr_array is a C-buffer with pixel format BGR or BGRA 

    e.g:
    # for 24-bit
    array_bck = brightness1d_copy(background.get_buffer(), 0.1, False)
    background = pygame.image.frombuffer(array_bck, (800, 600), 'BGR')
    
    # for 32-bit 
    array_bck = brightness1d_copy(background.get_buffer(), 0.1, True)
    background = pygame.image.frombuffer(array_bck, (800, 600), 'BGRA')
    
    :param bgr_array:
        numpy.ndarray shape(w,) uint8 data type, (unsigned char 0...255)
        containing pixels or bytearray buffer
        
    :param shift: 
        float; must be in range [ -1.0 ... 1.0 ]. 
        Brightness value.
        
    :param format_32: 
        bool;  True for 'BGRA' buffer type (32-bit) or False 'RGB' (24-bit)
         
    :return: 
        Return a copy of the original SDL surface with adjusted brightness
         
    """

    if shift == 0:
        return numpy.array(bgr_array)

    assert -1.0 <= shift <= 1.0, \
        "\nArgument shift must be in range[-1.0 ... 1.0]"

    # Only uint8 data is compatible
    if not is_uint8(bgr_array):
        raise TypeError("\nExpecting uint8 (unsigned char) data type got %s" % bgr_array.dtype)

    return brightness1d_copy_c(bgr_array, shift, format_32)




@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef inline object brightness_copy(object surface_, const float shift):
    """
    Brightness (return a copy)

    Apply the transformation to a new SDL surface
    This method control the pygame display brightness level

    Parameter shift is a float value in range [ -1.0 ... 1.0]. with +1.0 for the
    maximum brightness. A value of 0.0 will not perform any change to the original
    SDL surface

    Compatible 24, 32-bit (with or without alpha layer).

    e.g:
    new_surface = brightness_copy(surface, 0.2)

    :param surface_: 
        pygame.surface; Compatible 24 - 32 bit with or without alpha layer
        
    :param shift: 
        float must be in range [ -1.0 ... 1.0 ]
        
    :return: 
        pygame surface 24-bit format, without alpha layer
        
    """

    # If the shift is 0.0, there's no change needed, so return the surface as is
    if shift == 0.0:
        return surface_

    # Declare variables for width (w), height (h), and bit size (bit_size)
    cdef:
        Py_ssize_t w, h
        unsigned int bit_size

    # Get the width and height of the surface and the byte size of the pixel format
    w, h = surface_.get_size()  # surface_.get_size() returns a tuple (width, height)
    bit_size = surface_.get_bytesize()  # surface_.get_bytesize() returns the byte size per pixel

    # Check if the argument surface_ is a pygame.Surface type
    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)

    # Ensure that the shift value is between -1.0 and 1.0
    assert -1.0 <= shift <= 1.0, "\nArgument shift must be in range [-1.0 ... 1.0]"

    # Attempt to reference the pixel data of the surface as a 3D array (RGB format)
    cdef unsigned char [:,:,:] rgb_array
    try:
        rgb_array = surface_.get_view('3')  # '3' represents a 3D view (RGB) of the surface's pixel data

    except Exception as e:
        # If an exception occurs (e.g., unable to access the pixel data), raise an error
        raise ValueError("\nCannot reference source pixels into a 3d array.\n %s " % e)

    # Apply the brightness adjustment by passing the RGB array and shift value to the brightness_copy_c function
    return brightness_copy_c(rgb_array, shift)





@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef inline void brightness_exclude(
        object surface_,
        const float shift_,
        color_=(0, 0, 0)
):
    """

    Brightness adjustment with color exclusion (inplace)
    
    Exclusion:
    Set the parameter color to exclude a specific color from the transformation process.
    parameter shift control the brightness transformation, with +1.0 being the maximum 
    brightness possible. 
    
    Compatible with 24, 32-bit images 
    
    e.g
    # 24-bit
    image = pygame.image.load('../Assets/px.png').convert(24)
    brightness_exclude(image, +0.5, color=(0, 0, 0))
    
    # 32-bit
    image = pygame.image.load('../Assets/px.png').convert_alpha()
    brightness_exclude(image, +0.5, color=(0, 0, 0))

    :param surface_: 
        pygame.surface; 32|24-bit surface compatible
        
    :param shift_: 
        float; must be in range [ -1.0 ... +1.0 ], +1.0 is the maximum 
        brightness effect, zero will have no effect.
    
    :param color_: 
        tuple; RGB values to be excluded from the process e.g (10, 22, 0), 
        default is black tuple(0, 0, 0) 
        
    :return: 
        void
        
    """

   # Ensure the argument surface_ is a pygame.Surface object
    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)

    # If the shift value is 0.0, no change is needed, so return early
    if shift_ == 0.0:
        return

    # Ensure that the shift value is between -1.0 and 1.0
    assert -1.0 <= shift_ <= 1.0, \
        "Argument shift must be in range[-1.0 ... 1.0]"

    # Declare a variable to hold the RGB array of pixel data from the surface
    cdef unsigned char [:,:,:] rgb_array

    # Try to get a 3D view of the surface's pixel data (RGB format)
    try:
        rgb_array = surface_.get_view('3')  # '3' indicates a 3D view (RGB) of the surface

    # If an exception occurs while trying to access the pixel data, raise a ValueError
    except Exception as e:
        raise ValueError("\nCannot reference source pixels into a 3d array.\n %s " % e)

    # Call the function to adjust the brightness of the surface's pixel data, passing the shift and color values
    brightness_ex_c(rgb_array, shift_, color_)




@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef inline void brightness_bpf(
        object surface_,
        const float shift_,
        unsigned char bpf_threshold = 64):
    """
    
    Brightness adjustment with *bpf exclusion (inplace)
    *bpf stand for bright pass filter
    
    Exclusion:
    bpf_threshold is an integer value in range [0..255] that 
    determines the pixels threshold for the brightness algorithm. 
    The RGB sum below this threshold will not be included in the process.  
    
    Compatible with 24, 32-bit images
    
    e.g:
     24-bit
    image = pygame.image.load('../Assets/px.png').convert()
    brightness_bpf(image, 0.5, bpf_threshold=200)

    # 32-bit 
    image = pygame.image.load('../Assets/px.png').convert_alpha()
    brightness_bpf(image, 0.5, bpf_threshold=200)
    
    :param surface_: 
        Pygame.Surface compatible with 24 - 32 bit
         
    :param shift_:
        float; must be in range [-1.00 ... +1.00] this value control the brightness
         
    :param bpf_threshold: 
        integer; value in range [0 ... 255]. Bright pass filter value. 
        Equivalent to a threshold RGB. e.g sum of pixel values < threshold
        will not be modified. Default value is 64.
         
    :return: 
        void 

    """

    # If the shift_ value is 0, no change is needed, so return early
    if shift_ == 0:
        return

    # Ensure the argument surface_ is a pygame.Surface object
    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)

    # Ensure that the shift_ value is between -1.0 and 1.0
    assert -1.0 <= shift_ <= 1.0, \
        "Argument shift must be in range[-1.0 ... 1.0]"

    # Declare a variable to hold the RGB array of pixel data from the surface
    cdef unsigned char [:,:,:] rgb_array

    # Try to get a 3D view of the surface's pixel data (RGB format)
    try:
        rgb_array = surface_.get_view('3')  # '3' indicates a 3D view (RGB) of the surface

    # If an exception occurs while trying to access the pixel data, raise a ValueError
    except Exception as e:
        raise ValueError("\nCannot reference source pixels into a 3d array.\n %s " % e)

    # Call the function to adjust the brightness of the surface's pixel data with the given shift_ and bpf_threshold
    brightness_bpf_c(rgb_array, shift_, bpf_threshold)



# ---------------------------- SATURATION

@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef inline void saturation(object surface_, float shift_):
    """
    Saturation (inplace)

    This method control the saturation level of the pygame display or surface/texture

    e.g:
    saturation(surface, 0.2)
    
    
    :param surface_: 
        pygame.Surface; compatible 24 - 32 bit
        
    :param shift_  : 
        float must be in range [ -1.0 ... 1.0]
         
    :return: 
        void 
    
    """

    # If the shift_ value is 0, no change is needed, so return early
    if shift_ == 0:
        return

    # Ensure the argument surface_ is a pygame.Surface object
    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)

    # Clamp the shift_ value to be within the range [-1.0, 1.0]
    if shift_ < -1.0:
        shift_ = -1.0  # If shift_ is less than -1, set it to -1

    if shift_ > 1.0:
        shift_ = 1.0   # If shift_ is greater than 1, set it to 1

    # Declare a variable to hold the RGB array of pixel data from the surface
    cdef unsigned char [:, :, :] rgb_array

    # Try to get a 3D view of the surface's pixel data (RGB format)
    try:
        rgb_array = surface_.get_view('3')  # '3' indicates a 3D view (RGB) of the surface

    # If an exception occurs while trying to access the pixel data, raise a ValueError
    except Exception as e:
        raise ValueError("\nCannot reference source pixels into a 3d array.\n %s " % e)

    # Call the function to adjust the saturation of the surface's pixel data with the given shift_
    saturation_c(rgb_array, shift_)



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef inline void saturation3d(unsigned char [:, :, :] rgb_array, float shift):
    """
    Saturate 3d array directly (inplace)

    Modify the saturation level of an image by referencing
    the surface/image array data.
    The array must be type uint8 and shape (w, h, 3) containing RGB format pixel but
    any other format is also compatible
    The output image will be 24-bit format without layer alpha

    e.g:
    saturation3d(surface, 0.2)


    :param rgb_array:
        numpy.ndarray or memoryviewslice shape (w, h, 3) type uint8 containing
        RGB or any other pixel format and referencing an SDL surface or image.

    :param shift  :
        float must be in range [ -1.0 ... 1.0]

    :return:
        void (change apply inplace)

    """

    # If the shift value is 0, no change is needed, so return early
    if shift == 0:
        return

    # Clamp the shift value to be within the range [-1.0, 1.0]
    if shift < -1.0:
        shift = -1.0  # If shift is less than -1, set it to -1

    if shift > 1.0:
        shift = 1.0   # If shift is greater than 1, set it to 1

    # Declare variables for the dimensions and bit size of the image array
    cdef:
        Py_ssize_t w, h  # Width and height of the image
        Py_ssize_t bit_size = 0  # The bit size (number of color channels)

    # Try to get the dimensions (width, height) and bit size of the RGB array
    try:

        # Get the shape of the RGB array, which should return (height, width, channels)
        w, h, bit_size = rgb_array.shape[:3]

        # Check if the bit size (number of color channels) is either 3 (RGB) or 4 (RGBA)
        if bit_size not in (3, 4):
            raise ValueError('\nIncorrect bit_size, support only RGB(A)')  # Raise an error if bit size is invalid

    # If a ValueError occurs while retrieving the shape, print the array flags and raise the error
    except ValueError as e:
        # Check if the array is a memoryview slice
        if is_type_memoryview(rgb_array):
            print(numpy.array(rgb_array).flags)  # Print flags for memoryview slice
        # If it's a numpy array, print its flags
        else:
            print(rgb_array.flags)

        # Raise a ValueError with the shape of the array and the expected dimensions
        raise ValueError(
            "\n%s\nExpecting array shape (w, h, n), "
            "RGB(A) got (%s, %s, %s)" % (e, w, h, len(rgb_array[:3])))

        # Only uint8 data (unsigned char) is compatible
    if not is_uint8(rgb_array):
        raise TypeError("\nExpecting uint8 (unsigned char) data type got %s" % rgb_array.dtype)

    # Call the function to adjust the saturation of the RGB array with the given shift value
    saturation_c(rgb_array, shift)



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef inline void saturation1d(
    unsigned char [:] buffer,
    const float shift,
    bint format_32=False):

    """
    Saturate 1d array directly (inplace) 
    
    Control saturation level of an image given its C data buffer, 1d array shape (w, )
    type uint8 RGB(A) to BRG(A) or any other pixel format (inplace)
    
    Nevertheless for 32-bit image, the alpha channel must be place at the end of
    the pixel, such as RGB(A) or BGR(A)
    
    e.g:
    # for 32-bit 
    image = pygame.image.load("../Assets/px.png").convert_alpha()
    saturation1d(image.get_buffer(), -0.5, True)
    saturation1d(im.get_view('0'), 0.5, True)
    
    # for 24 
    image = pygame.image.load("../Assets/px.png").convert(24)
    saturation1d(image.get_buffer(), 0.3, False)
    
    :param buffer:  
        numpy.ndarray or memoryviewslice shape(w,) uint8 data type, (unsigned char 0...255) 
        containing RGB(A), BGR(A) or any other pixel format.
        
    :param shift: 
        float must be in range [ -1.0 ... 1.0 ], This value control the saturation level
    
    :param format_32:
        bool True|False. Set to False for 'RGB' buffer type (24-bit) or True for 'RGBA' (32-bit), 
        this bit enable/disable the layer alpha.
        
    :return:
        void, Inplace transformation. 
        Final image is same pixel format than input image 
    """

    if shift == 0:
        return

    assert -1.0 <= shift <= 1.0, \
        "\nArgument shift (float) must be in range[-1.0 ... 1.0]"

    # Only uint8 data is compatible
    if not is_uint8(buffer):
        raise TypeError("\nExpecting uint8 (unsigned char) data type got %s" % buffer.dtype)

    saturation1d_c(buffer, shift, format_32)


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef np.ndarray[np.uint8_t, ndim=1]  saturation1d_cp(
        const unsigned char [:] buffer,
        const float shift,
        bint format_32=False
):
    """
    Saturate am image using a C-buffer (return copy) 
    
    Control saturation level of an image given its C data buffer, 1d array shape (w, )
    type uint8 RGB(A) to BRG(A) or any other pixel format (inplace)
    
    Nevertheless for 32-bit image, the alpha layer must be place at the end of 
    the pixel format such as RGB(A) or BGR(A)
    
    e.g:
    # for buffer 32-bit 
    image = pygame.image.load("../Assets/px.png").convert_alpha()
    new_buffer = saturation1d_cp_c(image.get_buffer(), -0.5, True)
   
    # for 24-bit
    image = pygame.image.load("../Assets/px.png").convert(24)
    new_buffer = saturation1d_cp_c(im.get_view('0'), 0.5, False)
    
    Parameters
    ----------
    buffer: 
        numpy.ndarray or memoryviewslice shape(w,) uint8 data type, (unsigned char 0...255) 
        containing RGB(A)|BGR(A) or any other format pixels.
        
    shift: 
        float must be in range [ -1.0 ... 1.0 ]. 
        This value control the saturation level.
    
    format_32: 
        bool True | for 'RGB' buffer type (24-bit) or False 'RGBA' (32-bit), 
        this bit enable/disable the layer alpha it works indifferently of the pixel format.

    Returns
    ----------
    numpy.ndarray 1d array shape (w, ) type uint8 containing same pixel format than input array 

    """

    if shift == 0:
        return numpy.asarray(buffer)

    assert -1.0 <= shift <= 1.0, \
        "\nArgument shift (float) must be in range[-1.0 ... 1.0]"

        # Only uint8 data is compatible
    if not is_uint8(buffer):
        raise TypeError("\nExpecting uint8 (unsigned char) data type got %s" % buffer.dtype)

    return saturation1d_cp_c(buffer, shift, format_32)




# ------------------------------------ OTHER


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef inline void heatconvection(
        object surface_,
        float amplitude,
        float center = 0.0,
        float sigma = 2.0,
        float mu = 0.0):
    """

    Heat flow convection
    
    Convection (or convective heat transfer) is the transfer of heat from one place to another 
    due to the movement of fluid. Although often discussed as a distinct method of heat transfer, 
    convective heat transfer involves the combined processes of conduction (heat diffusion).    
    This effect can be use to simulate air turbulence or heat flow/convection
    it applies a gaussian transformation at the base of the image (vertical flow)   
    
    # for 32-24 bit image format 
     image = pygame.image.load("../Assets/fire.jpg").convert()
     b = math.cos(i * 3.14 / 180.0) * random.uniform(0, 2)
     heatconvection(image, abs(b) * random.uniform(20.0, 80.0),
         0, sigma = random.uniform(0.8, 4), mu_ = b)
    # Restore the original image 
     image = image_copy.copy()

    :param surface_  : 
        pygame.Surface; compatible 24 - 32 bit
         
    :param amplitude   : 
        Control the maximum amplitude (pixels displacement on the Y-axis,
        vertical effect) of the gaussian equation. No transformation if 
        amplitude equal zero. example of an variable amplitude issue from a 
        periodic function: b = math.cos(i * 3.14 / 180.0) * random.uniform(0, 2)
        with i linear.
         
    :param center   : 
        Control the center of the gaussian equation (if center equal zero,
        the Gauss equation is centered (default is 0.0)
                   
    :param sigma    : 
        float; sigma value of the gauss equation, a small value will create 
        a narrow effect while a stronger value will wider the effect. 
        Please refers to the gaussian distribution for further analysis on the 
        sigma values (default is 2.0).
        
    :param mu       : 
        float; mu value of the gauss equation. when mu is periodic such as
        a cosine trigonometric function, it allows to displace the effect 
        along the X-axis (default is 0.0).
        
    """

    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)

    if amplitude == <float>0.0:
        return

    cdef unsigned char [:,:,:] rgb_array
    try:
        rgb_array = surface_.get_view('3')

    except Exception as e:
        raise ValueError("\nCannot reference source pixels into a 3d array.\n %s " % e)

    heatconvection_inplace_c(rgb_array, amplitude, center, sigma, mu)




@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef inline void horizontal_glitch(
        object surface_,
        const float deformation,
        const float frequency,
        const float amplitude
):
    """
    Horizontal glitch (inplace)

    Deform the pygame display horizontally 
    
    e.g:
    # for 24 - 32 bit
    horizontal_glitch(background, deformation=0.5, frequency=0.08, amplitude=FRAME % 20)

    :param surface_    : 
        pygame.Surface; compatible 24 - 32 bit
         
    :param deformation : 
        float; Angle in radians, this value control the angle variation over time.
        
    :param frequency   : 
        float; signal frequency, factor that amplify the angle variation
        
    :param amplitude   : 
        float; cos amplitude value
        
    :return            : 
        void
    
    """

    # Assert that the surface_ argument is of type pygame.Surface
    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)

    # Declare a 3D array 'rgb_array' that will hold the image pixel data (in unsigned char format)
    cdef unsigned char [:,:,:] rgb_array

    # Attempt to reference the pixels of the surface_ as a 3D array
    try:
        rgb_array = surface_.get_view('3')  # Get the pixel data as a 3D memory view

    # If an exception occurs during the reference, raise a ValueError with a message
    except Exception as e:
        raise ValueError("\nCannot reference source pixels into a 3d array.\n %s " % e)

    # Call the horizontal_glitch_c function to apply a horizontal glitch effect to the image
    # The function is given the rgb_array and the deformation, frequency, and amplitude parameters
    horizontal_glitch_c(rgb_array, deformation, frequency, amplitude)




@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef inline void horizontal_sglitch(
        object surface_,
        object array_,
        const float deformation,
        const float frequency,
        const float amplitude
):
    """
    Glitch for static image background (inplace)

    Deform the pygame display to create a glitch effect

    e.g:
     # for 24 - 32 bit
    horizontal_sglitch(background, bgr_array, deformation=0.5, frequency=0.08, amplitude=FRAME % 20)

    :param surface_    : 
        pygame.Surface; compatible 24 - 32 bit
         
    :param array_      : 
        numpy.ndarray pixel copy
        
    :param deformation : 
        float; Angle in radians, this value control the angle variation over time
        
    :param frequency   : 
        float; signal frequency, factor that amplify the angle variation
        
    :param amplitude   : 
        float; cos amplitude value
        
    :return: 
        void
        
    """

    # Assert that the surface_ argument is of type pygame.Surface
    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)

    # Assert that the array_ argument is of type numpy.ndarray
    assert PyObject_IsInstance(array_, numpy.ndarray), \
        "\nArgument rgb_array must be a numpy.ndarray type, got %s " % type(array_)

    # Declare a 3D array 'rgb_array' to hold the pixel data from the surface_ (in unsigned char format)
    cdef unsigned char [:,:,:] rgb_array

    # Attempt to reference the pixels of the surface_ as a 3D array
    try:
        rgb_array = surface_.get_view('3')  # Retrieve the pixel data as a 3D memory view

    # If an exception occurs during the reference, raise a ValueError with a detailed message
    except Exception as e:
        raise ValueError("\nCannot reference source pixels into a 3d array.\n %s " % e)

    # Call the horizontal_sglitch_c function to apply a horizontal glitch effect to the image
    # This function takes the rgb_array (pixel data of the surface), array_ (an additional numpy array),
    # and parameters: deformation, frequency, and amplitude to control the glitch effect
    horizontal_sglitch_c(rgb_array, array_, deformation, frequency, amplitude)





@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef inline void bpf(object surface_, int threshold = 128):
    """
    
    BPF, bright pass filter (inplace)

    Conserve only the brightest pixels of a surface
    
    e.g:
    # 24 bit 
    image = pygame.image.load('../Assets/px.png').convert(24)
    bpf(image, threshold=60)
    
    # 32 bit 
    image = pygame.image.load('../Assets/px.png').convert_alpha()
    bpf(image, threshold=60)

    :param surface_ : 
        pygame.Surface; compatible 24 - 32 bit
         
    :param threshold: 
        integer; Bright pass threshold default is 128
        
    :return: 
        void 
    
    """

    # Assert that the surface_ argument is of type pygame.Surface
    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)

    # Declare a 3D array 'rgb_array' to hold the pixel data from the surface_ (in unsigned char format)
    cdef unsigned char [:,:,:] rgb_array

    # Attempt to reference the pixels of the surface_ as a 3D array
    try:
        rgb_array = surface_.get_view('3')  # Retrieve the pixel data as a 3D memory view

    # If an exception occurs during the reference, raise a ValueError with a detailed message
    except Exception as e:
        raise ValueError("\nCannot reference source pixels into a 3d array.\n %s " % e)

    # Declare variables for width (w) and height (h) of the image
    cdef int w, h
    w, h = rgb_array.shape[:2]  # Extract width and height from the shape of the rgb_array

    # Call the bpf_inplace_c function to apply the BPF (brightness-preserving filter) effect
    # The function takes the rgb_array (pixel data of the surface), width (w), height (h), and threshold
    # to control the intensity of the effect.
    bpf_inplace_c(rgb_array, w, h, threshold)



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef inline void bloom(
        object surface_,
        int threshold_,
        bint fast_=False,
        object mask_=None
):
    """
    Create a bloom effect (inplace)

    Compatible 24-32 bit SDL surface / image
    Optional mask to filter the bloom effect by controlling the transparency (alpha).
    Optional fast argument to improve the overall performance x10 - x80 improvement but reduce the 
    effect appearance. 

    e.g
    # check the demo_bloom_mask.py script in the Demo folder 
    bloom(SCREEN, threshold = BPF, fast=True, mask = mask)

    Parameters
    ----------
    
    surface_ : 
        pygame.Surface; Game display or texture compatible 24, 32-bit format
    
    threshold_ : 
        integer; Threshold value uint8 in range [0 ... 255].
        Bright pass filter threshold value to detect bright pixels within the texture or image.
    
    fast_ : 
        bool; True | False; If True the bloom effect will be approximated
        and only the x16 subsurface will be processed to maximize the overall processing time. 
        Default is False.
    
    mask_ : 
        numpy.ndarray or memoryviewslice shape (w, h) type uint8 containing values in range
        (0 .. 255) representing the mask alpha. Array (w, h) filled with 255 will render and bloom 
        the entire image. Array (w, h) filled with zero will disable the bloom effect. Any values
        in between ]0 and 255[ will filter the pixels and create selective bloom effect.
    
    Returns
    -------
    void
    
    """

    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)

    bloom_c(surface_, threshold_, fast_, mask_)



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef inline np.ndarray[np.uint32_t, ndim=2] fisheye_footprint(
        const int w,
        const int h,
        const unsigned int centre_x,
        const unsigned int centre_y
):
    """
    Create a fisheye lens model holding pixel coordinates of a surface

    The model variables w & h must have the same dimensions than the projected surface.  
    
    e.g 
     width, height = surface.get_size()
     f_model = fisheye_footprint(w=width, h=height, centre_x=width >> 1, centre_y=height >> 1)
     fisheye(surface, f_model)
    
    :param w       : 
        integer; width of the surface to project ino the fisheye model
         
    :param h       : 
        integer; height of the surface to project into the fisheye model
        
    :param centre_y: 
        integer; centre position y of the effect
        
    :param centre_x: 
        integer; centre position x of the effect
    
    :return        : 
        Return a numpy.ndarray type (w, h, 2) of unsigned int representing the
        fisheye model (coordinates of all pixels passing through the fisheye lens model)
      
    """

    return fisheye_footprint_c(w, h, centre_x, centre_y)



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef inline void fisheye_footprint_param(
        tmp_array_,
        float centre_x,
        float centre_y,
        float param1_,
        float focal_length,
        ):

    """
    Create a fisheye model to hold the pixel coordinates.

    This version contains two additional variables param1_ & focal_length 
    to control the fisheye model aspect.
    
    e.g
    check Demo/demo_magnifier.py for a real time example 
    
    tmp = numpy.ndarray((400, 400, 2), dtype=numpy.uint32, order='C')
    fisheye_footprint_param(tmp, 200, 200, 1., .6)
    
    :param tmp_array_: 
        numpy.ndarray shape (w, h, 2) of unsigned int.
        tmp_array_ array shape will determine the fisheye model.
        (Declare tmp_array_ outside the main loop).
        
    :param centre_x: 
        float; fisheye x centre coordinate.
        correspond to half the length of the fisheye model.
        
    :param centre_y: 
        float; fisheye y centre coordinate.
        Correspond to half the width of the fisheye model.
        
    :param param1_: 
        float; Control the fisheye aspect. param1_ > 1.0 converge to the centre, 
        param1_ < 1.0 diverge from the centre
        
    :param focal_length: 
        float; Control the fisheye type focal_length > 1 diverging lens
        focal_length < 0 converging lens
        
    :return: 
        void
         
    """

    assert isinstance(tmp_array_, numpy.ndarray), \
        "\nArgument tmp_array_ must be a numpy.ndarray type, got %s " % type(tmp_array_)

    fisheye_footprint_param_c(
        tmp_array_, centre_x, centre_y, param1_, focal_length)




@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef inline void fisheye_footprint_param_c(
        unsigned int [:, :, :] tmp_array_,
        float centre_x,
        float centre_y,
        float param1,
        float focal_length,
):
    """
    
    Create a fisheye model to hold the pixel coordinates.

    This version contains two additional variables param1_ & focal_length 
    to control the fisheye model aspect.
    
    e.g
    check Demo/demo_magnifier.py for a real time example 
    
    tmp = numpy.ndarray((400, 400, 2), dtype=numpy.uint32, order='C')
    fisheye_footprint_param_c(tmp, 200, 200, 1., .6)
    
    
    :param tmp_array_: 
        numpy.ndarray shape (w, h, 2) of unsigned int.
        tmp_array_ array shape will determine the fisheye model.
        (Declare tmp_array_ outside the main loop).
        
    :param centre_x: 
        float; fisheye x centre coordinate.
        correspond to half the length of the fisheye model.
        
    :param centre_y: 
        float; fisheye y centre coordinate.
        Correspond to half the width of the fisheye model.
        
    :param param1: 
        float; Control the fisheye aspect. param1_ > 1.0 converge to the centre, 
        param1_ < 1.0 diverge from the centre
        
    :param focal_length: 
        float; Control the fisheye type focal_length > 1 diverging lens
        focal_length < 0 converging lens
        
    :return: 
        void

    """

    # Declare variables with appropriate types for width (w), height (h), and other necessary values
    cdef:
        # Extract the width (w) and height (h) from the shape of tmp_array_
        Py_ssize_t w = <object>tmp_array_.shape[0]
        Py_ssize_t h = <object>tmp_array_.shape[1]
        
        # Declare other variables used for calculations, such as coordinates (x, y),
        # transformed coordinates (nx, ny), radius (r), angle (theta), and others
        int y, x
        float ny, ny2, nx, nx2, r, theta, nr
        
        # Constants for normalization and the center of the image
        float c1 = <float>2.0 / w  # Normalization constant for x-coordinates
        float c2 = <float>2.0 / h  # Normalization constant for y-coordinates
        float w2 = centre_x  # x-coordinate of the center of the image
        float h2 = centre_y  # y-coordinate of the center of the image

    # Use 'nogil' to allow for multithreading and parallelization (avoiding the Global Interpreter Lock)
    with nogil:
        
        # Loop over the x-coordinates (prange allows parallel execution across threads)
        for x in prange(w, schedule=SCHEDULE, num_threads=THREADS):
            # Normalize the x-coordinate and square it
            nx = x * c1 - <float>1.0
            nx2 = nx * nx
            
            # Loop over the y-coordinates
            for y in range(h):
                # Normalize the y-coordinate and square it
                ny = y * c2 - <float>1.0
                ny2 = ny * ny
                
                # Calculate the radial distance from the center
                r = <float>sqrt(nx2 + ny2)
                
                # If the distance is greater than 1 (outside of a unit circle), set the pixel value to black (0, 0)
                if r > 1:
                    tmp_array_[x, y, 0] = <unsigned int>0  # Red channel (0)
                    tmp_array_[x, y, 1] = <unsigned int>0  # Green channel (0)
                    continue  # Skip the remaining calculations and move to the next pixel
                
                # Calculate the new radius using the given parameters
                nr = (r + <float>param1 - <float>sqrt(<float>1.0 - (nx2 + ny2))) * <float>focal_length
                
                # Calculate the angle (theta) from the center using the arctangent function
                theta = <float>atan2(ny, nx)
                
                # Calculate the transformed coordinates using polar to cartesian conversion
                # Store the results in the tmp_array_ at the x, y location
                tmp_array_[x, y, 0] = <unsigned int> (nr * <float>cos(theta) * w2 + w2)  # New x-coordinate
                tmp_array_[x, y, 1] = <unsigned int> (nr * <float>sin(theta) * h2 + h2)  # New y-coordinate





@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef inline void fisheye(
        object surface_,
        unsigned int [:, :, ::1] fisheye_model
):
    """
    Display surface or gameplay throughout a lens effect (inplace)

    Compatible with 24-bit only 
    
    A fisheye lens is an ultra wide-angle lens that produces strong visual
    distortion intended to create a wide panoramic or hemispherical image
    
    Display a fisheye lens effect in real time.
    In order to accomplish a real time calculation, this algorithm is using 
    a pre-calculated lens model transformation stored in a numpy.ndarray, 
    argument fisheye_model (numpy.ndarray shape (w, h, 2) of type uint).
    The numpy array contains the pixel's coordinates of a surface after 
    a lens transformation. All calculation are performed upstream. 
    
    Use the function fisheye_footprint_c to create the pre-calculated array.
    This method needs to be called once only. 
    
    The fisheye lens transformation is applied inplace.
    
    e.g 
     width, height = surface.get_size()
     f_model = fisheye_footprint(w=width, h=height, centre_x=width >> 1, centre_y=height >> 1)
     fisheye(surface, f_model)

    
    Parameters
    ----------
    
    surface_ : 
        pygame.Surface; compatible 24 bit 
    
    fisheye_model : 
        numpy.ndarray shape (w, h, 2) int32, fisheye model containing uint values 
        x' & y'. x' & y' are the surface pixels coordinates after transformation. Values calculated
        upstream with the function fisheye_footprint_c 
        
    Returns
    -------
    void
    
    """

    # Ensure that 'surface_' is an instance of pygame.Surface
    # This verifies that the input surface is a valid pygame surface object.
    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)

    # Ensure that 'fisheye_model' is either a numpy.ndarray or a cython.memoryview type.
    # This checks the type of the fisheye model to ensure it is a valid memoryview or numpy array.
    assert PyObject_IsInstance(fisheye_model, (cython.view.memoryview, numpy.ndarray)), \
        "\nArgument fisheye_model must be a numpy.ndarray or a cython.view.memoryview type, " \
        "got %s " % type(fisheye_model)

    # Declare a memoryview of unsigned char to hold the pixel data of the surface.
    # This will allow access to the pixel data in an efficient way.
    cdef unsigned char [:,:,:] rgb_array

    # Attempt to reference the surface pixels into a 3D array (RGB data).
    # If the surface cannot be viewed as a 3D array, an error will be raised.
    try:
        rgb_array = surface_.get_view('3')

    except Exception as e:
        # If an error occurs while referencing the pixels, raise a ValueError with details.
        raise ValueError("\nCannot reference source pixels into a 3d array.\n %s " % e)

    # Apply the fisheye effect to the surface in-place.
    # The 'fisheye_inplace_c' function modifies the pixel data in 'rgb_array' based on the fisheye model.
    fisheye_inplace_c(rgb_array, fisheye_model)




@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef inline void tv_scan(surface_, int space=5):
    """
    TV scanline effect on pygame surface (inplace)

    The space between each scanline can by adjusted with the space value.

    e.g:
    tv_scan(image, space=10)

    Parameters
    ----------

    surface_ :
        pygame.Surface compatible 24, 32-bit format.

    space :
        integer; space between each lines.
        Choose a constant or use a variable for a dynamic effect

    Returns
    -------
    void

    """

    # Ensure that 'surface_' is an instance of pygame.Surface
    # This checks that the 'surface_' input is of the correct type (pygame.Surface).
    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)

    # Ensure that 'space' is greater than 0
    # This ensures that the value for 'space' is a positive number, as it is expected to control the space of the effect.
    assert space > 0, "Argument space must be >0"

    # Declare a memoryview of unsigned char to hold the pixel data of the surface
    # This memoryview will allow efficient access to the surface's pixel data in a 3D array format.
    cdef unsigned char [:, :, :] rgb_array

    # Attempt to reference the surface's pixel data into a 3D array (RGB data).
    # If this fails, an exception is raised and caught.
    try:
        rgb_array = surface_.get_view('3')

    except Exception as e:
        # If an error occurs while referencing the pixels, raise a ValueError with the exception message.
        raise ValueError("\nCannot reference source pixels into a 3d array.\n %s " % e)

    # Apply the TV scanline effect in-place on the surface's pixel data
    # The 'tv_scanline_c' function modifies the pixel data to create the TV scanline effect using the 'space' parameter.
    tv_scanline_c(rgb_array, space)



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef inline tuple ripple(
        int rows_,
        int cols_,
        const float [:, ::1] previous_,
        float [:, ::1] current_,
        unsigned char [:, :, :] array_,
        float dispersion_ = 0.008
):
    """
    
    Ripple effect without background deformation

    Check demo_ripple.py, demo_ripple1.py in the Demo folder 
    e.g:
    previous, current = ripple(width, height, previous, current, back_array,  dispersion_=0.008)
       
    rows_       : 
        integer; Screen width or surface width
        
    cols_       : 
        integer; Screen height or surface height
        
    previous_   : 
        numpy.ndarray type (w, h) type float; array use for the transformation. 
        Array holding the previous_ data
          
    current_    : 
        numpy.ndarray type (w, h) type float; array use for the transformation. 
        Array holding the current_ data
         
    rgb_array   : 
        numpy.ndarray type (w, h, 3) type unsigned char. Array containing the background image RGB pixels.
        The content of this array is invariant (static background image).
         
    dispersion_ :  
        float; ripple dampening factor, higher values decrease the ripple effect 
        radius default 0.008

    Returns
    -------
    Return a tuple containing 2 arrays (current_, previous_)
    see Parameters for each array sizes
        
    
    """

   # Ensure 'previous_' is a valid numpy.ndarray or cython memoryview
    # This ensures that 'previous_' is either a numpy ndarray or a cython memoryview. 
    # If it's not, an error message is shown indicating the incorrect type.
    assert PyObject_IsInstance(previous_, (numpy.ndarray, cython.view.memoryview)), \
        "\nArgument previous_ must be a numpy.ndarray type got %s " % type(previous_)

    # Ensure 'current_' is a valid numpy.ndarray or cython memoryview
    # This ensures that 'current_' is also either a numpy ndarray or a cython memoryview.
    assert PyObject_IsInstance(current_, (numpy.ndarray, cython.view.memoryview)), \
        "\nArgument current_ must be a numpy.ndarray type got %s " % type(current_)

    # Ensure 'array_' is a valid numpy.ndarray or cython memoryview
    # Similarly, this ensures that 'array_' (rgb_array) is either a numpy ndarray or a cython memoryview.
    assert PyObject_IsInstance(array_, (numpy.ndarray, cython.view.memoryview)), \
        "\nArgument rgb_array must be a numpy.ndarray type got %s " % type(array_)

    # Get dimensions (width and height) of 'previous_' array
    # Extract the width and height of the 'previous_' image array.
    cdef Py_ssize_t prev_w, prev_h
    prev_w, prev_h = previous_.shape[:2]

    # Get dimensions (width and height) of 'current_' array
    # Extract the width and height of the 'current_' image array.
    cdef Py_ssize_t curr_w, curr_h
    curr_w, curr_h = current_.shape[:2]

    # Get dimensions (width and height) of 'array_' array
    # Extract the width and height of the 'array_' image array.
    cdef Py_ssize_t arr_w, arr_h
    arr_w, arr_h = array_.shape[:2]

    # Ensure that all input arrays (previous_, current_, array_) have the same dimensions (width and height)
    # The widths and heights of 'previous_', 'current_', and 'array_' must match.
    # If they don't, an assertion error is raised with information about the mismatched dimensions.
    assert prev_w == curr_w and prev_w == arr_w \
        and prev_h == curr_h and prev_h == arr_h, \
        "\n Array sizes mismatch (previous_ w: %s, h: %s; " \
        "current_ w: %s, h: %s; bgr_array w: %s, h: %s " % (prev_w, prev_h, curr_w, curr_h,
        arr_w, arr_h)

    # Call the 'ripple_c' function to apply the ripple effect on the images.
    # This function takes the image dimensions (rows_, cols_), the previous, current, 
    # and array image data, and a dispersion factor to generate a ripple effect.
    return ripple_c(rows_, cols_, previous_, current_, array_, dispersion_)



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef inline tuple ripple_seabed(
    int cols_, int rows_,
    const float [:, ::1] previous_,                 # type numpy.float32 (w, h)
    float [:, ::1] current_,                        # type numpy.float32 (w, h)
    const unsigned char [:, :, ::1] texture_array_, # type numpy.ndarray (w, h, 3)
    unsigned char [:, :, :] background_array_,      # type numpy.ndarray (w, h, 3)
    float dispersion_ = 0.008
):
    """
    Ripple effect with background deformation

    Check demo_ripple_seabed.py in the Demo folder 
    
    e.g:
    previous, current, back_array = ripple_seabed(height, width, previous,\
      current, texture_array, back_array, dispersion_=0.009)
    
    
    Parameters
    ----------
    
    cols_ : 
        integer; Screen width or surface width
        
    rows_ : 
        integer; Screen height or surface height
        
    previous_ : 
        numpy.ndarray type (w, h) type float; array use for the transformation. 
        Array holding the previous_ data
         
    current_ : 
        numpy.ndarray type (w, h) type float; array use for the transformation.
        Array holding the current_ data 
        
    texture_array_ : 
        numpy.ndarray type (w, h, 3) type unsigned char. 
        Array containing the background image RGB pixels.
        The content of this array is invariant (static background image). 
        
    background_array_ :
        numpy.ndarray type (w, h, 3) of type unsigned char containing the background image RGB pixels.
        The background array is equivalent to the texture array with current_ ripple effect transformation.
         
    dispersion_ :
        float; ripple dampening factor, higher values decrease the ripple effect 
        radius default 0.008
 
    Returns
    -------
    Return a tuple containing 3 arrays (current_, previous_, bck_array)
    see Parameters for each array sizes

    """


    # Ensure 'previous_' is a valid numpy.ndarray or cython memoryview
    # This check ensures that 'previous_' is either a numpy ndarray or a cython memoryview. 
    # If it is not, an error is raised with an appropriate message indicating the incorrect type.
    assert PyObject_IsInstance(previous_, (numpy.ndarray, cython.view.memoryview)), \
        "\nArgument previous_ must be a numpy.ndarray type got %s " % type(previous_)

    # Ensure 'current_' is a valid numpy.ndarray or cython memoryview
    # Similarly, 'current_' is checked to ensure it is either a numpy ndarray or a cython memoryview.
    assert PyObject_IsInstance(current_, (numpy.ndarray, cython.view.memoryview)), \
        "\nArgument current_ must be a numpy.ndarray type got %s " % type(current_)

    # Ensure 'texture_array_' is a valid numpy.ndarray or cython memoryview
    # The 'texture_array_' argument is also validated to be either a numpy ndarray or a cython memoryview.
    assert PyObject_IsInstance(texture_array_, (numpy.ndarray, cython.view.memoryview)), \
        "\nArgument array must be a numpy.ndarray type got %s " % type(texture_array_)

    # Get the width and height of the 'previous_' array
    # The dimensions (width and height) of the 'previous_' array are extracted for later comparison.
    cdef Py_ssize_t prev_w, prev_h
    prev_w, prev_h = previous_.shape[:2]

    # Get the width and height of the 'current_' array
    # The dimensions (width and height) of the 'current_' array are extracted as well.
    cdef Py_ssize_t curr_w, curr_h
    curr_w, curr_h = current_.shape[:2]

    # Get the width and height of the 'texture_array_' array
    # The dimensions (width and height) of the 'texture_array_' array are also extracted.
    cdef Py_ssize_t arr_w, arr_h
    arr_w, arr_h = texture_array_.shape[:2]

    # Ensure that all input arrays (previous_, current_, texture_array_) have the same dimensions (width and height)
    # This 'assert' statement checks if all three arrays have matching width and height. 
    # If the dimensions don't match, an error message is raised showing the mismatched sizes.
    assert prev_w == curr_w and prev_w == arr_w \
        and prev_h == curr_h and prev_h == arr_h, \
        "\n Array sizes mismatch (previous_ w: %s, h: %s; " \
        "current_ w: %s, h: %s; texture_array_ w: %s, h: %s " % (prev_w, prev_h, curr_w, curr_h,
        arr_w, arr_h)

    # Call the 'ripple_seabed_c' function to apply the ripple effect using all the input arrays.
    # This function applies the ripple effect on the previous, current, and texture arrays, and it also 
    # uses a background array and a dispersion factor for the effect. It returns the modified result.
    return ripple_seabed_c(
        rows_, cols_,
        previous_,
        current_,
        texture_array_,
        background_array_,
        dispersion_)



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef inline void heatmap(object surface_, bint rgb_=True):
    """
    Transform an image into a heatmap equivalent (in-place).
    
    This function modifies the given image surface to apply a heatmap effect. 
    The transformation is applied directly to the surface, so no new object 
    is returned. It also allows the user to choose whether the heatmap 
    should be in RGB or BGR color model.

    Example:
        # Load an image, convert to an alpha surface, and apply heatmap
        image = pygame.image.load("../Assets/px.png").convert_alpha()
        heatmap(image, True)
    
    Parameters
    ----------
    surface_ : pygame.Surface
        A pygame.Surface object, typically in 24-bit or 32-bit image format 
        (compatible with pygame). The image to which the heatmap effect will 
        be applied.
    
    rgb_ : bool, optional
        If True, the image will be transformed into an RGB-based heatmap. 
        If False, the transformation will use the BGR-based heatmap model. 
        Default is True (RGB).

    Returns
    -------
    None
        This function operates in-place, meaning the input surface is modified 
        directly without returning a new object.
    
    Raises
    ------
    TypeError
        If 'surface_' is not a valid pygame.Surface object.
    """
    
    # Ensure 'surface_' is a valid pygame.Surface object
    # This assertion checks if the provided surface is an instance of 
    # pygame.Surface. If not, it raises an error with the type information.
    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)

    # Apply the heatmap transformation on the surface_
    # The actual transformation is done by the heatmap_c function, 
    # which processes the surface in-place based on the rgb_ parameter.
    heatmap_c(surface_, rgb_)




@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef inline predator_vision(
        object surface_,
        unsigned int sobel_threshold = 12,
        unsigned int bpf_threshold   = 50,
        unsigned int bloom_threshold = 50,
        bint inv_colormap            = False,
        bint fast                    = False,
        int blend                    = pygame.BLEND_RGB_ADD
):
    """
    Predator Vision Mode
    
    This function simulates the predator's vision using a series of image processing 
    filters. It applies Sobel edge detection, bright pass filter (BPF), bloom effects, 
    and a colormap inversion to the given surface. Optionally, it allows for a faster 
    processing mode and blending effects.

    Example:
        surface_ = predator_vision(
            image.copy(), sobel_threshold=80, bpf_threshold=0,
            bloom_threshold=0, inv_colormap=True, fast=True)
    
    Parameters
    ----------
    surface_ : pygame.Surface
        A pygame.Surface object, compatible with 24-bit or 32-bit formats. 
        This is the image to which the predator vision effect will be applied.
    
    sobel_threshold : int, optional
        The threshold for Sobel edge detection, used to detect edges in the image.
        The default value is 12.
    
    bpf_threshold : int, optional
        The threshold for the Bright Pass Filter (BPF), used to detect and highlight 
        bright pixels. The default value is 50.
    
    bloom_threshold : int, optional
        The intensity of the bloom effect, which adds a glow around bright pixels. 
        The default value is 50.
    
    inv_colormap : bool, optional
        If True, the colormap will be inverted, changing the color scheme to resemble 
        a predator's vision more closely. The default value is False.
    
    fast : bool, optional
        If True, a faster processing mode is used. This will reduce the quality 
        in exchange for faster processing time. The default value is False.
    
    blend : int, optional
        The blending mode to apply after all effects have been processed. This can 
        be a value from `pygame.BLEND_*` modes. The default is `pygame.BLEND_RGB_ADD`.

    Returns
    -------
    pygame.Surface
        A new pygame.Surface object with the predator vision effect applied. The 
        surface is in 24-bit format.
    
    """

    # Create a copy of the input surface to preserve the original
    cp = surface_.copy()

    # Apply Sobel edge detection filter: either fast or normal based on the 'fast' flag
    if fast:
        sobel_fast(cp, sobel_threshold, factor_=1)  # Use fast Sobel edge detection
    else:
        sobel(cp, sobel_threshold)  # Use normal Sobel edge detection

    # Apply Bright Pass Filter (BPF) to highlight bright areas
    bpf(surface_, bpf_threshold)

    # Apply a bloom effect to simulate glowing bright pixels
    shader_bloom_fast1(surface_, bloom_threshold)

    # Apply the heatmap effect with optional colormap inversion
    heatmap_c(surface_, inv_colormap)

    # Blend the processed surface ('cp') onto the original surface ('surface_')
    surface_.blit(cp, (0, 0), special_flags=blend)

    # Return the surface in 24-bit format after all effects are applied
    return surface_.convert(24)




@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef inline void blood(
        object surface_,
        const float [:, :] mask_,
        float percentage_
):
    """
    Blood effect (inplace)

    This function applies a blood effect to a given surface using a mask array 
    that defines the contour of the blood effect. The surface and mask must have 
    the same dimensions. The percentage parameter determines the intensity of the 
    blood effect, with 1.0 representing full intensity.

    Example:
        background = pygame.image.load("../Assets/Aliens.jpg").convert()
        background = pygame.transform.smoothscale(background, (800, 600))
        background.convert(32, RLEACCEL)
        image = background.copy()
        
        blood_surface = pygame.image.load("../Assets/redvignette.png").convert_alpha()
        blood_surface = pygame.transform.smoothscale(blood_surface, (800, 600))
        BLOOD_MASK = numpy.asarray(pygame.surfarray.pixels_alpha(blood_surface) / 255.0, numpy.float32)
        
        # Then call the method in your main loop (percentage must vary over time)
        blood(image, BLOOD_MASK, percentage)

    Parameters
    ----------
    surface_ : pygame.Surface
        The surface (e.g., game display) to which the blood effect will be applied.
        It must be in a compatible 24-bit or 32-bit format.

    mask_ : numpy.ndarray or cython.view.memoryview
        A normalized array or memoryview (of shape (w, h) with type float) 
        representing the blood mask. The values must be in the range [0.0, 1.0], 
        where 1.0 represents full intensity of the effect.

    percentage_ : float
        A value in the range [0.0, 1.0] that determines the intensity of the blood effect. 
        A value of 1.0 applies the full effect, while 0.0 applies no effect.

    Returns
    -------
    void
        The function modifies the given surface in place (i.e., it has no return value).
    """
    
    # Assert that the surface is a valid pygame.Surface
    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)

    # Assert that the mask is either a numpy.ndarray or cython memoryview
    assert PyObject_IsInstance(mask_, (numpy.ndarray, cython.view.memoryview)), \
        "\nArgument mask must be a numpy.ndarray or cython memoryview type, got %s " % type(mask_)

    # Get the size (width, height) of the surface
    cdef Py_ssize_t w, h
    w, h = surface_.get_size()

    # Get the dimensions of the mask
    cdef Py_ssize_t mask_w, mask_h
    mask_w, mask_h = mask_.shape[:2]

    # Ensure the dimensions of the surface and mask match
    assert w == mask_w and h == mask_h, "\nSurface size and mask size mismatch"

    # Try to obtain a reference to the surface's RGB array
    cdef unsigned char [:, :, :] rgb_array
    try:
        rgb_array = surface_.get_view('3')  # '3' denotes a 3D array view for RGB
    except Exception as e:
        raise ValueError("\nCannot reference source pixels into a 3d array.\n %s " % e)

    # Call the C function that applies the blood effect in place
    blood_inplace_c(w, h, rgb_array, mask_, percentage_)



# ---------------- MIRRORING


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef inline np.ndarray[np.uint8_t, ndim=3] mirroring_array(const unsigned char [:, :, :] rgb_array):

    """
    Mirroring numpy array (return a copy)

    This method returns a numpy.ndarray with mirrored pixels
    
    e.g
    rgb_array = mirroring_array(pixels3d(image))
    surface = make_surface(rgb_array)
     
    Parameters
    ----------
    
    rgb_array : 
        numpy.ndarray; Array shape (w, h, 3) of type uint8 containing RGB or any other pixel format 
        such as BGR etc. 

    Returns
    -------
    returns a numpy ndarray shape (w, h, 3) of type uint8 identical to the input array with 
    mirrored pixels
    
    """

    cdef:
        Py_ssize_t w, h
        Py_ssize_t byte_size

    w, h, byte_size = rgb_array.shape[:3]

    if byte_size not in (3, 4):
        raise ValueError(
            '\nArgument array must be shape (w, h, 3 | 4) got (%s, %s, %s)' % (w, h, byte_size))

    cdef unsigned char [:, :, :] new_array = empty((w, h, 3), uint8)

    # Transform the memoryviewslice into a numpy.ndarray
    return numpy.ndarray(
        shape=(w, h, 3),
        buffer=mirroring_c(w, h, rgb_array, new_array),
        dtype=uint8)




@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef inline void mirroring(object surface_):
    """
    Mirroring effect (inplace)

    This method creates a mirrored image of the given surface by reflecting it
    horizontally. The effect is applied directly to the surface (in-place).

    Example:
        # Load an image, apply the mirroring effect
        image = pygame.image.load("../Assets/px.png").convert()
        mirroring(image)
    
    Parameters
    ----------
    surface_ : pygame.Surface
        A pygame surface compatible with 24-bit or 32-bit formats that will 
        undergo the mirroring effect.
    
    Returns
    -------
    void
        The function modifies the given surface in place, meaning it does not 
        return a new surface, but rather alters the input surface.
    """

    # Assert that the input surface is a valid pygame.Surface object
    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)

    # Declare variables to hold the width and height of the surface
    cdef:
        Py_ssize_t w, h

    # Try to get the RGB data array of the surface
    cdef unsigned char [:, :, :] rgb_array
    try:
        rgb_array = surface_.get_view('3')  # Access the surface as a 3D array (RGB channels)
    except Exception as e:
        # If an error occurs while referencing the pixels, raise a ValueError
        raise ValueError(
            "\nCannot reference source pixels into a 3d array.\n %s " % e)

    # Get the width and height of the surface from the RGB array
    w, h = rgb_array.shape[:2]

    # Create a copy of the RGB array with column-major order (F order)
    cdef const unsigned char [::1, :, :] rgb_array_copy = numpy.asarray(rgb_array, order='F')

    # Apply the mirroring effect in place by calling the C function
    mirroring_inplace_c(w, h, rgb_array, rgb_array_copy)



# ------------------------


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef inline void sharpen(object surface_):
    """
    Sharpen an image (in-place) using a 3x3 kernel.

    This function applies a sharpening filter to the image on the given surface.
    The filter uses a 3x3 kernel to enhance the edges and details by increasing contrast
    around edges, making the image appear sharper.

    Example:
        # Load an image and apply the sharpen effect
        image = pygame.image.load("../Assets/px.png").convert()
        sharpen(image)

    Parameters
    ----------
    surface_ : pygame.Surface
        A Pygame surface compatible with 24-bit or 32-bit image formats
        that will be modified in-place to apply the sharpen effect.

    Returns
    -------
    void
        The function modifies the given surface in place, meaning no new surface is returned.
        The original surface will be sharpened.
    """

    # Ensure that the input is a valid pygame.Surface object
    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)

    # Declare a variable to hold the buffer of the surface
    cdef unsigned char[::1] bgr_array

    try:
        # Try to get a reference to the pixel data in the surface as a buffer
        bgr_array = surface_.get_buffer()

    except Exception as e:
        # If an error occurs while referencing the pixels, raise a ValueError
        raise ValueError("\nCannot reference source pixels into a buffer.\n %s " % e)

    # Ensure that the data is of type uint8 (8-bit unsigned integer)
    if not is_uint8(bgr_array):
        raise TypeError("\nExpecting uint8 (unsigned char) data type got %s" % bgr_array.dtype)

    # Get the width and the total length (size) of the surface
    cdef:
        Py_ssize_t w = surface_.get_width()  # Width of the surface
        Py_ssize_t l = bgr_array.shape[0]  # Total number of elements (length of the buffer)

    # Create a copy of the pixel data to allow in-place manipulation
    # This avoids modifying the original buffer directly, ensuring the transformation is applied correctly
    cdef unsigned char [::1] bgr_array_cp = numpy.ndarray(shape=l, buffer=bgr_array, dtype=uint8).copy()

    # Get the bit size of the surface (24-bit or 32-bit)
    cdef int bitsize = surface_.get_bitsize()

    # Validate that the surface is either 24-bit or 32-bit (common formats for color images)
    if bitsize not in (24, 32):
        raise ValueError('\nIncompatible surface format got %s bitsize, compatible 24 or 32-bit only.' % bitsize)

    # Call the C function to apply the sharpening effect using a 1D convolution with a 3x3 kernel
    sharpen_1d_c(w, l, bgr_array, bgr_array_cp, True if bitsize == 32 else False)





@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef inline void sharpen_1d(
        const Py_ssize_t w,
        const Py_ssize_t h,
        unsigned char [::1] bgr_array,
        bint format_32=False):

    """
    Sharpen array (inplace)
    
    Compatible with BGR or BGR(A) array types, but works with any other pixel format e.g RGB or RGB(A).
    The sharpen method can be apply directly to a 1d array using a 3 x 3 kernel described below.
    Set the variable format_32=True if the array contains alpha transparency (array type BGR(A))
    otherwise set it to False.
    
    pixels convoluted outside image edges will be set to adjacent edge value
        [0 , -1,  0]
        [-1,  5, -1]
        [0 , -1,  0]
        
    e.g 
    # for 32 bit array data BGR(A) 
    sharpen_1d(w, h, im.get_buffer(), True)
    
    # for 24-bit array data BGR type
    sharpen_1d(w, h, im.get_buffer(), False)
      
    Parameters
    ----------
    
    w : 
        integer; array width 
    
    h : 
        integer; array height
    
    bgr_array :  
        numpy.ndarray shape (w, ) of type uint8 containing RGB pixels
    
    format_32 : 
        bool True | 'BGR' for 24-bit array (BGR) or False | 'BGRA' for 32-bit array (BGRA) 

    Returns
    -------
    void
    
    """

    if not is_uint8(bgr_array):
        raise TypeError("\nExpecting uint8 (unsigned char) data type got %s" % bgr_array.dtype)

    cdef:

        Py_ssize_t l = bgr_array.shape[0]
        # below create a copy False of the array and do not reference the pixels.
        # The real time transformation of the identical copy of the array will not be functional as all the pixels
        # undergo constant transformations. It is then necessary to load the pixels from a copy of the source array
        # to implement the inplace transformation. Such as below
        unsigned char [::1] bgr_array_cp = numpy.ndarray(shape=l, buffer=bgr_array, dtype=uint8).copy()

    sharpen_1d_c(w, l, bgr_array, bgr_array_cp, format_32)



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef inline void sharpen_1d_c(
        const Py_ssize_t w,
        const Py_ssize_t l,
        unsigned char [::1] bgr_array,
        const unsigned char [::1] bgr_array_cp,
        bint format_32=False)nogil:

    """
    
    Sharpen array (inplace)
    
    Compatible with BGR or BGR(A) array type, but works with any other pixel format e.g RGB or RGB(A).
    
    The sharpen method can be applied directly to a 1d array using a 3 x 3 kernel described below.
    Set the variable format_32=True if the array contains alpha transparency; array type RGBA or BGRA
    otherwise set it to False.
    
    pixels convoluted outside image edges will be set to adjacent edge value
        [0 , -1,  0]
        [-1,  5, -1]
        [0 , -1,  0]

    e.g 
    # for 32 bit array data BGR(A) 
    sharpen_1d_c(w, l, image.get_buffer(), image_copy.get_buffer(), True)
    
    
    # for 24-bit array data BGR type
    sharpen_1d_c(w, l, image.get_buffer(), image_copy.get_buffer(), False)
    

    Parameters
    ----------
    
    w: 
        integer; array width 
    
    l: 
        integer; array total length such as w * h * byte_size
    
    bgr_array : 
        numpy.ndarray shape(w, ) uint8 BGR(A) 
        (unsigned char 0...255) containing the image pixels format.
    
    bgr_array_cp : 
        numpy.ndarray shape (w, ) uint8 BGR(A), empty array used during the transformation 
    
    format_32 : 
        bool True | 'BGR' for 24-bit array (BGR) or False | 'BGRA' for 32-bit array (BGRA) 

    Returns
    -------
    Void

    """

    cdef short bitsize
    bitsize = 3 if format_32 == False else 4

    cdef:

        int i, r, g, b
        unsigned int row = w * bitsize
        const unsigned char * p1
        const unsigned char * p2
        const unsigned char * p3
        const unsigned char * p4
        const unsigned char * p5

    for i in prange(0, l, bitsize, schedule=SCHEDULE, num_threads=THREADS):

        p3 = &bgr_array_cp[ i ]

        if row + bitsize < i < l - row - bitsize:

            # 3x3 kernel (sharpen)
            # [ 0, -1, 0 ]
            # [ -1, 5, -1 ]
            # [ 0, -1, 0 ]

            p1 = &bgr_array_cp[ i - row ]
            p2 = &bgr_array_cp[ i - bitsize ]
            p4 = &bgr_array_cp[ i + bitsize ]
            p5 = &bgr_array_cp[ i + row ]

            # blue
            b = -p1[0] -p2[0] +p3[0] * 5 -p4[0] -p5[0]

            # green
            g = -(p1 + 1)[0] -(p2 + 1)[0] +(p3 + 1)[0] * 5 -(p4 + 1)[0] -(p5 + 1)[0]

            # red
            r = -(p1 + 2)[0] -(p2 + 2)[0] +(p3 + 2)[0] * 5 -(p4 + 2)[0] -(p5 + 2)[0]

            if r < 0:
                r = <unsigned char>0

            if g < 0:
                g = <unsigned char>0

            if b < 0:
                b = <unsigned char>0

            if r > 255:
                r= <unsigned char>255

            if g > 255:
                g = <unsigned char>255

            if b > 255:
                b = <unsigned char>255

        else:
            # set pixels that cannot be convoluted.
            # pixels located on the edge of the image
            # mode BGR
            bgr_array[ i     ] = (p3 + 2)[0]
            bgr_array[ i + 1 ] = (p3 + 1)[0]
            bgr_array[ i + 2 ] = p3[0]
            continue


        # Alpha channel is unchanged
        bgr_array[ i    ] = b
        bgr_array[ i + 1] = g
        bgr_array[ i + 2] = r



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef inline np.ndarray[np.uint8_t, ndim=1] sharpen_1d_cp(
        const Py_ssize_t w,
        const Py_ssize_t h,
        unsigned char [::1] bgr_array,
        bint format_32=False
):
    """
    Sharpen array (return a new array)
    
    Return a new array instead of applying the changes inplace.
    Compatible with BGR or BGR(A) array types, but works with any other pixel format e.g RGB or RGB(A).
    The sharpen method can be apply directly to a 1d array using a 3 x 3 kernel described below.
    Set the variable format_32=True if the array contains alpha transparency (array type BGR(A))
    otherwise set it to False.
    

    pixels convoluted outside image edges will be set to adjacent edge value
        [0 , -1,  0]
        [-1,  5, -1]
        [0 , -1,  0]

    e.g 
    # for 32 bit array data BGR(A) 
    arr=sharpen_1d_cp(w, h, im.get_buffer(), True)
    im = pygame.image.frombuffer(arr, (w, h), "RGBA").convert_alpha()

    # for 24-bit array data BGR type
    arr=sharpen_1d_cp(w, h, im.get_buffer(), False)
    im = pygame.image.frombuffer(arr, (w, h), "RGB")

   
    Parameters
    ----------
    w : 
        integer; array width
    
    h : 
        integer; array height
    
    bgr_array : 
        numpy.ndarray shape (w, ) of type uint8 containing BGR pixels
    
    format_32 :  
        bool True | 'BGR' for 24-bit array (BGR) or False | 'BGRA' for 32-bit array (BGRA) 

    Returns
    -------
    1d numpy.ndarray shape (w, ) uint8 similar to the input array with sharpen pixels. 
     
    """

    if not is_uint8(bgr_array):
        raise TypeError("\nExpecting uint8 (unsigned char) data type got %s" % bgr_array.dtype)

    cdef:
        Py_ssize_t l = bgr_array.shape[0]
        unsigned char[ ::1 ] tmp_array = numpy.empty(l, uint8)

    return numpy.ndarray(
        shape=l,
        buffer=sharpen_1d_cp_c(w, l, bgr_array, tmp_array, format_32),
        dtype=uint8)




@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef inline unsigned char [::1] sharpen_1d_cp_c(
        const Py_ssize_t w,
        const Py_ssize_t l,
        unsigned char [::1] bgr_array,
        unsigned char [::1] bgr_array_cp,
        bint format_32=False)nogil:
    
    """
    Sharpen array (return a copy)
    
    Return a copy of the array instead of applying the changes inplace
    Compatible with BGR or BGR(A) array type, but works with any other pixel format e.g RGB or RGBA(A).
    The sharpen method can be applied directly to a 1d array using a 3 x 3 kernel described below.
    Set the variable format_32=True if the array contains alpha transparency; array type BGRA
    otherwise set it to False.

    pixels convoluted outside image edges will be set to adjacent edge value
        [0 , -1,  0]
        [-1,  5, -1]
        [0 , -1,  0]

    e.g 
    # for 32 bit array data BGR(A) 
    arr=sharpen_1d_cp_c(w, l, im.get_buffer(), True)

    # for 24-bit array data BGR type
    arr = sharpen_1d_cp(w, l, bytearray(pygame.image.tobytes(im, "RGB")), False)
    im = pygame.image.frombuffer(arr, (w, h), "RGB")


    Parameters
    ----------
    w : 
        integer; array width
    
    l : 
        integer; array length (width * height * bitsize)
    
    bgr_array : 
        numpy.ndarray shape (w, ) of type uint8 containing BGR pixels
    
    bgr_array_cp : 
        numpy.ndarray shape (w, ) uint8 BGR(A), empty array used during the transformation 
         
    format_32 :  
        bool True | 'BGR' for 24-bit array (BGR) or False | 'BGRA' for 32-bit array (BGRA) 

    Returns
    -------
    memoryviewslice shape (w, ) uint8 similar to the input array with sharpen pixels. 
     

    """
    cdef short bitsize
    bitsize = 3 if format_32 == False else 4

    cdef:

        int i, r, g, b
        unsigned int row = w * bitsize
        const unsigned char * p1
        const unsigned char * p2
        const unsigned char * p3
        const unsigned char * p4
        const unsigned char * p5

    for i in prange(0, l, bitsize, schedule = SCHEDULE, num_threads = THREADS):

        p3 = &bgr_array[ i ]

        if format_32:
            if (p3 + 3)[0] == 0:
                bgr_array_cp[ i     ] = p3[0]
                bgr_array_cp[ i + 1 ] = (p3 + 1)[0]
                bgr_array_cp[ i + 2 ] = (p3 + 2)[0]
                bgr_array_cp[ i + 3 ] = (p3 + 3)[ 0 ]
                continue

        if row + bitsize < i < l - row - bitsize:

            # 3x3 kernel (sharpen)
            # [ 0, -1, 0 ]
            # [ -1, 5, -1 ]
            # [ 0, -1, 0 ]

            p1 = &bgr_array[ i - row ]
            p2 = &bgr_array[ i - bitsize ]
            p4 = &bgr_array[ i + bitsize ]
            p5 = &bgr_array[ i + row ]

            # blue
            b = -p1[ 0 ] - p2[ 0 ] + p3[ 0 ] * 5 - p4[ 0 ] - p5[ 0 ]

            # green
            g = -(p1 + 1)[ 0 ] - (p2 + 1)[ 0 ] + (p3 + 1)[ 0 ] * 5 - (p4 + 1)[ 0 ] - (p5 + 1)[ 0 ]

            # red
            r = -(p1 + 2)[ 0 ] - (p2 + 2)[ 0 ] + (p3 + 2)[ 0 ] * 5 - (p4 + 2)[ 0 ] - (p5 + 2)[ 0 ]

            if r < 0:
                r = <unsigned char> 0

            if g < 0:
                g = <unsigned char> 0

            if b < 0:
                b = <unsigned char> 0

            if r > 255:
                r = <unsigned char> 255

            if g > 255:
                g = <unsigned char> 255

            if b > 255:
                b = <unsigned char> 255

        else:
            # set pixels that cannot be convoluted.
            # pixels located on the edge of the image
            # mode RGB
            bgr_array_cp[ i     ] = (p3 + 2)[ 0 ]
            bgr_array_cp[ i + 1 ] = (p3 + 1)[ 0 ]
            bgr_array_cp[ i + 2 ] = p3[ 0 ]

            if format_32:
                bgr_array_cp[ i + 3 ] = (p3 + 3)[ 0 ]

            continue

        # Set RGB
        bgr_array_cp[ i     ] = r
        bgr_array_cp[ i + 1 ] = g
        bgr_array_cp[ i + 2 ] = b

        # Alpha channel is unchanged
        if format_32:
            bgr_array_cp[ i + 3 ] = (p3 + 3)[0]

    return bgr_array_cp




@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef inline void sharpen32(object surface_):
    """

    Sharpen image using 3 x 3 kernel (inplace)
    
    Compatible with 24, 32-bit images 
    
    e.g:
    # for 32-bit 
    sharpen32(image)
    
    :param surface_: 
        pygame.Surface; compatible 24, 32-bit
         
    :return: 
        void
         
    """

    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)

    cdef:
        Py_ssize_t w, h

    w, h = surface_.get_size()

    # create a 3d array BGR or BGRA format depends on the input surface
    cdef unsigned char [:, :, ::1] bgra_array = \
        numpy.ascontiguousarray(surface_.get_buffer(), dtype = numpy.uint8).reshape(h, w, surface_.get_bytesize())

    sharpen_inplace_c(bgra_array)




@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef inline void dirt_lens(
        object surface_,
        object lens_model_,
        int flag_=BLEND_RGB_ADD,
        float light_ = 0.0
):
    """
    Dirt lens effect (inplace)
    
    This function display a dirt lens texture on the top of your game display to 
    simulate a camera artefact or realistic camera effect when the light from the
    scene is oriented directly toward the camera. 
    
    Choose a lens texture from the Assets directory (free textures provided in Assets directory 
    of this project). All textures are sizes 5184x3456 and would have to be re-sized beforehand.
    
    The setting `light_` is a float values cap between -1.0 to 0.2 and allow you to increase the 
    light source oriented toward the camera. Values <0.0 will decrease the lens dirt 
    effect and values >0.0 will increase the brightness of the display and increase the 
    amount of dirt on the camera lens (your display).
    
    Optionally the setting flag can be changed from BLEND_RGB_ADD to any other pygame optional 
    flags value. BLEND_RGB_ADD is the default setting and allow the pixels from the dirt lens 
    texture to be blended (added) to the display.
    
    e.g:
    dirt_lens(image, flag_=BLEND_RGB_ADD, lens_model_=lens, light_=VALUE)
    
    :param surface_: 
        Surface 24 - 32 bit represent the surface or the display 
    
    :param lens_model_: 
        Surface The Lens model is a pygame Surface. PygameShader provide 6 
        different surfaces that can be used as a layer to generate a dirt lens effect on your game 
        display. See below for the name of the free dirt lens textures. 
     
        Assets/Bokeh__Lens_Dirt_9.jpg
        Assets/Bokeh__Lens_Dirt_38.jpg
        Assets/Bokeh__Lens_Dirt_46.jpg
        Assets/Bokeh__Lens_Dirt_50.jpg
        Assets/Bokeh__Lens_Dirt_54.jpg
        Assets/Bokeh__Lens_Dirt_67.jpg
     
        The texture has to be loaded prior calling this effect and passed as an argument. By default 
        the textures sizes are 5184x3456 (w & h). The texture(s) have to be re-scale once to the game 
        display dimensions (e.g 1027x768)
     
    :param flag_: 
        integer; pygame flags such as BLEND_RGB_ADD, BLEND_RGB_MAX etc. These flags 
        will change the overall appearance of the effect. BLEND_RGB_ADD is the default flag and blend 
        together the dirt_lens image and the game display.
    
    :param light_: 
        float; Float value cap between [-1.0 ... 0.2] to increase or decrease 
        the overall brightness of the dirt lens texture. This setting can be used to simulate a 
        texture transition when sweeping the values from -1.0 toward 0.2 by a small increment.
        Values < 0 will tend to diminish the effect and values > 0 will increase the brightness 
        and the dirt lens effect. 
     
    :return: 
        void; Inplace transformation.
    
    """

    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)

    assert PyObject_IsInstance(lens_model_, pygame.Surface), \
        "\nArgument lens_model_ must be a pygame.Surface type, got %s " % type(lens_model_)

    assert PyObject_IsInstance(light_, float), \
        "\nArgument light_ must be a float type, got %s " % type(light_)

    if light_ > 0.2:
        light_ = 0.2

    elif light_ < -1.0:
        light_ = -1.0

    assert PyObject_IsInstance(flag_, int), \
        "\nArgument flag_ must be a int type, got %s " % type(flag_)

    dirt_lens_c(surface_, lens_model_, flag_, light_)


# *******************************************************************

@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef inline void end_game(object surface):
    """
    
    :param surface: 
    :return: 
    """
    raise NotImplementedError

@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef inline void level_clear(object surface):
    """
    
    :param surface: 
    :return: 
    """
    raise NotImplementedError



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef object dithering(object surface_):

    """
    Dithering Floyd Steinberg (copy)
    
    Dithering is used in computer graphics to create the illusion of "color depth" in images with
    a limited color palette - a technique also known as color quantization. In a dithered image,
    colors that are not available in the palette are approximated by a diffusion of colored pixels
    from within the available palette. The human eye perceives the diffusion as a mixture of 
    the colors within it (see color vision). Dithered images, particularly those with relatively
    few colors, can often be distinguished by a characteristic graininess or speckled appearance
    
    Take a pygame surface as argument format 24-32 bit and convert it to a 3d array format 
    (w, h, 3) type float (float32, single precision). 
    As the image is converted to a different data type format (uint8 to float32), 
    the transformation cannot be applied inplace. The image returned by the method dithering 
    is a copy of the original image without the alpha channel   
    
    Compatible with 24 - 32 bit surface. 
    The output image is 24-bit without the alpha channel 
    
    e.g:
    # for 24 - 32 bit 
    image = dithering(image)
    
    :param surface_: Pygame surface format 24-32 bit 
    :return        : 24-bit surface without the alpha channel  
    
    """

    assert PyObject_IsInstance(surface_, Surface), \
        '\nArgument surface_ must be a pygame.Surface got %s ' % type(surface_)

    cdef:
        # np.ndarray[np.float32_t, ndim=3] bgr_array
        np.ndarray[ np.uint8_t, ndim=3 ] rgb_array

    try:
        rgb_array = pixels3d(surface_)
        # Create a 3d array shape (w, h, 4)
        # bgr_array = numpy.asarray(
        #    surface_.get_view('0'), dtype = numpy.float32).reshape(surface_.get_width(), surface_.get_height(), 4)

    except Exception as e:
        raise ValueError("\nCannot reference source pixels into a 3d array.\n %s " % e)

    # return dithering_c(bgr_array/<float>255.0)
    return dithering_c(numpy.asarray(rgb_array/<float>255.0, dtype=numpy.float32))




@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef inline void dithering_inplace(object surface_):
    """
    Dithering Floyd Steinberg (inplace)
     
    Dithering is used in computer graphics to create the illusion of "color depth" in images with
    a limited color palette - a technique also known as color quantization. In a dithered image,
    colors that are not available in the palette are approximated by a diffusion of colored pixels
    from within the available palette. The human eye perceives the diffusion as a mixture of 
    the colors within it (see color vision). Dithered images, particularly those with relatively
    few colors, can often be distinguished by a characteristic graininess or speckled appearance

    Take a pygame surface as argument format 24-32 bit and convert it to a 3d array format 
    (w, h, 3) type float (float32, single precision). 
    As the image is converted to a different data type format (uint8 to float32), 
    the transformation cannot be applied inplace. The image returned by the method dithering 
    is a copy of the original image.   
    
    e.g:
    # for 24 - 32 bit 
    dithering_inplace(image)

    :param surface_: 
        Pygame surface format 24-32 bit
         
    :return        : 
        Output surface is the same format than input 24 or 32 bit
         

    """

    # Ensure that the input argument surface_ is a valid pygame.Surface object
    assert PyObject_IsInstance(surface_, Surface), \
        'Argument surface_ must be a pygame.Surface got %s ' % type(surface_)

    # Declare the variable rgb_array to reference the pixel data of the surface
    cdef unsigned char [:, :, :] rgb_array

    try:
        # Try to get a reference to the pixel data of the surface as a 3D array
        rgb_array = surface_.get_view('3')

    except Exception as e:
        # If an error occurs while referencing the pixels, raise a ValueError
        raise ValueError("\nCannot reference source pixels into a 3d array.\n %s " % e)

    # Convert the rgb_array data to a numpy float32 array (scaled between 0 and 1),
    # then call the dithering function to apply dithering to the image in-place
    dithering_inplace_c(numpy.asarray(rgb_array, dtype=numpy.float32)/<float>255.0, rgb_array)




@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef inline void dithering1d(
        Py_ssize_t w,
        Py_ssize_t h,
        unsigned char [::1] bgr_array,
        bint format_32=False
):
    """
    Dithering Floyd Steinberg (inplace)
    
    Dithering is used in computer graphics to create the illusion of "color depth" in images with
    a limited color palette - a technique also known as color quantization. In a dithered image,
    colors that are not available in the palette are approximated by a diffusion of colored pixels
    from within the available palette. The human eye perceives the diffusion as a mixture of 
    the colors within it (see color vision). Dithered images, particularly those with relatively
    few colors, can often be distinguished by a characteristic graininess or speckled appearance.
    
    Compatible with 24-32 bit image. If the dithering process is not applied to the image, 
    check the flag format_32. 
    format_32 should be set to True for image containing per pixel transparency or 
    equivalent array shape (w, h, 4). 
    For 24-bit image, set format_32 to False (array shape (w, h, 3))
    
    # for 32 bit image
    dithering1d(w, h, im.get_buffer(), True)
    
    # for 24 bit image
    dithering1d(w, h, im.get_buffer(), False)
    
    
    Parameters
    ----------
    w : 
        integer; width of the array
          
    h : 
        integer; height of the array
         
    bgr_array : 
        numpy.ndarray shape (w, h, 3|4) containing BGR pixels or any other pixel format
        
    format_32 : 
        bool True | 'BGR' for 24-bit array (BGR) or False | 'BGRA' for 32-bit array (BGRA)
         

    Returns
    -------
    void
    
    """

    # Declare the variable 'l' as the length of the first dimension of the bgr_array
    # This represents the width or number of pixels in a single row of the image
    cdef:
        Py_ssize_t l = bgr_array.shape[0]

        # Create a temporary array 'tmp_array' that holds the normalized pixel values (0.0 to 1.0)
        # We convert the original 'bgr_array' to a numpy ndarray, divide each pixel value by 255,
        # and then cast the resulting values into a numpy float32 array.
        float [::1] tmp_array = numpy.asarray(numpy.ndarray(
            shape=l, buffer=bgr_array, dtype=uint8)/<float>255.0, dtype=numpy.float32)

    # Call the dithering function with the width (w), height (h), row length (l), 
    # the original bgr_array, the temporary normalized tmp_array, and the format_32 flag
    dithering1d_c(w, h, l, bgr_array, tmp_array, format_32)






@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef inline void dithering1d_c(
        const Py_ssize_t w,
        const Py_ssize_t h,
        const Py_ssize_t l,
        unsigned char [::1] bgr_array,
        float [::1] tmp_array,
        bint format_32=False
)nogil:
    """
    Dithering Floyd Steinberg (inplace)
    
    Dithering is used in computer graphics to create the illusion of "color depth" in images with
    a limited color palette - a technique also known as color quantization. In a dithered image,
    colors that are not available in the palette are approximated by a diffusion of colored pixels
    from within the available palette. The human eye perceives the diffusion as a mixture of 
    the colors within it (see color vision). Dithered images, particularly those with relatively
    few colors, can often be distinguished by a characteristic graininess or speckled appearance.
    
    Compatible with 24-32 bit image. If the dithering process is not applied to the image, 
    check the flag format_32. 
    format_32 should be set to True for image containing per pixel transparency or 
    equivalent array shape (w, h, 4). 
    For 24-bit image, set format_32 to False (array shape (w, h, 3))
    
    # for 32 bit image
    dithering1d_c(w, h, im.get_buffer(), tmp_array, True)
    
    # for 24 bit image
    dithering1d_c(w, h, im.get_buffer(), tmp_array, False)
    
    Parameters
    ----------
    w : 
        integer; width of the array
          
    h : 
        integer; height of the array
         
    l : 
        integer; length of the array equivalent to w * h * bytesize
        
    bgr_array :
        numpy.ndarray shape (w, h, 3|4) of type uint8 containing BGR pixels or any other pixel format
        
    tmp_array :  
        numpy.ndarray shape (w, h, 3|4) type float32 containing normalized RGB pixels, 
        copy of the input array `bgr_array`
        
    format_32 : 
        bool True | 'BGR' for 24-bit array (BGR) or False | 'BGRA' for 32-bit array (BGRA)
         
    Returns
    -------
    void
    
    """


    if w * h == 0:
        raise ValueError('\nExpecting w and h non null! got w:%s h:%s' % (w, h))

    cdef int byte_size = l / (w * h)

    if format_32:
        if byte_size != 4:
            raise ValueError(
                "\nIs format_32 set correctly?\n"
                " bytesize value is %s and format_32 should be set to %s"
                % (byte_size, True if byte_size==4 else False))

    cdef:
        float new_red, new_green, new_blue
        float quantization_error_red, quantization_error_green, quantization_error_blue
        float oldr, oldg, oldb
        float * r1
        float * g1
        float * b1

    cdef:
        int i = 0
        unsigned int wb = w * byte_size

    # cannot use range(0, l, byte_size) as byte_size is not accepted as a cython value
    # The only way is to force the num_threads to be equal to 1 (1 active thread only), equivalent
    # to range.
    for i in prange(0, l, byte_size, schedule=SCHEDULE, num_threads=1):

        # note: skip transparent pixel
        if format_32 and byte_size==4:
            if bgr_array[ i + 3 ] == 0:
                continue

        oldr = tmp_array[ i ]
        oldg = tmp_array[ i + 1]
        oldb = tmp_array[ i + 2]

        new_red   = round_c(oldr)
        new_green = round_c(oldg)
        new_blue  = round_c(oldb)

        tmp_array[ i  ] = new_red
        tmp_array[ i + 1] = new_green
        tmp_array[ i + 2] = new_blue

        quantization_error_red   = <float>(oldr - new_red)
        quantization_error_green = <float>(oldg - new_green)
        quantization_error_blue  = <float>(oldb - new_blue)

        if i + 2 + byte_size < l:
            r1 = &tmp_array[i + byte_size]
            r1[0] += quantization_error_red * C1
            g1 = &tmp_array[i + 1 + byte_size]
            g1[0] +=quantization_error_green * C1
            b1 = &tmp_array[i + 2 + byte_size]
            b1[0] += quantization_error_blue * C1

        if i - byte_size + wb + 2 < l:
            r1 = &tmp_array[i - byte_size + wb]
            r1[0] += quantization_error_red * C2
            g1 = &tmp_array[i - byte_size + wb + 1]
            g1[0] += quantization_error_green * C2
            b1 = &tmp_array[i - byte_size + wb + 2]
            b1[0] += quantization_error_blue * C2

        if i + 2 + wb < l:
            r1 = &tmp_array[i + wb]
            r1[0] +=  quantization_error_red * C3
            g1 = &tmp_array[i + 1 + wb]
            g1[0] +=  quantization_error_green * C3
            b1 = &tmp_array[i + 2 + wb]
            b1[0] +=  quantization_error_blue * C3

        if i + byte_size + wb + 2 < l:
            r1 = &tmp_array[i + byte_size + wb]
            r1[0] += quantization_error_red * C4
            g1 = &tmp_array[i + byte_size + wb + 1]
            g1[0] += quantization_error_green * C4
            b1 = &tmp_array[i + byte_size + wb + 2]
            b1[0] += quantization_error_blue * C4

    # OK to run prange here
    for i in prange(0, l, byte_size, schedule=SCHEDULE, num_threads=THREADS):
        # no need to set alpha as the change is applied inplace
        bgr_array[i    ] = <unsigned char>(tmp_array[ i ] * <float>255.0)
        bgr_array[i + 1] = <unsigned char>(tmp_array[ i + 1] * <float>255.0)
        bgr_array[i + 2] = <unsigned char>(tmp_array[ i + 2] * <float>255.0)



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef inline dithering1d_cp(
        Py_ssize_t w,
        Py_ssize_t h,
        rgb_array,
        bint format_32 = False
):
    """
    
    Dithering Floyd Steinberg (copy)
    
    Dithering is used in computer graphics to create the illusion of "color depth" in images with
    a limited color palette - a technique also known as color quantization. In a dithered image,
    colors that are not available in the palette are approximated by a diffusion of colored pixels
    from within the available palette. The human eye perceives the diffusion as a mixture of 
    the colors within it (see color vision). Dithered images, particularly those with relatively
    few colors, can often be distinguished by a characteristic graininess or speckled appearance.
    
    Compatible with 24-32 bit image. If the dithering process is not applied to the image, 
    check the flag format_32. 
    format_32 should be set to True for image containing per pixel transparency or 
    equivalent array shape (w, h, 4). 
    For 24-bit image, set format_32 to False (array shape (w, h, 3))
    
    
    # for 24-bit image
    buff = pygame.image.tobytes(im, "RGB")
    arr = dithering1d_cp(w, h, buff, False)
    im = pygame.image.frombuffer(arr, (w, h), "RGB")

    # for 32-bit image
    buff = pygame.image.tobytes(im, "RGBA")
    arr = dithering1d_cp(w, h, buff, True)
    im = pygame.image.frombuffer(arr, (w, h), "RGBA")
    
    Parameters
    ----------
    w : 
        integer; width of the array
        
    h : 
        integer; height of the array
        
    rgb_array : 
        numpy.ndarray shape (w, h, 3|4) containing RGB pixels or any other pixel format
        
    format_32 : 
        bool True | 'RGB' for 24-bit array (RGB) or False | 'RGBA' for 32-bit array (RGBA)

    Returns
    -------
    copy of the input array with dithering effect applied to the array
    
    """

    # Declare the variable 'l' to represent the length of the 'rgb_array'
    # This is used to determine the number of pixels in the array (in one dimension)
    cdef:
        Py_ssize_t l = len(rgb_array)

        # Create a temporary array 'tmp_array' of type unsigned char and size 'l'.
        # This array is initialized using numpy.empty to hold the modified pixel values during the dithering process.
        unsigned char[::1] tmp_array = numpy.empty(l, uint8)

    # Call the dithering function 'dithering1d_cp_c' with the necessary parameters:
    # - 'w' (width) and 'h' (height) for the image dimensions
    # - 'l' for the length (number of pixels in one row)
    # - The normalized pixel values from 'rgb_array' (converted to float32 and divided by 255)
    # - The temporary array 'tmp_array' that will hold the processed pixel values
    # - 'format_32' to specify the format of the output (likely related to bit depth or processing)
    return dithering1d_cp_c(
        w, h, l,
        numpy.asarray(numpy.frombuffer(rgb_array, dtype=numpy.uint8)/float(255.0),
            dtype=numpy.float32), tmp_array, format_32)




@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef dithering1d_cp_c(
        const Py_ssize_t w,
        const Py_ssize_t h,
        const Py_ssize_t l,
        float [::1] bgr_array,
        unsigned char [::1] tmp_array,
        bint format_32=False):
    """
    
    Dithering Floyd Steinberg (copy)
    
    Dithering is used in computer graphics to create the illusion of "color depth" in images with
    a limited color palette - a technique also known as color quantization. In a dithered image,
    colors that are not available in the palette are approximated by a diffusion of colored pixels
    from within the available palette. The human eye perceives the diffusion as a mixture of 
    the colors within it (see color vision). Dithered images, particularly those with relatively
    few colors, can often be distinguished by a characteristic graininess or speckled appearance.
    
    Compatible with 24-32 bit image. If the dithering process is not applied to the image, 
    check the flag format_32. 
    format_32 should be set to True for image containing per pixel transparency or 
    equivalent array shape (w, h, 4). 
    For 24-bit image, set format_32 to False (array shape (w, h, 3))
    
    
    # for 24-bit image
    buff = pygame.image.tobytes(im, "RGB")
    arr = dithering1d_cp(w, h, buff, tmp_array, False)
    im = pygame.image.frombuffer(arr, (w, h), "RGB")

    # for 32-bit image
    buff = pygame.image.tobytes(im, "RGBA")
    arr = dithering1d_cp(w, h, buff, tmp_array, True)
    im = pygame.image.frombuffer(arr, (w, h), "RGBA")
    
    Parameters
    ----------
    w : 
        integer; width of the array
        
    h : 
        integer; height of the array
        
    l : 
        integer; length of the array
        
    bgr_array : 
        numpy.ndarray shape (w, h, 3|4) type float32, containing normalized BGR pixels range [0.0 ... 1.0] 
        or any other pixel format
        
    tmp_array :  
        numpy.ndarray shape (w, h, 3|4) type uint8 containing BGR pixels, 
        copy of the input array `bgr_array`
        
    format_32 : 
        bool True | 'BGR' for 24-bit array (BGR) or False | 'BGRA' for 32-bit array (BGRA)

    Returns
    -------
    memoryviewslice shape (w, ) type uint8 RGB or RGBA format (depends on the input array)
    copy of the input array with dithering effect applied to the array
    
    """

    if w * h == 0:
        raise ValueError('\nExpecting w and h non null! got w:%s h:%s' % (w, h))

    cdef int byte_size = l / (w * h)

    if format_32:
        if byte_size != 4:
            raise ValueError(
                "\nIs format_32 set correctly?\n"
                " bytesize value is %s and format_32 should be set to %s"
                % (byte_size, True if byte_size == 4 else False))

    cdef:
        float new_red, new_green, new_blue
        float quantization_error_red, quantization_error_green, quantization_error_blue
        float oldr, oldg, oldb
        float * r1
        float * g1
        float * b1

    cdef:
        int i = 0
        unsigned int wb = w * byte_size


    with nogil:

        for i in prange(0, l, byte_size, schedule=SCHEDULE, num_threads=1):

            # note: skip transparent pixel
            if format_32:
                if bgr_array[ i + 3] == 0:
                    continue

            oldr = bgr_array[ i ]
            oldg = bgr_array[ i + 1]
            oldb = bgr_array[ i + 2]

            new_red   = round_c(oldr)
            new_green = round_c(oldg)
            new_blue  = round_c(oldb)

            bgr_array[ i  ] = new_red
            bgr_array[ i + 1] = new_green
            bgr_array[ i + 2] = new_blue

            quantization_error_red   = <float>(oldr - new_red)
            quantization_error_green = <float>(oldg - new_green)
            quantization_error_blue  = <float>(oldb - new_blue)


            if i + 2 + byte_size < l:
                r1 = &bgr_array[i + byte_size]
                r1[0] = r1[0] + quantization_error_red * C1
                g1 = &bgr_array[i + 1 + byte_size]
                g1[0] = g1[0] + quantization_error_green * C1
                b1 = &bgr_array[i + 2 + byte_size]
                b1[0] = b1[0] + quantization_error_blue * C1


            if i - byte_size + wb + 2 < l:
                r1 = &bgr_array[i - byte_size + wb]
                r1[0] = r1[0] + quantization_error_red * C2
                g1 = &bgr_array[i - byte_size + wb + 1]
                g1[0] = g1[0] + quantization_error_green * C2
                b1 = &bgr_array[i - byte_size + wb + 2]
                b1[0] = b1[0] + quantization_error_blue * C2


            if i + 2 + wb < l:
                r1 = &bgr_array[i + wb]
                r1[0] = r1[0] + quantization_error_red * C3
                g1 = &bgr_array[i + 1 + wb]
                g1[0] = g1[0] + quantization_error_green * C3
                b1 = &bgr_array[i + 2 + wb]
                b1[0] = b1[0] + quantization_error_blue * C3


            if i + byte_size + wb + 2 < l:
                r1 = &bgr_array[i + byte_size + wb]
                r1[0] = r1[0] + quantization_error_red * C4
                g1 = &bgr_array[i + byte_size + wb + 1]
                g1[0] = g1[0] + quantization_error_green * C4
                b1 = &bgr_array[i + byte_size + wb + 2]
                b1[0] = b1[0] + quantization_error_blue * C4

        for i in prange(0, l, byte_size, schedule=SCHEDULE, num_threads=THREADS):
            tmp_array[i    ] = <unsigned char>(bgr_array[ i ] * <float>255.0)
            tmp_array[i + 1] = <unsigned char>(bgr_array[ i + 1] * <float>255.0)
            tmp_array[i + 2] = <unsigned char>(bgr_array[ i + 2] * <float>255.0)
            if format_32:
                tmp_array[ i + 3 ] = <unsigned char> (bgr_array[ i + 3 ] * <float> 255.0)

    return tmp_array



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef object dithering_atkinson(object surface_):
    """
    
    Dithering atkinson (copy)
    
    Atkinson dithering is a variant of Floyd–Steinberg dithering designed by 
    Bill Atkinson at Apple Computer, and used in the original Macintosh computer. 
    
    Dithering is used in computer graphics to create the illusion of "color depth" in images with
    a limited color palette - a technique also known as color quantization. In a dithered image,
    colors that are not available in the palette are approximated by a diffusion of colored pixels
    from within the available palette. The human eye perceives the diffusion as a mixture of 
    the colors within it (see color vision). Dithered images, particularly those with relatively
    few colors, can often be distinguished by a characteristic graininess or speckled appearance.
    
    Take a pygame surface as argument format 24-32 bit and convert it to a 3d array format 
    (w, h, 3) type float (float32, single precision). 
    
    As the image is converted to a different data type format (conversion from uint8 to float32), 
    the transformation cannot be applied inplace. 
    
    The image returned is a copy of the original image.   
    
    e.g:
    # for 24, 32-bit image format 
     image = dithering_atkinson(image)

    :param surface_:
        Pygame surface format 24, 32-bit
         
    :return:
        pygame surface format 24-bit 

    """

    # Ensure that the 'surface_' object is an instance of pygame.Surface
    # If the surface is not of the correct type, an error is raised with an informative message
    assert PyObject_IsInstance(surface_, Surface), \
        'Argument surface_ must be a pygame.Surface, got %s ' % type(surface_)

    try:
        # Attempt to extract the pixel data as a 3D array from the 'surface_' using the function 'pixels3d'.
        # This function assumes the surface contains the image data in 3D (e.g., RGB or RGBA format).
        rgb_array = pixels3d(surface_)

    except Exception as e:
        # If the extraction of pixel data fails, raise an error indicating the failure
        # and provide details about the exception that caused the issue.
        raise ValueError("\nCannot reference source pixels into a 3d array.\n %s " % e)

    # Normalize the pixel values (dividing by 255) to bring them into the range [0, 1],
    # then cast the array to a float32 type.
    # The resulting normalized array is passed to the 'dithering_atkinson_c' function
    # to apply the Atkinson dithering algorithm.

    return dithering_atkinson_c(numpy.asarray(rgb_array / <float> 255.0, dtype=numpy.float32))





@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef inline void dithering_atkinson1d(
    Py_ssize_t w,
    Py_ssize_t h,
    unsigned char [::1] c_buffer,
    bint format_32=False
):

    """
    
    Atkinson dithering for 1d array (inplace)
    
    Atkinson dithering is a variant of Floyd–Steinberg dithering designed by 
    Bill Atkinson at Apple Computer, and used in the original Macintosh computer. 
    
    Dithering is used in computer graphics to create the illusion of "color depth" in images with
    a limited color palette - a technique also known as color quantization. In a dithered image,
    colors that are not available in the palette are approximated by a diffusion of colored pixels
    from within the available palette. The human eye perceives the diffusion as a mixture of 
    the colors within it (see color vision). Dithered images, particularly those with relatively
    few colors, can often be distinguished by a characteristic graininess or speckled appearance.
    
    Compatible with 24-32 bit image. If the dithering process is not applied to the image, 
    check the flag format_32. 
    format_32 should be set to True for image containing per pixel transparency or 
    equivalent array shape (w, h, 4). 
    For 24-bit image, set format_32 to False (array shape (w, h, 3))
    
    # for 32 bit image
    dithering_atkinson1d(w, h, im.get_buffer(), True)
    
    # for 24 bit image
    dithering_atkinson1d(w, h, im.get_buffer(), False)
    
    Parameters
    ----------
    w : 
        integer; width of the array
          
    h : 
        integer; height of the array
         
    c_buffer : 
        C-Buffer or memoryviewslice or 1d numpy.ndarray containing BGR pixels or any other pixel format. 
        datatype uint8.
        
        
    format_32 : 
        bool; True for 'RGB' 24-bit array (BGR) or False for 32-bit array (BGRA)
         
    Returns
    -------
    void

    """

    # Define local variables within the 'cdef' block:
    cdef:
        # 'l' is the length of the 'c_buffer', representing the number of elements in the buffer.
        Py_ssize_t l = len(c_buffer)

        # 'tmp_buffer' is an array created by converting the 'c_buffer' into a 1D NumPy array of 'uint8' type.
        # Then, it's normalized by dividing by 255.0 to scale the pixel values to the range [0, 1].
        # The resulting array is cast to 'float32' type for further processing.
        float [ ::1 ] tmp_buffer = \
            numpy.asarray(
                numpy.ndarray(shape=l, buffer=c_buffer, dtype=uint8)/<float>255.0,  # Normalize to [0,1] by dividing by 255
                dtype=numpy.float32  # Convert the result to float32 for further processing
            )

    # Apply the Atkinson dithering algorithm to the image data using the 'dithering_atkinson1d_c' function.
    # Parameters passed include width ('w'), height ('h'), length ('l') of the buffer, the 'c_buffer' itself,
    # the temporary 'tmp_buffer' used for processing, and a format flag ('format_32').
    dithering_atkinson1d_c(w, h, l, c_buffer, tmp_buffer, format_32)






@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef inline void dithering_atkinson1d_c(
        const Py_ssize_t w,
        const Py_ssize_t h,
        const Py_ssize_t l,
        unsigned char [::1] c_buffer,
        float [::1] tmp_buffer,
        bint format_32=False
)nogil:

    """
    
    Atkinson dithering for 1d array (inplace)
    
    Atkinson dithering is a variant of Floyd–Steinberg dithering designed by 
    Bill Atkinson at Apple Computer, and used in the original Macintosh computer. 
    
    Dithering is used in computer graphics to create the illusion of "color depth" in images with
    a limited color palette - a technique also known as color quantization. In a dithered image,
    colors that are not available in the palette are approximated by a diffusion of colored pixels
    from within the available palette. The human eye perceives the diffusion as a mixture of 
    the colors within it (see color vision). Dithered images, particularly those with relatively
    few colors, can often be distinguished by a characteristic graininess or speckled appearance.
    
    
    # for 32 bit image
    dithering_atkinson1d_c(w, h, im.get_buffer(), tmp_buffer, True)
    
    # for 24 bit image
    dithering_atkinson1d_c(w, h, im.get_buffer(), tmp_buffer, False)
    
    Parameters
    ----------
    w : 
        integer; width of the array
          
    h : 
        integer; height of the array
         
    l : 
        integer; length of the array equivalent to w * h * bytesize
        
    c_buffer : 
        C buffer, memoryviewslice or 1d numpy.ndarray of type uint8 
        containing BGR pixels or any other pixel format
        
    tmp_buffer : 
        C buffer, memoryviewslice or 1d numpy.ndarray of type float32 
        containing BGR pixels or any other pixel format, copy of the input array with 
        normalized pixels.
        
    format_32 : 
        bool; True for 24-bit array (BGR) or False for 32-bit array (BGRA)
         
    Returns
    -------
    void

    """


    if w * h == 0:
        with gil:
            raise ValueError('\nExpecting w and h non null! got w:%s h:%s' % (w, h))

    cdef int byte_size = l / (w * h)

    if format_32:
        if byte_size != 4:
            with gil:
                raise ValueError(
                    "\nIs format_32 set correctly?\n"
                    " bytesize value is %s and format_32 should be set to %s"
                    % (byte_size, True if byte_size==4 else False))

    cdef:
        float new_red, new_green, new_blue
        float quantization_error_red, quantization_error_green, quantization_error_blue
        float oldr, oldg, oldb

        float * ptr


    cdef:
        int i = 0
        unsigned int wb = w * byte_size

    # cannot use range(0, l, byte_size) as byte_size is not accepted as a cython value
    # The only way is to force the num_threads to be equal to 1 (1 active thread only), equivalent
    # to range.
    for i in prange(0, l, byte_size, schedule=SCHEDULE, num_threads=1):

        # ptr = &tmp_buffer[ i ]

        # note: skip transparent pixel
        if format_32 and byte_size==4:
            if c_buffer[ i + 3 ] == 0:
                continue

        oldr = tmp_buffer[ i ]
        oldg = tmp_buffer[ i + 1 ]
        oldb = tmp_buffer[ i + 2 ]

        new_red   = round_c(oldr)
        new_green = round_c(oldg)
        new_blue  = round_c(oldb)

        tmp_buffer[ i ] = new_red
        tmp_buffer[ i + 1 ] = new_green
        tmp_buffer[ i + 2 ] = new_blue

        quantization_error_red   = <float>(oldr - new_red) * <float>0.125
        quantization_error_green = <float>(oldg - new_green) * <float>0.125
        quantization_error_blue  = <float>(oldb - new_blue) * <float>0.125

        ptr = &tmp_buffer[ i ]

        if i + 2 + byte_size < l:

            (ptr + byte_size)[0] += quantization_error_red
            (ptr + 1 + byte_size)[ 0 ] += quantization_error_green
            (ptr + 2 + byte_size)[ 0 ] += quantization_error_blue


        if i + 2 + 2 * byte_size < l:

            (ptr + 2 * byte_size)[ 0 ] += quantization_error_red
            (ptr + 1 + 2 * byte_size)[ 0 ] += quantization_error_green
            (ptr + 2 + 2 * byte_size)[ 0 ] += quantization_error_blue

        if i - byte_size + wb + 2 < l:

            (ptr - byte_size + wb)[ 0 ] += quantization_error_red
            (ptr - byte_size + wb + 1)[ 0 ] += quantization_error_green
            (ptr + byte_size + wb + 2)[ 0 ] += quantization_error_blue

        if i + 2 + wb < l:

            (ptr + wb)[ 0 ] += quantization_error_red
            (ptr + wb + 1)[ 0 ] += quantization_error_green
            (ptr + wb + 2)[ 0 ] += quantization_error_blue

        if i + byte_size + wb + 2 < l:

            (ptr + byte_size + wb)[ 0 ] += quantization_error_red
            (ptr + byte_size + wb + 1)[ 0 ] += quantization_error_green
            (ptr + byte_size + wb + 2)[ 0 ] += quantization_error_blue

        if i + 2 * wb + 2 < l:

            (ptr + 2 * wb)[ 0 ] += quantization_error_red
            (ptr + 2 * wb + 1)[ 0 ] += quantization_error_green
            (ptr + 2 * wb + 2)[ 0 ] += quantization_error_blue

    # OK to run prange here
    for i in prange(0, l, byte_size, schedule=SCHEDULE, num_threads=THREADS):
        # no need to set alpha as the change is applied inplace
        c_buffer[i    ] = <unsigned char>(tmp_buffer[ i ] * <float>255.0)
        c_buffer[i + 1] = <unsigned char>(tmp_buffer[ i + 1] * <float>255.0)
        c_buffer[i + 2] = <unsigned char>(tmp_buffer[ i + 2] * <float>255.0)


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef object pixelation(object surface_, unsigned int blocksize_=64):
    """
    
    Pixelate a pygame.Surface 
    
    In computer graphics, pixelation (also spelled pixellation in British English) 
    is caused by displaying a bitmap or a section of a bitmap at such a large size 
    that individual pixels, small single-colored square display elements that comprise
    the bitmap, are visible. Such an image is said to be pixelated (pixellated in the UK). 
    
    Return a new pixelated surface
    Blocksize represent the square pixel size (default is 64, 64x64 pixel block).
    
    e.g:
    # Compatible with 24, 32-bit images
    pix_image = pixelation(image)
    
    :param surface_: 
        pygame.Surface;
         
    :param blocksize_: 
        unsigned int; block size used for the pixelation process, default is 64
        
    :return: 
        pixelated surface
    """

    # Ensure that the 'surface_' is a valid pygame.Surface object.
    # If it's not, raise an error with a message that includes the type of the provided object.
    assert PyObject_IsInstance(surface_, Surface), \
        'Argument surface_ must be a pygame.Surface got %s ' % type(surface_)

    # Check if the 'blocksize_' is greater than 4.
    # If it's not, raise an error, as block size must be > 4.
    assert blocksize_ > 4, 'Invalid block size, blocksize must be > 4  got %s ' % blocksize_

    # Define local variables for the width ('w') and height ('h') of the surface.
    cdef Py_ssize_t w, h
    w, h = surface_.get_size()  # Get the dimensions of the surface (width, height).

    # Create a smaller version of the surface by scaling it down to the specified block size.
    # The 'smoothscale' function smoothly resizes the surface to the given dimensions.
    cdef object small = smoothscale(surface_, (blocksize_, blocksize_))

    # Scale the resized image back to the original surface dimensions ('w', 'h').
    # This effectively applies the block-based scaling.
    return scale(small, (w, h))




@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef object blend(object source, object destination, float percentage):
    """
    Alpha Blending 
    
    Blend two images together.
    
    e.g:
    # compatible 24, 32-bit
    transition = blend(source=image1, destination=image2, percentage =60)
    
    :param source: 
        pygame.Surface (Source), compatible 24, 32-bit 
        
    :param destination: 
        pygame.Surface (Destination), compatible 24, 32-bit
        
    :param percentage: 
        float; Percentage value between [0.0 ... 100.0]
        
    :return: return: 
        Return a new surface (24-bit) blend of both input images.
        
    """

    # Ensure that the 'source' is a valid pygame.Surface object.
    # If not, raise an error with a message that includes the type of the provided object.
    assert PyObject_IsInstance(source, Surface), \
        'Argument source must be a pygame.Surface got %s ' % type(source)

    # Ensure that 'destination' is also a valid pygame.Surface object.
    # If 'destination' is not a Surface, raise an error with a message.
    assert PyObject_IsInstance(destination, Surface), \
        'Argument destination must be a pygame.Surface got %s ' % type(destination)

    # Validate that the 'percentage' argument is within the valid range [0.0, 100.0].
    # If it’s not, raise an error with the out-of-range value.
    assert 0.0 <= percentage <= 100.0, \
        "\nIncorrect value for argument percentage should be [0.0 ... 100.0] got %s " % percentage

    # If percentage is 0.0, simply return the original source image (no blending needed).
    if percentage == 0.0:
        return source

    # Ensure that both the 'source' and 'destination' surfaces have the same dimensions.
    # If they don't match, raise an error with the dimensions of both surfaces.
    assert source.get_size() == destination.get_size(), \
        'Source and Destination surfaces must have same dimensions: ' \
        'Source (w:%s, h:%s), destination (w:%s, h:%s).' % \
        (*source.get_size(), *destination.get_size())

    # Define local variables for the 3D pixel arrays of the 'source' and 'destination' surfaces.
    cdef:
        unsigned char [:, :, :] source_array
        unsigned char[:, :, :] destination_array

    # Attempt to reference the pixel data of the source surface as a 3D array.
    # If an error occurs, raise a ValueError indicating the failure.
    try:
        source_array = source.get_view('3')
    except Exception as e:
        raise ValueError(
            "\nCannot reference source pixels into a 3d array.\n %s " % e)

    # Attempt to reference the pixel data of the destination surface as a 3D array.
    # If an error occurs, raise a ValueError indicating the failure.
    try:
        destination_array = destination.get_view('3')
    except Exception as e:
        raise ValueError(
            "\nCannot reference destination pixels into a 3d array.\n %s " % e)

    # Call the 'blend_c' function to perform the blending operation between the source and destination arrays.
    # The blending operation uses the specified 'percentage' value.
    return blend_c(source_array, destination_array, percentage)




@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef blend1d(
        const Py_ssize_t w,
        const Py_ssize_t h,
        const unsigned char [::1] source,
        const unsigned char [::1] destination,
        float percentage,
        modes,
        bint format_32 = False
):
    """
    Alpha blending 
    
    Blend 2 buffers together (1d array)

    Use this method to blend two surfaces together by providing 2 images buffers as source
    and destination. 
    
    With the argument `percentage`, you can control the % of both images
    for example, if percentage is set to 25%, the source image will be drawn at 25% while the
    destination image will be drawn at 75%.
    
    Argument `modes` can be RGB(X) for buffer format RGB(X) or BGR(X) for buffer pixel type BGR(X).
    
    Set `format_32` to True if the source and destination arrays contains per-pixel transparency (alpha values)

    NOTE:
    Both source & destination buffers must have the same length, 
    same data type (uint8) and same pixel format.
    Percentage must be in range [0...100]
    modes must be RGB(X) or BGR(X)

    # for 32 bit images.
    im = blend1d(w, h, im.get_buffer(), BCK.get_buffer(), 25, 'BGR(X)', True)

    # for 24 bit images
    im = blend1d(w, h, im.get_buffer(), BCK.get_buffer(), 25, 'BGR(X)', False)


    Parameters
    ----------
    w: 
        integer; with of the source array
        
    h: 
        integer; height of the source array
        
    source: 
        numpy.ndarray; shape (w, ) of type uint8 containing 
        RGB(A) or BGR(A) pixel format
        
    destination: 
        numpy.ndarray; numpy.ndarray; shape (w, ) of type uint8 
        containing pixels with format identical to the source array
        
    percentage: 
        float; blending value 0 to 100%
        
    modes: 
        str; can be RGB(X) or BGR(X). Use RGB(X) if the source array pixel
        format is equivalent to RGB or RGBA, otherwise select BGR(X)
    
    format_32: 
        bool; default is False. Select True if the source array 
        contains alpha transparency.

    Returns
    -------
    Returns a pygame.Surface with the blending effect.
    The final output image can be 24-32 bit format and depends on 
    the source & destination buffers shapes


    """

    # Validate that the 'percentage' argument is within the valid range [0.0, 100.0].
    # If it's not, raise an error with the out-of-range value.
    assert 0.0 <= percentage <= 100.0, \
        "\nIncorrect value for argument percentage should be [0.0 ... 100.0] got %s " % percentage

    # Define local variables for the length of 'source' and calculate the byte size for the image data.
    # The byte size is determined by dividing the length of the source by the product of image width (w) and height (h).
    cdef:
        Py_ssize_t l = len(source)   # Length of the source array (number of elements).
        int byte_size = l / (w * h)  # Determine the byte size (channels per pixel).

    # If the percentage is 0.0, simply return the destination as it is, using the appropriate color format.
    # 'RGBA' is used if the byte size is 4 (meaning the source has alpha), otherwise 'RGB' is used.
    if percentage == 0.0:
        return frombuffer(destination, (w, h), 'RGBA' if byte_size == 4 else 'RGB')

    # Ensure that the 'source' and 'destination' arrays have the same length.
    # If they don't, raise an error with the mismatched lengths.
    assert l == len(destination), \
        'Source and Destination arrays must have same dimensions: ' \
        'Source (w,) %s, destination (w, ) %s).' % (l, len(destination))

    # Call the 'blend1d_c' function to perform the blending operation.
    # The blending operation uses the specified 'percentage' value, image width (w), height (h),
    # and the byte size format (RGBA or RGB), as well as additional blending modes.
    return blend1d_c(w, h, l, source, destination, percentage, modes, format_32)



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef blend1d_c(
        const Py_ssize_t w,
        const Py_ssize_t h,
        const Py_ssize_t l,
        const unsigned char[::1] source_array,
        const unsigned char[::1] destination_array,
        float percentage,
        modes,
        bint format_32 = False):

    """
    Alpha blending 
    
    Blend 2 arrays together (1d array)

    Use this method to blend two arrays together by providing 2 buffers as source
    and destination. 
    
    With the argument `percentage`, you can control the % of both images
    for example, if percentage is set to 25%, the source image will be drawn at 25% while the
    destination image will be drawn at 75%.
    
    Argument `modes` can be RGB(X) for buffer format RGB(X) or BGR(X) for buffer pixel type BGR(X).
    
    Set `format_32` to True if the source and destination arrays contains per-pixel transparency (alpha values)

    NOTE:
    Both source & destination buffers must have the same length, 
    same data type (uint8) and same pixel format.
    Percentage must be in range [0...100]
    modes must be RGB(X) or BGR(X)

    # for 32 bit images.
    im = blend1d_c(w, h, im.get_buffer(), BCK.get_buffer(), 25, 'BGR(X)', True)

    # for 24 bit images
    im = blend1d_c(w, h, im.get_buffer(), BCK.get_buffer(), 25, 'BGR(X)', False)


    Parameters
    ----------
    w : 
        integer; with of the source array
        
    h : 
        integer; height of the source array
        
    l : 
        integer; length of the array
        
    source_array :
        numpy.ndarray; shape (w, ) of type uint8 containing RGB(A) or BGR(A) pixel format
        
    destination_array : 
        numpy.ndarray; numpy.ndarray; shape (w, ) of type uint8 containing pixels with format identical to
        the source array
        
    percentage : 
        float; blending value 0 to 100%
        
    modes : 
        str; can be RGB(X) or BGR(X). Use RGB(X) if the source array pixel format is equivalent to RGB or RGBA,
        otherwise select BGR(X)
        
    format_32 :
        bool; default is False. Select True if the source array contains alpha transparency.

    Returns
    -------
    Returns a pygame.Surface with the blending effect.
    The final output image can be 24-32 bit format and depends 
    on the source & destination buffers.


    """

    if w * h == 0:
        raise ValueError('\nExpecting w and h non null! got w:%s h:%s' % (w, h))

    cdef int byte_size = l / (w * h)

    if format_32:
        if byte_size != 4:
            raise ValueError(
                "\nIs format_32 set correctly?\n"
                " bytesize value is %s and format_32 should be set to %s"
                % (byte_size, True if byte_size==4 else False))

    cdef:
        unsigned char[ ::1 ] tmp_array = empty(l, dtype = uint8)
        int j=0
        float c4 = percentage * <float>0.01
        float tmp = <float> 1.0 - c4
        unsigned char * f_array
        const unsigned char * dst_array
        const unsigned char * src_array


    with nogil:
        for j in prange(0, l, byte_size, schedule=SCHEDULE, num_threads=THREADS):

                dst_array = &destination_array[ j ]
                src_array = &source_array[ j ]
                f_array = &tmp_array[ j ]

                if modes == 'BGR(X)':

                    # using pointer address instead of array indexing.
                    # In a contiguous buffer memory array the next element is the pointer address + 1
                    f_array[ 0 ] = min(<unsigned char> (<float> (src_array + 2)[0] * c4 + (dst_array + 2)[ 0 ] * tmp),
                                       <unsigned char> 255)

                    (f_array + 1)[ 0 ] = min(
                        <unsigned char> (<float> (src_array + 1)[ 0 ] * c4 + (dst_array + 1)[ 0 ] * tmp),
                        <unsigned char> 255)

                    (f_array + 2)[ 0 ] = min(
                        <unsigned char> (<float> src_array[ 0 ] * c4 + dst_array[ 0 ] * tmp),
                        <unsigned char> 255)

                    # force the alpha value to 255
                    if byte_size == 4:
                        (f_array + 3)[ 0 ] = <unsigned char>255

                # modes RGB(X)
                else:
                    f_array[ 0 ] = min(<unsigned char> (<float> src_array[ 0 ] * c4 + dst_array[ 0 ] * tmp),
                                       <unsigned char> 255)

                    (f_array + 1)[ 0 ] = min(
                        <unsigned char> (<float> (src_array + 1)[ 0 ] * c4 + (dst_array + 1)[ 0 ] * tmp),
                        <unsigned char> 255)

                    (f_array + 2)[ 0 ] = min(
                        <unsigned char> (<float> (src_array + 2)[ 0 ] * c4 + (dst_array + 2)[ 0 ] * tmp),
                        <unsigned char> 255)

                    # force the alpha value to 255
                    if byte_size == 4:
                        (f_array + 3)[ 0 ] = <unsigned char> 255

    return frombuffer(tmp_array, (w, h), 'RGBA' if byte_size == 4 else 'RGB')



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef void blend_inplace(
        object destination,
        object source,
        float percentage
        ):
    """
    
    Blend the source image into the destination (inplace) 
    
    source & destination Textures must be same sizes
    Compatible with 24 - 32 bit surface
    
    e.g:
    blend_inplace(destination, source, percentage = VALUE)
    
    :param destination     : 
        pygame.Surface, compatible 24, 32-bit 
    
    :param source: 
        pygame.Surface, compatible 24, 32-bit
    
    :param percentage : 
        float; Percentage value between [0.0 ... 100.0]
    
    :return:  
        void
        
    """

    # Ensure that 'source' is a valid Pygame Surface object.
    # If it's not, raise an error with the type of the argument.
    assert PyObject_IsInstance(source, Surface), \
        'Argument source must be a pygame.Surface got %s ' % type(source)

    # Ensure that 'destination' is also a valid Pygame Surface object.
    # If it's not, raise an error with the type of the argument.
    assert PyObject_IsInstance(destination, Surface), \
        'Argument destination must be a pygame.Surface got %s ' % type(destination)

    # Validate that the 'percentage' argument is within the valid range [0.0, 100.0].
    # If it's outside this range, raise an error with the provided percentage value.
    assert 0.0 <= percentage <= 100.0, \
        "\nIncorrect value for argument percentage should be [0.0 ... 100.0] got %s " % percentage

    # Check that the 'source' and 'destination' surfaces have the same size.
    # If the dimensions don't match, raise an error with the respective sizes.
    assert source.get_size() == destination.get_size(), \
        'Source and Destination surfaces must have same dimensions: ' \
        'Source (w:%s, h:%s), destination (w:%s, h:%s).' % \
        (*source.get_size(), *destination.get_size())

    # Attempt to get the pixel data of the source surface as a 3D array.
    # If this fails, raise a ValueError with the exception message.
    try:
        source_array = source.get_view('3')  # '3' refers to a 3D array (height, width, channels)
    except Exception as e:
        raise ValueError("\nCannot reference source pixels into a 3d array.\n %s " % e)

    # Attempt to get the pixel data of the destination surface as a 3D array.
    # If this fails, raise a ValueError with the exception message.
    try:
        destination_array = destination.get_view('3')
    except Exception as e:
        raise ValueError("\nCannot reference destination pixels into a 3d array.\n %s " % e)

    # Call the 'blend_inplace_c' function to perform the blending operation between the source and destination arrays.
    # The blending operation is done in-place based on the specified 'percentage' value.
    blend_inplace_c(destination_array, source_array, percentage)



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef cartoon(
        object surface_,
        unsigned int sobel_threshold = 128,
        unsigned int median_kernel   = 2,
        unsigned int color           = 8,
        unsigned int flag            = BLEND_RGB_ADD
):
    """
    Create a cartoon effect
    
    Compatible with 24 - 32 bit image 
    
    e.g:
    cartoon_image = cartoon(image)
    
    Parameters
    ----------
    surface_ : 
        pygame.Surface compatible 24 - 32 bit 
    
    sobel_threshold : 
        integer sobel threshold
    
    median_kernel : 
        integer median kernel 
    
    color : 
        integer; color reduction value (max color)
    
    flag : 
        integer; Blend flag e.g (BLEND_RGB_ADD, BLEND_RGB_SUB, 
        BLEND_RGB_MULT, BLEND_RGB_MAX, BLEND_RGB_MIN  

    Returns
    -------
    Return a pygame Surface with the cartoon effect 
    
    """

    # Check if the 'median_kernel' is less than 2.
    # If it is, raise a ValueError because the kernel size must be at least 2.
    if median_kernel < 2:
        raise ValueError("\nKernel size median_kernel must be >=2")

    # Check if the 'sobel_threshold' is within the valid range of 0 to 255.
    # If it's not, raise a ValueError because the threshold must be within this range.
    if not (0 <= sobel_threshold <= 255):
        raise ValueError("\nSobel threshold sobel_threshold must be in range 0...255")

    # Call the 'cartoon_c' function to apply a cartoon effect on the 'surface_'.
    # Pass the 'sobel_threshold', 'median_kernel', 'color', and 'flag' as parameters.
    # This function will perform the cartooning effect using these parameters.
    return cartoon_c(surface_, sobel_threshold, median_kernel, color, flag)






@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef inline void convert_27(object surface_):
    """
    Convert an image to a reduced color palette of 27 colors (inplace).

    This function applies an algorithm that converts a surface (image) to only 27 distinct colors,
    reducing the color depth of the image. It modifies the surface directly, so the result is
    stored back in the same surface.

    The algorithm is compatible with both 24-bit and 32-bit surface formats.

    Example:
    --------
    # Convert an image to 27 colors.
    convert_27(image)

    Parameters:
    -----------
    surface_ : pygame.Surface
        A Pygame surface (image) that is compatible with either 24-bit or 32-bit formats.

    Returns:
    --------
    void
        This function modifies the surface in place and does not return a new surface.
    """
    
    # Ensure that the provided surface is a valid Pygame surface.
    assert isinstance(surface_, Surface), \
        'Argument surface_ must be a valid Pygame.Surface, got %s ' % type(surface_)

    # Try to retrieve the 3D array view of the surface for manipulation.
    cdef unsigned char [:, :, :] array_
    
    try:
        array_ = surface_.get_view('3')
    except (pygame.error, ValueError):
        raise ValueError('\nCannot convert the surface into a 3D array.')

    # Get the dimensions of the surface (width and height).
    cdef Py_ssize_t w, h
    w, h = surface_.get_size()

    # Apply the conversion to 27 colors using the C function 'convert_27_c'.
    convert_27_c(w, h, array_)




@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef object bilateral(object image, const float sigma_s, const float sigma_i, unsigned int kernel_size = 3):
    """
    Apply bilateral filtering to an image and return a filtered copy.

    Bilateral filtering is a non-linear, edge-preserving, and noise-reducing 
    smoothing filter. It replaces the intensity of each pixel with a weighted 
    average of intensities from nearby pixels, with weights based on both 
    spatial proximity and intensity similarity. This allows the filter to blur 
    smooth regions while preserving sharp edges.

    The filter relies on two key parameters:
    - **sigma_s**: Spatial extent of the kernel. It defines the size of the 
      neighborhood around each pixel that influences the filter's operation.
    - **sigma_i**: Intensity range kernel. This controls how sensitive the 
      filter is to intensity differences. A smaller value of `sigma_i` preserves 
      edges more effectively, while a larger value allows for more uniform blurring.

    As `sigma_i` increases, the filter approaches a Gaussian blur (which is 
    applied uniformly across the image). A smaller value of `sigma_i` retains 
    more local detail by reducing the contribution of pixels with different intensities.

    **Example usage**:
        surface = bilateral(surface, 16.0, 18.0, 3)

    :param image: 
        Pygame Surface object (24-32 bit format). The alpha channel is ignored. 
        The image must be in the RGB format, and a view of it will be converted 
        to a 3D array for processing.

    :param sigma_s: 
        float; Spatial extent of the kernel. This value controls the size of 
        the neighborhood used to calculate the weighted average.

    :param sigma_i: 
        float; Intensity sensitivity. Defines the maximum intensity difference 
        that contributes to the blurring process. Smaller values preserve edges.

    :param kernel_size: 
        integer (default is 3); The size of the kernel. It defines how far 
        the filter will reach from each pixel, affecting the size of the 
        local neighborhood considered.

    :return: 
        Pygame Surface; A new surface with the bilateral filter applied.
    """

    # Ensure that the 'image' argument is a valid Pygame surface.
    assert isinstance(image, Surface), \
        'Argument image must be a valid Surface, got %s ' % type(image)

    # Ensure that 'sigma_s' is a valid float value for the spatial sigma in the bilateral filter.
    assert isinstance(sigma_s, float), \
        'Argument sigma_s must be a valid float, got %s ' % type(sigma_s)

    # Ensure that 'sigma_i' is a valid float value for the intensity sigma in the bilateral filter.
    assert isinstance(sigma_i, float), \
        'Argument sigma_i must be a valid float, got %s ' % type(sigma_i)

    # Try to retrieve a 3D array view of the 'image' surface for pixel manipulation.
    cdef unsigned char [:, :, :] array_

    try:
        # Get the 3D view of the image (for color channels).
        array_ = image.get_view('3')

    except (pygame.error, ValueError):
        # Raise an error if the surface cannot be converted into a 3D array.
        raise ValueError('\nCannot convert the surface into a 3D array.')

    # Apply the bilateral filter using the 'bilateral_c' C function with the image data and parameters.
    return bilateral_c(array_, sigma_s, sigma_i, kernel_size)




@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef object emboss(object surface_, unsigned short int flag_=0):
    """
    Apply an emboss filter to an image or surface, producing an embossed effect.
    
    The embossing filter creates a visual effect that makes the image appear raised, similar to a paper
    or metal embossing of the original image, which can be used to highlight edges or create artistic effects.
    
    e.g:
    # Apply the emboss effect and return a 24-bit image format.
    image = emboss(image)
    
    # Apply the emboss effect and return a 32-bit image format with blending.
    image = emboss(image, 1)
    
    Parameters
    ----------
    surface_ : pygame.Surface
        A Pygame surface compatible with 24-bit or 32-bit image formats.
    
    flag_ : int, optional, default=0
        A special Pygame blend flag such as BLEND_RGB_ADD, BLEND_RGB_MULT, etc.
        This flag modifies the image blending behavior and will affect the final output.

    Returns
    -------
    pygame.Surface
        A surface containing the embossed image. The output is a 24-bit format if flag is set to 0,
        otherwise, a 32-bit format is returned.
    """

    # Ensure the 'surface_' argument is a valid Pygame Surface object.
    assert isinstance(surface_, Surface), \
        'Argument surface_ must be a valid Surface, got %s ' % type(surface_)

    # Try to extract a 3D array view of the surface (for manipulation of color channels).
    cdef unsigned char [:, :, :] array_
    try:
        array_ = surface_.get_view('3')
    except (pygame.error, ValueError):
        # Raise an error if the surface cannot be converted to a 3D array.
        raise ValueError('\nCannot convert the surface into a 3D array.')

    # Apply the emboss effect by passing the 3D array to the C function.
    cdef object emb = emboss3d_c(array_)

    # If a blending flag is provided (non-zero), apply the blend mode and return the surface.
    if flag_ != 0:
        del array_
        surface_.blit(emb, (0, 0), special_flags=flag_)
        return surface_

    # Return the embossed image as a 24-bit surface when no blending is applied.
    return emb




@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef void emboss_inplace(object surface_, copy=None):

    """
    Emboss a surface (inplace)
    
    Applying an embossing filter to an image often results in an image resembling a paper
    or metal embossing of the original image, hence the name. 
    
    e.g:
    # 24-bit 
    image = pygame.image.load('../Assets/px.png').convert(24)
    pixel_copy = numpy.ascontiguousarray(array3d(image_copy).transpose(1, 0, 2))
    emboss_inplace(image, copy=pixel_copy)
    
    # 32-bit 
    image = pygame.image.load('../Assets/px.png').convert_alpha()
    pixel_copy = numpy.ascontiguousarray(array3d(image_copy).transpose(1, 0, 2))
    emboss_inplace(image, copy=pixel_copy)
    
    Parameters
    ----------
    surface_ : 
        Pygame.Surface to emboss 
        Changes apply inplace - meaning the surface will be directly modified once the process 
        is complete
        
    copy      : 
        numpy.ndarray shape (w, h, 3) type uint8 containing RGB pixels and must be the same sizes than 
        the input surface.Copy of the source array pixels (improve slightly the performance). 

    Returns
    -------
    void 
    
    """

    # Ensure the 'surface_' argument is a valid Pygame Surface object.
    assert isinstance(surface_, Surface), \
        'Argument surface_ must be a valid Surface, got %s ' % type(surface_)

    # Declare a 3D unsigned char array to hold pixel data for manipulation.
    cdef unsigned char [:, :, :] rgb_array

    # Try to get a 3D array view of the surface's pixels (for accessing color channels).
    try:
        rgb_array = surface_.get_view('3')
    except (pygame.error, ValueError):
        # Raise an error if the surface cannot be converted to a 3D array.
        raise ValueError('\nCannot convert the surface into a 3D array.')

    # Apply the embossing effect in-place by passing the 3D array and a copy flag.
    emboss3d_inplace_c(rgb_array, copy)




@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef inline void emboss1d(
        const Py_ssize_t w,
        const Py_ssize_t h,
        unsigned char [:] bgr_array,
        tmp_array = None,
        bint format_32 = False
):
    """
    Emboss directly a C-buffer type (inplace) 
    
    Applying an embossing filter to an image often results in an image resembling a paper
    or metal embossing of the original image, hence the name. 
    
    If you are using tmp_array to improve the performances, make sure to have the same 
    array size and shape than the source array bgr_array 
     
    e.g
    # 24 - bit 
    image = pygame.image.load('../Assets/px.png').convert(24)
    image = pygame.transform.smoothscale(image, (800, 600))
    image_copy = image.copy()
     
    emboss1d(800, 600, image.get_view('0'), image_copy.get_buffer(), False)
    
    # 32 - bit 
    image = pygame.image.load('../Assets/px.png').convert_alpha()
    image = pygame.transform.smoothscale(image, (800, 600))
    image_copy = image.copy()
     
    emboss1d(800, 600, image.get_view('0'), image_copy.get_buffer(), True)
    
     
    Parameters
    ----------
    w : 
        int; width of the surface 
        
    h : 
        int; height of the surface 
        
    bgr_array :
        numpy.ndarray or memoryviewslice shape (l, ) type uint8 containing BGR pixels or any other format 
        bgr_array represent the source data that will be modify. The changes are applied inplace - meaning 
        that the surface will be automatically changed after updating the source array data. 
        
    tmp_array : 
        numpy.ndarray or memoryviewslice shape (l, ) type uint8 containing BGR pixels or any other format 
        This array is a copy of the source array 
        
    format_32 :
        bool; default is False. Select True if the source array contains alpha transparency (32 - bit format).

    Returns
    -------
    void

    """

    cdef:
        Py_ssize_t l = bgr_array.shape[0]
        # below create a copy False of the array and do not reference the pixels.
        # The real time transformation of the identical copy of the array will not be functional as all the pixels
        # undergo constant transformations. It is then necessary to load the pixels from a copy of the source array
        # to implement the inplace transformation. Such as below
        const unsigned char [:] bgr_array_cp = numpy.ndarray(shape=l, buffer=bgr_array, dtype=uint8).copy() if\
            tmp_array is None else tmp_array

    # Only uint8 data is compatible
    if not is_uint8(bgr_array):
        raise TypeError("\nExpecting uint8 (unsigned char) data type got %s" % bgr_array.dtype)


    emboss1d_c(w, l, bgr_array, bgr_array_cp, format_32)


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef object emboss_gray(object surface_):
    """
    Apply a gray-scale embossing filter to an image or surface and return a modified copy.
    
    Embossing creates a raised effect on the image, often making it appear like it has been embossed
    onto paper or metal. This version applies the embossing effect in grayscale, making it ideal for 
    artistic or stylistic transformations. 

    Example usage:
        image = emboss_gray(image)
    
    Parameters
    ----------
    surface_ : pygame.Surface
        A Pygame surface compatible with 24-32 bit formats. The surface to which the emboss effect will be applied.
    
    Returns
    -------
    pygame.Surface
        A new surface with the gray-scale embossed image. The resulting image is always in 24-bit format.
    
    """

    # Ensure the 'surface_' argument is a valid Pygame Surface object.
    assert isinstance(surface_, Surface), \
        'Argument surface_ must be a valid Surface, got %s ' % type(surface_)

    # Declare a 3D unsigned char array to hold pixel data for manipulation (this will store the RGB channels).
    cdef unsigned char [:, :, :] array_

    # Attempt to get a 3D array view of the surface's pixels (RGB channels).
    try:
        array_ = surface_.get_view('3')
    except (pygame.error, ValueError):
        # Raise an error if the surface cannot be converted to a 3D array (e.g., invalid format).
        raise ValueError('\nCannot convert the surface into a 3D array.')

    # Apply the gray-scale embossing effect to the array and return the modified surface.
    return emboss3d_gray_c(array_)




@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef object bilinear(
    object surface_,
    tuple size_,
    fx=None,
    fy=None
    ):
    """
    Resize an image using the bilinear filter algorithm (returns a copy).

    This function applies the bilinear filter to resize an image. Bilinear filtering 
    smooths the image and is commonly used in image resizing tasks. The function 
    supports 32-bit input images, but the result is always returned in 24-bit format 
    (without the alpha channel).

    Example usage:
        image = bilinear(image, (600, 600))  # Resize image to 600x600
        image = bilinear(image, (600, 600), 2, 2)  # Resize with specific scaling factors

    Parameters
    ----------
    surface_ : pygame.Surface
        A Pygame surface, compatible with 24 or 32-bit formats, representing the image to be resized.
    
    size_ : tuple
        A tuple (width, height) specifying the new dimensions of the surface.
    
    fx : float, optional
        A scaling factor for the x-axis (width). If provided, it will override the width specified in `size_`.
        Default is None.
    
    fy : float, optional
        A scaling factor for the y-axis (height). If provided, it will override the height specified in `size_`.
        Default is None.
    
    Returns
    -------
    pygame.Surface
        A new Pygame surface of type 24-bit (without alpha channel), resized based on the input parameters.

    """
    
    # Ensure that the 'surface_' argument is a valid Pygame Surface object.
    assert isinstance(surface_, Surface), \
        'Argument surface_ must be a valid Surface, got %s ' % type(surface_)

    # Declare a 3D unsigned char array to hold pixel data for the surface (for RGB channels).
    cdef unsigned char [:,:,:] rgb_array

    # Try to obtain a 3D array view of the surface's pixels (RGB channels).
    try:
        rgb_array = surface_.get_view('3')
    except (pygame.error, ValueError):
        # Raise an error if the surface is not compatible for conversion to a 3D array.
        raise ValueError('\nTexture/image is not compatible.')

    # Call the C function 'bilinear_c' to perform the resizing and return the resized surface.
    return bilinear_c(rgb_array, size_, fx, fy)




@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef tuple tunnel_modeling24(
        const int screen_width,
        const int screen_height,
        object surface_
):

    """
    24-bit Tunnel modeling  
    This method will produce 24-bit rendering data 
    
    This algorithm uses a 256x256 texture but reshape it to 512x512 pixels for a
    better rendering
    
    e.g
    WIDTH = 800
    HEIGHT = 800
    BCK1 =  pygame.image.load("../Assets/space2.jpg").convert(24)
    BCK1 = pygame.transform.smoothscale(BCK1, (WIDTH, HEIGHT))
    BACKGROUND = pygame.image.load("../Assets/space1.jpg")
    BACKGROUND = pygame.transform.smoothscale(BACKGROUND, (WIDTH, HEIGHT))
    distances, angles, shades, scr_data = tunnel_modeling24(WIDTH, HEIGHT, BACKGROUND)
    
    Parameters
    ----------
    screen_width : 
        int; display's width or width of the tunnel effect
        
    screen_height : 
        int; display's height or height of the tunnel effect
        
    surface_ : 
        pygame.Surface; Tunnel texture effect compatible 24, 32-bit

    Returns
    -------
    python tuple containing 4 buffers (distances, angles, shades and scr_data)

    """

    assert screen_width > 0, "Argument screen_width must be > 0"
    assert screen_height > 0, "Argument screen_height must be > 0"

    cdef:
        int length = screen_width * screen_height * 4
        int [:] distances = numpy.empty(length, int32)
        int [:] angles    = numpy.empty(length, int32)
        int [:] shades    = numpy.empty(length, int32)
        unsigned int [:, :] indexes = \
            numpy.empty((screen_height * 2, screen_width * 2), dtype=numpy.uint32)
        int ii, jj, n

    cdef:
        int s_width  = 512
        int s_height = 512

    surface = surface_.convert()
    surface = smoothscale(surface, (s_width, s_height))

    cdef:
        unsigned char [::1] scr_data = surface.get_buffer()
        float sqy, sqx
        int x, y

    with nogil:

        n = 0

        for ii in range(0, screen_height * 2):
            for jj in range(0, screen_width * 2):
                indexes[ ii, jj ] = n
                n = n + 1

        for y in prange(0, screen_height * <unsigned short int>2):
            sqy = <float>pow(y - screen_height, <unsigned short int>2)

            for x in range(0, screen_width * <unsigned short int>2):

                sqx = <float>pow(x - screen_width, <unsigned short int>2)


                if (sqx + sqy) == 0:
                    distances[indexes[y, x]] = <unsigned short int>1
                else:
                    distances[indexes[y, x]] = \
                        <int>(<float>floor(<float>32.0 * <float>s_height / <float>sqrt(sqx + sqy))) % s_height

                angles[indexes[y, x]]    = <int>round_c(<float>s_width * <float>atan2(<float>y - <float>screen_height,
                                            <float>x - <float>screen_width) / (<float>M_PI))
                shades[indexes[y, x]]    = <int>min(<float>sqrt(sqx + sqy)* <float>10.0, <unsigned char>255)

    return distances, angles, shades, scr_data




@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef tunnel_render24(
        int t,
        const int screen_width,
        const int screen_height,
        const int screen_w2,
        const int screen_h2,
        const int [::1] distances,
        const int [::1] angles,
        const int [::1] shades,
        unsigned char [::1] scr_data,
        unsigned char [::1] dest_array):

    """
    Tunnel effect rendering 
    
    The output surface is 24-bit
    
    e.g
    surface_ = tunnel_render24(FRAME*5,WIDTH,HEIGHT,
        WIDTH >> 1,HEIGHT >> 1,distances,angles,shades,scr_data,dest_array) 
    SCREEN.blit(surface_, (0, 0), special_flags=pygame.BLEND_RGB_ADD)
        
    Parameters
    ----------
    t : 
        int; Timer or frame count. Control the speed of the effect 
        
    screen_width : 
        int; Display width or width of the tunnel effect 
        
    screen_height : 
        int; Display height or height of the tunnel effect 
        
    screen_w2 : 
        int; This is the screen_width divided by 2
        
    screen_h2 :
        int; This is the screen_height value divided by 2
        
    distances : 
        numpy.ndarray shape (l, ) C-buffer containing the distances.
        You need to get this data buffer from tunnel_modeling24 
        
    angles : 
        numpy.ndarray shape (l, ) C-buffer containing all the angles.
        You need to get this data buffer from tunnel_modeling24
        
    shades : 
        numpy.ndarray shape (l, ) C-buffer containing all the shades.
        You need to get this data buffer from tunnel_modeling24
        
    scr_data : 
        numpy.ndarray shape (l, ) C-buffer containing all the background pixels
        You need to get this data buffer from tunnel_modeling24
        
    dest_array : 
        numpy.ndarray shape (l, ) C-buffer empty. This is typically an empty buffer with 
        length equal to width * height * 4 (RGBA empty buffer)  
        This buffer will be used to build the final texture effect (image)

    Returns
    -------
    pygame.Surface; image compatible 24-bit 

    """

    cdef:
        unsigned int s_width  = 512
        unsigned int s_height = 512
        float timer = t * <float>1e-3
        unsigned int shiftx  = <unsigned int>floor(s_width * timer)
        unsigned int shifty  = <unsigned int>floor(s_height * <float>0.25 * timer)
        unsigned int centerx = <unsigned int>(screen_w2 +
                <float>floor((screen_w2 >> 1) * <float>sin(timer * <float>0.25)))
        unsigned int centery = \
            <unsigned int>(screen_h2 +
                           <float>floor((screen_h2 >> 1) * <float>sin(timer * <float>0.5)))
        unsigned int stride  = screen_width * <unsigned short int>2
        unsigned int dest_ofs = 0
        unsigned int pix_ofs = 0
        int u, v, x, y
        unsigned int shade, src_ofs

    with nogil:
        for y in prange(0,  screen_height,  schedule=SCHEDULE, num_threads=THREADS):
            for x in range(0, screen_width):

                src_ofs = y * stride + centerx + centery * stride + x
                dest_ofs = (y * <unsigned int> screen_height + x) << 2

                u = (distances[ src_ofs ] + shiftx) & 0xff
                v = (angles[ src_ofs ] + shifty) & 0xff

                shade = <int> (shades[ src_ofs ] * <float>ONE_255)

                pix_ofs = (u + (v << <unsigned short int> 9)) << <unsigned short int> 3

                dest_array[ dest_ofs ] = scr_data[ pix_ofs + <unsigned short int> 2 ] * shade
                dest_array[ dest_ofs + <unsigned short int> 1 ] = \
                    scr_data[ pix_ofs + <unsigned short int> 1 ] * shade
                dest_array[ dest_ofs + <unsigned short int> 2 ] = \
                    scr_data[ pix_ofs + <unsigned short int> 0 ] * shade
                # dest_array[ dest_ofs + <unsigned short int> 3 ] = <unsigned char> 255

    # !!! Convert improve the overall performances, do not remove (strip the alpha channel)
    return frombuffer(dest_array, (screen_width, screen_height), "RGBA").convert()



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef tuple tunnel_modeling32(
        const Py_ssize_t screen_width,
        const Py_ssize_t screen_height,
        object surface_
):
    """
    Generate 32-bit Tunnel Modeling Effect.

    This method simulates a tunnel effect and produces 32-bit rendering data based on 
    the provided surface texture. It uses a 256x256 texture but reshapes it to a 
    512x512 resolution for better rendering quality. The algorithm calculates various 
    parameters such as distances, angles, shades, and rendering data, which can be 
    used to visualize the tunnel effect.

    Example usage:
        WIDTH = 800
        HEIGHT = 800
        BCK1 = pygame.image.load("../Assets/space2.jpg").convert(24)
        BCK1 = pygame.transform.smoothscale(BCK1, (WIDTH, HEIGHT))
        BACKGROUND = pygame.image.load("../Assets/space1.jpg")
        BACKGROUND = pygame.transform.smoothscale(BACKGROUND, (WIDTH, HEIGHT))
        distances, angles, shades, scr_data = tunnel_modeling32(WIDTH, HEIGHT, BACKGROUND)

    Parameters
    ----------
    screen_width : int
        The width of the display or the width of the tunnel effect.

    screen_height : int
        The height of the display or the height of the tunnel effect.

    surface_ : pygame.Surface
        The texture surface used for the tunnel effect. The surface should be compatible 
        with 24 or 32-bit formats.

    Returns
    -------
    tuple
        A tuple containing four buffers:
        - distances: A buffer representing the calculated distances for the effect.
        - angles: A buffer representing the calculated angles for the effect.
        - shades: A buffer representing the calculated shades (brightness) for the effect.
        - scr_data: A buffer containing the texture data of the surface.
    """

    # Validate screen dimensions to ensure they are positive integers
    assert screen_width > 0, "Argument screen_width must be > 0"
    assert screen_height > 0, "Argument screen_height must be > 0"

    # Declare buffers for storing distances, angles, shades, and indexes (for pixel mapping)
    cdef:
        ssh4 = screen_width * screen_height * 4
        int [:] distances = numpy.empty(ssh4, int32)  # Buffer for distances
        int [:] angles    = numpy.empty(ssh4, int32)  # Buffer for angles
        int [:] shades    = numpy.empty(ssh4, int32)  # Buffer for shades (brightness)
        unsigned int[ :, : ] indexes = numpy.empty(
            (screen_height * 2, screen_width * 2), dtype = numpy.uint32)  # Pixel indexes for mapping

    # Convert the surface to 32-bit with alpha channel for processing
    surface = surface_.convert_alpha()

    # Resize the surface to 512x512 for better rendering
    cdef:
        int s_width  = 512
        int s_height = 512

    surface = smoothscale(surface, (s_width, s_height))  # Smooth scaling for texture resolution

    # Create a buffer for accessing the surface pixel data (RGBA)
    cdef:
        unsigned char [::1] scr_data = surface.get_buffer()  # Pixel data buffer
        float sqy, sqx  # Temporary variables for distance calculations
        int x, y        # Pixel coordinates
        int n, ii, jj   # Loop counters

    # Initialize pixel index counter for mapping
    n = 0
    with nogil:  # Release the Global Interpreter Lock for optimized loop performance
        # Create a grid of pixel indexes for the 2x scaled screen
        for ii in range(0, screen_height * 2):
            for jj in range(0, screen_width * 2):
                indexes[ii, jj] = n
                n = n + 1  # Increment pixel index

        # Loop through each pixel in the 2x scaled screen area
        for y in prange(0, screen_height * 2):  # Parallelize the outer loop for better performance
            sqy = <float>pow(y - screen_height, <unsigned short int>2)  # Calculate the squared Y distance from the center

            for x in range(0, screen_width * 2):
                sqx = <float>pow(x - screen_width, <unsigned short int>2)  # Calculate the squared X distance from the center

                # Calculate distances, angles, and shades based on pixel positions
                if (sqx + sqy) == 0:
                    distances[indexes[y, x]] = <unsigned short int>1  # Prevent division by zero
                else:
                    distances[indexes[y, x]] = <int>(<float>floor(
                        <float>32.0 * <float>s_height / <float>sqrt(sqx + sqy))) % s_height  # Calculate distance
                angles[indexes[y, x]] = <int>round_c(<float>s_width *
                    <float>atan2(<float>y - <float>screen_height, <float>x - <float>screen_width) / (<float>M_PI))  # Calculate angle
                shades[indexes[y, x]] = <int>min(<float>sqrt(sqx + sqy)* <float>10.0, <unsigned char>255)  # Calculate shade (brightness)

    # Return the calculated data as a tuple
    return distances, angles, shades, scr_data



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef tunnel_render32(
        int t,
        const Py_ssize_t screen_width,
        const Py_ssize_t screen_height,
        const int screen_w2,
        const int screen_h2,
        const int [::1] distances,
        const int [::1] angles,
        const int [::1] shades,
        unsigned char [::1] scr_data,
        unsigned char [::1] dest_array):
    """
    Tunnel effect rendering 
    
    The output surface is 32-bit
    
    e.g
    surface_ = tunnel_render32(FRAME*5,WIDTH,HEIGHT,
        WIDTH >> 1,HEIGHT >> 1,distances,angles,shades,scr_data,dest_array) 
    SCREEN.blit(surface_, (0, 0), special_flags=pygame.BLEND_RGB_ADD)
    
    Parameters
    ----------
    t : 
       int; Timer or frame count. Control the speed of the effect 
       
    screen_width : 
       int; Display width or width of the tunnel effect 
       
    screen_height : 
       int; Display height or height of the tunnel effect 
       
    screen_w2 : 
       int; This is the screen_width divided by 2
       
    screen_h2 :
       int; This is the screen_height value divided by 2
       
    distances : 
       numpy.ndarray shape (l, ) C-buffer containing the distances.
       You need to get this data buffer from tunnel_modeling32 
       
    angles : 
       numpy.ndarray shape (l, ) C-buffer containing all the angles.
       You need to get this data buffer from tunnel_modeling32
       
    shades : 
       numpy.ndarray shape (l, ) C-buffer containing all the shades.
       You need to get this data buffer from tunnel_modeling32
       
    scr_data : 
       numpy.ndarray shape (l, ) C-buffer containing all the background pixels
       You need to get this data buffer from tunnel_modeling32
       
    dest_array : 
       numpy.ndarray shape (l, ) C-buffer empty. This is typically an empty buffer with 
       length equal to width * height * 4 (RGBA empty buffer)  
       This buffer will be used to build the final texture effect (image)
    
    Returns
    -------
    pygame.Surface; image compatible 32-bit 

    """


    assert screen_width > 0, "Argument screen_width must be > 0"
    assert screen_height > 0, "Argument screen_height must be > 0"
    assert screen_w2 > 0, "Argument screen_w2 must be > 0"
    assert screen_h2 > 0, "Argument screen_h2 must be > 0"

    cdef:
        int s_width  = 512
        int s_height = 512
        float timer = t * <float>1e-3
        int shiftx  = <int>floor(s_width * timer)
        int shifty  = <int>floor(s_height * <float>0.25 * timer)
        int centerx = <int>(screen_w2 +
                            <float>floor((screen_w2 >> 1) * <float>sin(timer * <float>0.25)))
        int centery = <int>(screen_h2 + <float>floor(
            (screen_h2 >> 1) * <float>sin(timer * <float>0.5)))
        int stride  = <int>screen_width * <unsigned short int>2
        int dest_ofs = 0
        int src_ofs
        int u, v, x, y
        int pix_ofs, shade


    with nogil:

        for y in prange(0, screen_height, schedule=SCHEDULE, num_threads=4):

            for x in range(0, screen_width):

                src_ofs = y * stride + centerx + centery * stride + x
                dest_ofs = (y * <int>screen_height + x) << 2

                u = (distances[src_ofs] + shiftx) & 0xff
                v = (angles[src_ofs] + shifty) & 0xff

                shade = <int> (shades[src_ofs] * <float>ONE_255)

                pix_ofs = (u + (v << <unsigned short int>9)) << <unsigned short int>3

                dest_array[dest_ofs] = scr_data[pix_ofs + <unsigned short int>2] * shade
                dest_array[dest_ofs + <unsigned short int>1] = \
                    scr_data[pix_ofs + <unsigned short int>1] * shade
                dest_array[dest_ofs + <unsigned short int>2] =\
                    scr_data[pix_ofs + <unsigned short int>0] * shade
                dest_array[dest_ofs + <unsigned short int>3] = <unsigned char>255

    return pygame.image.frombuffer(dest_array, (screen_width, screen_height), "RGBA").convert_alpha()

# ----------------



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef inline float damped_oscillation(float t) nogil:
    """
    Compute the value of a damped oscillation at a given time.

    This function calculates a damped oscillation value based on the input time `t`.
    It combines an exponential decay function and a cosine function to model the oscillation.
    The decay rate is controlled by the factor `0.1` in the exponential term, and the frequency 
    of the oscillation is determined by the cosine function with a frequency of `π`.

    Example:
        y = damped_oscillation(2.0)
    
    Parameters
    ----------
    t : float
        The time variable (x) at which the damped oscillation is to be evaluated.
    
    Returns
    -------
    float
        The resulting value (y) of the damped oscillation function, where y = f(x).
        The oscillation value that decays over time while oscillating with a cosine wave.
    """

    # Calculate the damped oscillation value using exponential decay and cosine
    # The exponential decay is modeled with exp(-t * 0.1), which decays over time
    # The cosine oscillation is modeled with cos(M_PI * t), which oscillates between -1 and 1
    return <float>(<float>exp(-t * <float>0.1) * <float>cos(M_PI * t))
    



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cdef inline float gauss(float x, float c, float sigma=1.0, float mu=0.0) nogil:
    """
    Compute the value of a Gaussian function at a given point.

    This function evaluates the Gaussian (normal distribution) function based on the input value `x`.
    It calculates the probability density function (PDF) of a Gaussian distribution, with a mean (`mu`) 
    and standard deviation (`sigma`). The function can be shifted by a constant `c`.

    For more details on the mathematical formulation, refer to:
    https://en.wikipedia.org/wiki/Gaussian_function

    Example:
        y = gauss(2.0, 0.0, sigma=1.0, mu=0.0)
    
    Parameters
    ----------
    x : float
        The input variable at which to evaluate the Gaussian function.
    
    c : float
        A constant that shifts the input `x`. This effectively shifts the Gaussian curve along the x-axis.

    sigma : float, optional, default=1.0
        The standard deviation of the Gaussian distribution. It controls the width of the bell curve.
    
    mu : float, optional, default=0.0
        The mean (or center) of the Gaussian distribution. It controls the peak position of the bell curve.

    Returns
    -------
    float
        The value of the Gaussian function at the point `x`, shifted by `c`, with the specified `sigma` and `mu`.
        The result represents the probability density at that point in the Gaussian distribution.
    """

    # Shift the input `x` by the constant `c` to modify the position
    x -= c
    
    # Compute the Gaussian function using the formula:
    # exp(-0.5 * ((x - mu)^2) / sigma^2) scaled by 1 / (sigma * sqrt(2 * pi))
    # In this case, we use a constant `C1_` which is precomputed to represent 1 / sqrt(2 * pi)
    
    # return <float>((<float>1.0 / sigma * C1_) * exp(-<float>0.5 * ((x - mu) * (x - mu)) / (sigma * sigma)))
    return 0.0



# @cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef inline void bgr_c(unsigned char [:, :, :] rgb_array) nogil:
    """
    Convert an array of shape (w, h, n) from RGB(A) to BGR(A) format (in-place).

    This function performs an in-place conversion of the color channels in an image
    represented by a numpy array. The image is assumed to be in RGB(A) format (where 
    'n' is either 3 for RGB or 4 for RGBA) and the function converts it to BGR(A) format
    by swapping the red and blue channels.

    **Note**: The function operates in-place, meaning it modifies the original array.

    **Example usage**:
        bgr_c(bgr_array)

    :param rgb_array: 
        A numpy array of shape (w, h, n), where `w` is the image width, 
        `h` is the image height, and `n` is 3 (for RGB) or 4 (for RGBA). The array 
        contains 8-bit unsigned integer values (0-255) representing the pixel color 
        values in RGB(A) format.

    :return: 
        void; modifies the array in-place.
    """
    # Get the width (w) and height (h) of the image (array shape)
    w, h = rgb_array.shape[ :2 ]

    # Declare variables for the loop, temporary storage, and pointers for red and blue components
    cdef:
        int i = 0, j = 0  # Loop counters for x and y coordinates (pixels)
        unsigned char tmp  # Temporary variable for swapping colors
        unsigned char *r  # Pointer to the red channel of the current pixel
        unsigned char *b  # Pointer to the blue channel of the current pixel

    # Iterate over each row (y-coordinate) of the image, parallelizing the outer loop for performance
    for j in prange(h, schedule = SCHEDULE, num_threads = THREADS):
        # Iterate over each column (x-coordinate) of the image
        for i in range(w):
            # Set the pointer to the blue component of the current pixel
            b = &rgb_array[ i, j, 0 ]

            # Set the pointer to the red component of the current pixel (which is 2 indices ahead of the blue component)
            r = b + 2  # Equivalent to: &rgb_array[i, j, 2]

            # Swap the red and blue values in the pixel by using the temporary variable
            tmp = <unsigned char> b[ 0 ]  # Store the current blue value in tmp
            b[ 0 ] = <unsigned char> r[ 0 ]  # Assign the red value to the blue channel
            r[ 0 ] = tmp  # Assign the stored blue value to the red channel

@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef inline void bgr_1d_c(unsigned char [::1] rgb_array, bint format_32=False)nogil:
    """
    Convert a 1D array from RGB(A) to BGR(A) format in-place.

    This function rearranges the color channels of a 1D array containing pixel data,
    switching the red and blue channels to convert an image from RGB(A) order to 
    BGR(A) order. The operation is performed in-place, meaning the original array is modified.

    **Example:**
        bgr_1d_c(rgb_array)

    ### Parameters:
    
    - **rgb_array** (*numpy.ndarray* or memoryview slice, shape (w,), dtype uint8):  
      A 1D array containing pixel data in RGB(A) order. For an RGB image, the array length 
      should be a multiple of 3, and for an RGBA image, it should be a multiple of 4.
      
    - **format_32** (*bool*, default False):  
      Indicates the pixel format of the input array:
        - `True` for 32-bit (RGBA).
        - `False` for 24-bit (RGB).

    ### Returns:
    - **None**:  
      The function modifies the input array in-place and does not return a new array.
    """

    cdef unsigned int l = rgb_array.shape[0]

    cdef:
        int i=0
        unsigned char tmp
        unsigned char * r
        unsigned char * b
        unsigned short int bit = 3

    if format_32:
        bit = 4

    # noinspection SpellCheckingInspection
    for i in prange(0, l, bit, schedule=SCHEDULE, num_threads=THREADS):

        # BGR
        r = &rgb_array[ i     ]
        b = r + 2 # &rgb_array[ i + 2 ]

        tmp = r[0]
        r[0] = <unsigned char> b[0]
        b[0] = <unsigned char> tmp




@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef inline np.ndarray[np.uint8_t, ndim=1] bgr_1d_cp_c(
        unsigned char [::1] rgb_array, bint format_32=False):

    """
    Convert a 1D uint8 array from RGB(A) format to BGR(A) format and return a new array.
    
    This function takes a 1D array (or memoryview slice) representing image pixel data in 
    RGB or RGBA order and returns a new 1D numpy.ndarray with the red and blue channels swapped, 
    resulting in a BGR or BGRA order. This is useful when interfacing with libraries or systems 
    that require BGR(A) formatted data.
    
    **Example:**
        bgr_array = bgr_1d_cp_c(rgb_array)
    
    ### Parameters:
    - **rgb_array**:
        - *Type:* numpy.ndarray or memoryview slice
        - *Shape:* (w,), where w is the total number of elements. For an RGB image, w should be a 
          multiple of 3, and for an RGBA image, a multiple of 4.
        - *Description:* Contains pixel data in RGB(A) order (with values ranging from 0 to 255).
    
    - **format_32**:
        - *Type:* bool
        - *Description:* 
            - `True` if the input array is in 32-bit format (RGBA), 
            - `False` if in 24-bit format (RGB).
    
    ### Returns:
    - A new 1D numpy.ndarray of type uint8 with the same shape as the input, but with the color channels 
      converted to BGR(A) order.
    """
    cdef:
        unsigned int l = rgb_array.shape[0]

    cdef:
        int i=0
        unsigned char [::1] destination_array = numpy.empty(l, dtype=uint8)
        unsigned short int bit = 3
        unsigned char * index
        unsigned char * array_index


    if format_32: bit = 4

    # noinspection SpellCheckingInspection
    with nogil:

        if format_32:
            for i in prange(0, l, bit, schedule=SCHEDULE, num_threads=THREADS):
                # BGR
                index = &destination_array[ i ]
                array_index = &rgb_array[ i ]
                index[0] = <unsigned char> (array_index + 2)[0]
                (index + 1)[0] = <unsigned char> (array_index + 1)[0]
                (index + 2)[0] = <unsigned char> array_index[0]
                (index + 3)[0] = <unsigned char> (array_index + 3)[0]
        else:
            for i in prange(0, l, bit, schedule=SCHEDULE, num_threads=THREADS):
                # BGR
                index = &destination_array[ i ]
                array_index = &rgb_array[ i ]
                index[ 0 ] = <unsigned char> (array_index + 2)[ 0 ]
                (index + 1)[ 0 ] = <unsigned char> (array_index + 1)[ 0 ]
                (index + 2)[ 0 ] = <unsigned char> array_index[ 0 ]

    return numpy.ndarray(shape=l, buffer=destination_array, dtype=uint8)







@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef inline void brg_c(unsigned char [:, :, :] rgb_array):
    """
    Convert an image from RGB(A) to BRG(A) format in-place.

    This function swaps the red and green channels of an image while preserving 
    the blue channel and alpha channel (if present). It is useful for converting 
    game displays, textures, or Pygame surfaces to the equivalent BRG format.

    ### Example Usage:
        brg_c(rgb_array)

    ### Parameters:
    - **rgb_array** (*numpy.ndarray, shape (w, h, n), dtype=uint8*):  
      A 3D NumPy array containing image pixel data in RGB(A) format.  
      Each pixel is represented by unsigned 8-bit values (0–255).  
      The function supports both 24-bit (RGB) and 32-bit (RGBA) formats.

    ### Returns:
    - **None**:  
      The function modifies the input array in-place and does not return a new array.
    """

    cdef Py_ssize_t w, h
    w, h = rgb_array.shape[:2]

    cdef:
        int i=0, j=0
        unsigned char tmp_r, tmp_g

    with nogil:

        for j in prange(h, schedule=SCHEDULE, num_threads=THREADS):

            for i in range(w):

                tmp_r = rgb_array[i, j, 0]  # keep the red color
                tmp_g = rgb_array[i, j, 1]  # keep the green color
                rgb_array[i, j, <unsigned short int>0] = rgb_array[i, j, <unsigned short int>2] # r-->b
                rgb_array[i, j, <unsigned short int>1] = tmp_r  # g --> r
                rgb_array[i, j, <unsigned short int>2] = tmp_g



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef inline void brg_1d_c(
        unsigned char [::1] rgb_array, bint format_32=False):

    """
    Convert a 1D NumPy array of type uint8 from RGB(A) to BGR(A) in place.

    This function modifies the input array directly, swapping the red and blue channels.
    It supports both 24-bit (RGB) and 32-bit (RGBA) image buffers.

    Set `format_32` to `True` if the array contains 32-bit (RGBA) values.

    Example Usage:
    --------------
    # Convert a 24-bit RGB image
    brg_1d_c(rgb_array)

    # Convert a 32-bit RGBA image
    brg_1d_c(rgb_array, format_32=True)

    Parameters
    ----------
    rgb_array : numpy.ndarray
        A 1D NumPy array of shape (w,) and dtype uint8, containing image pixel data 
        in RGB(A) format.

    format_32 : bool, optional (default=False)
        If `False`, the function assumes an RGB (24-bit) buffer.
        If `True`, it assumes an RGBA (32-bit) buffer.

    Returns
    -------
    None
        The function modifies the input array in place.
    """

    cdef unsigned int l = rgb_array.shape[ 0 ]

    cdef:
        int i = 0
        unsigned char tmp_r
        unsigned char tmp_g
        unsigned short int bit = 3

    if format_32:
        bit = 4

    with nogil:

        # noinspection SpellCheckingInspection
        for i in prange(0, l, bit, schedule = SCHEDULE, num_threads = THREADS):
            tmp_r = rgb_array[ i     ]
            tmp_g = rgb_array[ i + 1 ]
            rgb_array[ i     ] = <unsigned char>rgb_array[ i + 2 ]
            rgb_array[ i + 1 ] = <unsigned char>tmp_r
            rgb_array[ i + 2 ] = <unsigned char>tmp_g


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef inline np.ndarray[np.uint8_t, ndim=1] brg_1d_cp_c(
        const unsigned char [::1] bgr_array, bint format_32=False):

    """
    Convert a 1D uint8 array from BGR(A) to BRG(A) format and return a new array.
    
    This function swaps the red and green channels while preserving the blue 
    and alpha channels (if present). It processes a 1D array representation 
    of an image and returns a new array with the modified pixel format.
    
    ### Example Usage:
        brg_array = brg_1d_cp_c(bgr_array)
    
    ### Parameters:
    - **bgr_array** (*numpy.ndarray*, shape *(w,)*, dtype *uint8*):  
      A 1D array containing image pixel data in BGR(A) format, where values 
      range from 0 to 255.
    
    - **format_32** (*bool*):  
      - `True`: Input is a 24-bit (BGR) format.  
      - `False`: Input is a 32-bit (BGRA) format.  
    
    ### Returns:
    - **numpy.ndarray** (*shape (w,), dtype uint8*):  
      A new 1D array with pixel data converted to BRG(A) format.
    """

    cdef:
        unsigned int l = bgr_array.shape[0]

    cdef:
        int i=0
        unsigned char [::1] destination_array = numpy.empty(l, dtype=uint8)
        unsigned short int bit = 3
        unsigned char * index

    if format_32:
        bit = 4


    with nogil:
        if format_32:
            # noinspection SpellCheckingInspection
            for i in prange(0, l, bit, schedule=SCHEDULE, num_threads=THREADS):
                # BRG
                index = &destination_array[ i ]
                index[0] = <unsigned char> bgr_array[ i + 2 ]
                (index+1)[0] = <unsigned char> bgr_array[ i     ]
                (index+2)[0] = <unsigned char> bgr_array[ i + 1 ]
                (index+3)[0] = <unsigned char> bgr_array[ i + 3 ]
        else:
            for i in prange(0, l, bit, schedule = SCHEDULE, num_threads = THREADS):
                # BRG
                index = &destination_array[ i ]
                index[ 0 ] = <unsigned char> bgr_array[ i + 2 ]
                (index + 1)[ 0 ] = <unsigned char> bgr_array[ i ]
                (index + 2)[ 0 ] = <unsigned char> bgr_array[ i + 1 ]

    return numpy.ndarray(shape=l, buffer=destination_array, dtype=uint8)



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef inline void grey_c(unsigned char [:, :, :] rgb_array):
    """
    Convert an image to grayscale while preserving luminosity (in-place).

    This function converts a 3-channel (RGB) or 4-channel (RGBA) image to grayscale by 
    calculating the luminosity of each pixel, which preserves the perceived brightness. 
    The alpha (transparency) channel will be ignored in 32-bit (RGBA) images.

    Example Usage:
    --------------
    grey_c(rgb_array)

    Parameters
    ----------
    rgb_array : numpy.ndarray
        A 3D NumPy array of shape (w, h, 3) for RGB images, or (w, h, 4) for RGBA images. 
        The array should be of type uint8, with pixel values ranging from 0 to 255. 
        For RGBA images, the alpha channel will be ignored.

    Returns
    -------
    None
        The function modifies the input array in place and does not return a new array.
    """


    cdef Py_ssize_t w, h
    w, h = rgb_array.shape[:2]

    cdef:
        int i, j
        unsigned char luminosity
        unsigned char *r
        unsigned char *g
        unsigned char *b

    with nogil:
        for j in prange(h, schedule=SCHEDULE, num_threads=THREADS):
            for i in range(w):
                r = &rgb_array[i, j, 0]
                g = &rgb_array[i, j, 1]
                b = &rgb_array[i, j, 2]
                luminosity = <unsigned char>(r[0] * <float>0.2126 + g[0] * <float>0.7152 + b[0] * <float>0.072)
                r[0] = luminosity
                g[0] = luminosity
                b[0] = luminosity


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef np.ndarray[np.uint8_t, ndim=2] grey_2d_c(unsigned char [:, :, :] rgb_array):
    """
    Convert a 3D RGB(A) array into a 2D grayscale array.

    This function converts an input RGB or RGBA array to a 2D grayscale array based 
    on the luminosity of each pixel. The resulting 2D array contains pixel intensities 
    that range from 0 (black) to 255 (white). The alpha (transparency) channel is 
    ignored for 32-bit images with transparency.

    Example Usage:
    --------------
    grey_array = grey_2d_c(rgb_array)

    Parameters
    ----------
    rgb_array : numpy.ndarray
        A 3D NumPy array of shape (w, h, 3) for RGB images, or (w, h, 4) for RGBA images.
        The array should have dtype uint8, with pixel values ranging from 0 to 255. 
        The alpha channel will be ignored in RGBA images.

    Returns
    -------
    numpy.ndarray
        A 2D NumPy array of shape (w, h) with dtype uint8, containing the grayscale 
        image data. Each pixel represents intensity, ranging from 0 (black) to 255 (white).
    """


    cdef Py_ssize_t w, h
    w, h = rgb_array.shape[:2]

    cdef:
        int i, j
        unsigned char luminosity
        unsigned char [:, :] grayscale = numpy.empty((w, h), dtype=uint8)

    with nogil:
        for j in prange(h, schedule=SCHEDULE, num_threads=THREADS):
            for i in range(w):
                luminosity = <unsigned char>(rgb_array[i, j, 0] * <float>0.2126 +
                                             rgb_array[i, j, 1] * <float>0.7152 +
                                             rgb_array[i, j, 2] * <float>0.072)
                grayscale[i, j] = luminosity
    return numpy.ndarray(shape=(w, h), buffer=grayscale, dtype=uint8)


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef inline void grey_1d_c(
        unsigned char [:] rgb_array, bint format_32=False):
    """
    Convert a 1D array of uint8 data (RGB(A)) to grayscale (with alpha) in place.

    A grayscale image has a single channel representing pixel intensity or brightness, 
    with pixel values typically ranging from 0 (black) to 255 (white). This function 
    converts the input RGB(A) array to grayscale while preserving the alpha channel 
    (if present). The conversion is performed in place, modifying the original array.

    Set `format_32` to `True` if the array is a 32-bit buffer containing RGBA values.

    Parameters
    ----------
    rgb_array : numpy.ndarray or bytearray
        A 1D array or buffer containing pixel data in RGB(A) format, with dtype uint8 
        (unsigned char values ranging from 0 to 255).

    format_32 : bool, optional (default=False)
        If `True`, the function assumes the input is a 32-bit buffer (RGBA).
        If `False`, the function assumes a 24-bit buffer (RGB).

    Returns
    -------
    None
        The function modifies the input array in place and does not return a new array.
    """

    cdef unsigned int l = rgb_array.shape[ 0 ]

    cdef:
        int i = 0
        unsigned char * r
        unsigned char * g
        unsigned char * b
        unsigned char luminosity = 0
        unsigned short int bit = 3

    if format_32:
        bit = 4


    with nogil:

        # noinspection SpellCheckingInspection
        for i in prange(0, l, bit, schedule = SCHEDULE, num_threads = THREADS):
            # Assuming an RGB array
            r = &rgb_array[ i     ]
            g = &rgb_array[ i + 1 ]
            b = &rgb_array[ i + 2 ]
            luminosity = <unsigned char> (   r[ 0 ] * <float> 0.2126
                                           + g[ 0 ] * <float> 0.7152
                                           + b[ 0 ] * <float> 0.072)
            r[ 0 ] = luminosity
            g[ 0 ] = luminosity
            b[ 0 ] = luminosity




@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef inline np.ndarray[np.uint8_t, ndim=1] grey_1d_cp_c(
    const unsigned char [::1] bgr_array,
    bint format_32=False):

    """
    Convert a 1D array of type uint8 from BGR(A) to grayscale (with alpha channel) and return a copy.

    This function converts the input BGR or BGRA array to grayscale based on the luminosity 
    formula, preserving the alpha channel (if present). The conversion generates a new 1D 
    NumPy array in grayscale format, where the pixel values represent intensity.

    Parameters
    ----------
    bgr_array : numpy.ndarray
        A 1D NumPy array of shape (w,) with dtype uint8, containing pixel data in 
        BGR(A) format. The pixel values should range from 0 to 255. 
        If the array represents a BGRA image, the alpha channel will be preserved.

    format_32 : bool, optional (default=False)
        If `True`, the input array is assumed to be in BGRA (32-bit) format. 
        If `False`, the array is assumed to be in BGR (24-bit) format.

    Returns
    -------
    numpy.ndarray
        A new 1D NumPy array of shape (w,) with dtype uint8, containing the grayscale 
        pixel data in grayscale format. If the input was in BGRA format, the alpha channel
        is preserved in the output.
    """

    cdef:
        Py_ssize_t l = bgr_array.shape[0]

    cdef:
        int i=0
        unsigned char [::1] destination_array = numpy.empty(l, dtype=uint8)
        unsigned short int channels = 3
        unsigned char luminosity = 0
        unsigned char * index
        const unsigned char * bgr_index

    if format_32:
        channels = 4


    with nogil:

        # noinspection SpellCheckingInspection
        for i in prange(0, l, channels, schedule=SCHEDULE, num_threads=THREADS):

            index = &destination_array[ i ]
            bgr_index = &bgr_array[ i ]
            luminosity = <unsigned char> ( (bgr_index + 2)[0] * <float> 0.2126
                                         + (bgr_index + 1)[0] * <float> 0.7152
                                         + bgr_index[0] * <float> 0.072)

            index[0] = luminosity
            (index + 1)[0] = luminosity
            (index + 2)[0] = luminosity
            if format_32:
                (index + 3)[0] = <unsigned char> (bgr_index + 3)[0]

    return numpy.ndarray(shape=l, buffer=destination_array, dtype=uint8)





@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef inline void sepia_c(unsigned char [:, :, :] rgb_array):

    """
    Apply a sepia tone filter to a 3D RGB image array (inplace).

    This function transforms an image (or video game surface) into a sepia-toned version by applying
    a filter that shifts the colors of the RGB channels toward a reddish-brown hue. The result gives
    the image a warm, vintage look, often associated with old photographs.

    The input array (`rgb_array`) must be a 3D NumPy array with shape (w, h, 3), where `w` is the width,
    `h` is the height, and `3` corresponds to the RGB channels. If you're working with a Pygame surface,
    you can use functions like `pixels3d` or `array3d` from the `surfarray` module to convert it into a 3D array.

    In sepia-toned images, unlike traditional black-and-white photographs that use standard grayscale,
    the color spectrum is adjusted to a warmer, reddish-brown tone, resulting in a softer, dreamier aesthetic.

    Parameters
    ----------
    rgb_array : numpy.ndarray
        A 3D NumPy array of shape (w, h, 3) with dtype uint8 (unsigned char 0...255), containing RGB pixel data.
        The array should represent an image in RGB format, typically from Pygame display or surface data.

    Returns
    -------
    void
        This function modifies the input array (`rgb_array`) directly and does not return a new array.

    Example
    -------
    # For a Pygame surface:
    sepia(surface)

    Notes
    -----
    - The input array should have dtype uint8, representing pixel values ranging from 0 to 255.
    - If working with Pygame surfaces, consider using `pixels3d` or `array3d` from `surfarray` 
        to convert the surface to a 3D array.
    - This function modifies the input array in place.
    """
    cdef Py_ssize_t w, h
    w, h = rgb_array.shape[:2]

    cdef:
        int i=0, j=0
        float rr, gg, bb
        unsigned char *r
        unsigned char *g
        unsigned char *b

    with nogil:
        for j in prange(h, schedule=SCHEDULE, num_threads=THREADS):
            for i in range(w):

                # RGB
                r = &rgb_array[i, j, <unsigned short int>0]
                g = &rgb_array[i, j, <unsigned short int>1]
                b = &rgb_array[i, j, <unsigned short int>2]

                rr = r[0] * <float>0.393 + g[0] * <float>0.769 + b[0] * <float>0.189
                gg = r[0] * <float>0.349 + g[0] * <float>0.686 + b[0] * <float>0.168
                bb = r[0] * <float>0.272 + g[0] * <float>0.534 + b[0] * <float>0.131

                r[0] = <unsigned char> rr if rr<255 else <unsigned char>255
                g[0] = <unsigned char> gg if gg<255 else <unsigned char>255
                b[0] = <unsigned char> bb if bb<255 else <unsigned char>255




@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef void sepia_1d_c(unsigned char [:] rgb_array, bint format_32=False):
    """
    Convert 1d array BGR(A) (uint8) into sepia equivalent model. 
    
    While traditional black-and-white photographs use a standard grayscale to create
    highlights and shadows, sepia-toned photos use a reddish-brown tone to create that spectrum.
    “Sepia is a softer manipulation of light,”. This gives them a softer, dreamier aesthetic.
    
    e.g 
    # image 24-bit  
    im = pygame.image.load("../Assets/px.png")
    w, h = im.get_width(), im.get_height()
    c = numpy.ndarray(shape=(w*h*3), buffer=im.get_view('0'), dtype=uint8)
    sepia_1d(c, False)   

    
    # image 32-bit 
    import pygame
    im = pygame.image.load("../Assets/px.png")
    w, h = im.get_width(), im.get_height()
    sepia_1d(im.get_view('0'), True)
    
    Parameters
    ----------
    
    rgb_array : 
        numpy.ndarray; memoryviewslice; 1d array of uint8 data type containing BGR(A) pixels
        or any other pixel format.
        
    format_32 : 
        bool True | for 'BGR' buffer type (24-bit) or False 'BGRA' (32-bit)

    Returns
    -------
    void
    
    """

    cdef Py_ssize_t l = rgb_array.shape[ 0 ]

    cdef:
        int i = 0
        int bit = 3
        float rr, gg, bb
        unsigned char * r
        unsigned char * g
        unsigned char * b

    if format_32:
        bit = 4


    with nogil:
        for i in prange(0, l, bit, schedule = SCHEDULE, num_threads = THREADS):

                # BGR format
                r = &rgb_array[ i + 2 ]
                g = &rgb_array[ i + 1 ]
                b = &rgb_array[ i     ]

                rr = r[ 0 ] * <float> 0.393 + g[ 0 ] * <float> 0.769 + b[ 0 ] * <float> 0.189
                gg = r[ 0 ] * <float> 0.349 + g[ 0 ] * <float> 0.686 + b[ 0 ] * <float> 0.168
                bb = r[ 0 ] * <float> 0.272 + g[ 0 ] * <float> 0.534 + b[ 0 ] * <float> 0.131

                r[ 0 ] = <unsigned char>rr if rr < 255 else <unsigned char>255
                g[ 0 ] = <unsigned char>gg if gg < 255 else <unsigned char>255
                b[ 0 ] = <unsigned char>bb if bb < 255 else <unsigned char>255



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef unsigned char [::1]  sepia_1d_cp_c(
    const unsigned char [::1] bgr_array,
    unsigned char [::1] destination_array,
    bint format_32=False)nogil:
    """
    Apply a sepia tone filter to a 1D array of image data (inplace).
    
    This function processes a 1D array representing pixel data in BGR(A) format and 
    applies a sepia filter. The sepia effect is achieved by adjusting the red, green, 
    and blue channels according to a set of predefined coefficients, creating a vintage, 
    warm-toned effect. The result is stored in the provided destination array.
    
    Parameters
    ----------
    bgr_array : numpy.ndarray
        A 1D array containing pixel data in BGR(A) format, with dtype uint8. Each pixel 
        is represented by 3 (for BGR) or 4 (for BGRA) values corresponding to the blue, green, 
        red, and optionally alpha channels. The pixel values should range from 0 to 255.
    
    destination_array : numpy.ndarray
        A 1D array of the same length as `bgr_array`, where the sepia-filtered pixel data will 
        be stored. This array is modified directly (inplace).
    
    format_32 : bool, optional (default=False)
        If `True`, the input array is assumed to be in BGRA format (32-bit), meaning each 
        pixel contains four channels. If `False`, the input is assumed to be in BGR format 
        (24-bit), meaning each pixel contains three channels (blue, green, red).
    
    Returns
    -------
    numpy.ndarray
        The `destination_array` with the sepia-toned pixel data. The array is modified 
        directly (inplace) and is returned for convenience.
    """

    cdef unsigned int l = bgr_array.shape[ 0 ]  # Get the length of the input array

    # Initialize variables
    cdef:
        int i = 0, channels = 3  # Initialize iteration index and number of channels
        float rr, gg, bb  # Variables to store the new red, green, and blue values
        const unsigned char * r  # Pointer to the red channel
        const unsigned char * g  # Pointer to the green channel
        const unsigned char * b  # Pointer to the blue channel

    if format_32:  # Check if input is in BGRA format
        channels = 4  # Update channel count for BGRA format

    with nogil:  # Release the Global Interpreter Lock for parallel execution

        # Iterate through the image data in steps of 'channels' (3 for BGR, 4 for BGRA)
        for i in prange(0, l, channels, schedule = SCHEDULE, num_threads = THREADS):

            # Get pointers to individual color channels (BGR)
            r = &bgr_array[ i + 2 ]  # Pointer to the red channel
            g = &bgr_array[ i + 1 ]  # Pointer to the green channel
            b = &bgr_array[ i ]  # Pointer to the blue channel

            # Apply sepia filter using the luminosity coefficients for each channel
            rr = r[ 0 ] * <float> 0.393 + g[ 0 ] * <float> 0.769 + b[ 0 ] * <float> 0.189
            gg = r[ 0 ] * <float> 0.349 + g[ 0 ] * <float> 0.686 + b[ 0 ] * <float> 0.168
            bb = r[ 0 ] * <float> 0.272 + g[ 0 ] * <float> 0.534 + b[ 0 ] * <float> 0.131

            # Ensure the resulting pixel values are capped at 255
            destination_array[ i ] = <unsigned char> rr if rr < 255 else <unsigned char> 255
            destination_array[ i + 1 ] = <unsigned char> gg if gg < 255 else <unsigned char> 255
            destination_array[ i + 2 ] = <unsigned char> bb if bb < 255 else <unsigned char> 255

            # If in BGRA format, preserve the alpha channel
            if format_32:
                destination_array[ i + 3 ] = bgr_array[ i + 3 ]

    return destination_array  # Return the modified destination array



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef inline void median_inplace_c(
        unsigned char [:, :, :] rgb_array_, int kernel_size_=2):

    """
    Median filter (inplace)
    
    The median filter is a non-linear digital filtering technique, 
    often used to remove noise from an image or signal. 
    Such noise reduction is a typical pre-processing step to improve 
    the results of later processing (for example, edge detection on an image).
    Median filtering is very widely used in digital image processing because, 
    under certain conditions, it preserves edges while removing noise 
    
    The output median effect strength is controlled by the kernel size variable.
    This method cannot be used for real time rendering

    The Array (rgb_array) must be a numpy array shape (w, h, 3) containing RGB(A) pixels,
     
    e.g
    # 24-bit 
    im = pygame.image.load("../Assets/background.jpg")
    im = scale(im, (800, 600))
    w, h = im.get_width(), im.get_height()
    median_inplace_c(pixels3d(im), fast=True)
    
    # 32-bit 
    im = pygame.image.load("../Assets/px.png").convert_alpha()
    im = scale(im, (800, 600))
    w, h = im.get_width(), im.get_height()
    median_inplace_c(pixels3d(im), fast=False)

    
    Parameters
    ----------
    rgb_array_ : 
        numpy.ndarray shape(w, h, 3) uint8 (unsigned char 0...255) containing the
        pygame display pixels format RGB
        
    kernel_size_ : 
        integer; size of the kernel

    Returns
    -------
    void
    """

    cdef Py_ssize_t w, h
    w, h = rgb_array_.shape[:2]

    cdef int k = kernel_size_ >> 1
    cdef int v = k * k * 3, kx, ky

    cdef:
        unsigned char [::1, :, :] rgb_array_copy = \
                    numpy.asarray(rgb_array_, order='F')

        int i=0, j=0
        Py_ssize_t ii=0, jj=0

        unsigned char *tmp_red   = <unsigned char *> malloc(v * sizeof(unsigned char))
        unsigned char *tmp_green = <unsigned char *> malloc(v * sizeof(unsigned char))
        unsigned char *tmp_blue  = <unsigned char *> malloc(v * sizeof(unsigned char))

        unsigned char [::1] tmp_red_ = <unsigned char[:v]>tmp_red
        unsigned char [::1] tmp_green_ = <unsigned char[:v]>tmp_green
        unsigned char [::1] tmp_blue_ = <unsigned char[:v]>tmp_blue

        unsigned char *tmpr
        unsigned char *tmpg
        unsigned char *tmpb

        int index = 0, val
        Py_ssize_t w_1 = w - 1, h_1 = h - 1


    with nogil:
        for j in prange(h, schedule=SCHEDULE, num_threads=THREADS, chunksize=2048):
            for i in range(w):

                index = 0

                for kx in range(-k, k):

                    ii = i + kx

                    if ii < 0:
                        ii = 0

                    elif ii >= w_1:
                        ii = <int> w_1

                    for ky in range(-k, k):

                        jj = j + ky

                        if jj < 0:
                            jj = 0

                        elif jj >= h_1:
                            jj = <int>h_1

                        tmp_red_[index]   = rgb_array_copy[ii, jj, 0]
                        tmp_green_[index] = rgb_array_copy[ii, jj, 1]
                        tmp_blue_[index]  = rgb_array_copy[ii, jj, 2]

                        index = index + 1


                # External C quicksort
                tmpr = new_quickSort(&tmp_red_[0], 0, v)
                tmpg = new_quickSort(&tmp_green_[0], 0, v)
                tmpb = new_quickSort(&tmp_blue_[0], 0, v)

                val = (index - 1) >> 1

                rgb_array_[i, j, 0] = tmpr[val]
                rgb_array_[i, j, 1] = tmpg[val]
                rgb_array_[i, j, 2] = tmpb[val]

    free(tmp_red)
    free(tmp_green)
    free(tmp_blue)



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef inline void median_grayscale_inplace_c(
        unsigned char [:, :, :] rgb_array_, int kernel_size_=2):

    """
    Median grayscale (inplace)
    
    The median filter is a non-linear digital filtering technique, 
    often used to remove noise from an image or signal. 
    Such noise reduction is a typical pre-processing step to improve 
    the results of later processing (for example, edge detection on an image).
    Median filtering is very widely used in digital image processing because, 
    under certain conditions, it preserves edges while removing noise 

    The output image is a greyscale image with a median filter effect. 
    The output median effect strength is controlled by the kernel size variable.  
    
    This method cannot be used for real time rendering. 

    The surface is compatible 24 - 32 bit with or without alpha layer
    
    e.g
    # 24-bit 
    im = pygame.image.load("../Assets/background.jpg")
    median_grayscale_inplace_c(im)
    
    # 32-bit 
    im = pygame.image.load("../Assets/px.png").convert_alpha()
    median_grayscale_inplace_c(im)
    

    :param rgb_array_: 
        numpy.ndarray shape(w, h, n) uint8 (unsigned char 0...255) containing the
        pygame display or surface pixels format RGB(A)
        
    :param kernel_size_: 
        integer; size of the kernel, default kernel_size_=2
        
    :return: 
        void
        
    """

    cdef:
        Py_ssize_t w, h
        int k = kernel_size_ >> 1, ky, kx
        unsigned char v = k * k * 4

    w, h = rgb_array_.shape[:2]

    cdef:

        unsigned char [::1, :, :] rgb_array_copy = \
            numpy.asarray(rgb_array_, order='F')

        int i=0, j=0
        Py_ssize_t ii=0, jj=0

        unsigned char *tmp_   = <unsigned char *> malloc(v * sizeof(unsigned char))
        unsigned char *tmp

        int index = 0
        unsigned char val

    # multiprocessing is not used here.
    # tmp_ share memory view cannot be shared across all the instances and
    # using multiprocess with chunks will be slower
    with nogil:
        for j in range(0, h):
            for i in range(0, w):

                index = 0

                for kx in range(-k, k):

                    ii = i + kx

                    if ii < 0:
                        ii = 0
                    elif ii > w - 1:
                        ii = w - 1

                    for ky in range(-k, k):

                        jj = j + ky

                        if jj < 0:
                            jj = 0
                        elif jj > h - 1:
                            jj = h - 1

                        tmp_[index]   = rgb_array_copy[ii, jj, 0]

                        index = index + 1

                tmp = new_quickSort(tmp_, 0, v)

                val = tmp[(v >> 1) - 1]

                rgb_array_[i, j, 0] = val
                rgb_array_[i, j, 1] = val
                rgb_array_[i, j, 2] = val

    free(tmp_)



cdef float ONE_255 = <float>1.0 / <float>255.0


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef inline void posterize_surface_c(
        unsigned char [:, :, :] rgb_array, int color_number):
    """
    Apply color reduction (posterization) to an image.

    Reduces the number of distinct colors in the given image or texture, creating a posterized effect.
    The function maps the RGB values of the image to a limited set of colors, effectively reducing the 
    color depth.

    Example usage:
    posterize_surface_c(surface, 8)  # Reduce the color depth of the surface to 8 colors.

    :param rgb_array: 
        A 3D numpy.ndarray with shape (width, height, 3), containing RGB values in the range [0, 255].
        This array represents the image or texture to be posterized. Modifying this array will directly 
        alter the underlying image or display surface.

    :param color_number: 
        An integer specifying the target number of colors to reduce the image to.
        A higher number retains more colors, while a lower number results in a more heavily posterized image.

    :return: 
        None. The function modifies the input `rgb_array` in place.
    """

    cdef Py_ssize_t w, h
    w, h = rgb_array.shape[:2]


    cdef:
        int x=0, y=0
        unsigned char *r
        unsigned char *g
        unsigned char *b
        float f = <float> 255.0 / <float> color_number
        float c1 = <float>color_number * <float>ONE_255

    with nogil:
        for y in prange(h, schedule=SCHEDULE, num_threads=THREADS):
            for x in range(0, w):

                r = &rgb_array[x, y, 0]
                g = &rgb_array[x, y, 1]
                b = &rgb_array[x, y, 2]
                if r[0]!=0:
                    r[0] = <unsigned char>(<int>(round_c(c1 * r[0]) * f))
                if g[0]!=0:
                    g[0] = <unsigned char>(<int>(round_c(c1 * g[0]) * f))
                if b[0]!=0:
                    b[0] = <unsigned char>(<int>(round_c(c1 * b[0]) * f))



cdef:
    short [:, ::1] GY = numpy.array(
        ([-1, 0, 1],
         [-2, 0, 2],
         [-1, 0, 1])).astype(dtype=int16, order='C')

    short [:, ::1] GX = numpy.array(
        ([-1, -2, -1],
         [0,   0,  0],
         [1,   2,  1])).astype(dtype=int16, order='c')

    unsigned short int KERNEL_HALF = 1



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef inline void sobel_inplace_c(
        unsigned char [:, :, :] rgb_array,
        float threshold=20.0
):
    """
    Sobel (edge detection)

    Transform the game display or a pygame surface into a sobel equivalent model
    The array must be greyscaled but non greyscale array will also work, 
    only the red channel will be used to code the sobel filter 

    e.g:
    sobel(surface, 64)

    :param rgb_array: 
        numpy.ndarray shape (w, h, 3) type uint8 referencing the game display or SDL surface
        containing RGB values. Any change to this array will modify the SDL surface directly
        
    :param threshold: 
        float; Threshold value (Default value = 20.0)
        
    :return: 
        void
        
    """

    cdef Py_ssize_t w, h
    w, h = rgb_array.shape[:2]

    cdef unsigned char [::1, :, :] source_array

    try:
        source_array = numpy.asarray(rgb_array, order='F')

    except Exception as e:
        raise ValueError("\nCannot reference source pixels into a 3d array.\n %s " % e)


    cdef:
        Py_ssize_t w_1 = w - 1
        Py_ssize_t h_1 = h - 1
        int kernel_offset_x, kernel_offset_y
        int x, y
        Py_ssize_t xx, yy
        float r_gx, r_gy
        unsigned char *gray
        unsigned char m
        float magnitude

    with nogil:

        for y in prange(h, schedule=SCHEDULE, num_threads=THREADS):

            for x in range(w):

                r_gx, r_gy = <float>0.0, <float>0.0

                for kernel_offset_x in range(-KERNEL_HALF, KERNEL_HALF + 1):
                    xx = x + kernel_offset_x
                    if xx > w_1:
                        xx = w_1
                    if xx < 0:
                        xx = <unsigned int>0

                    for kernel_offset_y in range(-KERNEL_HALF, KERNEL_HALF + 1):
                        yy = y + kernel_offset_y
                        if yy > h_1:
                            yy = h_1
                        if yy < 0:
                            yy = <unsigned int>0

                        # grayscale image red = green = blue
                        gray = &source_array[xx, yy, 0]

                        # if gray[0]!=0:

                        # if kernel_offset_x != 0:

                        r_gx = r_gx + <float> gray[0] * \
                               <float> GX[kernel_offset_x + KERNEL_HALF,
                                          kernel_offset_y + KERNEL_HALF]

                        # if kernel_offset_y != 0:

                        r_gy = r_gy + <float> gray[0] * \
                               <float> GY[kernel_offset_x + KERNEL_HALF,
                                          kernel_offset_y + KERNEL_HALF]

                magnitude = min(<float> sqrt(r_gx * r_gx + r_gy * r_gy), <float>255.0)

                # if magnitude > 255:
                #     magnitude = 255.0

                m = <unsigned char> magnitude if magnitude > threshold else 0

                # update the pixel if the magnitude is above threshold else black pixel
                rgb_array[x, y, 0] = m
                rgb_array[x, y, 1] = m
                rgb_array[x, y, 2] = m



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef inline void sobel_1d_c(
        const Py_ssize_t w,
        const Py_ssize_t l,
        unsigned char [::1] bgr_array,
        const unsigned char [::1] bgr_array_cp,
        float threshold=20.0,
        bint format_32=False,
        bint greyscale=False) nogil:
    """
    Applies a 1D Sobel edge detection filter to a C-style buffer (in-place).

    This function computes the Sobel gradient of an image to detect edges. 
    It operates on a raw C-buffer, modifying it in place. While the input 
    array should ideally be greyscale, the function can process non-greyscale 
    images as well. If `greyscale=True`, the computation is simplified by 
    processing only a single channel (typically the blue channel in BGR format). 

    For improved performance, a copy of the source array (`bgr_array_cp`) can be 
    provided, ensuring both arrays have the same dimensions.

    Example Usage:
    --------------
    # 24-bit Image
    image = pygame.image.load('../Assets/px.png').convert(24)
    image = pygame.transform.smoothscale(image, (800, 600))
    grey(image)  # Convert to greyscale before applying Sobel
    image_copy = image.copy()
    sobel_1d_c(800, 600, image.get_buffer(), image_copy.get_buffer(), threshold=25.0)

    # 32-bit Image
    image = pygame.image.load('../Assets/px.png').convert_alpha()
    image = pygame.transform.smoothscale(image, (800, 600))
    image_copy = image.copy()
    sobel_1d_c(800, 600, image.get_buffer(), image_copy.get_buffer(), threshold=25.0, format_32=True)

    Parameters:
    -----------
    w : int
        Width of the image or array.

    l : int
        Total length of the array (width * height * bytes per pixel).

    bgr_array : numpy.ndarray (1D C-buffer), dtype=uint8
        A contiguous C-buffer representing the image pixels in BGR or BGRA format.
        This buffer is modified in place.

    bgr_array_cp : numpy.ndarray (1D C-buffer), dtype=uint8
        A copy of `bgr_array` used to improve performance. It must have the same 
        dimensions as `bgr_array`.

    threshold : float, default=20.0
        The Sobel gradient threshold for edge detection.

    format_32 : bool, default=False
        - `True`: Input buffer is in 32-bit BGRA format.
        - `False`: Input buffer is in 24-bit BGR format.

    greyscale : bool, default=False
        - `True`: Performs the Sobel operation on a single channel (blue) for 
          improved efficiency when working with greyscale images.
        - `False`: Applies the filter to all three RGB channels.

    Returns:
    --------
    None
        The function modifies `bgr_array` in place.
    """



    cdef short bitsize
    bitsize = 3 if format_32 == False else 4

    cdef:

        int i, cx, cy
        unsigned int row = w * bitsize
        float magnitude
        unsigned char m
        const unsigned char * p1
        const unsigned char * p2
        const unsigned char * p3
        const unsigned char * p4



    for i in prange(0, l, bitsize, schedule=SCHEDULE, num_threads=THREADS):

        if row + bitsize < i < l - row - bitsize:

            p1 = &bgr_array_cp[ i - row - bitsize ]
            p2 = &bgr_array_cp[ i - row + bitsize ]
            p3 = &bgr_array_cp[ i + row - bitsize ]
            p4 = &bgr_array_cp[ i + row + bitsize ]

            # blue
            cy =  \
                - p1[0] \
                + p2[0] \
                - bgr_array_cp[ i - bitsize ] * <int>2 \
                + bgr_array_cp[ i + bitsize ] * <int>2 \
                - p3[0] \
                + p4[0]

            cx = \
                - p1[0] \
                - bgr_array_cp[ i - row ] * <int>2 \
                - p2[0] \
                + p3[0] \
                + bgr_array_cp[ i + row ] * <int>2 \
                + p4[0]

            if not greyscale:
                # green
                cy = cy \
                    - (p1+1)[ 0 ] \
                    + (p2+1)[ 0 ] \
                    - bgr_array_cp[ i - bitsize +1 ] * <int> 2 \
                    + bgr_array_cp[ i + bitsize +1] * <int> 2 \
                    - (p3+1)[ 0 ] \
                    + (p4+1)[ 0 ]

                cx = cx \
                    - (p1+1)[ 0 ] \
                    - bgr_array_cp[ i - row +1] * <int> 2 \
                    - (p2+1)[ 0 ] \
                    + (p3+1)[ 0 ] \
                    + bgr_array_cp[ i + row +1] * <int> 2 \
                    + (p4+1)[ 0 ]

                # red
                cy = cy \
                     - (p1 + 2)[ 0 ] \
                     + (p2 + 2)[ 0 ] \
                     - bgr_array_cp[ i - bitsize + 2 ] * <int> 2 \
                     + bgr_array_cp[ i + bitsize + 2 ] * <int> 2 \
                     - (p3 + 2)[ 0 ] \
                     + (p4 + 2)[ 0 ]

                cx = cx \
                     - (p1 + 2)[ 0 ] \
                     - bgr_array_cp[ i - row + 2 ] * <int> 2 \
                     - (p2 + 2)[ 0 ] \
                     + (p3 + 2)[ 0 ] \
                     + bgr_array_cp[ i + row + 2 ] * <int> 2 \
                     + (p4 + 2)[ 0 ]


        magnitude = min(<float> sqrt(cx * cx + cy * cy), <float> 255.0)
        m = <unsigned char> magnitude if magnitude > threshold else 0
        # update the pixel if the magnitude is above threshold else black pixel

        # Alpha channel is unchanged
        bgr_array[ i    ] = m
        bgr_array[ i + 1] = m
        bgr_array[ i + 2] = m




@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef inline void sobel_fast_inplace_c(
        surface_, int threshold_=20, unsigned short factor_=1):
    """
    Fast sobel (inplace)

    Transform the game display or a pygame surface into a sobel equivalent model
    This version is slightly fastest than sobel_inplace_c as
    it down-scale the array containing all the pixels and apply the sobel algorithm to a smaller
    sample. When the processing is done, the array is re-scale to its original dimensions.
    If this method is in theory faster than sobel_inplace_c, down-scaling and up-scaling
    an array does have a side effect such as decreasing the overall image definition
    (jagged lines non-antialiasing)
    
    The surface/array must be greyscaled but non greyscale surface/array will also work, 
    only the red channel will be used to code the sobel filter 

    e.g:
    sobel_fast_inplace_c(surface, 64, amplitude=1)

    :param surface_: 
        pygame.Surface
        
    :param threshold_: 
        integer; threshold value for the sobel filter
        
    :param factor_: 
        unsigned short (default value =1). Define the
        reduction factor of an image. 1 divide by 2, 2 divide by 4 etc
        
    :return: 
        void
        
    """

    cdef:
        int w, h, w_1, h_1
    w, h = surface_.get_size()

    cdef:
        int kernel_offset_x, kernel_offset_y
        int x, y
        Py_ssize_t xx, yy
        float r_gx, r_gy
        unsigned char *gray
        unsigned char m
        float magnitude

        unsigned char [:, :, :] source_array = surface_.get_view('3')
        unsigned char [:, :, :] rescale_array = \
            numpy.asarray(pixels3d(scale(surface_, (w >> factor_, h >> factor_))))
        unsigned char [:, :, :] new_array = empty((w >> factor_, h >> factor_, 3), uint8)

    h = h >> factor_
    w = w >> factor_
    w_1 = w - 1
    h_1 = h - 1

    with nogil:

        for y in prange(h, schedule=SCHEDULE, num_threads=THREADS):

            for x in range(w):

                r_gx, r_gy = <float>0.0, <float>0.0

                for kernel_offset_y in range(-KERNEL_HALF, KERNEL_HALF + 1):
                    yy = y + kernel_offset_y
                    if yy > h_1:
                        yy = h_1
                    if yy < 0:
                        yy = 0

                    for kernel_offset_x in range(-KERNEL_HALF, KERNEL_HALF + 1):

                        xx = x + kernel_offset_x
                        if xx > w_1:
                            xx = w_1
                        if xx < 0:
                            xx = 0

                        # grayscale image red = green = blue
                        gray = &rescale_array[xx, yy, 0]

                        if kernel_offset_x != 0:

                            r_gx = r_gx + <float> gray[0] * \
                                   <float> GX[kernel_offset_x + KERNEL_HALF,
                                              kernel_offset_y + KERNEL_HALF]

                        if kernel_offset_y != 0:

                            r_gy = r_gy + <float> gray[0] * \
                                   <float> GY[kernel_offset_x + KERNEL_HALF,
                                              kernel_offset_y + KERNEL_HALF]

                magnitude = <float> sqrt(r_gx * r_gx + r_gy * r_gy)

                if magnitude > 255:
                    magnitude = <float>255.0

                m = <unsigned char> magnitude if magnitude > threshold_ else 0

                # update the pixel if the magnitude is above threshold else black pixel
                new_array[x, y, 0] = m
                new_array[x, y, 1] = m
                new_array[x, y, 2] = m

    w = w << factor_
    h = h << factor_

    cdef unsigned char [:, :, :] new_ = resize_array_c(new_array, w, h)

    with nogil:

        for y in prange(h, schedule=SCHEDULE, num_threads=THREADS):
            for x in range(w):

                source_array[x, y, 0] = new_[x, y, 0]
                source_array[x, y, 1] = new_[x, y, 1]
                source_array[x, y, 2] = new_[x, y, 2]


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef inline void invert_inplace_c(
        unsigned char [:, :, :] rgb_array):

    """
    Invert the color values of all pixels in a given image or texture (in-place).

    This function performs an in-place inversion of all the pixel values in the provided image or texture,
    effectively creating a negative of the image. Each pixel's color components (Red, Green, Blue, and optionally 
    Alpha) are inverted by subtracting their current value from the maximum value (255 for 8-bit color channels).

    The function operates directly on the provided `rgb_array`, modifying the pixel values in place.

    Example:
        invert(surface)

    Parameters
    ----------
    rgb_array : numpy.ndarray
        A 3D NumPy array representing the image or texture, with shape (width, height, channels).
        The array should contain pixel values in the RGB or RGBA format, with each component being 
        an unsigned 8-bit integer (uint8), where each color channel ranges from 0 to 255.

    Returns
    -------
    void
        This function modifies the provided `rgb_array` in place and does not return any value.
        
    Notes
    -----
    The function modifies the input array directly, so there is no need to assign the result to a new variable.
    """


    cdef Py_ssize_t w, h
    w, h = rgb_array.shape[:2]

    cdef:
        int i=0, j=0
        unsigned char *r
        unsigned char *g
        unsigned char *b

    with nogil:
        for j in prange(h, schedule=SCHEDULE, num_threads=THREADS):
            for i in range(w):
                r = &rgb_array[i, j, 0]
                g = &rgb_array[i, j, 1]
                b = &rgb_array[i, j, 2]
                r[0] = <unsigned char>255 - r[0]
                g[0] = <unsigned char>255 - g[0]
                b[0] = <unsigned char>255 - b[0]



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef void invert3d_c(unsigned char [:, :, :] rgb_array):
    """
    
    Invert 3d array pixels (inplace)
    
    Invert a 3d array shape (w, h, n) uint8 data type
    
    Inverting an image means inverting the pixel values.
    Images are represented using RGB or Red Green Blue values. 
    Each can take up an integer value between 0 and 255 (both included). 
    For example, a red color is represent using (255, 0, 0), white with (255, 255, 255), 
    black with (0, 0, 0) and so on. Inverting an image means reversing the colors on the image.
    For example, the inverted color for red color will be (0, 255, 255). Note that 0 
    became 255 and 255 became 0. This means that inverting an image is essentially subtracting 
    the old RGB values from 255. 
    
    e.g
    # 24 bit
    image = pygame.image.load('../Assets/px.png').convert(24)
    invert3d_c(array3d)
    
    # 32 bit
    image = pygame.image.load('../Assets/px.png').convert_alpha()
    invert3d_c(array3d)
    
    Parameters
    ----------
    rgb_array: 
        numpy.ndarray shape (w, h, n) containing RGB(A) pixel format and 
        works with any other formats such as BGR, BGRA

    Returns
    -------
    void
    
    """
    cdef Py_ssize_t w, h
    w, h = rgb_array.shape[ :2 ]

    cdef:
        int i = 0, j = 0
        unsigned char *r
        unsigned char *g
        unsigned char *b

    with nogil:
        for j in prange(h, schedule = SCHEDULE, num_threads = THREADS):
            for i in range(w):
                r = &rgb_array[ i, j, 0 ]
                g = &rgb_array[ i, j, 1 ]
                b = &rgb_array[ i, j, 2 ]
                r[ 0 ] = <unsigned char>255 - r[ 0 ]
                g[ 0 ] = <unsigned char>255 - g[ 0 ]
                b[ 0 ] = <unsigned char>255 - b[ 0 ]




@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef void invert1d_c(unsigned char [:] rgb_array, bint format_32=False):
    """
    Invert directly a C-buffer pixel values 
    
    Invert a C-buffer uint8 data types RGB(A) format
    
    This method will works with other buffer format such as BGR, BGRA
    
    Inverting an image means inverting the pixel values.
    Images are represented using RGB or Red Green Blue values. 
    Each can take up an integer value between 0 and 255 (both included). 
    For example, a red color is represent using (255, 0, 0), white with (255, 255, 255), 
    black with (0, 0, 0) and so on. Inverting an image means reversing the colors on the image.
    For example, the inverted color for red color will be (0, 255, 255). Note that 0 
    became 255 and 255 became 0. This means that inverting an image is essentially subtracting 
    the old RGB values from 255. 
    
    e.g
    # 24 bit
    image = pygame.image.load('../Assets/px.png').convert(24)
    invert1d_c(image.get_buffer(), False)
    
    # 32 bit
    image = pygame.image.load('../Assets/px.png').convert_alpha()
    invert1d_c(image.get_buffer(), True)
    
    Parameters
    ----------
    rgb_array: 
        numpy.ndarray; memoryviewslice; 1d array of uint8 data type 
        containing RGB(A) pixels or any other pixel format such as BGR or BGRA etc
        
    format_32: 
        bool True | for 'RGB' buffer type (24-bit) or False 'RGBA' (32-bit)

    Returns
    -------
    void

    """
    cdef Py_ssize_t l = rgb_array.shape[ 0 ]

    cdef:
        int i = 0
        int bit = 3
        unsigned char *r
        unsigned char *g
        unsigned char *b


    if format_32:
        bit = 4


    with nogil:
        # noinspection SpellCheckingInspection
        for i in prange(0, l, bit, schedule = SCHEDULE, num_threads = THREADS):
                r = &rgb_array[ i     ]
                g = &rgb_array[ i + 1 ]
                b = &rgb_array[ i + 2 ]
                r[ 0 ] = <unsigned char>255 - r[ 0 ]
                g[ 0 ] = <unsigned char>255 - g[ 0 ]
                b[ 0 ] = <unsigned char>255 - b[ 0 ]





@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef np.ndarray[np.uint8_t, ndim=1] invert1d_cp_c(
        const unsigned char [:] rgb_array,
        bint format_32=False):
    """
    
    Invert directly a C-buffer pixel values (return a copy)
    
    Invert C buffer uint8 data types RGB(A) format
    
    Inverting an image means inverting the pixel values.
    Images are represented using RGB or Red Green Blue values. 
    Each can take up an integer value between 0 and 255 (both included). 
    For example, a red color is represent using (255, 0, 0), white with (255, 255, 255), 
    black with (0, 0, 0) and so on. Inverting an image means reversing the colors on the image.
    For example, the inverted color for red color will be (0, 255, 255). Note that 0 
    became 255 and 255 became 0. This means that inverting an image is essentially subtracting 
    the old RGB values from 255. 
    
    This method will works with other buffer format such as BGR, BGRA
    
    e.g:
    # 24-bit
    image = pygame.image.load('../Assets/px.png').convert(24)
    arr3d = invert1d_cp(image.get_buffer(), False)
    image = pygame.image.frombuffer(arr3d, (WIDTH, HEIGHT), "BGR")
    
    # 32-bit
    image = pygame.image.load('../Assets/px.png').convert_alpha()
    arr3d = invert1d_cp(image.get_buffer(), True)
    image = pygame.image.frombuffer(arr3d, (WIDTH, HEIGHT), "BGRA")

    Parameters
    ----------
    rgb_array: 
        numpy.ndarray; memoryviewslice; 1d array of uint8 data type 
        containing RGB(A) pixels or any other pixel format such as BGR or BGRA etc
        
    format_32: 
        bool True | for 'RGB' buffer type (24-bit) or False 'RGBA' (32-bit)

    Returns
    -------
    Return a copy of the input array inverted.
    numpy.ndarray 1d array uint8 data type with pixel format equivalent to the input format

    """
    cdef Py_ssize_t l = rgb_array.shape[ 0 ]

    cdef:
        int i = 0
        const unsigned char *r
        const unsigned char *g
        const unsigned char *b
        unsigned char [::1] destination = empty(l, dtype=uint8)

    with nogil:

        if format_32:
            # noinspection SpellCheckingInspection
            for i in prange(0, l, 4, schedule = SCHEDULE, num_threads = THREADS):
                r = &rgb_array[ i ]
                g = &rgb_array[ i + 1 ]
                b = &rgb_array[ i + 2 ]
                destination[ i ] = <unsigned char> 255 - r[ 0 ]
                destination[ i + 1 ] = <unsigned char> 255 - g[ 0 ]
                destination[ i + 2 ] = <unsigned char> 255 - b[ 0 ]
                destination[ i + 3 ] = rgb_array[ i + 3 ]

        else:
            # noinspection SpellCheckingInspection
            for i in prange(0, l, 3, schedule = SCHEDULE, num_threads = THREADS):
                    r = &rgb_array[ i     ]
                    g = &rgb_array[ i + 1 ]
                    b = &rgb_array[ i + 2 ]
                    destination[ i    ] = <unsigned char>255 - r[ 0 ]
                    destination[ i + 1] = <unsigned char>255 - g[ 0 ]
                    destination[ i + 2] = <unsigned char>255 - b[ 0 ]


    return numpy.ndarray(shape=l, buffer=destination, dtype=uint8)



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef inline void wave_inplace_c(
        unsigned char [:, :, :] rgb_array,
        float rad,
        int size = 5):
    """
    Wave effect (inplace)
    
    Create a wave effect to the game display or surface (inplace)
    Compatible with 24-bit surface 

    e.g:
    wave_inplace_c(pixels3d(surface), 8 * math.pi/180.0 + frame_number, 5)
    wave_inplace_c(pixels3d(surface), x * math.pi/180.0, 5)

    Parameters
    ----------
    
    rgb_array : 
        numpy.ndarray 3d array shape (w, h, 3) type uint8 of RGB format pixels
        referencing the game display or surface containing all the pixel values.
        Any change to this array will modify the game display or given surface directly
       
    rad : 
        float; angle in rad to rotate over time, default 0.139 
        
    size : 
        int; Number of sub-surfaces, default is 5 
        
    Returns
    -------
    void
    """

    cdef Py_ssize_t w, h
    w, h = rgb_array.shape[:2]

    cdef:
        unsigned char [::1, :, :] rgb = \
                numpy.asarray(rgb_array, order='F')
        int x, y, x_pos, y_pos, xx, yy
        int i=0, j=0
        float c1 = <float>1.0 / <float>(size * size)
        unsigned int w_1 = <unsigned int>w - 1
        unsigned int h_1 = <unsigned int>h - 1

    with nogil:

        for y in prange(0, h_1 - size, size, schedule=SCHEDULE, num_threads=THREADS):

            y_pos = y + size + <int>(<float>sin(rad + <float>y * c1) * <float>size)

            for x in prange(0, w_1 - size, size, schedule=SCHEDULE, num_threads=THREADS):

                x_pos = x + size + <int> (<float>sin(rad + <float> x * c1) * <float> size)

                for j in range(0, size + 1):
                    yy = y_pos + j
                    if yy > h_1:
                        # yy = h_1
                        continue
                    if yy < 0:
                        # yy = 0
                        continue

                    for i in range(0, size + 1):
                        xx = x_pos + i
                        if xx > w_1:
                            # xx = w_1
                            continue
                        if xx < 0:
                            # xx = 0
                            continue

                        rgb_array[xx, yy, 0] = rgb[x + i, y + j, 0]
                        rgb_array[xx, yy, 1] = rgb[x + i, y + j, 1]
                        rgb_array[xx, yy, 2] = rgb[x + i, y + j, 2]


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef inline void wave32_c(
        unsigned char [:, :, ::1] rgba_array,
        float rad,
        int size):

    """
    Create a wave effect (inplace)
        
    Method fully compatible with 32-bit SDL surface with layer alpha.
    The layer alpha pixels will also be displace by the wave effect.    
    The Array (rgba_array) must be a numpy array shape (w, h, 4) type uint8 
    containing RGB pixels   
    
    e.g:
    image = pygame.image.load("../Assets/px.png").convert_alpha()
    wave32(image, angle * math.pi / 180, 10)
    
    Parameters
    ----------
    rgba_array : 
        numpy.ndarray shape (w, h, 4) of type uint8 containing all the RGBA 
        values and represent the image pixels of the game display surface. This array is referencing 
        the game display and any changes to the array will affect the surface directly. 
        
    rad : 
        float; angle in radian to rotate the wave over time
        
    size : 
        int; Number of sub-surfaces

    Returns
    -------
    void

    """

    cdef:
        Py_ssize_t w, h
        Py_ssize_t bit_size

    w, h, bit_size = rgba_array.shape[:3]


    cdef:
        unsigned char [::1, :, :] rgb = \
                numpy.asarray(rgba_array, order='F')

        int x, y, x_pos, y_pos, xx, yy
        unsigned int i=0, j=0
        float c1 = <float>1.0 / <float>(size * size)
        unsigned int w_1 = <unsigned int>w - 1
        unsigned int h_1 = <unsigned int>h - 1
        unsigned int yj, xi


    with nogil:

        for x in prange(0, w_1 - size, size, schedule = SCHEDULE, num_threads = THREADS):
            x_pos = x + size + <int> (<float>sin(rad + <float> x * c1) * <float> size)


            for y in prange(0, h_1 - size, size, schedule=SCHEDULE, num_threads=THREADS):
                y_pos = y + size + <int>(<float>sin(rad + <float>y * c1) * <float>size)

                for i in range(0, size + 1):

                    xx = x_pos + i
                    if xx > w_1:
                        # xx = w_1
                        continue
                    if xx < 0:
                        # xx = 0
                        continue
                    xi = x + i

                    for j in range(0, size + 1):
                        yy = y_pos + j
                        if yy > h_1:
                            # yy = h_1
                            continue
                        if yy < 0:
                            # yy = 0
                            continue
                        yj = y + j

                        # if rgb[ xi, yj, 3 ] > 0:

                        rgba_array[xx, yy, 0] = rgb[xi, yj, 0]
                        rgba_array[xx, yy, 1] = rgb[xi, yj, 1]
                        rgba_array[xx, yy, 2] = rgb[xi, yj, 2]
                        rgba_array[ xx, yy, 3 ] = rgb[xi, yj, 3 ]
                        # else:
                        #     rgba_array[ xx, yy, 3 ] = 0



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef inline void wave_static_c(
        unsigned char [:, :, :] rgb_array,
        const unsigned char [:, :, :] rgb_array_cp,
        const float rad,
        const int size):

    """
    Create a wave effect to a static game display or image

    This function is different to the wave method as a copy of the 
    static background is passed to the function as an argument `rgb_array` to 
    improve the overall performances.
    
    The arguments rgb_array and rgb_array_cp are numpy arrays or a memoryviewslice, shape (w, h, 3) 
    type uint8 containing RGB pixels. 
   
    e.g:
    background = pygame.image.load('../Assets/px.png').convert(24)
    background = pygame.transform.smoothscale(background, (800, 600))
    background_cp = background.copy()
    wave_static_c(pixels3d(background), pixels3d(background_cp), FRAME * math.pi/180 , 5)
    SCREEN.blit(background, (0, 0))

    :param rgb_array: 
        numpy.ndarray shape (w, h, 3) of type uint8 containing all the 
        RGB pixels values and represent the image pixels of the game display surface.
        This array is referencing the game display and any changes to the array will
        alter the surface directly. 
    
    :param rgb_array_cp: 
        numpy.ndarray shape (w, h, 3) type uint8 copy of rgb_array
    
    :param rad: 
        float; angle in rad to rotate over time.
        
    :param size: 
        int; Number of sub-surfaces.
        
    :return: 
        void
    """

    cdef:
        Py_ssize_t w, h, ww, hh

    w, h = rgb_array.shape[:2]
    ww, hh = rgb_array_cp.shape[:2]

    if w!=ww or h!=hh:
        raise ValueError(
            "\nBoth the surface and the array must have the same sizes/dimensions")

    cdef:
        int x, y, x_pos, y_pos, xx, yy
        unsigned int i=0, j=0
        float c1 = <float>1.0 / <float>(size * size)
        int w_1 = <int>w - 1
        int h_1 = <int>h - 1

    with nogil:

        for y in prange(0, h_1 - size, size, schedule=SCHEDULE, num_threads=THREADS):

            y_pos = y + size + <int>(<float>sin(rad + <float>y * c1) * <float>size)

            for x in prange(0, w_1 - size, size, schedule=SCHEDULE, num_threads=THREADS):

                x_pos = x + size + <int> (<float>sin(rad + <float> x * c1) * <float> size)

                for j in range(0, size + 1):
                    yy = y_pos + j
                    if yy > h_1:
                        # yy = h_1
                        continue
                    if yy < 0:
                        # yy = 0
                        continue

                    for i in range(0, size + 1):
                        xx = x_pos + i
                        if xx > w_1:
                            # xx = w_1
                            continue
                        if xx < 0:
                            # xx = 0
                            continue

                        rgb_array[xx, yy, 0] = rgb_array_cp[x + i, y + j, 0]
                        rgb_array[xx, yy, 1] = rgb_array_cp[x + i, y + j, 1]
                        rgb_array[xx, yy, 2] = rgb_array_cp[x + i, y + j, 2]


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef inline void swirl_c(
        const Py_ssize_t w,
        const Py_ssize_t h,
        unsigned char [:, :, :] rgb_array,
        const unsigned char [:, :, :] rgb_array_cp,
        float degrees
) nogil:

    """
    Apply Swirl Effect to an Image (Inplace)

    The swirl effect distorts an image, creating a spiraling or twisting appearance 
    that draws attention and adds movement or dynamism to the design. It can be used 
    creatively in various contexts, such as digital art, social media graphics, or advertising.

    This effect works with 24-bit image formats but is not compatible with 32-bit formats 
    due to the presence of an alpha channel. For 32-bit images with alpha channels, the 
    alpha layer will remain unchanged during the transformation, causing the alpha to 
    "bleed" into the effect. If this is undesirable, convert the image to a 24-bit format 
    before applying the effect.

    For 32-bit images with an alpha channel, consider using the `swirl32` method instead, 
    which is specifically designed for such images.

    This algorithm uses a lookup table of cosine and sine values to apply the swirl effect.

    Both the `rgb_array` and `rgb_array_cp` must be numpy arrays of shape (w, h, 3) 
    containing RGB pixel values.

    Example usage:
        background = pygame.image.load("../Assets/background.jpg").convert(24)
        background = pygame.transform.smoothscale(background, (WIDTH, HEIGHT))
        background_cp = background.copy()

        # In the game loop:
        swirl_c(WIDTH, HEIGHT, pixels3d(background_cp), pixels3d(background), angle)
        SCREEN.blit(background_cp, (0, 0))

    Parameters
    ----------
    w : int
        The width of the source image array (in pixels).

    h : int
        The height of the source image array (in pixels).

    rgb_array : numpy.ndarray
        A numpy array with shape (w, h, 3) of type uint8, referencing the game 
        display or surface containing the RGB color values. Any modifications to this array 
        will directly affect the game display or surface.

    rgb_array_cp : numpy.ndarray
        A copy of `rgb_array` used to perform the in-place transformation. This copy is 
        modified by the swirl effect while the original array remains unchanged.

    degrees : float
        The swirl angle in degrees. This value controls the degree of rotation for the effect.

    Returns
    -------
    None
        This function modifies the `rgb_array_cp` in place to apply the swirl effect.
    """


    cdef:
        int i, j, angle
        int diffx, diffy
        float columns, rows, r, di, dj
        float * c1
        float * c2
        float r_max

    columns = <float>0.5 * (<float>w - <float>1.0)
    rows    = <float>0.5 * (<float>h - <float>1.0)

    r_max = <float>1.0/<float>sqrt(columns * columns + rows * rows)

    for j in prange(h, schedule=SCHEDULE, num_threads=THREADS):
        dj = <float> j - rows

        for i in range(w):
            di = <float>i - columns

            r = <float>sqrt(di * di + dj * dj) * r_max
            angle = <int>(degrees * r % 360)

            c1 = &COS_TABLE[angle]
            c2 = &SIN_TABLE[angle]

            diffx = <int>(di * c1[0] - dj * c2[0] + columns)
            diffy = <int>(di * c2[0] + dj * c1[0] + rows)

            if (diffx >-1) and (diffx < w) and \
               (diffy >-1) and (diffy < h):
                rgb_array[i, j, 0] = rgb_array_cp[diffx, diffy, 0]
                rgb_array[i, j, 1] = rgb_array_cp[diffx, diffy, 1]
                rgb_array[i, j, 2] = rgb_array_cp[diffx, diffy, 2]

            else:
                rgb_array[ i, j, 0 ] = 0
                rgb_array[ i, j, 1 ] = 0
                rgb_array[ i, j, 2 ] = 0




@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef inline void swirl32_c(
        Py_ssize_t w,
        Py_ssize_t h,
        unsigned char [:, :, ::1] rgb_array_,
        const unsigned char [:, :, :] rgb,
        const float degrees
):
    """
    Swirl an image (inplace)
    
    Compatible with 32-bit images.
    
    The swirl effect is a visual distortion that creates
    a spiraling appearance in an image or graphic. 
    This effect can draw attention to specific areas of a design
    and add a sense of movement or dynamism. It can be used creatively
    in various contexts, from social media graphics to advertising
    and digital art.
    
    both arrays (rgb_array, rgb) must be numpy arrays shape (w, h, 4) containing RGBA pixels.
    
    e.g:
    swirl32_c(w, h, rgb_array_, rgb, 30)
    
    :param w:
        array (rgb_array) width 
        
    :param h:
        array (rgb_array) height 
        
    :param rgb_array_: 
        numpy.ndarray shape (h, w, 4) type uint8 referencing the game display 
        or surface containing all the RGBA color values.Any change to this array will modify the game 
        display or given surface directly
        
    :param rgb:
        numpy.ndarray shape (h, w, 4) type uint8 copy of rgb_array_
         
    :param degrees: 
        float; swirl angle in degrees
    
    :return: 
        void
        
    """

    cdef:
        int i, j, angle
        int diffx, diffy
        float columns, rows, r, di, dj
        float * c1
        float * c2
        float r_max
        unsigned char * p1

    columns = <float>0.5 * (<float>w - <float>1.0)
    rows    = <float>0.5 * (<float>h - <float>1.0)

    r_max = <float>1.0/<float>sqrt(columns * columns + rows * rows)

    # inversion of w, h
    # The surface size returns w,h and the rgb_array_ returns h,w
    w, h = h, w

    with nogil:
        for i in prange(w, schedule=SCHEDULE, num_threads=THREADS):
            di = <float>i - columns

            for j in range(h):

                dj = <float> j - rows

                r = <float>sqrt(di * di + dj * dj) * r_max
                angle = <int>(degrees * r % 360)

                # angle = <int>fmodf(degrees * r, 360.0)

                c1 = &COS_TABLE[angle]
                c2 = &SIN_TABLE[angle]
                # c2 = <float>sqrt(1- c1[0] ** 2)
                # diffx = <int> (di * c1[ 0 ] - dj * c2 + columns)
                # diffy = <int> (di * c2 + dj * c1[ 0 ] + rows)

                diffx = <int>(di * c1[0] - dj * c2[0] + columns)
                diffy = <int>(di * c2[0] + dj * c1[0] + rows)

                p1 = &rgb_array_[ i, j, 0 ]

                if (diffx > -1) and (diffx < w) and \
                        (diffy > -1) and (diffy < h):

                    p1[ 0 ] = rgb[ diffx, diffy, 0 ]
                    (p1 + 1)[ 0 ] = rgb[ diffx, diffy, 1 ]
                    (p1 + 2)[ 0 ] = rgb[ diffx, diffy, 2 ]
                    (p1 + 3)[ 0 ] = rgb[ diffx, diffy, 3 ]

                else:
                    p1[ 0 ] = <unsigned char>0
                    (p1 + 1)[ 0 ] = <unsigned char>0
                    (p1 + 2)[ 0 ] = <unsigned char>0
                    (p1 + 3)[ 0 ] = <unsigned char>0





@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef inline void swirlf_c(
    const Py_ssize_t w,
    const Py_ssize_t h,
    unsigned char [:, :, :] rgb_array,
    const unsigned char [::1, :, :] rgb,
    const float degrees
)nogil:

    """
    
    Swirl an image (inplace) floating point accuracy 
    
    This algorithm DO NOT use COS and SIN tables, it determines the angles with 
    floating point accuracy instead.
    
    compatible with 24-bit image format only
    
    The swirl effect is a visual distortion that creates
    a spiraling appearance in an image or graphic. 
    This effect can draw attention to specific areas of a design
    and add a sense of movement or dynamism. It can be used creatively
    in various contexts, from social media graphics to advertising
    and digital art.
    
    The Array (rgb_array_) must be a numpy array shape (w, h, 3) containing RGB pixels
    
    e.g:
     swirlf_c(surface_, rgb,, angle)
    
    :param w:
        integer; rgb_array_ width in pixels 
        
    :param h:
        integer; rgb_array_ height in pixels 
        
    :param rgb_array: 
        numpy.ndarray shape (w, h, 3) type uint8 referencing the game display or 
        surface containing all the RGB color values. Any changes to this array will modify directly the 
        game display or surface.
    
    :param rgb:
        numpy.ndarray copy of rgb_array_ (fortran)  
        
    :param degrees: 
        float; swirl angle in degrees
        
    :return: 
        void
        
    """


    cdef:
        int i, j, diffx, diffy
        float columns, rows, r, di, dj, c1, c2, angle

        float rad = degrees * DEG_TO_RAD
        float r_max

    columns = <float>0.5 * (w - <float>1.0)
    rows    = <float>0.5 * (h - <float>1.0)
    r_max   = <float>1.0/<float>sqrt(columns * columns + rows * rows)

    for j in prange(h, schedule=SCHEDULE, num_threads=THREADS):
        dj = <float> j - rows
        for i in range(w):
            di = <float>i - columns

            r = <float>sqrt(di * di + dj * dj)
            angle = <float>(rad * r * r_max)

            c1 = <float>cos(angle)
            c2 = <float>sin(angle)
            diffx = <int>(di * c1 - dj * c2 + columns)
            diffy = <int>(di * c2 + dj * c1 + rows)

            if (diffx >-1) and (diffx < w) and \
               (diffy >-1) and (diffy < h):
                rgb_array[i, j, 0], rgb_array[i, j, 1],\
                    rgb_array[i, j, 2] = rgb[diffx, diffy, 0], \
                                          rgb[diffx, diffy, 1], rgb[diffx, diffy, 2]

            else:
                rgb_array[ i, j, 0 ] = 0
                rgb_array[ i, j, 1 ] = 0
                rgb_array[ i, j, 2 ] = 0



# Todo show examples and method name is incorrect in the example
@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef inline void plasma_inplace_c(
        unsigned char [:, :, :] rgb_array_,
        int frame,
        float hue_=<float>1.0/<float>6.0,
        float sat_=<float>1.0/<float>6.0,
        float value_=<float>1.0/<float>8.0,
        float a_=<float>ONE_255,
        float b_=<float>ONE_TWELVE,
        float c_=<float>ONE_TWELVE):
    """
    CREATE A BASIC PLASMA EFFECT ON THE TOP OF A PYGAME SURFACE

    The Array (rgb_array) must be a numpy array shape (w, h, 3) containing RGB pixels,
     please refer to pygame function pixels3d or array3d to convert an image into a
     3d array (library surfarray)

    e.g:
    plasma_config(surface, frame_number)

    :param a_: 
        float; default value 1.0/255.0 control the plasma equation
        
    :param b_: 
        float; default value 1.0/12.0 control the plasma equation
        
    :param c_: 
        float; default value 1.0/12.0 control the plasma equation
        
    :param value_: 
        float; default value 1.0/8.0 value factor
        
    :param sat_: 
        float; default value 1.0/6.0 saturation value
        
    :param hue_: 
        float; default value 1.0/6.0 hue value factor
        
    :param rgb_array_: 
        numpy.ndarray shape( w, h, 3) containing all the RGB color values
        
    :param frame: 
        integer; Variable that need to change over time
        
    :return: 
        void
        
    """

    cdef Py_ssize_t width, height
    width, height = rgb_array_.shape[:2]

    cdef:
        float xx, yy, t
        float h, s, v
        int i = 0, x, y
        float f, p, q, t_
        float hue, r, g, b
        unsigned char *rr
        unsigned char *gg
        unsigned char *bb

    t = <float>frame

    with nogil:
        for x in prange(width, schedule=SCHEDULE, num_threads=THREADS):
            xx = <float> x * <float> 0.5
            for y in range(height):
                yy = <float>y * <float>0.5

                hue = <float>4.0 + <float>sin((xx * <float>0.5 + yy * <float>0.5) * a_) \
                      + <float>sin((xx * b_ + t) * c_)

                h, s, v = hue * hue_, hue * sat_, hue * value_

                i = <int>(h * <float>6.0)
                f = (h * <float>6.0) - i
                p = v*(<float>1.0 - s)
                q = v*(<float>1.0 - s * f)
                t_ = v*(<float>1.0 - s * (<float>1.0 - f))
                i = i % 6

                if i == 0:
                    r, g, b =  v, t_, p
                if i == 1:
                     r, g, b = q, v, p
                if i == 2:
                     r, g, b = p, v, t_
                if i == 3:
                     r, g, b = p, q, v
                if i == 4:
                     r, g, b = t_, p, v
                if i == 5:
                     r, g, b = v, p, q

                if s == 0.0:
                     r, g, b = v, v, v

                rr = &rgb_array_[x, y, 0]
                gg = &rgb_array_[x, y, 1]
                bb = &rgb_array_[x, y, 2]

                if rr[0] < 256:
                    rr[0] = <unsigned char>(min(rr[0] + r * <float>128.0, <float>255.0))

                if gg[0] < 256:
                    gg[0] = <unsigned char>(min(gg[0] + g * <float>128.0, <float>255.0))

                if bb[0] < 256:
                    bb[0] = <unsigned char>(min(bb[0] + b * <float>128.0, <float>255.0))



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef inline void plasma_c(surface_, float frame, unsigned int [::1] palette_):

    """
    Apply Plasma Effect to a Surface (Inplace)

    This function creates a dynamic plasma effect on a pygame surface. The effect is generated 
    using a palette of colors, and it evolves over time based on the `frame` value. The plasma 
    effect gives a fluid, flowing appearance, which can be used creatively for backgrounds or 
    visual effects in games or applications.

    The effect is applied directly to the surface, modifying it in place. The surface can be either 
    24-bit or 32-bit, with or without an alpha channel.

    Example usage:
        plasma_c(surface, frame_number, palette)

    Parameters
    ----------
    surface_ : pygame.Surface
        A pygame surface compatible with 24-bit or 32-bit formats. The plasma effect is applied 
        directly to this surface, modifying its pixels in place.

    frame : float
        The current frame number, which drives the evolution of the plasma effect. This value 
        controls the shifting patterns in the plasma animation.

    palette_ : numpy.ndarray (1D)
        A 1D array containing the color palette (as integers). This array represents the colors 
        used to create the plasma effect. The colors are applied cyclically to the plasma pattern.

    Returns
    -------
    None
        This function modifies the `surface_` in place by applying the plasma effect to it.
    """

    cdef Py_ssize_t width, height
    width, height = surface_.get_size()

    cdef unsigned char [:, :, :] rgb_array
    try:
        rgb_array = surface_.get_view('3')

    except Exception as e:
        raise ValueError("\nCannot reference source pixels into a 3d array.\n %s " % e)

    cdef:
        int x, y, ii
        float color_
        float w2 = <float>width * <float>HALF
        float h2 = <float>height * <float>HALF
        Py_ssize_t length = len(palette_)

    with nogil:

        for y in prange(height, schedule=SCHEDULE, num_threads=THREADS):
            for x in range(width):

                color_ = \
                    <float>128.0 + \
                    <float>128.0 * <float>sin(x * <float>ONE_255 + frame) \
                    + <float>128.0 + \
                    <float>128.0 * <float>sin(frame * <float>ONE_32 + y * <float>ONE_64) \
                    + <float>128.0 \
                    + <float>128.0 * <float>sin(
                        sqrt((x - w2) * (x - w2) + (y - h2) * (y - h2)) * <float>ONE_255) \
                    + <float>128.0 + <float>128.0 * <float>sin(
                    <float>sqrt(x * x + y * y + frame) * <float>ONE_64)

                ii = palette_[<int>fmin(color_ / <float>8.0, length)]

                rgb_array[x, y, 0] = (ii >> 16) & <unsigned char>255
                rgb_array[x, y, 1] = (ii >> 8) & <unsigned char>255
                rgb_array[x, y, 2]= ii & <unsigned char>255






@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef void hsl_c(unsigned char [:, :, :] rgb_array, const float shift):

    """
    Rotate the Hue of a 3D Array (HSL Transformation)
    
    This function applies a hue rotation directly to a 3D array of pixel data. 
    The hue rotation is based on the HSL (Hue, Saturation, Lightness) color model, 
    which provides an intuitive way to describe and manipulate colors according to human perception. 
    
    The hue value (`shift`) controls the rotation of the colors, where `shift`
     corresponds to a value between 0.0 and 1.0, mapping to a 0° to 360° hue shift in the color space.
    
    Example usage:
        # For 24-bit image
        image = pygame.image.load('../Assets/px.png').convert(24)
        hsl_c(bgr_array, 0.2)
        
        # For 32-bit image with alpha
        image = pygame.image.load('../Assets/px.png').convert_alpha()
        hsl_c(bgr_array, 0.2)
    
    Parameters
    ----------
    rgb_array : numpy.ndarray
        A 3D array (shape: w, h, n) of uint8 type containing RGB(A) pixel values.
        This function supports any pixel format (e.g., RGB, BGR, BGRA, etc.), 
        and modifies the input array in place.
    
    shift : float
        A value in the range [0.0, 1.0], representing the amount to shift the hue. 
        A value of 0.0 corresponds to no shift (0°), and a value of 1.0 corresponds to a full 360° hue rotation.
    
    Returns
    -------
    None
        This function modifies the input `rgb_array` in place by rotating the hue of each pixel.
    """


    cdef Py_ssize_t w, h
    w, h = rgb_array.shape[:2]

    cdef:
        int i=0, j=0
        hsl hsl_
        rgb rgb_
        float h_
        unsigned char *r
        unsigned char *g
        unsigned char *b
        unsigned int sum_rgb


    with nogil:
        for j in prange(h, schedule=SCHEDULE, num_threads=THREADS):
            for i in range(w):

                r = &rgb_array[i, j, 0]
                g = &rgb_array[i, j, 1]
                b = &rgb_array[i, j, 2]

                sum_rgb = r[ 0 ] + g[ 0 ] + b[ 0 ]
                # hsl transform of null RGB (solid black) or solid white remain unchanged
                if sum_rgb == 0 or sum_rgb == 765:
                    continue

                hsl_ = struct_rgb_to_hsl(
                    r[0] * <float>ONE_255, g[0] *
                    <float>ONE_255, b[0] * <float>ONE_255)

                h_ = hsl_.h + shift
                rgb_ = struct_hsl_to_rgb(h_, hsl_.s, hsl_.l)

                r[0] = <unsigned char>(rgb_.r * <float>255.0)
                g[0] = <unsigned char>(rgb_.g * <float>255.0)
                b[0] = <unsigned char>(rgb_.b * <float>255.0)



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef inline void hsl1d_c(unsigned char [::1] bgr_array, const float shift, bint format_32=False):
    """
    HSL (C buffer) uint8 data types RGB(A) format (inplace)
    
    HSL (Hue, Saturation, Lightness) is another color representation 
    model used in digital imaging and graphics. It defines colors in 
    terms of their hue, saturation, and lightness, offering an intuitive 
    way to describe and manipulate colors based on human perception.
    
    This method will works with other buffer format such as BGR, BGRA
    
    Hue value (shift) must be in range [0.0 ...1.0] corresponding to 0.0 - 360.0 degrees rotation
    
    e.g:
    # 24-bit
    image = pygame.image.load('../Assets/px.png').convert(24)
    hsl1d_c(array3d, 0.2)
    
    # 32-bit
    image = pygame.image.load('../Assets/px.png').convert_alpha()
    hsl1d_c(array3d, 0.2, format_32=True)
    
    Parameters
    ----------
    bgr_array: 
        numpy.ndarray 1d array, memoryviewslice uint8 data type containing 
        BGR(A) pixel format, works also with other format pixel (BGR, BGRA etc)
         
    shift: 
        float; Hue value in range [0.0 ... 1.0] corresponding to 0.0 - 360.0 degrees rotation
        
    format_32: 
        bool True | for 'BGR' buffer type (24-bit) or False 'BGRA' (32-bit) 

    Returns
    -------
    void

    """

    cdef unsigned int l = bgr_array.shape[0]

    cdef:
        int i=0
        unsigned char tmp
        unsigned char * r
        unsigned char * g
        unsigned char * b
        unsigned short int bit = 3
        unsigned int sum_rgb
        hsl hsl_
        rgb rgb_
        float h_

    if format_32:
        bit = 4


    with nogil:
        # noinspection SpellCheckingInspection
        for i in prange(0, l, bit, schedule=SCHEDULE, num_threads=THREADS):

            r = &bgr_array[ i ]
            g = &bgr_array[ i + 1 ]
            b = &bgr_array[ i + 2 ]

            sum_rgb = r[ 0 ] + g[ 0 ] + b[ 0 ]
            # hsl transform of null RGB (solid black) or solid white remain unchanged
            if sum_rgb == 0 or sum_rgb == 765:
                continue

            hsl_ = struct_rgb_to_hsl(
                r[ 0 ] * <float> ONE_255, g[ 0 ] *
                <float> ONE_255, b[ 0 ] * <float> ONE_255)

            h_ = hsl_.h + shift
            rgb_ = struct_hsl_to_rgb(h_, hsl_.s, hsl_.l)

            r[ 0 ] = <unsigned char> (rgb_.r * <float> 255.0)
            g[ 0 ] = <unsigned char> (rgb_.g * <float> 255.0)
            b[ 0 ] = <unsigned char> (rgb_.b * <float> 255.0)




@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef np.ndarray[np.uint8_t, ndim=1] hsl1d_cp_c(
        const unsigned char [::1] bgr_array,
        const float shift,
        bint format_32=False
):
    """
    Rotate hue (HSL) directly to a C-buffer (return a copy)

    HSL (C buffer) uint8 data types RGB(A) format 
    
    This method will works with other buffer format such as BGR, BGRA
    
    HSL (Hue, Saturation, Lightness) is another color representation 
    model used in digital imaging and graphics. It defines colors in 
    terms of their hue, saturation, and lightness, offering an intuitive 
    way to describe and manipulate colors based on human perception.
    
    e.g:
    # 24-bit
    image = pygame.image.load('../Assets/px.png').convert(24)
    hsl1d_cp_c(array3d, 0.2)
    
    # 32-bit
    image = pygame.image.load('../Assets/px.png').convert_alpha()
    hsl1d_cp_c(array3d, 0.2, format_32=True)
    
    Hue value (shift) must be in range [0.0 ...1.0] corresponding to 0.0 - 360.0 degrees rotation
    
    Parameters
    ----------
    bgr_array: 
        numpy.ndarray 1d array, memoryviewslice uint8 data type containing 
        BGR(A) pixel format, works also with other format pixel (BGR, BGRA etc)
        
    shift:
        float; float value in range [0.0 ... 1.0] corresponding to 0.0 - 360.0 degrees rotation
        
    format_32: 
        bool True | for 'BGR' buffer type (24-bit) or False 'BGRA' (32-bit) 
    
    Returns
    -------
    numpy.ndarray 1d array type uint8 new array containing pixels with rotated hue

    """

    cdef unsigned int l = bgr_array.shape[0]

    cdef:
        int i=0
        unsigned char tmp
        unsigned char [::1] destination = numpy.empty(l, dtype=uint8)
        const unsigned char * r
        const unsigned char * g
        const unsigned char * b
        unsigned short int bit = 3
        unsigned int sum_rgb
        hsl hsl_
        rgb rgb_
        float h_

    if format_32:
        bit = 4


    with nogil:
        # noinspection SpellCheckingInspection
        for i in prange(0, l, bit, schedule=SCHEDULE, num_threads=THREADS):

            r = &bgr_array[ i ]
            g = &bgr_array[ i + 1 ]
            b = &bgr_array[ i + 2 ]

            sum_rgb = r[ 0 ] + g[ 0 ] + b[ 0 ]
            # hsl transform of null RGB (solid black) or solid white remain unchanged
            if sum_rgb == 0 or sum_rgb == 765:
                destination[ i ] = r[ 0 ]
                destination[ i + 1 ] = g[ 0 ]
                destination[ i + 2 ] = b[ 0 ]
                if format_32:
                    destination[ i + 3 ] = bgr_array[ i + 3 ]
                continue

            hsl_ = struct_rgb_to_hsl(
                r[ 0 ] * <float> ONE_255, g[ 0 ] *
                <float> ONE_255, b[ 0 ] * <float> ONE_255)

            h_ = hsl_.h + shift
            rgb_ = struct_hsl_to_rgb(h_, hsl_.s, hsl_.l)

            destination[ i     ] = <unsigned char> (rgb_.r * <float> 255.0)
            destination[ i + 1 ] = <unsigned char> (rgb_.g * <float> 255.0)
            destination[ i + 2 ] = <unsigned char> (rgb_.b * <float> 255.0)

            if format_32:
                destination[ i + 3 ] = bgr_array[ i + 3 ]

    return numpy.ndarray(shape=l, buffer=destination, dtype=uint8)



# ----------------------- HSV

@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef inline void hsv3d_c(unsigned char [:, :, :] rgb_array, const float shift_):

    """
    Rotate the hue directly from a 3d array (HSV conversion method)

    HSV (Hue, Saturation, Value) is a color model similar to HSL (Hue, Saturation, Lightness)
    but with some differences in how it represents and manipulates colors. 
    It’s often used in graphics software and computer vision applications for its 
    simplicity in specifying and adjusting color attributes.
    
    New hue value. Must be between [0.0 ... 1.0] corresponding to 0.0 - 360.0 degrees 
     (e.g 0.5 = 180 degrees)
    
    e.g:
    #Rotate the hue 72 degrees
    hsv3d_c(bgr_array, 0.2)

    :param rgb_array: 
        numpy.ndarray of shape(w, h, 3) of unsigned char, rgb values
        
    :param shift_: 
        float; Hue value in range [0.0 ... 1.0] corresponding to 0.0 - 360.0 degrees rotation. 
        New hue value
         
    :return: 
        void
        
    """

    cdef Py_ssize_t w, h
    w, h = rgb_array.shape[:2]

    cdef:
        int i=0, j=0
        hsv hsv_
        rgb rgb_
        float h_
        unsigned char *r
        unsigned char *g
        unsigned char *b
        unsigned int sum_rgb

    with nogil:
        for j in prange(h, schedule=SCHEDULE, num_threads=THREADS):
            for i in range(w):

                # Get the pixel color
                r = &rgb_array[i, j, 0]
                g = &rgb_array[i, j, 1]
                b = &rgb_array[i, j, 2]

                sum_rgb = r[ 0 ] + g[ 0 ] + b[ 0 ]
                # hsl transform of null RGB (solid black) or solid white remain unchanged
                if sum_rgb == 0 or sum_rgb == 765:
                    continue

                # Get the current hue values
                hsv_ = struct_rgb_to_hsv(
                    r[0] * <float>ONE_255, g[0] *
                    <float>ONE_255, b[0] * <float>ONE_255)

                # Rotate the hue
                h_ = hsv_.h + shift_

                # Conversion HSV to RGB
                rgb_ = struct_hsv_to_rgb(h_, hsv_.s, hsv_.v)

                # Change the pixel with new hue value.
                r[0] = <unsigned char>(rgb_.r * <float>255.0)
                g[0] = <unsigned char>(rgb_.g * <float>255.0)
                b[0] = <unsigned char>(rgb_.b * <float>255.0)


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef inline void hsv1d_c(
        unsigned char [::1] bgr_array, const float shift, bint format_32=False):
    """
     Rotate hue 1d array
    
    Rotate the hue directly from a C-buffer (1d array uint8 data types RGB(A) format)
     Changes apply inplace
    
    This method will works with other buffer format such as BGR, BGRA

    HSV (Hue, Saturation, Value) is a color model similar to HSL (Hue, Saturation, Lightness)
    but with some differences in how it represents and manipulates colors. 
    It’s often used in graphics software and computer vision applications for its 
    simplicity in specifying and adjusting color attributes.
    
    e.g 
    #compatible with 32 bits images 
    image = pygame.image.load('../Assets/px.png').convert_alpha()
    hsv1d_c(image.get_buffer(), angle/36.0, True)
    
    #compatible with 24 bits images 
    image = pygame.image.load('../Assets/px.png').convert(24)
    hsv1d_c(image.get_buffer(), angle/36.0, False) 

    Parameters
    ----------
    bgr_array: 
        numpy.ndarray 1d array, memoryviewslice uint8 data type containing 
        BGR(A) pixel format, works also with other format pixel (BGR, BGRA etc)
         
    shift:
        float; Hue value in range [0.0 ... 1.0]
        
    format_32:
        bool True | for 'BGR' buffer type (24-bit) or False 'BGRA' (32-bit) 

    Returns
    -------
    void

    """

    cdef unsigned int l = bgr_array.shape[0]

    cdef:
        int i=0
        unsigned char tmp
        unsigned char * r
        unsigned char * g
        unsigned char * b
        unsigned short int bit = 3
        hsv hsv_
        rgb rgb_
        float h_
        unsigned int sum_rgb

    if format_32:
        bit = 4


    with nogil:
        # noinspection SpellCheckingInspection
        for i in prange(0, l, bit, schedule=SCHEDULE, num_threads=THREADS):

            # get the pixel color
            r = &bgr_array[ i ]
            g = &bgr_array[ i + 1 ]
            b = &bgr_array[ i + 2 ]

            sum_rgb = r[ 0 ] + g[ 0 ] + b[ 0 ]
            # hsl transform of null RGB (solid black) or solid white remain unchanged
            if sum_rgb == 0 or sum_rgb == 765:
                continue

            # Get the HSV value
            hsv_ = struct_rgb_to_hsv(
                r[ 0 ] * <float> ONE_255, g[ 0 ] *
                <float> ONE_255, b[ 0 ] * <float> ONE_255)

            # Rotate the hue
            h_ = hsv_.h + shift

            # Convert HSV to RGB
            rgb_ = struct_hsv_to_rgb(h_, hsv_.s, hsv_.v)

            # Set the pixel with new RGB values
            r[ 0 ] = <unsigned char> (rgb_.r * <float> 255.0)
            g[ 0 ] = <unsigned char> (rgb_.g * <float> 255.0)
            b[ 0 ] = <unsigned char> (rgb_.b * <float> 255.0)



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef np.ndarray[np.uint8_t, ndim=1] hsv1d_cp_c(
        const unsigned char [::1] bgr_array,
        const float shift,
        bint format_32=False):

    """
    Rotate hue of a C-buffer array (return a copy)
    
    HSV 1d array (C buffer) uint8 data types BGR(A) format 

    This method will works with other buffer format such as BGR, BGRA

    e.g:
    # 32-bit image
    image = pygame.image.load('../Assets/px.png').convert_alpha()
    arr = hsv1d_cp_c(image.get_buffer(), angle/360.0, format_32=True) 
    image = pygame.image.frombuffer(arr, (WIDTH, HEIGHT), "BGRA")
    
    # 24-bit image 
    image = pygame.image.load('../Assets/px.png').convert(24)
    arr = hsv1d_cp_c(image.get_buffer(), angle/360.0, format_32=False)
    

    Parameters
    ----------
    bgr_array: 
        numpy.ndarray 1d array, memoryviewslice uint8 data type containing 
        BGR(A) pixel format, works also with other format pixel (RGB, RGBA etc)
         
    shift: 
        float; Hue value in range [0 ... 1.0] corresponding to 0.0 - 360.0 degrees
         
    format_32: 
        bool True | for 'BGR' buffer type (24-bit) or False 'BGRA' (32-bit)
         

    Returns
    -------
    numpy.ndarray 1d array uint8 type equivalent to the input array with rotated hue

    """

    cdef unsigned int l = bgr_array.shape[0]

    cdef:
        int i=0
        unsigned char tmp
        unsigned char [::1] destination = numpy.empty(l, dtype=uint8)
        const unsigned char * r
        const unsigned char * g
        const unsigned char * b
        unsigned short int bit = 3
        unsigned int sum_rgb
        hsv hsv_
        rgb rgb_
        float h_

    if format_32:
        bit = 4

    with nogil:
        # noinspection SpellCheckingInspection
        for i in prange(0, l, bit, schedule=SCHEDULE, num_threads=THREADS):

            r = &bgr_array[ i ]
            g = &bgr_array[ i + 1 ]
            b = &bgr_array[ i + 2 ]

            sum_rgb = r[ 0 ] + g[ 0 ] + b[ 0 ]
            # hsl transform of null RGB (solid black) or solid white remain unchanged
            if sum_rgb == 0 or sum_rgb == 765:
                destination[ i ] = r[ 0 ]
                destination[ i + 1 ] = g[ 0 ]
                destination[ i + 2 ] = b[ 0 ]
                if format_32:
                    destination[ i + 3 ] = bgr_array[ i + 3 ]
                continue

            hsv_ = struct_rgb_to_hsv(
                r[ 0 ] * <float> ONE_255, g[ 0 ] *
                <float> ONE_255, b[ 0 ] * <float> ONE_255)

            h_ = hsv_.h + shift
            rgb_ = struct_hsv_to_rgb(h_, hsv_.s, hsv_.v)

            destination[ i     ] = <unsigned char> (rgb_.r * <float> 255.0)
            destination[ i + 1 ] = <unsigned char> (rgb_.g * <float> 255.0)
            destination[ i + 2 ] = <unsigned char> (rgb_.b * <float> 255.0)

            if format_32:
                destination[ i + 3 ] = bgr_array[ i + 3 ]

    return numpy.ndarray(shape=l, buffer=destination, dtype=uint8)






















@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef inline void brightness_c(
        unsigned char [:, :, :] rgb_array,
        float shift=0):
    """
    Adjust the brightness of an image or surface (inplace).

    This method allows you to modify the brightness of the provided game display or surface.
    The array (rgb_array) should be a numpy array with the shape (w, h, 3), representing RGB pixel values.
    The brightness is controlled by adjusting the intensity of all colors in the image.

    e.g:
    brightness_c(pixels3d(surface), 0.2)

    Parameters
    ----------
    rgb_array : 
        numpy.ndarray of shape (w, h, 3) with dtype uint8. The array references the game display 
        or surface containing the RGB pixel values. Any modifications to this array will directly 
        affect the game display or SDL surface.

    shift : 
        float; a value in the range [-1.0 ... 1.0], where 0.0 means no change to the brightness.
        A value of -1.0 results in the lowest brightness (darkest), while 1.0 represents the maximum brightness 
        (brightest) achievable.

    Returns
    -------
    void
        This method modifies the array in-place and does not return any value.
    """


    cdef Py_ssize_t width, height
    width, height = rgb_array.shape[:2]

    cdef:
        int i=0, j=0
        unsigned char *r
        unsigned char *g
        unsigned char *b
        float l
        hsl hsl_
        rgb rgb_


    with nogil:
        for j in prange(height, schedule=SCHEDULE, num_threads=THREADS):
            for i in range(width):

                r, g, b = &rgb_array[i, j, 0], &rgb_array[i, j, 1], &rgb_array[i, j, 2]

                hsl_ = struct_rgb_to_hsl(
                    r[0] * <float>ONE_255,
                    g[0] * <float>ONE_255,
                    b[0] * <float>ONE_255
                )# struct_rgb_to_hsl returns floats, range 0.0 ... 1.0

                # l = min((hsl_.l + shift), <float> 1.0)
                # l = max(l, <float> 0.0)


                l = hsl_.l + shift

                # force white pixel, we do not need to run
                # struct_hsl_to_rgb to convert hsl to rgb as we know that
                # the color will be white
                if l >= <float>1.0:
                    r[ 0 ] = <unsigned char> 255
                    g[ 0 ] = <unsigned char> 255
                    b[ 0 ] = <unsigned char> 255
                    continue

                # force black pixel, we do not need to run
                # struct_hsl_to_rgb to convert hsl to rgb as we know that
                # the color will be black
                if l <= <float>0:
                    r[ 0 ] = <unsigned char> 0
                    g[ 0 ] = <unsigned char> 0
                    b[ 0 ] = <unsigned char> 0
                    continue

                rgb_ = struct_hsl_to_rgb(hsl_.h, hsl_.s, l)

                r[0] = <unsigned char> (rgb_.r * <float>255.0)
                g[0] = <unsigned char> (rgb_.g * <float>255.0)
                b[0] = <unsigned char> (rgb_.b * <float>255.0)



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef inline object brightness_copy_c(
        unsigned char [:, :, :] rgb_array, float shift=0):

    """
    Brightness (return a copy)

    Return a 24-bit surface type with adjusted brightness.  

    The Array (rgb_array) must be a numpy array shape (w, h, 3) containing RGB pixels

    e.g:
    surface = brightness_copy_c(pixels3d(surface), 0.2)

    :param rgb_array: 
        numpy ndarray shape (w, h, 3) containing RGB pixels values
        
    :param shift: 
        float; values in range [-1.0 ... 1.0], 0.0 no change,
        -1 lowest brightness, +1 max brightness
        
    :return: 
        24-bit pygame surface (copy)
        
    """

    cdef:
        Py_ssize_t width, height
        unsigned int bit_size

    width, height, bit_size = rgb_array.shape[:3]

    cdef:
        int i=0, j=0
        unsigned char *r
        unsigned char *g
        unsigned char *b
        float l
        hsl hsl_
        rgb rgb_
        unsigned char [:, :, :] array_tmp = \
            numpy.empty((height, width, bit_size), dtype=numpy.uint8)

    with nogil:
        for j in prange(height, schedule=SCHEDULE, num_threads=THREADS):
            for i in range(width):

                r, g, b = \
                    &rgb_array[ i, j, 0 ],\
                    &rgb_array[ i, j, 1 ],\
                    &rgb_array[ i, j, 2 ]

                hsl_ = struct_rgb_to_hsl(
                    r[ 0 ] * <float> ONE_255,
                    g[ 0 ] * <float> ONE_255,
                    b[ 0 ] * <float> ONE_255
                )

                l = min((hsl_.l + shift), <float>1.0)
                l = max(l, <float>0.0)

                # investigate bug below causing black layer

                # l = hsl_.l + shift
                #
                # # force white pixel, we do not need to run
                # # struct_hsl_to_rgb to convert hsl to rgb as we know that
                # # the color will be white
                # if l >= 1:
                #     r[ 0 ] = <unsigned char> 255
                #     g[ 0 ] = <unsigned char> 255
                #     b[ 0 ] = <unsigned char> 255
                #     continue
                #
                # # force black pixel, we do not need to run
                # # struct_hsl_to_rgb to convert hsl to rgb as we know that
                # # the color will be black
                # if l <= 0:
                #     r[ 0 ] = <unsigned char> 0
                #     g[ 0 ] = <unsigned char> 0
                #     b[ 0 ] = <unsigned char> 0
                #     continue

                rgb_ = struct_hsl_to_rgb(hsl_.h, hsl_.s, l)

                array_tmp[j, i, 0] = <unsigned char> (rgb_.r * <float>255.0)
                array_tmp[j, i, 1] = <unsigned char> (rgb_.g * <float>255.0)
                array_tmp[j, i, 2] = <unsigned char> (rgb_.b * <float>255.0)

    return frombuffer(array_tmp, (width, height), "RGB")




@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef inline void brightness1d_c(unsigned char [:] bgr_array, const float shift, bint format_32=False)nogil:
    """
    
    Brightness
    
    Control brightness of an image given its C buffer, 1d array shape (w, )
     BGRA or RGBA (inplace)

    e.g:
    # for 24-bit  
    array_bck = brightness1d_c(background.get_buffer(), 0.1, False)
    background = pygame.image.frombuffer(array_bck, (800, 600), 'BGR')
    
    # for 32-bit 
    array_bck = brightness1d_c(background.get_buffer(), 0.1, True)
    background = pygame.image.frombuffer(array_bck, (800, 600), 'BGRA')

    :param bgr_array: 
        numpy ndarray shape (w, h, 3) containing BGRA pixels values
        
    :param shift: 
        float; values in range [-1.0 ... 1.0], 0.0 no change,
        -1 lowest brightness, +1 max brightness
        
    :param format_32: 
        bool True for 'BGRA' buffer type (32-bit) or False 'BGR' (24-bit) 
        
    :return          : void
    
    """

    cdef unsigned int length = bgr_array.shape[0]

    cdef:
        int i=0
        unsigned char * r
        unsigned char * g
        unsigned char * b
        unsigned short int bit = 3
        float l
        hsl hsl_
        rgb rgb_

    if format_32:
        bit = 4

    # noinspection SpellCheckingInspection
    for i in prange(0, length, bit, schedule=SCHEDULE, num_threads=THREADS):

        r = &bgr_array[ i     ]
        g = &bgr_array[ i + 1 ]
        b = &bgr_array[ i + 2 ]

        hsl_ = struct_rgb_to_hsl(
            r[ 0 ] * <float> ONE_255,
            g[ 0 ] * <float> ONE_255,
            b[ 0 ] * <float> ONE_255
        )

        # compensate hsl_.l
        l = hsl_.l + shift

        # force white pixel, we do not need to run
        # struct_hsl_to_rgb to convert hsl to rgb as we know that
        # the color will be white
        if l  >= <float>1.0:
            r[ 0 ] = <unsigned char> 255
            g[ 0 ] = <unsigned char> 255
            b[ 0 ] = <unsigned char> 255
            continue

        # force black pixel, we do not need to run
        # struct_hsl_to_rgb to convert hsl to rgb as we know that
        # the color will be black
        if l <= <float>0.0:
            r[ 0 ] = <unsigned char> 0
            g[ 0 ] = <unsigned char> 0
            b[ 0 ] = <unsigned char> 0
            continue

        rgb_ = struct_hsl_to_rgb(hsl_.h, hsl_.s, l)

        r[ 0 ] = <unsigned char> (rgb_.r * <float> 255.0)
        g[ 0 ] = <unsigned char> (rgb_.g * <float> 255.0)
        b[ 0 ] = <unsigned char> (rgb_.b * <float> 255.0)



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef inline np.ndarray[np.uint8_t, ndim=1] brightness1d_copy_c(
        unsigned char [:] bgr_array,
        const float shift,
        bint format_32=False
):
    """

     Brightness control (return a copy)
    
    Control brightness of an image given its C buffer, 1d array shape (w, )
    The bgr_array is a C-buffer with pixel format BGR or BGRA 

    e.g:
    # for 24-bit 
    array_bck = brightness1d_copy_c(background.get_buffer(), 0.1, False)
    background = pygame.image.frombuffer(array_bck, (800, 600), 'BGR')
    
    # for 32-bit 
    array_bck = brightness1d_copy_c(background.get_buffer(), 0.1, True)
    background = pygame.image.frombuffer(array_bck, (800, 600), 'BGRA')

    :param bgr_array: 
        numpy ndarray shape (w, ) containing BGR or BGRA pixels values.
        
    :param shift: 
        float; values in range [-1.0 ... 1.0], 0.0 no change,
        -1 lowest brightness, +1 max brightness
        
    :param format_32: 
        bool; True for 'BGRA' buffer type (32-bit) or False 'RGB' (24-bit)
         
    :return: Return a copy of the original SDL surface with adjusted brightness
    
    """

    cdef unsigned int length = bgr_array.shape[0]

    cdef:
        int i=0
        unsigned char * r
        unsigned char * g
        unsigned char * b
        unsigned short int bit = 3
        float l
        hsl hsl_
        rgb rgb_
        unsigned char [::1] bgr_array_cp = numpy.ndarray(shape=length, buffer=bgr_array, dtype=uint8)

    if format_32:
        bit = 4

    with nogil:
        # noinspection SpellCheckingInspection
        for i in prange(0, length, bit, schedule=SCHEDULE, num_threads=THREADS):

            r = &bgr_array_cp[ i     ]
            g = &bgr_array_cp[ i + 1 ]
            b = &bgr_array_cp[ i + 2 ]

            hsl_ = struct_rgb_to_hsl(
                r[ 0 ] * <float> ONE_255,
                g[ 0 ] * <float> ONE_255,
                b[ 0 ] * <float> ONE_255
            )

            # compensate hsl_.l
            l = hsl_.l + shift

            # force white pixel, we do not need to run
            # struct_hsl_to_rgb to convert hsl to rgb as we know that
            # the color will be white
            if l >= <float>1.0:
                r[ 0 ] = <unsigned char> 255
                g[ 0 ] = <unsigned char> 255
                b[ 0 ] = <unsigned char> 255
                continue

            # force black pixel, we do not need to run
            # struct_hsl_to_rgb to convert hsl to rgb as we know that
            # the color will be black
            if l <= <float>0.0:
                r[ 0 ] = <unsigned char> 0
                g[ 0 ] = <unsigned char> 0
                b[ 0 ] = <unsigned char> 0
                continue

            rgb_ = struct_hsl_to_rgb(hsl_.h, hsl_.s, l)

            r[ 0 ] = <unsigned char> (rgb_.r * <float> 255.0)
            g[ 0 ] = <unsigned char> (rgb_.g * <float> 255.0)
            b[ 0 ] = <unsigned char> (rgb_.b * <float> 255.0)

    return numpy.ndarray(shape=length, buffer=bgr_array_cp, dtype=uint8)



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef inline void brightness_ex_c(
        unsigned char [:, :, :] rgb_array_, float shift_=0.0, color_=(0, 0, 0)):
    """
    
    Brightness adjustment with color exclusion (inplace)
    
    Exclusion:
    Set the parameter color to exclude a specific color from the transformation process.
    parameter shift control the brightness transformation, with +1.0 being the maximum 
    brightness possible. 
    
    Compatible with 24, 32-bit images 
    
    e.g
    # 24-bit
    image = pygame.image.load('../Assets/px.png').convert(24)
    brightness_ex_c(image, +0.5, color=(0, 0, 0))
    
    # 32-bit
    image = pygame.image.load('../Assets/px.png').convert_alpha()
    brightness_ex_c(image, +0.5, color=(0, 0, 0))

    :param rgb_array_:
        numpy ndarray shape (w, h, 3) containing RGB pixels values
        
    :param shift_: 
        float; values in range [-1.0 ... 1.0], 0 no change, -1.0 lowest brightness effect, 
        +1.0 highest brightness effect.
        
    :param color_: 
        tuple; Color to exclude from the brightness process, default black color tuple(0, 0, 0)
        
    :return: 
        void
        
    """

    cdef Py_ssize_t width, height
    width, height = rgb_array_.shape[:2]

    cdef:
        int i=0, j=0
        unsigned char * r
        unsigned char * g
        unsigned char * b
        float l, h, s
        hsl hsl_
        rgb rgb_
        unsigned char rr=color_[0], gg=color_[1], bb=color_[2]

    with nogil:
        for j in prange(height, schedule=SCHEDULE, num_threads=THREADS):
            for i in range(width):

                r, g, b =\
                    &rgb_array_[i, j, 0], \
                    &rgb_array_[i, j, 1], \
                    &rgb_array_[i, j, 2]

                if not ((r[0]==rr) and (g[0]==gg) and (b[0]==bb)):

                    hsl_ = struct_rgb_to_hsl(
                        r[0] * <float>ONE_255,
                        g[0] * <float>ONE_255,
                        b[0] * <float>ONE_255
                    )  # struct_rgb_to_hsl returns floats, range 0.0 ... 1.0

                    # l = min((hsl_.l + shift), <float>1.0)
                    # l = max(l, <float>0.0)

                    # compensate hsl_.l
                    l = hsl_.l + shift_

                    # force white pixel, we do not need to run
                    # struct_hsl_to_rgb to convert hsl to rgb as we know that
                    # the color will be white
                    if l >= <float>1.0:
                        r[ 0 ] = <unsigned char> 255
                        g[ 0 ] = <unsigned char> 255
                        b[ 0 ] = <unsigned char> 255
                        continue

                    # force black pixel, we do not need to run
                    # struct_hsl_to_rgb to convert hsl to rgb as we know that
                    # the color will be black
                    if l <= <float>0.0:
                        r[ 0 ] = <unsigned char> 0
                        g[ 0 ] = <unsigned char> 0
                        b[ 0 ] = <unsigned char> 0
                        continue

                    rgb_ = struct_hsl_to_rgb(hsl_.h, hsl_.s, l)

                    r[0] = <unsigned char> (rgb_.r * <float>255.0)
                    g[0] = <unsigned char> (rgb_.g * <float>255.0)
                    b[0] = <unsigned char> (rgb_.b * <float>255.0)


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef inline void brightness_bpf_c(
        unsigned char [:, :, :] rgb_array_,
        float shift_=0.0,
        unsigned char bpf_threshold_=64):
    """
    
    Brightness adjustment with *bpf filter (inplace)
    *bpf stand for bright pass filter
    
    Exclusion:
    bpf_threshold is an integer value in range [0..255] that 
    determines the pixels threshold for the brightness algorithm. 
    The RGB sum below this threshold will not be included in the process.  
    
    Compatible with 24, 32-bit images
    
    e.g:
     24-bit
    image = pygame.image.load('../Assets/px.png').convert()
    brightness_bpf_c(image, 0.5, bpf_threshold=200)

    # 32-bit 
    image = pygame.image.load('../Assets/px.png').convert_alpha()
    brightness_bpf_c(image, 0.5, bpf_threshold=200)
    
    :param rgb_array_: 
        Pygame.Surface compatible with 24 - 32 bit
         
    :param shift_: 
        float, must be in range [-1.00 ... +1.00] this value control the brightness
         
    :param bpf_threshold_: 
        integer value in range [0 ... 255]. Bright pass filter value. 
        Equivalent to a threshold RGB. e.g sum of pixel values < threshold will not be modified. Default is 64
        
    :return:
        void
         
    """


    cdef Py_ssize_t w, h
    w, h = rgb_array_.shape[:2]

    cdef:
        int i=0, j=0
        unsigned char * r
        unsigned char * g
        unsigned char * b
        float l
        hsl hsl_
        rgb rgb_

    with nogil:
        for j in prange(h, schedule=SCHEDULE, num_threads=THREADS):
            for i in range(w):

                r, g, b = \
                    &rgb_array_[i, j, 0],\
                    &rgb_array_[i, j, 1],\
                    &rgb_array_[i, j, 2]

                if r[0] + g[0] + b[0] < bpf_threshold_:
                    continue

                # divide RGB by 255.0
                hsl_ = struct_rgb_to_hsl(
                    r[0] * <float>ONE_255,
                    g[0] * <float>ONE_255,
                    b[0] * <float>ONE_255
                ) # struct_rgb_to_hsl returns floats, range 0.0 ... 1.0

                # compensate hsl_.l
                l = hsl_.l + shift_

                # force white pixel, we do not need to run
                # struct_hsl_to_rgb to convert hsl to rgb as we know that
                # the color will be white

                if l >= <float>1.0:
                    r[0] = <unsigned char> 255
                    g[0] = <unsigned char> 255
                    b[0] = <unsigned char> 255
                    continue

                # force black pixel, we do not need to run
                # struct_hsl_to_rgb to convert hsl to rgb as we know that
                # the color will be black
                if l <= <float>0.0:
                    r[0] = <unsigned char>0
                    g[0] = <unsigned char>0
                    b[0] = <unsigned char>0
                    continue

                rgb_ = struct_hsl_to_rgb(hsl_.h, hsl_.s, l)

                r[0] = <unsigned char> (rgb_.r * <float>255.0 )
                g[0] = <unsigned char> (rgb_.g * <float>255.0 )
                b[0] = <unsigned char> (rgb_.b * <float>255.0 )



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef inline void saturation_c(unsigned char [:, :, :] rgb_array_, const float shift_):
    """
    Adjust the saturation level of an image or surface (in-place).

    This function modifies the saturation of an image by adjusting the intensity of its colors. 
    A positive `shift_` value increases saturation (making colors more vivid), while a negative 
    value decreases it (making colors more grayscale). A shift of `0.0` leaves the image unchanged.

    The function operates directly on the provided `rgb_array_`, modifying the pixel values in place.

    Example:
        saturation_c(surface, 0.2)  # Increase saturation
        saturation_c(surface, -0.5) # Decrease saturation (closer to grayscale)

    Parameters
    ----------
    rgb_array_ : numpy.ndarray
        A 3D NumPy array with shape (width, height, 3), containing RGB pixel values.
        Each color channel should be an unsigned 8-bit integer (uint8) with values in the range [0, 255].

    shift_ : float
        A value in the range [-1.0, 1.0] that controls the saturation level:
        - `0.0`: No change.
        - Positive values increase saturation (enhance colors).
        - Negative values decrease saturation (desaturate colors toward grayscale).

    Returns
    -------
    void
        This function modifies `rgb_array_` in place and does not return a new array.

    Notes
    -----
    - The function applies a per-pixel transformation to adjust saturation.
    - Since the function operates in place, the original array is directly modified.
    - The function does not support RGBA images (alpha channels).
    """


    assert -1.0 <= shift_ <= 1.0, \
        "Argument shift must be in range[-1.0 ... 1.0]"

    cdef Py_ssize_t width, height
    width, height = rgb_array_.shape[:2]

    cdef:
        int i=0, j=0
        unsigned char *r
        unsigned char *g
        unsigned char *b
        unsigned int sum_rgb
        float s
        hsl hsl_
        rgb rgb_

    with nogil:
        for i in prange(width, schedule=SCHEDULE, num_threads=THREADS):
            for j in range(height):

                r, g, b = \
                    &rgb_array_[i, j, 0], \
                    &rgb_array_[i, j, 1], \
                    &rgb_array_[i, j, 2]

                sum_rgb = r[0] + g[0] + b[0]

                # hsl transformation of solid white or
                # solid black color is invariant
                if sum_rgb == 0 or sum_rgb == 765:
                    continue

                hsl_ = struct_rgb_to_hsl(
                    <float>r[0] * <float>ONE_255,
                    <float>g[0] * <float>ONE_255,
                    <float>b[0] * <float>ONE_255
                )

                # Modifying the saturation level and apply
                # the transformation to the pixel
                s = min((hsl_.s + shift_), <float> 1.0)
                s = max(s, <float> 0.0)

                rgb_ = struct_hsl_to_rgb(hsl_.h, s, hsl_.l)

                r[0] = <unsigned char>(rgb_.r * <float>255.0)
                g[0] = <unsigned char>(rgb_.g * <float>255.0)
                b[0] = <unsigned char>(rgb_.b * <float>255.0)



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef inline void saturation1d_c(
        unsigned char [:] buffer,
        const float shift,
        bint format_32=False
)nogil:

    """

    Saturate 1d array shape (w, ) (inplace)
     
    The array must be a numpy.ndarray or memoryviewslice shape (w, ) type uint8 containing
    RGB(A) or any other pixel format and referencing an SDL surface or image. 
    The pixel format must have tha alpha channel placed last such as RGBA or BGRA   

    e.g:
    # for 24-bit  
    image = pygame.image.load("../Assets/px.png").convert(24)
    saturation1d_c(image.get_buffer(), 0.5, False)
    
    # for 32-bit  
    image = pygame.image.load("../Assets/px.png").convert_alpha()
    saturation1d_c(image.get_buffer(), 0.5, True)

    :param buffer: 
        numpy.ndarray or memoryviewslice shape(w,) uint8 data type, (unsigned char 0...255) 
        containing RGB(A), BGR(A) or any other pixel format.
        
    :param shift: 
        float; values in range [-1.0 ... 1.0], 0.0 no change, -1 lowest saturation, +1 max saturation
        Shift value control the saturation level.
        
    :param format_32: 
        bool; False for 'RGB' buffer type (24-bit) or True for 'RGBA' (32-bit). 
        This bit enable/disable the alpha layer.
        
    :return: 
        void; Inplace transformation. 
        
    """

    assert -1.0 <= shift <= 1.0, \
        "Argument shift (float) must be in range[-1.0 ... 1.0]"

    cdef:
        Py_ssize_t length

    length  = len(buffer)

    cdef:
        int i = 0
        unsigned char *r
        unsigned char *g
        unsigned char *b
        unsigned int sum_rgb
        float s
        hsl hsl_
        rgb rgb_
        int bit = 3

    if format_32:
        bit = 4

    # noinspection SpellCheckingInspection
    for i in prange(0, length, bit, schedule = SCHEDULE, num_threads = THREADS):

        r = &buffer[ i ]
        g = &buffer[ i + 1 ]
        b = &buffer[ i + 2 ]

        # No transformation for invisible pixels
        if format_32:
           if buffer[i + 3] == 0:
               continue

        sum_rgb = r[ 0 ] + g[ 0 ] + b[ 0 ]

        # hsl transformation of solid white or
        # solid black color is invariant
        if sum_rgb == 0 or sum_rgb == 765:
            continue

        hsl_ = struct_rgb_to_hsl(
            <float> r[ 0 ] * <float> ONE_255,
            <float> g[ 0 ] * <float> ONE_255,
            <float> b[ 0 ] * <float> ONE_255
        )

        # Modifying the saturation level and apply
        # the transformation to the pixel
        s = min((hsl_.s + shift), <float> 1.0)
        s = max(s, <float> 0.0)

        rgb_ = struct_hsl_to_rgb(hsl_.h, s, hsl_.l)

        r[ 0 ] = <unsigned char> (rgb_.r * <float> 255.0)
        g[ 0 ] = <unsigned char> (rgb_.g * <float> 255.0)
        b[ 0 ] = <unsigned char> (rgb_.b * <float> 255.0)



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef np.ndarray[np.uint8_t, ndim=1] saturation1d_cp_c(
        const unsigned char [:] buffer,
        const float shift,
        bint format_32=False
):

    """

    Saturate 1d array shape (w, ) (return a copy)

    The array must be a numpy.ndarray|buffer or memoryviewslice shape (w, ) type uint8 containing
    RGB(A) or any other pixel format and referencing an SDL surface or image.
     
    The pixel format must have alpha channel placed last such as RGB(A) or BGR(A) 
    for 32-bit image data.   

    e.g:
    # for buffer 32-bit 
    image = pygame.image.load("../Assets/px.png").convert_alpha()
    buffer_cp = saturation1d_cp_c(image.get_buffer(), 0.5, True)
    
    # for 24-bit
    image = pygame.image.load("../Assets/px.png").convert(24)
    buffer_cp = saturation1d_cp_c(image.get_buffer(), 0.5, False)

    :param buffer: 
        numpy ndarray|buffer or memoryviewslice shape (w,) type uint8 containing RGB(A) or any other
        pixel format.
          
    :param shift: 
        float; values in range [-1.0 ... 1.0], 0.0 no change,-1 lowest saturation, +1 max saturation.
        Shift control the saturation level
        
    :param format_32: 
        bool; False for 'RGB' buffer type (24-bit) or True for 'RGBA' (32-bit). 
        This bit enable/disable the alpha layer
        
    :return: 
        numpy.ndarray 1d array shape (w, ) type uint8 containing same pixel 
        format than input array
         
    """

    assert -1.0 <= shift <= 1.0, \
        "Argument shift must be in range[-1.0 ... 1.0]"

    cdef:
        Py_ssize_t length

    length  = len(buffer)

    cdef:
        int i = 0
        const unsigned char *r
        const unsigned char *g
        const unsigned char *b
        unsigned int sum_rgb
        unsigned char [::1] buffer_cp = numpy.ndarray(shape=length, buffer=buffer, dtype=uint8)
        float s
        hsl hsl_
        rgb rgb_
        int bit = 3
        unsigned char * p1

    if format_32:
        bit = 4

    with nogil:
        # noinspection SpellCheckingInspection
        for i in prange(0, length, bit, schedule = SCHEDULE, num_threads = THREADS):

            b = &buffer[ i ]

            # No transformation for invisible pixels
            if format_32:
                if (b + 3)[ 0 ] == 0:
                    continue

            g = b + 1
            r = b + 2

            sum_rgb = r[ 0 ] + g[ 0 ] + b[ 0 ]

            # hsl transformation of solid white or
            # solid black color is invariant
            if sum_rgb == 0 or sum_rgb == 765:
                continue

            hsl_ = struct_rgb_to_hsl(
                <float> r[ 0 ] * <float> ONE_255,
                <float> g[ 0 ] * <float> ONE_255,
                <float> b[ 0 ] * <float> ONE_255
            )

            # Modifying the saturation level and apply
            # the transformation to the pixel
            s = min((hsl_.s + shift), <float> 1.0)
            s = max(s, <float> 0.0)

            rgb_ = struct_hsl_to_rgb(hsl_.h, s, hsl_.l)

            p1 = &buffer_cp[ i ]

            p1[ 0 ]     = <unsigned char> (rgb_.r * <float> 255.0)
            (p1+1)[ 0 ] = <unsigned char> (rgb_.g * <float> 255.0)
            (p1+2)[ 0 ] = <unsigned char> (rgb_.b * <float> 255.0)

            # buffer_cp is a copy of the input array.
            # The alpha array should be identical to the input so no need
            # to change it.
            # if format_32:
            #     (p1 + 3)[0] = <unsigned char> (b + 3)[ 0 ]

    return numpy.ndarray(shape=length, buffer=buffer_cp, dtype=uint8)



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef inline saturation_mask(
        const unsigned char [:, :, :] rgb_array,
        const float shift,
        const unsigned char [:, :] mask,
):
    """
    Apply a saturation mask to an image (returns a copy).

    This function modifies the saturation of an image (`rgb_array`) based on the given `shift` value, 
    while applying a mask to selectively disable the effect in certain areas.

    ### Parameters:
    - **rgb_array** (`numpy.ndarray` or `memoryviewslice`):  
    A 3D array of shape `(width, height, 3)`, with `dtype=uint8`, representing an image in RGB, BGR, or any similar pixel format.

    - **shift** (`float`):  
    A value in the range `[-1.0, 1.0]` that determines the saturation adjustment:
    - `[-1.0, 0.0]` → Decreases saturation.
    - `[0.0, 1.0]` → Increases saturation.

    - **mask** (`numpy.ndarray`):  
    A 2D array of shape `(width, height)`, with `dtype=uint8`, representing the mask layer.  
    Ideally, values should be either `255` (enable saturation effect) or `0` (disable saturation effect).  
    If `mask` and `rgb_array` have different dimensions, a `ValueError` is raised.

    ### Returns:
    - **pygame.Surface**:  
    A 24-bit surface without a transparency layer.

    ### Example Usage:
    ```python
    import pygame
    from pygame.surfarray import pixels3d

    # Load and prepare the mask
    mask = pygame.image.load('../Assets/alpha.png').convert_alpha()
    mask = pygame.transform.smoothscale(mask, (800, 600))
    mask_array = pygame.surfarray.pixels_alpha(mask)
    mask_array = BW(mask_array)  # Convert to black & white if needed

    # Apply saturation mask to the background
    background = saturation_mask(pixels3d(background), 0.5, mask_array)
    SCREEN.blit(background, (0, 0))


    """

    if shift == <float>0.0:
        return numpy.array(rgb_array)

    assert -1.0 <= shift <= 1.0, 'Argument shift must be in range [-1.0 .. 1.0].'

    if not is_uint8(rgb_array):
        raise TypeError(
            "\nExpecting uint8 (unsigned char) data type got %s" % rgb_array.dtype)

    if not is_uint8(mask):
        raise TypeError(
            "\nExpecting uint8 (unsigned char) data type got %s" % rgb_array.dtype)

    cdef Py_ssize_t w, h, w_mask, h_mask

    try:
        w, h = rgb_array.shape[ :2 ]

    except (ValueError, pygame.error):
        raise ValueError(
            '\nArray bgr_array shape not understood.')

    try:
        w_mask, h_mask = mask.shape[ :2 ]

    except (ValueError, pygame.error):
        raise ValueError(
            '\nArray mask shape not understood.')

    if w != w_mask or h != h_mask:
        raise ValueError(
            "\nExpecting array bgr_array (%s, %s)"
            " and mask (%s, %s) to have same shapes." % (w, h, w_mask, h_mask))

    cdef:
        const unsigned char *r
        const unsigned char *g
        const unsigned char *b
        unsigned int sum_rgb
        unsigned char [:, :, ::1] tmp_array = numpy.empty((h, w, 3), dtype = numpy.uint8, order='C')
        float s
        hsl hsl_
        rgb rgb_
        int i, j


    with nogil:
        for j in prange(h, schedule = SCHEDULE, num_threads = THREADS):
            for i in range(w):

                if mask[i, j] == 0:
                    continue

                r, g, b = \
                    &rgb_array[ i, j, 0 ], \
                    &rgb_array[ i, j, 1 ], \
                    &rgb_array[ i, j, 2 ]

                sum_rgb = r[0] + g[0] + b[0]

                if sum_rgb == 0 or sum_rgb == 765:
                    continue

                hsl_ = struct_rgb_to_hsl(
                    r[0] * <float>ONE_255,
                    g[0] * <float>ONE_255,
                    b[0] * <float>ONE_255
                )
                s = min((hsl_.s + shift), <float>1.0)
                s = max(s, <float>0.0)

                rgb_ = struct_hsl_to_rgb(hsl_.h, s, hsl_.l)

                tmp_array[ j, i, 0 ] = <unsigned char> (rgb_.r * <float> 255.0)
                tmp_array[ j, i, 1 ] = <unsigned char> (rgb_.g * <float> 255.0)
                tmp_array[ j, i, 2 ] = <unsigned char> (rgb_.b * <float> 255.0)

    return frombuffer(tmp_array, (w, h) , 'RGB')



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef inline void saturation_mask_inplace(
        unsigned char [:, :, :] rgb_array,
        const float shift,
        const unsigned char [:, :] mask,
        int w, int h
)nogil:
    """
    Saturation effect with mask designed for class area24_cc (light effect)
    
    rgb_array is a numpy.ndarray shape (w, h, 3) type uint8 with RGB or BGR or any
    other pixel format. This array reference the SDL surface or image and any changes
    to this array will modify the image directly.
    
    The mask layer should be a 2d array filled with uint8 values. Ideally the values
    must be in either 255 or 0. Zero will disable the effect
    Mask and rgb_array must have the same width and height,
    but mask (w, h) is transposed compared to rgb_array (h, w, 3) 
    
    e.g:
    mask = pygame.image.load('../Assets/alpha.png').convert_alpha()
    mask = pygame.transform.smoothscale(mask, (800, 600))
    mask_array = pygame.surfarray.pixels_alpha(mask)
    # mask_array is transposed 
    mask_array = BW(mask_array).T
    
    background = saturation_mask(pixels3d(background), 0.5, mask_array)
    SCREEN.blit(background, (0, 0))
       
    :param rgb_array: 
        3d numpy.ndarray or memoryviewslice shapes (h, w, 3) type uint8
        containing RGB or BGR or any other pixel format.
        This array reference the SDL surface or image
        
    :param shift: 
        Value must be in range [-1.0 ... 1.0],
        between [-1.0 ... 0.0] decrease saturation.
        between [0.0  ... 1.0] increase saturation.
                   
    :param mask: 
        unsigned char numpy.ndarray shape (w, h) type uint8, 
        layer mask to use for disabling the saturation effect
        
    :param w: 
        int width of the array
         
    :param h:
        int height of the array
    
    return:Void
    """

    cdef:
        unsigned char *r
        unsigned char *g
        unsigned char *b
        unsigned char sum_rgb
        float s
        hsl hsl_
        rgb rgb_
        int i, j


    for i in prange(w, schedule=SCHEDULE, num_threads=THREADS):
        for j in range(h):

                if mask[i, j] == 0:
                     continue

                r, g, b = \
                    &rgb_array[ j, i, 0 ], \
                    &rgb_array[ j, i, 1 ], \
                    &rgb_array[ j, i, 2 ]

                sum_rgb = r[ 0 ] + g[ 0 ] + b[ 0 ]

                if sum_rgb == 0 or sum_rgb == 765:
                    continue

                hsl_ = struct_rgb_to_hsl(
                    <float> r[ 0 ] * <float> ONE_255,
                    <float> g[ 0 ] * <float> ONE_255,
                    <float> b[ 0 ] * <float> ONE_255
                )

                s = min((hsl_.s + shift), <float> 1.0)
                s = max(s, <float> 0.0)

                rgb_ = struct_hsl_to_rgb(hsl_.h, s, hsl_.l)

                r[ 0 ] = <unsigned char> (rgb_.r * <float> 255.0)
                g[ 0 ] = <unsigned char> (rgb_.g * <float> 255.0)
                b[ 0 ] = <unsigned char> (rgb_.b * <float> 255.0)





# ------------------------------------ OTHER


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef inline void heatconvection_inplace_c(
        unsigned char [:, :, :] rgb_array,
        float amplitude,
        float center=0.0,
        float sigma=2.0,
        float mu=0.0):

    """
    Heat flow convection
    
    Convection (or convective heat transfer) is the transfer of heat from one place to another 
    due to the movement of fluid. Although often discussed as a distinct method of heat transfer, 
    convective heat transfer involves the combined processes of conduction (heat diffusion).    
    This effect can be use to simulate air turbulence or heat flow/convection
    it applies a gaussian transformation at the base of the image (vertical flow)   
    
    # for 32-24 bit image format 
     image = pygame.image.load("../Assets/fire.jpg").convert()
     b = math.cos(i * 3.14 / 180.0) * random.uniform(0, 2)
     heatconvection(image, abs(b) * random.uniform(20.0, 80.0),
         0, sigma = random.uniform(0.8, 4), mu_ = b)
    # Restore the original image 
     image = image_copy.copy()

    :param rgb_array: 
        numpy.ndarray shape (w, h, n) type uint8 containing RGB pixels or any other format
         
    :param amplitude: 
        Control the maximum amplitude (pixels displacement on the Y-axis, vertical effect) 
        of the gaussian equation. No transformation if amplitude equal zero. example of 
        an variable amplitude issue from a periodic function:
        b = math.cos(i * 3.14 / 180.0) * random.uniform(0, 2) with i linear.
         
    :param center: 
        Control the center of the gaussian equation (if center equal zero,
        the Gauss equation is centered (default is 0.0)
                   
    :param sigma: 
        float; sigma value of the gauss equation, a small value will create a 
        narrow effect while a stronger value will wider the effect. Please refers 
        to the gaussian distribution for further analysis on the sigma values (default is 2.0).
        
    :param mu: 
        float; mu value of the gauss equation. when mu is periodic such as a cosine trigonometric 
        function, it allows to displace the effect along the X-axis (default is 0.0).
        
    """

    cdef Py_ssize_t w, h

    w, h = rgb_array.shape[:2]

    cdef:
        unsigned char [:, :, ::1] array_cp = \
            numpy.asarray(rgb_array, dtype=uint8, order='C')
        int x = 0, y = 0
        unsigned int yy
        unsigned int h_1 = <int>h - 1
        float [::1] f_gauss = linspace(-4, 4, w, dtype=float32)

    with nogil:

        for x in prange(w, schedule=SCHEDULE, num_threads=THREADS):

            for y in range(h):

                yy =<int>(gauss(f_gauss[x], center, sigma, mu) * amplitude + y)

                if yy > h_1:
                    yy = h_1

                if yy < 0:
                    yy = 0

                rgb_array[x, y, 0] = array_cp[x, yy, 0]
                rgb_array[x, y, 1] = array_cp[x, yy, 1]
                rgb_array[x, y, 2] = array_cp[x, yy, 2]


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(False)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef inline void horizontal_glitch_c(
        unsigned char [:, :, :] rgb_array,
        const float deformation,
        const float frequency,
        const float amplitude):

    """
    Apply a horizontal glitch effect to an image (in-place).

    This function distorts an image (`rgb_array`) by shifting pixels horizontally 
    based on a cosine wave, creating a glitch effect.

    ### Parameters:
    - **rgb_array** (`numpy.ndarray`):  
    A 3D array of shape `(width, height, 3)`, with `dtype=uint8`, representing an image with RGB pixels.  
    The transformation is applied in-place.

    - **deformation** (`float`):  
    An angle in radians that controls the variation of the distortion over time.

    - **frequency** (`float`):  
    A factor that amplifies the deformation angle, controlling the periodicity of the distortion.

    - **amplitude** (`float`):  
    The amplitude of the cosine wave, determining the maximum horizontal displacement of pixels.

    ### Returns:
    - **None** (modifies `rgb_array` in-place).

    ### Example Usage:
    ```python
    horizontal_glitch_c(
        bgr_array, 
        deformation=0.5, 
        frequency=0.08, 
        amplitude=FRAME % 20
    )
    """
    cdef Py_ssize_t w, h
    w, h = rgb_array.shape[:2]

    cdef:
        int i=0, j=0
        float rad = <float>(<float>3.14/<float>180.0)
        float angle = <float>0.0
        float angle1 = <float>0.0
        unsigned char [::1, :, :] rgb_array_cp = \
            numpy.asarray(rgb_array, dtype=uint8, order='F')
        int ii=0

    with nogil:

        for j in range(h):

            for i in range(w):

                ii = (i + <int>(<float>cos(angle) * amplitude))
                if ii > w - 1:
                    ii = w - 1
                if ii < 0:
                    ii = 0

                rgb_array[i, j, 0] = rgb_array_cp[ii, j, 0]
                rgb_array[i, j, 1] = rgb_array_cp[ii, j, 1]
                rgb_array[i, j, 2] = rgb_array_cp[ii, j, 2]

            angle1 = angle1 + frequency * rad
            angle = angle + (deformation * rad + rand() % angle1 - rand() % angle1)



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(False)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef inline void horizontal_sglitch_c(
        unsigned char [:, :, :] bck_array,
        unsigned char [:, :, :] array,
        const float deformation,
        const float frequency,
        const float amplitude):

    """
    Glitch for static image/background (inplace)

    Deform the pygame display to create a glitch appearance.
    The Arrays (bgr_array, array) must both be numpy arrays shape (w, h, 3) containing RGB pixels.
    
    e.g:
     # for 24 - 32 bit
    horizontal_sglitch_c(
         bck_array,
         array, 
         deformation = 0.5,
         frequency   = 0.08, 
         amplitude   = FRAME % 20)
    
    :param bck_array: 
        numpy.ndarray shape (w, h, 3) uint8 containing RGB pixels
        
    :param array: 
        numpy.ndarray shape (w, h, 3) copy, background array copy.
        
    :param deformation: 
        float; Angle in radians, this value control the angle variation over the time
        
    :param frequency: 
        float; signal frequency, factor that amplify the angle variation
        
    :param amplitude: 
        float; cos amplitude value
        
    :return: 
    void
    
    """

    cdef Py_ssize_t w, h, ww, hh

    w, h = bck_array.shape[:2]
    ww, hh = array.shape[:2]

    if w!=ww and h!=hh:
        raise ValueError("\nBoth surface and bgr_array must have the same sizes/dimensions")

    cdef:
        int i=0, j=0
        float rad = <float>(<float>3.14/<float>180.0)
        float angle = <float>0.0
        float angle1 = <float>0.0
        int ii=0

    with nogil:

        for j in range(h):

            for i in range(w):

                ii = (i + <int>(<float>cos(angle) * amplitude))
                if ii > <int>w - 1:
                    ii = <int>w - 1
                if ii < 0:
                    ii = 0

                bck_array[i, j, 0],\
                bck_array[i, j, 1],\
                bck_array[i, j, 2] = array[ii, j, 0],\
                    array[ii, j, 1], array[ii, j, 2]

            angle1 = angle1 + frequency * rad
            angle = angle + (deformation * rad + rand() % angle1 - rand() % angle1)



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef inline void bpf_inplace_c(
        unsigned char [:, :, :] rgb_array_, int w, int h, int threshold = 128)nogil:
    
    """
    Apply a Bright Pass Filter (BPF) to an image in-place.

    This function retains only the brightest pixels in the given image (`rgb_array_`),  
    effectively filtering out darker regions. The modification is applied in-place.

    ### Parameters:
    - **rgb_array_** (`numpy.ndarray`):  
    A 3D array of shape `(width, height, 3)`, with `dtype=uint8`, representing an image with RGB pixels.  
    Pixels below the given `threshold` are set to zero.

    - **w** (`int`):  
    The width of the array.

    - **h** (`int`):  
    The height of the array.

    - **threshold** (`int`, default=128):  
    The brightness threshold.  
    Pixels with values below this threshold are set to zero.

    ### Returns:
    - **None** (modifies `rgb_array_` in-place).

    ### Example Usage:
    ```python
    # Apply a bright pass filter with a threshold of 60
    bpf_inplace_c(image, w=image.shape[0], h=image.shape[1], threshold=60)
    """
    

    assert 0 <= threshold <= 255, "Argument threshold must be in range [0 ... 255]"


    cdef:
        int i = 0, j = 0
        float lum, c
        unsigned char *r
        unsigned char *g
        unsigned char *b

    with nogil:
        for j in prange(0, h, schedule=SCHEDULE, num_threads=THREADS):
            for i in range(0, w):

                # ITU-R BT.601 luma coefficients
                r = &rgb_array_[i, j, 0]
                g = &rgb_array_[i, j, 1]
                b = &rgb_array_[i, j, 2]

                if r[0] + g[0] + b[0] == 0:
                    continue

                lum = r[0] * <float>0.299 + g[0] * <float>0.587 + b[0] * <float>0.114

                if lum < threshold:
                    r[ 0 ], g[ 0 ], b[ 0 ] = 0, 0, 0
                    continue

                c = (lum - threshold) / lum
                r[0] = <unsigned char>(r[0] * c)
                g[0] = <unsigned char>(g[0] * c)
                b[0] = <unsigned char>(b[0] * c)


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef inline void bpf_c(
        object surface_,
        int threshold = 128):

    """
    
    Apply a Bright Pass Filter (BPF) to a pygame surface.

    This function enhances the brightest areas of the given `surface_`  
    by filtering out pixels below the specified `threshold`.

    ### Parameters:
    - **surface_** (`pygame.Surface`):  
    The source surface to which the bright pass filter is applied.

    - **threshold** (`int`, default=128):  
    The brightness threshold.  
    Pixels with values below this threshold are suppressed.

    ### Returns:
    - **None** (modifies `surface_` in-place).

    ### Example Usage:
    ```python
    bpf_c(image, threshold=100)

    
    """

    assert 0 <= threshold <= 255, \
        "Argument threshold must be in range [0 ... 255]"

    cdef:
        Py_ssize_t w, h

    cdef unsigned char [:, :, :] rgb_array = pixels3d(surface_)

    w, h = rgb_array.shape[:2]

    cdef:
        int i = 0, j = 0
        float lum, c
        unsigned char *r
        unsigned char *g
        unsigned char *b


    with nogil:
        for j in prange(0, h, schedule=SCHEDULE, num_threads=THREADS):
            for i in range(0, w):

                # ITU-R BT.601 luma coefficients
                r = &rgb_array[i, j, 0]
                g = &rgb_array[i, j, 1]
                b = &rgb_array[i, j, 2]

                if r[0] + g[0] + b[0] == 0:
                    continue

                lum = r[0] * <float>0.299 + g[0] * <float>0.587 + b[0] * <float>0.114

                if lum < threshold:
                    r[ 0 ], g[ 0 ], b[ 0 ] = 0, 0, 0
                    continue

                c = (lum - threshold) / lum
                r[0] = <unsigned char>(r[0] * c)
                g[0] = <unsigned char>(g[0] * c)
                b[0] = <unsigned char>(b[0] * c)


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef inline bpf24_c(
        unsigned char [:, :, :] rgb_array,
        int threshold = 128,
        ):

    """
    
    Bright Pass Filter (bpf)

    Conserve only the brightest pixels in an array

    The Array (rgb_array) must be a numpy array shape (w, h, 3) containing RGB pixels

    :param rgb_array: 
        numpy.ndarray shape (w, h, 3) uint8 containing RGB pixels
        
    :param threshold: 
        float Bright pass threshold default 128
        
    :return: 
        Return the modified array shape (w, h, 3) uint8
        
    """
    assert 0 <= threshold <= 255, "Argument threshold must be in range [0 ... 255]"

    cdef:
        Py_ssize_t w, h
    w, h = rgb_array.shape[:2]

    cdef:
        int i = 0, j = 0
        float lum, c
        unsigned char [:, :, :] output_array_ = numpy.zeros((h, w, 3), uint8)
        unsigned char *r
        unsigned char *g
        unsigned char *b

    with nogil:
        for j in prange(0, h, schedule=SCHEDULE, num_threads=THREADS):
            for i in range(0, w):

                # ITU-R BT.601 luma coefficients
                r = &rgb_array[i, j, 0]
                g = &rgb_array[i, j, 1]
                b = &rgb_array[i, j, 2]

                if r[0] + g[0] + b[0] == 0:
                    continue

                lum = r[0] * <float>0.299 + g[0] * <float>0.587 + b[0] * <float>0.114

                if lum < threshold:
                    continue

                c = (lum - threshold) / lum
                output_array_[j, i, 0] = <unsigned char>(r[0] * c)
                output_array_[j, i, 1] = <unsigned char>(g[0] * c)
                output_array_[j, i, 2] = <unsigned char>(b[0] * c)

    return frombuffer(output_array_, (w, h), 'RGB')



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef inline void filtering24_c(object surface_, mask_):
    """
    Apply an internal optimized filtering to a 24/32-bit surface (in-place).

    This function processes a 24-bit or 32-bit `surface_` using a 32-bit `mask_`.  
    The mask's alpha values are converted to float and used to adjust the filtering effect.  
    Modifications are applied directly to `surface_`.

    ### Parameters:
    - **surface_** (`pygame.Surface`):  
      A 24-bit or 32-bit surface that will be filtered.

    - **mask_** (`pygame.Surface`):  
      A 32-bit surface containing an alpha channel.  
      The alpha values are used to modulate the effect during processing.  
      The mask must be a 32-bit surface; otherwise, a `ValueError` is raised.

    ### Returns:
    - **None** (modifies `surface_` in-place).

    ### Raises:
    - `ValueError`:  
      - If `mask_` is not a 32-bit surface (missing alpha channel or incorrect format).
      - If `surface_` cannot create a C buffer.

    ### Example Usage:
    ```python
    image = pygame.image.load('../Assets/px.png').convert(24)
    image = pygame.transform.smoothscale(image, (800, 600))

    mask = pygame.image.load('../Assets/alpha.png').convert_alpha()
    mask = pygame.transform.smoothscale(mask, (800, 600))

    filtering24_c(image, mask)
    ```
    """

    # Get surface dimensions
    cdef:
        int w, h
        unsigned short int byte_size  # Bytes per pixel for surface_
        unsigned short int m_bytesize # Bytes per pixel for mask_

    w, h = surface_.get_size()
    byte_size = surface_.get_bytesize()  # Determine surface pixel format
    m_bytesize = mask_.get_bytesize()    # Determine mask pixel format

    # Ensure the mask is 32-bit (RGBA)
    if m_bytesize != 4:
        raise ValueError("\nMask argument is incorrect, missing alpha channels or wrong shape.")

    # Attempt to retrieve a buffer from the surface
    cdef unsigned char [::1] rgb

    try:
        rgb = surface_.get_buffer()
    except (ValueError, pygame.error):
        raise ValueError("\nCannot create a C buffer from the given surface.")

    # Retrieve mask buffer
    cdef:
        unsigned char [::1] mask = mask_.get_buffer()
        int i
        Py_ssize_t l = (w * h) << 2 if byte_size == 4 else w * h * 3  # Compute buffer size
        unsigned char * p1
        float a  # Alpha multiplier

    # Perform pixel-wise modification in parallel with OpenMP
    with nogil:
        for i in prange(0, l, byte_size, schedule=SCHEDULE, num_threads=THREADS):

            # Compute alpha multiplier based on mask's alpha channel
            a = mask[i + 3] * <float>ONE_255 if byte_size == 4 \
                else mask[((i / byte_size) << 2) + 3] * <float>ONE_255

            # Apply alpha multiplication to each color channel
            p1 = &rgb[i]
            p1[0] = <unsigned char> (p1[0] * a)       # Red channel
            (p1 + 1)[0] = <unsigned char> ((p1 + 1)[0] * a)  # Green channel
            (p1 + 2)[0] = <unsigned char> ((p1 + 2)[0] * a)  # Blue channel


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef inline void filtering_inplace_c(object surface_, mask_):
    """
    
    Surface masking (inplace), use a 2d numpy.ndarray for masking

    selectively modify the opacity (transparency) of a surface with the 
    given mask alpha (array). 

    `mask_` is a numpy.ndarray or memoryviewslice shape (w, h) type uint8, 
    containing alpha values range [ 0 ... 255].
    it hides anything that falls outside of its transparency shape (zero values hiding pixels 
    and 255.0 giving full texture opacity.
    
    e.g:
    # creating the mask array
    mask = pygame.image.load('../Assets/alpha.png').convert_alpha()
    mask = pygame.transform.smoothscale(mask, (800, 600))
    mask_array = pygame.surfarray.pixels_alpha(mask)
    
    # in the main loop
    filtering_inplace_c(source, mask_array)

    :param surface_: 
        pygame.Surface compatible 24-32 bit
        
    :param mask_: 
        numpy.ndarray shape(w, h) containing alpha values range [ 0 ... 255] 
        Value 0 hides pixels and 255 gives full texture opacity
        
    :return: 
    void

    """
    cdef:
        int w, h, w_, h_

    w, h = surface_.get_size()

    try:
        w_, h_ = mask_.shape[:3]

    except (ValueError, pygame.error):
       raise ValueError(
           '\nArgument mask type not understood, '
           'expecting numpy.ndarray type (w, h) got %s ' % type(mask_))

    assert w == w_ and h == h_, \
        '\nSurface and mask size does not match (w:%s, h:%s), ' \
        '(w:%s, h:%s) ' % (w, h, w_, h_)

    cdef:
        unsigned char [:, :, :] rgb = surface_.get_view('3')
        float [:, :] mask = numpy.asarray(mask_, dtype=numpy.float32)
        int i, j
        unsigned char * r
        unsigned char * g
        unsigned char * b
        float a

    with nogil:

        for j in prange(0, h, schedule=SCHEDULE, num_threads=THREADS):

            for i in range(w):

                r = &rgb[i, j, 2]
                g = &rgb[i, j, 1]
                b = &rgb[i, j, 0]
                a = mask[i, j] * <float>ONE_255

                r[0] = <unsigned char>(r[0] * a)
                g[0] = <unsigned char>(g[0] * a)
                b[0] = <unsigned char>(b[0] * a)


# todo check if really compatible with 24 bit
#  surface if not delete same function that the optimized one above
@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef inline void filtering1d_inplace_c(object surface_, mask_):
    """

    Surface masking (inplace), use a surface for masking
    
    selectively modify the opacity (transparency) of a surface with the 
    given mask alpha (32-bit surface with channel alpha). 

    `mask_` is a 32-bit pygame.Surface containing RGB(A) color components. 
    The alpha transparency channel is used for masking the input surface.
    The pixels will be hidden if the alpha channel is closing to 0.0 and fully
    opaque when alpha is closing to 255.0

    e.g:
    # Creating the mask alpha
    mask = pygame.image.load('../Assets/alpha.png').convert_alpha()
    mask = pygame.transform.smoothscale(mask, (800, 600))
    
    # in the game loop
    filtering1d_inplace_c(source, mask)
    
    :param surface_: 
        pygame.Surface compatible 24-32 bit

    :param mask_: 
        32-bit Pygame.Surface format RGBA
        The alpha transparency channel(A) is used for masking the input surface.
        The pixels will be hidden if the alpha channel is closing to 0.0 and fully
        opaque when alpha is closing to 255.0

    :return: 
        void
    """

    cdef:
        int w, h, w_, h_
        short int byte_size = surface_.get_bytesize()
        short int byte_size1 = mask_.get_bytesize()

    w, h = surface_.get_size()
    w_, h_ = mask_.get_size()

    if byte_size1 != 4:
        raise AttributeError(
            '\n mask_ attribute must be a 32-bit pygame.Surface with layer alpha.')

    assert w * h == w_ * h_, \
        '\nSurface and mask size does not match ' \
        'surface (w:%s, h:%s), mask (w:%s, h:%s) ' % (w, h, w_, h_)

    cdef:
        unsigned char [::1] bgr = surface_.get_buffer()
        unsigned char [::1] mask = mask_.get_buffer()
        int i
        unsigned char *p1
        float a

    with nogil:

        # Here we are forcing the byte_size to 4 due to get_buffer()
        # get_buffer always returns BGRA pixels format regardless of the input image 24-bit for e.g
        for i in prange(0, w * h * 4, 4, schedule=SCHEDULE, num_threads=THREADS):

            p1 = &bgr[ i ]

            a = <float>mask[ i + 3 ] * <float>ONE_255

            p1[0] = <unsigned char>(p1[0] * a)
            (p1 + 1)[0] = <unsigned char>((p1 + 1)[0] * a)
            (p1 + 2)[0] = <unsigned char>((p1 + 2)[0] * a)


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef inline void bloom_c(
        surface_,
        int threshold_,
        bint fast_ = False,
        object mask_ = None
):
    """
    Apply a bloom effect to a 24/32-bit surface (in-place).

    This function enhances bright areas of an image by applying a bloom effect, 
    which creates a glowing appearance around high-intensity pixels. The effect 
    is applied by downscaling, filtering, and blending multiple layers.

    ### Parameters:
    - **surface_** (`pygame.Surface`):  
      The target surface (must be in 24-bit or 32-bit format).  
      The bloom effect is applied directly to this surface.

    - **threshold_** (`int`):  
      Bright pass filter threshold (`0 - 255`).  
      Pixels with brightness above this value will contribute to the bloom effect.

    - **fast_** (`bool`, optional):  
      Enables a faster approximation of the bloom effect (default: `False`).  
      When `True`, the function processes fewer downscaling layers for improved performance.

    - **mask_** (`numpy.ndarray` or `memoryviewslice`, optional):  
      A `float32` mask of shape `(w, h)` with values in the range `[0.0 - 1.0]`.  
      - A mask filled with `1.0` applies the bloom effect to the entire image.  
      - A mask filled with `0.0` disables the effect entirely.  
      - Intermediate values control selective bloom intensity.  
      If `None`, the effect is applied uniformly across the surface.

    ### Returns:
    - **None** (modifies `surface_` in-place).

    ### Raises:
    - `ValueError`:  
      - If `threshold_` is outside the valid range `[0 - 255]`.
      - If the image is too small for processing.

    ### Example Usage:
    ```python
    # Check demo_bloom_mask.py in the Demo folder
    bloom_c(SCREEN, threshold=128, fast=True, mask=mask_array)
    ```
    """

    # Ensure threshold is within the valid range
    assert 0 <= threshold_ <= 255, "Argument threshold must be in range [0 ... 255]"

    cdef:
        Py_ssize_t w, h  # Surface dimensions
        int bit_size      # Bit depth of the surface
        int w2, h2, w4, h4, w8, h8, w16, h16  # Downscaled dimensions
        bint x2, x4, x8, x16 = False  # Flags to control processing steps

    # Get surface dimensions and bit depth
    w, h = surface_.get_size()
    bit_size = surface_.get_bitsize()

    # Compute downscaled dimensions
    w2, h2   = <int>w >> 1, <int>h >> 1
    w4, h4   = w2 >> 1, h2 >> 1
    w8, h8   = w4 >> 1, h4 >> 1
    w16, h16 = w8 >> 1, h8 >> 1

    # Ensure image is large enough for processing
    if w16 == 0 or h16 == 0:
        raise ValueError(
            "\nImage too small and cannot be processed.\n"
            "Try increasing the image size."
        )

    # Determine which subsampling levels are possible
    x2 = w2 > 0 and h2 > 0
    x4 = w4 > 0 and h4 > 0
    x8 = w8 > 0 and h8 > 0
    x16 = w16 > 0 and h16 > 0

    # Skip processing if the image is too small for the first downscale step
    if not x2:
        return

    # Optimize processing levels for fast mode
    if fast_:
        # Skip x2 and x8 processing for improved speed
        x2, x4, x8, x16 = False, True, False, True

    # Initialize subsampled surfaces
    s2, s4, s8, s16 = None, None, None, None

    # Apply successive downscaling, filtering, and blending steps
    # Each step progressively applies the bloom effect at different levels of detail.

    # Step 1: Downscale x2, apply bright pass filter, blur, then upscale
    if x2:
        s2 = scale(surface_, (w2, h2))      # Downscale
        bpf_c(s2, threshold=threshold_)     # Apply bright pass filter
        blur4bloom_c(s2, npass=2)           # Apply blur
        s2 = smoothscale(s2, (w, h))        # Upscale back
        surface_.blit(s2, (0, 0), special_flags=BLEND_RGB_ADD)  # Blend back

    # Step 2: Downscale x4, apply filters, then upscale
    if x4:
        s4 = scale(surface_, (w4, h4))
        bpf_c(s4, threshold=threshold_)
        blur4bloom_c(s4)
        s4 = smoothscale(s4, (w, h))
        surface_.blit(s4, (0, 0), special_flags=BLEND_RGB_ADD)

    # Step 3: Downscale x8, apply filters, then upscale
    if x8:
        s8 = scale(surface_, (w8, h8))
        bpf_c(s8, threshold=threshold_)
        blur4bloom_c(s8)
        s8 = smoothscale(s8, (w, h))
        surface_.blit(s8, (0, 0), special_flags=BLEND_RGB_ADD)

    # Step 4: Downscale x16, apply filters, then upscale
    if x16:
        s16 = scale(surface_, (w16, h16))
        bpf_c(s16, threshold=threshold_)
        blur4bloom_c(s16)
        s16 = smoothscale(s16, (w, h))
        surface_.blit(s16, (0, 0), special_flags=BLEND_RGB_ADD)

    # Apply optional mask for selective bloom effect
    if mask_ is not None:
        filtering_inplace_c(surface_, mask_)






@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef inline object shader_bloom_fast(
        surface_,
        int threshold,
        bint fast = False,
        unsigned short int factor = 2
):
    """
    Applies a fast bloom effect to an input surface.

    The bloom effect brightens pixels in the image above a specified threshold 
    and then applies blur to create a glowing effect. This function performs a 
    series of downscaling operations and blurs, which are then combined to produce 
    the final bloom effect. 

    Example:
        image = shader_bloom_fast(image, 60)

    Parameters
    ----------
    surface_ : pygame.Surface
        A Pygame surface containing RGB pixel data (32-bit or 24-bit color format).

    threshold : int
        The brightness threshold for the bloom effect. Pixels with values above 
        this threshold will contribute to the bloom effect. A smaller value will 
        cause a stronger bloom.

    fast : bint, optional
        If True, the algorithm will prioritize speed over visual quality by only 
        applying the blur to the lowest downscaled surface (S16). Default is False.

    factor : int, optional
        A value between 0 and 4 that controls the level of downscaling for the 
        textures used in the bloom. Higher values result in more aggressive downscaling 
        (default is 2, which corresponds to a division by 4).

    Returns
    -------
    pygame.Surface
        A Pygame surface with the bloom effect applied, in 24-bit color format.

    Raises
    ------
    ValueError
        If the surface is too small to process (e.g., after downscaling).
    """

    # Check input validity
    assert isinstance(surface_, pygame.Surface), "Argument surface_ must be a pygame.Surface, got %s " % type(surface_)
    assert 0 <= threshold <= 255, "Argument threshold must be in range [0 ... 255], got %s " % threshold
    assert 0 <= factor <= 4, "Argument factor must be in range [0 ... 4], got %s " % factor
    assert isinstance(fast, bool), "Argument fast must be boolean True | False, got %s " % type(fast)

    cdef:
        Py_ssize_t w, h
        int bit_size
        int w2, h2, w4, h4, w8, h8, w16, h16
        bint x2, x4, x8, x16 = False

    # Make a copy of the original surface for manipulation
    cp = surface_.copy()

    # Scale down the surface based on the provided factor
    surface_ = smoothscale(surface_, (surface_.get_width() >> factor, surface_.get_height() >> factor))

    # Get the new width and height after scaling
    w, h = surface_.get_size()
    bit_size = surface_.get_bitsize()

    # Calculate various downscale sizes (x2, x4, x8, x16)
    w2, h2 = w >> 1, h >> 1
    w4, h4 = w2 >> 1, h2 >> 1
    w8, h8 = w4 >> 1, h4 >> 1
    w16, h16 = w8 >> 1, h8 >> 1

    # Check if the image is large enough to process after downscaling
    if w16 == 0 or h16 == 0:
        raise ValueError(
            "\nImage too small and cannot be processed.\n"
            "Try to increase the size of the image or decrease the factor value (default 2)"
        )

    # Set flags to indicate which downscaled versions are usable
    x2 = w2 > 0 and h2 > 0
    x4 = w4 > 0 and h4 > 0
    x8 = w8 > 0 and h8 > 0
    x16 = w16 > 0 and h16 > 0

    # Initialize surfaces for the different downscaled versions
    s2, s4, s8, s16 = None, None, None, None

    # If downscale x2 is not possible, abort the operation
    if not x2:
        return

    # If fast mode is enabled, limit processing to the x16 surface only
    if fast:
        x2, x4, x8 = False, False, False

    # Perform downscale and blur operations for the x2, x4, x8, and x16 surfaces

    # Apply the most expensive downscale (x2)
    if x2:
        s2 = scale(surface_, (w2, h2))
        s2 = bpf24_c(pixels3d(s2), threshold=threshold)
        s2_array = numpy.array(s2.get_view('3'), dtype=numpy.uint8)
        blur3d_c(s2_array)
        b2_blurred = make_surface(s2_array)
        s2 = smoothscale(b2_blurred, (w, h))

    # Apply second most expensive downscale (x4)
    if x4:
        s4 = scale(surface_, (w4, h4))
        s4 = bpf24_c(pixels3d(s4), threshold=threshold)
        s4_array = numpy.array(s4.get_view('3'), dtype=numpy.uint8)
        blur3d_c(s4_array)
        b4_blurred = make_surface(s4_array)
        s4 = smoothscale(b4_blurred, (w, h))

    # Apply third downscale (x8)
    if x8:
        s8 = scale(surface_, (w8, h8))
        s8 = bpf24_c(pixels3d(s8), threshold=threshold)
        s8_array = numpy.array(s8.get_view('3'), dtype=numpy.uint8)
        blur3d_c(s8_array)
        b8_blurred = make_surface(s8_array)
        s8 = smoothscale(b8_blurred, (w, h))

    # Apply the least significant downscale (x16)
    if x16:
        s16 = scale(surface_, (w16, h16))
        s16 = bpf24_c(pixels3d(s16), threshold=threshold)
        s16_array = numpy.array(s16.get_view('3'), dtype=numpy.uint8)
        blur3d_c(s16_array)
        blur3d_c(s16_array)  # Apply blur twice for stronger effect
        b16_blurred = make_surface(s16_array)
        s16 = smoothscale(b16_blurred, (w, h))

    # Combine the downscaled surfaces into the final image
    if fast:
        s16 = smoothscale(s16, (w << factor, h << factor))
        cp.blit(s16, (0, 0), special_flags=BLEND_RGB_ADD)
    else:
        s2.blit(s4, (0, 0), special_flags=BLEND_RGB_ADD)
        s2.blit(s8, (0, 0), special_flags=BLEND_RGB_ADD)
        s2.blit(s16, (0, 0), special_flags=BLEND_RGB_ADD)
        s2 = smoothscale(s2, (w << factor, h << factor))
        cp.blit(s2, (0, 0), special_flags=BLEND_RGB_ADD)

    return cp




@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef void shader_bloom_fast1(
        object surface_,
        unsigned short int smooth_ = 3,
        unsigned int threshold_ = 0,
        unsigned short int flag_ = BLEND_RGB_ADD,
        bint saturation_ = False,
        mask_ = None
):
    """
    Bloom effect inplace (simplified version for better performances)    
    
    The other bloom versions in PygameShader such as bloom and shader_bloom_fast, 
    cause the halo of light to be offset from moving objects due to the re-scaling (down sampling) 
    of the sub-surfaces in addition to the loss of accuracy.
    This version is compatible with moving object in the display since the algorithm do not use 
    the exact same technics. 
    
    The bloom effect can be adjust with the bright pass filter threshold (variable threshold).
    Adjust threshold value for the bloom intensity, zero being the maximum bloom.
    
    The smooth factor (smooth_) will help to spread the light homogeneously around the objects. 
    A large number of smooth will cast the bloom over the entire scene but diminished 
    the overall bloom effect, while a small value will pixelate the hallo around objects 
    but will generate the brightest effect on objects. 
    When smooth is below 3, the halo appear to be slightly pixelated. 
    
    You can use (saturation_) to generate saturated colors within the light bloom effect.
    
    The flag (flag) can be used for special effect with the light within the 
    pygame display. The default value is pygame.BLEND_RGB_ADD and allow to blend 
    the bloom to the display. Option are BLEND_RGB_MAX, BLEND_RGB_SUB etc, refers to Pygame blend 
    attributes. 
    
    This effect is applied inplace
    
    e.g:
    shader_bloom_fast1(image)
    
    :param surface_: 
        pygame.Surface; image compatible 32-24 bit
        
    :param smooth_: 
        integer; Smooth the hallow default 3 (gaussian kernel)
        
    :param threshold_: 
        integer; control the bloom intensity default value 0
        
    :param flag_: 
        integer; pygame flag to use (default is BLEND_RGB_ADD), refers to pygame bend attributes
        
    :param saturation_: 
        bool; True | False include saturation effect to the halo
        
    :param mask_:
        numpy.ndarray or memoryviewslice shape (w, h) type float32 containing values in range
        (0 .. 255) representing the mask alpha. Array (w, h) filled with 255 will render and 
        bloom the entire image. Array (w, h) filled with zero will disable the bloom effect. 
        Any values in between ]0 and 255[ will filter the pixels and create selective bloom effect.
        mask is optional.
    
          
    :return: 
        void
         
    """

    cdef:
        int  w, h
        unsigned int bit_size
        unsigned int  w16, h16
        unsigned int r

    assert isinstance(surface_, pygame.Surface), \
        "Argument surface_ must be a pygame.Surface got %s " % type(surface_)

    if flag_ < 0:
        raise ValueError("Argument flag cannot be < 0")
    if smooth_ < 0:
        raise ValueError("Argument smooth_ cannot be < 0")

    threshold_ %= <unsigned char>255

    w, h = surface_.get_size()
    bit_size = surface_.get_bitsize()

    w16, h16 = w >> 4, h >> 4

    if w16 == 0 or h16 == 0:
        raise ValueError(
            "\nImage too small and cannot be processed.\n"
                "Try to increase the size of the image")


    s2 = smoothscale(surface_, (w16, h16))
    s2.blit(s2, (0, 0), special_flags=BLEND_RGB_ADD)
    # blend_add_surface_c(s2, s2)

    cdef unsigned char [ :, :, : ] s2_array = s2.get_view('3')
    bpf_inplace_c(s2_array, w16, h16, threshold=threshold_)

    for r in range(smooth_):
        blur3d_c(s2_array)
        if saturation_ : saturation_c(s2_array, <float>0.3)

    pygame.surfarray.array_to_surface(s2, asarray(s2_array, dtype=uint8))
    s2 = smoothscale(s2, (w, h))

    if flag_ is not None and flag_!=0:
        surface_.blit(s2, (0, 0), special_flags=flag_)
    else:
        surface_.blit(s2, (0, 0))

    if mask_ is not None:
        filtering_inplace_c(surface_, mask_)


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef inline np.ndarray[np.uint32_t, ndim=2] fisheye_footprint_c(
        const Py_ssize_t w,
        const Py_ssize_t h,
        const unsigned int centre_x,
        const unsigned int centre_y
):

    """
    Create a fisheye lens footprint (return new array) 
    
    Create a fisheye lens model holding pixel coordinates 
    of a surface within a numpy array shape (w, h, 2). Third array dimension holds 
    the texture pixel coordinates (x, y) 

    The model variables w & h must have the same dimensions than the projected texture/surface.  
    
    e.g 
     width, height = surface.get_size()
     f_model = fisheye_footprint_c(w=width, h=height, centre_x=width >> 1, centre_y=height >> 1)
     fisheye(surface, f_model)
        
    Parameters
    ----------
    w : 
        integer; centre position x of the effect
        
    h : 
        integer; centre position y of the effect
        
    centre_x : 
        integer; width of the surface to project ino the fisheye model
         
    centre_y : 
        integer; height of the surface to project into the fisheye model

    Returns
    -------
        Return a numpy.ndarray type (w, h, 2) of unsigned int representing the
        fisheye model (coordinates of all pixels passing through the fisheye lens model)
      
    """

    assert w > 0, "Argument w must be > 0"
    assert h > 0, "Argument h must be > 0"

    cdef:
        unsigned int [:, :, :] image_fisheye_model = numpy.empty((w, h, 2), numpy.uint32)
        int y, x
        float ny, ny2, nx, nx2, r, theta, nr
        float c1 = <float>2.0/w
        float c2 = <float>2.0/h
        float w2 = centre_x
        float h2 = centre_y

    with nogil:

        for x in prange(w, schedule=SCHEDULE, num_threads=THREADS):
            nx = x * c1 - <float>1.0
            nx2 = nx * nx

            for y in range(h):
                ny = y * c2 - <float>1.0
                ny2 = ny * ny

                r = <float>sqrt(nx2 + ny2)

                if r > 1:
                    image_fisheye_model[ x, y, 0 ] = <unsigned int> 0
                    image_fisheye_model[ x, y, 1 ] = <unsigned int> 0
                    continue

                nr = (r + <float>1.0 - <float>sqrt(
                <float>1.0 - (nx2 + ny2))) * <float>0.5

                theta = <float>atan2(ny, nx)
                image_fisheye_model[ x, y, 0 ] = <unsigned int> (nr * <float>cos(theta) * w2 + w2)
                image_fisheye_model[ x, y, 1 ] = <unsigned int> (nr * <float>sin(theta) * h2 + h2)

    return numpy.ndarray(shape=(w, h, 2), dtype=numpy.uint32, order='C', buffer=image_fisheye_model)



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef inline void fisheye_inplace_c(
        unsigned char [:, :, :] rgb_array,
        const unsigned int [:, :, ::1] fisheye_model
):
    """
    Create a fisheye lens effect (inplace)
    
    Display a fisheye lens effect in real time.

    This method takes an array shape (w, h, 3) containing RGB pixel format as input.
    The array reference an SDL pygame.Surface. Any changes to the array's data will affect 
     the surface directly. The input array data must be compatible with 24-bit pixel format, this means 
    that the array shape must be (w, h, 3) format RGB. 
    
    In order to accomplish a real time calculation, this algorithm is using 
    a pre-calculated lens model transformation stored in a numpy.ndarray, 
    argument fisheye_model (numpy.ndarray shape (w, h, 2) of type uint).
    The numpy array contains the pixel's coordinates of a surface after 
    a lens transformation. All calculation are performed upstream. 
    
    Use the function fisheye_footprint_c to create the pre-calculated array.
    This method needs to be called once only. 
    
    The fisheye lens transformation is applied inplace.
    
    e.g 
     width, height = surface.get_size()
     f_model = fisheye_inplace_c(rgb_array, fisheye_model)
     fisheye(surface, f_model)
 
    Parameters
    ----------
    
    rgb_array :
        numpy.ndarray shape (w, h, 3) of type uint8 containing RGB pixels or any other
        pixel format. Any changes made to this array will affect the pygame surface directly.  
        
    fisheye_model : 
        numpy.ndarray shape (w, h, 2) int32, fisheye model containing uint values 
        x' & y'. x' & y' are the surface pixels coordinates after transformation. Values calculated 
        upstream with the function fisheye_footprint_c      

    Returns
    -------
    void
    
    """

    cdef:
        Py_ssize_t w, h

    w, h = rgb_array.shape[:2]

    cdef:
        int x, y
        const unsigned char [::1, :, :] rgb_array_copy = \
            numpy.asarray(rgb_array, order='F')

        unsigned int x2
        unsigned int y2
        unsigned int w_1 = w - 1
        unsigned int h_1 = h - 1
        const unsigned int * f_xy


    with nogil:
        for x in prange(w, schedule=SCHEDULE, num_threads=THREADS):
            for y in range(h):

                f_xy = &fisheye_model[x, y, 0]

                x2 = min(f_xy[0], w_1)
                y2 = min((f_xy+1)[0], h_1)

                if x2==0 and y2==0:
                    continue

                rgb_array[x, y, 0] = rgb_array_copy[x2, y2, 0]
                rgb_array[x, y, 1] = rgb_array_copy[x2, y2, 1]
                rgb_array[x, y, 2] = rgb_array_copy[x2, y2, 2]



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef inline void tv_scanline_c(
        unsigned char [:, :, :] rgb_array,
        int space):

    """
     TV scanline effect on pygame surface

    The space between each scanline can by adjusted with the space value.

    rgb_array - must be a numpy array shape (w, h, 3) type uint8 containing RGB pixels or any 
    other pixel format. Any change made to this array will affect the surface directly. 
    The referenced surface must be format 24 or 32-bit format.
    
    e.g:
    tv_scanline_c(pixels3d(image))
    
    Parameters
    ----------
    
    rgb_array : 
        numpy.ndarray shape (w, h, 3) type uint8 containing RGB pixels or any other pixel format. 
        Any changes made to this array will affect the surface directly 
    
    space : 
        integer; space between each lines, choose a constant or a variable for a dynamic effect

    Returns
    -------
    void
    
    """


    cdef:
        Py_ssize_t w, h
    w, h = rgb_array.shape[:2]

    cdef:
        int x, y, j
        unsigned char *r
        unsigned char *g
        unsigned char *b
        int frame_2 = space >> 1

    with nogil:
        for y in prange(0, h, space, schedule=SCHEDULE, num_threads=THREADS):
            for x in range(w):
                for j in range(frame_2):
                    if y + j < h - 1:
                        r = &rgb_array[x, y + j, <unsigned short int>0]
                        g = &rgb_array[x, y + j, <unsigned short int>1]
                        b = &rgb_array[x, y + j, <unsigned short int>2]
                    else:
                        r = &rgb_array[x, y, <unsigned short int>0]
                        g = &rgb_array[x, y, <unsigned short int>1]
                        b = &rgb_array[x, y, <unsigned short int>2]
                    r[0] = <unsigned char> (r[0] * <float>0.65)
                    g[0] = <unsigned char> (g[0] * <float>0.65)
                    b[0] = <unsigned char> (b[0] * <float>0.65)





@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef object split_channels(
        object surface_,
        char offset_,
        array_ = None):

    """
    RGB split effect (return copy) 
    
    e.g
    im = split_channels(im, 10)
    
    :param surface_: 
        Pygame surface, Image compatible 24 - 32 bit
         
    :param offset_: 
        char; Offset to add between each channels. offset_ must be in range [ -128 to 127] 
        When offset_ > 0 RGB channels are displayed in this order (BGR). 
        If offset_ < 0 RGB channels are display in following order RGB
         
    :param array_: 
        numpy.ndarray shape (w, h, 3) type uint8; (Optional). Speed-up the process when 
        an array is provided
        
    :return: 
        surface (copy) with split channels
       
    """

    return split_channels_c(surface_, offset_, array_)


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef object split_channels_c(
        object surface_,
        char offset_,
        array_ = None):

    """
    RGB split effect (return copy) 

    e.g
    im = split_channels_c(im, 10)

    :param surface_:
        Pygame surface, Image compatible 24 - 32 bit
         
    :param offset_: 
        char; Offset to add between each channels. offset_ must be in range [ -128 to 127] 
        When offset_ > 0 RGB channels are displayed in this order (BGR). 
        If offset_ < 0 RGB channels are display in following order RGB
         
    :param array_: 
        numpy.ndarray shape (w, h, 3) type uint8; (Optional). Speed-up the process when 
        an array is provided
        
    :return: 
        surface (copy) with split channels
        
    """

    cdef:
        Py_ssize_t w, h

    w, h = surface_.get_size()

    cdef unsigned char [:, :, :] rgb
    try:
        rgb   = surface_.get_view('3')

    except Exception as e:
        raise ValueError("\nCannot reference source pixels into a 3d array.\n %s " % e)

    cdef unsigned char [:, :, :] new_rgb

    try:
        new_rgb = empty((h, w, 3), dtype=numpy.uint8) if array_ is None else array_

    except Exception as e:
        raise ValueError("\nCannot reference source pixels into a 3d array.\n %s " % e)

    cdef:
        int i, j
        unsigned int ofs_x_green = w - offset_
        unsigned int ofs_y_green = h - offset_
        unsigned int ofs_x_blue  = w - offset_ * 2
        unsigned int ofs_y_blue  = h - offset_ * 2

    with nogil:

        for i in prange(0, w, schedule=SCHEDULE, num_threads=THREADS):

            for j in range(0, h):
                new_rgb[j, i, 0] = rgb[i, j, 0]
                new_rgb[j, i, 1] = rgb[i + offset_ if i < ofs_x_green else 0, j + offset_ if j < ofs_y_green else 0, 1]
                new_rgb[j, i, 2] = rgb[i + offset_ * 2 if i < ofs_x_blue else 0,
                                       j + offset_ * 2 if j < ofs_y_blue else 0, 2]

    return frombuffer(new_rgb, (w, h), 'RGB')



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef void split_channels_inplace(
        object surface_,
        char offset_,
        array_=None):

    """
    RGB split (inplace)
    
    e.g
    split_channels_inplace(im, 10)
    
    :param surface_: 
        pygame surface, image compatible 24 - 32 bit
         
    :param offset_: 
        char; offset to add between each channels. offset_ must be in range [ -128 to 127] 
        when offset_ > 0 rgb channels are displayed in this order (bgr), 
        offset_ < 0 rgb channels are display in order rgb
         
    :param array_: 
        numpy.ndarray shape (w, h, 3) type uint8; (optional). boost performances when 
        an array is provided
         
    :return: 
        void
        
    """
    split_channels_inplace_c(surface_, offset_, array_)



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef void split_channels_inplace_c(
        object surface_,
        char offset_,
        array_=None):

    """
    RGB split (inplace)
    
    e.g
    split_channels_inplace_c(im, 10)
    
    :param surface_: 
        pygame surface, image compatible 24 - 32 bit
         
    :param offset_: 
        char; offset to add between each channels. offset_ must be in range [ -128 to 127]
        when offset_ > 0 rgb channels are displayed in this order (bgr), offset_ < 0 rgb channels are display
        in order rgb
         
    :param array_: 
        numpy.ndarray shape (w, h, 3) type uint8; (optional). boost performances when 
        an array is provided 
     
    :return : 
        void
    """

    cdef:
        Py_ssize_t w, h

    w, h = surface_.get_size()

    cdef unsigned char [:, :, :] rgb

    try:
        rgb   = surface_.get_view('3')

    except Exception as e:
        raise ValueError("\nCannot reference source pixels into a 3d array.\n %s " % e)

    cdef unsigned char [::1, :, :] rgb_array_copy

    try:
        rgb_array_copy = numpy.asarray(rgb, order='F') if array_ is None else array_

    except Exception as e:
        raise ValueError("\nCannot reference source pixels into a 3d array.\n %s " % e)

    cdef:
        int i, j
        unsigned int ofs_x_green = w - offset_
        unsigned int ofs_y_green = h - offset_
        unsigned int ofs_x_blue  = w - offset_ * 2
        unsigned int ofs_y_blue  = h - offset_ * 2

    with nogil:

        for i in prange(0, w, schedule=SCHEDULE, num_threads=THREADS):

            for j in range(0, h):

                rgb[i, j, 1] = rgb_array_copy[i + offset_ if i < ofs_x_green else 0,
                        j + offset_ if j < ofs_y_green else 0, 1]
                rgb[i, j, 2] = rgb_array_copy[i + offset_ * 2 if i < ofs_x_blue else 0,
                        j + offset_ * 2 if j < ofs_y_blue else 0, 2]






@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef inline tuple ripple_c(
       Py_ssize_t rows_, Py_ssize_t cols_,
       const float [:, ::1] previous_,
       float [:, ::1] current_,
       unsigned char [:, :, :] array_,
       float dispersion_ = 0.008
       ):
    """
    Ripple effect without background deformation

    Check demo_ripple.py, demo_ripple1.py in the Demo folder 
    e.g:
    previous, current = ripple(width, height, previous, current, back_array,  dispersion_=0.008)
  
    rows_ : 
        integer; Screen width or surface width
        
    cols_ : 
        integer; Screen height or surface height
        
    previous_ : 
        numpy.ndarray type (w, h) type float; array use for the transformation. 
        Array holding the previous_ data
          
    current_ : 
        numpy.ndarray type (w, h) type float; array use for the transformation. 
        Array holding the current_ data
         
    array : 
        numpy.ndarray type (w, h, 3) type unsigned char. Array containing the 
        background image RGB pixels.
        The content of this array is invariant (static background image). 
    
    dispersion_ :
        float; ripple dampening factor, higher values decrease the ripple effect 
        radius default 0.008

    Returns
    -------
    Return a tuple containing 2 arrays (current_, previous_)
    see Parameters for each array sizes
        
    """

    cdef:
        int i, j, a, b
        float data
        float *c
        const float *d
        unsigned char *e
        float r
        unsigned int row_1 = rows_ - 1
        unsigned int col_1 = cols_ - 1

    with nogil:

        for j in prange(0, cols_, schedule=SCHEDULE, num_threads=THREADS):
            for i in range(0, rows_):

                data = (previous_[i + 1 if i <row_1 else 0, j]
                        + previous_[i - 1 if i > 1 else 0, j] +
                              previous_[i, j - 1 if j > 1 else 0] +
                        previous_[i, j + 1 if j <col_1 else 0]) * <float>0.5

                c = &current_[i, j]
                data = data - <float>c[0]
                c[0] = data - (data * dispersion_)
                d = &previous_[i,j]
                e = &array_[i, j, 0]
                e[0] = <unsigned char> d[0] if d[0] > 0 else 0
                array_[i, j, 1] = e[0]
                array_[i, j, 2] = e[0]

    return current_, previous_


# Todo this can be simplify with buffer
#  and pointers once all the array are contiguous
@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef inline ripple_seabed_c(
           int rows_, int cols_,
           const float [:, ::1] previous_,                 # type numpy.float32 (w, h)
           float [:, ::1] current_,                        # type numpy.float32 (w, h)
           const unsigned char [:, :, ::1] texture_array_, # type numpy.ndarray (w, h, 3)
           unsigned char [:, :, :] background_array_,      # type numpy.ndarray (w, h, 3)
           float dispersion_ = 0.008
           ):
    """
    Ripple effect with background deformation

    Check demo_ripple_seabed.py in the Demo folder 
    e.g:
    previous, current, back_array = ripple_seabed(height, width, previous,\
                   current, texture_array, back_array, dispersion_=0.009)
    
    Parameters
    ----------
    rows_              : 
        integer; Screen width or surface width
    
    cols_              : 
        integer; Screen height or surface height
    
    previous_          : 
        numpy.ndarray type (w, h) type float; array use for the transformation. 
        Array holding the previous_ data
          
    current_           : 
        numpy.ndarray type (w, h) type float; array use for the transformation. 
        Array holding the current_ data
         
    texture_array_     : 
        numpy.ndarray type (w, h, 3) type unsigned char. 
        Array containing the background image RGB pixels.
        The content of this array is invariant (static background image).
         
    background_array_  : 
        numpy.ndarray type (w, h, 3) of type unsigned char containing the background image RGB pixels.
        The background array is equivalent to the texture array with current_ ripple effect transformation.
    
    dispersion_        :  
        float; ripple dampening factor, higher values decrease the ripple effect 
        radius default 0.008

    Returns
    -------
    Return a tuple containing 3 arrays (current_, previous_, bck_array)
    see Parameters for each array sizes
        
    """

    cdef:
        float cols2 = cols_ >> 1
        float rows2 = rows_ >> 1
        int i, j
        unsigned int a, b
        unsigned int cols_1 = cols_ - 1
        unsigned int rows_1 = rows_ - 1
        float data
        unsigned char * index

    # from 1 to w - 1 to avoid python wraparound error
    # same for j (1 to h - 1)
    with nogil:
        for j in prange(1, cols_1, schedule=SCHEDULE, num_threads=THREADS):
            for i in range(1, rows_1):

                # data = (previous_[i + 1, j] + previous_[i - 1, j] +
                #         previous_[i, j - 1] + previous_[i, j + 1]) * <float>0.5

                data = (previous_[i + 1 if i < rows_1 else 0, j]
                        + previous_[i - 1 if i > 1 else 0, j] +
                        previous_[i, j - 1 if j > 1 else 0] +
                        previous_[i, j + 1 if j < rows_1 else 0]) * <float> 0.5


                data -= current_[i, j]
                data -= data * dispersion_
                current_[i, j] = data
                data = <float>1.0 - data * <float>ONE_1024
                a = max(<int>(((i - rows2) * data) + rows2) % rows_, 0)
                b = max(<int>(((j - cols2) * data) + cols2) % cols_, 0)

                background_array_[i, j, 0], background_array_[i, j, 1], background_array_[i, j, 2] = \
                    texture_array_[a, b, 0], texture_array_[a, b, 1], texture_array_[a, b, 2]

    return current_, previous_, background_array_



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef (int, int, int) wavelength2rgb(int wavelength, float gamma=1.0):
    """
    Color wavelength to RGB
    
    Return tuple of rgb components corresponding to a specific color wavelength
    wavelength_to_rgb is an External C routine with pre-defined wavelength such as :
    
    Color   Wavelength(nm) Frequency(THz)
    Red     620-750        484-400
    Orange  590-620        508-484
    Yellow  570-590        526-508
    Green   495-570        606-526
    Blue    450-495        668-606
    Violet  380-450        789-668
    
    e.g If the routine is called with a wavelength of 620, the returned color
    will be a red gradient.
    
    # Return orange color 255, 137, 0 (RGB)
    wavelength2rgb(610) 

    :param wavelength: 
        integer; Wavelength
        
    :param gamma: 
        float; Gamma value
        
    :return: 
        tuple RGB values uint8 (0 ... 255)
        
    
    """

    cdef  rgb_color_int rgb_c
    rgb_c = wavelength_to_rgb(wavelength, gamma)
    return rgb_c.r, rgb_c.g, rgb_c.b


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef tuple custom_map(int wavelength, int [::1] color_array, float gamma=1.0):

    """
    
    Return tuple RGB components corresponding to a customized wavelength domain.
    unlike wavelength2rgb, this method will return a tuple RGB color components corresponding 
    to a wavelength defined within a customized wavelength domain.
    
    see - demo_fire for a real time presentation.
    
    # example of customized nm wavelength. 
    # from 0-1 nm wavelength, the RGB components will be mostly black 
    # from 2 - 619 nm, the RGB components will be yellow 
    arr = numpy.array(
        [0, 1,       # violet is not used
         0, 1,       # blue is not used
         0, 1,       # green is not used
         570, 619,     # yellow, return a yellow gradient for values [2...619]
         620, 650,   # orange return a orange gradient for values [620 ... 650]
         651, 660    # red return a red gradient for values [651 ... 660]
         ], numpy.int32)

    heatmap = [custom_map(i - 20, arr, 1.0) for i in range(380, 800)]
    
    Parameters
    ----------
    
    wavelength : 
        integer; Wavelength in nm 
    
    color_array : 
        numpy.array (buffer) containing the min and max of each color (red,
        orange, yellow, green, blue, violet)
    
    gamma : 
        float; Gamma value
        
    Returns
    -------
    tuple RGB uint values in range (0 ... 255)
    
    """

    cdef  rgb_color_int rgb_c
    cdef int *p
    p = &color_array[0]
    rgb_c = wavelength_to_rgb_custom(wavelength, p, gamma)
    return rgb_c.r, rgb_c.g, rgb_c.b




cdef int i = 0
HEATMAP = [ wavelength2rgb(i, 1.0) for i in range(380, 750) ]

cdef float f_map = (<float>750.0 - <float>380.0 -<float>1.0) / (<float>255.0 * <float>3.0)

cdef unsigned char[:, ::1] heatmap_array = numpy.zeros((750 - 380, 3), numpy.uint8)
cdef tuple t
i = 0
for t in HEATMAP:
    heatmap_array[i, 0] = t[0]
    heatmap_array[i, 1] = t[1]
    heatmap_array[i, 2] = t[2]
    i += 1



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef inline void heatmap_c(object surface_, bint rgb_=True):
    """
    Transform an image into a heatmap equivalent

    e.g: 
    # for 24, 32-bit
    image = pygame.image.load("../Assets/px.png").convert_alpha()
    heatmap_c(pixels3d(image), True)
    
    :param surface_: 
        pygame.Surface 24, 32-bit 
        
    :param rgb_: 
        boolean; True transformed the image into a RGB heatmap model of False (BGR)
        
    :return: 
        void
    """

    cdef:
        unsigned int w, h
    w, h = surface_.get_size()

    cdef:
        unsigned char [::1] bgr_array  = surface_.get_buffer()
        unsigned int s
        int i
        int size = bgr_array.size
        unsigned int index = 0
        unsigned char *r
        unsigned char *g
        unsigned char *b
        short int bitsize = surface_.get_bitsize()
        short int bytesize = surface_.get_bytesize()

    with nogil:
        # RGB
        if rgb_:
            for i in prange(0, size, bytesize, schedule=SCHEDULE, num_threads=THREADS):

                # RGB model
                r = &bgr_array[i]
                g = &bgr_array[i + <unsigned short int>1]
                b = &bgr_array[i + <unsigned short int>2]

                s = r[0] + g[0] + b[0]
                index = <int>(s * f_map)

                #RGB 
                r[0] = <unsigned char>heatmap_array[index, <unsigned short int>0]
                g[0] = <unsigned char>heatmap_array[index, <unsigned short int>1]
                b[0] = <unsigned char>heatmap_array[index, <unsigned short int>2]
        # BGR
        else:
            for i in prange(0, size, bytesize, schedule=SCHEDULE, num_threads=THREADS):

                r = &bgr_array[i]
                g = &bgr_array[i + <unsigned short int>1]
                b = &bgr_array[i + <unsigned short int>2]

                s = r[0] + g[0] + b[0]
                index = <int>(s * f_map)

                # BGR
                r[0] = <unsigned char>heatmap_array[index, <unsigned short int>2]
                g[0] = <unsigned char>heatmap_array[index, <unsigned short int>1]
                b[0] = <unsigned char>heatmap_array[index, <unsigned short int>0]


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef (int, int, int) blue_map(int wavelength, float gamma=1.0):
    """
    Return tuple RGB components matching a specific wavelength in nm
    
    e.g
    rgb = blue_map(600)
    
    :param wavelength: 
        integer; Wavelength in nm
        
    :param gamma: 
        float; Gamma value
        
    :return: 
        tuple RGB uint8 values (0 ... 255)
    
    """

    cdef  rgb_color_int rgb_c
    rgb_c = wavelength_to_rgb(wavelength, gamma)
    return rgb_c.r, rgb_c.g, rgb_c.b

i = 0
BLUEMAP = [ blue_map(i, 1.0) for i in range(450, 495) ]

cdef float f_bluemap = (<float>495.0 - <float>450.0 - <float>1.0) / (<float>255.0 * <float>3.0)

cdef unsigned char[:, ::1] bluemap_array = numpy.zeros((495 - 450, 3), numpy.uint8)

i = 0
for t in BLUEMAP:
    bluemap_array[i, 0] = t[0]
    bluemap_array[i, 1] = t[1]
    bluemap_array[i, 2] = t[2]
    i += 1



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef inline void bluescale(object surface_):
    """
       
    Map an image into a blue equivalent format
    
    this algorithm is using the wavelength from 450-495 nm to
    represent the image in blue shades

    e.g:
    bluescale(image)
    
    :param surface_: 
        pygame.Surface to transform
        
    :return: 
        void
    
    """

    cdef:
        unsigned int w, h
    w, h = surface_.get_size()

    cdef:
        unsigned char [::1] bgr_array  = surface_.get_buffer()
        unsigned int s
        int i
        int size = bgr_array.size
        unsigned int index = 0
        unsigned char *r
        unsigned char *g
        unsigned char *b
        short int bytesize = surface_.get_bytesize()

    with nogil:
        for i in prange(0, size, bytesize, schedule=SCHEDULE, num_threads=THREADS):

            r = &bgr_array[i]
            g = &bgr_array[i + <unsigned short int>1]
            b = &bgr_array[i + <unsigned short int>2]

            s = r[0] + g[0] + b[0]
            index = <unsigned int>(s * f_bluemap)

            r[0] = <unsigned char>bluemap_array[index, <unsigned short int>2]
            g[0] = <unsigned char>bluemap_array[index, <unsigned short int>1]
            b[0] = <unsigned char>bluemap_array[index, <unsigned short int>0]



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef (int, int, int) red_map(int wavelength, float gamma=1.0):
    """
    Return tuple RGB components matching a specific wavelength in nm
    
    e.g:
    rgb = red_map(610)
    
    :param wavelength: 
        integer; Wavelength in nm
        
    :param gamma: 
        float; Gamma value
        
    :return: 
        tuple RGB uint8 values (0 ... 255)
    
    """

    cdef  rgb_color_int rgb_c
    rgb_c = wavelength_to_rgb(wavelength, gamma)
    return rgb_c.r, rgb_c.g, rgb_c.b

i = 0
REDMAP = [ red_map(i, 1.0) for i in range(620, 750) ]

cdef float f_redmap = (<float>750.0 - <float>620.0 - <float>1.0) / (<float>255.0 * <float>3.0)

cdef unsigned char[:, ::1] redmap_array = numpy.zeros((750 - 620, 3), numpy.uint8)

i = 0
for t in REDMAP:
    redmap_array[i, 0] = t[0]
    redmap_array[i, 1] = t[1]
    redmap_array[i, 2] = t[2]
    i += 1



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef inline void redscale(object surface_):
    """
    Apply a redscale effect to an image.

    This algorithm maps the input image into shades of red by adjusting the 
    color channels according to wavelengths typically associated with the 
    red portion of the visible light spectrum, ranging from 620 to 750 nm. 
    The redscale effect retains the intensity variations of the red color while 
    reducing or removing the influence of other color channels.

    Example:
        redscale(image)

    Parameters
    ----------
    surface_ : pygame.Surface
        A Pygame surface representing the image to be transformed into redscale.
        The surface must contain valid RGB pixel data.

    Returns
    -------
    void
        This function modifies the input surface directly and does not return a new surface.
    """


    cdef:
        unsigned int w, h
    w, h = surface_.get_size()

    cdef:
        unsigned char [::1] bgr_array  = surface_.get_buffer()
        unsigned int s
        int i
        int size = bgr_array.size
        unsigned int index = 0
        unsigned char *r
        unsigned char *g
        unsigned char *b
        short int bytesize = surface_.get_bytesize()

    with nogil:
        for i in prange(0, size, bytesize, schedule=SCHEDULE, num_threads=THREADS):

            r = &bgr_array[i]
            g = &bgr_array[i + <unsigned short int>1]
            b = &bgr_array[i + <unsigned short int>2]

            s = r[0] + g[0] + b[0]
            index = <unsigned int>(s * f_redmap)

            r[0] = <unsigned char>redmap_array[index, <unsigned short int>2]
            g[0] = <unsigned char>redmap_array[index, <unsigned short int>1]
            b[0] = <unsigned char>redmap_array[index, <unsigned short int>0]



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef inline void blood_inplace_c(
        Py_ssize_t w,
        Py_ssize_t h,
        unsigned char [:, :, :] rgb_array,
        const float [:, :] mask,
        float percentage)nogil:

    """
    Blood effect (inplace)
    
    The mask array determines the contour used for the blood effect.
    Compatible with surface 24, 32-bit format.
    
    rgb_array and mask must have the same dimensions.
    
    e.g 
    background = pygame.image.load("../Assets/Aliens.jpg").convert()
    background = pygame.transform.smoothscale(background, (800, 600))
    image = background.copy()
    blood_surface = pygame.image.load("../Assets/redvignette.png").convert_alpha()
    blood_surface = pygame.transform.smoothscale(blood_surface, (800, 600))
    BLOOD_MASK = numpy.asarray(pygame.surfarray.pixels_alpha(blood_surface) / 255.0, numpy.float32)
    
    # Then call the method in your main loop (percentage must vary overtime)
    blood_inplace_c(800, 600, pixels3d(image), BLOOD_MASK, percentage)
    
    
    Parameters
    ----------
    
    w : 
        width of the array 
    
    h : 
        height of the array 
    
    rgb_array : 
        numpy.ndarray or memoryviewslice shape (w, h, 3) type uint8 containing RGB pixels or any other 
        pixel format. Any changes to this array will modify the surface directly
    
    mask : 
        Normalised numpy.ndarray or cython memoryviewslice shape (w, h) of type float. 
        Values must be float in range [0.0...1.0].
        
    percentage : 
        float; Percentage value in range [0.0 ... 1.0] with 1.0 being 100%
        
    Returns
    -------
        void
    """
    assert 0.0 <= percentage <= 1.0, \
        "percentage variable must be in range[0.0 ... 1.0] got %s " % percentage

    cdef:
        unsigned int s
        int i, j
        unsigned int index = 0
        unsigned char *r
        unsigned char *g
        unsigned char *b
        float theta

    for j in prange(0, h, schedule=SCHEDULE, num_threads=THREADS):
        for i in range(0, w):

            r = &rgb_array[i, j, <unsigned short int>0]

            index = <int>(r[0] * f_redmap)
            theta = <float>(mask[i, j] * percentage)

            # ALTERNATIVE WITH BEST PERFORMANCES
            r[0] = <unsigned char> (
                min(r[0] + <float> redmap_array[index, 0] * theta, <unsigned char>255))



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef inline unsigned char [:, :, :] mirroring_c(
        Py_ssize_t w,
        Py_ssize_t h,
        const unsigned char[:, :, :] rgb_array,
        unsigned char [:, :, :] new_array
) nogil:
    """
    Apply a horizontal mirroring effect to a given RGB array.

    This function mirrors the input RGB array along the vertical axis (horizontally) 
    and stores the result in a new array. The mirrored pixels are written to the 
    `new_array`, which should have the same shape and size as the input array.

    Example:
        rgb_array = mirroring_c(800, 600, bgr_array, new_array)
        surface = make_surface(rgb_array)

    Parameters
    ----------
    w : int
        The width of the image array, typically the number of columns in the input array.

    h : int
        The height of the image array, typically the number of rows in the input array.

    rgb_array : numpy.ndarray or memoryviewslice
        A 3D array or memoryview of shape (w, h, 3) containing pixel data, where 
        each pixel is represented by three values (RGB or BGR format). The array must 
        be of type `uint8`.

    new_array : numpy.ndarray or memoryviewslice
        A 3D array or memoryview of shape (w, h, 3) that will store the mirrored 
        result. It should have the same shape and type as `rgb_array`, but initially 
        should be empty.

    Returns
    -------
    memoryviewslice
        A 3D memoryviewslice of shape (w, h, 3) containing the mirrored pixel data.
        This array will be filled with the horizontally mirrored pixels from the 
        `rgb_array`.

    Notes
    -----
    - The function modifies `new_array` in place and does not create a new array.
    - This function operates under the `nogil` context to improve performance when 
      used in multi-threaded environments.
    """


    cdef:
        int x, y
        int x2, x3
        const unsigned char *r
        const unsigned char *g
        const unsigned char *b


    for x in prange(w, schedule=SCHEDULE, num_threads=THREADS):

        x2 = x >> 1
        x3 = <int> w - x2 - 1

        for y in range(h):

            r = &rgb_array[x, y, 0]
            g = &rgb_array[x, y, 1]
            b = &rgb_array[x, y, 2]

            new_array[x2, y, 0] = r[0]
            new_array[x2, y, 1] = g[0]
            new_array[x2, y, 2] = b[0]

            new_array[x3, y, 0] = r[0]
            new_array[x3, y, 1] = g[0]
            new_array[x3, y, 2] = b[0]

    return new_array



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef inline void mirroring_inplace_c(
        const Py_ssize_t w,
        const Py_ssize_t h,
        unsigned char[:, :, :] rgb_array,
        const unsigned char [:, :, :] rgb_array_cp
) nogil:
    """
    Apply an in-place horizontal mirroring effect to the given RGB array.

    This function performs horizontal mirroring of the input `rgb_array` by 
    copying the mirrored data from the `rgb_array_cp` into `rgb_array` itself. 
    The original `rgb_array` is modified directly, and no new array is returned.

    Example:
        image_copy = image.copy()
        mirroring_inplace_c(800, 600, image.get_buffer(), image_copy.get_buffer())

    Parameters
    ----------
    w : int
        The width of the image array, representing the number of columns in the 
        input array.

    h : int
        The height of the image array, representing the number of rows in the 
        input array.

    rgb_array : numpy.ndarray or memoryviewslice
        A 3D array or memoryview of shape (w, h, 3) containing the pixel data 
        (RGB or BGR format). This array will be modified directly by the function.

    rgb_array_cp : numpy.ndarray or memoryviewslice
        A 3D array or memoryview of shape (w, h, 3) containing the original 
        pixel data. This serves as the source for the mirrored pixels and is 
        not modified by the function.

    Returns
    -------
    void
        This function modifies the `rgb_array` in-place and does not return any value.

    Notes
    -----
    - The function operates in-place, meaning that it modifies the original `rgb_array` directly.
    - The `rgb_array_cp` serves as a reference and should contain the original (non-mirrored) pixel data.
    - This function is designed to be used in performance-sensitive applications, with the `nogil` context allowing 
      it to run efficiently in multi-threaded environments.
    """



    cdef:
        int x, y
        int x2, x3
        const unsigned char *r
        const unsigned char *g
        const unsigned char *b

    for x in prange(w, schedule=SCHEDULE, num_threads=THREADS):

        x2 = x >> 1
        x3 = <int> w - x2 - 1

        for y in range(h):

            r = &rgb_array_cp[x, y, 0]
            g = &rgb_array_cp[x, y, 1]
            b = &rgb_array_cp[x, y, 2]


            rgb_array[x2, y, 0] = r[0]
            rgb_array[x2, y, 1] = g[0]
            rgb_array[x2, y, 2] = b[0]

            rgb_array[x3, y, 0] = r[0]
            rgb_array[x3, y, 1] = g[0]
            rgb_array[x3, y, 2] = b[0]



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef tuple dampening(
        object surface_,
        int frame,
        int display_width,
        int display_height,
        float amplitude = 50.0,
        int duration = 30,
        float freq = 20.0):

    """
    Apply a dampening effect to a surface.

    This effect simulates a gradual scaling or shrinking of the surface based on a 
    damped oscillation function. The length of the effect is determined by the product 
    of duration and frequency. The position of the surface is adjusted according to 
    its new size to maintain its centered position on the display.

    Example:
        surf, xx, yy = dampening(BCK, frame, w, h, amplitude=100, duration=40, freq=15)
        SCREEN.blit(surf, (xx, yy))

    Parameters
    ----------
    surface_ : pygame.Surface
        A Pygame surface that is compatible with 24-32 bit color depth.

    frame : int
        The current frame number in the animation sequence. This should be incremented 
        with each frame update.

    display_width : int
        The width of the game display window.

    display_height : int
        The height of the game display window.

    amplitude : float, optional
        The amplitude of the dampening effect, which determines the maximum amount 
        of scaling (default is 50.0).

    duration : int, optional
        The duration of the effect, which controls how long the effect lasts 
        in terms of frames (default is 30).

    freq : float, optional
        The frequency of the dampening effect, which affects how fast the scaling 
        oscillates. A smaller value will make the effect last longer, while a larger 
        value shortens the effect (default is 20.0).

    Returns
    -------
    Tuple
        A tuple containing:
            - A new Pygame Surface with the dampening effect applied.
            - The x-coordinate of the new position of the surface (top-left corner).
            - The y-coordinate of the new position of the surface (top-left corner).
            The surface is centered in the display area.
    """

    # Ensure that frequency and duration are positive
    assert freq > 0.0, "Argument freq must be > 0"
    assert duration > 0.0, "Argument duration must be > 0"

    # Calculate dampened oscillation effect based on the frame number
    cdef float t = damped_oscillation(<float>((<float>frame / freq) % duration))
    cdef int width, height
    cdef float tm = t * amplitude

    # Get the current width and height of the surface
    width, height = surface_.get_size()

    # Ensure that the surface does not shrink below a certain size
    if width + tm < 0:
        tm = 0
    if height + tm < 0:
        tm = 0

    # Apply the dampening effect by scaling the surface
    cdef object surf = smoothscale(
        surface_, 
        (<int>tm + <int>(width + <int>tm), 
         <int>tm + <int>(height + <int>tm))
    )

    # Get the new size of the scaled surface
    cdef int new_width, new_height
    new_width, new_height = surf.get_size()

    # Calculate the difference in position to center the surface
    cdef int diff_x = display_width - new_width
    cdef int diff_y = display_height - new_height

    # Return the modified surface and its centered position (x, y)
    return surf, diff_x >> 1, diff_y >> 1



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(False)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef inline float lateral_dampening(
        int frame,
        float amplitude = 50.0,
        int duration = 30,
        float freq = 20.0):

    """
    Apply lateral dampening effect to produce horizontal displacement.

    This method calculates the lateral displacement (x-coordinate) based on a dampened 
    oscillation function. The displacement value oscillates between positive and negative 
    values, gradually decaying according to the amplitude, frequency, and duration parameters.

    Example:
        tm = lateral_dampening(frame, amplitude=50.0, duration=35, freq=5.0)
        SCREEN.blit(BCK, (tm, 0), special_flags=0)

    Parameters
    ----------
    frame : int
        The current frame number in the animation sequence. This value must be incremented 
        with each frame to produce smooth animation.

    amplitude : float, optional
        The amplitude of the lateral dampening effect. This value controls the maximum 
        displacement of the surface. A higher value results in larger horizontal movement. 
        Default is 50.0.

    duration : int, optional
        The total duration of the effect, in terms of frames. This defines how long the 
        oscillations will last. The default value is 30 frames.

    freq : float, optional
        The frequency of the dampening oscillation. This controls how fast the oscillations 
        occur. A lower value makes the effect take longer to complete (slower oscillation), 
        while a higher value speeds up the oscillation. Default is 20.0.

    Returns
    -------
    float
        The lateral displacement value (x) that can be used to shift the object horizontally 
        on the screen (e.g., when blitting an image). The value will oscillate within a 
        range determined by the amplitude.

    Notes
    -----
    - The displacement follows a damped oscillation model, where the value decays 
      over time based on the frequency and duration parameters.
    """

    # Ensure that frequency and duration are positive to avoid invalid values
    assert freq > 0, "Argument freq must be > 0"
    assert duration > 0, "Argument duration must be > 0"

    # Calculate the dampened oscillation value based on the current frame
    # The damped_oscillation function should return a value that simulates 
    # oscillations over time, based on frequency and duration.
    cdef float t = damped_oscillation(<float>((<float>frame / freq) % duration)) * amplitude

    # Return the lateral displacement value (x)
    return t


# --------------------------------------------------------------------------------------------------------
# KERNEL DEFINITION FOR SHARPEN ALGORITHM
cdef const float [:, ::1] SHARPEN_KERNEL = numpy.array(([0, -1, 0],
                      [-1, 5, -1],
                      [0, -1, 0]), order='C').astype(dtype=float32)
cdef int HALF_KERNEL = <int>len(SHARPEN_KERNEL) >> 1



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef inline void sharpen_inplace_c(unsigned char [:, :, :] rgb_array):
    """
    Apply a sharpen effect on an image using a 3x3 convolution kernel (inplace).

    This function modifies the provided image (rgb_array) in place by applying a 
    sharpen filter using a 3x3 convolution kernel. The sharpen effect increases the 
    contrast of the image by emphasizing edges. The kernel used is as follows:

    [ 0, -1,  0 ]
    [-1,  5, -1 ]
    [ 0, -1,  0 ]

    The kernel works by increasing the central pixel's value (multiplied by 5) while 
    subtracting values from its immediate neighbors (multiplied by -1). This sharpens 
    the image by increasing edge contrasts.

    Note:
    - The function works in-place, so the original `rgb_array` is modified.
    - Pixels on the image edges are handled by setting them to adjacent edge values.

    Example:
        sharpen_inplace_c(bgr_array)

    Parameters
    ----------
    rgb_array : numpy.ndarray or memoryviewslice
        A 3D numpy array or memoryviewslice with shape (w, h, 3|4) where w and h 
        are the image width and height, and the third dimension contains RGB, RGBA, 
        BGR, or BGRA pixel values in uint8 format. The array is modified in place.

    Returns
    -------
    void
        The function modifies the `rgb_array` in place and does not return any value.
    """

    # texture sizes
    cdef Py_ssize_t w, h
    w, h = rgb_array.shape[:2]

    cdef:

        unsigned char [:, :, :] rgb_array_1 = numpy.empty((w, h, 3), uint8)
        int x, y, xx, yy
        short kernel_offset_y, kernel_offset_x
        float r, g, b
        const float * k
        unsigned char *rr
        unsigned char *gg
        unsigned char *bb
        int w_1 = <int>w - 1
        int h_1 = <int>h - 1

    with nogil:

        for y in prange(h, schedule=SCHEDULE, num_threads=THREADS):

            for x in range(w):

                rr = &rgb_array_1[x, y, 0]
                gg = &rgb_array_1[x, y, 1]
                bb = &rgb_array_1[x, y, 2]

                r, g, b = 0, 0, 0

                for kernel_offset_y in range(-HALF_KERNEL, HALF_KERNEL + 1):

                    yy = y + kernel_offset_y

                    if yy < 0:
                        yy = <unsigned short int> 0

                    if yy > h_1:
                        yy = h_1

                    for kernel_offset_x in range(
                            -HALF_KERNEL, HALF_KERNEL + 1):

                        xx = x + kernel_offset_x

                        if xx < 0:
                            xx = 0

                        if xx > w_1:
                            xx = w_1

                        k = &SHARPEN_KERNEL[kernel_offset_y + HALF_KERNEL,
                                           kernel_offset_x + HALF_KERNEL]
                        if k[0] != 0.0:
                            r = r + rgb_array[xx, yy, 0] * k[0]
                            g = g + rgb_array[xx, yy, 1] * k[0]
                            b = b + rgb_array[xx, yy, 2] * k[0]

                if r < 0:
                    r = <float>0

                if g < 0:
                    g = <float>0

                if b < 0:
                    b = <float>0

                if r > 255:
                    r= <float>255

                if g > 255:
                    g = <float>255

                if b > 255:
                    b = <float>255

                rr[0] = <unsigned char>r
                gg[0] = <unsigned char>g
                bb[0] = <unsigned char>b

        for y in prange(h, schedule=SCHEDULE, num_threads=THREADS):

            for x in range(w):

                rgb_array[x, y, 0] = rgb_array_1[x, y, 0]
                rgb_array[x, y, 1] = rgb_array_1[x, y, 1]
                rgb_array[x, y, 2] = rgb_array_1[x, y, 2]



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef inline void sharpen32_inplace_c(unsigned char [:, :, :] rgba_array_):
    """
    Apply a sharpen effect to an image using a 3x3 convolution kernel (in-place).

    This function modifies the provided image (rgba_array_) in place by applying 
    a sharpen filter using a 3x3 convolution kernel. The sharpen effect enhances 
    the contrast by emphasizing edges. The kernel used is:

    [ 0, -1,  0 ]
    [-1,  5, -1 ]
    [ 0, -1,  0 ]

    The kernel works by multiplying the central pixel by 5 and subtracting values 
    from its immediate neighbors (multiplied by -1). This sharpen filter emphasizes 
    edge details by increasing contrast around the edges.

    Note:
    - The operation is performed in-place, meaning the original `rgba_array_` will 
      be modified directly.
    - Pixels on the edges of the image are handled by replicating the adjacent edge values.

    Example:
        sharpen32_inplace_c(rgba_array)

    Parameters
    ----------
    rgba_array_ : numpy.ndarray or memoryviewslice
        A 3D numpy array or memoryviewslice with shape (w, h, 4) where `w` and `h` 
        are the image's width and height, and the third dimension contains RGBA pixel 
        values (each channel being uint8). The array will be modified in place.

    Returns
    -------
    void
        The function modifies the `rgba_array_` in place and does not return any value.
    """

    # texture sizes
    cdef Py_ssize_t w, h
    w, h = rgba_array_.shape[:2]

    cdef:

        unsigned char [:, :, :] rgba_array_1 = numpy.empty((w, h, 4), uint8)
        int x, y, xx, yy
        short kernel_offset_y, kernel_offset_x
        float r, g, b
        const float * k
        unsigned char *rr
        unsigned char *gg
        unsigned char *bb
        unsigned char *aa
        int w_1 = <int>w - 1
        int h_1 = <int>h - 1

    with nogil:

        for y in prange(h, schedule=SCHEDULE, num_threads=THREADS):

            for x in range(w):

                aa = &rgba_array_[ x, y, 3 ]
                # skip transparent pixels
                if aa[0] == 0:
                    continue

                rr = &rgba_array_1[x, y, 0]
                gg = &rgba_array_1[x, y, 1]
                bb = &rgba_array_1[x, y, 2]


                r, g, b = 0, 0, 0

                for kernel_offset_y in range(-HALF_KERNEL, HALF_KERNEL + 1):

                    yy = y + kernel_offset_y

                    if yy < 0:
                        yy = <unsigned short int> 0

                    if yy > h_1:
                        yy = h_1

                    for kernel_offset_x in range(
                            -HALF_KERNEL, HALF_KERNEL + 1):

                        xx = x + kernel_offset_x

                        if xx < 0:
                            xx = 0

                        if xx > w_1:
                            xx = w_1

                        k = &SHARPEN_KERNEL[kernel_offset_y + HALF_KERNEL,
                                            kernel_offset_x + HALF_KERNEL]
                        if k[0] != 0.0:
                            r = r + rgba_array_[xx, yy, 0] * k[0]
                            g = g + rgba_array_[xx, yy, 1] * k[0]
                            b = b + rgba_array_[xx, yy, 2] * k[0]

                if r < 0:
                    r = <float>0

                if g < 0:
                    g = <float>0

                if b < 0:
                    b = <float>0

                if r > 255:
                    r= <float>255

                if g > 255:
                    g = <float>255

                if b > 255:
                    b = <float>255

                rr[0] = <unsigned char>r
                gg[0] = <unsigned char>g
                bb[0] = <unsigned char>b


        for y in prange(h, schedule=SCHEDULE, num_threads=THREADS):

            for x in range(w):

                rgba_array_[x, y, 0] = rgba_array_1[x, y, 0]
                rgba_array_[x, y, 1] = rgba_array_1[x, y, 1]
                rgba_array_[x, y, 2] = rgba_array_1[x, y, 2]





# Added to version 1.0.1
@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef cartoon_c(
        object surface_,
        unsigned int sobel_threshold = 128,
        unsigned int median_kernel   = 2,
        unsigned int color           = 8,
        unsigned int flag            = BLEND_RGB_ADD):
    """
    Apply a cartoon effect to an image.

    This function applies a series of image processing techniques to simulate a 
    cartoon-style effect on an input image. It combines edge detection (using Sobel), 
    median filtering for smoothing, and color reduction to achieve a cartoonish look.

    Example:
        cartoon_image = cartoon_c(image)

    Parameters
    ----------
    surface_ : pygame.Surface
        A pygame.Surface object compatible with 24-bit or 32-bit images (RGB or RGBA).

    sobel_threshold : int, optional
        The threshold value for Sobel edge detection (default is 128). 
        Higher values result in stronger edge detection.

    median_kernel : int, optional
        The size of the kernel used for median filtering (default is 2).
        Larger values result in greater smoothing.

    color : int, optional
        The number of colors to reduce the image to (default is 8).
        This controls the level of color reduction to create a "posterized" effect.

    flag : int, optional
        The blending flag used for the final image composition (default is BLEND_RGB_ADD).
        It determines how the processed image is combined with the original.

    Returns
    -------
    pygame.Surface
        A pygame.Surface object with the cartoon effect applied.
    """
    
    # Create a copy of the surface to work with
    surface_branch_1 = surface_.copy()

    # Attempt to get a 3D array view of the pixel data
    try:
        array_ = surface_branch_1.get_view('3')
    except Exception as e:
        raise ValueError(
            "\nCannot reference source pixels into a 3D array.\n %s " % e)

    # Apply Sobel edge detection to the image to highlight edges
    sobel_inplace_c(array_, sobel_threshold)

    # Apply median filtering to smooth the image and reduce noise
    median_fast(surface_, kernel_size_=median_kernel, reduce_factor_=1)

    # Transfer the processed array back to the surface
    pygame.surfarray.array_to_surface(surface_branch_1, array_)

    # Clean up the array reference after use
    del array_

    # Blend the processed image with the original surface using the specified blend flag
    surface_.blit(surface_branch_1, (0, 0), special_flags=flag)

    # Posterize the surface to reduce the number of colors (to simulate cartoon shading)
    posterize_surface(surface_, color)

    # Return the surface with the cartoon effect applied
    return surface_


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef object blend_c(
        unsigned char[:, :, :] source_array,
        unsigned char[:, :, :] destination_array,
        float percentage):

    """
    Alpha Blending 
    
    Blend two images together.
    
    e.g:
    # compatible 24, 32-bit data
    image = blend_c(source_array, destination_array, percentage)


    :param source_array:
        numpy.ndarray or memoryviewslice shape (w, h, 3) type uint8
        
    :param destination_array: 
        numpy.ndarray or memoryviewslice shape (w, h, 3) type uint8
        
    :param percentage: 
        float; Percentage value between [0.0 ... 100.0]
        
    :return: return: 
        Return a 24-bit surface, blend of both input images
        
    """

    cdef:

        unsigned char c1, c2, c3
        int i=0, j=0
        Py_ssize_t w = source_array.shape[0]
        Py_ssize_t h = source_array.shape[1]
        unsigned char[:, :, ::1] final_array = numpy.ascontiguousarray(empty((h, w, 3), dtype=uint8))
        float c4 = percentage * <float>0.01
        float tmp = <float> 1.0 - c4
        unsigned char * v

    with nogil:
        for j in prange(h, schedule=SCHEDULE, num_threads=THREADS):
            for i in range(w):

                c1 = min(<unsigned char> (<float> destination_array[i, j, 0] * c4 +
                                source_array[i, j, 0] * tmp),
                         <unsigned char>255)
                c2 = min(<unsigned char> (<float> destination_array[i, j, 1] * c4 +
                                source_array[i, j, 1] * tmp),
                         <unsigned char>255)
                c3 = min(<unsigned char> (<float> destination_array[i, j, 2] * c4 +
                                source_array[i, j, 2] * tmp),
                         <unsigned char>255)

                v = &final_array[j, i, 0]

                v[0] = c1
                (v + 1)[0] = c2
                (v + 2)[0] = c3
                # final_array[j, i, 0] = c1 # if c1>0 else 0
                # final_array[j, i, 1] = c2 # if c2>0 else 0
                # final_array[j, i, 2] = c3 # if c3>0 else 0

    return frombuffer(final_array, (w, h), 'RGB')



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef inline void blend_inplace_c(
        unsigned char[:, :, :] destination,
        const unsigned char[:, :, :] source,
        const float percentage
       ):

    """
    Blend the source image into the destination (inplace) 
    
    source & destination Textures must be same sizes
    Compatible with 24 - 32 bit surface
    
    e.g:
    blend_inplace(DESTINATION, BACKGROUND, percentage = VALUE)
    
    :param destination: 
        numpy.ndarray shape(w, h, 3) or memoryviewslice type uint8 containing RGB pixel 
        format or any other pixel format.
        This array will receive the transformation (blend of both images,
        source and destination)
        
    :param source: 
        numpy.ndarray shape(w, h, 3) or memoryviewslice type uint8 containing RGB pixel 
        format or any other pixel format.
        This array will remain unchanged.
        
    :param percentage: 
        float; Percentage value between [0.0 ... 100.0]
        
    :return:
        void
    
    """

    cdef:

        unsigned char c1, c2, c3
        int i=0, j=0
        Py_ssize_t w = source.shape[0]
        Py_ssize_t h = source.shape[1]
        float c4 = percentage * <float> 0.01
        float tmp = <float> 1.0 - c4

    with nogil:

        for j in prange(h, schedule=SCHEDULE, num_threads=THREADS):
            for i in range(w):

                c1 = min(<unsigned char> (<float> source[i, j, 0] * c4 +
                                destination[i, j, 0] * tmp), <unsigned char>255)
                c2 = min(<unsigned char> (<float> source[i, j, 1] * c4 +
                                destination[i, j, 1] * tmp), <unsigned char>255)
                c3 = min(<unsigned char> (<float> source[i, j, 2] * c4 +
                                destination[i, j, 2] * tmp), <unsigned char>255)

                destination[ i, j, 0 ] = c1 # if c1 > 0 else 0
                destination[ i, j, 1 ] = c2 # if c2 > 0 else 0
                destination[ i, j, 2 ] = c3 # if c3 > 0 else 0



# new version 1.0.5
@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef object alpha_blending(source, destination):
    """
    Alpha blending (return new surface)
    
    Blend two 32-bit images together. 
    Both images must be format 32-bit with alpha layers.
    The source image will be draw on the top of the destination image.
    
    e.g:
    new_image = alpha_blending(source, destination)
    
    :param source     : 
        pygame.Surface (Source) 32-bit with alpha channel
        
    :param destination: 
        pygame.Surface (Destination) 32-bit with alpha channel
        
    :return: return   : 
        Return a 32 bit pygame.Surface with both images blended together
    """

    cdef:
        Py_ssize_t w, h, w2, h2
        int bit_size, bit_size2
        unsigned char [ ::1 ] source_array,
        unsigned char[ ::1 ] destination_array

    w, h = source.get_size()
    w2, h2 = destination.get_size()
    bit_size = source.get_bytesize()
    bit_size2 = destination.get_bytesize()

    if bit_size != 4:
        raise ValueError("\n Source image is not a 32-bit")

    if bit_size2 != 4:
        raise ValueError("\n Destination image is not a 32-bit")

    # source & destination array are normalized
    try:
        # source array is BGRA format
        source_array = numpy.ascontiguousarray(source.get_buffer(), dtype=uint8)

    except Exception as e:
        raise ValueError(
            "\nCannot reference source pixels into a 1d memoryviewslice array.\n %s " % e)

    try:
        # destination array is BGRA format
        destination_array = numpy.ascontiguousarray(destination.get_buffer(), dtype=uint8)


    except Exception as e:
        raise ValueError(
            "\nCannot reference destination pixels into a 1d memoryviewslice array.\n %s " % e)

    cdef:

        float rr, gg, bb, tmp
        int i=0
        int l = w * h * bit_size
        int l2 = w2 * h2 * bit_size2
        unsigned char[::1] final_array = numpy.ascontiguousarray(empty(l, dtype=uint8))
        float r, g, b
        float *a
        unsigned char *p1
        unsigned char *d1

    if l!=l2:
        raise ValueError("\nSource and destination must be same sizes.")

    with nogil:
        # noinspection SpellCheckingInspection
        for i in prange(0, l, bit_size, schedule=SCHEDULE, num_threads=THREADS):

                p1 = &source_array[ i ]
                d1 = &destination_array[ i ]

                tmp = (<float> 255.0 - (p1 + 3)[0]) *  (d1 + 3)[0] * <float>ONE_65025

                r = (p1 + 2)[0] * (p1 + 3)[0] * <float>ONE_65025
                g = (p1 + 1)[0] * (p1 + 3)[0] * <float>ONE_65025
                b = p1[0] * (p1 + 3)[0] * <float>ONE_65025

                # premult with alpha
                rr = r + (d1+2)[0] * <float>ONE_255 *  tmp
                gg = g + (d1+1)[0] * <float>ONE_255 *  tmp
                bb = b + d1[0] * <float>ONE_255 *  tmp

                # back to [0 ... 255]
                final_array[i]   = <unsigned char>min(rr * <float>255.0, <unsigned char>255)
                final_array[i+1] = <unsigned char>min(gg * <float>255.0, <unsigned char>255)
                final_array[i+2] = <unsigned char>min(bb * <float>255.0, <unsigned char>255)
                final_array[i+3] = <unsigned char>min(((p1 + 3)[0] + tmp * <float>255.0), <unsigned char>255)

    return pygame.image.frombuffer(final_array, (w, h), 'RGBA')


# new version 1.0.5
@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef void alpha_blending_inplace(object image1, object image2):
    """
    Alpha blending (inplace)

    Blend two images together, 
    image1 and image2 must have the same dimensions and be format 32-bit with 
    layer alpha. The pixels format must be RGB(A) in range (0 ... 255). 
    
    e.g:
    
    alpha_blending_inplace(source, destination)

    :param image2     : 
        pygame.Surface 32-bit must have an alpha channel
        an exception will be raised otherwise
        
    :param image1: 
        pygame.Surface 32-bit must have an alpha channel
        an exception will be raised otherwise
        
    :return: return    : 
        Return a 32 bit pygame.Surface, 
        with blended images. 
    """

    cdef:
        unsigned char [::1] image1_array
        unsigned char [::1] image2_array
        Py_ssize_t w, h, w2, h2
        int bit_size, bit_size2

    w, h = image2.get_size()
    w2, h2 = image1.get_size()
    bit_size = image2.get_bytesize()
    bit_size2 = image1.get_bytesize()

    if bit_size != 4:
        raise ValueError("\n Source image is not a 32-bit")

    if bit_size2 != 4:
        raise ValueError("\n Destination image is not a 32-bit")

    # source & destination array are normalized
    try:
        image1_array = numpy.ascontiguousarray(image2.get_buffer(), dtype = uint8)

    except Exception as e:
        raise ValueError(
            "\nCannot reference image2 pixels into a 1d memoryviewslice array.\n %s " % e)

    try:
        image2_array = numpy.ascontiguousarray(image1.get_buffer(), dtype = uint8)

    except Exception as e:
        raise ValueError(
            "\nCannot reference image1 pixels into a 1d memoryviewslice array.\n %s " % e)

    cdef:

        float rr, gg, bb, tmp
        int i=0, j=0
        int l = w * h * bit_size
        int l2 = w2 * h2 * bit_size2
        float r, g, b
        float alpha, beta
        unsigned char *p1
        unsigned char *p2


    if l!=l2:
        raise ValueError("\nimage1 and image2 must be same sizes.")

    with nogil:
        # noinspection SpellCheckingInspection
        for i in prange(0, l, bit_size, schedule=SCHEDULE, num_threads=THREADS):

                p1 = &image1_array[ i ]
                p2 = &image2_array[ i ]

                # image 1alpha
                alpha = (p1 + 3)[0] * <float>ONE_255
                # image2 alpha
                beta = (p2 + 3)[0] * <float>ONE_255

                # RGB premultiplied
                r = (p1 + 2)[0] * <float>ONE_255 * alpha
                g = (p1 + 1)[0] * <float>ONE_255 * alpha
                b = p1[ 0 ] * <float>ONE_255 * alpha

                tmp = (<float> 1.0 - alpha) * <float>ONE_255 * beta

                # premult with alpha
                rr = r + (p2 + 2)[0] * tmp
                gg = g + (p2 + 1)[0] * tmp
                bb = b + p2[ 0 ] * tmp

                # back to [0 ... 255]
                # Source array buffer is BGRA format
                (p1 + 2)[0] = <unsigned char> min(rr * <float> 255.0, <unsigned char> 255)
                (p1 + 1)[0] = <unsigned char> min(gg * <float> 255.0, <unsigned char> 255)
                p1[ 0 ] = <unsigned char> min(bb * <float> 255.0, <unsigned char> 255)
                (p1 + 3)[0] = <unsigned char> min(
                    (alpha + beta * (1 - alpha)) * <float> 255.0, <unsigned char> 255)








@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef inline void dirt_lens_c(
        object source_,
        object lens_model_,
        int flag_,
        float light_=0.0):

    """
    Dirt lens effect (inplace)
    
    This function display a dirt lens texture on the top of your game display to 
    simulate a camera artefact or realistic camera effect when the light from the
    scene is oriented directly toward the camera. 
    
    Choose a lens texture from the Assets directory (free textures provided in Assets directory 
    of this project). All textures are sizes 5184x3456 and would have to be re-sized beforehand.
    
    The setting `light_` is a float values cap between -1.0 to 0.2 and allow you to increase the 
    light source oriented toward the camera. Values <0.0 will decrease the lens dirt 
    effect and values >0.0 will increase the brightness of the display and increase the 
    amount of dirt on the camera lens (your display).
    
    Optionally the setting flag can be changed from BLEND_RGB_ADD to any other pygame optional 
    flags value. BLEND_RGB_ADD is the default setting and allow the pixels from the dirt lens 
    texture to be blended (added) to the display.
    
    e.g:
    dirt_lens(image, flag_=BLEND_RGB_ADD, lens_model_=lens, light_=VALUE)
    
    :param source_: 
        Surface 24 - 32 bit represent the surface or the display 
    
    :param lens_model_: 
        Surface The Lens model is a pygame Surface. PygameShader provide 6 
        different surfaces that can be used as a layer to generate a dirt lens effect on your game 
        display. See below for the name of the free dirt lens textures. 
     
        Assets/Bokeh__Lens_Dirt_9.jpg
        Assets/Bokeh__Lens_Dirt_38.jpg
        Assets/Bokeh__Lens_Dirt_46.jpg
        Assets/Bokeh__Lens_Dirt_50.jpg
        Assets/Bokeh__Lens_Dirt_54.jpg
        Assets/Bokeh__Lens_Dirt_67.jpg
     
        The texture has to be loaded prior calling this effect and passed as an argument. By default 
        the textures sizes are 5184x3456 (w & h). The texture(s) have to be re-scale once to the game 
        display dimensions (e.g 1027x768)
     
    :param flag_: 
        integer; pygame flags such as BLEND_RGB_ADD, BLEND_RGB_MAX etc. These flags 
        will change the overall appearance of the effect. BLEND_RGB_ADD is the default flag and blend 
        together the dirt_lens image and the game display.
    
    :param light_: 
        float; Float value cap between [-1.0 ... 0.2] to increase or decrease 
        the overall brightness of the dirt lens texture. This setting can be used to simulate a 
        texture transition when sweeping the values from -1.0 toward 0.2 by a small increment.
        Values < 0 will tend to diminish the effect and values > 0 will increase the brightness 
        and the dirt lens effect. 
     
    :return: 
        void; Inplace transformation.
    """

    if light_!=0.0:

        lens_model_ = brightness_copy_c(lens_model_.get_view('3'), light_)

    source_.blit(lens_model_, (0, 0), special_flags=flag_)



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef object dithering_c(float [:, :, :] rgb_array_):
    """
    Apply Floyd-Steinberg Dithering to an image.

    Dithering is a technique used in computer graphics to simulate the appearance of colors
    that are not present in a limited color palette. It works by diffusing the error between 
    the desired color and the available color to neighboring pixels, creating the illusion of 
    more colors. The human eye perceives this diffusion as a smooth transition between colors, 
    even when fewer colors are available. Dithering can give images a characteristic grainy or 
    speckled appearance, particularly when the number of colors is low.

    The Floyd-Steinberg algorithm is a widely used dithering method that disperses the quantization
    error to adjacent pixels, producing visually appealing results.

    Example:
        image = dithering_c(bgr_array)

    Parameters
    ----------
    rgb_array_ : numpy.ndarray
        A 3D numpy array (w, h, 3) of type float32 representing the input image in RGB format, 
        where each pixel value is normalized in the range [0.0, 1.0].

    Returns
    -------
    numpy.ndarray
        A 2D numpy array representing the output image, in a 24-bit color format (8 bits per channel),
        regardless of the input image format.
    
    Notes
    -----
    - This function performs Floyd-Steinberg dithering to reduce the color depth of the image.
    - The input image must be in RGB format with pixel values normalized between 0.0 and 1.0.
    - The output image will be in 24-bit format (8 bits per channel).
    """


    cdef Py_ssize_t w, h
    w = <object> rgb_array_.shape[ 0 ]
    h = <object> rgb_array_.shape[ 1 ]

    cdef:
        int x=0, y=0
        float new_red, new_green, new_blue
        float quantization_error_red, quantization_error_green, quantization_error_blue
        float oldr, oldg, oldb
        float * r1
        float * g1
        float * b1

    with nogil:

        # prange cause the data to overlap between threads, this is why I am defining a chunksize of
        # h*3 to separate the threads with an interval of h*3. However the algorithm will run faster
        # without prange on that occasion
        for y in range(0, h): #, schedule=SCHEDULE, num_threads=THREADS , chunksize=h*3):

            for x in range(0, w):

                oldr = rgb_array_[x, y, 0]
                oldg = rgb_array_[x, y, 1]
                oldb = rgb_array_[x, y, 2]

                new_red   = <float>round_c(oldr)
                new_green = <float>round_c(oldg)
                new_blue  = <float>round_c(oldb)

                rgb_array_[x, y, 0] = new_red
                rgb_array_[x, y, 1] = new_green
                rgb_array_[x, y, 2] = new_blue

                quantization_error_red   = <float>(oldr - new_red)
                quantization_error_green = <float>(oldg - new_green)
                quantization_error_blue  = <float>(oldb - new_blue)

                if x < w - 1:
                    r1 = &rgb_array_[x + 1, y, 0]
                    r1[0] = r1[0] + quantization_error_red * C1
                    g1 = &rgb_array_[x + 1, y, 1]
                    g1[0] = g1[0] + quantization_error_green * C1
                    b1 = &rgb_array_[x + 1, y, 2]
                    b1[0] = b1[0] + quantization_error_blue * C1

                if y < h - 1:

                    if x > 0:
                        r1 = &rgb_array_[x - 1, y + 1, 0]
                        r1[0] = r1[0] + quantization_error_red * C2
                        g1 = &rgb_array_[x - 1, y + 1, 1]
                        g1[0] = g1[0] + quantization_error_green * C2
                        b1 = &rgb_array_[x - 1, y + 1, 2]
                        b1[0] = b1[0] + quantization_error_blue * C2

                    r1 = &rgb_array_[x, y + 1, 0]
                    r1[0] = r1[0] + quantization_error_red * C3
                    g1 = &rgb_array_[x, y + 1, 1]
                    g1[0] = g1[0] + quantization_error_green * C3
                    b1 = &rgb_array_[x, y + 1, 2]
                    b1[0] = b1[0] + quantization_error_blue * C3

                    if x < w - 1:
                        r1 = &rgb_array_[x + 1, y + 1, 0]
                        r1[0] = r1[0] + quantization_error_red * C4
                        g1 = &rgb_array_[x + 1, y + 1, 1]
                        g1[0] =  g1[0] + quantization_error_green * C4
                        b1 = &rgb_array_[x + 1, y + 1, 2]
                        b1[0] = b1[0] + quantization_error_blue * C4

    arr = (asarray(rgb_array_).transpose(1, 0, 2) * <float> 255.0).astype(dtype=numpy.uint8, order='C')
    return frombuffer(arr, (w, h), "RGB")



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef inline void dithering_inplace_c(float [:, :, :] rgb_array_, unsigned char[:, :, :] tmp):
    """
    Dithering Floyd Steinberg (inplace)
    
    Dithering is used in computer graphics to create the illusion of "color depth" in images with
    a limited color palette - a technique also known as color quantization. In a dithered image,
    colors that are not available in the palette are approximated by a diffusion of colored pixels
    from within the available palette. The human eye perceives the diffusion as a mixture of 
    the colors within it (see color vision). Dithered images, particularly those with relatively
    few colors, can often be distinguished by a characteristic graininess or speckled appearance
    
    Take a pygame surface as argument format 24-32 bit and convert it to a 3d array format 
    (w, h, 3) type float (float32, single precision). 
    As the image is converted to a different data type format (uint8 to float32), 
    the transformation cannot be applied inplace. The image returned by the method dithering 
    is a copy of the original image.   
    
    e.g:
    # for 24 - 32 bit 
    dithering_inplace_c(image)
       
    :param rgb_array_: 
        numpy.ndarray shape (w, h, 3) of type float32 containing RGB pixel format, all the pixels 
        are normalized between 0.0 ... 1.0
          
    :param tmp: 
        numpy.ndarray shape (w, h, 3) of type uint8 containing the source RGB pixels
          
    :return: 
        void
             
    """

    cdef Py_ssize_t w, h
    w = <object> rgb_array_.shape[ 0 ]
    h = <object> rgb_array_.shape[ 1 ]

    cdef:
        int x=0, y=0
        float new_red, new_green, new_blue
        float quantization_error_red, quantization_error_green, quantization_error_blue
        float oldr, oldg, oldb
        float * r1
        float * g1
        float * b1

    with nogil:

        for y in prange(0, h, schedule=SCHEDULE, num_threads=THREADS, chunksize=h):

            for x in range(0, w):

                oldr = rgb_array_[x, y, <unsigned short int>0]
                oldg = rgb_array_[x, y, <unsigned short int>1]
                oldb = rgb_array_[x, y, <unsigned short int>2]

                new_red   = round_c(oldr)
                new_green = round_c(oldg)
                new_blue  = round_c(oldb)

                rgb_array_[x, y, <unsigned short int>0] = new_red
                rgb_array_[x, y, <unsigned short int>1] = new_green
                rgb_array_[x, y, <unsigned short int>2] = new_blue

                quantization_error_red   = <float>(oldr - new_red)
                quantization_error_green = <float>(oldg - new_green)
                quantization_error_blue  = <float>(oldb - new_blue)

                if x < w - 1:
                    r1 = &rgb_array_[x + 1, y, 0]
                    r1[0] = r1[0] + quantization_error_red * C1
                    g1 = &rgb_array_[x + 1, y, 1]
                    g1[0] = g1[0] + quantization_error_green * C1
                    b1 = &rgb_array_[x + 1, y, 2]
                    b1[0] = b1[0] + quantization_error_blue * C1

                if y < h - 1:

                    if x > 0:
                        r1 = &rgb_array_[x - 1, y + 1, 0]
                        r1[0] = r1[0] + quantization_error_red * C2
                        g1 = &rgb_array_[x - 1, y + 1, 1]
                        g1[0] = g1[0] + quantization_error_green * C2
                        b1 = &rgb_array_[x - 1, y + 1, 2]
                        b1[0] = b1[0] + quantization_error_blue * C2

                    r1 = &rgb_array_[x, y + 1, 0]
                    r1[0] = r1[0] + quantization_error_red * C3
                    g1 = &rgb_array_[x, y + 1, 1]
                    g1[0] = g1[0] + quantization_error_green * C3
                    b1 = &rgb_array_[x, y + 1, 2]
                    b1[0] = b1[0] + quantization_error_blue * C3

                    if x < w -1:
                        r1 = &rgb_array_[x + 1, y + 1, 0]
                        r1[0] = r1[0] + quantization_error_red * C4
                        g1 = &rgb_array_[x + 1, y + 1, 1]
                        g1[0] = g1[0] + quantization_error_green * C4
                        b1 = &rgb_array_[x + 1, y + 1, 2]
                        b1[0] = b1[0] + quantization_error_blue * C4


        for y in prange(1, h, schedule=SCHEDULE, num_threads=THREADS):
            for x in range(1, w):
                tmp[x, y, 0] = <unsigned char>(rgb_array_[x, y, 0] * <float>255.0)
                tmp[x, y, 1] = <unsigned char>(rgb_array_[x, y, 1] * <float>255.0)
                tmp[x, y, 2] = <unsigned char>(rgb_array_[x, y, 2] * <float>255.0)



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef object dithering_atkinson_c(float [:, :, :] rgb_array_):

    """
    
    Dithering atkinson (copy)
    
    Atkinson dithering is a variant of Floyd–Steinberg dithering designed by 
    Bill Atkinson at Apple Computer, and used in the original Macintosh computer. 
    
    Dithering is used in computer graphics to create the illusion of "color depth" in images with
    a limited color palette - a technique also known as color quantization. In a dithered image,
    colors that are not available in the palette are approximated by a diffusion of colored pixels
    from within the available palette. The human eye perceives the diffusion as a mixture of 
    the colors within it (see color vision). Dithered images, particularly those with relatively
    few colors, can often be distinguished by a characteristic graininess or speckled appearance.
    
    Take a pygame surface as argument format 24-32 bit and convert it to a 3d array format 
    (w, h, 3) type float (float32, single precision) containing RGB pixels format
    
    As the image is converted to a different data type format (conversion from uint8 to float32), 
    the transformation cannot be applied inplace. 
    
    The image returned is a copy of the original image.   
    
    e.g:
    # for 24, 32-bit image format 
    rgb_array = dithering_atkinson_c(rgb_array)
    
    :param rgb_array_:
        numpy.ndarray shape (w, h, 3) of type float32 containing all the normalized pixels format RGB
        
    :return: 
        pygame surface 24-bit format.
           
    """

    cdef Py_ssize_t w, h
    w = <object> rgb_array_.shape[ 0 ]
    h = <object> rgb_array_.shape[ 1 ]

    cdef:
        int x=0, y=0
        float new_red, new_green, new_blue
        float quantization_error_red, quantization_error_green, quantization_error_blue
        float oldr, oldg, oldb
        float * r1
        float * g1
        float * b1

    with nogil:

        # prange cause the data to overlap between threads, this is why I am defining a chunksize of
        # h*3 to separate the threads with an interval of h*3. However the algorithm will run faster
        # without prange on that occasion
        for y in range(0, h): #, schedule=SCHEDULE, num_threads=THREADS):# , chunksize=h*3):

            for x in range(0, w):

                oldr = rgb_array_[x, y, 0]
                oldg = rgb_array_[x, y, 1]
                oldb = rgb_array_[x, y, 2]

                new_red   = round_c(oldr)
                new_green = round_c(oldg)
                new_blue  = round_c(oldb)

                rgb_array_[x, y, 0] = new_red
                rgb_array_[x, y, 1] = new_green
                rgb_array_[x, y, 2] = new_blue

                quantization_error_red   = <float>(oldr - new_red) * <float>0.125
                quantization_error_green = <float>(oldg - new_green) * <float>0.125
                quantization_error_blue  = <float>(oldb - new_blue) * <float>0.125

                if x < w - 1:
                    r1 = &rgb_array_[x + 1, y, 0]
                    r1[0] += quantization_error_red
                    g1 = &rgb_array_[x + 1, y, 1]
                    g1[0] += quantization_error_green
                    b1 = &rgb_array_[x + 1, y, 2]
                    b1[0] += quantization_error_blue

                if x < w - 2:
                    r1 = &rgb_array_[x + 2, y, 0]
                    r1[0] += quantization_error_red
                    g1 = &rgb_array_[x + 2, y, 1]
                    g1[0] += quantization_error_green
                    b1 = &rgb_array_[x + 2, y, 2]
                    b1[0] += quantization_error_blue

                if y < h - 1:
                        if x > 0:
                            r1 = &rgb_array_[x - 1, y + 1, 0]
                            r1[0] += quantization_error_red
                            g1 = &rgb_array_[x - 1, y + 1, 1]
                            g1[0] +=  quantization_error_green
                            b1 = &rgb_array_[x - 1, y + 1, 2]
                            b1[0] += quantization_error_blue

                        r1 = &rgb_array_[x, y + 1, 0]
                        r1[0] += quantization_error_red
                        g1 = &rgb_array_[x, y + 1, 1]
                        g1[0] += quantization_error_green
                        b1 = &rgb_array_[x, y + 1, 2]
                        b1[0] += quantization_error_blue

                        if x < w - 1:
                            r1 = &rgb_array_[x + 1, y + 1, 0]
                            r1[0] += quantization_error_red
                            g1 = &rgb_array_[x + 1, y + 1, 1]
                            g1[0] += quantization_error_green
                            b1 = &rgb_array_[x + 1, y + 1, 2]
                            b1[0] += quantization_error_blue

                if y < h - 2:
                    r1 = &rgb_array_[x, y + 2, 0]
                    r1[0] += quantization_error_red
                    g1 = &rgb_array_[x, y + 2, 1]
                    g1[0] += quantization_error_green
                    b1 = &rgb_array_[x, y + 2, 2]
                    b1[0] += quantization_error_blue

    arr = (asarray(rgb_array_).transpose(1,0,2) * <float>255.0).astype(dtype=numpy.uint8, order='C')
    return frombuffer(arr, (w, h), "RGB")



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef dithering1D_atkinson_c(
        const Py_ssize_t w,
        const Py_ssize_t h,
        unsigned char [::1] bgr_array,
        bint format_32 = False
        ):
    """
    Apply 1D Atkinson dithering to a BGR or BGRA image buffer.

    This function uses the Atkinson dithering algorithm, which is a type of ordered dithering 
    designed to reduce the color depth of an image while maintaining visual quality. The Atkinson 
    algorithm works by diffusing the quantization error to adjacent pixels, creating a visually 
    smoother transition between colors in low-color images.

    This implementation operates on a C-buffer (1D array), which can represent either a BGR or BGRA 
    image format. The dithering process alters the pixel values in place to simulate a larger color 
    palette in the final image.

    Example:
        dithering1D_atkinson_c(w, h, bgr_array, format_32=True)

    Parameters
    ----------
    w : int
        The width of the image (number of pixels along the x-axis).
    
    h : int
        The height of the image (number of pixels along the y-axis).
    
    bgr_array : unsigned char [::1]
        A 1D C-buffer (array) containing the pixel data of the image in BGR format. 
        The image can either be in BGR or BGRA format, depending on the value of `format_32`.
    
    format_32 : bool, optional, default=False
        If True, the input image is in BGRA format (with 4 channels per pixel); 
        if False, the input image is in BGR format (3 channels per pixel).
    
    Returns
    -------
    void
        The function modifies the input `bgr_array` in place, applying the Atkinson dithering 
        algorithm to reduce the color depth of the image.

    Notes
    -----
    - The dithering algorithm works by diffusing quantization errors to neighboring pixels in 
      the image, creating a smoother gradient with fewer colors.
    - This function modifies the image buffer directly and does not return a new buffer.
    - The `bgr_array` should be a C-buffer (1D array) containing BGR or BGRA pixel data, 
      depending on the `format_32` parameter.
    """


    cdef short int bytesize = 4 if format_32==True else 3
    cdef unsigned int row = w * bytesize

    cdef:
        int i
        int l = w * h * bytesize
        float new_red, new_green, new_blue
        float quantization_error_red, quantization_error_green, quantization_error_blue
        float oldr, oldg, oldb
        float * p1
        float [::1] tmp = numpy.asarray(bgr_array, dtype=numpy.float32) * <float>ONE_255

    with nogil:

        for i in prange(0, l, bytesize,  schedule=SCHEDULE, num_threads=1): #=THREADS):

                p1 = &tmp[ i ]

                # skip when alpha is null.
                # neighborhood pixels are most likely transparent as well
                if bytesize == 4:
                    if (p1 + 3)[ 0 ] == 0:
                        continue

                oldr = p1[0]
                oldg = (p1 + 1)[ 0 ]
                oldb = (p1 + 2)[ 0 ]

                new_red   = round_c(oldr)
                new_green = round_c(oldg)
                new_blue  = round_c(oldb)

                p1[ 0 ]= new_red
                (p1 + 1)[ 0 ] = new_green
                (p1 + 2)[ 0 ] = new_blue

                quantization_error_red   = <float>(oldr - new_red) * <float>0.125
                quantization_error_green = <float>(oldg - new_green) * <float>0.125
                quantization_error_blue  = <float>(oldb - new_blue) * <float>0.125

                if i < l - bytesize:
                    p1 = &tmp[ i + bytesize ]
                    p1[0] += quantization_error_red
                    (p1 + 1)[0] += quantization_error_green
                    (p1 + 2)[0] += quantization_error_blue

                if i < l - 2 * bytesize:
                    p1 = &tmp[ i + bytesize * 2 ]
                    p1[0] += quantization_error_red
                    (p1 + 1)[0] += quantization_error_green
                    (p1 + 2)[0] += quantization_error_blue

                if i < l - (bytesize + row):
                    p1 = &tmp[ i - bytesize + row ]
                    p1[0] += quantization_error_red
                    (p1 + 1)[0] +=  quantization_error_green
                    (p1 + 2)[0] += quantization_error_blue

                if i < l - row:
                    p1 = &tmp[ i + row ]
                    p1[0] += quantization_error_red
                    (p1 + 1)[0] += quantization_error_green
                    (p1 + 2)[0] += quantization_error_blue

                if i < l - (bytesize + row):
                    p1 = &tmp[ i + bytesize + row ]
                    p1[0] += quantization_error_red
                    (p1 + 1)[0] += quantization_error_green
                    (p1 + 2)[0] += quantization_error_blue

                if i < l - row * 2 :
                    p1 = &tmp[ i + row * 2 ]
                    p1[ 0 ] += quantization_error_red
                    (p1 + 1)[ 0 ] += quantization_error_green
                    (p1 + 2)[ 0 ] += quantization_error_blue

        for i in prange(0, l, schedule=SCHEDULE, num_threads=THREADS):
            bgr_array [ i ] = <unsigned char>(tmp [ i ] * <float>255.0)





@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef inline void convert_27_c(
        const Py_ssize_t w,
        const Py_ssize_t h,
        unsigned char [:, :, :] rgb_array) nogil:
    """
    Convert an image to 27 unique colors using a quantization algorithm.

    This function reduces the colors in the input image to a palette of only 27 distinct colors 
    by applying a color quantization technique. This technique is often used in image processing 
    to reduce the color depth of an image while preserving its visual structure as much as possible. 
    The result is an image with limited color options, making it suitable for applications like 
    low-color displays or stylistic effects.

    Example:
        convert_27_c(800, 600, bgr_array)

    Parameters
    ----------
    w : int
        The width of the input image (in pixels).
    
    h : int
        The height of the input image (in pixels).
    
    rgb_array : unsigned char[:, :, :]
        A 3D numpy array or memory view with shape (w, h, 3) containing the RGB pixel data 
        of the image. This array must reference a pygame.Surface. 
        Any modifications to this array will directly alter the referenced surface.

    Returns
    -------
    void
        This function modifies the input `rgb_array` in place, converting the pixel colors 
        to one of the 27 distinct colors.

    Notes
    -----
    - The algorithm quantizes the RGB color space to only 27 colors. This is achieved 
      by selecting the closest matching color from a predefined palette.
    - The input `rgb_array` must have a shape of (w, h, 3), where `w` and `h` are the image 
      dimensions and the third dimension holds the RGB values (3 channels).
    - This function operates in place, meaning the original `rgb_array` will be modified directly.
    """


    cdef:
        int x=0
        int y=0
        unsigned char *r
        unsigned char *g
        unsigned char *b
        float f = <float> 255.0 / <float> 2
        float c1 = <float>2 / <float>255.0
        int index = 0


    with nogil:
        for y in prange(h, schedule=SCHEDULE, num_threads=THREADS):
            for x in range(0, w):

                r = &rgb_array[x, y, 0]
                g = &rgb_array[x, y, 1]
                b = &rgb_array[x, y, 2]

                r[ 0 ] = <unsigned char>(round_c(c1 * <float> r[ 0 ] ) * f)
                g[ 0 ] = <unsigned char>(round_c(c1 * <float> g[ 0 ] ) * f)
                b[ 0 ] = <unsigned char>(round_c(c1 * <float> b[ 0 ] ) * f)




@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef bilateral_c(
        const unsigned char [:, :, :] rgb_array_,
        const float sigma_s_,
        const float sigma_i_,
        unsigned int kernel_size = 3
):
    """
    Applies bilateral filtering to an image and returns a filtered copy.
    
    Bilateral filtering is a non-linear, edge-preserving, and noise-reducing 
    smoothing filter. It replaces the intensity of each pixel with a weighted 
    average of nearby pixel intensities. The weight for each neighboring pixel 
    is determined by both spatial proximity (distance) and intensity similarity.
    
    The filter is controlled by two parameters:
    - sigma_s_: Spatial standard deviation, which defines the size of the neighborhood.
    - sigma_i_: Intensity standard deviation, which controls the degree to which 
      intensity differences affect the weight.
    
    - A smaller value for `sigma_i_` keeps sharp edges, while larger values result in 
      more smoothing across edges.
    - As `sigma_i_` increases, the filter approximates a Gaussian blur.
    
    **Example:**
        image = bilateral_c(bgr_array, 16, 18, 3)
    
    :param rgb_array_: 
        3D numpy array representing the image, with dimensions (height, width, 3).
        Each pixel contains RGB values (ignores alpha channel if present).
    
    :param sigma_s_: 
        Float; Spatial standard deviation. Determines the size of the spatial neighborhood 
        to consider during filtering.
    
    :param sigma_i_: 
        Float; Intensity standard deviation. Controls how much intensity difference affects the weight.
    
    :param kernel_size: 
        Integer (default 3); The size of the square kernel. Defines the extent of the neighborhood 
        around each pixel to be considered in the filtering.
    
    :return: 
        A filtered image as a 24-bit RGB surface.
    
    """

    cdef Py_ssize_t w, h
    w = <object>rgb_array_.shape[ 0 ]
    h = <object> rgb_array_.shape[ 1 ]

    cdef:
        unsigned char [:, :, :] bilateral = empty((h, w, 3), dtype=uint8)
        int x, y, xx, yy, kx, ky
        int k = kernel_size
        float sigma_i2 = 1.0 / (2.0 * sigma_i_ * sigma_i_)
        float sigma_s2 = 1.0 / (2.0 * sigma_s_ * sigma_s_)
        float ir, ig, ib
        float wr, wg, wb
        float wpr, wpg, wpb
        unsigned char rr, gg, bb
        unsigned char r, g, b
        # Precompute the Gaussian spatial kernel (weights based on distance)
        float [:, :] spatial_kernel = empty((2*k + 1, 2*k + 1), dtype=numpy.float32)
        float dist, gs


    for ky in range(-k, k + 1):  # Loop over vertical range of the kernel
        for kx in range(-k, k + 1):  # Loop over horizontal range of the kernel
            dist = <float>kx * kx + <float>ky * ky  # Calculate the squared distance
            gs = <float>exp(-dist * sigma_s2)  # Compute the Gaussian weight based on distance
            spatial_kernel[ky + k, kx + k] = gs  # Store the spatial weight in the kernel

    # Start the bilateral filtering with parallelization
    with nogil:

        for x in prange(0, w, schedule=SCHEDULE, num_threads=THREADS):

            for y in range(0, h):

                ir, ig, ib = <float>0.0, <float>0.0, <float>0.0
                wpr, wpg, wpb = <float>0.0, <float>0.0, <float>0.0

                rr = rgb_array_[x, y, 0]
                gg = rgb_array_[x, y, 0]
                bb = rgb_array_[x, y, 0]

                for ky in range(-k, k + 1):
                    yy = y + ky

                    if yy < 0:
                        continue
                    elif yy >= h:
                        continue

                    for kx in range(-k, k + 1):
                        xx = x + kx

                        if xx < 0:
                            continue
                        elif xx >= w:
                            continue

                        # Get the spatial weight based on distance from the center
                        gs = spatial_kernel[ky + k, kx + k]

                        # Get the RGB values of the neighboring pixel

                        r = rgb_array_[xx, yy, 0]
                        g = rgb_array_[xx, yy, 1]
                        b = rgb_array_[xx, yy, 2]

                        # Compute the range weights based on intensity differences
                        wr = exp(-((r - rr) ** 2) * sigma_i2) * gs
                        wg = exp(-((g - gg) ** 2) * sigma_i2) * gs
                        wb = exp(-((b - bb) ** 2) * sigma_i2) * gs

                        ir = ir + r * wr
                        ig = ig + g * wg
                        ib = ib + b * wb

                        wpr = wpr + wr
                        wpg = wpg + wg
                        wpb = wpb + wb

                ir = ir / wpr
                ig = ig / wpg
                ib = ib / wpb

                bilateral[y, x, 0] = <unsigned int>ir
                bilateral[y, x, 1] = <unsigned int>ig
                bilateral[y, x, 2] = <unsigned int>ib

    return frombuffer(bilateral, (w, h), 'RGB')





EMBOSS_KERNEL3x3 = \
    numpy.array((
        [-2, -1, 0],
        [-1,  1, 1],
        [ 0,  1, 2],
    )).astype(dtype=numpy.int8, order='C')



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef object emboss3d_c(const unsigned char [:, :, :] rgb_array_):
    """
    Apply a 3D emboss filter to an image, returning a modified copy.

    This function applies an embossing filter to an image, which creates a 3D-like effect 
    by simulating the appearance of raised or recessed areas on the image's surface, similar to 
    the look of an embossed paper or metal design. This effect highlights edges and creates 
    a visually distinct texture that simulates depth.

    Example:
        # Apply the emboss effect and return a 24-bit pygame.Surface
        embossed_image = emboss3d_c(bgr_array)

    Parameters
    ----------
    rgb_array_ : unsigned char[:, :, :]
        A 3D numpy array or memoryview with shape (w, h, 3) containing RGB pixel data 
        (or any other compatible pixel format). The image to which the emboss filter is applied.
    
    Returns
    -------
    pygame.Surface
        A pygame.Surface object with the embossed effect applied. The surface is in 24-bit format.
        This is a copy of the original image, with the 3D emboss effect applied.

    Notes
    -----
    - The resulting image will have a pseudo-3D effect, with edges being more pronounced.
    - The function operates by simulating depth through pixel value manipulation, highlighting 
      contrasts based on the original image's structure.
    - The returned surface is a copy of the input image and will not modify the original image.
    """


    cdef:
        short kernel_half = <unsigned int>len(EMBOSS_KERNEL3x3) >> 1

    # texture sizes
    cdef int w = <object>rgb_array_.shape[0]
    cdef int h = <object>rgb_array_.shape[1]

    cdef:
        char [:, ::1] kernel = EMBOSS_KERNEL3x3
        unsigned char [:, :, ::1] emboss = empty((h, w, 3), order='C', dtype=uint8)
        int x, y
        unsigned int xx, yy
        unsigned short red, green, blue,
        short kernel_offset_y, kernel_offset_x
        int r, g, b
        short k

    with nogil:

        for y in prange(0, h, schedule=SCHEDULE, num_threads=THREADS):

            for x in range(0, w):

                r, g, b = <unsigned char>0, <unsigned char>0, <unsigned char>0

                for kernel_offset_y in range(-kernel_half, kernel_half + 1):

                    yy = y + kernel_offset_y

                    if yy < 0:
                        continue
                    elif yy > h - 1:
                        continue

                    for kernel_offset_x in range(-kernel_half, kernel_half + 1):

                        xx = x + kernel_offset_x

                        if xx < 0:
                            continue
                        elif xx > w - 1:
                            continue

                        red, green, blue = \
                            rgb_array_[xx, yy, 0], \
                            rgb_array_[xx, yy, 1],\
                            rgb_array_[xx, yy, 2]

                        k = <short>kernel[kernel_offset_y + kernel_half, kernel_offset_x + kernel_half]
                        r = r + red * k
                        g = g + green * k
                        b = b + blue * k

                if r < 0:
                    r = <unsigned char>0
                if g < 0:
                    g = <unsigned char>0
                if b < 0:
                    b = <unsigned char>0
                if r > 255:
                    r= <unsigned char>255
                if g > 255:
                    g = <unsigned char>255
                if b > 255:
                    b = <unsigned char>255

                emboss[y, x, 0] = <unsigned char>r
                emboss[y, x, 1] = <unsigned char>g
                emboss[y, x, 2] = <unsigned char>b

    return frombuffer(emboss, (w, h), 'RGB')




@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef object emboss3d_gray_c(const unsigned char [:, :, :] rgb_array_):
    """
    Apply a grayscale 3D emboss filter to an image, returning a modified copy.

    This function applies an embossing filter to an image, similar to traditional embossing, 
    but with a grayscale effect. The result simulates raised or recessed areas, giving the image 
    a depth-like appearance with grayscale shading based on the original image's luminance.

    Example:
        # Apply the grayscale emboss effect and return a 24-bit pygame.Surface
        embossed_image = emboss3d_gray_c(bgr_array)

    Parameters
    ----------
    rgb_array_ : unsigned char[:, :, :]
        A 3D numpy array or memoryview with shape (w, h, 3) containing RGB pixel data 
        (or any other compatible pixel format). The image to which the grayscale emboss filter is applied.

    Returns
    -------
    pygame.Surface
        A pygame.Surface object with the grayscale embossed effect applied. 
        The surface is in 24-bit format, representing the modified version of the input image.

    Notes
    -----
    - The grayscale effect focuses on the brightness of the image and applies the emboss effect 
      based on the lightness or darkness of the original pixels.
    - The function operates by modifying the pixel values to create a pseudo-3D depth effect using 
      grayscale luminance values.
    - The returned surface is a copy of the input image and does not modify the original image.
    """

    cdef:
        short kernel_half = <unsigned int>len(EMBOSS_KERNEL3x3) >> 1

    # texture sizes
    cdef int w = <object>rgb_array_.shape[0]
    cdef int h = <object>rgb_array_.shape[1]

    cdef:
        char [:, ::1] kernel = EMBOSS_KERNEL3x3
        unsigned char [:, :, ::1] emboss = empty((h, w, 3), order='C', dtype=uint8)
        int x, y
        unsigned int xx, yy
        unsigned short red, green, blue,
        short kernel_offset_y, kernel_offset_x
        short k,
        unsigned char grey
        int g

    with nogil:

        for y in prange(0, h, schedule=SCHEDULE, num_threads=THREADS):

            for x in range(0, w):

                g = <unsigned char>0

                for kernel_offset_y in range(-kernel_half, kernel_half + 1):

                    yy = y + kernel_offset_y
                    if yy < 0:
                        continue
                    elif yy > h - 1:
                        continue

                    for kernel_offset_x in range(-kernel_half, kernel_half + 1):

                        xx = x + kernel_offset_x

                        if xx < 0:
                            continue
                        elif xx > w - 1:
                            continue

                        grey = <unsigned char>((
                                rgb_array_[xx, yy, 0] +
                                rgb_array_[xx, yy, 1] +
                                rgb_array_[xx, yy, 2])/<float>3.0)

                        k = kernel[kernel_offset_y + kernel_half, kernel_offset_x + kernel_half]

                        g = g + grey * k

                if g < 0:
                    g = <unsigned char>0
                if g > 255:
                    g = <unsigned char>255

                emboss[y, x, 0] = <unsigned char>g
                emboss[y, x, 1] = <unsigned char>g
                emboss[y, x, 2] = <unsigned char>g

    return frombuffer(emboss, (w, h), 'RGB')






@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef inline void emboss3d_inplace_c(unsigned char [:, :, :] rgb_array, copy = None):
    """
    Apply an embossing effect to a 3D RGB array in-place.

    The embossing effect highlights edges in an image, producing a raised or relief-like appearance 
    by emphasizing intensity changes between neighboring pixels. This effect mimics an engraving or 
    embossing process on materials like paper or metal.

    Example Usage:
    --------------
    # Apply embossing effect to a 24-bit image
    image = pygame.image.load('../Assets/px.png').convert(24)
    pixel_copy = numpy.ascontiguousarray(array3d(image).transpose(1, 0, 2))
    emboss3d_inplace_c(image, copy=pixel_copy)

    # Apply embossing effect to a 32-bit image with alpha channel
    image = pygame.image.load('../Assets/px.png').convert_alpha()
    pixel_copy = numpy.ascontiguousarray(array3d(image).transpose(1, 0, 2))
    emboss3d_inplace_c(image, copy=pixel_copy)

    Parameters
    ----------
    rgb_array : numpy.ndarray, shape (w, h, 3), dtype uint8
        A 3D NumPy array representing an RGB image with shape (width, height, 3). 
        The embossing effect is applied directly to this array in-place, modifying 
        the original image data. The array should contain pixel data in RGB format (uint8).

    copy : numpy.ndarray, shape (w, h, 3), dtype uint8, optional (default=None)
        A copy of the original `rgb_array` that can be used as a reference during processing.
        Providing a `copy` array is optional but may improve performance in certain cases. 
        If not provided, the function will operate directly on `rgb_array`.

    Returns
    -------
    None
        This function modifies the input `rgb_array` in-place and does not return a value.

    Notes
    -----
    - The embossing effect is applied by calculating pixel intensity differences with neighboring 
      pixels, enhancing edges to create a 3D relief effect.
    - The input `rgb_array` is modified in-place, so ensure to provide a copy if preserving the 
      original data is necessary.
    """

    cdef:
        short kernel_half = len(EMBOSS_KERNEL3x3) >> 1

    # texture sizes
    cdef int w = <object>rgb_array.shape[0]
    cdef int h = <object>rgb_array.shape[1]

    cdef:
        char [:, ::1] kernel = EMBOSS_KERNEL3x3

        # below create a copy False of the array and do not reference the pixels.
        # The real time transformation of the identical copy of the array will not be functional as all the pixels
        # undergo constant transformations. It is then necessary to load the pixels from a copy of the source array
        # to implement the inplace transformation. Such as below
        unsigned char [:, :, :] rgb_array_cp = \
            numpy.ndarray(shape=(h, w, 3), buffer=rgb_array, dtype=uint8).copy() \
                if copy is None else copy

        # The below will also works, but this is slightly slower.
        # unsigned char [::1, :, :] rgb_array_cp = numpy.asarray(bgr_array, order='F')
        # This works with the below pixels within the xx and yy loops
        # red, green, blue = \
        #     rgb_array_cp[ xx, yy, 0 ], \
        #     rgb_array_cp[ xx, yy, 1 ], \
        #     rgb_array_cp[ xx, yy, 2 ]

        int x, y
        unsigned int xx, yy
        unsigned char * red
        unsigned char * green
        unsigned char * blue
        short kernel_offset_y, kernel_offset_x
        int r, g, b
        char * k

    with nogil:

        for y in prange(0, h, schedule=SCHEDULE, num_threads=THREADS):

            for x in range(0, w):

                r, g, b = <unsigned char> 0, <unsigned char> 0, <unsigned char> 0

                for kernel_offset_y in range(-kernel_half, kernel_half + 1):

                    yy = y + kernel_offset_y

                    if yy < 0:
                        continue
                    elif yy > h - 1:
                        continue

                    for kernel_offset_x in range(-kernel_half, kernel_half + 1):

                        xx = x + kernel_offset_x

                        if xx < 0:
                            continue
                        elif xx > w - 1:
                            continue

                        if copy is not None:
                            red   = &rgb_array_cp[ yy, xx, 0 ]
                            green = &rgb_array_cp[ yy, xx, 1 ]
                            blue  = &rgb_array_cp[ yy, xx, 2 ]

                        else:
                            red   = &rgb_array_cp[ yy, xx, 0 ]
                            green = &rgb_array_cp[ yy, xx, 2 ]
                            blue  = &rgb_array_cp[ yy, xx, 1 ]


                        k = &kernel[ kernel_offset_y + kernel_half, kernel_offset_x + kernel_half ]

                        r = r + red[0]   * k[0]
                        g = g + green[0] * k[0]
                        b = b + blue[0]  * k[0]

                if r < 0:
                    r = <unsigned char> 0
                if g < 0:
                    g = <unsigned char> 0
                if b < 0:
                    b = <unsigned char> 0
                if r > 255:
                    r = <unsigned char> 255
                if g > 255:
                    g = <unsigned char> 255
                if b > 255:
                    b = <unsigned char> 255

                rgb_array[ x, y, 0 ] = <unsigned char> r
                rgb_array[ x, y, 1 ] = <unsigned char> g
                rgb_array[ x, y, 2 ] = <unsigned char> b



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef inline void emboss1d_c(
        const Py_ssize_t w,
        const Py_ssize_t l,
        unsigned char [:] bgr_array,
        const unsigned char [:] bgr_array_cp,
        bint format_32 = False)nogil:

    """
    
    Emboss directly a C-buffer type (inplace) 
    
    Applying an embossing filter to an image often results in an image resembling a paper
    or metal embossing of the original image, hence the name. 
    
    If you are using tmp_array to improve the performances, make sure to have the same 
    array size and shape than the source array bgr_array
    
    e.g
    # 24 - bit 
    image = pygame.image.load('../Assets/px.png').convert(24)
    image = pygame.transform.smoothscale(image, (800, 600))
     
    emboss1d_c(800, 800*600*3, image.get_view('0'), None, False)
    
    # 32 - bit 
    image = pygame.image.load('../Assets/px.png').convert_alpha()
    image = pygame.transform.smoothscale(image, (800, 600))
    image_copy = image.copy()
     
    emboss1d_c(800, 800*600*4, image.get_view('0'), image_copy.get_buffer(), True)
    
    Parameters
    ----------
    w : 
        int; width of the surface 
         
    l : 
        int; length of the raw data 
        
    bgr_array :
        numpy.ndarray or memoryviewslice shape (l, ) type uint8 containing BGR pixels or any other format 
        bgr_array represent the source data that will be modify. The changes are applied inplace - meaning 
        that the surface will be automatically changed after updating the source array data. 
    
    bgr_array_cp : 
        numpy.ndarray or memoryviewslice shape (l, ) type uint8 containing BGR pixels or any other format 
        This array is a copy of the source array 
    
    format_32 :
        bool; default is False. Select True if the source array contains alpha transparency (32 - bit format).

    Returns
    -------
    void
    
    """

    cdef short bitsize
    bitsize = 3 if format_32 == False else 4

    cdef:

        int i
        int r, g, b
        unsigned int row = w * bitsize
        const unsigned char * p1
        const unsigned char * p2
        const unsigned char * p3
        const unsigned char * p4
        const unsigned char * p5
        const unsigned char * p6
        const unsigned char * p7

    for i in prange(0, l, bitsize, schedule=SCHEDULE, num_threads=THREADS):

        p4 = &bgr_array_cp[ i ]

        if row + bitsize < i < l - row - bitsize:

            # Emboss kernel 3x3
            # [-2, -1, 0]
            # [-1,  1, 1]
            # [ 0,  1, 2]

            p1 = &bgr_array_cp[ i - row - bitsize ]
            p2 = &bgr_array_cp[ i - row ]
            p3 = &bgr_array_cp[ i - bitsize ]

            p5 = &bgr_array_cp[ i + bitsize ]
            p6 = &bgr_array_cp[ i + row ]
            p7 = &bgr_array_cp[ i + row + bitsize ]

            b = -p1[ 0 ] * <int> 2 - p2[ 0 ] - p3[ 0 ] + p4[ 0 ] + p5[ 0 ] + p6[ 0 ] + p7[ 0 ] * <int> 2

            g = -(p1 + 1)[ 0 ] * <int> 2 - (p2 + 1)[ 0 ] - (p3 + 1)[ 0 ] + (p4 + 1)[ 0 ] + (p5 + 1)[ 0 ] +\
                (p6 + 1)[ 0 ] + (p7 + 1)[ 0 ] * <int> 2

            r = -(p1 + 2)[ 0 ] * <int> 2 - (p2 + 2)[ 0 ] - (p3 + 2)[ 0 ] + (p4 + 2)[ 0 ] + (p5 + 2)[ 0 ] +\
                (p6 + 2)[ 0 ] + (p7 + 2)[ 0 ] * <int> 2

        else:
            bgr_array[ i     ] = p4[ 0 ]
            bgr_array[ i + 1 ] = (p4 + 1)[ 0 ]
            bgr_array[ i + 2 ] = (p4 + 2)[ 0 ]
            continue

        if r < 0:
            r = <unsigned char> 0
        if r > 255:
            r = <unsigned char> 255
        if g < 0:
            g = <unsigned char> 0
        if g > 255:
            g = <unsigned char> 255
        if b < 0:
            b = <unsigned char> 0
        if b > 255:
            b = <unsigned char> 255

        bgr_array[ i     ] = <unsigned char> b
        bgr_array[ i + 1 ] = <unsigned char> g
        bgr_array[ i + 2 ] = <unsigned char> r




@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef object bilinear_c(
        unsigned char [:, :, :] rgb_array_,
        tuple size_,
        fx=None,
        fy=None
):
    """
    Resize an image using the bilinear filter algorithm (returns a copy).

    Bilinear interpolation is a resampling method used to resize images. It 
    calculates pixel values based on a weighted average of the four nearest 
    pixels. This method provides smoother results compared to nearest neighbor 
    interpolation.

    This function accepts images in 24-bit or 32-bit formats, but always returns 
    the resized image in 24-bit format without alpha transparency.

    Example Usage:
    --------------
    # Resize an image to 600x600
    image = bilinear(image, (600, 600))
    
    # Resize an image with scaling factors for both axes
    image = bilinear(image, (600, 600), fx=2, fy=2)

    Parameters
    ----------
    rgb_array_ : numpy.ndarray
        A 3D NumPy array of type uint8 representing an RGB image with shape 
        (width, height, n), where `n` is typically 3 for RGB images or 4 for RGBA 
        images. The image is resized using bilinear interpolation.

    size_ : tuple
        A tuple (width, height) representing the new dimensions (in pixels) for 
        the resized image.

    fx : float, optional
        The scaling factor for the width. If provided, it takes precedence over 
        the width value from `size_`. Default is None.

    fy : float, optional
        The scaling factor for the height. If provided, it takes precedence over 
        the height value from `size_`. Default is None.

    Returns
    -------
    pygame.Surface
        A new pygame.Surface object with the resized image in 24-bit format (RGB),
        without alpha transparency.

    """

    # Extract the current image width (w) and height (h)
    cdef Py_ssize_t w, h
    w, h = rgb_array_.shape[:2]

    # Initialize scale factors
    cdef:
        float rowscale, colscale
        float original_x, original_y
        unsigned int bl, br, tl, tr, \
            modxiplusonelim, modyiplusonelim
        unsigned int modxi, modyi
        int x, y
        float modxf, modyf, b, t, xf, yf
        unsigned int new_width = size_[0]
        unsigned int new_height = size_[1]

    # Apply scaling factors if provided
    if fx is not None:
        new_width = <unsigned int> (w * fx)
    if fy is not None:
        new_height = <unsigned int>(h * fy)

    # Calculate the scale for resizing (row and column scaling)
    rowscale = <float> w / <float> new_width
    colscale = <float> h / <float> new_height

    # Prepare an empty array to hold the resized image
    cdef unsigned char [: , :, ::1] new_rgb = \
        numpy.empty((new_height, new_width, 3), dtype=numpy.uint8)

    # Perform the resizing using bilinear interpolation (no GIL)
    with nogil:
        for y in prange(0, new_height, schedule=SCHEDULE, num_threads=THREADS):
            # Calculate the source y-coordinate (original position in the image)
            original_y = <float> y * colscale
            modyi = <unsigned int> original_y
            modyf = original_y - modyi
            modyiplusonelim = min(modyi + 1, h - 1)
            yf = <float> 1.0 - modyf

            for x in range(new_width):
                # Calculate the source x-coordinate (original position in the image)
                original_x = <float> x * rowscale
                modxi = <unsigned int> original_x
                modxf = original_x - modxi
                modxiplusonelim = min(modxi + 1, w - 1)
                xf = <float> 1.0 - modxf

                # Interpolate the red channel values
                bl = rgb_array_[modxi, modyi, 0]
                br = rgb_array_[modxiplusonelim, modyi, 0]
                tl = rgb_array_[modxi, modyiplusonelim, 0]
                tr = rgb_array_[modxiplusonelim, modyiplusonelim, 0]
                b = modxf * br + xf * bl
                t = modxf * tr + xf * tl
                new_rgb[y, x, 0] = <unsigned int> (modyf * t + yf * b + <float> 0.5)

                # Interpolate the green channel values
                bl = rgb_array_[modxi, modyi, 1]
                br = rgb_array_[modxiplusonelim, modyi, 1]
                tl = rgb_array_[modxi, modyiplusonelim, 1]
                tr = rgb_array_[modxiplusonelim, modyiplusonelim, 1]
                b = modxf * br + xf * bl
                t = modxf * tr + xf * tl
                new_rgb[y, x, 1] = <unsigned int> (modyf * t + yf * b + <float> 0.5)

                # Interpolate the blue channel values
                bl = rgb_array_[modxi, modyi, 2]
                br = rgb_array_[modxiplusonelim, modyi, 2]
                tl = rgb_array_[modxi, modyiplusonelim, 2]
                tr = rgb_array_[modxiplusonelim, modyiplusonelim, 2]
                b = modxf * br + xf * bl
                t = modxf * tr + xf * tl
                new_rgb[y, x, 2] = <unsigned int> (modyf * t + yf * b + <float> 0.5)

    # Return the resized image as a 24-bit pygame Surface
    return frombuffer(new_rgb, (new_width, new_height), 'RGB')






@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef tuple render_light_effect24(
        int x,
        int y,
        np.ndarray[np.uint8_t, ndim=3] background_rgb,
        np.ndarray[np.uint8_t, ndim=2] mask_alpha,
        float intensity        = 1.0,
        float [:] color        = numpy.asarray([128.0, 128.0, 128.0], dtype=numpy.float32),
        bint smooth            = False,
        bint saturation        = False,
        float sat_value        = 0.2,
        bint bloom             = False,
        unsigned int threshold = 110,
        bint heat              = False,
        float frequency        = 1
):
    """
    Generates a realistic lighting effect on a Pygame surface or texture.

    This function simulates a light effect that can be blended onto a surface 
    using additive blending (`BLEND_RGBA_ADD` in Pygame). The effect is generated 
    based on a mask texture and can include optional enhancements such as bloom, 
    heat waves, and saturation adjustments.

    Example Usage:
    --------------
    lit_surface, sw, sh = render_light_effect24(
        MOUSE_POS[0],
        MOUSE_POS[1],
        background_rgb,
        lalpha,
        intensity=5,
        color=c,
        smooth=False,
        saturation=False,
        sat_value=0.2,
        bloom=True,
        threshold=64,
        heat=False,
        frequency=1
    )

    Parameters:
    -----------
    x : int
        X-coordinate of the light source (must be within the screen width).

    y : int
        Y-coordinate of the light source (must be within the screen height).

    background_rgb : numpy.ndarray, shape (w, h, 3), dtype uint8
        A 3D NumPy array representing the RGB values of the background surface.

    mask_alpha : numpy.ndarray, shape (w, h), dtype uint8
        A 2D NumPy array containing the alpha (transparency) values of the light mask.
        Using a radial gradient mask with maximum intensity at the center is recommended.

    intensity : float, default=1.0
        Light intensity in the range [0.0, 20.0]. Higher values produce a stronger effect.

    color : numpy.ndarray, default=[128.0, 128.0, 128.0]
        A 3-element NumPy array representing the RGB color of the light in float format 
        (values in the range [0.0, 255.0]).

    smooth : bool, default=False
        If True, applies a blur effect to smooth the lighting.

    saturation : bool, default=False
        If True, applies a saturation effect to enhance color vibrancy.

    sat_value : float, default=0.2
        Adjusts the saturation level. The valid range is [-1.0, 1.0].

    bloom : bool, default=False
        If True, enables a bloom effect, which enhances the brightness of intense areas.

    threshold : int, default=110
        The brightness threshold used in the bloom effect. Pixels above this value 
        contribute to the bloom.

    heat : bool, default=False
        If True, applies a heat wave effect that distorts the lighting dynamically.

    frequency : float, default=1
        Determines the frequency of the heat wave effect. Must be an increasing value.

    Returns:
    --------
    tuple
        A tuple containing:
        - A 24-bit Pygame surface representing the generated light effect.
        - The surface width (`sw`).
        - The surface height (`sh`).

    Notes:
    ------
    - The output surface does not contain per-pixel alpha information.
    - Use `BLEND_RGBA_ADD` when blitting the surface to achieve an additive lighting effect.
    """


    assert intensity >= <float>0.0, '\nIntensity value cannot be > 0.0'


    cdef int w, h, lx, ly, ax, ay
    try:
        w, h = background_rgb.shape[:2]
    except (ValueError, pygame.error) as e:
        raise ValueError('\nArray shape not understood.')

    try:
        ax, ay = (<object>mask_alpha).shape[:2]
    except (ValueError, pygame.error) as e:
        raise ValueError('\nArray shape not understood.')

    # Return an empty surface if the x or y are not within the normal range.
    if (x < <unsigned short int>0) or (x > w - <unsigned short int>1) or \
            (y < <unsigned short int>0) or (y > h - <unsigned short int>1):
        return Surface((ax, ay), SRCALPHA), ax, ay

    # return an empty Surface when intensity = 0.0
    if intensity == <float>0.0:
        return Surface((ax, ay), SRCALPHA), ax, ay

    lx = ax >> 1
    ly = ay >> 1

    cdef:
        np.ndarray[np.uint8_t, ndim=3] rgb = empty((ax, ay, <unsigned short int>3), uint8)
        np.ndarray[np.uint8_t, ndim=2] alpha = empty((ax, ay), uint8)
        int i=0, j=0
        float f
        int w_low = lx
        int w_high = lx
        int h_low = ly
        int h_high = ly

    if x < lx:
        w_low = x
    elif x > w - lx:
        w_high = w - x

    if y < ly:
        h_low = y
    elif y >  h - ly:
        h_high = h - y

    rgb = background_rgb[x - w_low:x + w_high, y - h_low:y + h_high, :]
    alpha = mask_alpha[lx - w_low:lx + w_high, ly - h_low:ly + h_high]

    ax, ay = rgb.shape[:2]
    cdef:
        unsigned char [:, :, ::1] new_array = empty((ay, ax, 3), numpy.uint8)

    # NOTE the array is transpose
    with nogil:
        for i in prange(ax, schedule=SCHEDULE, num_threads=THREADS):
            for j in range(ay):
                f = alpha[i, j] * <float>ONE_255 * intensity
                new_array[j, i, 0] = <unsigned char>fmin(rgb[i, j, 0] * f * color[0], <float>255.0)
                new_array[j, i, 1] = <unsigned char>fmin(rgb[i, j, 1] * f * color[1], <float>255.0)
                new_array[j, i, 2] = <unsigned char>fmin(rgb[i, j, 2] * f * color[2], <float>255.0)

    ay, ax = new_array.shape[:2]

    if smooth:
        blur3d_c(new_array, npass=1)

    if saturation:
        saturation_c(new_array, sat_value)

    if heat:
        new_array = \
            heatwave_array24_horiz_c(
                numpy.asarray(new_array).transpose(1, 0, 2),
                alpha,
                frequency,
                (frequency % <unsigned short int>8) / <float>1000.0,
                attenuation=100,
                threshold=<unsigned short int>10
            )

    surface = pygame.image.frombuffer(new_array, (ax, ay), "RGB")

    if bloom:
        shader_bloom_fast1(surface, mask_=alpha)

    return surface, ax, ay




@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef inline tuple bpf24_c2(image, int threshold = 128, bint transpose=False):
    """

    Bright pass filter for 24-bit image (return tuple)
    
    This method is used internally.
    Calculate the luminance of every pixels and applied an attenuation c = lum2 / lum
    with lum2 = max(lum - threshold, 0) and
    lum = rgb[i, j, 0] * 0.299 + rgb[i, j, 1] * 0.587 + rgb[i, j, 2] * 0.114
    The output image will keep only bright areas. 
    You can adjust the threshold value default is 128 in order to get the desire changes.
    
    return a tuple ( 24-bit surface, 3d array )

    e.g:
    surf = bpf24_c2(image)
    
    :param transpose: 
        Transpose the final array (w and h are transpose if True)
        
    :param image: 
        pygame.Surface 24-bit format (RGB)  without per-pixel information
        
    :param threshold: 
        integer; Threshold to consider for filtering pixels luminance values,
        default is 128 range [0..255] unsigned char (python integer)
    
    :return: 
        Return a tuple (24-bit surface, 3d array format (w, h, 3) or
        (h, w, 3) if transposed).
        
    """

    # Fallback to default threshold value if argument
    # threshold value is incorrect
    if 0 > threshold > 255:
        raise AttributeError(
            '\nArgument threshold must be in range [0...255], fallback to default value 128.')

    assert isinstance(image, pygame.Surface), \
           "\nExpecting pygame surface for argument image, got %s " % type(image)

    try:
        rgb_array = image.get_view('3')

    except (pygame.error, ValueError):
        raise ValueError('\nCannot create a valid 3d array from the given surface.')

    cdef unsigned char [:, :, :] rgb = rgb_array

    cdef:
        int w, h

    w, h = rgb.shape[:2]

    # check sizes
    assert w>0 and h>0,\
        'Incorrect surface dimensions should be (w>0, h>0) got (w:%s, h:%s)' % (w, h)

    cdef:
        unsigned char [:, :, ::1] out_rgb= numpy.empty((w, h, 3), numpy.uint8) if \
            transpose == False else numpy.empty((h, w, 3), numpy.uint8)

        int i = 0, j = 0
        float lum, c
        unsigned char *r
        unsigned char *g
        unsigned char *b


    with nogil:
        for j in prange(0, h, schedule=SCHEDULE, num_threads=THREADS):
            for i in range(0, w):
                # ITU-R BT.601 luma coefficients
                r = &rgb[i, j, 0]
                g = &rgb[i, j, 1]
                b = &rgb[i, j, 2]

                lum = r[0] * <float>0.299 + g[0] * <float>0.587 + b[0] * <float>0.114

                if lum > threshold:

                    c = (lum - threshold) / lum

                    if not transpose:
                        out_rgb[ i, j, 0 ] = <unsigned char> (r[ 0 ] * c)
                        out_rgb[ i, j, 1 ] = <unsigned char> (g[ 0 ] * c)
                        out_rgb[ i, j, 2 ] = <unsigned char> (b[ 0 ] * c)

                    else:
                        out_rgb[j, i, 0] = <unsigned char>(r[0] * c)
                        out_rgb[j, i, 1] = <unsigned char>(g[0] * c)
                        out_rgb[j, i, 2] = <unsigned char>(b[0] * c)
                else:
                    out_rgb[ j, i, 0 ] = 0
                    out_rgb[ j, i, 1 ] = 0
                    out_rgb[ j, i, 2 ] = 0


    return frombuffer(out_rgb, (w, h), 'RGB'), out_rgb



# todo rename
@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef inline object bloom_effect_array24_c2(
        object surface_,
        const unsigned char threshold_,
        int smooth_ = 1,
        mask_       = None,
        bint fast_  = False):
    """

    Create a bloom effect on a Pygame surface (compatible 24 bit surface)

    Definition:
    Bloom is a computer graphics effect used in video games, demos,
    and high dynamic range rendering to reproduce an imaging artifact of real-world cameras.

    e.g:
    image = pygame.image.load('../Assets/px.png').convert_alpha()
    image = pygame.transform.smoothscale(image, (800, 600))

    mask = pygame.image.load('../Assets/alpha.png').convert_alpha()
    mask = pygame.transform.smoothscale(mask, (800, 600))
    
    # in the main loop 
    image = bloom_effect_array24_c2(image, 0, 1, mask, True)
    

    :param fast_: 
        bool; True | False. Speed up the bloom process using only the x16 surface and using
        an optimized bright pass filter (texture size downscale x4 prior processing)

    :param mask_: 
        Pygame.Surface representing the mask alpha. 
        Alpha values of 255 will render and bloom the entire image while zero will disable 
        the bloom effect (hide the pixels). 
    
    :param surface_: 
        pygame.Surface 24-bit format surface
        
    :param threshold_: 
        integer; Threshold value used by the bright pass algorithm (default 128)
        
    :param smooth_: 
        Number of Gaussian blur 5x5 to apply to downside images.
        
    :return : 
        Returns a pygame.Surface with a bloom effect (24 bit surface)


    """


    assert smooth_ > 0, \
        "Argument smooth_ must be > 0, got %s " % smooth_
    assert -1 < threshold_ < 256, \
        "Argument threshold must be in range [0...255] got %s " % threshold_

    cdef:
        Py_ssize_t w, h
        unsigned short int bit_size
        Py_ssize_t w2, h2, w4, h4, w8, h8, w16, h16
        bint x2, x4, x8, x16 = False

    w, h = surface_.get_size()
    bit_size = surface_.get_bitsize()

    with nogil:
        w2, h2 = w >> 1, h >> 1
        w4, h4 = w2 >> 1, h2 >> 1
        w8, h8 = w4 >> 1, h4 >> 1
        w16, h16 = w8 >> 1, h8 >> 1


        if w2 > 0 and h2 > 0:
            x2 = True
        else:
            x2 = False

        if w4 > 0 and h4 > 0:
            x4 = True
        else:
            x4 = False

        if w8 > 0 and h8 > 0:
            x8 = True
        else:
            x8 = False

        if w16 > 0 and h16 > 0:
            x16 = True
        else:
            x16 = False

    # check if the first reduction is > 0
    # if not we cannot blur that surface (too small)
    if not x2:
        return pygame.Surface((1, 1))

    if fast_:

        x2, x4, x8 = False, False, False

        s4 = scale(surface_, (w >> 2, h >> 2))
        bpf_surface, bpf_array = bpf24_c2(s4, threshold=threshold_, transpose=True)
        bpf_surface = scale(bpf_surface, (w, h))

    else:
        # BRIGHT PASS FILTER
        bpf_surface, bpf_array = bpf24_c2(surface_, threshold=threshold_, transpose=True)

    if x2:
        s2 = scale(bpf_surface, (w2, h2))

        s2_array = blur1d_cp_c(
            s2.get_buffer(),
            w2, h2,
            npass=smooth_,
            format_32=False
        )
        b2_blurred = frombuffer(s2_array, (w2, h2), 'RGB')

        s2 = smoothscale(b2_blurred, (w, h))

        surface_.blit(s2, (0, 0), special_flags=BLEND_RGB_ADD)

    if x4:
        s4 = scale(bpf_surface, (w4, h4))

        s4_array = blur1d_cp_c(
            s4.get_buffer(),
            w4,
            h4,
            npass = smooth_,
            format_32 = False
        )

        b4_blurred = frombuffer(s4_array, (w4, h4), 'RGB')
        s4 = smoothscale(b4_blurred, (w, h))
        surface_.blit(s4, (0, 0), special_flags=BLEND_RGB_ADD)

    if x8:
        s8 = scale(bpf_surface, (w8, h8))


        s8_array = blur1d_cp_c(
            s8.get_buffer(),
            w8,
            h8,
            npass = smooth_,
            format_32 = False
        )

        b8_blurred = frombuffer(s8_array, (w8, h8), 'RGB')
        s8 = smoothscale(b8_blurred, (w, h))
        surface_.blit(s8, (0, 0), special_flags=BLEND_RGB_ADD)

    # if x16:
    #     s16 = scale(bpf_surface, (w16, h16))
    #     s16_array = numpy.array(s16.get_view('3'))
    #
    #     if smooth_ > 1:
    #         for r in range(smooth_):
    #             s16_array = blur3d_cp_c(s16_array)
    #     else:
    #         s16_array = blur3d_cp_c(s16_array)
    #     b16_blurred = frombuffer(s16_array, (w16, h16), 'RGB')
    #     s16 = smoothscale(b16_blurred, (w, h))
    #     surface_.blit(s16, (0, 0), special_flags = BLEND_RGB_ADD)

    # Alternate way of x16 slightly faster
    if x16:

        s16_array = blur1d_cp_c(
            bytearray(tobytes(scale(bpf_surface, (w16, h16)), 'RGB')),
            w16,
            h16,
            npass = smooth_,
            format_32 = False
        )

        b16_blurred = frombuffer(s16_array, (w16, h16), 'RGB')
        s16 = smoothscale(b16_blurred, (w, h))
        surface_.blit(s16, (0, 0), special_flags=BLEND_RGB_ADD)

    if mask_ is not None:
        filtering24_c(surface_, mask_)

    return surface_


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef area24_cc(
        int x, int y,
        np.ndarray[np.uint8_t, ndim=3] background_rgb,
        np.ndarray[np.uint8_t, ndim=2] mask_alpha,
        float intensity=1.0,
        float [::1] color=numpy.asarray([128.0, 128.0, 128.0], dtype=numpy.float32),
        bint smooth=False,
        bint saturation=False,
        float sat_value=0.2,
        bint bloom=False,
        unsigned char bloom_threshold=64
):
    """
    Generates a realistic lighting effect on a Pygame surface or texture.

    This function simulates a light effect that can be blended onto a surface using
    additive blending (`BLEND_RGBA_ADD` in Pygame). It supports optional enhancements
    such as bloom, smoothing, and saturation adjustments.

    Lighting Modes:
    ---------------
    - **Smooth**: Applies a Gaussian blur with a 5x5 kernel to soften the lighting effect.
    - **Saturation**: Adjusts color intensity using HSL color conversion. A value range 
      of [-1.0, 1.0] is supported, with higher values increasing vibrancy and values below 
      zero desaturating the output.
    - **Bloom**: Enhances brightness by applying a bloom effect, making bright areas 
      appear more intense.

    Lighting Parameters:
    --------------------
    - **Intensity**: Defines the brightness of the light. If set to zero, the function 
      returns an empty `pygame.Surface` with the `RLEACCEL` flag.
    - **Color**: Specifies the light’s RGB coloration. Defaults to (128, 128, 128).

    Example Usage:
    --------------
    lit_surface, sw, sh = area24_cc(
        MOUSE_POS[0], MOUSE_POS[1], background_rgb, lalpha, 
        intensity=5, color=c, smooth=False, saturation=False, 
        sat_value=0.2, bloom=True, bloom_threshold=0
    )

    Parameters:
    -----------
    x : int
        X-coordinate of the light source (must be within screen width).

    y : int
        Y-coordinate of the light source (must be within screen height).

    background_rgb : numpy.ndarray, shape (w, h, 3), dtype uint8
        A 3D NumPy array representing the RGB values of the background surface.

    mask_alpha : numpy.ndarray, shape (w, h), dtype uint8
        A 2D NumPy array containing the alpha values of the light mask.
        Using a radial gradient mask with maximum intensity at the center is recommended.

    color : numpy.ndarray, default=[128.0, 128.0, 128.0]
        A 3-element NumPy array representing the RGB color of the light in float format
        (values in the range [0.0, 255.0]).

    intensity : float, default=1.0
        Light intensity in the range [0.0, 20.0]. Higher values produce a stronger effect.

    smooth : bool, default=False
        If True, applies a blur effect to soften the lighting.

    saturation : bool, default=False
        If True, increases color intensity using HSL conversion.

    sat_value : float, default=0.2
        Adjusts the saturation level. The valid range is [-1.0, 1.0].
        Higher values increase vibrancy, while negative values desaturate the effect.

    bloom : bool, default=False
        If True, enables a bloom effect, enhancing brightness.

    bloom_threshold : int, default=64
        The brightness threshold for the bloom effect, in the range [0, 255].
        Lower values create a stronger bloom effect.

    Returns:
    --------
    tuple
        A tuple containing:
        - A 24-bit Pygame surface representing the generated light effect.
        - The surface width (`sw`).
        - The surface height (`sh`).

    Notes:
    ------
    - The output surface does not contain per-pixel alpha information.
    - Use `BLEND_RGBA_ADD` when blitting the surface to achieve an additive lighting effect.
    """


    if intensity < 0.0:
        raise ValueError('\nIntensity value cannot be < 0.0')

    cdef int w, h, lx, ly, ax, ay
    try:
        w, h = background_rgb.shape[:2]
    except (ValueError, pygame.error) as e:
        raise ValueError('\nArray shape not understood.')

    try:
        ax, ay = mask_alpha.shape[:2]
    except (ValueError, pygame.error) as e:
        raise ValueError('\nArray shape not understood.')

    # return an empty Surface when intensity = 0.0
    if intensity == 0:
        return Surface((ax, ay), SRCALPHA), ax, ay

    lx = ax >> 1
    ly = ay >> 1

    cdef:
        int i=0, j=0
        float f
        unsigned int w_low = lx
        unsigned int w_high = lx
        unsigned int h_low = ly
        unsigned int h_high = ly
        unsigned int x1, x2, y1, y2

    with nogil:
        if x < lx:
            w_low = x
        elif x > w - lx:
            w_high = w - x

        if y < ly:
            h_low = y
        elif y >  h - ly:
            h_high = h - y

        x1 = x - w_low
        x2 = x + w_high
        y1 = y - h_low
        y2 = y + h_high
        x1 = max(x1, <unsigned short int>0)
        x1 = min(x1, w)
        x2 = max(x2, <unsigned short int>0)
        x2 = min(x2, w)
        y1 = max(y1, <unsigned short int>0)
        y1 = min(y1, h)
        y2 = max(y2, <unsigned short int>0)
        y2 = min(y2, h)

    # RGB block and ALPHA
    cdef:
        unsigned char [:, : , :] rgb = background_rgb[x1:x2, y1:y2, :]
        unsigned char [:, :] alpha = mask_alpha[lx - w_low:lx + w_high, ly - h_low:ly + h_high]

    # RGB ARRAY IS TRANSPOSED IN THE LOOP
    ax, ay = rgb.shape[:2]
    cdef:
        unsigned char [:, :, ::1] new_array = \
            numpy.ascontiguousarray(empty((ay, ax, 3), numpy.uint8))
        float c1 = <float>ONE_255 * intensity
        float red   = color[0]
        float green = color[1]
        float blue  = color[2]
        unsigned char * index

    with nogil:

        for j in prange(ay, schedule=SCHEDULE, num_threads=THREADS):
            for i in range(ax):
                index = &new_array[ j, i, 0 ]
                if alpha[i, j] == 0:
                    index[0] = <unsigned char>0
                    (index + 1)[0] = <unsigned char>0
                    (index + 2)[0] = <unsigned char>0
                    continue
                f = alpha[i, j] * c1
                index[0] = <unsigned char>min(rgb[i, j, 0] * f * red, <unsigned char>255)
                (index + 1)[0] = <unsigned char>min(rgb[i, j, 1] * f * green, <unsigned char>255)
                (index + 2)[0] = <unsigned char>min(rgb[i, j, 2] * f * blue, <unsigned char>255)

    # As the array is transposed we
    # we need to adjust ax and ay (swapped).
    ay, ax = new_array.shape[:2]

    # Return an empty surface if the x or y are not within the normal range.
    if ax <1 or ay < 1:
        return Surface((ax, ay), SRCALPHA), ax if ax > 0 else 0, ay if ay > 0 else 0

    if saturation:
        saturation_mask_inplace(
            new_array,
            sat_value,
            alpha,
            w = ax, h = ay
        )

    cdef unsigned char [:, :, :] n_cp =\
        numpy.asarray(new_array, dtype=uint8)

    if bloom:
        # surf = bpf24_c(new_array, threshold = bloom_threshold)
        # blend_add_array_c(new_array, surf.get_view('3'))

        bpf_inplace_c(new_array, ay, ax, threshold=bloom_threshold)
        blend_add_array_c(new_array, n_cp)


    if smooth:
        blur3d_c(new_array, npass=1)

    surface = frombuffer(new_array, (ax, ay), 'RGB')
    surface.set_colorkey((0, 0, 0, 0), pygame.RLEACCEL)

    return surface, ax, ay



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef object chromatic(
        surface_,
        unsigned int delta_x,
        unsigned int delta_y,
        float zoom=0.9999,
        float fx=0.02
):
    """
    Chromatic aberration (return a new surface)

    Create a chromatic aberration with an amplitude proportional to the
    distance from the centre of the effect
    
    e.g:
    source = chromatic(source, 400, 300, 0.999, fx=0.04)
    
    :param surface_:
        pygame.Surface 24, 32-bit compatible
        
    :param delta_x: 
        int; chromatic centre effect coordinate X, must be in range [0 ... w]
        
    :param delta_y: 
        int; chromatic centre effect coordinate Y, must be in range [0 ... h]
        
    :param zoom: 
        float; zoom factor 0.9999 (no zoom, full image), < 1.0 zoom-in. Must
        be in range [0.0 ... 0.9999]
        
    :param fx: 
        channel rgb layer offset default 0.02. Must be in range [0.0 ... 0.2]
        
    :return: 
        a chromatic aberration effect
        
    """
    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)

    cdef Py_ssize_t w, h
    w, h = surface_.get_size()


    if w == 0 or h == 0:
        raise ValueError("Surface w or h cannot be null!")

    if delta_x < 0 or delta_y < 0:
        raise ValueError("Arguments delta_x and delta_y must be > 0")

    delta_x %= w
    delta_y %= h

    if zoom < 0 or <float>floor(zoom) > <float>0.99999999:
        raise ValueError("Argument zoom must be in range [0.0 ... 0.999]")

    if 0 > <float>floor(fx) > 0.2:
        raise ValueError("Argument fx must be in range [0.0 ... 0.2]")

    cdef unsigned char [:, :, :] rgb_array

    try:
        rgb_array = surface_.get_view('3')

    except Exception as e:
        raise ValueError("\nCannot reference source pixels into a 3d array.\n %s " % e)

    cdef:
        unsigned char [:, :, ::1] new_array = \
            numpy.ascontiguousarray(empty((h, w, 3), dtype=numpy.uint8))
        int i = 0, j = 0
        float dw = <float>delta_y / <float>w
        float dh = <float>delta_x / <float>h
        float nx, ny, theta_rad, dist, new_dist, cos_, sin_
        unsigned int new_j, new_i

    with nogil:
        for j in prange(0, h, schedule=SCHEDULE, num_threads=THREADS):
            ny = <float> (<float> j / <float> w) - dw
            for i in range(w):
                nx = <float>(<float>i / <float>h) - dh

                theta_rad = <float>atan2 (ny,nx)
                cos_ = <float>cos(<float>theta_rad)
                sin_ = <float>sin(<float>theta_rad)

                dist = <float>sqrt(nx * nx + ny * ny)
                new_dist = dist * (zoom - fx)

                new_j = <unsigned int>((sin_ * new_dist + dw) * <float>w)
                new_i = <unsigned int>((cos_ * new_dist + dh) * <float>h)
                new_array[j, i, 0] = rgb_array[new_i, new_j, 0]

                new_dist = dist * (zoom  - fx * <float>2.0)

                new_i = <unsigned int>((cos_ * new_dist + dh) * <float>h)
                new_j = <unsigned int> ((sin_ * new_dist + dw) * <float> w)
                new_array[j, i, 1] = rgb_array[new_i, new_j, 1]

                new_dist = dist * (zoom  - fx * <float>3.0)

                new_j = <unsigned int>((sin_ * new_dist + dw) * <float>w)
                new_i = <unsigned int>((cos_ * new_dist + dh) * <float>h)

                new_array[j, i, 2] = rgb_array[new_i, new_j, 2]

    return frombuffer(new_array, (w, h), 'RGB')



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef object chromatic_inplace(
        surface_,
        unsigned int delta_x,
        unsigned int delta_y,
        float zoom=0.9999,
        float fx=0.02
):
    """
    Chromatic aberration (inplace)

    Create a chromatic aberration with an amplitude proportional to the
    distance from the centre of the effect
    
    e.g:
    surf = chromatic_inplace(background, MOUSE_POS.x, MOUSE_POS.y, 0.999, fx=0.04)

    :param surface_:
        pygame.Surface 24, 32-bit compatible
        
    :param delta_x: 
        int; chromatic centre effect coordinate X, must be in range [0 ... w]
        
    :param delta_y: 
        int; chromatic centre effect coordinate Y, must be in range [0 ... h]
        
    :param zoom: 
        float; zoom factor 0.9999 (no zoom, full image), < 1.0 zoom-in. Must
        be in range [0.0 ... 0.9999]
        
    :param fx: 
        channel rgb layer offset default 0.02. Must be in range [0.0 ... 0.2]
        
    :return: 
        a chromatic aberration effect
        
    """

    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)

    cdef Py_ssize_t w, h
    w, h = surface_.get_size()


    if w == 0 or h == 0:
        raise ValueError("Surface w or h cannot be null!")

    if delta_x < 0 or delta_y < 0:
        raise ValueError("Arguments delta_x and delta_y must be > 0")

    delta_x %= w
    delta_y %= h

    if zoom < 0 or <float>floor(zoom) > <float>0.99999999:
        raise ValueError("Argument zoom must be in range [0.0 ... 0.999]")

    if 0 > <float>floor(fx) > 0.2:
        raise ValueError("Argument fx must be in range [0.0 ... 0.2]")

    cdef unsigned char [:, :, :] rgb_array

    try:
        # bgr_array = numpy.array(pixels3d(surface_), copy=True, dtype=uint8)
        rgb_array = numpy.asarray(surface_.get_view('3'), dtype=uint8, order='F')
    except Exception as e:
        raise ValueError("\nCannot reference source pixels into a 3d array.\n %s " % e)

    cdef:
        unsigned char [:, :, :] new_array = surface_.get_view('3')
        int i = 0, j = 0
        float dw = <float>delta_y / <float>w
        float dh = <float>delta_x / <float>h
        float nx, ny, theta_rad, dist, new_dist, cos_, sin_
        unsigned int new_j, new_i

    with nogil:
        for j in prange(0, h, schedule=SCHEDULE, num_threads=THREADS):
            ny = <float> (<float> j / <float> w) - dw
            for i in range(w):
                nx = <float>(<float>i / <float>h) - dh

                theta_rad = <float>atan2 (ny,nx)
                cos_ = <float>cos(<float>theta_rad)
                sin_ = <float>sin(<float>theta_rad)

                dist = <float>sqrt(nx * nx + ny * ny)
                new_dist = dist * (zoom - fx)

                new_j = <unsigned int>((sin_ * new_dist + dw) * <float>w)
                new_i = <unsigned int>((cos_ * new_dist + dh) * <float>h)
                new_array[i, j, 0] = rgb_array[new_i, new_j, 0]

                new_dist = dist * (zoom  - fx * <float>2.0)

                new_i = <unsigned int>((cos_ * new_dist + dh) * <float>h)
                new_j = <unsigned int>((sin_ * new_dist + dw) * <float> w)
                new_array[i, j, 1] = rgb_array[new_i, new_j, 1]

                new_dist = dist * (zoom  - fx * <float>3.0)

                new_j = <unsigned int>((sin_ * new_dist + dw) * <float>w)
                new_i = <unsigned int>((cos_ * new_dist + dh) * <float>h)

                new_array[i, j, 2] = rgb_array[new_i, new_j, 2]



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef object zoom(surface_, unsigned int delta_x, unsigned int delta_y, float zx=0.9999):
    """
    Zoom within an image (return a copy)

    Zoom-in or zoom-out (factor zx) toward a given centre point (delta_x, delta_y)
    Compatible 24, 32-bit image format.
    The final output image will be same than input format.  
    This algorithm will have slightly better performances with 24-bit image
    
    e.g:
    surf = zoom(background, MOUSE_POS.x, MOUSE_POS.y, z)
    
    :param surface_ : 
        pygame.Surface
        
    :param delta_x  : 
        int; Zoom centre x coordinate must be in range [0 ... w]
        
    :param delta_y  : 
        int; Zoom centre y coordinate must be in range [0 ... h]
        
    :param zx       : 
        float; Zoom factor must be in range ]0.0 ... 1.0[, zoom intensity will be 
        attenuated around 1.0 and maximum around 0.0
         
    :return         : 
        Returns a zoomed image.
        the output image is same format than input image.  
        
    """

    cdef int w, h
    w, h = surface_.get_size()

    cdef short int byte_size

    byte_size = surface_.get_bytesize()


    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)

    if w == 0 or h == 0:
        raise ValueError("Surface w or h cannot be null!")

    if delta_x < 0 or delta_y < 0:
        raise ValueError("Arguments delta_x and delta_y must be > 0")

    delta_x %= w
    delta_y %= h

    if zx <= 0.0 or zx >= 1.0:
        raise ValueError('Argument zx must be in range ]0.0 ... 1.0[')


    cdef unsigned char [:, :, :] rgb_array

    try:
        rgb_array = surface_.get_view('3') if byte_size == 3 else \
        numpy.asarray(surface_.get_view('0'), dtype = numpy.uint8).reshape(h, w, byte_size)

    except Exception as e:
        raise ValueError("\nCannot reference source pixels into a 3d array.\n %s " % e)

    cdef:
        unsigned char [:, :, :] new_array = \
            empty((h, w, 4), dtype=numpy.uint8) if byte_size == 4 else \
                empty((h, w, 3), dtype = numpy.uint8)

        int i = 0, j = 0
        float dw = delta_y / <float>w
        float dh = delta_x / <float>h
        float nx, ny, theta, new_dist
        unsigned int new_j, new_i

    with nogil:
        for j in prange(0, h, schedule=SCHEDULE, num_threads=THREADS):

            ny = <float> (<float> j / <float> w) - dw

            for i in range(w):

                nx = <float>(<float>i / <float>h) - dh

                theta = <float>atan2 (ny,nx)

                new_dist = <float>sqrt(nx * nx + ny * ny) * zx

                new_j = <unsigned int> ((<float> sin(<float> theta) * new_dist + dw) * <float> w)
                new_i = <unsigned int> ((<float> cos(<float> theta) * new_dist + dh) * <float> h)

                if byte_size == 3:

                    new_array[ j, i, 0 ] = rgb_array[ new_i, new_j, 0 ]
                    new_array[ j, i, 1 ] = rgb_array[ new_i, new_j, 1 ]
                    new_array[ j, i, 2 ] = rgb_array[ new_i, new_j, 2 ]

                else:
                    new_array[ j, i, 0 ] = rgb_array[ new_j, new_i, 0 ]
                    new_array[ j, i, 1 ] = rgb_array[ new_j, new_i, 1 ]
                    new_array[ j, i, 2 ] = rgb_array[ new_j, new_i, 2 ]
                    new_array[ j, i, 3 ] = rgb_array[ new_j, new_i, 3 ]

    return frombuffer(new_array, (w, h), 'RGB' if byte_size == 3 else 'BGRA')




@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef inline void zoom_inplace(
        surface_,
        unsigned int delta_x,
        unsigned int delta_y,
        float zx=0.9999
):
    """
    Zoom within an image (inplace)

    Zoom-in or zoom-out (factor zx) toward a given centre point (delta_x, delta_y)
    Compatible with 24, 32-bit image format. 
    The input image format is unchanged during the process.
    
    
    e.g:
    surf = zoom(background, MOUSE_POS.x, MOUSE_POS.y, z)

    :param surface_ : 
        pygame.Surface compatible 24, 32-bit 
        
    :param delta_x  : 
        int; Zoom centre x coordinate must be in range [0 ... w]
        
    :param delta_y  : 
        int; Zoom centre y coordinate must be in range [0 ... h]
        
    :param zx       : 
        float; Zoom factor must be in range ]0.0 ... 1.0[
        The zoom effect will be attenuated close to 1.0 and max around 0.0 
        default is 0.9999 
        
    :return         : 
        void
    """

    cdef int w, h
    w, h = surface_.get_size()

    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)

    if w == 0 or h == 0:
        raise ValueError("Surface w or h cannot be null!")

    if delta_x < 0 or delta_y < 0:
        raise ValueError("Arguments delta_x and delta_y must be > 0")

    delta_x %= w
    delta_y %= h

    if zx <=0.0 or zx >= 1.0:
         raise ValueError('Argument zx must be in range ]0.0 ... 1.0[')

    cdef:
        unsigned char [:, :, :] rgb_array

    try:
        rgb_array = surface_.get_view('3')

    except Exception as e:
        raise ValueError("\nCannot reference source pixels into a 3d array.\n %s " % e)

    cdef unsigned char [::1, :, :] rgb_array_cp = numpy.asarray(rgb_array, order='F')

    cdef:
        int i = 0, j = 0
        float dw = delta_y / <float>w
        float dh = delta_x / <float>h
        float nx, ny, theta, new_dist
        unsigned int new_j, new_i

    with nogil:
        for j in prange(0, h, schedule=SCHEDULE, num_threads=THREADS):

            ny = <float> (<float> j / <float> w) - dw

            for i in range(w):

                nx = <float>(<float>i / <float>h) - dh

                theta = <float>atan2 (ny,nx)

                new_dist = <float>sqrt(nx * nx + ny * ny) * zx

                new_j = <unsigned int>((<float>sin(<float>theta) * new_dist + dw) * <float>w)
                new_i = <unsigned int>((<float>cos(<float>theta) * new_dist + dh) * <float>h)

                rgb_array[i, j, 0] = rgb_array_cp[new_i, new_j, 0]
                rgb_array[i, j, 1] = rgb_array_cp[new_i, new_j, 1]
                rgb_array[i, j, 2] = rgb_array_cp[new_i, new_j, 2]



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef void Luma_GreyScale(object surface_):
    """
    Convert image into greyscale using yiq (luma information)
    
    e.g:
    Luma_GreyScale(image)
    
    :param surface_: 
        pygame.Surface;
        
    :return: 
        void
        
    """
    cdef unsigned char [:,:,:] arr = surface_.get_view('3')
    Luma_GreyScale_c(arr)



# -------------------------------------------------------------------------------------------------------------------


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef heatwave_array24_horiz_c(
            unsigned char [:, :, :] rgb_array,
            unsigned char [:, :] mask_array,
            float frequency, float amplitude, float attenuation=0.10,
            unsigned char threshold=64
):
    """
    HORIZONTAL HEATWAVE 

    DISTORTION EQUATION: 
    distortion = sin(x * attenuation + frequency) * amplitude * mask[x, y]
    Amplitude is equivalent to ((frequency % 2) / 1000.0) and will define the maximum pixel displacement.
    The highest the frequency the lowest the heat wave  
    
    e.g
    new_array = \
            heatwave_array24_horiz_c(
                numpy.asarray(new_array).transpose(1, 0, 2),
                alpha,
                frequency,
                (frequency % <unsigned short int>8) / <float>1000.0,
                attenuation=100,
                threshold=<unsigned short int>10
            )


    :param rgb_array: 
        numpy.ndarray or MemoryViewSlice, array shape (w, h, 3) containing RGB values
        
    :param mask_array: 
        numpy.ndarray or  MemoryViewSlice shape (w, h) containing alpha values
        
    :param frequency: 
        float; increment value. The highest the frequency the lowest the heat wave
          
    :param amplitude: 
        float; variable amplitude. Max amplitude is 10e-3 * 255 = 2.55 
        when alpha is 255 otherwise 10e-3 * alpha.
        
    :param attenuation: 
        float; default 0.10
        
    :param threshold: 
        unsigned char; Compare the alpha value with the threshold.
        if alpha value > threshold, apply the displacement to the texture otherwise no change
        
    :return: 
        Return a pygame.Surface 24 bit format
         
    """


    cdef int w, h
    w, h = (<object>rgb_array).shape[:2]

    cdef:
        unsigned char [:, :, ::1] new_array = empty((h, w, 3), dtype=numpy.uint8)
        int x = 0, y = 0, xx, yy
        float distortion


    with nogil:
        for x in prange(0, w, schedule=SCHEDULE, num_threads=THREADS):

            for y in range(h):
                distortion = sin(x * attenuation + frequency) * amplitude * mask_array[x, y]

                xx = <int>(x  + distortion + rand() * <float>0.0002)
                if xx > w - 1:
                    xx = w - 1
                if xx < 0:
                    xx = 0

                if mask_array[x, y] > threshold:
                    new_array[y, x, 0] = rgb_array[xx, y, 0]
                    new_array[y, x, 1] = rgb_array[xx, y, 1]
                    new_array[y, x, 2] = rgb_array[xx, y, 2]
                else:
                    new_array[y, x, 0] = rgb_array[x, y, 0]
                    new_array[y, x, 1] = rgb_array[x, y, 1]
                    new_array[y, x, 2] = rgb_array[x, y, 2]

    return numpy.asarray(new_array)



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef inline void Luma_GreyScale_c(unsigned char [:, :, :] rgb_array):
    """
    Convert an RGB image to grayscale using the YIQ color model (luma information).

    The YIQ color space separates the brightness (luma) and color (chroma) components of an image.
    This function uses only the luma component (Y) from the YIQ model to convert an image to grayscale,
    preserving brightness but discarding the color information. The conversion results in a grayscale 
    image based on the luminance (brightness) of the original image.

    Example Usage:
    --------------
    # Convert an image to grayscale based on luminance (Y channel)
    Luma_GreyScale(image)

    Parameters
    ----------
    rgb_array : numpy.ndarray
        A 3D NumPy array of type uint8 representing an RGB image with shape (width, height, 3).
        The image is converted to grayscale in-place, modifying the original `rgb_array`.
        Any changes to this array will affect the original surface that the array references.

    Returns
    -------
    void
        This function modifies the `rgb_array` in place and does not return any value.
    """
    
    # Get the width (w) and height (h) of the image
    cdef Py_ssize_t w, h
    w, h = rgb_array.shape[:2]

    # Declare loop variables and temporary structures for color conversion
    cdef:
        int i = 0, j = 0
        yiq yiq_
        rgb rgb_
        float h_
        unsigned char *r
        unsigned char *g
        unsigned char *b

    # Perform the grayscale conversion in parallel (using OpenMP parallelization)
    with nogil:
        # Iterate through each row of the image
        for j in prange(h, schedule=SCHEDULE, num_threads=THREADS):
            # Iterate through each column of the current row
            for i in range(w):
                # Access the RGB values of the current pixel
                r = &rgb_array[i, j, 0]
                g = &rgb_array[i, j, 1]
                b = &rgb_array[i, j, 2]
                
                # Convert the RGB values to YIQ color space (luma information)
                yiq_ = rgb_to_yiq(r[0] * <float>ONE_255, g[0] * <float>ONE_255, b[0] * <float>ONE_255)

                # Replace the RGB values with the Y (luma) value from YIQ, turning it into grayscale
                # The grayscale value is derived from the Y component of the YIQ model
                r[0] = <unsigned char>min(<unsigned char>(yiq_.y * <float>255.0), <unsigned char>255)
                g[0] = <unsigned char>min(<unsigned char>(yiq_.y * <float>255.0), <unsigned char>255)
                b[0] = <unsigned char>min(<unsigned char>(yiq_.y * <float>255.0), <unsigned char>255)








