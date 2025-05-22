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

import pygame

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=ImportWarning)

import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

from libc.math cimport roundf as round_c

try:
    cimport cython
    from cython.parallel cimport prange, parallel

except ImportError:
    raise ImportError("\n<cython> library is missing on your system."
          "\nTry: \n   C:\\pip install cython on a window command prompt.")

from PygameShader.config import OPENMP, THREAD_NUMBER, __VERSION__

cdef int THREADS = 1

if OPENMP:
     THREADS = THREAD_NUMBER

DEF SCHEDULE = 'static'

DEF ONE_255      = 1.0/255.0

@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef (float, float, float) rgb_2_yiq(
        const unsigned char r,
        const unsigned char g,
        const unsigned char b):
    """
    Convert RGB color values into the YIQ color model.

    The YIQ color model was used in NTSC television broadcasting (1953).  
    - Y (Luma) represents brightness and is the only component used by black-and-white TVs.  
    - I (In-phase) represents the orange-blue chrominance.  
    - Q (Quadrature) represents the purple-green chrominance.  

    This function takes RGB values in the range [0, 255] and converts them into  
    YIQ color space using normalized floating-point calculations.

    :param r: 
        Unsigned char (integer) representing the red channel, in range [0, 255].

    :param g: 
        Unsigned char (integer) representing the green channel, in range [0, 255].

    :param b: 
        Unsigned char (integer) representing the blue channel, in range [0, 255].

    :raises ValueError:
        If any input value is outside the valid range [0, 255].

    :return: 
        A tuple (Y, I, Q) where:
        - Y (float): Luma component (brightness).
        - I (float): Orange-blue chrominance.
        - Q (float): Purple-green chrominance.
    """

    # Convert RGB values from integer [0, 255] to floating-point [0, 1]
    # using ONE_255 = 1.0 / 255.0 for normalization
    cdef yiq yiq_
    yiq_ = rgb_to_yiq(r * <float> ONE_255, g * <float> ONE_255, b * <float> ONE_255)

    # Return the YIQ components as a tuple
    return yiq_.y, yiq_.i, yiq_.q


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef (unsigned char, unsigned char, unsigned char) yiq_2_rgb(
        const float y,
        const float i,
        const float q
):
    """
    Convert YIQ color model values into equivalent RGB values.

    The YIQ color model was used in NTSC television broadcasting (1953).  
    - Y (Luma) represents brightness and is the only component used by black-and-white TVs.  
    - I (In-phase) represents the orange-blue chrominance.  
    - Q (Quadrature) represents the purple-green chrominance.  

    This function converts YIQ values into RGB format.  
    The resulting RGB values are scaled to the [0, 255] range and rounded to the nearest integer.

    :param y: 
        Float representing the Luma (brightness) component.
        
    :param i: 
        Float representing the In-phase chrominance (orange-blue contrast).
        
    :param q: 
        Float representing the Quadrature chrominance (purple-green contrast).

    :return: 
        A tuple (R, G, B) where:
        - R (unsigned char): Red value in the range [0, 255].
        - G (unsigned char): Green value in the range [0, 255].
        - B (unsigned char): Blue value in the range [0, 255].

    Output RGB values are rounded to the nearest integer.
    """
    cdef rgb rgb_

    # Convert YIQ to RGB using the transformation function
    rgb_ = yiq_to_rgb(y, i, q)

    # Scale floating-point RGB values from [0,1] to [0,255] and round
    return <unsigned char> round_c(rgb_.r * <float> 255.0), \
           <unsigned char> round_c(rgb_.g * <float> 255.0), \
           <unsigned char> round_c(rgb_.b * <float> 255.0)





@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef void RGB_TO_YIQ_inplace(
        object image_surface,  # The image surface containing the RGB data (pygame.Surface)
        bint include_y = True,  # Flag to include the Y (luma) component in the conversion
        bint include_i = False,  # Flag to include the I (orange-blue contrast) component
        bint include_q = False   # Flag to include the Q (purple-green contrast) component
):
    """
    Converts a 24-bit or 32-bit RGB image into the YIQ color model in-place.

    The YIQ color model separates brightness (luma) and color information (chroma). 
    - Y (luma) represents the brightness of the image and is used in black-and-white television.
    - I and Q (chrominance) represent color information, with I corresponding to orange-blue contrast 
      and Q corresponding to purple-green contrast.

    This function modifies the pixel data of the given `image_surface` directly.

    Example:
        RGB_TO_YIQ_inplace(image)

    :param image_surface: 
        A `pygame.Surface` object, compatible with 24-bit or 32-bit images.

    :param include_y: 
        A boolean flag (default: `True`). If `True`, the Y (luma) component is included in the conversion.

    :param include_i: 
        A boolean flag (default: `False`). If `True`, the I (orange-blue contrast) component is included.

    :param include_q: 
        A boolean flag (default: `False`). If `True`, the Q (purple-green contrast) component is included.

    :return: 
        None. This function modifies the image data in place.
    """
    # Check that the image_surface is a valid pygame.Surface
    if not isinstance(image_surface, pygame.Surface):
        raise TypeError("The provided image_surface must be a valid pygame.Surface object.")

    # Check that the flags are boolean values
    if not isinstance(include_y, bool) or not isinstance(include_i, bool) or not isinstance(include_q, bool):
        raise ValueError("Flags 'include_y', 'include_i', and 'include_q' must be boolean values.")

    # Check if the image surface has the correct pixel format (24 or 32-bit)
    if image_surface.get_size() == (0, 0):
        raise ValueError("The provided image_surface has an invalid size (0x0). Ensure the image is loaded correctly.")

    # Get a 3D view of the image data (RGB values) as an unsigned char array
    cdef unsigned char [:, :, :] rgb_array = image_surface.get_view('3')

    cdef Py_ssize_t w, h
    w, h = (<object>rgb_array).shape[ :2 ]

    try:
        # Call the C function to perform the in-place conversion to the YIQ color model
        # This function modifies the image's pixel values according to the flags provided
        RGB_TO_YIQ_inplace_c(w, h, rgb_array, include_y, include_i, include_q)

    except Exception as e:
        # Catch any exceptions that might arise during the conversion and raise a custom error
        raise RuntimeError(f"An error occurred during the YIQ conversion: {e}")




@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef inline void RGB_TO_YIQ_inplace_c(
    const Py_ssize_t w,  # The width of the image (in pixels)
    const Py_ssize_t h,  # The height of the image (in pixels)
    unsigned char [:, :, :] rgb_array,  # The 3D array representing the RGB color values of the image
    bint luma = True,  # Flag indicating whether to include the Y (luma) component in the conversion
    bint in_phase = False,  # Flag indicating whether to include the I (orange-blue contrast) component
    bint quadrature = False   # Flag indicating whether to include the Q (purple-green contrast) component
) nogil:
    """
    Converts a 24-bit or 32-bit RGB image into the YIQ color model in-place.

    The YIQ color model separates brightness (luma) and color information (chroma). 
    - Y (luma) represents the brightness of the image and is used in black-and-white television receivers.
    - I and Q (chrominance) represent color information, with I corresponding to orange-blue contrast 
      and Q corresponding to purple-green contrast.

    The function modifies the pixel data directly in the provided `rgb_array` in place.

    Example:
        RGB_TO_YIQ_inplace_c(width, height, rgb_array)

    :param w: 
        The width (number of pixels) of the image.

    :param h: 
        The height (number of pixels) of the image.

    :param rgb_array: 
        A 3D numpy.ndarray (or memoryview) with shape (w, h, 3) and dtype uint8.
        This array holds the RGB values for each pixel. Changes to this array will modify the image 
        or surface that it references directly. It may be in RGB format or any other compatible pixel format.

    :param luma: 
        A boolean flag (default: `True`). If `True`, the Y (luma) component will be included in the conversion.

    :param in_phase: 
        A boolean flag (default: `False`). If `True`, the I (orange-blue contrast) component will be included.

    :param quadrature: 
        A boolean flag (default: `False`). If `True`, the Q (purple-green contrast) component will be included.

    :return: 
        None. The function modifies the image data in place.
    """

    cdef:
        int i = 0, j = 0  # Loop counters for iterating through the image pixels
        yiq yiq_  # Variable to hold the YIQ color model values for the current pixel
        rgb rgb_  # Variable to hold the converted RGB values for the current pixel
        float h_  # Unused here, could be for hue in HSL/HSV if applicable
        unsigned char *r  # Pointer to the red component of the current pixel
        unsigned char *g  # Pointer to the green component of the current pixel
        unsigned char *b  # Pointer to the blue component of the current pixel

    with nogil:  # Ensure the operation runs in parallel without holding the Python Global Interpreter Lock (GIL)
        # Iterate over the height of the image using parallel processing (prange)
        for j in prange(h, schedule=SCHEDULE, num_threads=THREADS):
            # Iterate over the width of the image
            for i in range(w):
                # Get pointers to the RGB components of the current pixel
                r = &rgb_array[i, j, 0]
                g = &rgb_array[i, j, 1]
                b = &rgb_array[i, j, 2]

                # Convert the current RGB values to YIQ
                # Scale each RGB component to the range [0, 1] by multiplying with ONE_255
                yiq_ = rgb_to_yiq(r[0] * <float>ONE_255, g[0] * <float>ONE_255, b[0] * <float>ONE_255)

                # Convert YIQ back to RGB, conditionally including each component based on the flags
                rgb_ = yiq_to_rgb(
                    yiq_.y if luma else <float>0.0,  # Include Y if luma is True, otherwise set to 0
                    yiq_.i if in_phase else <float>0.0,  # Include I if in_phase is True, otherwise set to 0
                    yiq_.q if quadrature else <float>0.0   # Include Q if quadrature is True, otherwise set to 0
                )

                # Store the converted RGB values back into the array, scaling the result to [0, 255]
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
cpdef inline (float, float, float) rgb_pixel_to_hsl(
        const unsigned char r,  # The red component of the pixel (uint8)
        const unsigned char g,  # The green component of the pixel (uint8)
        const unsigned char b  # The blue component of the pixel (uint8)
) nogil:
    """
    Converts RGB color components (uint8) to the HSL color model (float tuple).

    This function takes RGB color components in the range [0, 255] and converts them to the 
    equivalent HSL (Hue, Saturation, Lightness) model. The HSL values are rescaled to:
    - Hue (H) in the range [0, 360] degrees,
    - Saturation (S) in the range [0, 100],
    - Lightness (L) in the range [0, 100].

    The conversion process follows the standard formula for RGB to HSL conversion.

    Example:
        r, g, b = 255, 0, 0
        h, s, l = rgb_pixel_to_hsl(r, g, b)  # Output will be (0.0, 100.0, 50.0) for pure red

    :param r: 
        The red component of the pixel, an integer (uint8) in the range [0, 255].

    :param g: 
        The green component of the pixel, an integer (uint8) in the range [0, 255].

    :param b: 
        The blue component of the pixel, an integer (uint8) in the range [0, 255].

    :return: 
        A tuple of three float values representing the HSL components:
        - `h` (Hue): in the range [0, 360] degrees.
        - `s` (Saturation): in the range [0, 100].
        - `l` (Lightness): in the range [0, 100].
    """

    cdef:
        hsl hsl_  # A structure to hold the HSL values

    # Convert RGB components (in the range [0, 255]) to the range [0.0, 1.0]
    hsl_ = struct_rgb_to_hsl(
        r * <float> ONE_255,
        g * <float> ONE_255,
        b * <float> ONE_255
    )  # struct_rgb_to_hsl returns HSL values between 0.0 and 1.0

    # Rescale HSL values to the desired ranges
    hsl_.h *= <float> 360.0  # Hue rescaled to [0, 360]
    hsl_.s *= <float> 100.0  # Saturation rescaled to [0, 100]
    hsl_.l *= <float> 100.0  # Lightness rescaled to [0, 100]

    # Return the rescaled HSL values
    return hsl_.h, hsl_.s, hsl_.l



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef inline (unsigned char, unsigned char, unsigned char) hsl_to_rgb_pixel(
        const float h,  # Normalized hue value (h/360.0)
        const float s,  # Normalized saturation value (s/100.0)
        const float l  # Normalized lightness value (l/100.0)
) nogil:
    """
    Converts HSL (Hue, Saturation, Lightness) values to RGB pixel components.

    The input HSL values are expected to be normalized as follows:
    - Hue (h) should be in the range [0, 1] (i.e., h/360.0).
    - Saturation (s) and Lightness (l) should be in the range [0, 1] (i.e., s/100.0 and l/100.0).

    The function converts the normalized HSL values to RGB, rounding the output to the nearest integer 
    and ensuring that the resulting RGB values fall within the range [0, 255], 
    which is typical for pixel color values.

    Example:
        h, s, l = rgb_pixel_to_hsl(255, 128, 64)
        r, g, b = hsl_to_rgb_pixel(h/360.0, s/100.0, l/100.0)
        # Returns RGB values for the equivalent color

    :param h: 
        float; Normalized hue value in the range [0.0, 1.0], 
        where 0.0 corresponds to 0° and 1.0 corresponds to 360°.

    :param s: 
        float; Normalized saturation value in the range [0.0, 1.0], 
        where 0.0 represents no saturation and 1.0 represents full saturation.

    :param l: 
        float; Normalized lightness value in the range [0.0, 1.0],
         where 0.0 represents black, 1.0 represents white, and 0.5 represents the pure color.

    :return: 
        tuple of three unsigned char values (r, g, b) representing 
        the RGB pixel color equivalent to the input HSL values.
        The RGB values are rounded and fall within the range [0, 255].

    :raises ValueError:
        If any of the input values (h, s, or l) are outside the valid normalized range [0.0, 1.0].
        
    """

    # Input validation checks
    if not (0.0 <= h <= 1.0):
        raise ValueError(f"Hue value {h} is out of range. It should be in [0.0, 1.0].")
    if not (0.0 <= s <= 1.0):
        raise ValueError(f"Saturation value {s} is out of range. It should be in [0.0, 1.0].")
    if not (0.0 <= l <= 1.0):
        raise ValueError(f"Lightness value {l} is out of range. It should be in [0.0, 1.0].")

    cdef:
        rgb rgb_  # Structure to hold the RGB components after conversion

    # Convert the input normalized HSL values to RGB values (in the range [0.0, 1.0])
    rgb_ = struct_hsl_to_rgb(
        h, s, l
    )  # struct_hsl_to_rgb returns RGB values as floating-point numbers in the range [0.0, 1.0]

    # Rescale the RGB values to the [0, 255] range
    rgb_.r *= <float> 255.0
    rgb_.g *= <float> 255.0
    rgb_.b *= <float> 255.0

    # Round each RGB component and return as unsigned char values in the range [0, 255]
    return <unsigned char> round_c(rgb_.r), \
           <unsigned char> round_c(rgb_.g), \
           <unsigned char> round_c(rgb_.b)


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef inline (float, float, float) rgb_pixel_to_hsv(
        const unsigned char r,  # Red component of the pixel (uint8)
        const unsigned char g,  # Green component of the pixel (uint8)
        const unsigned char b  # Blue component of the pixel (uint8)
) nogil:

    """
    Converts RGB pixel color components to the equivalent HSV model.

    The input RGB values are expected to be in the range [0, 255].
    The function converts the RGB values into the HSV (Hue, Saturation, Value) model, 
    and the output HSV values are rescaled as follows:
    - Hue (H) is scaled to [0, 360] degrees.
    - Saturation (S) is scaled to [0, 100].
    - Value (V) is scaled to [0, 100].

    Example:
        r, g, b = 255, 0, 0
        h, s, v = rgb_pixel_to_hsv(r, g, b)  # Output will be (0.0, 100.0, 100.0) for pure red.

    :param r: 
        uint8; The red component of the pixel, in the range [0, 255].

    :param g: 
        uint8; The green component of the pixel, in the range [0, 255].

    :param b: 
        uint8; The blue component of the pixel, in the range [0, 255].

    :return: 
        tuple of float values representing the HSV equivalent of the given RGB components.
        - `h` (Hue) is in the range [0, 360].
        - `s` (Saturation) is in the range [0, 100].
        - `v` (Value) is in the range [0, 100].
    """

    cdef:
        hsv hsv_  # Structure to hold the HSV components after conversion

    # Normalize RGB values by dividing by 255.0 to get values in the range [0.0, 1.0]
    hsv_ = struct_rgb_to_hsv(
        r * <float> ONE_255,
        g * <float> ONE_255,
        b * <float> ONE_255
    )  # struct_rgb_to_hsv returns HSV values in the range [0.0, 1.0]

    # Rescale HSV values to the desired ranges
    hsv_.h *= <float> 360.0  # Hue scaled to [0, 360]
    hsv_.s *= <float> 100.0  # Saturation scaled to [0, 100]
    hsv_.v *= <float> 100.0  # Value scaled to [0, 100]

    # Return the HSV values
    return hsv_.h, hsv_.s, hsv_.v



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef inline (unsigned char, unsigned char, unsigned char) hsv_to_rgb_pixel(
        const float h,  # Normalized Hue (h/360.0)
        const float s,  # Normalized Saturation (s/100.0)
        const float v  # Normalized Value (v/100.0)
) nogil:
    """
    Converts HSV (Hue, Saturation, Value) values to RGB pixel components (uint8).

    The input HSV values are expected to be normalized as follows:
    - Hue (h) should be in the range [0, 1] (i.e., h/360.0).
    - Saturation (s) and Value (v) should be in the range [0, 1] (i.e., s/100.0 and v/100.0).

    The function converts the normalized HSV values to RGB, rounding the output RGB values to the nearest integer,
    and the resulting RGB values are in the range [0, 255], which is typical for pixel color values.

    Example:
        h, s, v = 0.0, 1.0, 1.0  # Pure red in HSV
        r, g, b = hsv_to_rgb_pixel(h, s, v)  # Returns (255, 0, 0) for pure red.

    :param h: 
        float; Normalized hue value in the range [0.0, 1.0], 
        where 0.0 corresponds to 0° and 1.0 corresponds to 360°.

    :param s: 
        float; Normalized saturation value in the range [0.0, 1.0], 
        where 0.0 represents no saturation and 1.0 represents full saturation.

    :param v: 
        float; Normalized value (brightness) in the range [0.0, 1.0],
         where 0.0 represents black and 1.0 represents full brightness.

    :return: 
        tuple of three unsigned char values (r, g, b) representing the equivalent RGB pixel color.
        The RGB values are rounded to the nearest integer and fall within the range [0, 255].

    :raises ValueError:
        If any of the input values (h, s, or v) are outside the valid normalized range [0.0, 1.0].
    """

    # Input validation checks
    if not (0.0 <= h <= 1.0):
        raise ValueError(f"Hue value {h} is out of range. It should be in [0.0, 1.0].")
    if not (0.0 <= s <= 1.0):
        raise ValueError(f"Saturation value {s} is out of range. It should be in [0.0, 1.0].")
    if not (0.0 <= v <= 1.0):
        raise ValueError(f"Value {v} is out of range. It should be in [0.0, 1.0].")

    cdef:
        rgb rgb_

    # Convert the input normalized HSV values to RGB values (in the range [0.0, 1.0])
    rgb_ = struct_hsv_to_rgb(
        h, s, v
    )  # struct_hsv_to_rgb returns RGB values as floating-point numbers in the range [0.0, 1.0]

    # Rescale the RGB values to the [0, 255] range
    rgb_.r *= <float> 255.0
    rgb_.g *= <float> 255.0
    rgb_.b *= <float> 255.0

    # Round each RGB component and return as unsigned char values in the range [0, 255]
    return <unsigned char> round_c(rgb_.r), \
           <unsigned char> round_c(rgb_.g), \
           <unsigned char> round_c(rgb_.b)


