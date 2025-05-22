RGBConvert
========================================

:mod:`RGBConvert.pyx`

=====================

.. currentmodule:: RGBConvert


1. Color Space Conversion Library
---------------------------------

This Python library provides efficient color space conversion functions,
primarily implemented using Cython for optimized performance. The library
enables conversions between RGB and other color models like **YIQ**, **HSL**,
and **HSV**, which are essential for various image processing and computer vision tasks.

2. Purpose of the Library
-------------------------

The primary goal of this library is to offer fast and efficient color space conversions.
These conversions are commonly used in applications such as image processing, computer vision,
color correction, and visual effects. The library leverages Cython to speed up computations
and minimize memory overhead.

3. Key Functions and Their Roles
--------------------------------

**YIQ ↔ RGB Conversions**

The YIQ color model separates luma (brightness) from chroma (color),
making it particularly useful for broadcast and image processing.

- **`yiq_2_rgb(y, i, q)` → (r, g, b)**:
  Converts a pixel from YIQ (luma, in-phase, quadrature) to RGB color space.

- **`rgb_2_yiq(r, g, b)` → (y, i, q)**:
  Converts a pixel from RGB to YIQ color space.

- **`RGB_TO_YIQ_inplace(image_surface, include_y, include_i, include_q)`**:
  Converts an entire image from RGB to YIQ in-place, modifying the given image surface.

- **`RGB_TO_YIQ_inplace_c(w, h, rgb_array, luma, in_phase, quadrature)`**:
  A Cython-optimized, low-level function for in-place RGB-to-YIQ conversion, minimizing Python overhead.

**✅ Why YIQ?**

YIQ is widely used in **NTSC television broadcasting** and in image processing applications where the separation of luma (brightness) and chroma (color) enhances processing and compression.

**HSL ↔ RGB Conversions**

HSL stands for **Hue**, **Saturation**, and **Lightness**. It's commonly used for color adjustments, such as adjusting brightness or saturation.

- **`hsl_to_rgb_pixel(h, s, l)` → (r, g, b)**:
  Converts a pixel from HSL (Hue, Saturation, Lightness) to RGB color space.

- **`rgb_pixel_to_hsl(r, g, b)` → (h, s, l)**:
  Converts a pixel from RGB to HSL color space.

**✅ Why HSL?**

HSL is extremely useful for **color adjustment** in graphics applications. By working with the hue, saturation, and lightness, users can easily create effects like tints, shades, and saturation adjustments.

**HSV ↔ RGB Conversions**

HSV stands for **Hue**, **Saturation**, and **Value**, and it is widely used in color selection tools and image segmentation.

- **`hsv_to_rgb_pixel(h, s, v)` → (r, g, b)**:
  Converts a pixel from HSV (Hue, Saturation, Value) to RGB color space.

- **`rgb_pixel_to_hsv(r, g, b)` → (h, s, v)**:
  Converts a pixel from RGB to HSV color space.

**✅ Why HSV?**

HSV is frequently used in **color selection tools** and **image segmentation**, as it separates chromatic content from brightness, making it easier to manipulate colors based on their hue, saturation, or value.

4. Optimization Features
------------------------

This library is designed with performance in mind. Here are the key optimization features:

- **Cython-based optimization** (`cdef`, `cpdef`, `nogil`):
  Critical functions are compiled into C using Cython, significantly improving speed.

- **In-place operations** (`RGB_TO_YIQ_inplace`):
  By modifying image arrays directly, the library reduces memory overhead and avoids unnecessary copies.

- **No GIL** (`nogil`):
  The Cython implementation enables multi-threading, allowing for parallel execution and further performance improvements.

5. Use Cases
------------

This library is highly useful for various fields where color manipulation is essential. Some key use cases include:

- **Image Processing**:
  Transform images into different color spaces for tasks such as filtering, thresholding, or general analysis.

- **Computer Vision**:
  Utilize HSV or YIQ for **color-based object detection**, image segmentation, or feature extraction.

- **Graphics Applications**:
  Adjust color properties like hue, saturation, lightness, and brightness to create visual effects and improve contrast.

- **Broadcasting & Video Processing**:
  Convert between RGB and YIQ color spaces for **NTSC signals** in television broadcasting or video processing applications.

6. Conclusion
-------------

This library provides fast, efficient, and easy-to-use color space conversions, optimized for performance through Cython. Its versatility makes it invaluable for image processing, computer vision, graphics applications, and more.

|

7. Cython list methods
----------------------

.. py:function:: rgb_2_yiq(r, g, b)

    Summary:
    Converts RGB color values into the YIQ color model. The YIQ model was historically used in NTSC television broadcasting and separates the brightness (Y) from the chrominance components (I and Q). This function performs the conversion using normalized floating-point calculations.

    Example Usage:

    .. code-block:: python

        # Convert RGB values to YIQ
        y, i, q = rgb_2_yiq(255, 0, 0)  # Red color

    Parameters:

    - **r** (unsigned char):
      An integer representing the red channel, in the range [0, 255].

    - **g** (unsigned char):
      An integer representing the green channel, in the range [0, 255].

    - **b** (unsigned char):
      An integer representing the blue channel, in the range [0, 255].

    Returns:

    - **tuple**:

    A tuple `(Y, I, Q)` where:
    - **Y** (float): The Luma component, representing brightness.
    - **I** (float): The In-phase chrominance, representing the orange-blue hue.
    - **Q** (float): The Quadrature chrominance, representing the purple-green hue.


    Raises:

    - **ValueError**:
      If any input value is outside the valid range [0, 255].

    Notes:

    - The YIQ color model is used primarily in NTSC broadcasting and separates color information (chrominance) from brightness (luma).
    - The conversion is done using normalized floating-point calculations for accurate representation in the YIQ color space.


.. py:function:: yiq_2_rgb(y, i, q)

    **Summary**:
    Converts YIQ color model values into equivalent RGB values. The YIQ color model was historically used in NTSC television broadcasting and separates the brightness (Y) from the chrominance components (I and Q). This function converts the YIQ values back into RGB format, with the resulting values scaled to the [0, 255] range and rounded to the nearest integer.

    **Example Usage**:

    .. code-block:: python

        # Convert YIQ values to RGB
        r, g, b = yiq_2_rgb(0.5, 0.2, 0.1)

    **Parameters**:

    - **y** (float):
      The Luma (brightness) component of the YIQ color model.

    - **i** (float):
      The In-phase chrominance, representing the orange-blue contrast.

    - **q** (float):
      The Quadrature chrominance, representing the purple-green contrast.

    **Returns**:

    - **tuple**:

    A tuple `(R, G, B)` where:
        - **R** (unsigned char): Red value in the range [0, 255].
        - **G** (unsigned char): Green value in the range [0, 255].
        - **B** (unsigned char): Blue value in the range [0, 255].


    **Raises**:

    - **ValueError**:
      If the input values for `y`, `i`, or `q` are outside the expected range for the YIQ model.

    **Notes**:

    - The YIQ color model separates brightness (Y) from chrominance (I and Q), which allows black-and-white TVs to display just the Y component.
    - The function scales and rounds the RGB values to fit within the [0, 255] range, making them suitable for display or further processing.

|

.. py:function:: RGB_TO_YIQ_inplace(image_surface, include_y=True, include_i=False, include_q=False)


    **Summary**:
    Converts a 24-bit or 32-bit RGB image into the YIQ color model in-place. The YIQ model separates brightness (luma) from color information (chroma), where Y represents brightness (luma), and I and Q represent chrominance (orange-blue and purple-green contrasts). This function modifies the pixel data of the given `image_surface` directly without creating a new image.

    **Example Usage**:

    .. code-block:: python

        # Convert an image surface to YIQ with the default parameters
        RGB_TO_YIQ_inplace(image)

    **Parameters**:

    - **image_surface** (pygame.Surface):
      The input image surface (24-bit or 32-bit), which contains the RGB pixel data to be converted to YIQ.

    - **include_y** (bint, default=True):
      A flag indicating whether to include the Y (luma) component in the conversion. If `True`, the Y component will be included; otherwise, it will be omitted.

    - **include_i** (bint, default=False):
      A flag indicating whether to include the I (orange-blue contrast) component in the conversion. If `True`, the I component will be included.

    - **include_q** (bint, default=False):
      A flag indicating whether to include the Q (purple-green contrast) component in the conversion. If `True`, the Q component will be included.

    **Returns**:

    - **None**:
      This function modifies the `image_surface` in-place and does not return any value.

    **Raises**:

    - **TypeError**:
      If `image_surface` is not a valid `pygame.Surface` object.

    - **ValueError**:
      If the input image surface is not compatible with the conversion (e.g., if the surface does not support 24-bit or 32-bit color formats).

    **Notes**:

    - The function processes the pixel data in-place, meaning the original image surface is directly modified.
    - You can choose to include any combination of the Y, I, and Q components based on the flags `include_y`, `include_i`, and `include_q`.
    - The conversion will be done to all pixels in the surface, and the color channels will be replaced by their respective YIQ values.

|

**Function**:

.. py:function:: rgb_pixel_to_hsl(r, g, b)

**Summary**:
Converts RGB color components (uint8) to the HSL color model (float tuple). This function converts RGB values in the range [0, 255] to the equivalent HSL (Hue, Saturation, Lightness) model, rescaling the values as follows:
- Hue (H) in the range [0, 360] degrees.
- Saturation (S) in the range [0, 100].
- Lightness (L) in the range [0, 100].

**Example Usage**:

.. code-block:: python

    r, g, b = 255, 0, 0
    h, s, l = rgb_pixel_to_hsl(r, g, b)  # Output will be (0.0, 100.0, 50.0) for pure red

**Parameters**:

- **r** (unsigned char):
    The red component of the pixel, an integer (uint8) in the range [0, 255].

- **g** (unsigned char):
    The green component of the pixel, an integer (uint8) in the range [0, 255].

- **b** (unsigned char):
    The blue component of the pixel, an integer (uint8) in the range [0, 255].

**Returns**:
A tuple of three float values representing the HSL components:
- **h** (float): Hue, in the range [0, 360] degrees.
- **s** (float): Saturation, in the range [0, 100].
- **l** (float): Lightness, in the range [0, 100].

|

**Function**:

.. py:function:: hsl_to_rgb_pixel(h, s, l)

**Summary**:
Converts HSL (Hue, Saturation, Lightness) values to RGB pixel components. The input HSL values should be normalized as follows:
- Hue (h) should be in the range [0, 1] (i.e., h/360.0).
- Saturation (s) and Lightness (l) should be in the range [0, 1] (i.e., s/100.0 and l/100.0).

The function converts the normalized HSL values to RGB, rounding the output to the nearest integer and ensuring that the resulting RGB values fall within the range [0, 255].

**Example Usage**:

.. code-block:: python

    h, s, l = rgb_pixel_to_hsl(255, 128, 64)
    r, g, b = hsl_to_rgb_pixel(h/360.0, s/100.0, l/100.0)

**Parameters**:

- **h** (float):
    Normalized hue value in the range [0.0, 1.0], where 0.0 corresponds to 0° and 1.0 corresponds to 360°.

- **s** (float):
    Normalized saturation value in the range [0.0, 1.0], where 0.0 represents no saturation and 1.0 represents full saturation.

- **l** (float):
    Normalized lightness value in the range [0.0, 1.0], where 0.0 represents black, 1.0 represents white, and 0.5 represents the pure color.

**Returns**:
A tuple of three unsigned char values (r, g, b) representing the RGB pixel color equivalent to the input HSL values. The RGB values are rounded and fall within the range [0, 255].

**Raises**:
- **ValueError**: If any of the input values (h, s, or l) are outside the valid normalized range [0.0, 1.0].


|

**Function**:

.. py:function:: rgb_pixel_to_hsv(r, g, b)

**Summary**:
Converts RGB pixel color components to the equivalent HSV model. The input RGB values are expected to be in the range [0, 255]. The function converts the RGB values into the HSV (Hue, Saturation, Value) model, and the output HSV values are rescaled as follows:
- Hue (H) is scaled to [0, 360] degrees.
- Saturation (S) is scaled to [0, 100].
- Value (V) is scaled to [0, 100].

**Example Usage**:

.. code-block:: python

    r, g, b = 255, 0, 0
    h, s, v = rgb_pixel_to_hsv(r, g, b)

**Parameters**:

- **r** (uint8):
    The red component of the pixel, in the range [0, 255].

- **g** (uint8):
    The green component of the pixel, in the range [0, 255].

- **b** (uint8):
    The blue component of the pixel, in the range [0, 255].

**Returns**:
A tuple of float values representing the HSV equivalent of the given RGB components:
- **h** (Hue): in the range [0, 360].
- **s** (Saturation): in the range [0, 100].
- **v** (Value): in the range [0, 100].

**Raises**:
- **ValueError**: If any of the input values (r, g, or b) are outside the valid range [0, 255].

|

**Function**:

.. py:function:: hsv_to_rgb_pixel(h, s, v)

**Summary**:
Converts HSV (Hue, Saturation, Value) values to RGB pixel components (uint8). The input HSV values are expected to be normalized as follows:
- Hue (h) should be in the range [0, 1] (i.e., h/360.0).
- Saturation (s) and Value (v) should be in the range [0, 1] (i.e., s/100.0 and v/100.0).

The function converts the normalized HSV values to RGB, rounding the output RGB values to the nearest integer, and the resulting RGB values are in the range [0, 255], which is typical for pixel color values.

**Example Usage**:

.. code-block:: python

    h, s, v = 0.0, 1.0, 1.0
    r, g, b = hsv_to_rgb_pixel(h, s, v)  # Returns (255, 0, 0) for pure red.

**Parameters**:

- **h** (float):
    Normalized hue value in the range [0.0, 1.0], where 0.0 corresponds to 0° and 1.0 corresponds to 360°.

- **s** (float):
    Normalized saturation value in the range [0.0, 1.0], where 0.0 represents no saturation and 1.0 represents full saturation.

- **v** (float):
    Normalized value (brightness) in the range [0.0, 1.0], where 0.0 represents black and 1.0 represents full brightness.

**Returns**:
A tuple of three unsigned char values (r, g, b) representing the equivalent RGB pixel color. The RGB values are rounded to the nearest integer and fall within the range [0, 255].

**Raises**:
- **ValueError**: If any of the input values (h, s, or v) are outside the valid normalized range [0.0, 1.0].

|




