Palette
========================================

:mod:`Palette.pyx`

=====================

.. currentmodule:: Palette

|

1. Library Overview
-------------------

The functions in this library are designed for **image manipulation** and **palette management**,
making it suitable for applications such as **graphical rendering**, **image processing**,
and **creating custom color palettes**. The library is ideal for use in **graphics software**
or **visual applications** where images need to be rendered with specific color palettes or where
images must be dynamically modified based on palette data.

2. Key Concepts
---------------

- **Image Surface**:
  Represents an object corresponding to an image, where individual pixels can be accessed and manipulated. This object provides an abstraction for working with pixel data in various image processing tasks.

- **Palette**:
  A collection of colors that can be applied to an image. Palettes are commonly used in **visual design**, **digital art**, and **simulations** to ensure consistency in color usage throughout an image or graphic.

- **Temporary Buffer**:
  A storage area used to hold intermediate results during computations. The temporary buffer helps reduce the need to perform expensive operations multiple times, improving performance when working with complex image manipulations.

3. Purpose of the Library
-------------------------

The primary purpose of this library is to provide efficient tools for manipulating images at the **color level**. It allows users to modify or generate custom **color palettes**, providing greater flexibility in image rendering and processing tasks. By optimizing image handling, the library supports a variety of graphical applications, including dynamic image modifications based on color palettes.

4. Use Cases
------------

- **Graphics Software**:
  The library can be used in applications that need to render images with specific palettes, such as image editors, animation tools, or graphic design applications.

- **Image Processing**:
  Modify or analyze images by dynamically adjusting their color palettes, which can be useful in tasks such as color correction or filtering.

- **Digital Art**:
  Artists can use the library to create and manipulate custom palettes, allowing for unique, controlled color schemes in their digital artwork.

- **Simulations**:
  In simulations, this library can be used to generate or modify color palettes for visual effects or data visualization purposes, ensuring that simulated environments maintain consistent and visually appealing color schemes.

5. Conclusion
-------------

This library offers powerful tools for managing and manipulating images at the color level,
making it invaluable for anyone working with custom color palettes in graphical and image processing
applications.


6. Cython list methods
----------------------

|

.. py:function:: palette_change(image_surface, color_palette, temp_buffer)

    |

    **Summary:**
    Applies a color palette transformation to an image in-place. The function modifies the colors of the input image (a Pygame surface) using the specified color palette. The transformation is performed directly on the surface without creating a new image, optimizing memory usage.

    **Example Usage:**

    .. code-block:: python

        # Apply a color palette from IRIDESCENTCRYSTAL to the surface
        palette_change(surface, IRIDESCENTCRYSTAL, temp_buffer)

    **Parameters:**

    - **image_surface** (pygame.Surface):
      The Pygame surface representing the image to be modified. This is the input image on which the color palette will be applied.

    - **color_palette** (numpy.ndarray):
      A NumPy array containing the color palette to apply. The array should have a shape of (w, 3), where `w` is the number of colors, and the dtype must be `float32` with values in the range [0.0, 255.0].

      **Example:**

      .. code-block:: python

          from PygameShader import IRIDESCENTCRYSTAL

    - **temp_buffer** (numpy.ndarray):
      A temporary NumPy array used for optimization. It should be a contiguous array (C_CONTIGUOUS) and have the shape `(image_surface.get_width() * image_surface.get_height(), len(color_palette))`. This array helps to avoid memory reallocation during the function call, improving performance.

      Example of temporary buffer declaration:

      .. code-block:: python

          temp_buffer = numpy.ascontiguousarray(numpy.ndarray(
              (SURFACE.get_width() * SURFACE.get_height(), IRIDESCENTCRYSTAL.shape[0]),
              dtype=float32
          ))

    **Returns:**

    - **None**:
      This function modifies the `image_surface` in-place without returning a new object.

    **Raises:**

    - **ValueError**:
      If the input surface is not compatible or if an invalid palette is provided.

    - **TypeError**:
      If any of the input arguments are of an incorrect type (e.g., the surface is not a `pygame.Surface`, the palette is not a `numpy.ndarray`, or the buffer is not a contiguous array).

    - **ValueError**:
      If there is a mismatch in the dimensions of the `image_surface` and the `temp_buffer`.

    **Notes:**

    - This function relies on the efficient handling of large arrays for color transformation. Make sure to pre-allocate the `temp_buffer` before calling the function to avoid unnecessary memory allocation during each frame.

|

.. function:: make_palette(width: int, fh: float, fs: float, fl: float)

    |

    **Summary:**
    Generates a palette of RGB colors by mapping HSL values. The function scales hue, clips saturation,
    and adjusts lightness based on the given parameters, returning a NumPy array of RGB values.

    **Example Usage:**

    .. code-block:: python

        # Create a palette of 256 colors where:
        # - Hue is scaled by a factor of 6
        # - Saturation is fixed at 255
        # - Lightness is scaled by a factor of 2
        palette = make_palette(256, 6, 255, 2)

        # Another palette with different settings:
        palette = make_palette(256, 4, 255, 2)

    **Parameters:**

    - **width** (*int*): The number of colors (palette size) to generate.
    - **fh** (*float*): A factor by which to scale the hue value for each color.
    - **fs** (*float*): The saturation value, which must be in the range (0.0 ... 255.0).
      This limits the saturation intensity.
    - **fl** (*float*): A factor by which to scale the lightness value for each color.

    **Returns:**

    - (*numpy.ndarray*): A 1D NumPy array of RGB color values corresponding to the generated palette.

    .. note::
        The function relies on converting from HSL to RGB, applying the specified factors and clipping values.
        The output is a NumPy array, which can be directly used for visualization or further processing.

|

.. py:function:: create_surface_from_palette(palette_c)

    |

    **Summary:**
    Converts a 1D array of RGB palette colors into a Pygame Surface (line). The function processes a 1D array of 32-bit unsigned integers (representing colors in ARGB format) and converts it into a Pygame `Surface` object. The result is a surface that can be used for rendering or visualization.

    **Example Usage:**

    .. code-block:: python

        # Assuming 'palette' is a palette generated with make_palette:
        surface = create_surface_from_palette(palette)

    **Parameters:**

    - **palette_c** (numpy.ndarray or memoryview):
      A 1D array (or memoryview) containing the palette of colors. Each color should be represented as a 32-bit unsigned integer (in ARGB format). The array must have a `numpy.uint32` dtype and be contiguous in memory.

      Example:

    .. code-block:: python

      from PygameShader import make_palette

      # Create a palette of 256 colors
      palette = make_palette(256, 6, 255, 2)

    **Returns:**

    - **object**:
      A Pygame `Surface` object created from the palette.

    **Raises:**

    - **TypeError**:
      If `palette_c` is not a `numpy.ndarray` or memoryview, or if it has an incorrect dtype.

    - **ValueError**:
      If `palette_c` is empty, not contiguous, or has an invalid size.

    - **RuntimeError**:
      If an error occurs during the creation of the Pygame surface.

    **Notes:**

    - The input palette should contain RGB values packed into 32-bit integers (ARGB format).
    - The function processes the palette array in parallel for performance optimization.
    - The output is a Pygame `Surface` object, which can be directly used for visualization or rendering.


