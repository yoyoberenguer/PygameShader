Misc
========================================

:mod:`Misc.pyx`

---------------


.. currentmodule:: Misc

|

1. Python Library Summary
-------------------------

This library is part of a Python module for image manipulation and processing,
focusing on **gradient generation**, **image channel manipulation**, **color space calculations**,
and **sorting algorithms**. Below is a breakdown of the key functionality and purpose
of the different functions and methods in the module:

**Gradient Generation Functions**

These functions create various types of gradients, which are often used in graphical applications for
creating smooth transitions between colors.

- **`create_line_gradient_rgb`**: Generates a linear gradient of RGB values between two specified colors (`start_rgb` and `end_rgb`).
- **`create_line_gradient_rgba`**: Similar to the above, but for RGBA colors (adds an alpha channel for transparency).
- **`horizontal_rgb_gradient`**: Creates a horizontal gradient in RGB color space.
- **`horizontal_rgba_gradient`**: Creates a horizontal gradient in RGBA color space.
- **`create_radial_gradient`**: Creates a radial gradient that radiates outward from a point, with customizable start and end colors and offsets. A precomputed gradient can be provided to optimize performance.
- **`create_radial_gradient_alpha`**: Similar to `create_radial_gradient`, but specifically for handling RGBA with transparency.
- **`create_quarter_radial_gradient`** and **`create_quarter_radial_gradient_alpha`**: These create gradients for a quarter of a circle (sector-based gradients), useful for specialized visual effects.

**Color Space and Color Matching Functions**

These functions are used to work with different color spaces (HSV, HSL, RGB) and find closest color matches, which is useful for color-based image processing or palette generation.

- **`color_dist_hsv`**: Calculates the distance between two colors in the HSV color space.
- **`color_dist_hsl`**: Calculates the distance between two colors in the HSL color space.
- **`closest_hsv_color`** and **`closest_hsl_color`**: Find the closest color in a given palette to a target color, using the HSV or HSL color space respectively.
- **`close_color`**: Finds a color in a palette that is close to a given color (presumably using RGB space).
- **`use_palette`**: Similar to `close_color`, but selects colors from a palette based on a defined selection criterion.

**Image Manipulation Functions**

These functions focus on manipulating images or buffers, particularly for scrolling, blending, and combining channels.

- **`swap_image_channels`**: Swaps color channels (such as RGB) in an image surface, useful when working with different color formats or performing transformations.
- **`scroll_surface_24bit`**, **`scroll24_inplace`**, **`scroll_rgb_array_inplace`**, **`scroll_alpha_inplace`**: These functions allow for scrolling of an image or image channels in the x and y directions, which can be used for animations or visual effects. The `c` versions are likely optimized implementations written in C for performance.
- **`combine_rgba_buffers`**: Combines a BGR image buffer and an alpha channel into a single RGBA image buffer. This is useful for working with images that have separate color and alpha channels.
- **`normalize_2d_array`**: Normalizes a 2D array, possibly for color or pixel intensity normalization.

**Utility Functions**

These functions perform a variety of utility tasks, including sorting and random number generation.

- **`rgb_to_int`** and **`int_to_rgb`**: Convert between RGB values and their integer representation (often used for packing/unpacking colors).
- **`_randf`**, **`_randi`**: Generate random float and integer values within a specified range, likely for random color or pixel generation.
- **`bubble_sort`**, **`insertion_sort`**, **`quick_sort`**, **`heap_sort`**: These are different sorting algorithms implemented for sorting arrays of pixel or image data. `heap_sort` and `quick_sort` are especially useful for sorting large datasets or pixel values efficiently.
- **`partition_cython`**, **`_quick_sort`**, **`heapify`**: These are helper functions for sorting algorithms like Quick Sort and Heap Sort, optimized for performance in Cython.

**Image Format and Type Checking**

These functions help determine the type of image data or buffers.

- **`get_image_format`**: Likely checks the format of an image (whether it is in RGBA, RGB, etc.).
- **`is_type_memoryview`**, **`is_uint8`**, **`is_float64`**, **`is_int32`**: These functions check the type and datatype of a given image array or buffer, useful for validating input and ensuring the correct data format.

2. Overall Purpose of the Library
---------------------------------

The library is designed for **image processing** and **graphical operations**,
with a strong emphasis on **gradient creation**, **image channel manipulation**,
**color transformations**, **sorting algorithms**, and **optimization**.
It provides functions to create smooth color transitions (gradients), manipulate pixel data,
check image formats, and optimize computationally expensive operations using algorithms
like QuickSort and HeapSort. The functions are highly optimized with `cpdef` and `cdef`
to allow seamless integration between Python and C for performance-critical tasks.

3. Use Cases
------------

- **Image Processing**: Manipulating and transforming images, handling RGBA/BGR channels, scrolling image pixels, and applying color transformations.
- **Graphics Rendering**: Generating gradients (linear, radial, quarter radial) for background effects or visual transitions.
- **Color Matching**: Finding the closest matching colors in a given color palette based on various color spaces (HSV, HSL).
- **Performance Optimization**: Utilizing sorting and color manipulation algorithms to efficiently process large images or datasets, with performance improvements using Cython (`cpdef`/`cdef`).


4. Conclusion
--------------
This library would be useful in applications like **image editors**, **graphic design tools**,
**data visualization**, or **real-time graphical applications**.



5. Cython list methods
-----------------------

|

.. py:function:: swap_image_channels(image_surface, channel_order)

    |

    **Summary**:
    Swaps or nullifies the channels of an image based on the specified channel order string. The function allows you to swap the color channels (RGB) of an image represented as a pygame.Surface. You can specify the new channel order using a string where each letter represents a color channel (R, G, B) or '0' to nullify a channel (remove it).

    **Example Usage**:

    .. code-block:: python

        new_surface = swap_image_channels(image_surface, 'R0B')
        # Swaps red and blue channels and removes green

    **Parameters**:

    - **image_surface** (pygame.Surface):
        The image (pygame Surface) whose channels are to be swapped.

    - **channel_order** (str):
        A string representing the desired channel order. It must contain exactly 3 characters, where 'R', 'G', and 'B' represent the respective color channels, and '0' nullifies a channel. Example values: 'RGB', 'RBG', 'GRB', 'BGR', 'BRG', 'R0B', etc.

    **Returns**:
    A new 24-bit pygame surface with the swapped channels.

    **Raises**:
    - **ValueError**:

        If the channel order string is not valid or if the surface format is incompatible.

|

.. py:function:: create_line_gradient_rgb(num_pixels, start_rgb=(255, 0, 0), end_rgb=(0, 255, 0))

    |

    **Summary**:
    Generates a 2D horizontal gradient of RGB colors from a start color to an end color. The function creates an array of RGB values representing a smooth horizontal gradient transitioning from the specified `start_rgb` to the `end_rgb`. The gradient consists of `num_pixels` number of pixels, with the color gradually changing from the start color to the end color across the 1D array.

    **Example Usage**:

    .. code-block:: python

        gradient = create_line_gradient_rgb(256, start_rgb=(255, 0, 0), end_rgb=(0, 0, 255))
        # Generates a horizontal gradient from red to blue with 256 pixels.

    **Parameters**:

    - **num_pixels** (int):
        The number of pixels in the gradient (must be > 0). Determines the length of the gradient in the horizontal direction.

    - **start_rgb** (tuple, optional):
        A tuple representing the RGB values of the starting color (default is red: (255, 0, 0)).

    - **end_rgb** (tuple, optional):
        A tuple representing the RGB values of the ending color (default is green: (0, 255, 0)).

    **Returns**:
    - **numpy.ndarray**:

        A 2D numpy array of shape (num_pixels, 3), where each row represents an RGB color. The array contains the pixel values of the gradient, with the color transitioning from the `start_rgb` to the `end_rgb`.

    **Raises**:
    - **ValueError**:

        If the `num_pixels` is less than or equal to 0. If `start_rgb` or `end_rgb` are not valid RGB tuples of length 3, or if any of the color values are out of the valid RGB range [0, 255]. If `num_pixels` is not a positive integer.

    - **TypeError**:

        If `start_rgb` or `end_rgb` are not tuples, or if they contain non-integer values.

|


.. py:function:: create_line_gradient_rgba(num_pixels, start_rgba=(255, 0, 0, 255), end_rgba=(0, 255, 0, 0))

    |

    **Summary**:
    Generates a 2D horizontal gradient of RGB(A) colors from a start color to an end color. The function creates an array of RGBA values representing a smooth horizontal gradient transitioning from the specified `start_rgba` to the `end_rgba`. The gradient consists of `num_pixels` number of pixels, with the color gradually changing from the start color to the end color across the 1D array.

    **Example Usage**:

    .. code-block:: python

        gradient = create_line_gradient_rgba(256, start_rgba=(255, 0, 0), end_rgba=(0, 0, 255))
        # Generates a horizontal gradient from red to blue with 256 pixels.

    **Parameters**:

    - **num_pixels** (int):
        The number of pixels in the gradient (must be > 0). Determines the length of the gradient in the horizontal direction.

    - **start_rgba** (tuple, optional):
        A tuple representing the RGBA values of the starting color (default is red: (255, 0, 0, 255)).

    - **end_rgba** (tuple, optional):
        A tuple representing the RGBA values of the ending color (default is green: (0, 255, 0, 0)).

    **Returns**:
    - **numpy.ndarray**:

        A 2D numpy array of shape (num_pixels, 4), where each row represents an RGBA color. The array contains the pixel values of the gradient, with the color transitioning from the `start_rgba` to the `end_rgba`.

    **Raises**:
    - **ValueError**:

        If the `num_pixels` is less than or equal to 0. If `start_rgba` or `end_rgba` are not valid RGBA tuples of length 4, or if any of the color values are out of the valid RGBA range [0, 255]. If `num_pixels` is not a positive integer.

    - **TypeError**:

        If `start_rgba` or `end_rgba` are not tuples, or if they contain non-integer values.

|


.. py:function:: horizontal_rgb_gradient(w, h, color_start=(255, 0, 0), color_end=(0, 255, 0))

    |

    **Summary**:
    Generates a 24-bit horizontal gradient between two RGB colors. The function creates a smooth transition from `color_start` to `color_end` across a surface of the specified dimensions.

    **Example Usage**:

    .. code-block:: python

        # Create a gradient from red to green
        gradient = horizontal_rgb_gradient(500, 300, (255, 0, 0), (0, 255, 0))

    **Parameters**:

    - **w** (int):
        The width of the gradient surface in pixels (must be > 1).

    - **h** (int):
        The height of the gradient surface in pixels (must be > 0).

    - **color_start** (tuple, optional):
        The starting color as an (R, G, B) tuple, where each value is in the range [0, 255]. The default is (255, 0, 0), which represents red.

    - **color_end** (tuple, optional):
        The ending color as an (R, G, B) tuple, where each value is in the range [0, 255]. The default is (0, 255, 0), which represents green.

    **Returns**:
    - **pygame.Surface**:

        A 24-bit RGB pygame.Surface object of size (`w`, `h`) with the gradient applied.

    **Raises**:
    - **ValueError**:

        If `w` or `h` is not valid. If `color_start` or `color_end` is not a valid RGB tuple.

|

.. py:function:: horizontal_rgba_gradient(w, h, color_start=(255, 0, 0, 255), color_end=(0, 255, 0, 0))

    |

    **Summary**:
    Generates a 32-bit horizontal gradient with transparency between two RGBA colors. The function creates a smooth horizontal transition from `color_start` to `color_end` across the given width while maintaining the specified height.

    **Example Usage**:

    .. code-block:: python

        # Create a gradient from red to green
        gradient = horizontal_rgba_gradient(500, 300, (255, 0, 0, 0), (0, 255, 0, 255))

    **Parameters**:

    - **w** (int):
        The width of the gradient surface in pixels (must be > 1).

    - **h** (int):
        The height of the gradient surface in pixels (must be > 0).

    - **color_start** (tuple, optional):
        The starting color in RGBA format, where each component (R, G, B, A) is in the range 0-255.
        Default: (255, 0, 0, 255) (opaque red).

    - **color_end** (tuple, optional):
        The ending color in RGBA format, where each component (R, G, B, A) is in the range 0-255.
        Default: (0, 255, 0, 0) (transparent green).

    **Returns**:
    - **pygame.Surface**:

        A 32-bit per pixel `pygame.Surface` with dimensions (w, h), containing the generated gradient with an alpha channel.

    **Raises**:
    - **ValueError**:

        - If `w` or `h` is not a valid positive integer.
        - If `color_start` or `color_end` is not a tuple of four integers (0-255).

|

.. py:function:: create_radial_gradient(w, h, offset_x=0.5, offset_y=0.5, color_start=(255, 0, 0), color_end=(0, 0, 0), precomputed_gradient=None, scale_factor=r_max, threads=8)

    |

    **Summary**:
    Generates a radial gradient (24-bit RGB) with a smooth transition between two colors, centered within a rectangular surface of given width and height. The gradient radiates from the center of the surface, and the transition between colors is based on the distance from the center, allowing for various gradient effects (such as circular or radial). Optionally, a precomputed gradient can be used, or one can be generated dynamically.

    **Example Usage**:

    .. code-block:: python

        precomputed_gradient = create_line_gradient_rgb(
            math.sqrt(800 ** 2 + 800 ** 2),
            start_rgb = (255, 0, 0),
            end_rgb = (0, 0, 0))

        surface = create_radial_gradient(800, 800,
            offset_x = 0.5,
            offset_y = 0.5,
            color_start = (255, 0, 0),
            color_end = (0, 0, 0),
            precomputed_gradient = precomputed_gradient
        )

    **Parameters**:

    - **w** (int):
        The width of the output gradient surface in pixels (must be > 0).

    - **h** (int):
        The height of the output gradient surface in pixels (must be > 0).

    - **offset_x** (float, optional):
        The horizontal offset of the gradient center, where 0.5 represents the center of the surface (default is 0.5).

    - **offset_y** (float, optional):
        The vertical offset of the gradient center, where 0.5 represents the center of the surface (default is 0.5).

    - **color_start** (tuple, optional):
        The starting color of the gradient in RGB format (0-255). Default is (255, 0, 0), which corresponds to opaque red.

    - **color_end** (tuple, optional):
        The ending color of the gradient in RGB format (0-255). Default is (0, 0, 0), which corresponds to black.

    - **precomputed_gradient** (numpy.array, optional):
        A precomputed gradient 2D array (shape: (n, 3), containing RGB values). If not provided, a new gradient will be computed.

    - **scale_factor** (float, optional):
        A scaling factor that adjusts the radius of the gradient. A value greater than 1 will increase the size of the gradient, and values smaller than 1 will reduce its radius. Default is 1.4. Must be > 0.

    - **threads** (int, optional):
        The number of concurrent threads to use for gradient computation. Default is 8.

    **Returns**:
    - **pygame.Surface**:

        A `pygame.Surface` object with the generated radial gradient, centered at (w/2, h/2), in 24-bit RGB format.

    **Raises**:
    - **ValueError**:

        - If `color_start` or `color_end` is not a tuple of 3 integers.
        - If any RGB value in `color_start` or `color_end` is out of range (0-255).
        - If `scale_factor` is None or <= 0.
        - If `w` or `h` is <= 1.

    - **TypeError**:

        - If `precomputed_gradient` is not a contiguous array.

    - **ValueError**:

        - If `precomputed_gradient` is not a 2D array, not of type uint8, has length <= 1, or is not in RGB format.

    **Credits**:
    Function improved with collaboration from ChatGPT.

|

.. py:function:: create_radial_gradient_alpha(w, h, offset_x=0.5, offset_y=0.5, color_start=(255, 0, 0, 255), color_end=(0, 0, 0, 0), precomputed_gradient=None, scale_factor=r_max, threads=8)

    |

    **Summary**:
    Creates a 32-bit radial gradient with transparency, blending smoothly between two RGBA colors (`color_start` and `color_end`), and supports optional precomputed gradient arrays for performance optimization.

    **Parameters**:

    - **w** (int):
      The width of the surface in pixels (must be > 1).

    - **h** (int):
      The height of the surface in pixels (must be > 1).

    - **offset_x** (float, optional):
      The X-coordinate of the gradient center as a fraction of the width (default is 0.5).

    - **offset_y** (float, optional):
      The Y-coordinate of the gradient center as a fraction of the height (default is 0.5).

    - **color_start** (tuple, optional):
      The starting RGBA color of the gradient (default is (255, 0, 0, 255)).

    - **color_end** (tuple, optional):
      The ending RGBA color of the gradient (default is (0, 0, 0, 0)).

    - **precomputed_gradient** (numpy.ndarray or None, optional):
      A precomputed gradient array of shape (w, 4) in uint8 format for optimization (default is None).

    - **scale_factor** (float, optional):
      The scaling factor for the gradient (must be > 0, default is `r_max`).

    - **threads** (unsigned short int, optional):
      The number of threads to use for parallel computation (default is 8).

    **Returns**:
    - **pygame.Surface**: A surface containing the generated radial gradient with per-pixel transparency.

    **Raises**:
    - **ValueError**: If `color_start` or `color_end` is not a tuple of four integers, if any RGBA value is out of range (0-255), or if `scale_factor` is None or <= 0.
    - **ValueError**: If `w` or `h` is <= 1.
    - **TypeError**: If `precomputed_gradient` is not a contiguous array.
    - **ValueError**: If `precomputed_gradient` is not a 2D array, not of type uint8, has length <= 1, or is not in RGBA format.

    **Example Usage**:

    .. code-block:: python

        precomputed_gradient = create_line_gradient_rgba(
            int(math.sqrt(800 ** 2 + 800 ** 2)),
            start_rgba=(255, 0, 0, 255),
            end_rgba=(0, 0, 0, 0)
        )

        surface = create_radial_gradient_alpha(
            800, 800,
            precomputed_gradient=precomputed_gradient
        ).convert_alpha()


|

.. py:function:: create_quarter_radial_gradient(width_, height_, start_color_=(255, 0, 0), end_color_=(0, 0, 0), gradient_array_=None, factor_=1.4, threads_=8)

    |

    **Summary**:
    Generates a quarter radial gradient (24-bit opaque).
    This function creates a **radial gradient** by iterating over the **northwest (NW) quarter**
    of the surface (`width_/2`, `height_/2`) and mirroring the computed pixels to the remaining three
    quadrants (**northeast (NE), southeast (SE), and southwest (SW)**).

    The gradient transitions from `start_color_` at the center to `end_color_` at the edges.

    **Parameters**:

    - **width_** (int):
      The width of the surface in pixels (must be > 1).

    - **height_** (int):
      The height of the surface in pixels (must be > 1).

    - **start_color_** (tuple, optional):
      The starting RGB color at the center of the gradient.
      Default is **(255, 0, 0) [Red]**.

    - **end_color_** (tuple, optional):
      The ending RGB color at the outer edge of the gradient.
      Default is **(0, 0, 0) [Black]**.

    - **gradient_array_** (numpy.ndarray or None, optional):
      A NumPy array of shape `(width_, 3)`, containing precomputed RGB color values (`uint8`).
      If `None`, the gradient is computed dynamically.

    - **factor_** (float, optional):
      Controls the intensity and spread of the gradient.
      **Must be > 0**. Default is **1.4**.

    - **threads_** (int, optional):
      Number of concurrent threads to use for computation. Default is **8**.

    **Returns**:
    - **pygame.Surface**:

      A `pygame.Surface` object containing the radial gradient,
      **centered at (width_/2, height_/2)**.

    **Example Usage**:

    .. code-block:: python

        surface = create_quarter_radial_gradient(
            width_=800,
            height_=800,
            start_color_=(255, 0, 0),
            end_color_=(0, 0, 0),
            factor_=1.4
        )

|

.. py:function:: create_quarter_radial_gradient_alpha(width_, height_, start_color_=(255, 0, 0, 255), end_color_=(0, 0, 0, 0), gradient_array_=None, factor_=1.4, threads_=8)

    |

    **Summary**:
    Generates a quarter radial gradient with alpha transparency (32-bit).
    This function creates a **radial gradient with transparency** by iterating over the
    **northwest (NW) quarter** of the surface (`width_/2`, `height_/2`) and mirroring the computed
    pixels to the remaining three quadrants (**northeast (NE), southeast (SE), and southwest (SW)**).

    The gradient smoothly transitions from `start_color_` at the center to `end_color_` at the edges,
    incorporating **alpha blending for transparency**.

    **Parameters**:

    - **width_** (int):
      The width of the surface in pixels (must be > 1).

    - **height_** (int):
      The height of the surface in pixels (must be > 1).

    - **start_color_** (tuple, optional):
      The color at the **center** of the gradient in **RGBA format**.
      Default is **(255, 0, 0, 255) [Opaque Red]**.

    - **end_color_** (tuple, optional):
      The color at the **outer edge** of the gradient in **RGBA format**.
      Default is **(0, 0, 0, 0) [Fully Transparent Black]**.

    - **gradient_array_** (numpy.ndarray or None, optional):
      A NumPy array of shape `(width_, 4)`, containing precomputed RGBA color values (`uint8`).
      If `None`, the gradient is computed dynamically.

    - **factor_** (float, optional):
      Controls the intensity and spread of the gradient.
      **Must be > 0**. Default is **1.4**.

    - **threads_** (int, optional):
      Number of concurrent threads to use for computation. Default is **8**.

    **Returns**:
    - **pygame.Surface**:

      A `pygame.Surface` object containing the radial gradient,
      **centered at (width_/2, height_/2)**,
      with **alpha transparency for smooth blending**.

    **Example Usage**:

    .. code-block:: python

        precomputed_gradient = create_line_gradient_rgba(
            int(math.sqrt(800 ** 2 + 800 ** 2)),
            start_rgba=(255, 0, 0, 255),
            end_rgba=(0, 0, 0, 0)
        )

        surface = create_quarter_radial_gradient_alpha(
            width_=800,
            height_=800,
            start_color_=(255, 0, 0, 255),
            end_color_=(0, 0, 0, 0),
            precomputed_gradient=precomputed_gradient
        ).convert_alpha()

|

.. py:function:: scroll_surface_24bit(surface, dx, dy)

    |

    **Summary**:
    Scrolls a **24-bit or 32-bit** Pygame surface **horizontally and/or vertically**.

    This function creates and returns a **new 24-bit surface** with the pixel data
    shifted according to the specified `dx` and `dy` values.
    Pixels that are shifted out of bounds will **wrap around** to the opposite side.

    **Parameters**:

    - **surface** (pygame.Surface):
      A Pygame `Surface` object in **24-bit or 32-bit** format.

    - **dx** (int):
      The number of pixels to scroll **horizontally**.
      - A **positive `dx`** shifts the surface to the **right**.
      - A **negative `dx`** shifts the surface to the **left**.

    - **dy** (int):
      The number of pixels to scroll **vertically**.
      - A **positive `dy`** shifts the surface **downward**.
      - A **negative `dy`** shifts the surface **upward**.

    **Returns**:
    - **pygame.Surface**:

      A **new** 24-bit `pygame.Surface` object with the applied scroll effect.

    **Raises**:
    - **TypeError**:

      If `surface` is **not** a `pygame.Surface` object.

    **Example Usage**:

    .. code-block:: python

        new_surface = scroll_surface_24bit(original_surface, dx=10, dy=-5)

    This shifts the surface **10 pixels to the right** and **5 pixels upward**.

|

.. py:function:: scroll24_inplace(surface, dx, dy)

    |

    **Summary**:
    Scrolls a **24-bit Pygame surface** horizontally and/or vertically **in place**.

    This function **modifies** the given surface **directly** by shifting its pixel data
    based on the specified horizontal (`dx`) and vertical (`dy`) displacement.
    Scrolling **wraps around**, meaning pixels that move off one edge **reappear**
    on the opposite edge.

    **Parameters**:

    - **surface** (pygame.Surface):

      A `pygame.Surface` object in **24-bit or 32-bit** format (compatible with pixel access).

    - **dx** (int):
      Horizontal scroll amount.
      - **Positive values** move pixels **to the right**.
      - **Negative values** move pixels **to the left**.

    - **dy** (int):
      Vertical scroll amount.
      - **Positive values** move pixels **downward**.
      - **Negative values** move pixels **upward**.

    **Returns**:

    - **None** (modifies the surface **in place**).

    **Raises**:
    - **TypeError**:

      If `surface` is **not** a `pygame.Surface` object.

    **Example Usage**:

    .. code-block:: python

        scroll24_inplace(image_surface, 1, 0)

    This shifts the surface **1 pixel to the right** without creating a new surface.

|

.. py:function:: scroll_rgb_array_inplace(rgb_array, dx, dy)

    |

    **Summary**:
    Scrolls a **3D RGB pixel array** horizontally and/or vertically **in place**.

    This function **modifies** the given `rgb_array` **directly** by shifting pixel values
    based on the specified horizontal (`dx`) and vertical (`dy`) offsets.
    Scrolling **wraps around**, meaning pixels shifted beyond the array's boundaries
    reappear on the opposite side.

    **Parameters**:

    - **rgb_array** (numpy.ndarray):
      - A **3D NumPy array** containing **RGB pixel values**.
      - The array must reference **all pixels** to ensure modification in place.
      - Expected shape: **(height, width, 3)** (RGB format).

    - **dx** (int, optional, default=0):
      - **Horizontal shift amount**.
      - **Negative values** move pixels **left**.
      - **Positive values** move pixels **right**.

    - **dy** (int, optional, default=0):
      - **Vertical shift amount**.
      - **Negative values** move pixels **up**.
      - **Positive values** move pixels **down**.

    **Returns**:
    - **None** (modifies `rgb_array` **in place**).

    **Raises**:

    - **TypeError**:
      If `rgb_array` is **not** a `numpy.ndarray` or `memoryviewslice`.

    - **ValueError**:
      - If `rgb_array` is **not a 3D array** with shape `(height, width, 3)`.
      - If `rgb_array.dtype` is **not** `numpy.uint8` (expected **8-bit RGB format**).
      - If `rgb_array` is **empty** (`shape[0] == 0`).

    **Notes**:

    - If both `dx` and `dy` are zero, the function performs **no operation**.
    - This function is optimized for **performance** and operates **without copying data**.
    - The array must be **contiguous** and properly referenced to avoid unexpected behavior.

    **Example Usage**:

    .. code-block:: python

        scroll_rgb_array_inplace(rgb_array, dx=5, dy=-2)

    This shifts the array **5 pixels to the right** and **2 pixels upward**.

|

.. py:function:: scroll_alpha_inplace(alpha_array, dx, dy)

    |

    **Summary**:
    Scrolls an **alpha channel array** (2D) **horizontally and/or vertically** **in place**.

    This function **modifies** the given `alpha_array` **directly** by shifting alpha values
    based on the specified horizontal (`dx`) and vertical (`dy`) offsets.
    Scrolling **wraps around**, meaning pixels shifted beyond the array's boundaries
    reappear on the opposite side.

    **Parameters**:

    - **alpha_array** (numpy.ndarray):
      - A **2D NumPy array** containing **alpha channel pixel values**.
      - The array must reference **all alpha pixel values** for in-place transformation.
      - Expected shape: **(height, width)** (grayscale alpha format).

    - **dx** (int, optional, default=0):
      - **Horizontal shift amount**.
      - **Negative values** move pixels **left**.
      - **Positive values** move pixels **right**.

    - **dy** (int, optional, default=0):
      - **Vertical shift amount**.
      - **Negative values** move pixels **up**.
      - **Positive values** move pixels **down**.

    **Returns**:
    - **None** (modifies `alpha_array` **in place**).

    **Notes**:

    - If both `dx` and `dy` are zero, the function performs **no operation**.
    - This function is optimized for **performance** and operates **without copying data**.
    - The array must be **contiguous** and properly referenced to avoid unexpected behavior.
    - **Alpha channel values** typically represent pixel transparency, where:
      - `0` = **Fully Transparent**
      - `255` = **Fully Opaque**

    **Example Usage**:

    .. code-block:: python

        scroll_alpha_inplace(alpha_array, dx=3, dy=-1)

    This shifts the alpha channel **3 pixels to the right** and **1 pixel upward**.

|

.. py:function:: scroll32_inplace(surface, dx, dy)

    |

    **Summary**:
    Scrolls a **32-bit Pygame surface** **horizontally and/or vertically** **in place**.

    This function **modifies** the given `surface` **directly** by shifting pixel data
    based on the specified horizontal (`dx`) and vertical (`dy`) offsets.
    Scrolling **wraps around**, meaning pixels shifted beyond the surface boundaries
    reappear on the opposite side.

    **Parameters**:

    - **surface** (pygame.Surface):
      - A `pygame.Surface` object with a **32-bit pixel format**.
      - The surface's pixel data will be **modified directly**.

    - **dx** (int, optional, default=0):
      - **Horizontal shift amount**.
      - **Negative values** move pixels **left**.
      - **Positive values** move pixels **right**.

    - **dy** (int, optional, default=0):
      - **Vertical shift amount**.
      - **Negative values** move pixels **up**.
      - **Positive values** move pixels **down**.

    **Returns**:
    - **None** (modifies `surface` **in place**).

    **Exceptions**:
    - **TypeError**: If `surface` is not a valid `pygame.Surface`.

    **Notes**:

    - If both `dx` and `dy` are zero, the function performs **no operation**.
    - This function ensures **efficient, in-place modification** without copying data.
    - Scrolling is **cyclic (toroidal wrap-around)**, meaning pixels that exit one edge **reappear** on the opposite edge.

    **Example Usage**:

    .. code-block:: python

        surface = pygame.Surface((200, 200))
        scroll32_inplace(surface, dx=15, dy=-10)  # Moves 15 pixels right and 10 pixels up.

|

.. py:function:: rgb_to_int(red, green, blue)

    |

    **Summary**:
    Convert RGB color values into a single 32-bit integer, similar to pygame's map_rgb() function.

    This Cython `cpdef` function allows direct calls without requiring a Python hook function. It efficiently encodes red, green, and blue values into a packed integer representation.

    **Parameters**:

    - **red** (unsigned char):
      - Red color component (0-255).

    - **green** (unsigned char):
      - Green color component (0-255).

    - **blue** (unsigned char):
      - Blue color component (0-255).

    **Returns**:

    - **unsigned int**:
      - A 32-bit unsigned integer representing the combined RGB color.

|

.. py:function:: int_to_rgb(n)

    |

    **Summary**:
    Converts a **32-bit unsigned integer** into an **RGB color representation**.

    This function extracts the **red, green, and blue** components from a **packed integer format**
    (similar to `pygame.Color.unmap_rgb()`) and returns them as a **tuple**
    containing **values in the range [0, 255]**.

    **Parameters**:

    - **n** (int):
      - A **32-bit unsigned integer** representing an **RGB color**.

    **Returns**:

    - **tuple (unsigned char, unsigned char, unsigned char)**:
      - A tuple containing the **extracted** `(red, green, blue)` **values**.

    **Example Usage**:

    .. code-block:: python

        r, g, b = int_to_rgb(0xFF8000)  # Extracts (255, 128, 0) from integer.

|

.. py:function:: _randf(lower, upper)

    |

    **Summary**:

    Generates a **random floating-point number** in the range **[lower, upper)**.

    This function is an **optimized alternative** to `random.uniform(lower, upper)`,
    leveraging an **external C function** (`randRangeFloat`) for improved performance.

    **Note:**

    - This function operates **without the Python GIL (`nogil`)**,
      making it **safe for parallel execution in Cython**.

    **Parameters**:

    - **lower** (float):

      - The **lower bound** of the range (**inclusive**).

    - **upper** (float):

      - The **upper bound** of the range (**exclusive**).

    **Returns**:
    - **float**:

      - A **random float value** in the range **[lower, upper)**.

    **Example Usage**:

    .. code-block:: python

        value = _randf(0.0, 1.0)  # Generates a random float between 0.0 and 1.0.

|

.. py:function:: _randi(lower, upper)

    |

    **Summary**:
    Generates a **random integer** in the range **[lower, upper]**.

    This function is an **optimized alternative** to `random.randint(lower, upper)`,
    leveraging an **external C function** (`randRange`) for improved performance.

    **Performance Notes**:

    - Declared as **`inline`** for **faster execution**.
    - Operates **without the Python Global Interpreter Lock (`nogil`)**,
      making it **safe for parallel execution in Cython**.

    **Parameters**:

    - **lower** (int):
      - The **lower bound** of the range (**inclusive**).

    - **upper** (int):
      - The **upper bound** of the range (**inclusive**).

    **Returns**:

    - **int**:
      - A **random integer** in the range **[lower, upper]**.

    **Important Notes**:

    - The function **does not check** whether `lower <= upper`.
      **Incorrect usage** may result in **undefined behavior**,
      depending on the `randRange` implementation.

    **Example Usage**:

    .. code-block:: python

        value = _randi(1, 10)  # Generates a random integer between 1 and 10.

|

.. py:function:: combine_rgba_buffers(w, h, bgr_buffer, alpha_channel, output_rgba_buffer, transpose_output=False)

    |

    **Summary**:
    Combines separate **BGR** and **Alpha** memory buffers into a single **RGBA** buffer.

    This function takes **separate BGR** (Blue-Green-Red) and **Alpha** memory buffers,
    stacks them together into a single **RGBA buffer**, and optionally transposes the result.
    It is optimized using **Cython** and parallelized with **OpenMP** for improved performance.

    If `transpose_output` is **True**, the output memory view is flipped by swapping
    rows and columns.

    **Parameters**:

    - **w** (int):
      - The **width** of the texture.

    - **h** (int):
      - The **height** of the texture.

    - **bgr_buffer** (memoryview [unsigned char [::1]]):
      - A **1D contiguous memory view** containing packed **BGR values** (uint8).
      - Expected size: `width * height * 3`.

    - **alpha_channel** (memoryview [unsigned char [::1]]):
      - A **1D contiguous memory view** containing **alpha values** (uint8).
      - Expected size: `width * height`.

    - **output_rgba_buffer** (memoryview [unsigned char [::1]]):
      - A pre-allocated **1D memory view** to store the resulting **RGBA values** (uint8).
      - Expected size: `width * height * 4`.

    - **transpose_output** (bool, optional, default=False):
      - If **True**, the resulting **RGBA buffer** is transposed (flipped).

    **Returns**:
    - **memoryview [unsigned char [::1]]**:

      - A contiguous **memory view** containing the combined **RGBA pixel values** (uint8).

    **Raises**:

    - **ValueError**:

      - If `w` or `h` is **non-positive**.
      - If `bgr_buffer` size does not match `width * height * 3`.
      - If `alpha_channel` size does not match `width * height`.
      - If `output_rgba_buffer` is not large enough to store `width * height * 4`.

    **Notes**:

    - Uses **`memcpy`** for **efficient memory copying**.
    - Supports optional **transposition** of the buffer.
    - Designed for **performance-critical applications**.

    **Example Usage**:

    .. code-block:: python

        rgba_buffer = combine_rgba_buffers(
            w=800, h=600,
            bgr_buffer=bgr_data,
            alpha_channel=alpha_data,
            output_rgba_buffer=output_data
        )

|

.. py:function:: normalize_2d_array(array2d)

    |

    **Summary**:
    Normalize a **2D array** of unsigned 8-bit integers (**uint8**) to floating-point values in the range **[0, 1]**.

    This function takes a **2D array** with shape `(width, height)` containing **uint8 values** (0-255) and converts it into a **MemoryViewSlice (2D array)** of floats, where each value is rescaled to the range **[0, 1]**.

    **Parameters**:

    - **array2d** (memoryview [unsigned char[:, :]]):
      - A **2D array** of shape `(width, height)` containing **uint8 values** representing pixel intensities.

    **Returns**:
    - **memoryview [float[:, :]]**:

      - A **2D array** of shape `(width, height)` with **float values** normalized to the range [0, 1].

    **Raises**:

    - **ValueError**:
      - If the input array does not have **exactly two dimensions**.

    **Notes**:
    - Uses **`prange`** with **OpenMP** for **parallelized operations** to improve performance.
    - Performs element-wise normalization using the constant `ONE_255 = 1/255.0`.
    - Designed for **fast image processing** and **machine learning applications**.

    **Example Usage**:

    .. code-block:: python

        normalized_array = normalize_2d_array(array2d)

|

.. py:function:: generate_spectrum_surface(width, height, gamma=1.0)

    |

    **Summary**:
    Create a **pygame surface** displaying the **light spectrum** ranging from **380-750 nm**.

    The function visualizes the light spectrum by displaying colors corresponding to specific wavelengths. The spectrum includes the following color regions:

    - **Red**: 620-750 nm, 484-400 THz
    - **Orange**: 590-620 nm, 508-484 THz
    - **Yellow**: 570-590 nm, 526-508 THz
    - **Green**: 495-570 nm, 606-526 THz
    - **Blue**: 450-495 nm, 668-606 THz
    - **Violet**: 380-450 nm, 789-668 THz

    **Parameters**:

    - **width** (int):
      - The width of the surface (image) in pixels.

    - **height** (int):
      - The height of the surface (image) in pixels.

    - **gamma** (float, optional, default=1.0):
      - The gamma value for color adjustment.

    **Returns**:
    - **pygame.Surface**:

      - A **24-bit pygame surface** with shape `(width, height)` containing the light spectrum. It is optimized for fast **blit** operations.

    **Example Usage**:

    .. code-block:: python

        spectrum_surface = generate_spectrum_surface(800, 600, gamma=1.2)


|

.. py:function:: bubble_sort(nums, size)

    |

    **Summary**:

    Sorts an array of **unsigned chars** (e.g., bytes or pixel values) using the **bubble sort algorithm**.

    Bubble sort iteratively compares adjacent elements in the array and swaps them if they are in the wrong order. The process continues until the array is fully sorted. The algorithm stops when no swaps are made during a complete pass.

    **Parameters**:

    - **nums** (unsigned char [::1]):
      - A one-dimensional memoryview or array of unsigned char values to be sorted. Typically, this array represents byte values or pixel data (e.g., RGB values).

    - **size** (int):
      - The number of elements in the `nums` array to sort.

    **Returns**:

    - **None**:
      - This function sorts the array **in place**. No value is returned; the input array is modified directly.

    **Example Usage**:

    .. code-block:: python

        nums = bytearray([4, 2, 3, 1, 5])
        bubble_sort(nums, len(nums))
        print(nums)
        # Output: bytearray(b'\x01\x02\x03\x04\x05')

    **Notes**:
    - This is an **in-place sorting algorithm**, meaning it modifies the original array.
    - Bubble sort is not the most efficient sorting algorithm, but it is simple to implement.
    - The algorithm runs in **O(n^2)** time in the worst and average cases, which makes it less efficient for large datasets.

|

.. py:function:: insertion_sort(nums, size)

    |

    **Summary**:
    Sorts an array of **unsigned char** elements using the **Insertion Sort algorithm**.

    Insertion sort iterates through the array, picking each element one by one, and inserts it into the sorted portion of the array while shifting the other elements accordingly. This implementation uses **parallel processing** for improved performance.

    **Example Usage**:

    .. code-block:: python

        nums = bytearray([5, 2, 9, 1, 5, 6])
        insertion_sort(nums, len(nums))

    **Parameters**:

    - **nums** (unsigned char [::1]):
      - A 1D numpy array or memory view of unsigned char (uint8) elements, representing the array to be sorted.

    - **size** (int):
      - The size of the array `nums`. It specifies the number of elements in the array.

    **Returns**:

    - **None**:
      - This function sorts the array **in place**. No value is returned; the input array is modified directly.

    **Notes**:
    - The algorithm runs in **O(n^2)** time complexity in the worst and average cases, making it less efficient for large datasets.
    - Insertion sort is particularly efficient for small datasets or arrays that are already nearly sorted.
    - Parallel processing is utilized in this implementation to improve performance.

|

.. py:function:: partition_cython(nums, low, high)

    |

    **Summary**:
    Partition function for the **quicksort algorithm**.

    This function selects a pivot element and partitions the input array `nums` such that all elements less than the pivot are moved to its left, and all elements greater than the pivot are moved to its right. It returns the index of the partition point.

    **Parameters**:

    - **nums** (unsigned char [::1]):
      - A 1D memoryview of unsigned char values, representing the input array to be partitioned.

    - **low** (int):
      - The starting index of the subarray to be partitioned.

    - **high** (int):
      - The ending index of the subarray to be partitioned.

    **Returns**:

    - **int**:
      - The index of the partition point in the array.

    **Notes**:
    - This function is used as part of the quicksort algorithm.
    - It ensures that elements less than the pivot are on the left, and elements greater than the pivot are on the right.

|

.. py:function:: _quick_sort(items, low, high)

    |

    **Summary**:
    In-place quicksort implementation for an array of unsigned char values.

    This function recursively sorts the array `items` by selecting a pivot and partitioning the array such that elements smaller than the pivot move to the left and larger elements move to the right. It is optimized to reduce recursion depth by sorting the smaller partition first.

    **Parameters**:

    - **items** (unsigned char [::1]):
      - A 1D memoryview of unsigned char values, representing the input array to be sorted.

    - **low** (unsigned int):
      - The starting index of the subarray to be sorted.

    - **high** (unsigned int):
      - The ending index of the subarray to be sorted.

|

.. py:function:: heapify(nums, heap_size, root_index)

    |

    **Summary**:
    Ultra-optimized iterative heapify function for maintaining the max heap property.

    This function “sifts down” the value at the given root index so that the subtree rooted at that index becomes a valid max heap. It is assumed that the subtrees below are already valid heaps. For full heapsort, make sure to build the heap by calling this function on all non-leaf nodes in reverse order.

    **Parameters**:

    - **nums** (unsigned char [::1]):
      - A 1D memoryview of unsigned char values, representing the heap.

    - **heap_size** (unsigned int):
      - The number of elements in the heap.

    - **root_index** (unsigned int):
      - The index of the root node of the subtree to heapify.

|

.. py:function:: heap_sort(nums, n)

    |

    **Summary**:
    Performs in-place heap sort on an array of unsigned char values.

    This function first builds a max heap from the input array, ensuring the largest element is at the root. Then, it repeatedly extracts the maximum element (swaps it with the last element) and re-heapifies the reduced heap to maintain heap properties until the entire array is sorted.

    **Time Complexity**:
    - Heap construction: O(n)
    - Extraction and re-heapification: O(n log n)
    - Overall: O(n log n)

    **Space Complexity**: O(1) (in-place sorting)

    **Parameters**:

    - **nums** (unsigned char [::1]):
      - A 1D memoryview of unsigned char values to be sorted.

    - **n** (unsigned int):
      - The number of elements in the array.

|




