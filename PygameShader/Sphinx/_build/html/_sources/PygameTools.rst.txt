PygameTools
========================================

:mod:`PygameTools.pyx`

=====================

.. currentmodule:: PygameTools

|

1. Purpose of This Library
--------------------------

This library is designed for **high-performance image processing** with a focus on **efficiency and low-level memory management**.
It leverages **Cython and NumPy** to provide fast image operations, including **resizing, format conversions, pixel manipulations, and color channel analysis**.
By using memory views and Cython optimizations, the library ensures minimal overhead when processing images in **RGB and RGBA formats**.

2. Key Features
---------------

- **Efficient Image Manipulation**: Functions to resize, transform, and reformat images with minimal processing overhead.
- **Memory-Efficient Buffer Handling**: Converts images to and from compact buffer representations for efficient storage and processing.
- **Pixel-Level Operations**: Supports pixel indexing, blending, and mapping/unmapping between color representations.
- **Alpha Channel Handling**: Enables operations such as alpha removal, binary masking, and blending with transparency.
- **Performance Optimizations**: Uses Cython’s `nogil` operations for multi-threading and fast execution.
- **Comparison and Analysis Tools**: Functions for comparing PNG images and analyzing color channels.

3. Library Functional Overview
------------------------------

This library provides several core functionalities for working with images:

**Image Resizing & Buffer Manipulation**

- ``resize_array()``: Resize an RGB image efficiently.
- ``bufferize()`` / ``unbufferize()``: Convert images to and from compact buffer formats.
- ``flip_bgra_buffer()`` / ``bgr_buffer_transpose()``: Perform buffer-based transformations.

**Pixel Mapping & Indexing**

- ``mapping_array()`` / ``unmapping_array()``: Convert RGB images to indexed formats and vice versa.
- ``index3d_to_1d()`` / ``index1d_to_3d()``: Handle pixel coordinate transformations.

**Alpha Channel Processing**

- ``RemoveAlpha()``: Strip the alpha channel from an RGBA image.
- ``make_rgba_array()``: Combine an RGB image with an alpha mask.
- ``binary_mask()``: Generate binary masks from alpha values.

**Image Blending & Comparison**

- ``blend_pixel_mapped_arrays()``: Blend images using mapped pixel values.
- ``compare_png24bit()`` / ``compare_png32bit()``: Compare images for similarity.

**Image Analysis**

- ``get_rgb_channel_means()``: Compute average color values per channel.
- ``analyze_image_channels()``: Extract statistical insights from an image.

4. Target Applications
----------------------

- **Game Development**: Handling and processing game textures efficiently.
- **Computer Vision**: Preprocessing images for machine learning models.
- **Graphics Software**: Performing transformations, blending, and format conversions.
- **Embedded Systems**: Optimized image handling in memory-constrained environments.
- **Scientific Image Analysis**: Extracting statistics and performing pixel-level computations.

5. Summary
----------

This **Cython-based image processing library** provides **high-performance image transformations, buffer manipulation, and alpha channel handling**.
With optimized **resizing, mapping, blending, and analysis functions**, it is designed for applications needing **fast, memory-efficient image processing**.
Its **low-level optimizations** make it particularly well-suited for **real-time graphics, computer vision, and game development**.

6. Cython list methods
----------------------

|

.. py:function:: RemoveAlpha(rgba_array)

   |

   Convert an RGBA or BGRA array to RGB by removing the alpha channel.

   This function is used to process images with an alpha transparency channel
   and convert them to a format with no alpha, reducing a 32-bit image to a 24-bit image.

   The function accepts a 3D numpy array or memoryview containing RGBA/BGRA data,
   and outputs a new 3D array with the alpha channel removed (RGB).

   **Example:**

   .. code-block:: python

      rgb_array = RemoveAlpha(rgba_array)


   **Parameters**

      rgba_array : numpy.ndarray or memoryview
          A numpy array or memoryviewslice with shape (w, h, 4) and dtype uint8.
          The array represents an image in RGBA or BGRA format (with alpha transparency).

   **Returns**

      numpy.ndarray
          A new numpy array with shape (w, h, 3) and dtype uint8, containing the RGB
          values (with no alpha transparency).

   **Raises**

      ValueError
          If the input array does not have the expected shape (w, h, 4).

      TypeError
          If the input array is not of type uint8.

|

.. py:function:: resize_array(rgb_array, w2, h2)

   |

   Rescale an array (returning a new array).

   This function rescales a 3D RGB or RGBA array (of type uint8)
   to a new width and height. The output will have the shape (w2, h2, 3|4),
   preserving the number of channels (3 for RGB or 4 for RGBA).

   **Example:**

   .. code-block:: python


      result = resize_array(rgb_array, 800, 600)

   **Parameters**

      rgb_array : numpy.ndarray or memoryview
          A numpy array or memoryviewslice of shape (w, h, 3|4).
          Contains pixels in RGB or RGBA format (uint8).

      w2 : int
          The width of the output array.

      h2 : int
          The height of the output array.

   **Returns**

      numpy.ndarray or memoryview
          A new array with the shape (w2, h2, 3|4) and type uint8.
          The format of the pixel data (RGB or RGBA) is the same as the input.

   **Raises**

      ValueError
          If the input array does not have a valid shape (w, h, 3|4) or the `bit_size` is not supported.

      TypeError
          If the input array is not of type uint8.

      TypeError
          If the input is not a numpy array or memoryviewslice, or if the data type is not uint8.

|

.. py:function:: resize_array_c(rgb_array, w2, h2)

   |

   Array rescale (return a new array)

   This function rescales an image array in RGB or RGBA format (uint8) to a specified width (`w2`)
   and height (`h2`). The resulting array maintains the pixel format of the input and has a shape
   of (w2, h2, 3|4).

   **Example:**

   .. code-block:: python

      new_array = resize_array_c(rgb_array, 800, 400)


   **Parameters**

      rgb_array : numpy.ndarray or memoryview
          A 3D array with a shape of (w, h, 3|4) representing the pixel data of the image.
          The array must be in RGB (3 channels) or RGBA (4 channels) format and have a dtype of uint8.

      w2 : int
          The width of the resized image. Must be a positive integer greater than 0.

      h2 : int
          The height of the resized image. Must be a positive integer greater than 0.

   **Returns**

      memoryview
          A resized image array with shape (w2, h2, 3|4), dtype uint8, and the same pixel format
          (RGB or RGBA) as the input.

   **Raises**

      ValueError
          If `w2` or `h2` is not greater than 0, or if the shape of `rgb_array` is not understood
          (must be (w, h, 3|4)).

      TypeError
          If `rgb_array` is not a numpy.ndarray or memoryviewslice, or if its dtype is not uint8.

|

.. py:function:: make_rgba_array(rgb_, alpha_, transpose_=False)

   |

   Create RGBA array from RGB and Alpha arrays.

   This function generates a new RGBA memoryview array by combining an RGB array
   (shape (w, h, 3)) and an Alpha array (shape (w, h)). The resulting RGBA array has
   a shape of (w, h, 4) or (h, w, 4) if `transpose_` is set to `True`. The function
   is useful for creating images with transparency, such as 32-bit PNG images.

   **Example Usage:**

   .. code-block:: python


      im = pygame.image.load("../Assets/px.png").convert_alpha()
      w, h = im.get_width(), im.get_height()
      rgb = pygame.surfarray.pixels3d(im)
      alpha = pygame.surfarray.pixels_alpha(im)
      rgba = make_rgba_array(rgb, alpha, transpose_=True)
      surf = pygame.image.frombuffer(rgba, (w, h), "RGBA").convert_alpha()


   **Parameters**

      **rgb_** : numpy.ndarray or memoryview
          A 3D array with shape (w, h, 3) containing RGB pixel values (uint8). The dimensions
          must match the `alpha_` array.

      **alpha_** : numpy.ndarray or memoryview
          A 2D array with shape (w, h) containing alpha (transparency) values (uint8). The
          dimensions must match the `rgb_` array.

      **transpose_** : bool
          A flag to indicate if the output array should be transposed. If `True`, the shape
          of the output will be (h, w, 4) instead of (w, h, 4).

   **Returns**

      memoryview
          A Cython memoryview with shape (w, h, 4) or (h, w, 4) containing RGBA pixel
          values (uint8). This array can be directly used for image processing or converted
          to a numpy.ndarray.

   **Raises**

      TypeError
          If `rgb_` or `alpha_` is not a numpy.ndarray or memoryviewslice, or if the data
          type is not uint8.

      ValueError
          If the dimensions of `rgb_` and `alpha_` do not match.

   **Notes**

      - Both input arrays (`rgb_` and `alpha_`) must be of uint8 data type.
      - The function is designed for high-performance processing using memoryviewslice.

|

.. py:function:: create_rgba_surface(rgb_, alpha_, tmp_array_=None)

   |

   Create a 32-bit image/surface from RGB and Alpha arrays.

   This function generates a 32-bit image surface (with alpha transparency) by combining:
     - An RGB array with shape (w, h, 3) and dtype uint8.
     - An Alpha array with shape (w, h) and dtype uint8.

   Both input arrays must have the same width and height (w, h), and must be of type uint8.
   Optionally, a pre-allocated temporary array can be passed to optimize performance.

   **Example Usage:**

   .. code-block:: python


      im = pygame.image.load("../Assets/alpha.png").convert_alpha()
      rgb = pygame.surfarray.pixels3d(im)          # Extract the RGB pixel values.
      alpha = pygame.surfarray.pixels_alpha(im)    # Extract the Alpha (transparency) values.

      # Create the surface. Use convert_alpha() for optimal performance:
      image = create_rgba_surface(rgb, alpha).convert_alpha()


   **Parameters**


      **rgb_** : numpy.ndarray or memoryview
          A 3D array with shape (w, h, 3) containing RGB pixel values (dtype=uint8).

      **alpha_** : numpy.ndarray or memoryview
          A 2D array with shape (w, h) containing alpha (transparency) values (dtype=uint8).

      **tmp_array_** : numpy.ndarray or memoryview, optional
          A 3D array with shape (w, h, 4) used as a temporary array for creating the RGBA surface.
          Passing this pre-allocated array can speed up the process by avoiding allocation overhead.


   **Returns**


      pygame.Surface
          A 32-bit pygame surface with RGBA values.
          Use `convert_alpha()` to enhance performance when rendering the surface.
          Note: `convert_alpha()` requires the video display to be initialized.


   **Notes**


      - The input arrays must have compatible shapes and types.
      - For optimal performance:
          - Use pygame's `convert_alpha()` method.
          - Ensure the video display is initialized before calling `convert_alpha()`.

|


.. py:function:: alpha_array_to_surface(array)

   |

   Convert a 2D alpha array (shape w x h, type uint8) into a 24-bit Pygame surface (RGB).

   This function takes a 2D alpha array and creates a new 24-bit surface (RGB).
   Each pixel in the output surface has its R, G, and B channels set to the
   corresponding alpha value from the input array.

   **Example Usage:**

   .. code-block:: python


      im = pygame.image.load("../Assets/alpha.png").convert_alpha()
      alpha = pygame.surfarray.pixels_alpha(im)
      image = alpha_array_to_surface(alpha)


   **Parameters**


      array : numpy.ndarray or memoryview
          A 2D numpy array or memoryview with shape (w, h) and dtype uint8.


   **Returns**


      pygame.Surface
          A 24-bit Pygame surface (RGB).
          For better in-game performance, use `pygame.Surface.convert()` after creation.

|

.. py:function:: bufferize(rgb_array)

   |

   Create a C-buffer from a 3D numpy array.

   This function takes a 3D numpy array representing an RGB or RGBA image and converts it into a
   1D C-style buffer. The numpy array should have a shape of (width, height, 3|4) with uint8 type.

   **Parameters**

      rgb_array : numpy.ndarray
          A numpy array of shape (w, h, 3|4), representing an RGB or RGBA image.
          The last dimension (3 or 4) corresponds to the color channels (RGB or RGBA).

   **Returns**

      memoryview
          A memoryview slice (C buffer) containing the RGB or RGBA pixel values.
          This is a 1D array that can be accessed directly in C.

|


.. py:function:: unbufferize(c_buffer, width, height, bit_size)

   |

   Convert a 1D array (buffer) of uint8 data type into a 3D numpy array (copy).

   This function takes a 1D buffer of image data (such as RGB or RGBA values) and reconstructs it into
   a 3D array (numpy array) of shape (width, height, bit_size), where bit_size can be 3 (RGB) or 4 (RGBA).


   **Example:**

   .. code-block:: python


      c_buffer = <some_1d_buffer>
      width = 256
      height = 256
      bit_size = 3  # RGB
      result = unbufferize(c_buffer, width, height, bit_size)


   **Parameters**


      c_buffer : memoryview
          The input 1D buffer array (as a memoryview) containing pixel values in uint8 format.

      width : int
          The width of the resulting 3D array (image width in pixels).

      height : int
          The height of the resulting 3D array (image height in pixels).

      bit_size : int
          The number of channels in the image. Typically 3 (RGB) or 4 (RGBA).


   **Returns**


      memoryview
          A memoryview slice (3D array) of shape (width, height, bit_size),
          containing the RGB or RGBA pixel values.


   **Raises**


      ValueError
          If `bit_size == 0`, as it's an invalid value for image data.

|


.. py:function:: bgr_buffer_transpose(width, height, buffer, transposed_buffer=None)

   |

   Transpose rows & columns of a BGR image buffer.

   This function transposes a BGR image by rearranging the pixel data from the
   input `buffer` into the `transposed_buffer`. It assumes the image is represented as
   a 1D array of BGR pixels in row-major order.

   This is equivalent to performing a transpose operation on the pixel data:
   transposing rows and columns of BGR blocks.

   **Example Input/Output:**

   BGR image's pixels represented as:
   .. code-block::

      [BGR1, BGR2,  BGR3,  BGR4]
      [BGR5, BGR6,  BGR7,  BGR8]
      [BGR9, BGR10, BGR11, BGR12]

   After transpose:

   output image's pixels represented as:
   .. code-block::

      [BGR1, BGR5, BGR9]
      [BGR2, BGR6, BGR10]
      [BGR3, BGR7, BGR11]
      [BGR4, BGR8, BGR12]

   **Example Usage**

   .. code-block:: python

      source = pygame.image.load('../Assets/px.png').convert(24)
      source = pygame.transform.smoothscale(source, (800, 600))
      arr = numpy.empty(800 * 600 * 3, dtype=numpy.uint8)
      arr = bgr_buffer_transpose(800, 600, source.get_buffer(), arr)


   **Parameters**


      width : int
          The width of the video frame. Must be greater than 0.

      height : int
          The height of the video frame. Must be greater than 0.

      buffer : memoryview
          A 1D memoryview containing the pixel data (typically in BGR format).

      transposed_buffer : memoryview, optional
          An optional 1D memoryview to store the transposed pixel data. If not provided,
          a new buffer will be created.


   **Returns**


      numpy.ndarray
          A NumPy array containing the transposed pixel data.


   **Raises**


      ValueError
          If `width` or `height` is less than or equal to 0.
          If `buffer` is None or its size does not match `width * height * 3`.
          If `transposed_buffer` is provided but its size does not match `width * height * 3`.

      TypeError
          If `buffer` or `transposed_buffer` is not a memoryview or NumPy array.
          If `buffer` is a NumPy array and not contiguous.


   **Notes**


      - This function assumes the buffer data is in BGR format and performs transpose.
      - The operation is done in-place if a `transposed_buffer` is provided.

|


.. py:function:: flip_bgra_buffer(width, height, buffer, flipped_buffer)

   |

   Perform flipping of a BGRA image buffer.

   This core function flips a BGRA image by rearranging the pixel data from the
   input `buffer` into the `flipped_buffer`. It assumes the image is represented as
   a 1D array of BGRA pixels in row-major order.

   This is equivalent to performing a transpose operation on the pixel data:
   flipping rows and columns of BGRA blocks.

   **Example Input/Output:**

   Input buffer:
   .. code-block::

      buffer = [BGRA1, BGRA2, BGRA3, BGRA4, BGRA5, BGRA6, BGRA7, BGRA8, BGRA9]
      Represented as:
          [BGRA1, BGRA2, BGRA3]
          [BGRA4, BGRA5, BGRA6]
          [BGRA7, BGRA8, BGRA9]

   After flipping:

   output buffer:
   .. code-block::

      [BGRA1, BGRA4, BGRA9, BGRA2, BGRA5, BGRA8, BGRA3, BGRA6, BGRA9]
      Represented as:
          [BGRA1, BGRA4, BGRA9]
          [BGRA2, BGRA5, BGRA8]
          [BGRA3, BGRA6, BGRA9]

   **Parameters**

      width : int
          The width of the image in pixels.

      height : int
          The height of the image in pixels.

      buffer : memoryview
          The input 1D buffer containing BGRA pixel data. The buffer size must be
          `width * height * 4` to accommodate all pixels in the image.

      flipped_buffer : memoryview
          The output 1D buffer to store the flipped BGRA data. The size
          must also be `width * height * 4`.


   **Returns**

      numpy.ndarray
          A 1D NumPy array containing the flipped BGRA pixel data, stored in row-major order.


   **Raises**

      ValueError
          If `width` or `height` is less than or equal to 0.
          If `buffer` is not 1D or its size does not match `width * height * 4`.
          If `flipped_buffer` is not 1D or its size does not match `width * height * 4`.

      TypeError
          If `buffer` or `flipped_buffer` is not a memoryview or NumPy array.


   **Notes**

      - The flipping operation is performed in parallel using OpenMP's `prange` for better
        performance on multi-threaded systems.
      - If `flipped_buffer` is not provided, a new writable buffer is allocated and returned.

   **ChatGPT Collaboration:**
      - This code was enhanced with the assistance of **ChatGPT**.

|


.. py:function:: binary_mask(mask_alpha_)

   |

   Convert a 2D memoryview or a numpy.array into a black-and-white mask alpha array.

   This function processes a 2D memoryview of type `uint8` (values ranging from 0 to 255)
   and converts it into a binary black-and-white mask. Pixels with a value of `0`
   are converted to `0` (black), while pixels with a value greater than `0` are
   converted to `255` (white). The function returns a memoryview, not a NumPy array.

   **Parameters**

      **mask_alpha_** : memoryview
          A 2D memoryview of shape (W, H) and type `uint8`. Each element represents
          an alpha value ranging from 0 (fully transparent) to 255 (fully opaque).

   **Returns**

      memoryview
          A 2D memoryview of shape (W, H) and type `uint8`, where each element is either:
          - `0` (black): Corresponds to pixels with an original value of `0`.
          - `255` (white): Corresponds to pixels with an original value greater than `0`.

   **Raises**

      TypeError
          If the input `mask_alpha_` is not a `numpy.ndarray` or a memoryview slice.

      ValueError
          If:
          - `mask_alpha_` is not of type `uint8`.
          - `mask_alpha_` does not have exactly 2 dimensions.

   **Example**

   **Loading and converting an image's alpha channel to a binary mask:**
   .. code-block::

      # Load an image with alpha channel
      im = pygame.image.load("../Assets/alpha.png").convert_alpha()
      # Get the alpha channel as a NumPy array
      alpha = pygame.surfarray.pixels_alpha(im)
      # Convert the alpha channel to a black-and-white solid transparency mask
      solid_transparency = binary_mask(alpha)

   **Notes**

      - This function modifies the input memoryview `mask_alpha_` in place for performance reasons.
      - It operates on Cython memoryviews directly, which are more efficient than NumPy arrays
        for this type of low-level operation.
      - The operation is performed in parallel using OpenMP's `prange` for improved performance.
      - Ensure that `mask_alpha_` is writable before calling this function.

|


.. py:function:: mask32(rgb_, alpha_, mask_alpha_)

   |

   Apply a mask to an image split into its respective RGB & alpha values.

   This method creates a new alpha channel by combining the original alpha transparency
   with the mask alpha. It generates a 32-bit image with a new alpha channel for fast blitting.
   The new surface is already formatted for optimal performance, so there is no need to use
   the `convert_alpha` method from Pygame.

   **Parameters**

      **rgb_** : numpy.ndarray
          A 3D array with shape (w, h, 3) and type `uint8`, representing the RGB values of the image.

      **alpha_** : numpy.ndarray
          A 2D array with shape (w, h) and type `uint8`, representing the alpha (transparency) values of the image.

      **mask_alpha_** : numpy.ndarray
          A 2D array with shape (w, h) and type `uint8`, representing the mask's alpha values to apply.

   **Returns**

      object
          A 32-bit Pygame surface with a new alpha channel combining the original transparency and the mask alpha.

   **Example**

   **Applying a mask to an image and creating a new alpha channel:**
   .. code-block::

      new_image = mask32(rgb, alpha, mask_alpha).convert_alpha()

   **Notes**

      - The input arrays must have the same dimensions (w, h).
      - The function assumes that all arrays are of type `uint8`.
      - The resulting surface is already compatible for fast blitting and does not require further processing.

|


.. py:function:: channels_to_surface(red_channel, green_channel, blue_channel, alpha_channel, output_rgba_buffer=None)

   |

   Combine separate red, green, blue, and alpha channels into a single 32-bit Pygame surface.

   This function takes four 2D arrays representing the red, green, blue, and alpha channels of an image,
   and combines them into a single 32-bit RGBA surface. The result is a surface that supports transparency
   and is ready for rendering in Pygame.

   To ensure optimal performance and proper alpha transparency handling, use `pygame.convert_alpha()`
   after creating the surface.

   **Parameters**

      red_channel : numpy.ndarray or memoryview
          A 2D array or memoryview with shape (w, h) and type `uint8`, representing the red channel values
          for each pixel. Values range from 0 to 255.

      green_channel : numpy.ndarray or memoryview
          A 2D array or memoryview with shape (w, h) and type `uint8`, representing the green channel values
          for each pixel. Values range from 0 to 255.

      blue_channel : numpy.ndarray or memoryview
          A 2D array or memoryview with shape (w, h) and type `uint8`, representing the blue channel values
          for each pixel. Values range from 0 to 255.

      alpha_channel : numpy.ndarray or memoryview
          A 2D array or memoryview with shape (w, h) and type `uint8`, representing the alpha (transparency)
          channel values for each pixel. Values range from 0 to 255.

      output_rgba_buffer : numpy.ndarray, optional
          A temporary 3D array or memoryview with shape (h, w, 4) and type `uint8` for storing intermediate
          RGBA data. If not provided, a new array will be allocated.

   **Returns**

      pygame.Surface
          A 32-bit Pygame surface in RGBA format, suitable for rendering and supporting alpha transparency.

   **Raises**

      ValueError
          - If the input channel arrays or memoryviews do not have the same shape.
          - If `output_rgba_buffer` has an incorrect shape.

      TypeError
          - If any of the input arrays or memoryviews is not a 2D NumPy array of type `uint8`.

   **Example**

      **Creating a surface from RGBA channels:**
      .. code-block::

         import pygame
         im = pygame.image.load("../Assets/rgba_image.png").convert_alpha()
         red = pygame.surfarray.pixels_red(im)
         green = pygame.surfarray.pixels_green(im)
         blue = pygame.surfarray.pixels_blue(im)
         alpha = pygame.surfarray.pixels_alpha(im)

         surface = channels_to_surface(red, green, blue, alpha)
         surface = surface.convert_alpha()  # Optimize the surface for rendering

   **Notes**

      - This function works with both NumPy arrays and memoryviews, offering efficient data processing.
      - For proper transparency handling, use `pygame.Surface.convert_alpha()` on the returned surface.

|


.. py:function:: compare_png24bit(surface1, surface2)

   |

   Compute the pixel-wise difference between two 24-bit Pygame surfaces.

   This function takes two Pygame surfaces in 24-bit format (PNG) and calculates the absolute
   difference between their RGB values. The output is a new 24-bit Pygame surface where each pixel
   represents the difference between the corresponding pixels in `surface1` and `surface2`.

   **Example Usage**

   .. code-block:: python

      im1 = pygame.image.load("image1.png").convert(24)
      im2 = pygame.image.load("image2.png").convert(24)

      diff_surface = compare_png24bit(im1, im2)
      diff_surface = diff_surface.convert()  # Optimize for rendering

   **Parameters**

      surface1 : pygame.Surface
          A 24-bit Pygame surface representing the first image.

      surface2 : pygame.Surface
          A 24-bit Pygame surface representing the second image.

   **Returns**

      pygame.Surface
          A new 24-bit Pygame surface where pixel values indicate the absolute difference
          between `surface1` and `surface2`.

   **Raises**

      ValueError
          If the input surfaces do not have the same size or are not in 24-bit format.

   **Notes**

      - **Incompatible with JPEG images**, as JPEG compression alters pixel values due to its
        lossy nature.
      - Both input surfaces **must** have identical dimensions and be in 24-bit mode.
      - The output surface does not include an alpha channel (transparency).
      - For faster rendering, use `pygame.Surface.convert()` on the returned surface.

|

.. py:function:: compare_png32bit(surface1, surface2)

   |

   Compute the pixel-wise difference between two 32-bit Pygame surfaces.

   This function compares two Pygame surfaces in 32-bit format (PNG with an alpha channel)
   and calculates the absolute difference between their RGBA values. The output is a new
   32-bit Pygame surface where each pixel represents the difference between the corresponding
   pixels in `surface1` and `surface2`.

   **Example Usage**

   .. code-block:: python

      im1 = pygame.image.load("image1.png").convert_alpha()
      im2 = pygame.image.load("image2.png").convert_alpha()

      diff_surface = compare_png32bit(im1, im2)
      diff_surface = diff_surface.convert_alpha()  # Optimize for rendering

   **Parameters**

      surface1 : pygame.Surface
          A 32-bit Pygame surface (RGBA) representing the first image.

      surface2 : pygame.Surface
          A 32-bit Pygame surface (RGBA) representing the second image.

   **Returns**

      pygame.Surface
          A new 32-bit Pygame surface where pixel values indicate the absolute difference
          between `surface1` and `surface2`, including differences in transparency.

   **Raises**

      ValueError
          If the input surfaces do not have the same size or are not in 32-bit format.

   **Notes**

      - **Incompatible with JPEG images**, as JPEG compression alters pixel values due to its
        lossy nature.
      - Both input surfaces **must** have identical dimensions and be in 32-bit mode.
      - The output surface includes an alpha channel, preserving transparency differences.
      - For faster rendering, use `pygame.Surface.convert_alpha()` on the returned surface.

|

.. py:function:: unmapping_array(indexed_array_, tmp_array_=None)

   |

    Convert a 2D indexed array (int32) into a 3D RGB array (uint8).

    This function transforms a 2D array where each element is a 32-bit integer
    representing an RGB pixel into a 3D NumPy array, where each pixel's red,
    green, and blue channels are stored separately as uint8 values.

    **Purpose**

    This is the inverse of `pygame.surfarray.map_array()`, which converts a
    3D RGB array into a 2D int32 pixel map. It is useful for reconstructing RGB
    values from an indexed format.

    **Example Usage**

    .. code-block:: python

       # Convert a 2D indexed array into a 3D RGB array
       indexed_array = pygame.surfarray.map_array(surface, pygame.surfarray.pixels3d(surface))
       rgb_array = unmapping_array(indexed_array)

       # Create a Pygame surface from the RGB array
       new_surface = pygame.Surface((800, 600))
       pygame.pixelcopy.array_to_surface(new_surface, rgb_array)

       # Using a preallocated array for better performance
       preallocated_array = numpy.empty((800, 600, 3), dtype=numpy.uint8)
       rgb_array = unmapping_array(indexed_array, preallocated_array)

    **Parameters**

       **indexed_array_** : numpy.ndarray or memoryview
           A 2D array of shape (w, h) with int32 values, where each element
           represents an RGB pixel encoded as a single integer.

       **tmp_array_** : numpy.ndarray, optional
           A preallocated 3D array of shape (w, h, 3) and type uint8 to store
           the decomposed RGB values. If provided, this array is modified in place.

           - Shape: (w, h, 3)
           - Data type: uint8

    **Returns**

       numpy.ndarray[np.uint8_t, ndim=3]
           A 3D NumPy array of shape (w, h, 3), where:
           - `[..., 0]` contains the red (R) channel values.
           - `[..., 1]` contains the green (G) channel values.
           - `[..., 2]` contains the blue (B) channel values.

    **Raises**

       ValueError
           If `indexed_array_` is not of shape (w, h) or not of type int32.
           If `tmp_array_` is provided but has an incorrect shape or type.

       TypeError
           If inputs are not valid NumPy arrays or memoryviews.

    **Notes**

       - If `tmp_array_` is not provided, a new array is allocated.
       - Supplying a preallocated array is recommended for large datasets to
         improve performance.
       - This function is optimized for speed and operates directly on memory views.

|


.. py:function:: mapping_array(rgb_array, tmp_array_=None)

    |

    Convert a 3D RGB array (W, H, 3) into a 2D mapped integer array.

    This function transforms a 3D array of RGB values (uint8) into a 2D array where
    each RGB triplet is represented as a single int32 value. It is optimized as an
    equivalent to `pygame.surfarray.map_array(Surface, array3d)`, allowing efficient
    processing of NumPy arrays with shape (W, H, 3).

    **Parameters**

    **rgb_array** : memoryview (unsigned char[:, :, :])

        A 3D array or memoryview of shape (W, H, 3), containing pixel data in `uint8` format.
        Each pixel is represented by three values: red, green, and blue.

    **tmp_array_** : memoryview or numpy.ndarray, optional

        A preallocated 2D array of shape (W, H) with data type `int32`. If provided, the function
        modifies this array in place, avoiding additional memory allocation.

    **Returns**

    memoryview (int32[:, :])

        A 2D array of shape (W, H) with data type `int32`, where each element represents
        the mapped integer value of the corresponding RGB triplet.

    **Raises**

    ValueError

        - If `rgb_array` is `None`.
        - If `rgb_array` does not have the required shape `(W, H, 3)`.
        - If `rgb_array` has an invalid data type (not `uint8`).
        - If `tmp_array_` has an incorrect shape, data type, or other inconsistencies.

    TypeError

        - If `rgb_array` or `tmp_array_` is not a NumPy array or memoryview slice.

    **Notes**

    - This function is optimized for processing large images efficiently.
    - The input `rgb_array` must be compatible with `pygame.surfarray.array3d` output
      (an array of unsigned 8-bit integers).
    - If `tmp_array_` is provided, it must be writable.
    - If `tmp_array_` is not provided, a new 2D array is allocated internally.

    **Example Usage**

    .. code-block:: python


        import pygame
        import numpy as np

        # Load an image and extract RGB data
        surface = pygame.image.load("image.png").convert(24)  # Ensure it's a 24-bit surface
        rgb_array = pygame.surfarray.array3d(surface)

        # Convert the RGB array into a 2D mapped integer array
        mapped_array = mapping_array(rgb_array)

        # Create a new surface using the mapped array
        new_surface = pygame.Surface(surface.get_size())
        pygame.surfarray.blit_array(new_surface, mapped_array)

        # Using a preallocated array for efficiency
        preallocated_array = np.empty((rgb_array.shape[0], rgb_array.shape[1]), dtype=np.int32)
        mapped_array = mapping_array(rgb_array, preallocated_array)

|


.. py:function:: blend_pixel_mapped_arrays(target_pixels, blend_pixels, special_flags=0)

    |

    Blend two 2D mapped pixel arrays together with optional special effects.

    This function blends `target_pixels` and `blend_pixels` using the specified `special_flags`.
    The operation is performed **in-place**, directly modifying `target_pixels`.

    **Parameters**

    **target_pixels** : memoryview (unsigned int[:, :])

        A **2D Cython memoryview** (or a NumPy array converted via `numpy.asarray()`),
        where each element represents a pixel with mapped RGB values.
        The blend will be applied directly to this array (modifying it in-place).

    **blend_pixels** : memoryview (unsigned int[:, :])

        A **2D Cython memoryview** (or a NumPy array converted via `numpy.asarray()`),
        containing the pixel data to blend into `target_pixels`.
        The shape must match `target_pixels`.

    **special_flags** : unsigned char, optional (default=0)

        A blending mode flag that determines how `target_pixels` and `blend_pixels` are combined.
        The following **Pygame-style BLEND modes** are supported:

        - **BLEND_RGB_ADD** (1) → Additive blending
        - **BLEND_RGB_SUB** (2) → Subtractive blending
        - **BLEND_RGB_MULT** (3) → Multiplicative blending
        - **BLEND_RGB_MIN** (4) → Minimum value blending
        - **BLEND_RGB_MAX** (5) → Maximum value blending

        If `special_flags = 0`, no blending is applied (default behavior).

    **Returns**

    void

        The function modifies `target_pixels` **in-place**, so it does not return a new array.

    **Raises**

    ValueError

        - If `target_pixels` and `blend_pixels` have different shapes.
        - If `special_flags` is not in the valid range `[0-5]`.

    TypeError

        - If `target_pixels` or `blend_pixels` is not a **Cython memoryview** or **NumPy array**.

    **Notes**

    - This function is optimized for performance using **Cython (`cpdef inline`)**.
    - The blending operation modifies `target_pixels` **directly**, so pass a copy if
      you need to retain the original data.
    - Compatible with **Pygame's surface blending operations**.
    - Ensure both input arrays are of the same shape before calling this function.

    **Example Usage**

    .. code-block:: python

        import numpy as np

        # Create two random mapped pixel arrays
        target_pixels = np.random.randint(0, 255, (800, 600), dtype=np.uint32)
        blend_pixels = np.random.randint(0, 255, (800, 600), dtype=np.uint32)

        # Apply additive blending (BLEND_RGB_ADD = 1)
        blend_pixel_mapped_arrays(target_pixels, blend_pixels, special_flags=1)

        # The `target_pixels` array is now modified with blended values.
