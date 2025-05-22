Shader
======

:mod:`Shader.pyx`

=====================


.. currentmodule:: Shader


1. High-Performance Image Processing Library
--------------------------------------------

This library is a Cython-based module designed for high-performance image processing and manipulation.
By leveraging Cython’s ability to combine Python’s ease of use with C-like performance, it is optimized
for computationally intensive tasks such as real-time image processing, game development, computer vision,
and artistic effects.

2. Key Features
---------------

- **Color space conversions** (RGB, grayscale, HSL, HSV)
- **Image transformations** (mirroring, pixelation, resizing)
- **Visual effects** (distortions, filters, artistic effects)
- **Advanced image processing** (edge detection, sharpening, blending)
- **Physical simulations** (ripple effects, heat convection)
- **Performance optimizations** (multi-threading, memory-efficient operations)


3. Library Functional Overview
------------------------------

**Color Manipulation**

- **Color Conversions**: `bgr`, `brg`, `grey`, `sepia`, `hsl_effect`, `hsv_effect`
- **Brightness & Saturation**: `brightness`, `saturation`, `brightness_exclude`, `brightness_bpf`
- **Color Inversion**: `invert`

**Visual Effects**

- **Filters**: `median`, `sobel`, `bloom`, `posterize_surface`
- **Distortions**: `wave`, `swirl`, `fisheye`, `horizontal_glitch`, `horizontal_sglitch`
- **Artistic Effects**: `painting`, `cartoon`, `dithering`, `dithering_atkinson`

**Advanced Image Processing**

- **Edge Detection & Sharpening**: `sobel`, `sobel_fast`, `sharpen`, `sharpen32`
- **Blending & Compositing**: `blend`, `alpha_blending`, `alpha_blending_inplace`
- **Heatmap Effects**: `heatmap`, `predator_vision`

**Physical Simulations**

- **Ripple Effects**: `ripple`, `ripple_seabed`
- **Heat Convection**: `heatconvection`
- **Plasma Effects**: `plasma`, `plasma_config`

**Utility Functions**

- **Image Transformations**: `mirroring`, `pixelation`, `bilinear`
- **Color Mapping**: `wavelength2rgb`, `custom_map`
- **Optimized Functions**: `bgr_1d`, `grey_1d`, `invert1d`

**Special Effects**

- **Retro TV Scanlines**: `tv_scan`
- **Blood Effects**: `blood`
- **Dirt & Lens Distortion**: `dirt_lens`

**Performance & Optimization**

- **Memory Efficiency**: Optimized for `1D` and `3D` pixel arrays
- **Parallel Processing**: Uses `nogil` for multi-threaded execution


4. Target Applications
----------------------

This library is ideal for:
- **Game Development**: Real-time visual effects, distortions, blending
- **Computer Vision**: Preprocessing images for machine learning & analysis
- **Artistic Applications**: Creative effects for images and videos
- **Scientific Visualization**: Simulating physical phenomena (heat, ripples)


5. Summary
----------

This Cython-based library is a powerful toolkit for image processing,
combining high-performance optimizations with a vast range of visual effects
and transformations. It is particularly suited for real-time and high-performance
applications, making it an essential tool for game developers, computer vision
engineers, and creative professionals.


6. Cython list methods
----------------------

|

.. py:function:: bgr(object surface_)

   |

   Convert an image from RGB(A) to BGR(A) format (in-place).
   Converts the game display, image, or surface from RGB(A) to BGR(A) format.
   The alpha channel will be ignored in the process, but it is retained in case of a 32-bit surface.

   RGB is commonly used in image editing and display applications, where the order is red, green, and blue.
   On the other hand, BGR is often used in image processing applications, where the order is blue, green, and red.

   **Note**: This function operates in-place, meaning it modifies the original surface.

   **Parameters:**

   **surface_** (Pygame surface)
   Pygame surface (either display or image) with a compatible format
   (24-bit or 32-bit, with or without transparency/alpha channel).

   **Returns:**

   void; modifies the surface in-place.

   |

.. py:function:: bgr_copy(object surface_)

   |

   Convert an image format from RGB(A) to BGR(A) and return a new surface (copy).
   This function converts the pixel data of an input image from RGB(A) to BGR(A) format.
   The Alpha channel (if present) will be retained, but the order of the color channels
   is switched from RGB to BGR. This is useful when working with image processing libraries
   that expect the BGR format (such as OpenCV), while the RGB format is commonly used in
   display and image editing applications.

   .. code-block:: python

      new_surface = bgr_copy(surface)

   **Parameters:**

   **surface_** (Pygame Surface)
   A Pygame Surface object representing the image. The surface can be in 24-bit or
   32-bit format (with or without an alpha channel). The function assumes the image
   is in RGB(A) format and will convert it to BGR(A).

   **Returns:**

   A new Pygame Surface object with the converted BGR(A) pixel format.

|

.. py:function:: bgr_3d(unsigned char [:, :, :] rgb_array)

   |

   Convert an RGB(A) array (shape: w, h, n) with uint8 data type to BGR(A) format (inplace).
   This function directly processes a 3D array (such as an image or surface) from RGB(A) format
   to BGR(A) format. It assumes that the input array is in RGB or RGBA format, and it switches
   the red and blue channels to convert it to BGR or BGRA format respectively. The operation
   is done in place, modifying the original array.

   RGB is the common color order used in display and image editing applications, where the color
   channels are arranged as red, green, blue. On the other hand, BGR is often used in image
   processing applications, where the color channels are arranged as blue, green, red.

   .. code-block:: python

      bgr_3d(array)

   **Parameters:**

   **rgb_array** (numpy.ndarray)
   A 3D numpy array of shape (w, h, n), where `w` is the width, `h` is the height,
   and `n` is the number of channels (3 for RGB, 4 for RGBA). The array should have
   uint8 data type (values between 0 and 255), and contain pixel data in RGB(A) format.

   **Returns:**

   void
   This function modifies the input array in place and does not return any value.

   **Raises:**

   **ValueError**
   If the input array shape is not compatible with RGB(A) (i.e., it does not have
   the shape (w, h, 3) or (w, h, 4)).

   **TypeError**
   If the input array does not have the uint8 (unsigned byte) data type.

|

.. py:function:: bgr_1d(unsigned char [::1] rgb_array, bint format_32=False)

   |

   Convert a 1D array of uint8 data type from RGB(A) to BGR(A) format (inplace).
   This function processes a 1D array directly, converting the color channels from
   the RGB(A) format to the BGR(A) format. The conversion is done in place, modifying
   the original array. The `format_32` flag determines whether the input is in RGB
   (24-bit) or RGBA (32-bit) format.

   RGB is the standard color order used in many image editing and display applications,
   where the order of the color channels is red, green, and blue. In contrast, BGR is
   often used in image processing, where the color channels are arranged as blue,
   green, and red.

   .. code-block:: python

      bgr_1d(array)

   **Parameters:**

   **rgb_array** (numpy.ndarray or bytearray)
   A 1D array (or buffer) of pixel data in RGB(A) format, with uint8 data type
   (values between 0 and 255). The array length should be a multiple of 3 (for RGB)
   or 4 (for RGBA). The array contains the pixel color values that will be converted
   from RGB(A) to BGR(A) format.

   **format_32** (bool, optional)
   A boolean flag indicating the format of the input array.
   - `True` indicates the array is in 'RGB' (24-bit, 3 channels).
   - `False` (default) indicates the array is in 'RGBA' (32-bit, 4 channels).

   **Returns:**

   void
   The function modifies the input array in place and does not return any value.

   **Raises:**

   **TypeError**
   If the input array does not have a `uint8` data type.

|

.. py:function:: bgr_1d_cp(unsigned char [::1] rgb_array, bint format_32=False)

   |

   Convert a 1D array from RGB(A) to BGR(A) format and return a new copy.
   This function takes a 1D array (or memoryview slice) that contains pixel data in
   RGB(A) order and produces a new 1D array with the color channels reordered to
   BGR(A). This conversion is useful when interfacing with libraries or routines
   that expect pixels in BGR(A) format instead of the more common RGB(A) order.

   The function assumes that the input array is of type uint8, and its length should
   be a multiple of 3 (for RGB data) or 4 (for RGBA data). The `format_32` flag indicates
   whether the input contains 32-bit pixels (True for RGBA, False for RGB).

   .. code-block:: python

      new_bgr_array = bgr_1d_cp(rgb_array)

   **Parameters:**

   **bgr_array** (numpy.ndarray or memoryview slice, shape `(w,)`, dtype `uint8`)
   A 1D array or buffer containing pixel data in RGB(A) order.
   (For instance, if the image is RGB, the array length should be 3 * number_of_pixels.)

   **format_32** (bool, optional)
   A flag indicating the pixel format:
   - `False` (default): the input is assumed to be 24-bit (RGB, 3 channels).
   - `True`: the input is assumed to be 32-bit (RGBA, 4 channels).

   **Returns:**

   **numpy.ndarray**
   A new 1D array (uint8) with the pixel data converted to BGR(A) order.

|

.. py:function:: brg(object surface_)

   Convert a Pygame surface from RGB(A) to BRG(A) format in-place.
   This function modifies the given surface by swapping the red and green color channels,
   converting an image from RGB(A) order to BRG(A) order. The alpha channel, if present,
   is preserved but ignored during the conversion process.

   **Example Usage:**

   .. code-block:: python

      brg(surface)

   **Parameters:**

   **surface_** (pygame.Surface)
   A Pygame surface or display surface compatible with 24-bit (RGB) or 32-bit (RGBA) pixel formats.
   The function operates directly on the provided surface.

   **Returns:**

   **None**
   The function modifies the input surface in-place and does not return a new surface.

|

.. py:function:: brg_copy(object surface_)

   Convert an image from RGB(A) to BRG(A) format and return a new copy.
   This function swaps the red and green channels of an image while preserving
   the blue channel. It creates and returns a new surface with the modified
   color format, leaving the original surface unchanged.

   **Example Usage:**

   .. code-block:: python

      brg_surface = brg_copy(surface)

   **Parameters:**

   **surface_** (pygame.Surface)
   A Pygame surface or display surface in 24-bit (RGB) or 32-bit (RGBA) format.
   The alpha channel (if present) will be ignored.

   **Returns:**

   **pygame.Surface**
   A new surface with the color channels converted to BRG format.

|

.. py:function:: brg_3d(unsigned char [:, :, :] rgb_array)

   Convert a 3D array from RGB(A) to BRG(A) format (inplace).
   This function swaps the red and green channels while preserving the blue
   and alpha channels (if present). It modifies the input array directly
   without creating a copy.

   **Example Usage:**

   .. code-block:: python

      brg_3d(rgb_array)

   **Parameters:**

   **rgb_array** (numpy.ndarray, shape (w, h, n), dtype uint8)
   A 3D array representing an image, where pixel values range from 0 to 255.
   The last dimension (n) must be 3 (RGB) or 4 (RGBA).

   **Returns:**

   **None**
   The function operates inplace and does not return a new array.

|

.. py:function:: brg_1d(unsigned char [::1] rgb_array, bint format_32=False)

   Converts a 1D array of RGB(A) pixel data to BRG(A) format in-place.
   This function swaps the red and blue channels of an input array or buffer
   representing RGB or RGBA pixel data. It works on both 24-bit (RGB) and
   32-bit (RGBA) formats.

   **Example Usage:**

   .. code-block:: python

      brg_1d(rgb_array)  # Converts an RGB(A) buffer to BRG(A)

   **Parameters:**

   **rgb_array** (numpy.ndarray or bytearray)
   A 1D array of shape (w,) containing uint8 pixel data (values 0-255).
   Can be either a NumPy array or a raw byte buffer.

   **format_32** (bool, optional, default = False)
   - True: Treats the input as an RGB (24-bit) buffer.
   - False: Treats the input as an RGBA (32-bit) buffer.

   **Returns:**

   **None**
   The operation is performed in-place, modifying `rgb_array` directly.

|

.. py:function:: brg_1d_cp(const unsigned char [::1] bgr_array, bint format_32=False)

   Converts a 1D array of uint8 BGR(A) pixel data to BRG(A) format and returns a new array.
   Unlike the in-place version (`brg_1d`), this function creates and returns a
   new array with the red and blue channels swapped.

   BRG stands for Blue, Red, Green.

   **Example Usage:**

   .. code-block:: python

      bgr_array = brg_1d_cp(bgr_array)  # Converts an BGR(A) buffer to BRG(A)

   **Parameters:**

   **bgr_array** (numpy.ndarray or bytearray)
   A 1D array of shape (w,) containing uint8 pixel data (values 0-255).
   Can be either a NumPy array or a raw byte buffer.

   **format_32** (bool, optional, default = False)
   - True: Treats the input as a BGR (24-bit) buffer.
   - False: Treats the input as a BGRA (32-bit) buffer.

   **Returns:**

   **numpy.ndarray**
   A new array of shape (w,) with the BRG(A) pixel format (copied).

|

.. py:function:: grey(object surface_)

   Convert an image to grayscale while preserving luminosity (in-place).
   A grayscale image has a single channel representing pixel intensity or brightness,
   where pixel values range from 0 (black) to 255 (white). This function computes the
   grayscale values based on luminosity, preserving perceived brightness from the original color image.

   **Example Usage:**

   .. code-block:: python

      grey(surface)

   **Parameters:**

   **surface_** (pygame.Surface)
   A Pygame surface or display surface compatible object, with an image in
   24-bit or 32-bit format. The surface may include transparency or an alpha
   channel.

   **Returns:**

   **None**
   The function modifies the input surface in place and does not return a new surface.

|

.. py:function:: grey_copy(object surface_)

   Convert an image to grayscale while preserving luminosity and return a new surface.
   A grayscale image has a single channel representing pixel intensity or brightness,
   with pixel values typically ranging from 0 (black) to 255 (white). This function
   converts the original image to grayscale based on luminosity, preserving perceived
   brightness from the color image. The alpha channel is preserved in images with
   transparency (RGBA format), but it is not altered.

   **Example Usage:**

   .. code-block:: python

      im = grey_copy(surface)

   **Parameters:**

   **surface_** (pygame.Surface)
   A Pygame surface or display surface compatible object, with an image in
   24-bit or 32-bit format. The surface may include transparency (alpha channel).

   **Returns:**

   **pygame.Surface**
   A new surface object containing the grayscale image. The original surface remains unchanged.

|

.. py:function:: grey_2d(surface_)

   Convert an image into a 2D grayscale array.
   A grayscale image has a single channel representing pixel intensity or brightness,
   with pixel values typically ranging from 0 (black) to 255 (white). This function
   converts the input image to grayscale based on luminosity, preserving the intensity
   from the original color image. The alpha channel, if present, will be ignored in the
   output.

   **Example Usage:**

   .. code-block:: python

      gray = grey_2d(surface)

   **Parameters:**

   **surface_** (pygame.Surface)
   A Pygame surface or display surface-compatible object, with an image in
   24-bit or 32-bit format. The surface may include transparency (alpha channel),
   which will be ignored during the conversion.

   **Returns:**

   **numpy.ndarray**
   A 2D NumPy array containing the grayscale image data. The array will have
   shape (w, h) and dtype uint8, where each value represents pixel intensity.

|

.. py:function:: grey_3d(rgb_array)

   Convert a 3D array (w, h, n) in RGB(A) format to grayscale (with alpha) in place.
   This function processes a 3D array directly, converting it to grayscale based on
   luminosity while preserving the alpha channel (if present). The pixel values in
   the resulting grayscale array represent intensity or brightness, ranging from
   0 (black) to 255 (white).

   **Example Usage:**

   .. code-block:: python

      # For a 24-bit image/surface
      grey_3d(pixels3d(im))

      # For a 32-bit image with alpha channel
      grey_3d(numpy.asarray(im.get_view('0'), dtype=uint8).reshape(w, h, 4))

   **Parameters:**

   **rgb_array** (numpy.ndarray)
   A 3D NumPy array of shape (w, h, n) with dtype uint8, containing pixel data
   in RGB(A) format. The values in the array range from 0 to 255. For 32-bit arrays
   (RGBA), the alpha channel will be preserved but ignored in the grayscale conversion.

   **Returns:**

   **None**
   This function modifies the input array in place and does not return a new array.

|

.. py:function:: grey_1d(rgb_array, format_32=False)

   Convert a 1D array of uint8 data (RGB(A)) to grayscale (with alpha) in place.
   A grayscale image has a single channel representing pixel intensity or brightness,
   with pixel values typically ranging from 0 (black) to 255 (white). This function
   converts the input RGB(A) array to grayscale while preserving the alpha channel
   (if present). The conversion is performed in place, modifying the original array.

   **Example Usage:**

   .. code-block:: python

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

   **Parameters:**

   **rgb_array** (numpy.ndarray or bytearray)
   A 1D array or buffer containing pixel data in RGB(A) format, with dtype uint8
   (unsigned char values ranging from 0 to 255).

   **format_32** (bool, optional)
   If `True`, the function assumes the input is a 32-bit buffer (RGBA).
   If `False`, the function assumes a 24-bit buffer (RGB).

   **Returns:**

   **None**
   The function modifies the input array in place and does not return a new array.

|

.. py:function:: grey_1d_cp(bgr_array, format_32=False)

   Convert a 1D array of uint8 BGR(A) data to grayscale (with alpha channel) and return a copy.
   This function checks that the input array has the expected uint8 data type, and then converts
   the pixel data from BGR or BGRA format to grayscale. The conversion preserves the alpha channel
   (if present). It returns a new 1D NumPy array in grayscale format.

   The grayscale conversion is based on the luminosity formula, which calculates the
   brightness based on the red, green, and blue channels.

   **Parameters:**

   **bgr_array** (numpy.ndarray)
   A 1D array containing pixel data in BGR(A) format, with dtype uint8. The pixel values
   should range from 0 to 255. If the array represents a BGRA image, the alpha channel
   will be preserved.

   **format_32** (bool, optional)
   If `True`, the input array is assumed to be in BGRA (32-bit) format.
   If `False`, the array is assumed to be in BGR (24-bit) format.

   **Returns:**

   **numpy.ndarray**
   A new 1D NumPy array of shape (w,) with dtype uint8, containing the grayscale
   pixel data. If the input was in BGRA format, the alpha channel is preserved in the output.

   **Raises:**

   **TypeError**
   If the input array does not have dtype uint8, a `TypeError` will be raised.

|

.. py:function:: sepia(surface_)

   Apply a sepia tone filter to an image, surface, or video game graphics (inplace).
   Sepia toning is a technique used in photography and imaging where the image is given
   a reddish-brown tint, simulating the warm tones of old photographs. It creates a
   softer, dreamier aesthetic compared to standard grayscale, adding depth and a vintage look.

   This function transforms the provided surface into an equivalent sepia-toned model,
   adjusting the pixel colors accordingly while preserving the original structure. The
   transformation is applied directly to the surface (inplace), and no new surface is returned.

   **Example:**

   .. code-block:: python

       sepia(surface)

|

.. py:function:: sepia_copy(surface_)

   Sepia model (New Surface)

   Transform your video game/image or surface into an equivalent sepia model.

   While traditional black-and-white photographs use a standard grayscale to create
   highlights and shadows, sepia-toned photos use a reddish-brown tone to create that spectrum.
   “Sepia is a softer manipulation of light.” This gives them a softer, dreamier aesthetic.

   **Example:**

   .. code-block:: python

       im = sepia_copy(surface)

|

.. py:function:: sepia_3d(rgb_array)

   Apply a sepia tone filter to a 3D RGB(A) image array (inplace).

   This function processes a 3D NumPy array representing pixel data in RGB(A) format
   and applies a sepia filter. The sepia effect is achieved by adjusting the red,
   green, and blue channels according to predefined coefficients, creating a warm-toned,
   vintage effect. The function modifies the input array directly and does not return anything.

   Only arrays with shapes (w, h, 3) for RGB or (w, h, 4) for RGBA are supported.

   **Parameters:**

   **rgb_array** (numpy.ndarray)
   A 3D array with shape (w, h, 3) for RGB or (w, h, 4) for RGBA pixel data, where
   `w` is the width, `h` is the height, and `3` or `4` corresponds to the RGB or RGBA channels.
   The array must have dtype uint8 (unsigned char) with pixel values ranging from 0 to 255.

   **Raises:**

   **ValueError**
   If the input array does not have the expected shape (w, h, 3) or (w, h, 4).

   **TypeError**
   If the input array does not have dtype uint8.

   **Notes:**

   - This function modifies the input `rgb_array` directly (inplace).
   - If the input is in RGBA format, the alpha channel is preserved.
   - The sepia effect is applied by adjusting the luminosity of the RGB channels using a set of coefficients.

   **Example:**

   .. code-block:: python

      # Assuming rgb_array is a 3D NumPy array with shape (w, h, 3) for RGB or (w, h, 4) for RGBA:
      sepia_3d(rgb_array)

|

.. py:function:: sepia_1d(rgb_array, format_32=False)

   Convert a 1D array of uint8 data (RGB(A)) to sepia equivalent (inplace).

   While traditional black-and-white photographs use a standard grayscale to create
   highlights and shadows, sepia-toned photos use a reddish-brown tone to create that spectrum.
   "Sepia is a softer manipulation of light," creating a softer, dreamier aesthetic.

   **Example Usage:**

   For a 24-bit image (RGB):

   .. code-block:: python

      im = pygame.image.load("../Assets/px.png")
      w, h = im.get_width(), im.get_height()
      c = numpy.ndarray(shape=(w*h*3), buffer=im.get_view('0'), dtype=uint8)
      sepia_1d(c, False)

   For a 32-bit image (RGBA):

   .. code-block:: python

      im = pygame.image.load("../Assets/px.png")
      w, h = im.get_width(), im.get_height()
      sepia_1d(im.get_view('0'), True)

   or

   .. code-block:: python

      im = pygame.image.load("../Assets/px.png")
      w, h = im.get_width(), im.get_height()
      sepia_1d(numpy.ndarray(shape=(w*h*4), buffer=im.get_view('1'), dtype=uint8), True)

   **Parameters:**

   **rgb_array** (numpy.ndarray)
   A 1D array of shape (w,) containing uint8 pixel data (values ranging from 0 to 255) in RGB(A) format.
   Can be either a NumPy array or a raw byte buffer.

   **format_32** (bool, optional, default=False)
   If `True`, the input is treated as a 24-bit RGB buffer.
   If `False`, the input is treated as a 32-bit RGBA buffer.

   **Returns:**

   None
   The function modifies the input array in place and does not return a new array.

|

.. py:function:: median(surface_, kernel_size_=2, fast_=True, reduce_factor_=1)

   Apply a median filter to a surface (inplace).

   The median filter is a non-linear image filtering technique commonly used for
   removing noise from an image or signal. It works by replacing each pixel value
   with the median of the pixel values in a neighborhood defined by the kernel size.
   This technique is widely used for noise reduction in digital image processing,
   especially for preserving edges while removing noise.

   The strength of the effect is controlled by the `kernel_size` parameter, with
   larger kernel sizes producing stronger filtering effects. However, larger kernel
   sizes may also slow down the process significantly.

   **Note:** This filter is not suitable for real-time rendering in games or animations
   due to its computational cost.

   **Example Usage:**

   For a 24-bit image:

   .. code-block:: python

      im = pygame.image.load("../Assets/background.jpg")
      im = scale(im, (800, 600))
      median(im, fast=True)

   For a 32-bit image:

   .. code-block:: python

      im = pygame.image.load("../Assets/px.png").convert_alpha()
      im = scale(im, (800, 600))
      median(im, fast=False)

   **Parameters:**

   **surface_** (pygame.Surface)
   A Pygame surface object (24-bit or 32-bit) containing the image to be processed.
   It can have or lack an alpha channel for transparency.

   **kernel_size_** (unsigned short int, optional, default=2)
   The size of the kernel or neighborhood of pixels used for the median calculation.
   Increasing the kernel size improves the filter effect but decreases performance.

   **fast_** (bool, optional, default=True)
   A flag to enable fast calculation. If `True`, the filter uses a smaller texture
   size to improve performance, which may reduce the quality based on the `reduce_factor_` argument.

   **reduce_factor_** (unsigned short int, optional, default=1)
   A factor to reduce the size of the surface before processing. A value of 1 divides
   the surface by 2, and a value of 2 reduces the surface by 4. Values larger than 2
   may degrade the image quality. Has no effect if `fast_` is `False`.

   **Returns:**

   None
   The function modifies the surface directly (in-place) and does not return a value.

   **Raises:**

   **ValueError**
   If `surface_` is not a valid `pygame.Surface` or if the `kernel_size_` or
   `reduce_factor_` are out of valid ranges.

|

.. py:function:: painting(surface_)

   Apply a painting effect (in-place) to a Pygame surface.

   This function transforms an image to resemble a hand-painted artistic style.
   It achieves this by using a fast median filter algorithm (`median_fast`), which
   smooths pixel values while maintaining edge details, giving the image a brushstroke-like appearance.

   **Note:**
   - This method **modifies the input surface in place**.
   - It **is not optimized for real-time rendering** and is intended for offline processing of images.

   **Example Usage:**

   For a 24-bit or 32-bit image:

   .. code-block:: python

      im = pygame.image.load("../Assets/background.jpg").convert(24)
      im = scale(im, (800, 600))
      painting(im)

   **Parameters:**

   **surface_** (pygame.Surface)
   A Pygame-compatible surface (24-bit or 32-bit), with or without per-pixel transparency (alpha channel).

   **Returns:**

   None
   The function modifies the input surface directly.

   **Raises:**

   ValueError
   If `surface_` is not a valid `pygame.Surface`.

|

.. py:function:: pixels(surface_)

   Apply a pixelation effect to an image (INPLACE).

   Pixelation is a visual effect where an image is displayed at a low resolution,
   making individual pixels (small, single-colored square elements) clearly visible.
   This technique is often used in digital graphics to create artistic styles,
   censor parts of an image, or simulate retro, low-resolution video game graphics.

   This function modifies the input surface in-place, reducing fine details and
   emphasizing blocky pixel structures.

   **Example Usage:**
   

   For both 24-bit and 32-bit images:

   .. code-block:: python

      import pygame
      im = pygame.image.load("../Assets/background.jpg").convert()
      im = pygame.transform.scale(im, (800, 600))  # Rescale image
      pixels(im)  # Apply pixelation effect

   **Parameters:**

   **surface_** (pygame.Surface)
   A Pygame-compatible surface (24-bit or 32-bit image) with or without per-pixel transparency (alpha channel).

   **Returns:**

   None
   The input surface is modified directly (in-place).

|

.. py:function:: median_grayscale(surface_, kernel_size_=2)

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

   **Example Usage:**
   

   Apply median filter to a 24-bit image:

   .. code-block:: python

      im = pygame.image.load("../Assets/background.jpg")
      median_grayscale(im)

   Apply median filter to a 32-bit image (with alpha channel):

   .. code-block:: python

      im = pygame.image.load("../Assets/px.png").convert_alpha()
      median_grayscale(im)

   **Parameters:**

   **surface_** (pygame.Surface)
   A Pygame-compatible surface (24-bit or 32-bit, with or without alpha).

   **kernel_size_** (int, optional, default=2)
   The size of the kernel (neighborhood of pixels considered for filtering).
   Must be greater than 0. Increasing the kernel size enhances the filtering
   effect but significantly impacts performance.

   **Returns:**

   None
   The function modifies the input surface directly (in-place).

|

.. py:function:: posterize_surface(surface_, color_=8)

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

   **Example Usage:**
   

   Reduce the number of colors in a Pygame surface to 8:

   .. code-block:: python

      color_reduction(surface, 8)

   **Parameters:**

   **surface_** (pygame.Surface)
   A Pygame-compatible surface (24-bit or 32-bit, with or without alpha).

   **color_** (int, optional, default=8)
   The number of colors to reduce the image to.
   Must be greater than 0. Lower values produce a more dramatic effect.

   **Returns:**

   None
   The function modifies the input surface directly (in-place).

|

.. py:function:: sobel(surface_, threshold_=64)

   Apply Sobel Edge Detection (Inplace)

   The Sobel operator, also known as the Sobel-Feldman operator, is commonly used in image processing
   for edge detection. It highlights edges in an image by emphasizing areas with high intensity
   gradients.

   This function applies the Sobel edge detection to a Pygame surface, transforming the surface to emphasize
   its edges based on the gradient of pixel intensities.

   The input surface must be in grayscale (24 - 32 bit) for best results. If the surface is not in grayscale,
   only the red channel will be used for the edge detection.

   **Example usage:**
   
   Apply Sobel edge detection with a threshold of 64:

   .. code-block:: python

      sobel(surface, 64)

   **Parameters:**

   **surface_** (pygame.Surface)
   A Pygame.Surface object (24 - 32 bit depth), which may or may not have an alpha channel.
   The surface should be greyscaled, although non-greyscale images will use the red channel
   for edge detection.

   **threshold_** (int, optional, default=64)
   An integer representing the threshold for detecting edges.
   The threshold determines the sensitivity of edge detection, with higher values requiring
   stronger gradients to be considered an edge.

   **Returns:**

   None
   The function modifies the input surface in place, updating it with the Sobel edge detection result.

|

.. py:function:: sobel_1d(w, h, bgr_array, tmp_array=None, threshold=64, format_32=False, greyscale=False)

   Apply 1D Sobel Edge Detection (Inplace)

   This function applies the 1D Sobel operator to a given image (or surface), emphasizing
   the edges in the horizontal or vertical direction based on the gradient of pixel intensities.
   It modifies the input buffer(s) in place.

   **Example usage:**
   
   For 24-bit image:

   .. code-block:: python

      image = pygame.image.load('../Assets/px.png').convert(24)
      image = pygame.transform.smoothscale(image, (800, 600))
      grey(image)
      image_copy = image.copy()
      sobel_1d(800, 600, image.get_buffer(), image_copy.get_buffer(), threshold=25)

   For 32-bit image (with alpha):

   .. code-block:: python

      image = pygame.image.load('../Assets/px.png').convert_alpha()
      image = pygame.transform.smoothscale(image, (800, 600))
      image_copy = image.copy()
      sobel_1d(800, 600, image.get_buffer(), image_copy.get_buffer(), threshold=25, format_32=True)

   **Parameters:**

   **w** (int)
   Width of the Pygame surface that the source array (`bgr_array`) is referencing.

   **h** (int)
   Height of the Pygame surface that the source array (`bgr_array`) is referencing.

   **bgr_array** (numpy.ndarray)
   A 1D C-buffer of type uint8 containing pixel data in BGR format. If the image is not
   greyscale, the algorithm processes all three RGB channels. If greyscale is enabled, only
   the blue channel is used for edge detection.

   **tmp_array** (numpy.ndarray, optional)
   A 1D C-buffer of type uint8 containing pixel data in BGR format. It is a copy of the
   source `bgr_array`. Both `bgr_array` and `tmp_array` must have the same size and
   data format. This is used as a temporary buffer during processing.

   **threshold** (int, optional, default=64)
   The threshold for edge detection. Pixels with gradient values above this threshold will
   be considered edges.

   **format_32** (bool, optional, default=False)
   If `True`, the input array is assumed to be in 32-bit BGRA format. If `False`, the input
   array is assumed to be in 24-bit BGR format.

   **greyscale** (bool, optional, default=False)
   If `True`, the algorithm processes only the blue channel for edge detection, which can
   simplify the computation for greyscale images. If `False`, all three RGB channels are
   used in the Sobel operator.

   **Returns:**

   None
   The function modifies the input buffers (`bgr_array` and `tmp_array`) in place.

|

.. py:function:: sobel_fast(surface_, threshold_=64, factor_=1)

   Fast sobel (inplace)

   The Sobel operator, sometimes called the Sobel–Feldman operator or Sobel filter,
   is used in image processing and computer vision, particularly within edge detection
   algorithms where it creates an image emphasising edges.

   This function transforms the game display or a Pygame surface into a Sobel equivalent model.
   It is faster than the standard Sobel operator (`sobel_inplace_c`) as it down-scales the
   array containing all the pixels and applies the Sobel algorithm to a smaller sample. After
   processing, the array is re-scaled to its original dimensions. While this method is theoretically
   faster, down-scaling and up-scaling an array results in a decrease in overall image quality
   (e.g., jagged lines, no anti-aliasing).

   **Compatible with 24-bit and 32-bit surfaces**, with or without an alpha layer.
   The surface must be greyscale, but non-greyscale images will also work; however, only
   the red channel will be used for Sobel edge detection.

   **Example usage:**
   

   .. code-block:: python

      sobel_fast(surface, 64, factor=1)

   **Parameters:**

   **surface_** (pygame.Surface)
   A Pygame-compatible surface (24-bit or 32-bit) with or without an alpha layer.

   **threshold_** (int, optional, default=64)
   An integer representing the threshold for edge detection. Pixels with gradient values
   above this threshold are considered edges.

   **factor_** (unsigned short int, optional, default=1)
   An integer that controls the down-scaling factor of the image. A value of 1 means no down-scaling,
   and larger values reduce the size of the image before processing.

   **Returns:**

   None
   This function modifies the input surface directly (in-place).

|

.. py:function:: invert(surface_)

   Invert pixels (inplace)

   Inverting an image means inverting the pixel values. Images are represented using RGB
   or Red-Green-Blue values, where each channel can take an integer value between 0 and 255
   (both inclusive). For example, red is represented as (255, 0, 0), white as (255, 255, 255),
   black as (0, 0, 0), and so on. Inverting an image means reversing the colors. For instance,
   the inverted color for red will be (0, 255, 255), where 255 becomes 0 and 0 becomes 255.
   Effectively, inverting an image is equivalent to subtracting the original RGB values from 255.

   **Compatible with 24-bit and 32-bit surfaces**, with or without an alpha channel.

   **Example usage:**
   

   .. code-block:: python

      invert(surface)

   **Parameters:**

   **surface_** (pygame.Surface)
   A Pygame-compatible surface (24-bit or 32-bit) with or without an alpha channel.

   **Returns:**

   None
   This function modifies the input surface directly (in-place).

|

.. py:function:: invert_copy(surface_)

   Invert pixels and return a copy

   Inverting an image means inverting the pixel values. Images are represented using RGB
   or Red-Green-Blue values, where each channel can take an integer value between 0 and 255
   (both inclusive). For example, red is represented as (255, 0, 0), white as (255, 255, 255),
   black as (0, 0, 0), and so on. Inverting an image means reversing the colors. For instance,
   the inverted color for red will be (0, 255, 255), where 255 becomes 0 and 0 becomes 255.
   Effectively, inverting an image is equivalent to subtracting the original RGB values from 255.

   **Compatible with 24-bit and 32-bit surfaces**, with or without an alpha channel.

   **Example usage:**
   

   .. code-block:: python

      inv = invert_copy(surface)

   **Parameters:**

   **surface_** (pygame.Surface)
   A Pygame-compatible surface (24-bit or 32-bit) with or without an alpha channel.

   **Returns:**

   (pygame.Surface)
   A new surface with inverted pixels, leaving the original surface unchanged.

|

.. py:function:: invert3d(rgb_array)

   Invert 3D array pixels (inplace)

   Inverting an image means inverting the pixel values. Images are represented using RGB
   or Red-Green-Blue values, where each channel can take an integer value between 0 and 255
   (both inclusive). For example, red is represented as (255, 0, 0), white as (255, 255, 255),
   black as (0, 0, 0), and so on. Inverting an image means reversing the colors. For instance,
   the inverted color for red will be (0, 255, 255), where 255 becomes 0 and 0 becomes 255.
   Effectively, inverting an image is equivalent to subtracting the original RGB values from 255.

   **Example usage:**
   

   .. code-block:: python

      # 24-bit image
      image = pygame.image.load('../Assets/px.png').convert(24)
      invert3d(array3d)

      # 32-bit image (with alpha channel)
      image = pygame.image.load('../Assets/px.png').convert_alpha()
      invert3d(array3d)

   **Parameters:**

   **rgb_array** (numpy.ndarray)
   A 3D array with shape (w, h, n), where `w` and `h` represent the width and height of the image,
   and `n` is the number of channels (e.g., 3 for RGB, 4 for RGBA). The array contains pixel data in
   RGB(A) or similar formats, such as BGR, BGRA.

   **Returns:**

   None
   The function modifies the input array in place.

|

.. py:function:: invert1d(rgb_array, format_32=False)

   Invert directly a C-buffer pixel values

   Inverting an image means inverting the pixel values. Images are represented using RGB
   or Red-Green-Blue values, where each channel can take an integer value between 0 and 255
   (both inclusive). For example, red is represented as (255, 0, 0), white as (255, 255, 255),
   black as (0, 0, 0), and so on. Inverting an image means reversing the colors. For instance,
   the inverted color for red will be (0, 255, 255), where 255 becomes 0 and 0 becomes 255.
   Effectively, inverting an image is equivalent to subtracting the original RGB values from 255.

   **Example usage:**
   

   .. code-block:: python

      # 24-bit image
      image = pygame.image.load('../Assets/px.png').convert(24)
      invert1d(image.get_buffer(), False)

      # 32-bit image (with alpha channel)
      image = pygame.image.load('../Assets/px.png').convert_alpha()
      invert1d(image.get_buffer(), True)

   **Parameters:**

   **rgb_array** (numpy.ndarray)
   A 1D array (C-buffer) with uint8 data type containing pixel data in RGB(A) format. The method
   works with other formats such as BGR, BGRA, etc.

   **format_32** (bool, optional, default=False)
   If `True`, the array is assumed to be in RGBA format (32-bit). If `False`, the array is assumed
   to be in RGB format (24-bit).

   **Returns:**

   None
   The function modifies the input array in place.

|

.. py:function:: invert1d_cp(rgb_array, format_32=False)

   Invert directly a C-buffer pixel values (return a copy)

   Inverting an image means inverting the pixel values. Images are represented using RGB
   or Red-Green-Blue values, where each channel can take an integer value between 0 and 255
   (both inclusive). For example, red is represented as (255, 0, 0), white as (255, 255, 255),
   black as (0, 0, 0), and so on. Inverting an image means reversing the colors. For instance,
   the inverted color for red will be (0, 255, 255), where 255 becomes 0 and 0 becomes 255.
   Effectively, inverting an image is equivalent to subtracting the original RGB values from 255.

   This function returns a copy of the input array with the pixel values inverted.

   **Example usage:**
   

   .. code-block:: python

      # 24-bit image
      image = pygame.image.load('../Assets/px.png').convert(24)
      arr3d = invert1d_cp(image.get_buffer(), False)
      image = pygame.image.frombuffer(arr3d, (WIDTH, HEIGHT), "BGR")

      # 32-bit image (with alpha channel)
      image = pygame.image.load('../Assets/px.png').convert_alpha()
      arr3d = invert1d_cp(image.get_buffer(), True)
      image = pygame.image.frombuffer(arr3d, (WIDTH, HEIGHT), "BGRA")

   **Parameters:**

   **rgb_array** (numpy.ndarray)
   A 1D array (C-buffer) with uint8 data type containing pixel data in RGB(A) format. The method
   works with other formats such as BGR, BGRA, etc.

   **format_32** (bool, optional, default=False)
   If `True`, the array is assumed to be in RGBA format (32-bit). If `False`, the array is assumed
   to be in RGB format (24-bit).

   **Returns:**

   numpy.ndarray
   A 1D array with uint8 data type, representing a copy of the input buffer with inverted pixel values.

|

.. py:function:: hsl_effect(surface_, shift)

   Apply Hue Rotation to an Image (HSL Color Space)

   This function directly modifies the hue of a Pygame surface using the HSL (Hue, Saturation,
   Lightness) color model. Hue rotation shifts the colors of the surface in a way that corresponds
   to a rotation on the color wheel, allowing you to alter the overall color tone.

   The surface must be compatible with 24-bit or 32-bit color depth, with or without an alpha layer.
   If the `shift` value is 0.0, the surface remains unchanged.

   The hue shift value must be within the range [0.0, 1.0], where 0.0 represents no rotation, and 1.0
   represents a 360-degree rotation.

   **Example usage:**
   

   .. code-block:: python

      hsl_effect(surface, 0.2)  # Apply a 72-degree hue shift to the surface.

   **Parameters:**

   **surface_** (pygame.Surface)
   A Pygame surface, compatible with 24-bit or 32-bit color formats (with or without alpha).

   **shift** (float)
   A float value in the range [0.0, 1.0], where 0.0 corresponds to no hue shift,
   and 1.0 corresponds to a full 360-degree rotation of the hue.

   **Returns:**

   None
   The function modifies the input `surface_` in place, applying the hue shift effect.

|

.. py:function:: hsl3d(rgb_array, shift)

   Apply Hue Rotation to a 3D Array (HSL Color Space)

   This function applies hue rotation to a 3D numpy array representing an image in the HSL (Hue, Saturation,
   Lightness) color space. The hue shift is applied directly to the array, modifying the color tone of the image.

   The array must be in the shape (w, h, n), where `w` is the width, `h` is the height, and `n` is the number
   of color channels (3 for RGB or 4 for RGBA, etc.). The data type should be uint8.

   The `shift` value must be within the range [0.0, 1.0], where 0.0 represents no hue rotation, and 1.0 represents
   a full 360-degree hue rotation.

   **Example usage:**
   

   .. code-block:: python

      # For 24-bit RGB image
      image = pygame.image.load('../Assets/px.png').convert(24)
      array3d = pygame.surfarray.pixels3d(image)
      hsl3d(array3d, 0.2)  # Apply a 72-degree hue shift

      # For 32-bit RGBA image
      image = pygame.image.load('../Assets/px.png').convert_alpha()
      array3d = pygame.surfarray.pixels3d(image)
      hsl3d(array3d, 0.2)  # Apply a 72-degree hue shift

   **Parameters:**

   **rgb_array** (numpy.ndarray)
   A 3D numpy array with shape (w, h, n) containing RGB or RGBA pixel data.
   The array can also represent other formats such as BGR, BGRA.

   **shift** (float)
   A float value in the range [0.0, 1.0], representing the hue rotation.
   A value of 0.0 means no rotation, while 1.0 corresponds to a full 360-degree hue shift.

   **Returns:**

   None
   The function modifies the input `rgb_array` in place, applying the hue shift.

|

.. py:function:: hsl1d(bgr_array, shift, format_32=False)

   Apply Hue Rotation to a C-buffer (HSL Color Space)

   This function performs hue rotation on a C-buffer (1D array) containing pixel data in RGB(A) format
   using the HSL (Hue, Saturation, Lightness) color model. It modifies the array in place.

   The function supports pixel formats like RGB, BGR, RGBA, and BGRA, adjusting the hue of each pixel
   according to the specified `shift`. The `shift` value should be in the range [0.0, 1.0], which
   corresponds to a rotation of 0.0 to 360.0 degrees on the hue color wheel.

   **Example usage:**
   

   .. code-block:: python

      # For 24-bit RGB image
      image = pygame.image.load('../Assets/px.png').convert(24)
      hsl1d(image.get_buffer(), 0.2)

      # For 32-bit RGBA image
      image = pygame.image.load('../Assets/px.png').convert_alpha()
      hsl1d(image.get_buffer(), 0.2, format_32=True)

   **Parameters:**

   **bgr_array** (numpy.ndarray)
   A 1D numpy array (C-buffer) of type uint8 containing pixel data in RGB(A) format.
   The array can also represent other formats such as BGR, BGRA.

   **shift** (float)
   A float value in the range [0.0, 1.0] representing the hue rotation.
   A value of 0.0 means no rotation, while 1.0 corresponds to a full 360-degree rotation.

   **format_32** (bool, optional, default=False)
   If `True`, the input array is assumed to be in 32-bit RGBA format.
   If `False`, the array is assumed to be in 24-bit RGB format.

   **Returns:**

   None
   The function modifies the input `bgr_array` in place by applying the hue shift.

|

.. py:function:: hsl1d_cp(bgr_array, shift, format_32=False)

   Rotate hue (HSL) directly to a C-buffer (return a copy)

   This function performs a hue rotation in the HSL (Hue, Saturation, Lightness) color space on a C-buffer
   (1D array) containing pixel data in RGB(A) format, returning a modified copy of the array.

   The method works with pixel formats like RGB, BGR, RGBA, and BGRA. The hue shift (`shift`) value should
   be in the range [0.0, 1.0], corresponding to a rotation of 0.0 to 360.0 degrees.

   **Example usage:**
   

   .. code-block:: python

      # For 24-bit RGB image
      image = pygame.image.load('../Assets/px.png').convert(24)
      arr = hsl1d_cp(image.get_buffer(), 0.2, format_32=False)
      image = pygame.image.frombuffer(arr, (WIDTH, HEIGHT), "BGR")

      # For 32-bit RGBA image
      image = pygame.image.load('../Assets/px.png').convert_alpha()
      arr = hsl1d_cp(image.get_buffer(), 0.2, format_32=True)
      image = pygame.image.frombuffer(arr, (WIDTH, HEIGHT), "BGRA")

   **Parameters:**

   **bgr_array** (numpy.ndarray)
   A 1D numpy array (C-buffer) of type uint8 containing pixel data in RGB(A) format.
   The array can also represent other formats such as BGR, BGRA.

   **shift** (float)
   A float value in the range [0.0, 1.0], representing the hue rotation.
   A value of 0.0 means no rotation, while 1.0 corresponds to a full 360-degree rotation.

   **format_32** (bool, optional, default=False)
   If `True`, the input array is assumed to be in 32-bit RGBA format.
   If `False`, the array is assumed to be in 24-bit RGB format.

   **Returns:**

   **numpy.ndarray**
   A 1D numpy array of type uint8 containing the pixel data with the rotated hue.

|

.. py:function:: hsv_effect(surface_, shift)

   Apply Hue Rotation to a Surface (HSV Color Space)

   This function applies a hue rotation to a Pygame surface using the HSV (Hue, Saturation, Value) color model.
   It modifies the surface in place, rotating the hue of the colors on the surface based on the specified shift.

   The surface must be compatible with 24-bit or 32-bit color formats, with or without an alpha channel.
   The hue shift is specified as a float value in the range [0.0, 1.0], where 0.0 corresponds to no hue change,
   and 1.0 represents a full 360-degree hue rotation.

   **Example usage:**
   

   .. code-block:: python

      surface = pygame.image.load('../Assets/px.png').convert_alpha()
      hsv_effect(surface, 0.2)  # Rotate the hue by 72 degrees (0.2 * 360)

   **Parameters:**

   **surface_** (pygame.Surface)
   A Pygame surface, compatible with 24-bit or 32-bit color formats (with or without alpha).

   **shift** (float)
   A float value in the range [0.0, 1.0], specifying the hue rotation.
   A value of 0.0 means no rotation, while 1.0 corresponds to a full 360-degree rotation.

   **Returns:**

   **None**
   The function modifies the input surface in place by applying the hue shift.

|

.. py:function:: hsv3d(rgb_array, shift)

   Rotate hue 3d array

   Rotate the hue (HSV conversion method), directly from a 3d array

   Compatible 24 - 32 bit with or without alpha layer

   HSV (Hue, Saturation, Value) is a color model similar to HSL (Hue, Saturation, Lightness)
   but with some differences in how it represents and manipulates colors.
   It’s often used in graphics software and computer vision applications for its
   simplicity in specifying and adjusting color attributes.

   New Shift value. Must be between [0.0 ... 1.0] corresponding to 0.0 - 360.0 degrees
   (e.g 0.5 = 180 degrees)

   **Example usage:**
   

   .. code-block:: python

      array3d = pygame.surfarray.pixels3d(image)
      hsv3d(array3d, 0.2)  # Rotate the hue by 72 degrees

   **Parameters:**

   **rgb_array** (numpy.ndarray)
   A 3D numpy array with shape (w, h, n) containing RGB or RGBA pixel data.
   The array can also represent other formats such as BGR, BGRA.

   **shift** (float)
   A float value in the range [0.0, 1.0] representing the hue rotation.
   A value of 0.0 means no rotation, while 1.0 corresponds to a full 360-degree rotation.

   **Returns:**

   **None**
   The function modifies the input `rgb_array` in place by applying the hue shift.

|

.. py:function:: hsv1d(bgr_array, shift, format_32=False)

   Rotate hue 1d array

   Rotate the hue directly from a C-buffer (1d array uint8 data types RGB(A) format)
   Changes apply inplace

   This method works with other buffer formats such as BGR, BGRA.

   HSV (Hue, Saturation, Value) is a color model similar to HSL (Hue, Saturation, Lightness)
   but with some differences in how it represents and manipulates colors.
   It’s often used in graphics software and computer vision applications for its
   simplicity in specifying and adjusting color attributes.

   **Example usage:**
   

   .. code-block:: python

      # Compatible with 32-bit images
      image = pygame.image.load('../Assets/px.png').convert_alpha()
      hsv1d(image.get_buffer(), angle / 36.0, format_32=True)

      # Compatible with 24-bit images
      image = pygame.image.load('../Assets/px.png').convert(24)
      hsv1d(image.get_buffer(), angle / 36.0, format_32=False)

   **Parameters:**

   **bgr_array** (numpy.ndarray)
   A 1D numpy array (C-buffer) of type uint8 containing pixel data in BGR(A) format.
   The array can also represent other formats such as BGR, BGRA.

   **shift** (float)
   A float value in the range [0.0, 1.0] representing the hue rotation.
   A value of 0.0 means no rotation, while 1.0 corresponds to a full 360-degree rotation.

   **format_32** (bool, optional, default=False)
   If `True`, the input array is assumed to be in 32-bit BGRA format.
   If `False`, the array is assumed to be in 24-bit BGR format.

   **Returns:**

   **None**
   The function modifies the input `bgr_array` in place by applying the hue shift.

|

.. py:function:: hsv1d_cp(bgr_array, shift, format_32=False)

   Rotate the hue 1d array (return a copy)

   HSV 1d array (C buffer) uint8 data types RGB(A) format

   This method works with other buffer formats such as BGR, BGRA.

   HSV (Hue, Saturation, Value) is a color model similar to HSL (Hue, Saturation, Lightness)
   but with some differences in how it represents and manipulates colors.
   It’s often used in graphics software and computer vision applications for its
   simplicity in specifying and adjusting color attributes.

   **Example usage:**
   

   .. code-block:: python

      # 32-bit image
      image = pygame.image.load('../Assets/px.png').convert_alpha()
      arr = hsv1d_cp(image.get_buffer(), angle / 360.0, format_32=True)
      image = pygame.image.frombuffer(arr, (WIDTH, HEIGHT), "BGRA")

      # 24-bit image
      image = pygame.image.load('../Assets/px.png').convert(24)
      arr = hsv1d_cp(image.get_buffer(), angle / 360.0, format_32=False)

   **Parameters:**

   **bgr_array** (numpy.ndarray)
   A 1D numpy array (C-buffer) of type uint8 containing pixel data in RGB(A) format.
   The array can also represent other formats such as BGR, BGRA.

   **shift** (float)
   A float value in the range [0.0, 1.0] representing the hue rotation.
   A value of 0.0 means no rotation, while 1.0 corresponds to a full 360-degree rotation.

   **format_32** (bool, optional, default=False)
   If `True`, the input array is assumed to be in 32-bit BGRA format.
   If `False`, the array is assumed to be in 24-bit BGR format.

   **Returns:**

   **numpy.ndarray**
   A new 1D numpy array of type uint8 containing the pixels with the rotated hue.

|

.. py:function:: wave(surface_, rad, size=5)

   Apply Wave Effect to a Surface (Inplace)

   This function applies a wave effect to a Pygame surface, modifying it in place.
   The effect is applied to the surface based on an angle (in radians) and the
   number of sub-surfaces. It is compatible with 24-bit surfaces.

   The wave effect creates a dynamic, wave-like distortion, often used for water
   or other fluid-like visual effects in games.

   **Example usage:**
   

   .. code-block:: python

      wave(surface, 8 * math.pi / 180.0 + frame_number, 5)  # Animate with a changing angle
      wave(surface, x * math.pi / 180.0, 5)  # Apply wave with a fixed angle

   **Parameters:**

   **surface_** (pygame.Surface)
   A Pygame surface, compatible with 24-bit color depth.

   **rad** (float)
   The angle in radians for the wave effect. This value controls the wave's
   motion over time.

   **size** (int, optional, default=5)
   The number of sub-surfaces used to create the wave effect. A higher number
   results in a more complex wave.

   **Returns:**

   **None**
   The function modifies the input surface in place, applying the wave effect.

|

.. py:function:: wave32(surface_, rad, size=5)

   Apply Wave Effect to a 32-bit Surface (Inplace)

   This function applies a wave effect to a 32-bit Pygame surface, modifying it in place.
   The effect is applied to both the RGB and alpha channels, meaning the wave will
   also displace the alpha layer (transparency) of the surface. It is fully compatible
   with 32-bit SDL surfaces, including those with an alpha channel.

   The wave effect creates a dynamic distortion that simulates the motion of waves,
   often used for effects like water or fluid movement in games.

   **Example usage:**
   

   .. code-block:: python

      wave32(surface, x * math.pi / 180.0, 5)  # Apply wave effect with a rotating angle

   **Parameters:**

   **surface_** (pygame.Surface)
   A Pygame surface, compatible with 32-bit color depth and alpha channel (RGBA).

   **rad** (float)
   The angle in radians for the wave effect, controlling its rotation over time.

   **size** (int, optional, default=5)
   The number of sub-surfaces used to create the wave effect. A higher value results
   in a more detailed wave.

   **Returns:**

   **None**
   The function modifies the input surface in place, applying the wave effect to both
   the color and alpha channels.

|

.. py:function:: wave_static(surface_, array_, rad, size=5)

   Wave effect for static background (inplace)

   This function is different from the `wave` method as a copy of the static
   background or game display is passed to the function as an argument `array_` to
   improve overall performance.

   Compatible with 24-bit surfaces.

   **Example usage:**
   

   .. code-block:: python

      background = pygame.image.load('../Assets/px.png').convert(24)
      background = pygame.transform.smoothscale(background, (800, 600))
      background_cp = background.copy()
      wave_static(pixels3d(background), pixels3d(background_cp), FRAME * math.pi / 180, 5)
      SCREEN.blit(background, (0, 0))

   **Parameters:**

   **surface_** (pygame.Surface)
   A Pygame surface, compatible with 24-bit color depth.

   **array_** (numpy.ndarray)
   A 3D numpy array with shape (w, h, 3), type `uint8`, representing a copy of the
   game display or image to be modified.

   **rad** (float)
   The angle in radians for the wave effect, controlling its rotation over time.

   **size** (int, optional, default=5)
   The number of sub-surfaces used to create the wave effect. A higher value results
   in a more detailed wave.

   **Returns:**

   **None**
   The function modifies the input `array_` in place, applying the wave effect.

|

.. py:function:: swirl(surface_, rgb_array_cp, degrees)

   Swirl an image (inplace)

   The swirl effect is a visual distortion that creates a spiraling appearance in an image or graphic.
   This effect can draw attention to specific areas of a design and add a sense of movement or dynamism.
   It can be used creatively in various contexts, from social media graphics to advertising and digital art.

   **Works with:** 24-bit and 32-bit image formats, but not compatible with 32-bit due to the alpha layer.

   If the image is in 32-bit with an alpha channel, the alpha layer will remain unchanged during the transformation,
   causing the alpha layer to bleed over the effect. To avoid this, convert the image to 24-bit.
   For 32-bit images with an alpha layer, use the `swirl32` method (designed for 32-bit).

   This algorithm uses a table of cosines and sines to achieve the effect.

   **Example usage:**
   

   .. code-block:: python

      background = pygame.image.load("../Assets/background.jpg").convert(24)
      background = pygame.transform.smoothscale(background, (WIDTH, HEIGHT))
      background_cp = background.copy()

      # in the game loop
      swirl(background_cp, pixels3d(background), angle)
      SCREEN.blit(background_cp, (0, 0))

   **Parameters:**

   **surface_** (pygame.Surface)
   A Pygame surface, compatible with 24-bit and 32-bit formats. However, not compatible with 32-bit due to the alpha channel.

   **rgb_array_cp** (numpy.ndarray)
   A 3D numpy array with shape (w, h, 3), containing RGB pixel data. A copy of the image to swirl. Both `surface_` and `rgb_array_cp` must have the same shape and type.

   **degrees** (float)
   The angle in degrees for the swirl effect.

   **Returns:**

   **None**
   The function modifies the input `surface_` in place by applying the swirl effect.

|

.. py:function:: swirl32(surface_, degrees)

   Swirl an image (inplace)

   **Compatible with:** 24-bit and 32-bit formats (with or without alpha layer).

   The swirl effect is a visual distortion that creates a spiraling appearance in an image or graphic.
   This effect can draw attention to specific areas of a design and add a sense of movement or dynamism.
   It can be used creatively in various contexts, from social media graphics to advertising and digital art.

   This algorithm uses a table of cosines and sines for angle approximation.

   Unlike the `swirl` method, `swirl32` takes into account the alpha layer during the transformation,
   meaning it works with 32-bit images that have per-pixel transparency, preserving the alpha channel.

   **Example usage:**
   

   .. code-block:: python

      swirl32(image, angle)

   **Parameters:**

   **surface_** (pygame.Surface)
   A Pygame surface, compatible with 24-bit and 32-bit formats, including alpha channels.

   **degrees** (float)
   The angle in degrees for the swirl effect.

   **Returns:**

   **None**
   The function modifies the input `surface_` in place by applying the swirl effect.

|

.. py:function:: swirlf(surface_, degrees)

   Swirl an image (inplace) with floating point accuracy

   **Compatible with:** 24-bit format only.

   This algorithm does not use cosine and sine tables; instead, it calculates the angles with floating-point accuracy.

   The swirl effect is a visual distortion that creates a spiraling appearance in an image or graphic.
   This effect can draw attention to specific areas of a design and add a sense of movement or dynamism.
   It can be used creatively in various contexts, from social media graphics to advertising and digital art.

   **Example usage:**
   

   .. code-block:: python

      swirlf(surface_, angle)

   **Parameters:**

   **surface_** (pygame.Surface)
   A Pygame surface, compatible with 24-bit formats.

   **degrees** (float)
   The angle in degrees for the swirl effect.

   **Returns:**

   **None**
   The function modifies the input `surface_` in place by applying the swirl effect.

|

.. py:function:: plasma_config(surface_, frame, hue_=1.0/6.0, sat_=1.0/6.0, value_=1.0/8.0, a_=1.0/255.0, b_=1.0/12.0, c_=1.0/12.0)

   Create a basic plasma effect on the top of a Pygame surface (inplace)

   **Compatible with:** 24 - 32-bit surfaces, with or without an alpha layer.

   This function creates a plasma effect that can be applied to a Pygame surface. It modifies the surface in place.
   The effect simulates plasma-like visual distortions using mathematical factors such as hue, saturation, and value.

   **Example usage:**
   

   .. code-block:: python

      plasma_config(surface, frame_number)

   **Parameters:**

   **surface_** (pygame.Surface)
   A Pygame surface, compatible with 24-bit or 32-bit formats.

   **frame** (int)
   A variable that changes over time, controlling the plasma effect's progression.

   **hue_** (float, optional)
   A factor controlling the hue, default value is 1.0/6.0.

   **sat_** (float, optional)
   A factor controlling the saturation, default value is 1.0/6.0.

   **value_** (float, optional)
   A factor controlling the value, default value is 1.0/8.0.

   **a_** (float, optional)
   A control parameter for the plasma equation, default value is 1.0/255.0.

   **b_** (float, optional)
   A control parameter for the plasma equation, default value is 1.0/12.0.

   **c_** (float, optional)
   A control parameter for the plasma equation, default value is 1.0/12.0.

   **Returns:**

   **None**
   The function modifies the input `surface_` in place by applying the plasma effect.

|

.. py:function:: plasma(surface_, frame, palette_)

   Apply plasma effect in place to a surface

   This function generates a dynamic plasma effect on a Pygame surface. The effect evolves
   over time, producing a fluid, glowing visual pattern. The plasma effect is applied directly
   to the surface, modifying its pixels in place.

   The function works with both 24-bit and 32-bit surfaces (with or without an alpha channel).

   **Example usage:**
   

   .. code-block:: python

      plasma(surface, frame_number, palette)

   **Parameters:**

   **surface_** (pygame.Surface)
   A Pygame surface, compatible with 24-bit or 32-bit formats. The plasma effect is applied
   directly to this surface, modifying its pixels in place.

   **frame** (float)
   The current frame number, which drives the evolution of the plasma effect. This value
   determines the shifting patterns and animation in the plasma effect.

   **palette_** (numpy.ndarray)
   A 1D array containing a palette of colors (as unsigned integers) used to generate the
   plasma effect. The colors are applied cyclically to create the visual effect.

   **Returns:**

   **None**
   The function modifies the input `surface_` in place by applying the plasma effect.

|

.. py:function:: brightness(surface_, shift_)

   Adjust brightness in place

   This function controls the brightness level of a Pygame display or SDL surface. The `shift_`
   parameter is a float value in the range [-1.0, 1.0], where +1.0 represents the maximum brightness,
   and 0.0 applies no changes to the surface. Negative values darken the surface.

   The function works with both 24-bit and 32-bit surfaces (with or without an alpha channel).

   **Example usage:**
   

   .. code-block:: python

      brightness(surface, 0.2)

   **Parameters:**

   **surface_** (pygame.Surface)
   A Pygame surface compatible with 24-bit or 32-bit formats (with or without an alpha channel).

   **shift_** (float)
   A float value in the range [-1.0, 1.0], where +1.0 increases brightness, 0.0 keeps the surface unchanged,
   and negative values decrease brightness.

   **Returns:**

   **None**
   The function modifies the input `surface_` in place by adjusting its brightness.

|

.. py:function:: brightness3d(rgb_array, shift)

   Adjust brightness of a 3D image array in place

   This function adjusts the brightness of an image by modifying its 3D array in place. The array
   should be of shape (w, h, n) where `n` is typically 3 (RGB) or 4 (RGBA). The `shift` parameter
   controls the brightness level and must be a float in the range [-1.0, 1.0]. Positive values
   increase brightness, negative values decrease it, and 0.0 results in no change.

   **Example usage:**
   

   .. code-block:: python

      brightness3d(rgb_array, 0.2)

   **Parameters:**

   **rgb_array** (numpy.ndarray)
   A 3D numpy array of shape (w, h, n), where `n` is the number of color channels (3 for RGB or 4 for RGBA).
   The array should contain uint8 data (0–255 values for each color channel).

   **shift** (float)
   A float value in the range [-1.0, 1.0] that controls the brightness level. Positive values increase brightness,
   negative values decrease it, and 0.0 leaves the array unchanged.

   **Returns:**

   **None**
   The function modifies the input `rgb_array` in place by adjusting its brightness.

|

.. py:function:: brightness1d(bgr_array, shift, format_32=False)

   Control brightness of an image from a C buffer (in place)

   This function adjusts the brightness of an image based on a 1D array buffer. The buffer should
   represent the pixel data of the image in either the BGR(A) or RGB(A) format. The brightness is
   adjusted by a `shift` value, which must be a float in the range [-1.0, 1.0]. The `format_32` parameter
   indicates whether the buffer is 32-bit (`True` for BGRA) or 24-bit (`False` for BGR).

   **Example usage:**
   

   .. code-block:: python

      # For 24-bit image (BGR format)
      array_bck = brightness1d_copy(background.get_buffer(), 0.1, False)
      background = pygame.image.frombuffer(array_bck, (800, 600), 'BGR')

      # For 32-bit image (BGRA format)
      array_bck = brightness1d_copy(background.get_buffer(), 0.1, True)
      background = pygame.image.frombuffer(array_bck, (800, 600), 'BGRA')

   **Parameters:**

   **bgr_array** (numpy.ndarray or bytearray)
   A 1D numpy array or bytearray containing pixel data in BGR(A) or RGB(A) format. The array should
   contain uint8 data (0–255 values for each color channel).

   **shift** (float)
   A float value in the range [-1.0, 1.0] that controls the brightness level. Positive values increase brightness,
   negative values decrease it, and 0.0 leaves the array unchanged.

   **format_32** (bool)
   A boolean indicating whether the buffer is 32-bit (`True` for BGRA) or 24-bit (`False` for BGR).

   **Returns:**

   **None**
   The function modifies the input `bgr_array` in place by adjusting its brightness.

|

.. py:function:: brightness1d_copy(bgr_array, shift, format_32=False)

   Control brightness of an image (return a copy)

   This function adjusts the brightness of an image given its C-buffer. The `bgr_array` should be a 1D
   array containing pixel data in either the BGR or BGRA format. The function returns a new array with
   the brightness adjusted based on the `shift` value, which must be a float in the range [-1.0, 1.0].
   The `format_32` parameter specifies whether the image is 32-bit (BGRA) or 24-bit (BGR).

   **Example usage:**
   

   .. code-block:: python

      # For 24-bit image (BGR format)
      array_bck = brightness1d_copy(background.get_buffer(), 0.1, False)
      background = pygame.image.frombuffer(array_bck, (800, 600), 'BGR')

      # For 32-bit image (BGRA format)
      array_bck = brightness1d_copy(background.get_buffer(), 0.1, True)
      background = pygame.image.frombuffer(array_bck, (800, 600), 'BGRA')

   **Parameters:**

   **bgr_array** (numpy.ndarray or bytearray)
   A 1D numpy array or bytearray containing pixel data in BGR(A) or RGB(A) format. The array should
   contain uint8 data (0–255 values for each color channel).

   **shift** (float)
   A float value in the range [-1.0, 1.0] that controls the brightness level. Positive values increase brightness,
   negative values decrease it, and 0.0 leaves the array unchanged.

   **format_32** (bool)
   A boolean indicating whether the buffer is 32-bit (`True` for BGRA) or 24-bit (`False` for BGR).

   **Returns:**

   **numpy.ndarray**
   A new numpy array containing the pixel data with adjusted brightness. The original array is not modified.

|

.. py:function:: brightness_copy(surface_, shift)

   Brightness (return a copy)

   This function applies a brightness transformation to a new SDL surface. The brightness level of the
   surface is modified based on the `shift` parameter, which is a float in the range [-1.0, 1.0]. A value of
   +1.0 corresponds to the maximum brightness, while a value of 0.0 will leave the surface unchanged.

   **Example usage:**
   

   .. code-block:: python

      new_surface = brightness_copy(surface, 0.2)

   **Parameters:**

   **surface_** (pygame.Surface)
   A Pygame surface compatible with either 24-bit or 32-bit formats, with or without an alpha channel.

   **shift** (float)
   A float value in the range [-1.0, 1.0] that adjusts the brightness of the surface. Positive values increase brightness,
   negative values decrease it, and 0.0 leaves the surface unchanged.

   **Returns:**

   **pygame.Surface**
   A new Pygame surface with adjusted brightness, in 24-bit format without an alpha channel.

|

.. py:function:: brightness_exclude(surface_, shift_, color_=(0, 0, 0))

   Brightness adjustment with color exclusion (inplace)

   This function modifies the brightness of an image, excluding a specific color from the transformation process.
   The `shift_` parameter controls the brightness level, with +1.0 being the maximum brightness. The color exclusion
   allows you to avoid affecting specific colors during the transformation.

   **Example usage:**
   

   .. code-block:: python

      # 24-bit image
      image = pygame.image.load('../Assets/px.png').convert(24)
      brightness_exclude(image, +0.5, color=(0, 0, 0))

      # 32-bit image
      image = pygame.image.load('../Assets/px.png').convert_alpha()
      brightness_exclude(image, +0.5, color=(0, 0, 0))

   **Parameters:**

   **surface_** (pygame.Surface)
   A Pygame surface compatible with 24-bit or 32-bit formats.

   **shift_** (float)
   A float value in the range [-1.0, +1.0] controlling the brightness adjustment.
   +1.0 corresponds to the maximum brightness, and 0.0 will leave the surface unchanged.

   **color_** (tuple, optional)
   A tuple of RGB values (e.g., (10, 22, 0)) to exclude from the brightness transformation.
   Default is (0, 0, 0) for black.

   **Returns:**

   **None**
   This function modifies the `surface_` in place, adjusting its brightness while excluding the specified color.

|

.. py:function:: brightness_bpf(surface_, shift_, bpf_threshold=64)

   Brightness adjustment with bpf exclusion (inplace)
   This function adjusts the brightness of an image using a bright pass filter (bpf).
   Pixels with an RGB sum below the specified `bpf_threshold` are excluded from the transformation.
   This helps to avoid modifying darker pixels while still adjusting the brightness of others.

   The `shift_` parameter controls the brightness level, with +1.0 being the maximum brightness.
   The function works with both 24-bit and 32-bit images.

   **Example usage:**
   

   .. code-block:: python

      # 24-bit image
      image = pygame.image.load('../Assets/px.png').convert()
      brightness_bpf(image, 0.5, bpf_threshold=200)

      # 32-bit image
      image = pygame.image.load('../Assets/px.png').convert_alpha()
      brightness_bpf(image, 0.5, bpf_threshold=200)

   **Parameters:**

   **surface_** (pygame.Surface)
   A Pygame surface compatible with 24-bit or 32-bit formats.

   **shift_** (float)
   A float value in the range [-1.0, +1.0] controlling the brightness adjustment.
   +1.0 corresponds to the maximum brightness, and 0.0 will leave the surface unchanged.

   **bpf_threshold** (int, optional)
   An integer value in the range [0, 255] that determines the threshold for the brightness filter.
   Pixels with an RGB sum below this threshold will not be modified. Default value is 64.

   **Returns:**

   **None**
   This function modifies the `surface_` in place, adjusting its brightness based on the given parameters.

|

.. py:function:: saturation(surface_, shift_)

   Saturation adjustment (inplace)

   This function adjusts the saturation level of a Pygame surface or texture.
   A positive `shift_` increases the saturation, while a negative value decreases it.
   The saturation shift is applied in place to the surface, meaning the original surface is modified directly.

   **Example usage:**
   

   .. code-block:: python

      saturation(surface, 0.2)

   **Parameters:**

   **surface_** (pygame.Surface)
   A Pygame surface compatible with 24-bit or 32-bit formats.

   **shift_** (float)
   A float value in the range [-1.0, +1.0] controlling the saturation level.
   A value of 1.0 corresponds to maximum saturation, 0.0 will leave the surface unchanged, and -1.0 will result in no color saturation.

   **Returns:**

   **None**
   This function modifies the `surface_` in place, adjusting its saturation.

|

.. py:function:: saturation3d(rgb_array, shift)

   Saturate 3D array directly (inplace)

   This function adjusts the saturation level of an image by modifying its 3D array data.
   The array should be in the `uint8` format with a shape of `(w, h, 3)` and should contain RGB pixel data.
   Other pixel formats may also be compatible. The resulting image will be in 24-bit format without an alpha layer.

   **Example usage:**
   

   .. code-block:: python

      saturation3d(rgb_array, 0.2)

   **Parameters:**

   **rgb_array** (numpy.ndarray)
   A 3D numpy array (or memory view) with shape `(w, h, 3)` and type `uint8`, containing RGB or another pixel format.
   It should reference an SDL surface or image data.

   **shift** (float)
   A float value in the range [-1.0, +1.0] controlling the saturation level.
   A value of 1.0 corresponds to maximum saturation, 0.0 leaves the image unchanged, and -1.0 decreases saturation.

   **Returns:**

   **None**
   This function modifies the `rgb_array` in place, adjusting the saturation of the referenced image data.

|

.. py:function:: saturation1d(buffer, shift, format_32=False)

   Saturate 1D array directly (inplace)

   This function adjusts the saturation level of an image by modifying its 1D data buffer.
   The buffer should be in `uint8` format with a shape of `(w,)`, and it can contain RGB(A),
   BGR(A), or other pixel formats. For 32-bit images, the alpha channel must be at the end
   of the pixel (e.g., RGB(A) or BGR(A)).

   **Example usage:**
   

   .. code-block:: python

      # For 32-bit image (RGBA format)
      image = pygame.image.load("../Assets/px.png").convert_alpha()
      saturation1d(image.get_buffer(), -0.5, True)
      saturation1d(image.get_view('0'), 0.5, True)

      # For 24-bit image (RGB format)
      image = pygame.image.load("../Assets/px.png").convert(24)
      saturation1d(image.get_buffer(), 0.3, False)

   **Parameters:**

   **buffer** (numpy.ndarray or memoryviewslice)
   A 1D array (or memory view) with shape `(w,)` and type `uint8`, containing pixel data
   in RGB(A), BGR(A), or other formats. This should reference an SDL surface or image buffer.

   **shift** (float)
   A float value in the range [-1.0, +1.0] controlling the saturation level.
   A value of 1.0 corresponds to maximum saturation, 0.0 leaves the image unchanged, and -1.0 decreases saturation.

   **format_32** (bool, optional)
   A flag indicating the pixel format:
   - `True` for 'RGBA' (32-bit format with an alpha channel).
   - `False` for 'RGB' (24-bit format without alpha).

   **Returns:**

   **None**
   This function modifies the `buffer` in place, adjusting the saturation of the referenced image data.
   The final image retains the same pixel format as the input image.

|

.. py:function:: saturation1d_cp(buffer, shift, format_32=False)

   Saturate an image using a C-buffer (return copy)

   This function adjusts the saturation level of an image by modifying its 1D data buffer,
   and returns a new array with the adjusted saturation. The `buffer` should be in `uint8` format
   with a shape of `(w,)`, and can contain RGB(A), BGR(A), or other pixel formats. For 32-bit images,
   the alpha channel must be placed at the end of the pixel format (e.g., RGB(A) or BGR(A)).

   **Example usage:**
   

   .. code-block:: python

      # For 32-bit image (RGBA format)
      image = pygame.image.load("../Assets/px.png").convert_alpha()
      new_buffer = saturation1d_cp(image.get_buffer(), -0.5, True)

      # For 24-bit image (RGB format)
      image = pygame.image.load("../Assets/px.png").convert(24)
      new_buffer = saturation1d_cp(image.get_buffer(), 0.5, False)

   **Parameters:**

   **buffer** (numpy.ndarray or memoryviewslice)
   A 1D array (or memory view) with shape `(w,)` and type `uint8`, containing pixel data
   in RGB(A), BGR(A), or other formats. This should reference an SDL surface or image buffer.

   **shift** (float)
   A float value in the range [-1.0, +1.0] controlling the saturation level.
   A value of 1.0 corresponds to maximum saturation, 0.0 leaves the image unchanged, and -1.0 decreases saturation.

   **format_32** (bool, optional)
   A flag indicating the pixel format:
   - `True` for 'RGBA' (32-bit format with an alpha channel).
   - `False` for 'RGB' (24-bit format without alpha).

   **Returns:**

   **numpy.ndarray**
   A new 1D array with shape `(w,)` and type `uint8`, containing the same pixel format as the input array,
   but with the adjusted saturation level.

|

.. py:function:: heatconvection(surface_, amplitude, center=0.0, sigma=2.0, mu=0.0)

   Heat flow convection

   This function simulates heat flow convection (or convective heat transfer) by applying
   a Gaussian transformation to an image, creating a vertical flow effect. The transformation
   is influenced by the parameters, such as amplitude, center, sigma, and mu, which modify the
   intensity and direction of the effect.

   The convection effect can be used to simulate phenomena like air turbulence or heat flow.
   A periodic variable for `amplitude` (e.g., using a cosine function) can create dynamic, changing effects.

   **Example usage:**
   

   .. code-block:: python

      # For 32-24 bit image format
      image = pygame.image.load("../Assets/fire.jpg").convert()
      b = math.cos(i * 3.14 / 180.0) * random.uniform(0, 2)
      heatconvection(image, abs(b) * random.uniform(20.0, 80.0), 0, sigma=random.uniform(0.8, 4), mu=b)
      # Restore the original image
      image = image_copy.copy()

   **Parameters:**

   **surface_** (pygame.Surface)
   A Pygame surface compatible with 24-bit or 32-bit formats. The transformation is applied directly to this surface.

   **amplitude** (float)
   The maximum amplitude (vertical pixel displacement) of the Gaussian transformation. If amplitude is zero,
   no transformation will be applied. A periodic function can be used to create a dynamic amplitude.

   **center** (float, optional)
   The center of the Gaussian transformation. The default is 0.0, which centers the effect.

   **sigma** (float, optional)
   The sigma value of the Gaussian equation. A small value creates a narrow effect, while a large value widens the effect.
   The default is 2.0.

   **mu** (float, optional)
   The mu value of the Gaussian equation. When mu is periodic (e.g., using a cosine function), it allows the effect
   to move horizontally. The default is 0.0.

   **Returns:**

   **None**
   The function modifies the `surface_` in place, applying the heat convection effect.

|

.. py:function:: horizontal_glitch(surface_, deformation, frequency, amplitude)

   Horizontal glitch (inplace)

   This function applies a horizontal glitch effect to a Pygame surface, deforming the image
   horizontally based on a signal defined by the given parameters. The effect is achieved
   by modifying the image's pixels in a glitchy, random manner, influenced by the deformation,
   frequency, and amplitude parameters.

   **Example usage:**
   

   .. code-block:: python

      # For 24-bit and 32-bit
      horizontal_glitch(background, deformation=0.5, frequency=0.08, amplitude=FRAME % 20)

   **Parameters:**

   **surface_** (pygame.Surface)
   A Pygame surface compatible with 24-bit or 32-bit formats. The transformation is applied directly to this surface.

   **deformation** (float)
   The angle in radians that controls the variation of the horizontal deformation. This value influences the overall
   glitch pattern.

   **frequency** (float)
   A factor that amplifies the angle variation. It determines the rate at which the glitch effect changes over time.

   **amplitude** (float)
   The amplitude of the cos function used to modulate the glitch effect. A higher amplitude leads to a more intense deformation.

   **Returns:**

   **None**
   The function modifies the `surface_` in place, applying the horizontal glitch effect.

|

.. py:function:: horizontal_sglitch(surface_, array_, deformation, frequency, amplitude)

   Glitch for static image background (inplace)

   This function applies a horizontal glitch effect to a Pygame surface, modifying the image by
   deforming it based on a signal defined by the deformation, frequency, and amplitude parameters.
   It creates a glitch effect specifically for static image backgrounds.

   **Example usage:**
   

   .. code-block:: python

      # For 24-bit and 32-bit
      horizontal_sglitch(background, bgr_array, deformation=0.5, frequency=0.08, amplitude=FRAME % 20)

   **Parameters:**

   **surface_** (pygame.Surface)
   A Pygame surface compatible with 24-bit or 32-bit formats. The transformation is applied directly to this surface.

   **array_** (numpy.ndarray)
   A numpy array containing a pixel copy that is used for the glitch effect. This array serves as a reference for the transformation.

   **deformation** (float)
   The angle in radians that controls the variation of the horizontal deformation. This value determines the degree of the glitch effect.

   **frequency** (float)
   A factor that amplifies the angle variation, influencing the rate at which the glitch effect changes over time.

   **amplitude** (float)
   The amplitude of the cos function that modulates the glitch effect. Higher amplitude results in more intense deformation.

   **Returns:**

   **None**
   The function modifies the `surface_` in place, applying the horizontal glitch effect using the provided parameters.

|

.. py:function:: bpf(surface_, threshold=128)

   BPF, bright pass filter (inplace)

   This function applies a bright pass filter to a Pygame surface, conserving only the brightest pixels.
   The pixels that fall below the specified brightness threshold are excluded from the surface, effectively
   brightening the image.

   **Example usage:**
   

   .. code-block:: python

      # For 24-bit
      image = pygame.image.load('../Assets/px.png').convert(24)
      bpf(image, threshold=60)

      # For 32-bit
      image = pygame.image.load('../Assets/px.png').convert_alpha()
      bpf(image, threshold=60)

   **Parameters:**

   **surface_** (pygame.Surface)
   A Pygame surface compatible with 24-bit or 32-bit formats. The transformation is applied directly to this surface.

   **threshold** (int)
   The bright pass threshold, with a default value of 128. Pixels with brightness below this threshold will be excluded from the transformation.

   **Returns:**

   **None**
   This function modifies the `surface_` in place by applying the bright pass filter.

|

.. py:function:: bloom(surface_, threshold_, fast_=False, mask_=None)

   Create a bloom effect (inplace)

   This function applies a bloom effect to a Pygame surface or image. The bloom effect brightens the
   image and adds a glowing effect around bright areas. The effect can be selectively filtered using
   a mask, and performance can be improved with the fast option at the cost of reduced effect appearance.

   **Example usage:**
   

   .. code-block:: python

      # Check the demo_bloom_mask.py script in the Demo folder
      bloom(SCREEN, threshold=BPF, fast=True, mask=mask)

   **Parameters:**

   **surface_** (pygame.Surface)
   A Pygame surface compatible with 24-bit or 32-bit formats. This surface will be transformed in place by applying the bloom effect.

   **threshold_** (int)
   An integer threshold value in the range [0...255]. This value is used to detect bright pixels within the texture or image to apply the bloom effect.

   **fast_** (bool)
   A boolean value that, when set to `True`, approximates the bloom effect for performance improvement (10x - 80x), but reduces the visual intensity of the effect. Default is `False`.

   **mask_** (numpy.ndarray or memoryviewslice)
   A 2D array of type uint8 representing the mask alpha, with shape `(w, h)`. Values in the range [0..255] control the transparency, and thus, the selective application of the bloom effect. An array filled with 255 renders the full bloom effect, while an array filled with 0 disables the effect. Intermediate values create partial bloom effects.

   **Returns:**

   **None**
   This function modifies the `surface_` in place by applying the bloom effect.

|

.. py:function:: fisheye_footprint(w, h, centre_x, centre_y)

   Create a fisheye lens model holding pixel coordinates of a surface

   This function generates a fisheye lens model that holds the pixel coordinates for each pixel
   on a given surface. The model can be used to apply a fisheye effect to a surface by mapping
   its coordinates into a fisheye lens projection.

   **Example usage:**
   

   .. code-block:: python

      width, height = surface.get_size()
      f_model = fisheye_footprint(w=width, h=height, centre_x=width >> 1, centre_y=height >> 1)
      fisheye(surface, f_model)

   **Parameters:**

   **w** (int)
   The width of the surface to project into the fisheye model. The value must match the surface's width.

   **h** (int)
   The height of the surface to project into the fisheye model. The value must match the surface's height.

   **centre_x** (int)
   The x-coordinate of the centre of the fisheye effect.

   **centre_y** (int)
   The y-coordinate of the centre of the fisheye effect.

   **Returns:**

   **numpy.ndarray**
   A 2D array of type `np.uint32_t` with shape `(w, h, 2)`, representing the fisheye model and holding the coordinates of all pixels projected through the fisheye lens model.

|

.. py:function:: fisheye_footprint_param(tmp_array_, centre_x, centre_y, param1_, focal_length)

   Create a fisheye model to hold the pixel coordinates with additional parameters

   This function generates a fisheye model that holds the pixel coordinates of a surface,
   with the added ability to control the fisheye model's aspect and focal length.

   **Example usage:**

   .. code-block:: python

      tmp = numpy.ndarray((400, 400, 2), dtype=numpy.uint32, order='C')
      fisheye_footprint_param(tmp, 200, 200, 1.0, 0.6)

   **Parameters:**

   **tmp_array_** (numpy.ndarray)
   A 3D array with shape `(w, h, 2)` of unsigned integers. The shape of this array will determine
   the fisheye model.

   **centre_x** (float)
   The x-coordinate of the fisheye effect’s center. Corresponds to half the length of the fisheye model.

   **centre_y** (float)
   The y-coordinate of the fisheye effect’s center. Corresponds to half the width of the fisheye model.

   **param1_** (float)
   A parameter that controls the fisheye model’s aspect. Values greater than 1.0 cause the effect to converge to the center,
   while values less than 1.0 cause the effect to diverge from the center.

   **focal_length** (float)
   Controls the type of fisheye lens. Values greater than 1.0 create a diverging lens effect, while values less than 0
   create a converging lens effect.

   **Returns:**

   **void**
   This function performs an inplace transformation on the provided `tmp_array_`, updating it with the fisheye model's coordinates.

|

.. function:: fisheye(surface_, fisheye_model)

   Display surface or gameplay through a lens effect (inplace).

   Compatible with 24-bit surfaces only.

   A fisheye lens is an ultra wide-angle lens that produces strong visual
   distortion intended to create a wide panoramic or hemispherical image.

   This function applies a fisheye lens effect in real-time. To achieve this,
   the algorithm utilizes a pre-calculated lens model transformation that is
   stored in a numpy ndarray, passed as the `fisheye_model` argument. The
   `fisheye_model` array has a shape of (w, h, 2) and contains pixel coordinates
   of the surface after lens transformation. All calculations are performed upstream.

   Use the function :func:`fisheye_footprint_c` to create the pre-calculated array.
   This method needs to be called only once.

   The fisheye lens transformation is applied inplace.

   **Example usage**:

   .. code-block:: python

      width, height = surface.get_size()
      f_model = fisheye_footprint(w=width, h=height, centre_x=width >> 1, centre_y=height >> 1)
      fisheye(surface, f_model)

   **Parameters**:
   - `surface_` (pygame.Surface): A surface object that is compatible with 24-bit color depth.
   - `fisheye_model` (numpy.ndarray): A numpy array with shape (w, h, 2) containing uint32 values that represent the transformed pixel coordinates (`x'`, `y'`). These values are pre-calculated using the :func:`fisheye_footprint_c` function.

   **Returns**:
   - `None`: The transformation is applied inplace, and no value is returned.

|

.. function:: tv_scan(surface_, space=5)

   Apply a TV scanline effect on a pygame surface (inplace).

   The space between each scanline can be adjusted with the `space` value.

   **Example usage**:

   .. code-block:: python

      tv_scan(image, space=10)

   **Parameters**:
   - `surface_` (pygame.Surface): A surface object compatible with 24-bit or 32-bit color formats.
   - `space` (int, optional): The space between each scanline. You can set a constant value or use a variable for a dynamic effect. Default is 5.

   **Returns**:
   - `None`: The effect is applied inplace, and no value is returned.

|

.. function:: ripple(rows_, cols_, previous_, current_, array_, dispersion_=0.008)

   Apply a ripple effect without background deformation.

   This function simulates a ripple effect over a surface, using previous and current states
   of the ripple, along with a background image. The effect is applied without distorting the
   background image, which remains static.

   **Example usage**:

   .. code-block:: python

      previous, current = ripple(width, height, previous, current, back_array, dispersion_=0.008)

   **Parameters**:
   - `rows_` (int): The screen width or surface width.
   - `cols_` (int): The screen height or surface height.
   - `previous_` (numpy.ndarray): A float array with shape (w, h) used for the transformation. It holds the previous state of the ripple effect.
   - `current_` (numpy.ndarray): A float array with shape (w, h) used for the transformation. It holds the current state of the ripple effect.
   - `array_` (numpy.ndarray): A uint8 array with shape (w, h, 3) containing the static background image in RGB format. This array remains unchanged.
   - `dispersion_` (float, optional): The ripple dampening factor, which controls the ripple effect's intensity. Higher values reduce the ripple effect radius. The default value is 0.008.

   **Returns**:
   - `tuple`: A tuple containing two arrays `(current_, previous_)`, which represent the updated states of the ripple. See the Parameters section for the sizes of each array.

|

.. function:: ripple_seabed(cols_, rows_, previous_, current_, texture_array_, background_array_, dispersion_=0.008)

   Apply a ripple effect with background deformation.

   This function simulates a ripple effect over a surface, where the background image
   is deformed as part of the effect. The function uses the previous and current states
   of the ripple along with a texture and background image.

   **Example usage**:

   .. code-block:: python

      previous, current, back_array = ripple_seabed(height, width, previous,
         current, texture_array, back_array, dispersion_=0.009)

   **Parameters**:
   - `cols_` (int): The screen width or surface width.
   - `rows_` (int): The screen height or surface height.
   - `previous_` (numpy.ndarray): A float array with shape (w, h) used for the transformation. It holds the previous state of the ripple effect.
   - `current_` (numpy.ndarray): A float array with shape (w, h) used for the transformation. It holds the current state of the ripple effect.
   - `texture_array_` (numpy.ndarray): A uint8 array with shape (w, h, 3) containing the static background image in RGB format. This array is unchanged during the ripple transformation.
   - `background_array_` (numpy.ndarray): A uint8 array with shape (w, h, 3) containing the background image in RGB format. This array is transformed by the ripple effect.
   - `dispersion_` (float, optional): The ripple dampening factor, which controls the ripple effect's intensity. Higher values reduce the ripple effect radius. The default value is 0.008.

   **Returns**:
   - `tuple`: A tuple containing three arrays `(current_, previous_, bck_array)`. The arrays represent the updated states of the ripple effect and background deformation. See the Parameters section for the sizes of each array.

|

.. function:: heatmap(surface_, rgb_=True)

   Transform an image into a heatmap equivalent (in-place).

   This function modifies the given image surface to apply a heatmap effect.
   The transformation is applied directly to the surface, so no new object
   is returned. It also allows the user to choose whether the heatmap
   should be in RGB or BGR color model.

   **Example usage**:

   .. code-block:: python

      # Load an image, convert to an alpha surface, and apply heatmap
      image = pygame.image.load("../Assets/px.png").convert_alpha()
      heatmap(image, True)

   **Parameters**:
   - `surface_` (pygame.Surface): A pygame.Surface object, typically in 24-bit or 32-bit image format (compatible with pygame). The image to which the heatmap effect will be applied.
   - `rgb_` (bool, optional): If True, the image will be transformed into an RGB-based heatmap. If False, the transformation will use the BGR-based heatmap model. Default is True (RGB).

   **Returns**:
   - `None`: This function operates in-place, meaning the input surface is modified directly without returning a new object.

   **Raises**:
   - `TypeError`: If `surface_` is not a valid pygame.Surface object.

|

.. function:: predator_vision(surface_, sobel_threshold=12, bpf_threshold=50, bloom_threshold=50, inv_colormap=False, fast=False, blend=pygame.BLEND_RGB_ADD)

   Simulate Predator Vision Mode.

   This function simulates the predator's vision using a series of image processing
   filters. It applies Sobel edge detection, bright pass filter (BPF), bloom effects,
   and a colormap inversion to the given surface. Optionally, it allows for faster
   processing and blending effects.

   **Example usage**:

   .. code-block:: python

      surface_ = predator_vision(
         image.copy(), sobel_threshold=80, bpf_threshold=0,
         bloom_threshold=0, inv_colormap=True, fast=True)

   **Parameters**:
   - `surface_` (pygame.Surface): A pygame.Surface object, compatible with 24-bit or 32-bit formats. This is the image to which the predator vision effect will be applied.
   - `sobel_threshold` (int, optional): The threshold for Sobel edge detection, used to detect edges in the image. The default value is 12.
   - `bpf_threshold` (int, optional): The threshold for the Bright Pass Filter (BPF), used to detect and highlight bright pixels. The default value is 50.
   - `bloom_threshold` (int, optional): The intensity of the bloom effect, which adds a glow around bright pixels. The default value is 50.
   - `inv_colormap` (bool, optional): If True, the colormap will be inverted, changing the color scheme to resemble a predator's vision more closely. The default value is False.
   - `fast` (bool, optional): If True, a faster processing mode is used. This will reduce the quality in exchange for faster processing time. The default value is False.
   - `blend` (int, optional): The blending mode to apply after all effects have been processed. This can be a value from `pygame.BLEND_*` modes. The default is `pygame.BLEND_RGB_ADD`.

   **Returns**:
   - `pygame.Surface`: A new pygame.Surface object with the predator vision effect applied. The surface is in 24-bit format.

|

.. function:: blood(surface_, mask_, percentage_)

   Apply a blood effect (in-place).

   This function applies a blood effect to a given surface using a mask array that defines
   the contour of the blood effect. The surface and mask must have the same dimensions.
   The `percentage` parameter determines the intensity of the blood effect, with 1.0 representing
   full intensity.

   **Example usage**:

   .. code-block:: python

      background = pygame.image.load("../Assets/Aliens.jpg").convert()
      background = pygame.transform.smoothscale(background, (800, 600))
      background.convert(32, RLEACCEL)
      image = background.copy()

      blood_surface = pygame.image.load("../Assets/redvignette.png").convert_alpha()
      blood_surface = pygame.transform.smoothscale(blood_surface, (800, 600))
      BLOOD_MASK = numpy.asarray(pygame.surfarray.pixels_alpha(blood_surface) / 255.0, numpy.float32)

      # Then call the method in your main loop (percentage must vary over time)
      blood(image, BLOOD_MASK, percentage)

   **Parameters**:
   - `surface_` (pygame.Surface): The surface (e.g., game display) to which the blood effect will be applied. It must be in a compatible 24-bit or 32-bit format.
   - `mask_` (numpy.ndarray or cython.view.memoryview): A normalized array or memoryview (shape: (w, h), type: float) representing the blood mask. The values must be in the range [0.0, 1.0], where 1.0 represents full intensity of the effect.
   - `percentage_` (float): A value in the range [0.0, 1.0] that determines the intensity of the blood effect. A value of 1.0 applies the full effect, while 0.0 applies no effect.

   **Returns**:
   - `None`: The function modifies the given surface in place (i.e., it has no return value).

|

.. function:: mirroring_array(rgb_array)

   Return a mirrored numpy array.

   This method returns a numpy.ndarray with mirrored pixels, flipping the image along the
   horizontal axis. The output array has the same shape and type as the input, with the
   pixel values mirrored.

   **Example usage**:

   .. code-block:: python

      rgb_array = mirroring_array(pixels3d(image))
      surface = make_surface(rgb_array)

   **Parameters**:
   - `rgb_array` (numpy.ndarray): An array with shape (w, h, 3) of type uint8, containing RGB or any other pixel format (such as BGR).

   **Returns**:
   - `numpy.ndarray`: A numpy ndarray with shape (w, h, 3) of type uint8, identical to the input array but with mirrored pixels.

|

.. function:: mirroring(surface_)

   Apply a mirroring effect (in-place).

   This method creates a mirrored image of the given surface by reflecting it
   horizontally. The effect is applied directly to the surface (in-place).

   **Example usage**:

   .. code-block:: python

      # Load an image, apply the mirroring effect
      image = pygame.image.load("../Assets/px.png").convert()
      mirroring(image)

   **Parameters**:
   - `surface_` (pygame.Surface): A pygame surface compatible with 24-bit or 32-bit formats that will undergo the mirroring effect.

   **Returns**:
   - `None`: The function modifies the given surface in place, meaning it does not return a new surface but rather alters the input surface.

|

.. function:: sharpen(surface_)

   Sharpen an image (in-place) using a 3x3 kernel.

   This function applies a sharpening filter to the image on the given surface.
   The filter uses a 3x3 kernel to enhance the edges and details by increasing contrast
   around edges, making the image appear sharper.

   **Example usage**:

   .. code-block:: python

      # Load an image and apply the sharpen effect
      image = pygame.image.load("../Assets/px.png").convert()
      sharpen(image)

   **Parameters**:

   - `surface_` (pygame.Surface): A Pygame surface compatible with 24-bit or 32-bit image formats
     that will be modified in-place to apply the sharpen effect.

   **Returns**:

   - `None`: The function modifies the given surface in place, meaning no new surface is returned.
     The original surface will be sharpened.

|

.. function:: sharpen_1d(w, h, bgr_array, format_32=False)

   Sharpen array (in-place) using a 3x3 kernel.

   This function applies a sharpening filter directly to a 1D array using a 3x3 kernel.
   It is compatible with BGR or BGR(A) array types, and can also work with other pixel formats
   such as RGB or RGB(A). The sharpening filter enhances the edges by increasing contrast around
   them, making the image appear sharper. If the array contains alpha transparency, set `format_32=True`.

   **Example usage**:

   .. code-block:: python

      # for 32-bit array data (BGR(A))
      sharpen_1d(w, h, im.get_buffer(), True)

      # for 24-bit array data (BGR type)
      sharpen_1d(w, h, im.get_buffer(), False)

   **Parameters**:
   - `w` (int): The width of the array.
   - `h` (int): The height of the array.
   - `bgr_array` (numpy.ndarray): A 1D array (shape: (w, )) of type uint8, containing the BGR or RGB pixel values.
   - `format_32` (bool, optional): If `True`, the array contains alpha transparency (BGRA format). If `False`, the array is BGR format (default is `False`).

   **Returns**:
   - `None`: The function modifies the given array in place (i.e., no new array is returned).

|

.. function:: sharpen_1d_cp(w, h, bgr_array, format_32=False)

   Sharpen array (return a new array).

   This function returns a new array with a sharpening filter applied, using a 3x3 kernel.
   It is compatible with BGR or BGR(A) array types, and can also work with other pixel formats
   such as RGB or RGB(A). The sharpening filter enhances edges by increasing contrast around
   them, making the image appear sharper. If the array contains alpha transparency, set `format_32=True`.

   **Example usage**:

   .. code-block:: python

      # for 32-bit array data (BGR(A))
      arr = sharpen_1d_cp(w, h, im.get_buffer(), True)
      im = pygame.image.frombuffer(arr, (w, h), "RGBA").convert_alpha()

      # for 24-bit array data (BGR type)
      arr = sharpen_1d_cp(w, h, im.get_buffer(), False)
      im = pygame.image.frombuffer(arr, (w, h), "RGB")

   **Parameters**:
   - `w` (int): The width of the array.
   - `h` (int): The height of the array.
   - `bgr_array` (numpy.ndarray): A 1D array (shape: (w, )) of type uint8, containing the BGR or RGB pixel values.
   - `format_32` (bool, optional): If `True`, the array contains alpha transparency (BGRA format). If `False`, the array is BGR format (default is `False`).

   **Returns**:
   - `numpy.ndarray`: A 1D numpy array (shape: (w, )) of type uint8, with sharpened pixels similar to the input array.

|

.. function:: sharpen32(surface_)

   Sharpen image using a 3x3 kernel (in-place).

   This function applies a sharpening filter directly to the image on the given surface.
   The filter uses a 3x3 kernel to enhance edges and details by increasing contrast around
   edges, making the image appear sharper. It is compatible with both 24-bit and 32-bit images.

   **Example usage**:

   .. code-block:: python

      # for 32-bit images
      sharpen32(image)

   **Parameters**:
   - `surface_` (pygame.Surface): A Pygame surface compatible with 24-bit or 32-bit formats to be sharpened.

   **Returns**:
   - `None`: The function modifies the given surface in place (i.e., no new surface is returned).

|

.. function:: dirt_lens(surface_, lens_model_, flag_=BLEND_RGB_ADD, light_=0.0)

   Dirt lens effect (in-place).

   This function applies a dirt lens texture on top of the game display to simulate a camera artifact or
   realistic camera effect, particularly when light from the scene is oriented directly toward the camera.
   The function blends the lens texture with the display, creating a dirt effect on the lens.

   **Example usage**:

   .. code-block:: python

      dirt_lens(image, lens_model, flag_=pygame.BLEND_RGB_ADD, light_=0.1)

   **Parameters**:
   - `surface_` (pygame.Surface): The surface (display or game screen) that will be modified by the dirt lens effect. It must be in a 24-bit or 32-bit format.
   - `lens_model_` (pygame.Surface): The lens model texture (e.g., dirt lens) to be applied to the surface. You can choose from various lens textures provided in the Assets directory. These textures must be resized to fit the display dimensions.
   - `flag_` (int, optional): The blending flag to apply to the lens texture. Default is `pygame.BLEND_RGB_ADD`, which adds the lens texture on top of the surface. You can use other Pygame blending flags like `BLEND_RGB_MAX`, etc.
   - `light_` (float, optional): A float value between -1.0 and 0.2 that adjusts the brightness of the lens texture. Values less than 0 will reduce the lens effect, while values greater than 0 will brighten the display and increase the dirt lens effect. Default is 0.0.

   **Returns**:
   - `None`: This function modifies the input surface in-place and does not return a new object.

|

.. function:: dithering(surface_)

   Dithering Floyd-Steinberg (copy).

   This function applies the Floyd-Steinberg dithering algorithm to a Pygame surface to create the illusion
   of color depth in images with a limited color palette. Dithering approximates colors that are not available
   in the palette by diffusing colored pixels from within the available palette. The result is a dithered image
   with a characteristic graininess or speckled appearance.

   The input image is converted from a Pygame surface to a 3D array (w, h, 3) with a float data type. As the image
   is converted to a different data type (from `uint8` to `float32`), the transformation is not applied in place,
   and a copy of the original image (without the alpha channel) is returned.

   **Example usage**:

   .. code-block:: python

      image = dithering(image)

   **Parameters**:
   - `surface_` (pygame.Surface): A Pygame surface object in 24-bit or 32-bit format. The surface to be dithered.

   **Returns**:
   - `pygame.Surface`: A new Pygame surface in 24-bit format (without the alpha channel) representing the dithered version of the input image.

|

.. function:: dithering_inplace(surface_)

   Dithering Floyd-Steinberg (inplace).

   This function applies the Floyd-Steinberg dithering algorithm to a Pygame surface to create the illusion
   of color depth in images with a limited color palette. Dithering approximates colors that are not available
   in the palette by diffusing colored pixels from within the available palette. The result is a dithered image
   with a characteristic graininess or speckled appearance. Unlike the non-inplace version, this function modifies
   the image surface in place.

   The input image is converted from a Pygame surface to a 3D array (w, h, 3) with a float data type. The image is
   modified directly, and no new object is returned.

   **Example usage**:

   .. code-block:: python

      dithering_inplace(image)

   **Parameters**:
   - `surface_` (pygame.Surface): A Pygame surface object in 24-bit or 32-bit format. The surface to be dithered.

   **Returns**:
   - `None`: The input surface is modified in place (i.e., no new surface is returned).

|

.. function:: dithering1d(w, h, bgr_array, format_32=False)

   Dithering Floyd-Steinberg (inplace) on a 1D array.

   This function applies the Floyd-Steinberg dithering algorithm to a 1D array, simulating the illusion of
   color depth in images with a limited color palette. Dithering approximates colors that are not available
   in the palette by diffusing colored pixels from within the available palette. The result is a dithered image
   with a characteristic graininess or speckled appearance. This function modifies the input array in place.

   The dithering process can handle both 24-bit and 32-bit images, with the `format_32` flag indicating whether
   the input is a 32-bit image (with transparency) or a 24-bit image.

   **Example usage**:

   .. code-block:: python

      dithering1d(w, h, im.get_buffer(), True)   # for 32-bit image
      dithering1d(w, h, im.get_buffer(), False)  # for 24-bit image

   **Parameters**:

   - `w` (int): The width of the image array.
   - `h` (int): The height of the image array.
   - `bgr_array` (numpy.ndarray): A 1D array with shape `(w, h, 3)` for 24-bit images (BGR) or `(w, h, 4)` for
     32-bit images (BGRA).
   - `format_32` (bool, optional): A flag indicating the image format. Set to `True` for 32-bit (BGRA) images,
     or `False` for 24-bit (BGR) images. The default is `False`.

   **Returns**:
   - `None`: This function modifies the input array in place (i.e., no new array is returned).

|

.. function:: dithering1d_cp(w, h, rgb_array, format_32=False)

   Dithering Floyd-Steinberg (copy) on a 1D array.

   This function applies the Floyd-Steinberg dithering algorithm to a 1D array, simulating the illusion of
   color depth in images with a limited color palette. Dithering approximates colors that are not available
   in the palette by diffusing colored pixels from within the available palette. The result is a dithered image
   with a characteristic graininess or speckled appearance. Unlike `dithering1d`, this function returns a new array
   with the dithering effect applied, rather than modifying the array in place.

   The dithering process can handle both 24-bit and 32-bit images. The `format_32` flag indicates whether the input
   is a 32-bit image (with transparency) or a 24-bit image.

   **Example usage**:

   .. code-block:: python

      buff = pygame.image.tobytes(im, "RGB")
      arr = dithering1d_cp(w, h, buff, False)  # for 24-bit image
      im = pygame.image.frombuffer(arr, (w, h), "RGB")

      buff = pygame.image.tobytes(im, "RGBA")
      arr = dithering1d_cp(w, h, buff, True)   # for 32-bit image
      im = pygame.image.frombuffer(arr, (w, h), "RGBA")

   **Parameters**:

   - `w` (int): The width of the image array.
   - `h` (int): The height of the image array.
   - `rgb_array` (numpy.ndarray): A 1D array with shape `(w, h, 3)` for 24-bit images (RGB) or `(w, h, 4)` for
     32-bit images (RGBA).
   - `format_32` (bool, optional): A flag indicating the image format. Set to `True` for 32-bit (RGBA) images,
     or `False` for 24-bit (RGB) images. The default is `False`.

   **Returns**:

   - `numpy.ndarray`: A new 1D array with the dithering effect applied. The array shape is `(w, h, 3)` for 24-bit
     images or `(w, h, 4)` for 32-bit images.

|

.. function:: dithering_atkinson(surface_)

   Dithering Atkinson (copy)

   Atkinson dithering is a variant of Floyd–Steinberg dithering, developed by Bill Atkinson at Apple Computer
   and used in the original Macintosh computer. This dithering technique is used to create the illusion of "color
   depth" in images with a limited color palette, also known as color quantization. Colors that are not available in
   the palette are approximated by diffusing colored pixels from within the available palette. The human eye perceives
   this diffusion as a mixture of the colors, often resulting in a characteristic grainy or speckled appearance.

   This function applies Atkinson dithering to an image represented by a Pygame surface. The image is converted to
   a 3D array format with the shape `(w, h, 3)` and type `float32` (single precision). Since the image is converted
   to a different data type (from `uint8` to `float32`), the transformation cannot be applied in place.

   The function returns a new image with the dithering effect applied.

   **Example usage**:

   .. code-block:: python

      image = dithering_atkinson(image)  # for 24 or 32-bit image format

   **Parameters**:

   - `surface_` (pygame.Surface): A Pygame surface in 24-bit or 32-bit format that will undergo Atkinson dithering.

   **Returns**:

   - `pygame.Surface`: A new Pygame surface in 24-bit format with the dithering effect applied (without the alpha channel).

|

.. function:: dithering_atkinson1d(w, h, c_buffer, format_32=False)

   Atkinson Dithering for 1D Array (inplace)

   Atkinson dithering is a variant of Floyd–Steinberg dithering, developed by Bill Atkinson at Apple Computer,
   and used in the original Macintosh computer. This dithering technique is employed to create the illusion of
   "color depth" in images with a limited color palette, a process also known as color quantization. In a dithered
   image, colors not available in the palette are approximated by diffusing colored pixels from within the available
   palette. The human eye perceives this diffusion as a mixture of colors, often leading to a characteristic grainy
   or speckled appearance.

   This function applies Atkinson dithering to an image represented by a 1D array in memory. It operates directly
   on the array (in-place), modifying the pixel values to achieve the dithering effect.

   The function is compatible with both 24-bit and 32-bit images. If the dithering process is not applied to the image,
   check the `format_32` flag. The flag should be set to `True` for images containing per-pixel transparency or
   equivalent array shapes (w, h, 4). For 24-bit images, set `format_32` to `False` (array shape (w, h, 3)).

   **Example usage**:

   .. code-block:: python

      # for 32-bit image
      dithering_atkinson1d(w, h, im.get_buffer(), True)

      # for 24-bit image
      dithering_atkinson1d(w, h, im.get_buffer(), False)

   **Parameters**:

   - `w` (int): The width of the array.
   - `h` (int): The height of the array.
   - `c_buffer` (C-Buffer or memoryviewslice or 1d numpy.ndarray): The 1D array or memory view containing BGR pixels or any other pixel format (datatype uint8).
   - `format_32` (bool, optional):
     - `True` if the array represents a 32-bit image (BGRA format with transparency).
     - `False` for a 24-bit image (BGR format). The default is `False`.

   **Returns**:

   - `None`: This function modifies the array in place. It does not return a new array.

|

.. function:: pixelation(surface_, blocksize_=64)

   Pixelate a Pygame Surface

   Pixelation in computer graphics refers to the effect caused by displaying a bitmap or a section of a bitmap at
   such a large size that individual pixels become visible. When applied to an image, pixelation reduces the resolution
   of the image by increasing the size of individual pixels, creating a blocky, pixelated appearance.

   This function pixelates a Pygame surface by dividing the image into square blocks, with each block representing the
   average color of the pixels inside it. The block size is specified by the `blocksize_` parameter (default is 64).

   **Example usage**:

   .. code-block:: python

      # For 24 or 32-bit image
      pix_image = pixelation(image)

   **Parameters**:

   - `surface_` (pygame.Surface): The Pygame surface to be pixelated.
   - `blocksize_` (unsigned int, optional): The block size used for the pixelation process, default is 64.
     This value determines the size of each pixelated block (e.g., 64x64 pixel blocks).

   **Returns**:

   - `pygame.Surface`: A new surface with the pixelation effect applied.

|

.. function:: blend(source, destination, percentage)

   Alpha Blending

   Blend two images together using alpha blending, where one image is overlaid on top of another with a specified
   transparency level. The function allows you to blend the `source` image onto the `destination` image based on a
   percentage value, where 0% means no blending and 100% means full blending of the source image over the destination.

   **Example usage**:

   .. code-block:: python

      # For 24 or 32-bit image
      transition = blend(source=image1, destination=image2, percentage=60)

   **Parameters**:

   - `source` (pygame.Surface): The source image (compatible with 24 or 32-bit).
   - `destination` (pygame.Surface): The destination image (compatible with 24 or 32-bit).
   - `percentage` (float): The percentage value between 0.0 and 100.0 that determines the level of blending.
     0.0 means the source image is completely transparent, and 100.0 means the source image completely replaces the
     destination image.

   **Returns**:

   - `pygame.Surface`: A new surface (24-bit) that represents the blended result of the `source` and `destination` images.

|

.. function:: blend1d(w, h, source, destination, percentage, modes, format_32=False)

   Alpha Blending (1D Array)

   Blend two 1D image buffers together using alpha blending. This method allows you to blend two images by
   providing the image buffers as the source and destination. The `percentage` argument allows you to control
   how much of each image contributes to the final result. For example, if `percentage` is set to 25%,
   the source image will contribute 25% while the destination image contributes 75%.

   The `modes` argument specifies the pixel format of the source and destination buffers, which can either
   be 'RGB(X)' or 'BGR(X)', where 'X' refers to the alpha channel if present. The `format_32` argument
   should be set to `True` if the source and destination buffers contain alpha transparency (RGBA format).

   **Example usage**:

   .. code-block:: python

      # For 32-bit images with alpha transparency
      im = blend1d(w, h, im.get_buffer(), BCK.get_buffer(), 25, 'BGR(X)', True)

      # For 24-bit images without alpha transparency
      im = blend1d(w, h, im.get_buffer(), BCK.get_buffer(), 25, 'BGR(X)', False)

   **Parameters**:

   - `w` (int): Width of the source array.
   - `h` (int): Height of the source array.
   - `source` (numpy.ndarray): 1D array of type uint8 representing the source image in RGB(A) or BGR(A) format.
   - `destination` (numpy.ndarray): 1D array of type uint8 representing the destination image in the same format as `source`.
   - `percentage` (float): Blending percentage value between 0 and 100. A value of 0 means only the destination image is visible, while 100 means only the source image is visible.
   - `modes` (str): A string representing the pixel format of the source and destination arrays. Use 'RGB(X)' for RGB or RGBA format and 'BGR(X)' for BGR or BGRA format.
   - `format_32` (bool): If `True`, the source and destination arrays contain alpha transparency (RGBA or BGRA format). Default is `False`.

   **Returns**:

   - `pygame.Surface`: A new surface with the blended effect applied, which can be either 24-bit or 32-bit depending on the source and destination array formats.

|

.. function:: blend_inplace(destination, source, percentage)

   Alpha Blending (Inplace)

   This function blends the `source` image into the `destination` image, modifying the destination image directly.
   The `source` and `destination` textures must be of the same size. The `percentage` argument controls how much of
   each image contributes to the final result. For example, if the `percentage` is set to 25%, the source image will
   contribute 25% while the destination image will contribute 75%.

   The function is compatible with both 24-bit and 32-bit surfaces.

   **Example usage**:

   .. code-block:: python

      blend_inplace(destination, source, percentage=50)

   **Parameters**:

   - `destination` (pygame.Surface): The surface that will be modified, compatible with 24-bit or 32-bit surfaces.
   - `source` (pygame.Surface): The surface to blend into the destination, compatible with 24-bit or 32-bit surfaces.
   - `percentage` (float): A value between 0.0 and 100.0 indicating the blending percentage. A value of 0.0 means
     no effect from the source, and 100.0 means the source completely replaces the destination.

   **Returns**:

   - `void`: The operation is performed inplace, meaning the destination surface is modified directly.

|

.. function:: cartoon(surface_, sobel_threshold=128, median_kernel=2, color=8, flag=BLEND_RGB_ADD)

   Apply a cartoon effect to an image.

   This function applies a cartoon-like effect to a given surface. It utilizes a Sobel filter for edge detection
   and combines it with median filtering and color reduction to create a simplified and stylized cartoon effect.
   The function is compatible with both 24-bit and 32-bit images.

   **Example usage**:

   .. code-block:: python

      cartoon_image = cartoon(image)

   **Parameters**:

   - `surface_` (pygame.Surface): The surface to which the cartoon effect will be applied. It can be either 24-bit or 32-bit.
   - `sobel_threshold` (unsigned int, optional): The Sobel filter threshold for edge detection. Default is 128.
   - `median_kernel` (unsigned int, optional): The size of the kernel used for median filtering. Default is 2.
   - `color` (unsigned int, optional): The maximum color reduction level (i.e., the number of colors in the image after reduction). Default is 8.
   - `flag` (unsigned int, optional): The blend flag for blending the final cartoon image. Default is `BLEND_RGB_ADD`.

   **Returns**:

   - `pygame.Surface`: The input surface with the cartoon effect applied.

|

.. function:: convert_27(surface_)

   Convert an image to a reduced color palette of 27 colors (inplace).

   This function reduces the color depth of an image by applying an algorithm that converts it
   to only 27 distinct colors. The operation is performed in place, meaning that the original
   surface is modified directly.

   **Example usage**:

   .. code-block:: python

      convert_27(image)

   **Parameters**:

   - `surface_` (pygame.Surface): The surface (image) that will be converted to a reduced
     color palette. It is compatible with both 24-bit and 32-bit image formats.

   **Returns**:

   - `void`: The function modifies the surface in place and does not return a new surface.

|

.. function:: bilateral(image, sigma_s, sigma_i, kernel_size=3)

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

   .. code-block:: python

      surface = bilateral(surface, 16.0, 18.0, 3)

   **Parameters**:

   - `image` (pygame.Surface): The surface (image) to which the bilateral
     filter will be applied. The image must be in 24-32 bit RGB format,
     and the alpha channel will be ignored. The image is converted into
     a 3D array for processing.
   - `sigma_s` (float): The spatial extent of the kernel. This parameter
     controls the size of the neighborhood used to compute the weighted average.
   - `sigma_i` (float): Intensity sensitivity. Defines the maximum intensity
     difference that contributes to the blur. Smaller values will preserve edges.
   - `kernel_size` (int, optional): The size of the kernel (default is 3).
     This controls how far the filter reaches from each pixel.

   **Returns**:

   - `pygame.Surface`: A new Pygame surface with the bilateral filter applied.

|

.. function:: emboss(surface_, flag_=0)

   Apply an emboss filter to an image or surface, producing an embossed effect.

   The embossing filter creates a visual effect that makes the image appear raised,
   similar to a paper or metal embossing of the original image, which can be used
   to highlight edges or create artistic effects.

   **Example usage**:

   .. code-block:: python

      image = emboss(image)
      image = emboss(image, 1)

   **Parameters**:

   - `surface_` (pygame.Surface): A Pygame surface compatible with 24-bit or 32-bit image formats.
   - `flag_` (int, optional): A special Pygame blend flag (default is 0). This flag can be one of
     the blend modes such as `BLEND_RGB_ADD`, `BLEND_RGB_MULT`, etc. It modifies the image blending
     behavior and will affect the final output.

   **Returns**:

   - `pygame.Surface`: A surface containing the embossed image. If `flag_` is set to 0, the output
     is a 24-bit format; otherwise, a 32-bit format is returned.

|

.. function:: emboss_inplace(surface_, copy=None)

   Emboss a surface (inplace)

   This function applies an embossing filter to an image in-place, meaning the image itself is modified
   rather than creating a new image. The embossing effect produces a raised, textured effect, similar to
   paper or metal embossing. It can highlight edges and create artistic effects in the image.

   **Example usage**:

   .. code-block:: python

      image = pygame.image.load('../Assets/px.png').convert(24)
      pixel_copy = numpy.ascontiguousarray(array3d(image_copy).transpose(1, 0, 2))
      emboss_inplace(image, copy=pixel_copy)

      image = pygame.image.load('../Assets/px.png').convert_alpha()
      pixel_copy = numpy.ascontiguousarray(array3d(image_copy).transpose(1, 0, 2))
      emboss_inplace(image, copy=pixel_copy)

   **Parameters**:

   - `surface_` (pygame.Surface): A Pygame surface that will be embossed. The surface will be modified
     in place once the process is complete.
   - `copy` (numpy.ndarray, optional): A numpy array with shape `(w, h, 3)` of type `uint8` containing
     the RGB pixels. It must be the same size as the input surface. This array is used as a copy of the
     source array to slightly improve performance.

   **Returns**:

   - `void`: The surface is modified in place and no new surface is returned.

|

.. function:: emboss1d(w, h, bgr_array, tmp_array=None, format_32=False)

   Emboss directly a C-buffer type (inplace)

   This function applies an embossing filter to an image in-place, directly modifying the provided
   C-buffer array (`bgr_array`). The embossing effect produces a raised texture, similar to paper or
   metal embossing, making edges more prominent and creating artistic effects.

   If the `tmp_array` is provided, it is used to improve performance, but it must have the same shape and size
   as the source array (`bgr_array`).

   **Example usage**:

   .. code-block:: python

      image = pygame.image.load('../Assets/px.png').convert(24)
      image = pygame.transform.smoothscale(image, (800, 600))
      image_copy = image.copy()
      emboss1d(800, 600, image.get_view('0'), image_copy.get_buffer(), False)

      image = pygame.image.load('../Assets/px.png').convert_alpha()
      image = pygame.transform.smoothscale(image, (800, 600))
      image_copy = image.copy()
      emboss1d(800, 600, image.get_view('0'), image_copy.get_buffer(), True)

   **Parameters**:

   - `w` (int): The width of the surface (image).
   - `h` (int): The height of the surface (image).
   - `bgr_array` (numpy.ndarray or memoryviewslice): A 1D array of type `uint8` containing BGR pixel data
     or any other pixel format. This array represents the source data that will be modified in place.
   - `tmp_array` (numpy.ndarray or memoryviewslice, optional): A 1D array of type `uint8` containing BGR
     pixel data or any other pixel format. This array should be a copy of the source array to improve performance.
   - `format_32` (bool, default is False): Set to `True` if the source array contains alpha transparency
     (32-bit format).

   **Returns**:

   - `void`: The function modifies the `bgr_array` in place and does not return a new array.

|

.. function:: emboss_gray(surface_)

   Apply a gray-scale embossing filter to an image or surface and return a modified copy.

   This function applies an embossing effect to the given image or surface in grayscale. The embossing effect
   creates a raised texture that mimics paper or metal embossing. The result is a stylized effect where edges
   and contours are emphasized, but in a monochromatic grayscale palette.

   **Example usage**:

   .. code-block:: python

      image = emboss_gray(image)

   **Parameters**:

   - `surface_` (pygame.Surface): The input surface (image) to which the gray-scale emboss effect will be applied.
     This surface must be compatible with 24-bit or 32-bit formats.

   **Returns**:

   - `pygame.Surface`: A new Pygame surface with the gray-scale embossed image. The resulting image is always in 24-bit format.

|

.. function:: bilinear(surface_, size_, fx=None, fy=None)

   Resize an image using the bilinear filter algorithm (returns a copy).

   This function applies the bilinear filter to resize an image. Bilinear filtering smooths the image
   and is commonly used in image resizing tasks. The function supports 32-bit input images, but the
   result is always returned in 24-bit format (without the alpha channel).

   **Example usage**:

   .. code-block:: python

      image = bilinear(image, (600, 600))  # Resize image to 600x600
      image = bilinear(image, (600, 600), 2, 2)  # Resize with specific scaling factors

   **Parameters**:

   - `surface_` (pygame.Surface): The input surface (image) to be resized. It must be compatible with 24-bit or 32-bit formats.
   - `size_` (tuple): A tuple `(width, height)` specifying the new dimensions for the resized surface.
   - `fx` (float, optional): A scaling factor for the x-axis (width). If provided, it will override the width specified in `size_`. Default is None.
   - `fy` (float, optional): A scaling factor for the y-axis (height). If provided, it will override the height specified in `size_`. Default is None.

   **Returns**:

   - `pygame.Surface`: A new Pygame surface of type 24-bit (without the alpha channel), resized based on the input parameters.

|

.. function:: tunnel_modeling24(screen_width, screen_height, surface_)

   24-bit Tunnel modeling
   This method will produce 24-bit rendering data to simulate a tunnel effect.

   The algorithm uses a 256x256 texture but reshapes it to 512x512 pixels for better rendering.
   It generates a tunnel effect by manipulating distances, angles, shades, and rendering data.

   **Example usage**:

   .. code-block:: python

      WIDTH = 800
      HEIGHT = 800
      BCK1 = pygame.image.load("../Assets/space2.jpg").convert(24)
      BCK1 = pygame.transform.smoothscale(BCK1, (WIDTH, HEIGHT))
      BACKGROUND = pygame.image.load("../Assets/space1.jpg")
      BACKGROUND = pygame.transform.smoothscale(BACKGROUND, (WIDTH, HEIGHT))
      distances, angles, shades, scr_data = tunnel_modeling24(WIDTH, HEIGHT, BACKGROUND)

   **Parameters**:

   - `screen_width` (int): The width of the display or the width of the tunnel effect.
   - `screen_height` (int): The height of the display or the height of the tunnel effect.
   - `surface_` (pygame.Surface): The tunnel texture effect, compatible with 24-bit or 32-bit formats.

   **Returns**:

   - `tuple`: A tuple containing four buffers:
     - `distances`: Buffer containing distance data.
     - `angles`: Buffer containing angle data.
     - `shades`: Buffer containing shading data.
     - `scr_data`: Buffer containing the screen rendering data.

|

.. function:: tunnel_render24(t, screen_width, screen_height, screen_w2, screen_h2, distances, angles, shades, scr_data, dest_array)

   Tunnel effect rendering

   This function renders the tunnel effect based on pre-calculated data, creating a 24-bit output surface.

   **Example usage**:

   .. code-block:: python

      surface_ = tunnel_render24(FRAME*5, WIDTH, HEIGHT, WIDTH >> 1, HEIGHT >> 1, distances, angles, shades, scr_data, dest_array)
      SCREEN.blit(surface_, (0, 0), special_flags=pygame.BLEND_RGB_ADD)

   **Parameters**:

   - `t` (int): Timer or frame count. Controls the speed of the tunnel effect.
   - `screen_width` (int): The width of the display or tunnel effect.
   - `screen_height` (int): The height of the display or tunnel effect.
   - `screen_w2` (int): The screen width divided by 2.
   - `screen_h2` (int): The screen height divided by 2.
   - `distances` (numpy.ndarray): A 1D C-buffer containing the distances. This buffer is obtained from the `tunnel_modeling24` function.
   - `angles` (numpy.ndarray): A 1D C-buffer containing the angles. This buffer is obtained from the `tunnel_modeling24` function.
   - `shades` (numpy.ndarray): A 1D C-buffer containing the shades. This buffer is obtained from the `tunnel_modeling24` function.
   - `scr_data` (numpy.ndarray): A 1D C-buffer containing the background pixels. This buffer is obtained from the `tunnel_modeling24` function.
   - `dest_array` (numpy.ndarray): A 1D C-buffer, typically empty. It should have the same size as the output image (width * height * 4 for RGBA). This buffer is used to build the final texture effect.

   **Returns**:

   - `pygame.Surface`: A 24-bit surface with the rendered tunnel effect.

|

.. function:: tunnel_modeling32(screen_width, screen_height, surface_)

   Generate 32-bit Tunnel Modeling Effect.

   This function simulates a tunnel effect and generates 32-bit rendering data based
   on the provided surface texture. It uses a 256x256 texture, which is reshaped to
   a 512x512 resolution for better rendering quality. The algorithm computes various
   parameters such as distances, angles, shades, and rendering data, which can be
   used to visualize the tunnel effect.

   **Example usage**:

   .. code-block:: python

      WIDTH = 800
      HEIGHT = 800
      BCK1 = pygame.image.load("../Assets/space2.jpg").convert(24)
      BCK1 = pygame.transform.smoothscale(BCK1, (WIDTH, HEIGHT))
      BACKGROUND = pygame.image.load("../Assets/space1.jpg")
      BACKGROUND = pygame.transform.smoothscale(BACKGROUND, (WIDTH, HEIGHT))
      distances, angles, shades, scr_data = tunnel_modeling32(WIDTH, HEIGHT, BACKGROUND)

   **Parameters**:

   - `screen_width` (int): The width of the display or the tunnel effect.
   - `screen_height` (int): The height of the display or the tunnel effect.
   - `surface_` (pygame.Surface): The texture surface used for the tunnel effect. The surface should be compatible with either 24 or 32-bit formats.

   **Returns**:

   - `tuple`: A tuple containing four buffers:
     - `distances`: A buffer representing the calculated distances for the effect.
     - `angles`: A buffer representing the calculated angles for the effect.
     - `shades`: A buffer representing the calculated shades (brightness) for the effect.
     - `scr_data`: A buffer containing the texture data of the surface.

|

.. function:: tunnel_render32(t, screen_width, screen_height, screen_w2, screen_h2, distances, angles, shades, scr_data, dest_array)

   Tunnel effect rendering.

   This function renders the tunnel effect using 32-bit rendering data. It takes
   the parameters calculated by the `tunnel_modeling32` function and applies them
   to generate the final tunnel effect. The output is a 32-bit surface that can
   be used in Pygame rendering.

   **Example usage**:

   .. code-block:: python

      surface_ = tunnel_render32(FRAME * 5, WIDTH, HEIGHT, WIDTH >> 1, HEIGHT >> 1, distances, angles, shades, scr_data, dest_array)
      SCREEN.blit(surface_, (0, 0), special_flags=pygame.BLEND_RGB_ADD)

   **Parameters**:

   - `t` (int): The timer or frame count. This controls the speed of the effect.
   - `screen_width` (int): The width of the display or the tunnel effect.
   - `screen_height` (int): The height of the display or the tunnel effect.
   - `screen_w2` (int): This is half the width of the screen (`screen_width // 2`).
   - `screen_h2` (int): This is half the height of the screen (`screen_height // 2`).
   - `distances` (numpy.ndarray): A C-buffer of distances, which is generated from the `tunnel_modeling32` function.
   - `angles` (numpy.ndarray): A C-buffer of angles, which is generated from the `tunnel_modeling32` function.
   - `shades` (numpy.ndarray): A C-buffer of shades (brightness), which is generated from the `tunnel_modeling32` function.
   - `scr_data` (numpy.ndarray): A C-buffer of background pixel data, which is generated from the `tunnel_modeling32` function.
   - `dest_array` (numpy.ndarray): An empty C-buffer, typically an empty buffer with a length of `width * height * 4` (RGBA). This buffer will be used to build the final tunnel effect image.

   **Returns**:

   - `pygame.Surface`: A surface containing the 32-bit image of the rendered tunnel effect.

|

.. function:: shader_bloom_fast(surface_, threshold, fast=False, factor=2)

   Applies a fast bloom effect to an input surface.

   The bloom effect brightens pixels in the image above a specified threshold
   and then applies a blur to create a glowing effect. This function performs a
   series of downscaling operations and blurs, which are then combined to produce
   the final bloom effect. The algorithm can be optimized for speed, trading
   off some visual quality.

   **Example usage**:

   .. code-block:: python

      image = shader_bloom_fast(image, 60)

   **Parameters**:

   - `surface_` (pygame.Surface):
     A Pygame surface containing RGB pixel data (32-bit or 24-bit color format).

   - `threshold` (int):
     The brightness threshold for the bloom effect. Pixels with values above
     this threshold will contribute to the bloom effect. A smaller value will
     cause a stronger bloom.

   - `fast` (bint, optional):
     If True, the algorithm will prioritize speed over visual quality by only
     applying the blur to the lowest downscaled surface (S16). Default is False.

   - `factor` (int, optional):
     A value between 0 and 4 that controls the level of downscaling for the
     textures used in the bloom. Higher values result in more aggressive downscaling.
     Default is 2, which corresponds to a division by 4.

   **Returns**:

   - `pygame.Surface`:
     A Pygame surface with the bloom effect applied, in 24-bit color format.

   **Raises**:

   - `ValueError`:
     If the surface is too small to process (e.g., after downscaling).

|

.. function:: shader_bloom_fast1(surface_, smooth_=3, threshold_=0, flag_=BLEND_RGB_ADD, saturation_=False, mask_=None)

   Bloom effect applied in-place (simplified version for better performance).

   The `shader_bloom_fast1` function applies a bloom effect that is optimized for moving objects
   in the display. Unlike other bloom methods, this version does not cause the light halo to be
   offset from moving objects, as it avoids the downsampling technique. This makes it better suited
   for dynamic scenes where objects are in motion.

   **Parameters**:

   - `surface_` (pygame.Surface):
     A Pygame surface with compatible 32-bit or 24-bit color format (RGB).

   - `smooth_` (int, optional):
     The smooth factor for spreading the halo of light. The default value is 3.
     Higher values spread the bloom over the entire scene but diminish the effect, while smaller values
     create a more pixelated halo but intensify the effect on objects. If smooth is below 3, the halo
     becomes pixelated.

   - `threshold_` (int, optional):
     The threshold for the bloom intensity. A threshold of 0 corresponds to the maximum bloom.
     The default value is 0, meaning no threshold is applied.

   - `flag_` (int, optional):
     A Pygame blend flag for special blending effects with the light in the display.
     The default is `pygame.BLEND_RGB_ADD`. Other options include `BLEND_RGB_MAX`, `BLEND_RGB_SUB`,
     and other Pygame blend attributes.

   - `saturation_` (bool, optional):
     If True, the bloom effect will include a saturation effect to the halo. The default is False.

   - `mask_` (numpy.ndarray or memoryviewslice, optional):
     A numpy array or memoryview representing the mask alpha. It should have the shape (w, h) and type
     `float32`, with values in the range (0..255). A mask filled with 255 will render and bloom the entire
     image, while a mask filled with 0 will disable the bloom effect. Values between 0 and 255 create a
     selective bloom effect. The mask is optional.

   **Returns**:

   - `void`:
     This effect is applied in-place, modifying the input surface directly.

   **Example usage**:

   .. code-block:: python

      shader_bloom_fast1(image)

|

.. function:: split_channels(surface_, offset_, array_=None)

   RGB split effect (returns a copy).

   This function applies an RGB split effect to an image by shifting the channels based on the provided
   offset. The result is an image with the red, green, and blue channels offset from their original positions.
   The offset value controls the direction and magnitude of the shift.

   **Parameters**:

   - `surface_` (pygame.Surface):
     A Pygame surface compatible with 24-bit or 32-bit color format (RGB). The input image on which the
     RGB split effect will be applied.

   - `offset_` (char):
     An integer value that specifies the offset to apply between the RGB channels. The offset must be in the
     range [-128, 127]. If the offset is positive, the channels are displayed in the BGR order. If the offset
     is negative, the channels are displayed in the RGB order.

   - `array_` (numpy.ndarray, optional):
     A numpy array with the shape (w, h, 3) and type `uint8`. If provided, this array will speed up the process
     by directly applying the transformation to it. This parameter is optional.

   **Returns**:

   - `pygame.Surface`:
     A new surface (copy) with the RGB split effect applied, where the RGB channels are offset according to the
     specified value.

   **Example usage**:

   .. code-block:: python

      image = split_channels(image, 10)

|

.. function:: split_channels_inplace(surface_, offset_, array_=None)

   RGB split effect (inplace).

   This function applies an RGB split effect to an image directly, modifying the original image without
   creating a new copy. The function shifts the red, green, and blue channels based on the provided offset
   value, either rearranging the channels or applying the specified offset.

   **Parameters**:

   - `surface_` (pygame.Surface):
     A Pygame surface compatible with 24-bit or 32-bit color format (RGB). The input image on which the
     RGB split effect will be applied, and the changes will be applied directly to this surface.

   - `offset_` (char):
     An integer value that specifies the offset to apply between the RGB channels. The offset must be in the
     range [-128, 127]. If the offset is positive, the channels are displayed in the BGR order. If the offset
     is negative, the channels are displayed in the RGB order.

   - `array_` (numpy.ndarray, optional):
     A numpy array with the shape (w, h, 3) and type `uint8`. If provided, this array will be used to speed up
     the process by directly applying the transformation to it. This parameter is optional.

   **Returns**:

   - `void`:
     The changes are applied directly to the input surface, so no new surface is returned.

   **Example usage**:

   .. code-block:: python

      split_channels_inplace(image, 10)

|

.. function:: wavelength2rgb(wavelength, gamma=1.0)

   Convert a wavelength to RGB color.

   This function maps a given wavelength (in nanometers) to an RGB color. The wavelength is interpreted
   within the visible spectrum (380–750 nm), and the corresponding color is returned in the RGB color model.
   The function optionally allows for gamma correction, which adjusts the color brightness.

   **Parameters**:

   - `wavelength` (int):
     The wavelength of light in nanometers (nm), typically within the visible spectrum range of 380–750 nm.
     The function uses this value to determine the corresponding color.

   - `gamma` (float, optional):
     A gamma correction factor applied to the color. The default value is `1.0`, meaning no correction is applied.
     Values greater than `1.0` increase the brightness of the color, while values less than `1.0` darken it.

   **Returns**:

   - `tuple`:
     A tuple of three integers representing the RGB components of the color, with each value in the range of 0 to 255.

   **Example usage**:

   .. code-block:: python

      # Returns RGB values for orange (255, 137, 0)
      wavelength2rgb(610)

      # Returns RGB values for red with gamma correction
      wavelength2rgb(620, gamma=1.2)

|

.. function:: custom_map(wavelength, color_array, gamma=1.0)

   Map a wavelength to a customized RGB color based on a user-defined wavelength domain.

   Unlike the `wavelength2rgb` function that returns RGB values corresponding to a wavelength
   within the standard visible spectrum (380–750 nm), this function allows the user to define
   a customized wavelength domain and returns the corresponding RGB color for a wavelength in that domain.

   **Parameters**:

   - `wavelength` (int):
     The wavelength of light in nanometers (nm). This value is mapped to the customized wavelength domain defined in `color_array`.

   - `color_array` (numpy.ndarray):
     A 1D array of integers representing the wavelength boundaries of different colors. The array should contain pairs of values for each color (e.g., [min, max] ranges for colors such as yellow, orange, red, etc.). These ranges define the domain in which the colors will be applied.

   - `gamma` (float, optional):
     A gamma correction factor applied to the resulting RGB color. The default value is `1.0`, which means no gamma correction. Values greater than `1.0` will brighten the color, while values less than `1.0` will darken it.

   **Returns**:

   - `tuple`:
     A tuple containing the RGB values (0-255) for the color corresponding to the given wavelength within the customized domain.

   **Example usage**:

   .. code-block:: python

      # Create a custom wavelength domain with different color gradients
      arr = numpy.array(
          [0, 1,       # Violet not used
           0, 1,       # Blue not used
           0, 1,       # Green not used
           570, 619,   # Yellow gradient from 2 to 619 nm
           620, 650,   # Orange gradient from 620 to 650 nm
           651, 660    # Red gradient from 651 to 660 nm
      ], dtype=numpy.int32)

      # Get RGB color for a wavelength of 600 nm
      rgb_color = custom_map(600, arr)

      # Generate a heatmap of colors for wavelengths from 380 nm to 799 nm
      heatmap = [custom_map(i - 20, arr, 1.0) for i in range(380, 800)]

|

.. function:: blue_map(wavelength, gamma=1.0)

   Map a specific wavelength to a blue-toned RGB color.

   This function takes a wavelength (in nanometers) and returns the corresponding RGB values
   with a focus on generating a blue color. It uses a simple mapping based on the provided
   wavelength and applies a gamma correction for more refined color adjustments.

   **Parameters**:

   - `wavelength` (int):
     The wavelength of light in nanometers (nm). This value is mapped to the blue-toned RGB value.

   - `gamma` (float, optional):
     A gamma correction factor applied to the resulting RGB color. The default value is `1.0`, which means no gamma correction. Values greater than `1.0` will brighten the color, while values less than `1.0` will darken it.

   **Returns**:

   - `tuple`:
     A tuple containing the RGB values (0-255) for the blue-toned color corresponding to the given wavelength.

   **Example usage**:

   .. code-block:: python

      # Get the blue RGB color correspondi

|

.. function:: bluescale(surface_)

   Apply a blue color filter to an image, transforming it into a blue-toned version.

   This function uses the wavelength range from 450 to 495 nm, which corresponds to
   blue light in the visible spectrum, to convert the image into various shades of blue.

   **Parameters**:

   - `surface_` (pygame.Surface):
     The Pygame surface (image) that will be transformed into blue shades.

   **Returns**:

   - `void`:
     This function modifies the provided surface in place, so there is no return value.

   **Example usage**:

   .. code-block:: python

      # Transform an image into shades of blue
      bluescale(image)

|

.. function:: red_map(wavelength, gamma=1.0)

   Return the RGB components corresponding to a specific wavelength in the red spectrum.

   This function maps a given wavelength (in nm) to the appropriate RGB values for
   the red portion of the visible spectrum. It can be used to generate red-based
   colors based on a given wavelength.

   **Parameters**:

   - `wavelength` (int):
     The wavelength in nanometers (nm) for which the corresponding RGB values are calculated.

   - `gamma` (float, optional):
     A gamma correction value to adjust the brightness of the colors. The default is `1.0` (no correction).

   **Returns**:

   - `tuple` (int, int, int):
     A tuple of RGB components as integers in the range (0 ... 255), corresponding to the specified wavelength.

   **Example usage**:

   .. code-block:: python

      # Get the RGB color for a wavelength of 610 nm (Red color)
      rgb = red_map(610)

|

.. function:: redscale(surface_)

   Apply a redscale effect to an image.

   This algorithm maps the input image into shades of red by adjusting the
   color channels according to wavelengths typically associated with the
   red portion of the visible light spectrum, ranging from 620 to 750 nm.
   The redscale effect retains the intensity variations of the red color while
   reducing or removing the influence of other color channels.

   **Parameters**:

   - `surface_` (pygame.Surface):
     A Pygame surface representing the image to be transformed into redscale.
     The surface must contain valid RGB pixel data.

   **Returns**:

   - `void`:
     This function modifies the input surface directly and does not return a new surface.

   **Example usage**:

   .. code-block:: python

      # Apply redscale effect to an image
      redscale(image)

|

.. function:: dampening(surface_, frame, display_width, display_height, amplitude=50.0, duration=30, freq=20.0)

   Apply a dampening effect to a surface.

   This effect simulates a gradual scaling or shrinking of the surface based on a
   damped oscillation function. The length of the effect is determined by the product
   of duration and frequency. The position of the surface is adjusted according to
   its new size to maintain its centered position on the display.

   **Parameters**:

   - `surface_` (pygame.Surface):
     A Pygame surface that is compatible with 24-32 bit color depth.

   - `frame` (int):
     The current frame number in the animation sequence. This should be incremented
     with each frame update.

   - `display_width` (int):
     The width of the game display window.

   - `display_height` (int):
     The height of the game display window.

   - `amplitude` (float, optional):
     The amplitude of the dampening effect, which determines the maximum amount
     of scaling. Default is 50.0.

   - `duration` (int, optional):
     The duration of the effect, which controls how long the effect lasts
     in terms of frames. Default is 30.

   - `freq` (float, optional):
     The frequency of the dampening effect, which affects how fast the scaling
     oscillates. A smaller value will make the effect last longer, while a larger
     value shortens the effect. Default is 20.0.

   **Returns**:

   - tuple:
     A tuple containing:

       - A new Pygame Surface with the dampening effect applied.
       - The x-coordinate of the new position of the surface (top-left corner).
       - The y-coordinate of the new position of the surface (top-left corner).

       The surface is centered in the display area.

   **Example usage**:

   .. code-block:: python

      # Apply dampening effect to a surface
      surf, xx, yy = dampening(BCK, frame, w, h, amplitude=100, duration=40, freq=15)
      SCREEN.blit(surf, (xx, yy))

|

.. function:: lateral_dampening(frame, amplitude=50.0, duration=30, freq=20.0)

   Apply lateral dampening effect to produce horizontal displacement.

   This method calculates the lateral displacement (x-coordinate) based on a dampened
   oscillation function. The displacement value oscillates between positive and negative
   values, gradually decaying according to the amplitude, frequency, and duration parameters.

   **Parameters**:

   - `frame` (int):
     The current frame number in the animation sequence. This value must be incremented
     with each frame to produce smooth animation.

   - `amplitude` (float, optional):
     The amplitude of the lateral dampening effect. This value controls the maximum
     displacement of the surface. A higher value results in larger horizontal movement.
     Default is 50.0.

   - `duration` (int, optional):
     The total duration of the effect, in terms of frames. This defines how long the
     oscillations will last. The default value is 30 frames.

   - `freq` (float, optional):
     The frequency of the dampening oscillation. This controls how fast the oscillations
     occur. A lower value makes the effect take longer to complete (slower oscillation),
     while a higher value speeds up the oscillation. Default is 20.0.

   **Returns**:

   - float:
     The lateral displacement value (x) that can be used to shift the object horizontally
     on the screen (e.g., when blitting an image). The value will oscillate within a
     range determined by the amplitude.

   **Example usage**:
   .. code-block:: python

      # Apply lateral dampening effect to a surface
      tm = lateral_dampening(frame, amplitude=50.0, duration=35, freq=5.0)
      SCREEN.blit(BCK, (tm, 0), special_flags=0)

   **Notes**:

   - The displacement follows a damped oscillation model, where the value decays
     over time based on the frequency and duration parameters.

|

.. function:: alpha_blending(source, destination)

   Perform alpha blending of two 32-bit images.

   This function blends two 32-bit images together, taking into account their
   alpha channels (transparency). The `source` image is drawn on top of the
   `destination` image. The blending uses the alpha channel to determine how
   transparent each pixel is in the source image.

   **Parameters**:

   - `source` (pygame.Surface):
     The source image to be blended with the destination image. It must be a
     32-bit image with an alpha channel.

   - `destination` (pygame.Surface):
     The destination image onto which the source image will be blended. It
     must also be a 32-bit image with an alpha channel.

   **Returns**:

   - pygame.Surface:
     A new 32-bit Pygame surface with the two images blended together, based
     on their alpha values.

   **Example usage**:

   .. code-block:: python

      # Blend two images together using alpha blending
      new_image = alpha_blending(source, destination)

   **Notes**:

   - This function returns a new surface and does not modify the original `source` or `destination` images.
   - Both images must have an alpha channel to ensure proper blending of transparency.

|

.. function:: alpha_blending_inplace(image1, image2)

   Perform alpha blending on two images in-place.

   This function blends two images with the same dimensions together, modifying
   the first image (`image1`). The blending is performed based on the alpha channel,
   where the second image (`image2`) is blended into the first image.

   Both images must be 32-bit with an alpha channel and should have the same dimensions.
   The blending process uses the alpha values of both images to calculate the transparency
   of the pixels.

   **Parameters**:

   - `image1` (pygame.Surface):
     The first image to be blended with the second image. It must be a 32-bit
     image with an alpha channel. This image is modified in place.

   - `image2` (pygame.Surface):
     The second image to be blended into the first image. It must also be a 32-bit
     image with an alpha channel.

   **Returns**:

   - None:
     This function modifies the first image (`image1`) in place and does not return a new surface.

   **Example usage**:

   .. code-block:: python

      # Blend two images together in-place
      alpha_blending_inplace(source, destination)

   **Notes**:

   - This function performs blending in place, modifying the `image1` surface.
   - Both `image1` and `image2` must have the same dimensions and include an alpha channel.
   - If either of the images does not have an alpha channel or have mismatched dimensions,
     an exception will be raised.

|

.. function:: render_light_effect24(x, y, background_rgb, mask_alpha, intensity=1.0, color=numpy.asarray([128.0, 128.0, 128.0], dtype=numpy.float32), smooth=False, saturation=False, sat_value=0.2, bloom=False, threshold=110, heat=False, frequency=1)

   Generates a realistic lighting effect on a Pygame surface or texture.

   This function simulates a light effect that can be blended onto a surface
   using additive blending (`BLEND_RGBA_ADD` in Pygame). The effect is generated
   based on a mask texture and can include optional enhancements such as bloom,
   heat waves, and saturation adjustments.

   **Example Usage**:

   .. code-block:: python

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

   **Parameters**:

   - `x` (int):
     The X-coordinate of the light source. It must be within the screen width.

   - `y` (int):
     The Y-coordinate of the light source. It must be within the screen height.

   - `background_rgb` (numpy.ndarray, shape (w, h, 3), dtype uint8):
     A 3D NumPy array representing the RGB values of the background surface.

   - `mask_alpha` (numpy.ndarray, shape (w, h), dtype uint8):
     A 2D NumPy array containing the alpha (transparency) values of the light mask.
     Using a radial gradient mask with maximum intensity at the center is recommended.

   - `intensity` (float, optional, default=1.0):
     Light intensity in the range [0.0, 20.0]. Higher values produce a stronger effect.

   - `color` (numpy.ndarray, optional, default=[128.0, 128.0, 128.0]):
     A 3-element NumPy array representing the RGB color of the light in float format
     (values in the range [0.0, 255.0]).

   - `smooth` (bool, optional, default=False):
     If `True`, applies a blur effect to smooth the lighting.

   - `saturation` (bool, optional, default=False):
     If `True`, applies a saturation effect to enhance color vibrancy.

   - `sat_value` (float, optional, default=0.2):
     Adjusts the saturation level. The valid range is [-1.0, 1.0].

   - `bloom` (bool, optional, default=False):
     If `True`, enables a bloom effect, which enhances the brightness of intense areas.

   - `threshold` (int, optional, default=110):
     The brightness threshold used in the bloom effect. Pixels above this value
     contribute to the bloom.

   - `heat` (bool, optional, default=False):
     If `True`, applies a heat wave effect that distorts the lighting dynamically.

   - `frequency` (float, optional, default=1):
     Determines the frequency of the heat wave effect. Must be an increasing value.

   **Returns**:

   - `tuple`:
     A tuple containing:

       - A 24-bit Pygame surface representing the generated light effect.
       - The surface width (`sw`).
       - The surface height (`sh`).

   **Notes**:

   - The output surface does not contain per-pixel alpha information.
   - Use `BLEND_RGBA_ADD` when blitting the surface to achieve an additive lighting effect.

|

.. function:: bloom_effect_array24_c2(surface_, threshold_, smooth_=1, mask_=None, fast_=False)

   Create a bloom effect on a Pygame surface (compatible 24-bit surface).

   **Definition**:

   Bloom is a computer graphics effect used in video games, demos, and high dynamic range rendering
   to reproduce an imaging artifact of real-world cameras. The effect simulates the bright areas of an image
   glowing and spreading to adjacent areas, enhancing the lighting and visual quality of the scene.

   **Example Usage**:

   .. code-block:: python

      image = pygame.image.load('../Assets/px.png').convert_alpha()
      image = pygame.transform.smoothscale(image, (800, 600))

      mask = pygame.image.load('../Assets/alpha.png').convert_alpha()
      mask = pygame.transform.smoothscale(mask, (800, 600))

      # In the main loop
      image = bloom_effect_array24_c2(image, 0, 1, mask, True)

   **Parameters**:

   - `surface_` (pygame.Surface):
     A Pygame surface in 24-bit format on which the bloom effect will be applied.

   - `threshold_` (unsigned char):
     Threshold value used by the bright pass algorithm. Default is 128, and it determines which pixels are considered
     "bright enough" to contribute to the bloom effect.

   - `smooth_` (int, optional, default=1):
     The number of Gaussian blur (5x5) iterations to apply to downscaled images. Increasing this value will
     create a stronger smoothing effect.

   - `mask_` (pygame.Surface, optional, default=None):
     A Pygame surface representing the mask alpha. The alpha values in this mask determine how much of the
     surface should be affected by the bloom effect. Alpha values of 255 will fully render and bloom the image,
     while zero will hide the corresponding pixels.

   - `fast_` (bool, optional, default=False):
     If `True`, speeds up the bloom process by using a 16x surface and an optimized bright pass filter,
     which involves downscaling the texture size by a factor of 4 before processing.

   **Returns**:

   - `pygame.Surface`:
     A new Pygame surface with the applied bloom effect (24-bit surface).

   **Notes**:

   - The function applies a bloom effect by first performing a bright pass on the image, then applying
     Gaussian blur and blending the result with the original surface.
   - The `mask_` parameter can be used to control which parts of the image will bloom based on the alpha values.
   - The `fast_` parameter allows for a quicker, albeit lower-quality, version of the bloom effect.

|

.. function:: area24_cc(x, y, background_rgb, mask_alpha, intensity=1.0, color=numpy.asarray([128.0, 128.0, 128.0], dtype=numpy.float32), smooth=False, saturation=False, sat_value=0.2, bloom=False, bloom_threshold=64)

   Generates a realistic lighting effect on a Pygame surface or texture.

   This function simulates a light effect that can be blended onto a surface using additive blending (`BLEND_RGBA_ADD` in Pygame). It supports optional enhancements such as bloom, smoothing, and saturation adjustments.

   **Lighting Modes**:
   --
   - **Smooth**: Applies a Gaussian blur with a 5x5 kernel to soften the lighting effect.
   - **Saturation**: Adjusts color intensity using HSL color conversion. A value range of [-1.0, 1.0] is supported, with higher values increasing vibrancy and values below zero desaturating the output.
   - **Bloom**: Enhances brightness by applying a bloom effect, making bright areas appear more intense.

   **Lighting Parameters**:
   -
   - **Intensity**: Defines the brightness of the light. If set to zero, the function returns an empty `pygame.Surface` with the `RLEACCEL` flag.
   - **Color**: Specifies the light’s RGB coloration. Defaults to (128, 128, 128).

   **Example Usage**:

   
   .. code-block:: python

      lit_surface, sw, sh = area24_cc(
          MOUSE_POS[0], MOUSE_POS[1], background_rgb, lalpha,
          intensity=5, color=c, smooth=False, saturation=False,
          sat_value=0.2, bloom=True, bloom_threshold=0
      )

   **Parameters**:

   - `x` (int):
     X-coordinate of the light source (must be within screen width).

   - `y` (int):
     Y-coordinate of the light source (must be within screen height).

   - `background_rgb` (numpy.ndarray, shape (w, h, 3), dtype uint8):
     A 3D NumPy array representing the RGB values of the background surface.

   - `mask_alpha` (numpy.ndarray, shape (w, h), dtype uint8):
     A 2D NumPy array containing the alpha values of the light mask. Using a radial gradient mask with maximum intensity at the center is recommended.

   - `color` (numpy.ndarray, optional, default=[128.0, 128.0, 128.0]):
     A 3-element NumPy array representing the RGB color of the light in float format (values in the range [0.0, 255.0]).

   - `intensity` (float, optional, default=1.0):
     Light intensity in the range [0.0, 20.0]. Higher values produce a stronger effect.

   - `smooth` (bool, optional, default=False):
     If `True`, applies a blur effect to soften the lighting.

   - `saturation` (bool, optional, default=False):
     If `True`, increases color intensity using HSL conversion.

   - `sat_value` (float, optional, default=0.2):
     Adjusts the saturation level. The valid range is [-1.0, 1.0]. Higher values increase vibrancy, while negative values desaturate the effect.

   - `bloom` (bool, optional, default=False):
     If `True`, enables a bloom effect, enhancing brightness.

   - `bloom_threshold` (unsigned char, optional, default=64):
     The brightness threshold for the bloom effect, in the range [0, 255]. Lower values create a stronger bloom effect.

   **Returns**:
   
   tuple:
     A tuple containing:
     - A 24-bit Pygame surface representing the generated light effect.
     - The surface width (`sw`).
     - The surface height (`sh`).

   **Notes**:
   
   - The output surface does not contain per-pixel alpha information.
   - Use `BLEND_RGBA_ADD` when blitting the surface to achieve an additive lighting effect.

|

.. function:: chromatic(surface_, delta_x, delta_y, zoom=0.9999, fx=0.02)

   Chromatic aberration (returns a new surface).

   This function applies a chromatic aberration effect to a given surface. The amplitude of the effect is proportional to the distance from the center of the effect, creating a distortion of the RGB color channels. The chromatic aberration effect simulates the color shift that occurs due to optical imperfections in lenses.

   **Example Usage**:
   
   .. code-block:: python

      source = chromatic(source, 400, 300, 0.999, fx=0.04)

   **Parameters**:
   
   - `surface_` (pygame.Surface):
     A 24-bit or 32-bit compatible Pygame surface to which the chromatic aberration effect will be applied.

   - `delta_x` (unsigned int):
     The X-coordinate of the chromatic center effect. This value must be within the range [0, surface width].

   - `delta_y` (unsigned int):
     The Y-coordinate of the chromatic center effect. This value must be within the range [0, surface height].

   - `zoom` (float, optional, default=0.9999):
     The zoom factor for the effect. A value of 0.9999 means no zoom (full image), while a value less than 1.0 will zoom in. It must be in the range [0.0, 0.9999].

   - `fx` (float, optional, default=0.02):
     The offset applied to the RGB channels. This value determines the intensity of the chromatic aberration effect and must be in the range [0.0, 0.2].

   **Returns**:
   
   pygame.Surface:
     A new Pygame surface with the chromatic aberration effect applied.

   **Notes**:
   
   - The function returns a new surface with the chromatic aberration effect applied.
   - The `delta_x` and `delta_y` parameters define the point at which the chromatic effect is centered, and the distortion becomes more pronounced as the distance from this point increases.
   - The `zoom` and `fx` parameters control the extent and intensity of the effect, respectively.

|

.. function:: chromatic_inplace(surface_, delta_x, delta_y, zoom=0.9999, fx=0.02)

   Chromatic aberration (inplace).

   This function applies a chromatic aberration effect to a given surface in place. The amplitude of the effect is proportional to the distance from the center of the effect. Unlike the `chromatic` function, this function modifies the original surface rather than returning a new surface. It simulates the optical effect of color distortion typically observed due to lens imperfections.

   **Example Usage**:
   
   .. code-block:: python

      surf = chromatic_inplace(background, MOUSE_POS.x, MOUSE_POS.y, 0.999, fx=0.04)

   **Parameters**:
   
   - `surface_` (pygame.Surface):
     A 24-bit or 32-bit compatible Pygame surface to which the chromatic aberration effect will be applied in place.

   - `delta_x` (unsigned int):
     The X-coordinate of the chromatic center effect. This value must be within the range [0, surface width].

   - `delta_y` (unsigned int):
     The Y-coordinate of the chromatic center effect. This value must be within the range [0, surface height].

   - `zoom` (float, optional, default=0.9999):
     The zoom factor for the effect. A value of 0.9999 means no zoom (full image), while a value less than 1.0 will zoom in. It must be in the range [0.0, 0.9999].

   - `fx` (float, optional, default=0.02):
     The offset applied to the RGB channels. This value determines the intensity of the chromatic aberration effect and must be in the range [0.0, 0.2].

   **Returns**:
   
   pygame.Surface:
     The same Pygame surface with the chromatic aberration effect applied in place.

   **Notes**:
   
   - The function modifies the original surface rather than returning a new one.
   - The `delta_x` and `delta_y` parameters define the point at which the chromatic effect is centered. The effect becomes more pronounced the farther away the pixel is from this center point.
   - The `zoom` and `fx` parameters control the extent and intensity of the chromatic aberration effect.
   - This function is useful when you want to apply the chromatic aberration effect directly to a surface without needing to create a new surface.

|

.. function:: zoom(surface_, delta_x, delta_y, zx=0.9999)

   Zoom within an image (return a copy).

   This function zooms in or zooms out on an image toward a given center point (delta_x, delta_y) using a specified zoom factor (zx). The function supports 24-bit and 32-bit image formats, and the output image will have the same format as the input. The zoom effect is more performant on 24-bit images.

   **Example Usage**:
   
   .. code-block:: python

      surf = zoom(background, MOUSE_POS.x, MOUSE_POS.y, z)

   **Parameters**:
   
   - `surface_` (pygame.Surface):
     The Pygame surface to apply the zoom effect on. This can be a 24-bit or 32-bit compatible surface.

   - `delta_x` (unsigned int):
     The X-coordinate of the zoom center. This value must be within the range [0, surface width].

   - `delta_y` (unsigned int):
     The Y-coordinate of the zoom center. This value must be within the range [0, surface height].

   - `zx` (float, optional, default=0.9999):
     The zoom factor. This value must be in the range (0.0, 1.0). A value of 1.0 will result in no zoom (full image), while values closer to 0.0 zoom in. The zoom intensity is attenuated around 1.0, with the maximum zoom effect occurring as zx approaches 0.0.

   **Returns**:
   
   pygame.Surface:
     A new Pygame surface with the zoom effect applied. The output image will have the same format (24-bit or 32-bit) as the input image.

   **Notes**:
   
   - The zoom effect centers around the point specified by `delta_x` and `delta_y`. The pixels closer to this center will appear less zoomed, while those farther away will experience more pronounced zooming.
   - This function returns a new surface with the zoom effect applied, leaving the original surface unchanged.
   - Performance is generally better with 24-bit images due to fewer data channels to process.

|

.. function:: zoom_inplace(surface_, delta_x, delta_y, zx=0.9999)

   Zoom within an image (inplace).

   This function zooms in or zooms out on an image toward a given center point (delta_x, delta_y) using a specified zoom factor (zx), modifying the image in place. The function supports 24-bit and 32-bit image formats, and the input image format remains unchanged during the process.

   **Example Usage**:
   
   .. code-block:: python

      zoom_inplace(background, MOUSE_POS.x, MOUSE_POS.y, z)

   **Parameters**:
   
   - `surface_` (pygame.Surface):
     The Pygame surface to apply the zoom effect on. This can be a 24-bit or 32-bit compatible surface.

   - `delta_x` (unsigned int):
     The X-coordinate of the zoom center. This value must be within the range [0, surface width].

   - `delta_y` (unsigned int):
     The Y-coordinate of the zoom center. This value must be within the range [0, surface height].

   - `zx` (float, optional, default=0.9999):
     The zoom factor. This value must be in the range (0.0, 1.0). A value of 1.0 will result in no zoom (full image), while values closer to 0.0 zoom in. The zoom effect is attenuated around 1.0, with the maximum zoom effect occurring as zx approaches 0.0.

   **Returns**:
   
   None:
     This function modifies the `surface_` in place. It does not return a new surface.

   **Notes**:
   
   - The zoom effect centers around the point specified by `delta_x` and `delta_y`. The pixels closer to this center will appear less zoomed, while those farther away will experience more pronounced zooming.
   - This function modifies the input surface directly, without creating a new one.
   - Performance is generally better with 24-bit images due to fewer data channels to process.

|

.. function:: Luma_GreyScale(surface_)

   Convert image into greyscale using YIQ (luma information).

   This function converts the provided image into a greyscale image by utilizing the YIQ color model, which focuses on the luma (brightness) information. The function modifies the input image in place.

   **Example Usage**:
   
   .. code-block:: python

      Luma_GreyScale(image)

   **Parameters**:
   
   - `surface_` (pygame.Surface):
     The Pygame surface to convert into greyscale. This surface can be in any color format supported by Pygame.

   **Returns**:
   
   None:
     The function modifies the input `surface_` in place, converting it to greyscale. It does not return a new surface.

   **Notes**:
   
   - The conversion uses the YIQ color model, specifically the luma (Y) component, which represents the brightness of the image.
   - The input surface is modified directly, and no new surface is returned.


