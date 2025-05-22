GaussianBlur5x5
========================================

:mod:`GaussianBlur5x5.pyx`

=====================

.. currentmodule:: GaussianBlur5x5

|

1. Summary of the Library
-------------------------

This Python library provides fast and efficient **image processing** functions,
focusing on **blurring**, **Gaussian filtering**, and **edge detection**.
It is implemented in **Cython** for high performance, making it ideal for applications
in **computer vision**, **graphics**, and **image preprocessing**.

2. Purpose
----------

The library is designed to perform fast blurring and edge detection, which are essential for:

- **Noise reduction** (smoothing images).
- **Edge detection** (for object recognition and feature extraction).
- **Graphics effects** (motion blur, bloom effects).
- **Preprocessing for computer vision tasks** (image segmentation, filtering).

3. Main Features
----------------

**Blurring Functions**

These functions apply various types of blurring to smooth images:

- **`blur(surface_, npass)`**:

Blurs an image surface, where `npass` controls intensity.

- **`blur3d(rgb_array, npass)`**:

Blurs a 3D (RGB) image array.

- **`blur1d(bgr_array, width, height, npass, format_32)`**:

Optimized 1D blurring for efficiency.

- **`blur1d_cp(bgr_array, width, height, npass, format_32)`**:

Returns a new blurred array instead of modifying in-place.

- **`blur4bloom_c(surface_, npass, tmp_array)`**:

Likely used for bloom effects, enhancing bright areas.

Several internal **Cython-optimized** versions (`blur3d_c`, `blur1d_c`, etc.) run without the **Global Interpreter Lock (GIL)** for multi-threading support.

**Gaussian Filtering**

- **`gauss_filter24(surface_)`**:

Applies Gaussian blur to a 24-bit image.

- **`gauss_filter32(surface_)`**:

Applies Gaussian blur to a 32-bit image.

📌 **Why Gaussian Blur?**

It helps smooth images before edge detection, reducing false edges caused by noise.

**Edge Detection (Canny Algorithm)**

The **Canny algorithm** is widely used in computer vision to detect object boundaries.

- **`canny3d(rgb_array, threshold, low, high)`**:

Detects edges in an RGB image.

- **`canny1d(bgr_array, width, height, format_32, threshold)`**:

Detects edges in a linear 1D image representation for efficiency.

There are also **Cython-optimized** versions (`canny3d_c`, `canny1d_c`) that improve performance using multi-threading.

**Optimization Features**

- **Cython-based** (`cdef`, `cpdef`, `nogil`) → Direct C-level performance.
- **In-place operations** (`blur1d`, `blur3d`) → Saves memory.
- **Multi-threading** (`nogil`) → Faster execution on multi-core processors.
- **Optimized memory handling** (`[::1]` memory views) → Reduces Python overhead.

4. Use Cases
------------

- **Computer Vision** → Object recognition, feature extraction.
- **Graphics & Gaming** → Motion blur, bloom effects.
- **Image Processing Pipelines** → Preprocessing before machine learning models.
- **Medical Imaging** → Enhancing and detecting features in scans.

5. Conclusion
-------------

This library is highly optimized for fast blurring, Gaussian filtering, and edge detection,
making it a great choice for **computer vision**, **graphics**, and **machine learning
applications** where speed is critical.


6. Cython list methods
----------------------

|

.. py:function:: blur(surface_, npass=1)

    |

    **Parameters**:
    - `surface_` (*pygame.Surface*): A Pygame surface object (24-bit or 32-bit).
    - `npass` (*int*, optional): The number of blur passes to apply (default is 1). Must be a positive integer.

    **Returns**: None

    **Description**:
    Apply an in-place Gaussian blur effect to a Pygame surface.

    This function performs a **two-pass convolution** using a **5x5 Gaussian kernel**
    to blur the input surface. The first pass applies the blur horizontally, and the
    second pass applies the blur vertically. For pixels outside the image boundaries,
    the function assigns the nearest edge value to reduce visual artifacts.


    .. math::

        \text{Horizontal pass: }
        \begin{bmatrix}
        \frac{1}{16} & \frac{4}{16} & \frac{6}{16} & \frac{4}{16} & \frac{1}{16}
        \end{bmatrix}

    .. math::

        \text{Vertical pass: }
        \begin{bmatrix}
        \frac{1}{16} \\ \frac{4}{16} \\ \frac{6}{16} \\ \frac{4}{16} \\ \frac{1}{16}
        \end{bmatrix}


    **Features**

    - Supports **24-bit (RGB)** and **32-bit (RGBA)** Pygame surfaces.
    - Uses a **5x5 Gaussian kernel** for smoothing the image.
    - Allows **multiple passes** (`npass > 1`) for enhanced blur intensity.
    - Ignores the **alpha channel** during processing to avoid altering transparency.

    **Parameters**

    - **surface_** (*pygame.Surface*):
      A Pygame surface object to which the blur effect will be applied. It must be either
      **24-bit RGB** or **32-bit RGBA**. The function modifies the surface in-place.

    - **npass** (*int*, optional):
      The number of blur passes to apply. Default is **1**.
      Must be a **positive integer** (`npass > 0`).
      Multiple passes result in a stronger blur effect but may increase processing time.

    **Returns**

    - **None** – The function modifies `surface_` in place, with no return value.

    **Raises**

    - **TypeError**:
      If `surface_` is not a valid `pygame.Surface` object.

    - **ValueError**:
      If the surface format is incompatible (e.g., non-24-bit or non-32-bit format) or cannot be processed.

    - **ValueError**:
      If `npass` is not a positive integer (`npass <= 0`).

    **Implementation Details**

    - The function accesses the pixel buffer of the surface directly using `get_view('3')`,
      which allows it to process **only the RGB channels**, ignoring the alpha channel.
    - It calls an optimized Cython function, `blur3d_c()`, to perform the in-place blurring with minimal overhead.
    - If `npass > 1`, multiple blur passes are applied for a stronger effect, but this increases the processing time.

    **Memory Efficiency**

    - This function is **memory-efficient**, as it operates directly on the surface buffer without creating additional copies of the image data.
    - If the surface has an **alpha channel**, it is **ignored** during processing, meaning any transparency information will remain unchanged.

    **Performance Considerations**

    - The function is optimized using Cython to minimize overhead and improve performance.
    - It supports **multi-threading** when using Cython's `nogil` feature for enhanced execution speed on multi-core systems.

    **Example Usage**

    .. code-block:: python

        import pygame
        from your_module import blur

        # Create a 24-bit Pygame surface
        surface = pygame.Surface((100, 100))

        # Apply a single blur pass (default)
        blur(surface)

        # Apply three blur passes for a stronger blur effect
        blur(surface, npass=3)


    **See Also**

    - `blur3d_c`: Optimized Cython function for 3D image blurring.
    - `gaussian_filter`: Applies a Gaussian blur with more control over the kernel size.

    |

.. py:function:: blur3d(rgb_array, npass=1)

    |

    **Parameters**:
    - `rgb_array` (*numpy.ndarray* or *memoryview slice*): A 3D array of shape `(w, h, n)`, representing an image in RGB(A) format.
    - `npass` (*int*, optional): The number of blur passes to apply (**default is `1`**). More passes result in a stronger blur.

    **Returns**: None

    **Description**

    Apply an in-place Gaussian blur to a **3D image array or memoryview slice**.

    This function performs a **Gaussian blur** on an image represented as a 3D array with shape `(w, h, n)`,
    where `w` and `h` are the width and height, and `n` is the number of color channels (typically **3 for RGB** or **4 for RGBA**).
    It uses a **5x5 Gaussian kernel** for smoothing, applied through **two-pass convolution** (horizontal + vertical).
    The function modifies the input array in-place, avoiding unnecessary memory allocation.

    **Features**

    - Supports **RGB(A) image data** with any number of channels (`n = 3` for RGB, `n = 4` for RGBA).
    - Performs **Gaussian blur** using a **5x5 convolution kernel**.
    - Uses **two-pass convolution** (horizontal and vertical) for an efficient blur.
    - Handles **edge pixels** by setting them to the nearest edge values.
    - Allows **multiple blur passes** (`npass > 1`) for increased smoothing.

    **Parameters**

    - **rgb_array** (*numpy.ndarray* or *memoryview slice*):
      A **3D array** of shape `(w, h, n)`, containing **uint8** values representing the image in **RGB(A) format**.
      The function modifies this array in-place.

    - **npass** (*int*, optional):
      The number of blur passes to apply. **Default is `1`**.
      Higher values will **increase blur strength** but require more processing time.

    **Returns**

    - **None** – The function modifies `rgb_array` in place and does not return anything.

    **Implementation Details**

    - The function applies a **5x5 Gaussian blur** using the kernel:

     .. math::

        \frac{1}{256}
        \begin{bmatrix}
        1  &  4  &  6  &  4  &  1  \\
        4  & 16  & 24  & 16  &  4  \\
        6  & 24  & 36  & 24  &  6  \\
        4  & 16  & 24  & 16  &  4  \\
        1  &  4  &  6  &  4  &  1
        \end{bmatrix}

    - Uses **two-pass convolution** (horizontal and vertical) to apply the blur efficiently.
    - Handles **edge pixels** by assigning the **nearest edge values** to prevent artifacts.
    - Optimized for **image processing**, **real-time rendering**, and **graphics effects**.

    **Memory Efficiency**

    - Operates **directly on the image buffer**, avoiding extra memory allocations.
    - Can be used on large images without significant memory overhead.
    - Optimized using **Cython** for high performance.

    **Example Usage**

    .. code-block:: python

        import numpy as np
        from your_module import blur3d

        # Create a sample 3D RGB image array (100x100 with 3 channels)
        rgb_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

        # Apply a single blur pass
        blur3d(rgb_image)

        # Apply multiple blur passes for stronger blur effect
        blur3d(rgb_image, npass=3)

    **Use Cases**

    - **Computer Vision** → Preprocessing for image segmentation and object detection.
    - **Graphics & Gaming** → Motion blur, bloom effects, and real-time rendering.
    - **Image Processing Pipelines** → Reducing noise and enhancing visual quality.
    - **Medical Imaging** → Smoothing image data before feature extraction.

    **See Also**

    - `blur` → Blurs a Pygame surface directly.
    - `gauss_filter24` → Applies Gaussian filtering to a 24-bit image.
    - `canny3d` → Performs edge detection on an RGB image.

    |

    **Apply an in-place Gaussian blur to a 1D array or memoryview slice representing pixel data.**

    This function applies a **Gaussian blur** on a 1D array (or memoryview slice) with shape `(w,)`,
    where `w` is the number of pixels. The array can represent either **BGR** (24-bit) or **BGRA** (32-bit)
    pixel formats or any other format compatible with the `uint8` data type. The blur operation is
    performed **in-place** on the provided array.

    The function performs **npass** passes of convolution using a **5x5 Gaussian kernel**. This can be
    useful for image processing tasks where you need to apply a blur effect, especially on pixel data
    represented as a 1D array (such as when the image data is flattened).

    **Parameters:**

    - **bgr_array** (*1D numpy.ndarray* or *memoryview slice*):
      A 1D array or memoryview slice with shape `(w,)` containing pixel data (e.g., in **BGR(A)** or any other
      pixel format). The array should be of type **uint8**.

    - **width** (*int*):
      The width (`w`) of the original image. Used for proper handling during the blur operation.

    - **height** (*int*):
      The height (`h`) of the original image. Required for correct image processing.

    - **npass** (*int*, optional):
      The number of blur passes to apply. **Default is 1 pass**. More passes result in a stronger blur.

    - **format_32** (*bool*):
      A flag indicating whether the pixel format is **BGR (24-bit)** or **BGRA (32-bit)**.
      `False` for **BGR (24-bit)** (3 channels) and `True` for **BGRA (32-bit)** (4 channels).

    **Returns:**

    - **None**:
      The function performs the blur in-place on `bgr_array` and does not return a value.

    **Notes:**

    - The function uses **Gaussian convolution** with a 5x5 kernel used in two passes.

    .. math::

        \frac{1}{256} \begin{bmatrix}
        1 & 4 & 6 & 4 & 1 \\
        4 & 16 & 24 & 16 & 4 \\
        6 & 24 & 36 & 24 & 6 \\
        4 & 16 & 24 & 16 & 4 \\
        1 & 4 & 6 & 4 & 1
        \end{bmatrix}

    - The image is processed in two passes (horizontal and vertical) for efficiency.
    - **Edge handling**: Pixels outside the image boundaries are replaced by adjacent edge values.
    - The function operates **in-place**, meaning the original `bgr_array` is modified directly.
    - This function is compatible with **BGR** and **BGRA** pixel formats (or other formats with the same shape and data type).

    |


.. py:function:: blur1d_cp(bgr_array, width, height, npass=1, format_32=False)

    |

    Apply a blur effect to a 1D buffer (memoryview slice) and return a copy.

    This function takes a **1D buffer or memoryview slice** containing pixel data in **BGR(A)** format
    (or any other compatible pixel format) and applies a **Gaussian blur**. The blurred result is
    returned as a new array, leaving the original buffer unmodified.

    The function supports both **24-bit (BGR)** and **32-bit (BGRA)** pixel formats. The `format_32`
    flag must be set accordingly:

    - **`format_32=False`** → for **BGR (24-bit)** format.
    - **`format_32=True`** → for **BGRA (32-bit)** format (includes an alpha channel).

    **Example Usage:**

    **For a 24-bit image (BGR format):**

    .. code-block:: python

        arr = blur1d_cp(
            image.get_buffer(),
            width=800, height=600,
            npass=1,
            format_32=False
        )
        image = pygame.image.frombuffer(arr, (800, 600), 'BGR')

    **For a 32-bit image (BGRA format):**

    .. code-block:: python

        arr = blur1d_cp(
            source.get_buffer(),
            width=800, height=600,
            npass=5,
            format_32=True
        )
        image = pygame.image.frombuffer(arr, (800, 600), 'BGRA')

    **Parameters:**

    - **bgr_array** (*C-buffer, numpy.ndarray, or memoryview slice*):
      A **1D array** with shape `(w,)` containing **BGR(A)** pixel data (or any compatible format).

    - **width** (*int*):
      The width of the original image in pixels.

    - **height** (*int*):
      The height of the original image in pixels.

    - **npass** (*int*, optional):
      The number of blur passes to apply (**default is 1**). Increasing this value intensifies the blur effect.

    - **format_32** (*bool*):
      - `True` → **BGRA (32-bit)** format (includes an alpha channel).
      - `False` → **BGR (24-bit)** format.

    **Returns:**

    - **numpy.ndarray** or **memoryview slice** (`shape (w,)`, `dtype=uint8`):
      A **blurred copy** of the input array, maintaining the original pixel format.

    **Notes:**

    - This function does **not modify the input buffer**; it returns a **new** blurred array.

    - The blur operation uses **Gaussian convolution** in multiple passes.

    - The **pixel format of the output array is identical** to the input buffer.

    - Providing an appropriate `format_32` flag ensures correct handling of alpha transparency.


    |

.. py:function:: gauss_filter24(surface_)

       |

       Apply a 5x5 Gaussian filter to smooth an image or surface.
       This function performs Gaussian blurring using a 5x5 convolution kernel, which helps reduce noise
       and detail in an image. It is particularly useful for preprocessing images before edge detection
       or other computer vision tasks.


       **5x5 Gaussian Kernel:**

       .. math::

          \begin{bmatrix}
          2 & 4 & 5 & 4 & 2 \\
          4 & 9 & 12 & 9 & 4 \\
          5 & 12 & 15 & 12 & 5 \\
          4 & 9 & 12 & 9 & 4 \\
          2 & 4 & 5 & 4 & 2
          \end{bmatrix}

       - The kernel values are normalized internally to ensure proper smoothing.
       - The filtering process preserves edges better than a simple average blur.

       **Parameters:**

       - **surface_ (pygame.Surface, 24-bit or 32-bit image):**

       - The input image or surface to be blurred.
       - If a 32-bit surface (with an alpha channel) is provided, the alpha layer is ignored in the output,
         resulting in a 24-bit image.

       **Returns:**

       - **pygame.Surface (24-bit filtered image):**
         - A new blurred surface with reduced noise and smoothed pixel transitions.
         - The output image is always 24-bit, even if the input was 32-bit.

       **Notes:**

       - Gaussian blur is widely used in computer vision and graphics for reducing high-frequency noise.
       - This method is optimized for performance and maintains image integrity.

       |



.. py:function:: gauss_filter32(surface_)

    |

    Apply a 5x5 Gaussian blur filter to an image or pygame surface.
    This function smooths an image by convolving it with a 5x5 Gaussian kernel,
    reducing noise and fine details while preserving essential structures.
    5x5 Gaussian Kernel:

    .. math::

        \begin{bmatrix}
        2 & 4 & 5 & 4 & 2 \\
        4 & 9 & 12 & 9 & 4 \\
        5 & 12 & 15 & 12 & 5 \\
        4 & 9 & 12 & 9 & 4 \\
        2 & 4 & 5 & 4 & 2
        \end{bmatrix}

    - The kernel values are normalized internally to ensure proper smoothing.
    - This method helps remove high-frequency noise while retaining important image structures.

    **Parameters:**

    - **surface_ (pygame.Surface, 32-bit or 24-bit image):**
    - The input image or surface to be blurred.
    - Both 24-bit (RGB) and 32-bit (RGBA) surfaces are supported.

    **Returns:**

    - pygame.Surface (32-bit filtered image):
    - A new blurred surface where pixel transitions are smoothed.
    - If the original surface includes per-pixel transparency (alpha channel), it is preserved in the output.

    **Notes:**

    - Gaussian blur is commonly used in image preprocessing, noise reduction, and artistic effects.
    - The filtering process is optimized for performance while maintaining high image fidelity.

    |

.. py:function:: canny3d(rgb_array,threshold = 50,low = 0,high = 20)

   |

   Apply Canny edge detection to a 3D array or memoryview slice.
   The Canny edge detection algorithm is a multi-stage process that identifies edges
   in an image by detecting areas with significant intensity changes. It was developed
   by John F. Canny (1986) and remains widely used in computer vision.

   **Steps of the Canny Algorithm:**

   1. Gradient Computation – Detects intensity changes.
   2. Non-Maximum Suppression – Removes non-edge pixels.
   3. Double Thresholding – Filters weak edges.
   4. Edge Tracking by Hysteresis – Retains strong edges.

   **Parameters:**

   - **rgb_array** (numpy.ndarray or memoryviewslice, shape (w, h, n), dtype uint8):
     Input image as a 3D array with RGB(A), BGRA, or other pixel formats.
   - **threshold** (unsigned char, default = 50):
     Lower-bound threshold to suppress weak edges in gradient magnitude.
   - **low** (unsigned char, default = 0):
     Lower hysteresis threshold for edge tracking.
     Weak edges below this value are discarded.
   - **high** (unsigned char, default = 20):
     Upper hysteresis threshold for edge tracking.
     Strong edges above this value are retained.

   **Returns:**

   - pygame.Surface – A new surface/image with detected edges.

   **Notes:**

   - Higher `threshold` values produce fewer edges (less noise).
   - Hysteresis (`low`, `high`) ensures only meaningful edges are kept.
   - This function supports multiple pixel formats and operates on 3D arrays directly.

   |


.. py:function:: canny1d(bgr_array, width, height, format_32=False, threshold = 70)

    |

    Apply Canny edge detection to a 1D buffer, NumPy array, or memoryview slice.
    This function processes a flat (1D) image buffer representing a grayscale or color image
    and applies the Canny edge detection algorithm to highlight edges.

    **Parameters:**

    - **bgr_array** (numpy.ndarray or memoryviewslice, shape (w,), dtype uint8):
      1D buffer containing pixel data in BGR(A), BGR, or other formats.

    - **width** (int):
      Width of the original image in pixels.

    - **height** (int):
      Height of the original image in pixels.

    - **format_32** (bool):
      True for 32-bit buffers (BGRA, BGRA).
      False for 24-bit buffers (BGR, BGR).

    - **threshold** (int, default = 70):
      Gradient magnitude threshold for edge detection.
      Higher values suppress weaker edges and reduce noise.

    **Returns:**

    - pygame.Surface – A new image with Canny edge detection applied.

    **Notes:**

    - This function supports various pixel formats (e.g., RGB, BGR, RGBA, BGRA).
    - Thresholding affects edge strength—adjust for optimal results.
    - Operates directly on a 1D buffer, making it memory-efficient.

    |


.. py:function:: blur3d_cp(rgb_array)

   |

   Apply a blur effect to a 3D numpy array or memoryviewslice (returns a copy).
   This function performs a blur operation on a 3D array, such as an image, and returns a blurred copy.
   It supports RGB(A) or other pixel formats, maintaining the original format in the returned array.

   **Parameters:**

   - **rgb_array** (numpy.ndarray or memoryviewslice, shape (w, h, n), dtype uint8):
     3D array containing pixel data in RGB(A) or any compatible pixel format (e.g., BGRA).
     The shape should be (width, height, channels) where channels typically represent color
     components (3 for RGB, 4 for RGBA, etc.).

   **Returns:**

   - numpy.ndarray: A new array (blurred copy) with the same shape and pixel format as `rgb_array`.
     The returned array will have the same width, height, and pixel format as the input,
     but with the blur effect applied.

   **Notes:**

   - The function does not modify the input array in-place; it returns a new blurred array.
   - The output will preserve the same color channels (e.g., RGB or RGBA).
