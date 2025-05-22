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
Summary of the Library
This Python library provides fast and efficient image processing functions, 
focusing on blurring, Gaussian filtering, and edge detection. It is implemented 
in Cython for high performance, making it ideal for applications in computer vision,
 graphics, and image preprocessing.

Purpose
The library is designed to perform fast blurring and edge detection, which are essential for:

Noise reduction (smoothing images).
Edge detection (for object recognition and feature extraction).
Graphics effects (motion blur, bloom effects).
Preprocessing for computer vision tasks (image segmentation, filtering).
Main Features
1. Blurring Functions
These functions apply various types of blurring to smooth images:

blur(surface_, npass): Blurs an image surface, where npass controls intensity.
blur3d(rgb_array, npass): Blurs a 3D (RGB) image array.
blur1d(bgr_array, width, height, npass, format_32): Optimized 1D blurring for efficiency.
blur1d_cp(bgr_array, width, height, npass, format_32): Returns a new blurred array instead of modifying in-place.
blur4bloom_c(surface_, npass, tmp_array): Likely used for bloom effects, enhancing bright areas.
Several internal Cython-optimized versions (blur3d_c, blur1d_c, etc.) run without the Global
 Interpreter Lock (GIL) for multi-threading support.

2. Gaussian Filtering
gauss_filter24(surface_): Applies Gaussian blur to a 24-bit image.
gauss_filter32(surface_): Applies Gaussian blur to a 32-bit image.
📌 Why Gaussian Blur?
It helps smooth images before edge detection, reducing false edges caused by noise.

3. Edge Detection (Canny Algorithm)
The Canny algorithm is widely used in computer vision to detect object boundaries.

canny3d(rgb_array, threshold, low, high): Detects edges in an RGB image.
canny1d(bgr_array, width, height, format_32, threshold): Detects edges in a linear 1D image representation for efficiency.
There are also Cython-optimized versions (canny3d_c, canny1d_c) that improve performance using multi-threading.

Optimization Features
Cython-based (cdef, cpdef, nogil) → Direct C-level performance.
In-place operations (blur1d, blur3d) → Saves memory.
Multi-threading (nogil) → Faster execution on multi-core processors.
Optimized memory handling ([::1] memory views) → Reduces Python overhead.
Use Cases
Computer Vision → Object recognition, feature extraction.
Graphics & Gaming → Motion blur, bloom effects.
Image Processing Pipelines → Preprocessing before machine learning models.
Medical Imaging → Enhancing and detecting features in scans.
Conclusion
This library is highly optimized for fast blurring, Gaussian filtering, and edge detection,
making it a great choice for computer vision, graphics, and machine learning applications where speed is critical.

"""


import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

# NUMPY IS REQUIRED
import pygame
from pygame.surfarray import pixels3d, array_alpha
from pygame import Surface
from pygame.image import frombuffer

try:
    import numpy
    from numpy import ndarray, zeros, empty, uint8, int32, float64, float32, \
        dstack, full, ones, asarray, ascontiguousarray, full_like, add, putmask,\
        int16, arange, repeat, newaxis, sum, divide
except ImportError:
    print("\n<numpy> library is missing on your system."
          "\nTry: \n   C:\\pip install numpy on a window command prompt.")
    raise SystemExit

try:
    cimport cython
    from cython.parallel cimport prange

except ImportError:
    raise ImportError("\n<cython> library is missing on your system."
          "\nTry: \n   C:\\pip install cython on a window command prompt.")

try:
    from cpython cimport PyObject_CallFunctionObjArgs, PyObject, \
            PyList_SetSlice, PyObject_HasAttr, PyObject_IsInstance, \
            PyObject_CallMethod, PyObject_CallObject
except ImportError:
    raise ImportError("\n<cython> library is missing on your system."
          "\nTry: \n   C:\\pip install cython on a window command prompt.")

from PygameShader.config import OPENMP, THREAD_NUMBER, __VERSION__
from PygameShader.misc cimport get_image_format, is_uint8, is_float64, is_int32
from PygameShader.misc cimport is_type_memoryview
from PygameShader.PygameTools cimport index1d_to_3d, index3d_to_1d, vmap_buffer_c


from libc.math cimport sqrtf as sqrt, atan2f as atan2, M_PI, roundf as round_c
from libc.stdio cimport printf
import ctypes



cdef int THREADS = 1

if OPENMP:
     THREADS = THREAD_NUMBER

DEF SCHEDULE = 'static'

cdef float[::1] GAUSS_KERNEL = numpy.array(
    [<float>(1.0/16.0), <float>(4.0/16.0),
     <float>(6.0/16.0), <float>(4.0/16.0), <float>(1.0/16.0)], dtype=numpy.float32)

# kernel definition
GAUSS_FILTER_CANNY = numpy.array(([ 2.0, 4.0, 5.0, 4.0, 2.0 ],
                        [ 4.0, 9.0, 12.0, 9.0, 4.0 ],
                        [ 5.0, 12.0, 15.0, 12.0, 5.0 ],
                        [ 4.0, 9.0, 12.0, 9.0, 4.0 ],
                        [ 2.0, 4.0, 5.0, 4.0, 2.0 ])).astype(dtype = float32, order = 'C')

cdef float[ :, : ] GAUSS_FILTER = divide(GAUSS_FILTER_CANNY, sum(GAUSS_FILTER_CANNY), dtype = float32)

cdef float [:, :] CANNY_GY = numpy.array(([-1, 0, 1],
                               [-2, 0, 2],
                               [-1, 0, 1]), dtype=float32)
cdef float [:, :] CANNY_GX = numpy.array(([-1, -2, -1],
                               [0, 0, 0],
                               [1, 2, 1]), dtype=float32)



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef inline void blur(object surface_, unsigned int npass=1):
    """
    Apply an in-place Gaussian blur effect to a Pygame surface.

    This function performs a **two-pass convolution** to apply a **5x5 Gaussian blur** 
    to the input surface. The first pass convolves the image horizontally, and the 
    second pass applies the blur vertically. Pixels outside the image boundaries are 
    set to the nearest edge value to minimize artifacts.

    ### Features:
    - Supports **24-bit (RGB) and 32-bit (RGBA)** Pygame surfaces.
    - Uses a **5x5 Gaussian kernel** for effective smoothing.
    - Allows multiple passes (`npass > 1`) for increased blur intensity.
    - Ignores the alpha channel during processing to avoid unwanted transparency effects.

    ### Parameters:
    - **surface_** (*pygame.Surface*):  
      A Pygame surface object (must be either 24-bit or 32-bit).

    - **npass** (*int*, optional):  
      Number of blur passes to apply (**default is `1`**).  
      Must be **a positive integer (`npass > 0`)**.

    ### Returns:
    - **None** – The function modifies `surface_` in place.

    ### Raises:
    - **TypeError**: If `surface_` is not a `pygame.Surface` object.
    - **ValueError**: If the surface format is incompatible or cannot be processed.
    - **ValueError**: If `npass` is not a positive integer.

    ### Implementation Details:
    - The function references the surface pixel buffer directly using `get_view('3')`, 
      extracting **only the RGB channels** while **ignoring the alpha channel** (if present).
    - Calls `blur3d_c()`, an optimized Cython function, to perform fast in-place blurring.
    - Additional blur passes (`npass > 1`) will further smooth the image but increase processing time.

    ### Notes:
    - The function is **memory-efficient** since it operates directly on the surface buffer.
    - If the surface has an **alpha channel**, the function **ignores it** during processing, 
      meaning transparency data may not be preserved.
    """

    if not isinstance(surface_, pygame.Surface):
        raise TypeError("\nArgument surface_ must be a pygame."
                        "Surface type, got %s " % type(surface_))

    if npass <= 0:
        raise ValueError(f"\nnpass must be a positive integer, got {npass}")

    cdef unsigned char [:, :, :] rgb_array
    try:
        rgb_array = surface_.get_view('3')

    except Exception as e:
        raise ValueError("\nCannot reference source pixels into a 3d array.\n %s " % e)

    blur3d_c(rgb_array, npass)




@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef inline void blur3d(unsigned char [:, :, :] rgb_array, unsigned int npass=1):
    """
    Apply an in-place Gaussian blur to a 3D array or memoryview slice.

    This function performs a **Gaussian blur** on an array with shape `(w, h, n)`,
    where `w` and `h` are the width and height, and `n` is the number of color channels (RGB or RGBA).  
    The array should be of the **uint8** data type. The blur is applied directly to the array (in-place), and it
    works by performing **npass** passes of convolution using a **5x5 Gaussian kernel**.

    ### Features:
    - Supports **RGB(A)** format with any number of channels `n` (typically 3 for RGB or 4 for RGBA).
    - Performs **Gaussian blur** using a **5x5 kernel**.
    - **Two-pass convolution** (horizontal + vertical) to apply the blur effect.
    - Handles **edge pixels** by setting them to the adjacent edge values.
    - Allows for **multiple blur passes** by setting `npass` (default is 1 pass).

    ### Parameters:
    - **rgb_array** (*numpy.ndarray* or *memoryview slice*):  
      A 3D array with shape `(w, h, n)` containing **uint8** values representing image data in **RGB(A) format**.
      
    - **npass** (*int*, optional):  
      The number of blur passes to apply. **Default is 1**. More passes will increase the blur strength.

    ### Returns:
    - **None**: The function modifies the `rgb_array` in place and does not return anything.

    ### Notes:
    - The blur effect is applied **in-place** to the `rgb_array`.
    - The Gaussian kernel used for the blur is a **5x5 kernel** as shown below:
      \[
      \frac{1}{256} \begin{bmatrix} 
      1 & 4 & 6 & 4 & 1 \\ 
      4 & 16 & 24 & 16 & 4 \\ 
      6 & 24 & 36 & 24 & 6 \\ 
      4 & 16 & 24 & 16 & 4 \\ 
      1 & 4 & 6 & 4 & 1 
      \end{bmatrix}
      \]
    - The function uses **two-pass convolution** (horizontal and vertical) for 
        efficient blurring.
    - **Edge handling**: Pixels outside the image boundaries are replaced by 
        adjacent edge values.
    - This function is suitable for **image processing**, **graphics effects**,
        and **real-time rendering** applications.
    """

    # Check that the input array is a valid type
    if not (isinstance(rgb_array, numpy.ndarray) or rgb_array.__class__.__name__ == '_memoryviewslice'):
        raise TypeError("Input rgb_array must be a numpy.ndarray or memoryviewslice.")

    if not is_uint8(rgb_array):
        raise TypeError(
            "\nrgb_array Expecting uint8 (unsigned char) "
            "data type got %s" % rgb_array.dtype)

    cdef:
        Py_ssize_t w, h, bit_size = (<object>rgb_array).shape[ :3 ]

    if npass <= 0:
        raise ValueError("\nnpass must be greater than 0, got %s" % npass)

    # Only RGB(A) array supported (w, h, 3|4)
    if bit_size not in (3, 4):
        raise ValueError('\nIncorrect bit_size, support only RGB or RGBA format.')

    blur3d_c(rgb_array, npass)





@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef inline void blur1d(
        unsigned char [::1] bgr_array,
        const unsigned int width,
        const unsigned int height,
        unsigned int npass = 1,
        bint format_32     = False
):
    """
    Apply an in-place blur to a 1D array or memoryview slice representing pixel data.

    This function applies a **Gaussian blur** on a 1D array (or memoryview slice) with shape `(w,)`, 
    where `w` is the number of pixels. The array can represent either **BGR** (24-bit) or **BGRA** (32-bit) 
    pixel formats or any other format compatible with the `uint8` data type. The blur operation is 
    performed **in-place** on the provided array.

    The function performs **npass** passes of convolution using a **5x5 Gaussian kernel**. This can be 
    useful for image processing tasks where you need to apply a blur effect, especially on pixel data 
    represented as a 1D array (such as when the image data is flattened).

    ### Parameters:
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

    ### Returns:
    - **None**:  
      The function performs the blur in-place on `bgr_array` and does not return a value.

    ### Notes:
    - The function uses **Gaussian convolution** with a 5x5 kernel 
      used in two passes.
      \[
      \frac{1}{256} \begin{bmatrix} 
      1 & 4 & 6 & 4 & 1 \\ 
      4 & 16 & 24 & 16 & 4 \\ 
      6 & 24 & 36 & 24 & 6 \\ 
      4 & 16 & 24 & 16 & 4 \\ 
      1 & 4 & 6 & 4 & 1 
      \end{bmatrix}
      \]
    - The image is processed in two passes (horizontal and vertical) for efficiency.
    - **Edge handling**: Pixels outside the image boundaries are replaced by adjacent edge values.
    - The function operates **in-place**, meaning the original `bgr_array` is modified directly.
    - This function is compatible with **BGR** and **BGRA** pixel formats
        (or other formats with the same shape and data type).
    """

    # Check that the input array is a valid type
    if not (isinstance(bgr_array, numpy.ndarray) or bgr_array.__class__.__name__ == '_memoryviewslice'):
        raise TypeError("Input bgr_array must be a numpy.ndarray or memoryviewslice.")

    if not is_uint8(bgr_array):
        raise TypeError(
            "\nbgr_array Expecting uint8 (unsigned char) data type got %s" % bgr_array.dtype)

    if npass <= 0:
        raise ValueError("\nnpass must be greater than 0, got %s" % npass)

    cdef:
        Py_ssize_t length = len(bgr_array)

    blur1d_c(bgr_array, width, height, npass, format_32)



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef unsigned char [::1] blur1d_cp(
        const unsigned char [::1] bgr_array,
        const unsigned int width,
        const unsigned int height,
        unsigned int npass = 1,
        bint format_32     = False
):
    """
    Apply a blur effect to a 1D buffer (memoryview slice) and return a copy.

    This function takes a **1D buffer or memoryview slice** containing pixel data in **BGR(A)** format 
    (or any other compatible pixel format) and applies a **Gaussian blur**. The blurred result is 
    returned as a new array, leaving the original buffer unmodified.

    The function supports both **24-bit (BGR)** and **32-bit (BGRA)** pixel formats. The `format_32` 
    flag must be set accordingly:  
    - **`format_32=False`** → for **BGR (24-bit)** format.  
    - **`format_32=True`** → for **BGRA (32-bit)** format (includes an alpha channel).  

    ### Example Usage:
    **For a 24-bit image (BGR format):**
    ```python
    arr = blur1d_cp(
        image.get_buffer(),
        width=800, height=600,
        npass=1,
        format_32=False
    )
    image = pygame.image.frombuffer(arr, (800, 600), 'BGR')
    ```

    **For a 32-bit image (BGRA format):**
    ```python
    arr = blur1d_cp(
        source.get_buffer(),
        width=800, height=600,
        npass=5,
        format_32=True
    )
    image = pygame.image.frombuffer(arr, (800, 600), 'BGRA')
    ```

    ### Parameters:
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

    ### Returns:
    - **numpy.ndarray** or **memoryview slice** (`shape (w,)`, `dtype=uint8`):  
      A **blurred copy** of the input array, maintaining the original pixel format.

    ### Notes:
    - This function does **not modify the input buffer**; it returns a **new** blurred array.
    - The blur operation uses **Gaussian convolution** in multiple passes.
    - The **pixel format of the output array is identical** to the input buffer.
    - Providing an appropriate `format_32` flag ensures correct handling of **alpha transparency**.
    
    """

    # Check that the input array is a valid type
    if not (isinstance(bgr_array, numpy.ndarray) or bgr_array.__class__.__name__ == '_memoryviewslice'):
        raise TypeError("Input bgr_array must be a numpy.ndarray or memoryviewslice.")

    if not is_uint8(bgr_array):
        raise TypeError(
            "\nbgr_array Expecting uint8 (unsigned char) "
            "data type got %s" % bgr_array.dtype)


    assert npass > 0, \
        "\nArgument npass must be > 0, got %s " % npass

    cdef:
        Py_ssize_t length = 0

    try:
        length = len(bgr_array)

    except Exception as e:
        raise ValueError(
            "\nArray length is 'undefined'.\n%s " % e)

        # Only uint8 data is compatible
    if not is_uint8(bgr_array):
        raise TypeError("\nExpecting uint8 (unsigned char) "
                        "data type got %s" % bgr_array.dtype)

    return blur1d_cp_c(bgr_array, width, height, npass, format_32)




@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef inline void non_max_suppression(
         unsigned char [:, :, :] rgb_array,
         float [:, :] angles):

    """
    Perform Non-Maximum Suppression (NMS) as part of the Canny edge detection algorithm.

    Non-Maximum Suppression (NMS) refines detected edges by preserving only the most significant 
    gradient magnitudes along the gradient direction, while suppressing weaker neighboring pixels.
    This step helps in **thinning the edges**, ensuring that detected edges are **sharp and well-defined**.

    ### How It Works:
    - Each pixel's **gradient magnitude and orientation** is evaluated.
    - The pixel is compared with its **two neighboring pixels along the gradient direction**.
    - If the current pixel is not the local maximum, it is suppressed (set to zero).

    ### Parameters:
    - **rgb_array** (*numpy.ndarray* or *memoryview slice*, `shape=(w, h, 3)`, `dtype=uint8`):  
      - The **grayscale or RGB image** where non-maximum suppression is applied.  
      - The input image should already have **gradient magnitude values** computed.  
    - **angles** (*numpy.ndarray* or *memoryview slice*, `shape=(w, h)`, `dtype=uint8`):  
      - Stores **gradient directions** (angles) for each pixel.  
      - Typically quantized into **four discrete directions** (0°, 45°, 90°, 135°).  

    ### Returns:
    - **void**:  
      - The function operates **in-place**, modifying `rgb_array` directly.  

    ### Notes:
    - This function is a critical step in **Canny edge detection** for refining edges.
    - It assumes that **gradient magnitudes** are already computed in `rgb_array`.
    - **Edges are preserved only if they are local maxima** along their gradient direction.
    - The `angles` array should be in **quantized form** (e.g., 0, 45, 90, 135 degrees).
    """


    cdef Py_ssize_t w, h, bit_size
    w, h, bit_size = rgb_array.shape[:3]


    cdef:
        int i, j
        int w_1 = w - 1
        int h_1 = h  -1
        float * angle
        unsigned char value_to_compare


    with nogil:

        for i in prange(0, w_1, schedule=SCHEDULE, num_threads=THREADS):
            for j in range(1, h_1):

                angle = &angles[i, j]

                if (<float>0.0 <= angle[0] < <float>22.5) or (<float>157.5 <= angle[0] <= <float>180.0):
                    value_to_compare = max(rgb_array[ i, j - 1, 0], rgb_array[ i, j + 1, 0 ])

                elif <float>22.5 <= angle[0] < <float>67.5:
                    value_to_compare = max(rgb_array[ i - 1, j - 1, 0 ], rgb_array[ i + 1, j + 1, 0 ])

                elif <float>67.5 <= angle[0] < <float>112.5:
                    value_to_compare = max(rgb_array[ i - 1, j, 0 ], rgb_array[ i + 1, j, 0 ])

                else:
                    value_to_compare = max(rgb_array[ i + 1, j - 1, 0 ], rgb_array[ i - 1, j + 1, 0 ])

                if rgb_array[ i, j, 0 ] < value_to_compare:
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
cdef double_threshold_hysteresis_inplace(
        unsigned char [:, :, :] rgb_array,
        unsigned int low,
        unsigned int high
):
    """
    Perform edge tracking by hysteresis in the Canny edge detection algorithm.

    Edge tracking by hysteresis is the final step in the Canny edge detection process. 
    It ensures that only **strong edges** (above a high threshold) and **weak edges** 
    (connected to strong edges) are preserved, while isolated weak edges are suppressed.

    ### Parameters:
    - **rgb_array** (*numpy.ndarray* or *memoryview slice*, `shape=(w, h, 3)`, `dtype=uint8`):  
      - The input image where edge tracking by hysteresis is applied.  
      - Pixels should already be classified into **strong and weak edges**.  

    - **low** (*unsigned char*, `default=0`):  
      - The **lower threshold** for edge tracking.  
      - Edges with gradient magnitudes below this value are **discarded**.  

    - **high** (*unsigned char*, `default=20`):  
      - The **upper threshold** for edge tracking.  
      - Edges with gradient magnitudes above this value are **considered strong** and retained.  
      - Weak edges are kept **only if they are connected to strong edges**.  

    ### Returns:
    - **void**:  
      - The function modifies `rgb_array` **in-place**, finalizing edge detection.  

    ### Notes:
    - **Hysteresis ensures continuity in detected edges**, preventing broken or incomplete edges.  
    - **Weak edges (low < magnitude < high) are removed** unless they are linked to strong edges.  
    - This function assumes that `rgb_array` has already undergone **gradient calculation and non-maximum suppression**.  
    """

    cdef:
        unsigned char weak = 50
        unsigned char strong = 255
        Py_ssize_t w, h, bit_size, w_1, h_1
        int i, j

    w, h, bit_size = rgb_array.shape[:3]
    w_1 = w - 1
    h_1 = h - 1

    cdef:

        unsigned char * r
        unsigned char * g
        unsigned char * b

    with nogil:
        # Double threshold
        for i in prange(0, w_1):#, schedule=SCHEDULE, num_threads=THREADS):
            for j in range(0, h_1):

                r = &rgb_array[ i, j, 0 ]
                g = &rgb_array[ i, j, 1 ]
                b = &rgb_array[ i, j, 2 ]

                if low < r[ 0 ] <= high:
                    r[0] = weak
                    g[0] = weak
                    b[0] = weak

                elif r[ 0 ] > high:
                    r[0] = strong
                    g[0] = strong
                    b[0] = strong

        # Edge tracking by hysteresis
        for i in prange(1, w_1, schedule=SCHEDULE, num_threads=THREADS):

            for j in range(1, h_1):

                r = &rgb_array[ i, j, 0 ]
                g = &rgb_array[ i, j, 1 ]
                b = &rgb_array[ i, j, 2 ]

                if r[ 0 ] == weak:

                    if (rgb_array[ i + 1, j - 1, 0 ] == strong
                    or (rgb_array[ i + 1, j, 0 ] == strong)
                    or (rgb_array[ i + 1, j + 1, 0 ] == strong)
                    or (rgb_array[ i, j - 1, 0 ] == strong)
                    or (rgb_array[ i, j + 1, 0 ] == strong)
                    or (rgb_array[ i - 1, j - 1, 0 ] == strong)
                    or (rgb_array[ i - 1, j, 0 ] == strong)
                    or (rgb_array[ i - 1, j + 1, 0 ] == strong)):
                        r[ 0 ] = strong
                        g[ 0 ] = strong
                        b[ 0 ] = strong

                    else:
                        r[ 0 ] = 0
                        g[ 0 ] = 0
                        b[ 0 ] = 0




@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef gauss_filter24(surface_):
    """
    Apply a 5x5 Gaussian filter to smooth an image or surface.

    This function performs Gaussian blurring using a **5x5 convolution kernel**, which helps reduce noise 
    and detail in an image. It is particularly useful for **preprocessing images** before edge detection 
    or other computer vision tasks.

    ### 5x5 Gaussian Kernel:
    ```
    [ 2,  4,  5,  4,  2 ],
    [ 4,  9, 12,  9,  4 ],
    [ 5, 12, 15, 12,  5 ],
    [ 4,  9, 12,  9,  4 ],
    [ 2,  4,  5,  4,  2 ]
    ```
    - The kernel values are **normalized internally** to ensure proper smoothing.  
    - The filtering process **preserves edges better** than a simple average blur.

    ### Parameters:
    - **surface_** (*pygame.Surface*, 24-bit or 32-bit image):  
      - The input image or surface to be blurred.  
      - If a **32-bit surface** (with an alpha channel) is provided, the **alpha layer is ignored** in the output, 
        resulting in a 24-bit image.

    ### Returns:
    - **pygame.Surface** (24-bit filtered image):  
      - A new **blurred surface** with reduced noise and smoothed pixel transitions.  
      - The output image is always **24-bit**, even if the input was 32-bit.  

    ### Notes:
    - Gaussian blur is widely used in **computer vision and graphics** for reducing high-frequency noise.  
    - This method is optimized for performance and maintains image integrity.  
    """


    assert isinstance(surface_, Surface),\
        '\nArgument image must be a valid Surface, got %s ' % type(surface_)

    cdef:
        Py_ssize_t w, h

    cdef unsigned char [:, :, :] rgb_array

    try:
        rgb_array = surface_.get_view('3')

    except (surface_.error, ValueError):
        raise ValueError(
            '\nTexture/image is not compatible.')

    w, h = rgb_array.shape[:2]

    assert w != 0 or h !=0, \
        '\nimage with incorrect dimensions (w>0, h>0) got (%s, %s) ' % (w, h)

    if not is_uint8(rgb_array):
        raise TypeError(
            "\nbgr_array Expecting uint8 (unsigned char) data type got %s" % rgb_array.dtype)


    cdef:
        short kernel_half = <short>(len(GAUSS_FILTER) >> 1)
        unsigned char [:, :, ::1] output_array = empty((h, w, 3), order='C', dtype=uint8)
        int x, y, xx, yy
        unsigned short red, green, blue,
        short kernel_offset_y, kernel_offset_x
        float r, g, b, k
        Py_ssize_t h_1 = h - 1
        Py_ssize_t w_1 = w - 1



    with nogil:

        for y in prange(0, h_1, schedule=SCHEDULE, num_threads=THREADS):

            for x in range(0, w_1):

                r, g, b = 0, 0, 0

                for kernel_offset_y in range(-kernel_half, kernel_half + 1):

                    for kernel_offset_x in range(-kernel_half, kernel_half + 1):

                        xx = x + kernel_offset_x
                        yy = y + kernel_offset_y

                        if xx < 0:
                            xx = 0
                        elif xx > w :
                            xx = w

                        if yy < 0:
                            yy = 0
                        elif yy > h :
                            yy = h

                        red   = rgb_array[xx, yy, 0]
                        green = rgb_array[xx, yy, 1]
                        blue  = rgb_array[xx, yy, 2]

                        if red + green + blue == 0:
                            continue

                        k = GAUSS_FILTER[kernel_offset_y + kernel_half,
                                         kernel_offset_x + kernel_half]

                        r += red * k
                        g += green * k
                        b += blue * k

                if r > 255:
                    r = <float>255.0
                if g > 255:
                    g = <float>255.0
                if b > 255:
                    b = <float>255.0

                output_array[y, x, 0] = <unsigned char>r
                output_array[y, x, 1] = <unsigned char>g
                output_array[y, x, 2] = <unsigned char>b


    return frombuffer(output_array, (w, h), "RGB")



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef gauss_filter32(surface_):
    """
    Apply a 5x5 Gaussian blur filter to an image or pygame surface.

    This function smooths an image by convolving it with a **5x5 Gaussian kernel**, 
    reducing noise and fine details while preserving essential structures.

    ### 5x5 Gaussian Kernel:
    ```
    [ 2,  4,  5,  4,  2 ],
    [ 4,  9, 12,  9,  4 ],
    [ 5, 12, 15, 12,  5 ],
    [ 4,  9, 12,  9,  4 ],
    [ 2,  4,  5,  4,  2 ]
    ```
    - The kernel values are **normalized internally** to ensure proper smoothing.
    - This method helps remove high-frequency noise while retaining important image structures.

    ### Parameters:
    - **surface_** (*pygame.Surface*, 32-bit or 24-bit image):  
      - The input image or surface to be blurred.  
      - Both **24-bit (RGB)** and **32-bit (RGBA)** surfaces are supported.

    ### Returns:
    - **pygame.Surface** (32-bit filtered image):  
      - A new **blurred surface** where pixel transitions are smoothed.  
      - If the **original surface includes per-pixel transparency (alpha channel)**, it is preserved in the output.  

    ### Notes:
    - Gaussian blur is commonly used in **image preprocessing, noise reduction, and artistic effects**.
    - The filtering process is optimized for performance while maintaining high image fidelity.
    """


    assert isinstance(surface_, Surface), \
        'Argument image must be a valid Surface, got %s ' % type(surface_)

    cdef:
        Py_ssize_t w, h
    w, h = surface_.get_size()

    cdef:
        unsigned char [:, :, :] rgb_array
        unsigned char [:, :] alpha

    try:
        rgb_array = surface_.get_view('3')
        alpha = array_alpha(surface_)

    except (surface_.error, ValueError):
        raise ValueError('\nInvalid texture or image. '
                         'This version is compatible with 32-bit image format '
                         'with per-pixel transparency.')

    assert w != 0 or h !=0,\
        '\nImage with incorrect dimensions (w>0, h>0) got (%s, %s) ' % (w, h)

    if not is_uint8(rgb_array):
        raise TypeError(
            "\nbgr_array Expecting uint8 (unsigned char) data type got %s" % rgb_array.dtype)

    cdef:
        short kernel_half = <short>(len(GAUSS_FILTER) >> 1)
        unsigned char [:, :, :] output_array = empty((h, w, 4), dtype=uint8)
        int x, y, xx, yy
        unsigned char red, green, blue,
        short kernel_offset_y, kernel_offset_x
        float r, g, b, k
        int w_1 = w - 1
        int h_1 = h - 1
        unsigned char * a


    with nogil:

        for y in prange(0, h_1, schedule=SCHEDULE, num_threads=THREADS):

            for x in range(0, w_1):

                r, g, b = 0, 0, 0

                a = &alpha[x, y]
                # Skip transparent pixel
                if a[0] == 0:
                    output_array[ y, x, 0 ] = 0
                    output_array[ y, x, 1 ] = 0
                    output_array[ y, x, 2 ] = 0
                    output_array[ y, x, 3 ] = 0
                    continue

                for kernel_offset_y in range(-kernel_half, kernel_half + 1):

                    for kernel_offset_x in range(-kernel_half, kernel_half + 1):

                        xx = x + kernel_offset_x
                        yy = y + kernel_offset_y

                        if xx < 0:
                            xx = 0
                        elif xx > w_1 :
                            xx = w_1

                        if yy < 0:
                            yy = 0
                        elif yy > h_1 :
                            yy = h_1

                        red   = rgb_array[xx, yy, 0]
                        green = rgb_array[xx, yy, 1]
                        blue  = rgb_array[xx, yy, 2]

                        if red + green + blue == 0:
                            continue

                        k = GAUSS_FILTER[kernel_offset_y + kernel_half,
                                         kernel_offset_x + kernel_half]

                        r += red * k
                        g += green * k
                        b += blue * k

                if r > 255:
                    r = <float>255.0
                if g > 255:
                    g = <float>255.0
                if b > 255:
                    b = <float>255.0

                output_array[y, x, 0] = <unsigned char>r
                output_array[y, x, 1] = <unsigned char>g
                output_array[y, x, 2] = <unsigned char>b
                output_array[y, x, 3] = <unsigned char>a[0]

    return frombuffer(output_array, (w, h), "RGBA")



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef canny3d(unsigned char [:, :, :] rgb_array,
              unsigned char threshold = 50,
              unsigned char low = 0,
              unsigned char high = 20
              ):
    """
    Apply Canny edge detection to a 3D array or memoryview slice.

    The **Canny edge detection algorithm** is a multi-stage process that identifies edges 
    in an image by detecting areas with significant intensity changes. It was developed 
    by **John F. Canny (1986)** and remains widely used in computer vision.

    ### Steps of the Canny Algorithm:
    1. **Gradient Computation** – Detects intensity changes.
    2. **Non-Maximum Suppression** – Removes non-edge pixels.
    3. **Double Thresholding** – Filters weak edges.
    4. **Edge Tracking by Hysteresis** – Retains strong edges.

    ### Parameters:
    - **rgb_array** (*numpy.ndarray* or *memoryviewslice*, shape *(w, h, n)*, dtype *uint8*):  
      - Input image as a 3D array with **RGB(A), BGRA, or other pixel formats**.
      
    - **threshold** (*unsigned char*, default = `50`):  
      - Lower-bound threshold to **suppress weak edges** in gradient magnitude.
      
    - **low** (*unsigned char*, default = `0`):  
      - Lower hysteresis threshold for **edge tracking**.
      - Weak edges **below this value** are discarded.
      
    - **high** (*unsigned char*, default = `20`):  
      - Upper hysteresis threshold for **edge tracking**.
      - Strong edges **above this value** are retained.

    ### Returns:
    - **pygame.Surface** – A new surface/image with **detected edges**.

    ### Notes:
    - **Higher `threshold` values** produce fewer edges (less noise).
    - **Hysteresis (`low`, `high`) ensures** only meaningful edges are kept.
    - This function **supports multiple pixel formats** and operates on 3D arrays directly.
    """

    # Check that the input array is a valid type
    if not (isinstance(rgb_array, numpy.ndarray) or rgb_array.__class__.__name__ == '_memoryviewslice'):
        raise TypeError("Input rgb_array must be a numpy.ndarray or memoryviewslice.")

    if not is_uint8(rgb_array):
        raise TypeError(
            "\nbgr_array Expecting uint8 (unsigned char) data type got %s" % rgb_array.dtype)

    cdef:
        Py_ssize_t w, h
        int bit_size = 0

    try:

        w, h, bit_size = rgb_array.shape[ :3 ]

        # Only RGB(A) array supported (w, h, 3|4)
        if bit_size not in (3, 4):
            raise ValueError(
                '\nIncorrect bit_size, support only RGB(A)')

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

    return canny3d_c(rgb_array, threshold, low, high)



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef canny3d_c(unsigned char [:, :, :] rgb_array,
               unsigned char threshold = 50,
               unsigned char low = 0,
               unsigned char high = 20
               ):
    """
    Apply Canny edge detection to a 3D array or memoryview slice.

    The **Canny edge detection algorithm** is a multi-stage process designed to 
    identify edges in an image by detecting significant intensity changes. 
    Developed by **John F. Canny (1986)**, it is widely used in image processing and 
    computer vision applications.

    ### Algorithm Steps:
    1. **Gradient Calculation** – Computes image intensity gradients.
    2. **Non-Maximum Suppression** – Eliminates weak gradient responses.
    3. **Double Thresholding** – Classifies pixels as strong, weak, or non-edges.
    4. **Edge Tracking by Hysteresis** – Preserves valid edges while discarding weak ones.

    ### Parameters:
    - **rgb_array** (*numpy.ndarray* or *memoryviewslice*, shape *(w, h, n)*, dtype *uint8*):  
      - Input image as a 3D array in **RGB(A), BGRA, or other pixel formats**.
    - **threshold** (*unsigned char*, default = `50`):  
      - Defines a **gradient magnitude threshold** to suppress weak edges and noise.
    - **low** (*unsigned char*, default = `0`):  
      - **Lower threshold** for **hysteresis-based edge tracking**.
      - Pixels with a gradient below this value are **discarded**.
    - **high** (*unsigned char*, default = `20`):  
      - **Upper threshold** for **hysteresis-based edge tracking**.
      - Pixels with a gradient above this value are **considered strong edges**.

    ### Returns:
    - **pygame.Surface** – A new image with **detected edges**.

    ### Notes:
    - **Higher `threshold` values** reduce noise but may **miss weak edges**.
    - **Hysteresis (`low`, `high`) ensures** meaningful edges are retained.
    - This function **operates directly** on a 3D array and supports **various pixel formats**.
    """

    cdef:
        Py_ssize_t w, h
        unsigned short int bit_size = 3
        str rgb_format = 'RGB'

    w, h, bit_size = rgb_array.shape[:3]

    if bit_size == 4:
        rgb_format = 'RGBA'

    # kernel 5x5 separable
    cdef:
        short int kernel_half = 1
        int x, y, xx, yy, kernel_offset_y, kernel_offset_x
        int index_x, index_y
        float gx, gy, angle
        char kernel_offset
        unsigned char c
        float magnitude
        float * kxv
        float * kyv
        unsigned char [:, :, :] output_array = numpy.empty((h, w, bit_size), dtype = uint8)
        float [:, :] angles = numpy.empty((h, w), dtype = float32)

    with nogil:
        for y in prange(0, h-1): # , schedule=SCHEDULE, num_threads=THREADS):

            for x in range(0, w-1):

                gx, gy = 0, 0

                if bit_size == 4 and rgb_array[x, y, 3] == 0:
                    output_array[ y, x, 0 ] = 0
                    output_array[ y, x, 1 ] = 0
                    output_array[ y, x, 2 ] = 0
                    continue

                for kernel_offset_y in range(-kernel_half, kernel_half + 1):

                    for kernel_offset_x in range(-kernel_half, kernel_half + 1):

                        xx = x + kernel_offset_x
                        yy = y + kernel_offset_y

                        if xx < 0:
                            xx = 0
                        elif xx > w :
                            xx = w

                        if yy < 0:
                            yy = 0
                        elif yy > h :
                            yy = h

                        c = rgb_array[xx, yy, 0]

                        if c == 0:
                            continue

                        index_x = kernel_offset_x + kernel_half
                        index_y = kernel_offset_y + kernel_half

                        kvx = &CANNY_GX[index_x, index_y]
                        kvy = &CANNY_GY[index_x, index_y]

                        gx = gx + c * kvx[0]
                        gy = gy + c * kvy[0]

                magnitude = <float>sqrt(gx ** 2 + gy ** 2)

                if magnitude < 0 or magnitude < threshold:
                    magnitude = 0
                if magnitude > 255:
                    magnitude = 255

                output_array[y, x, 0] = <unsigned char>magnitude
                output_array[y, x, 1] = <unsigned char>magnitude
                output_array[y, x, 2] = <unsigned char>magnitude
                if bit_size == 4:
                    output_array[ y, x, 3 ] = rgb_array[x, y, 3]

                angle = <float>atan2(gy, gx) * <float>180.0 / <float>M_PI
                if angle < 0:
                    angle = angle + <float>180.0
                angles[y, x] = round_c(angle)

    non_max_suppression(output_array, angles)
    # output_array = double_threshold_hysteresis(output_array, low, high)
    double_threshold_hysteresis_inplace(output_array, low, high)
    return frombuffer(output_array, (w, h), rgb_format)





@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef canny1d(
        unsigned char [::1] bgr_array,
        unsigned int width,
        unsigned int height,
        bint format_32=False,
        unsigned char threshold = 70
):
    """
    Apply Canny edge detection to a 1D buffer, NumPy array, or memoryview slice.

    This function processes a **flat (1D) image buffer** representing a **grayscale or color image** 
    and applies the **Canny edge detection algorithm** to highlight edges.

    ### Parameters:
    - **bgr_array** (*numpy.ndarray* or *memoryviewslice*, shape *(w,)*, dtype *uint8*):  
      - **1D buffer containing pixel data** in **BGR(A), BGR, or other formats**.
      
    - **width** (*int*):  
      - **Width of the original image** in pixels.
      
    - **height** (*int*):  
      - **Height of the original image** in pixels.
      
    - **format_32** (*bool*):  
      - `True` for **32-bit** buffers (**BGRA, BGRA**).  
      - `False` for **24-bit** buffers (**BGR, BGR**).
      
    - **threshold** (*int*, default = `70`):  
      - **Gradient magnitude threshold** for edge detection.  
      - Higher values **suppress weaker edges** and reduce noise.

    ### Returns:
    - **pygame.Surface** – A new image with **Canny edge detection applied**.

    ### Notes:
    - **This function supports various pixel formats** (e.g., RGB, BGR, RGBA, BGRA).  
    - **Thresholding affects edge strength**—adjust for optimal results.  
    - Operates **directly on a 1D buffer**, making it **memory-efficient**.  
    """

    # Check that the input array is a valid type
    if not (isinstance(bgr_array, numpy.ndarray) or bgr_array.__class__.__name__ == '_memoryviewslice'):
        raise TypeError("Input bgr_array must be a numpy.ndarray or memoryviewslice.")

    if not is_uint8(bgr_array):
        raise TypeError(
            "\nbgr_array Expecting uint8 (unsigned char) data type got %s" % bgr_array.dtype)

    cdef:
        Py_ssize_t w, h
        int bit_size = 0

    w, h, bit_size = bgr_array.shape[ :3 ]

    # Only BGR(A) array supported (w, h, 3|4)
    if bit_size not in (3, 4):
        raise ValueError(
            '\nIncorrect bit_size, support only BGR or BGR(A)')

    return canny1d_c(bgr_array, width, height, format_32, threshold)


# todo this can be improved using 1d arrays
@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef canny1d_c(
        unsigned char [::1] rgb_array,
        unsigned int width,
        unsigned int height,
        bint format_32=False,
        unsigned char threshold = 70
):
    """
    Apply Canny edge detection to a 1D buffer, NumPy array, or memoryviewslice.

    This function processes a **flat (1D) image buffer** representing a **grayscale or color image** 
    and applies the **Canny edge detection algorithm** to highlight edges.

    ### Parameters:
    - **rgb_array** (*numpy.ndarray* or *memoryviewslice*, shape *(w,)*, dtype *uint8*):  
      - **1D buffer containing pixel data** in **RGB(A), BGR(A), or other formats**.
      
    - **width** (*unsigned int*):  
      - **Width of the original image** in pixels.
      
    - **height** (*unsigned int*):  
      - **Height of the original image** in pixels.
      
    - **format_32** (*bool*, default = `False`):  
      - `True` for **32-bit** buffers (**RGBA, BGRA**).  
      - `False` for **24-bit** buffers (**RGB, BGR**).
      
    - **threshold** (*unsigned char*, default = `70`):  
      - **Gradient magnitude threshold** for edge detection.  
      - Higher values **suppress weaker edges** and reduce noise.

    ### Returns:
    - **numpy.ndarray** – A new image buffer with **Canny edge detection applied**.

    ### Notes:
    - **Supports various pixel formats** (e.g., RGB, BGR, RGBA, BGRA).  
    - **Thresholding affects edge strength**—adjust for optimal results.  
    - Operates **directly on a 1D buffer**, making it **memory-efficient**.  
    """

    cdef:
        unsigned short int bit_size = 3
        str rgb_format = 'RGB'

    if format_32:
        bit_size = 4
        rgb_format = 'RGBA'

    cdef:
        Py_ssize_t length = 0

    try:
        length = len(rgb_array)

    except Exception as e:
        raise ValueError(
            "\nArray length is 'undefined'.\n%s " % e)

    cdef unsigned char [::1] convolve

    try:
        convolve = numpy.empty((width * height * bit_size), dtype=uint8)

    except Exception as e:
        raise ValueError(
            "\nCannot reference source pixels into a 3d array.\n %s " % e)

    # kernel 5x5 separable
    cdef:
        short int kernel_half = 1
        int x, y, xx, yy
        float gx, gy, magnitude
        char k_offsx, k_offsy
        float kx, ky
        unsigned char cx, cy
        int i


    with nogil:
       # horizontal convolution
        # goes through all RGB values of the buffer and apply the convolution
        for i in prange(0, length, bit_size, schedule=METHOD, num_threads=THREADS):

            gx, gy = 0, 0


            if format_32 and rgb_array[i + 3] == 0:
                convolve[ i     ] = 0
                convolve[ i + 1 ] = 0
                convolve[ i + 2 ] = 0
                continue

            for k_offsx in range(-kernel_half, kernel_half + 1):
                for k_offsy in range(-kernel_half, kernel_half + 1):

                    kx = CANNY_GX[ k_offsx + kernel_half, k_offsy + kernel_half ]
                    xx = i + k_offsx * bit_size
                    if xx < 0 or xx > length:
                        continue

                    cx = rgb_array[ xx ]
                    gx = gx + cx * kx

                    ky = CANNY_GY[ k_offsx + kernel_half, k_offsy + kernel_half ]
                    yy = i + k_offsy * bit_size * width
                    if yy < 0 or yy > length:
                        continue

                    cy = rgb_array[ yy ]

                    gy = gy + cy * ky

            magnitude = <float> sqrt(gx ** 2 + gy ** 2)

            if magnitude < 0 or magnitude < threshold:
                magnitude = <unsigned char>0
            if magnitude > 255:
                magnitude = <unsigned char>255


            convolve[ i     ] = <unsigned char> magnitude
            convolve[ i + 1 ] = <unsigned char> magnitude
            convolve[ i + 2 ] = <unsigned char> magnitude
            if bit_size == 4:
                convolve[ i + 3 ] = rgb_array[i + 3]

    return frombuffer(convolve, (height, width), rgb_format)




@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef inline void blur3d_c(
        unsigned char [:, :, :] rgb_array,
        unsigned int npass=1,
        ):

    """
    Apply an in-place Gaussian blur to a 3D array (RGB/RGBA or any pixel format).

    This function performs **a two-pass Gaussian blur** on a **memoryview slice (w, h, n)**  
    or numpy.array using a **5x5 convolution kernel**. It is optimized for **uint8 image data**.

    ### Features:
    - Supports **RGB(A) or any (w, h, n) uint8 format**.
    - Uses a **5x5 Gaussian kernel** for smooth blurring.
    - Applies **two-pass convolution** (horizontal first, then vertical).
    - Handles **edge pixels** by setting them to the adjacent edge values.
    - Allows **multiple blur passes** (`npass` > 1) for stronger effects.

    ### Gaussian Kernel (5x5) Used:
    \[
    \frac{1}{256} \begin{bmatrix} 
    1 & 4 & 6 & 4 & 1 \\ 
    4 & 16 & 24 & 16 & 4 \\ 
    6 & 24 & 36 & 24 & 6 \\ 
    4 & 16 & 24 & 16 & 4 \\ 
    1 & 4 & 6 & 4 & 1 
    \end{bmatrix}
    \]

    ### Parameters:
    - **rgb_array** (*numpy.ndarray* or *memoryview slice*):  
      A `(width, height, channels)` uint8 array (e.g., RGB or RGBA format).
      
    - **npass** (*int*, optional):  
      Number of blur passes to apply (**default is `1`**).
      
    ### Returns:
    - **None** (modifies `rgb_array` in place).

    ### Notes:
    - Uses **convolution** to efficiently apply the blur effect.
    - **Two-pass approach** (horizontal + vertical) ensures efficient processing.
    - Suitable for **image processing, graphics effects, and real-time rendering**.

    """
    # Check that the input array is a valid type
    if not (isinstance(rgb_array, numpy.ndarray) or rgb_array.__class__.__name__ == '_memoryviewslice'):
        raise TypeError("Input rgb_array must be a numpy.ndarray or memoryviewslice.")

    if not is_uint8(rgb_array):
        raise TypeError(
            "\nbgr_array Expecting uint8 (unsigned char) data type got %s" % rgb_array.dtype)

    cdef:
        Py_ssize_t w, h
        unsigned short int bit_size

    w, h, bit_size = rgb_array.shape[:3]

    # kernel 5x5 separable
    cdef:
        short int kernel_half = 2
        int x, y, xx, yy, n
        float r, g, b
        char kernel_offset
        unsigned char red, green, blue
        float *k


    with nogil:
        for n in range(npass):
            # horizontal convolution
            for x in prange(0, w, schedule=SCHEDULE, num_threads=THREADS):

                for y in prange(0, h):  # range [0..w-1]

                    r, g, b = <unsigned char> 0, <unsigned char> 0, <unsigned char> 0

                    if bit_size==4 and rgb_array[x, y, 3] == 0:
                        continue

                    for kernel_offset in range(-kernel_half, kernel_half + 1):

                        k = &GAUSS_KERNEL[kernel_offset + kernel_half]

                        xx = x + kernel_offset

                        # check boundaries.
                        # Fetch the edge pixel for the convolution
                        if xx < 0:
                            continue

                        if xx > w - 1:
                            continue

                        red, green, blue = rgb_array[xx, y, 0],\
                            rgb_array[xx, y, 1], rgb_array[xx, y, 2]

                        if red + green + blue == <unsigned short int>0:
                            continue

                        r = r + red * k[0]
                        g = g + green * k[0]
                        b = b + blue * k[0]

                    rgb_array[x, y, 0] = <unsigned char>r
                    rgb_array[x, y, 1] = <unsigned char>g
                    rgb_array[x, y, 2] = <unsigned char>b

            # Vertical convolution
            for x in prange(0,  w, schedule=SCHEDULE, num_threads=THREADS):

                for y in prange(0, h):

                    r, g, b = <unsigned char> 0, <unsigned char> 0, <unsigned char> 0

                    if bit_size == 4 and rgb_array[x, y, 3] == 0:
                        continue

                    for kernel_offset in range(-kernel_half, kernel_half + 1):

                        k = &GAUSS_KERNEL[kernel_offset + kernel_half]
                        yy = y + kernel_offset

                        if yy < 0:
                            continue

                        if yy > h-1:
                            continue

                        red, green, blue = rgb_array[x, yy, 0],\
                            rgb_array[x, yy, 1], rgb_array[x, yy, 2]
                        if red + green + blue == <unsigned short int>0:
                            continue

                        r = r + red * k[0]
                        g = g + green * k[0]
                        b = b + blue * k[0]

                    rgb_array[x, y, 0], \
                    rgb_array[x, y, 1],\
                    rgb_array[x, y, 2] = \
                        <unsigned char>r, <unsigned char>g, <unsigned char>b



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef inline void blur4bloom_c(
        object surface_,
        unsigned int npass=1,
        tmp_array=None):

    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)
    assert npass > 0, \
        "\nArgument npass must be > 0, got %s " % npass

    cdef unsigned char [:, :, :] rgb_array
    try:
        rgb_array = surface_.get_view('3')

    except Exception as e:
        raise ValueError("\nCannot reference source pixels into a 3d array.\n %s " % e)

    if not is_uint8(rgb_array):
        raise TypeError(
            "\nbgr_array Expecting uint8 (unsigned char) data type got %s" % rgb_array.dtype)

    if tmp_array is not None:
        if not is_uint8(tmp_array):
            raise TypeError(
                "\ntmp_array Expecting uint8 (unsigned char) data type got %s" % rgb_array.dtype)

    cdef:
        Py_ssize_t w, h
        unsigned short int bit_size

    w, h, bit_size = rgb_array.shape[:3]

    cdef unsigned char [:, :, ::1] convolve

    try:
        convolve = numpy.empty((w, h, bit_size), dtype=uint8) \
            if tmp_array is None else tmp_array

    except Exception as e:
        raise ValueError("\nCannot reference source pixels into a 3d array.\n %s " % e)

    if convolve is not None and hasattr(convolve, 'shape'):
        if (w, h, bit_size) != numpy.asarray(convolve).shape[:3]:
            raise ValueError(
                "\nInput arrays have different sizes, "
                "bgr_array (%s, %s, %s) and tmp_array %s" %
                (w, h, bit_size, numpy.asarray(convolve).shape[:3]))

    # kernel 5x5 separable
    cdef:
        short int kernel_half = 2
        int x, y, xx, yy, n
        float r, g, b
        char kernel_offset
        unsigned char red, green, blue
        float *k


    with nogil:
        for n in range(npass):
            # horizontal convolution
            for x in prange(0, w, schedule=SCHEDULE, num_threads=THREADS):

                for y in prange(0, h):  # range [0..w-1]

                    r, g, b = <unsigned char> 0, <unsigned char> 0, <unsigned char> 0

                    if bit_size==4 and rgb_array[x, y, 3] == 0:
                        continue

                    for kernel_offset in range(-kernel_half, kernel_half + 1):

                        k = &GAUSS_KERNEL[kernel_offset + kernel_half]

                        xx = x + kernel_offset

                        # check boundaries.
                        # Fetch the edge pixel for the convolution
                        if xx < 0:
                            continue

                        if xx > w - 1:
                            continue

                        red, green, blue = rgb_array[xx, y, 0],\
                            rgb_array[xx, y, 1], rgb_array[xx, y, 2]

                        if red + green + blue == <unsigned short int>0:
                            continue

                        r = r + red * k[0]
                        g = g + green * k[0]
                        b = b + blue * k[0]

                    convolve[x, y, 0] = <unsigned char>r
                    convolve[x, y, 1] = <unsigned char>g
                    convolve[x, y, 2] = <unsigned char>b

            # Vertical convolution
            for x in prange(0,  w, schedule=SCHEDULE, num_threads=THREADS):

                for y in prange(0, h):

                    r, g, b = <unsigned char> 0, <unsigned char> 0, <unsigned char> 0

                    if bit_size == 4 and rgb_array[x, y, 3] == 0:
                        continue

                    for kernel_offset in range(-kernel_half, kernel_half + 1):

                        k = &GAUSS_KERNEL[kernel_offset + kernel_half]
                        yy = y + kernel_offset

                        if yy < 0:
                            continue

                        if yy > h-1:
                            continue

                        red, green, blue = convolve[x, yy, 0],\
                            convolve[x, yy, 1], convolve[x, yy, 2]
                        if red + green + blue == <unsigned short int>0:
                            continue

                        r = r + red * k[0]
                        g = g + green * k[0]
                        b = b + blue * k[0]

                    rgb_array[x, y, 0], \
                    rgb_array[x, y, 1],\
                    rgb_array[x, y, 2] = \
                        <unsigned char>r, <unsigned char>g, <unsigned char>b



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef inline void blur1d_c(
        unsigned char [::1] bgr_array,  # 1D array representing an image buffer
        const unsigned int width,       # Width of the original image
        const unsigned int height,      # Height of the original image
        int npass      = 1,             # Number of blur passes (default is 1)
        bint format_32 = False          # Flag indicating whether the buffer is 32-bit (BGRA) or 24-bit (BGR)
):
    """
    Apply an in-place blur to a 1D array or memoryview slice representing pixel data.

    This function applies a **Gaussian blur** on a 1D array (or memoryview slice) with shape `(w,)`, 
    where `w` is the number of pixels. The array can represent either **BGR** (24-bit) or **BGRA** (32-bit) 
    pixel formats or any other format compatible with the `uint8` data type. The blur operation is 
    performed **in-place** on the provided array.

    The function performs **npass** passes of convolution using a **5x5 Gaussian kernel**. This can be 
    useful for image processing tasks where you need to apply a blur effect, especially on pixel data 
    represented as a 1D array (such as when the image data is flattened).

    ### Parameters:
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

    ### Returns:
    - **None**:  
      The function performs the blur in-place on `bgr_array` and does not return a value.

    ### Notes:
    - The function uses **Gaussian convolution** with a 5x5 kernel 
      used in two passes.
      \[
      \frac{1}{256} \begin{bmatrix} 
      1 & 4 & 6 & 4 & 1 \\ 
      4 & 16 & 24 & 16 & 4 \\ 
      6 & 24 & 36 & 24 & 6 \\ 
      4 & 16 & 24 & 16 & 4 \\ 
      1 & 4 & 6 & 4 & 1 
      \end{bmatrix}
      \]
    - The image is processed in two passes (horizontal and vertical) for efficiency.
    - **Edge handling**: Pixels outside the image boundaries are replaced by adjacent edge values.
    - The function operates **in-place**, meaning the original `bgr_array` is modified directly.
    - This function is compatible with **BGR** and **BGRA** pixel formats
        (or other formats with the same shape and data type).
    """

    cdef:
        Py_ssize_t length = 0  # Stores the length of the image buffer

    # Get the length of bgr_array
    try:
        length = len(bgr_array)

    except Exception as e:
        raise ValueError(f"\nArray length is 'undefined'.\n{e}")


    # Define convolution kernel parameters
    cdef:
        float r, g, b  # Temporary variables for storing RGB values
        unsigned short int bit_size = 3  # Default to 3 bytes per pixel (RGB)
        int i, n
        unsigned char * p1
        float * p2

    # If format_32 is True, adjust bit_size to 4 bytes per pixel (BGRA)
    if format_32:
        bit_size = 4

    # Compute the row length in bytes (width * bytes per pixel)
    cdef Py_ssize_t row = width * bit_size

    with nogil:
        for n in range(npass):  # Perform the blur operation multiple times if needed

            # Horizontal convolution pass
            for i in prange(0, length, bit_size, schedule=METHOD, num_threads=THREADS):

                r, g, b = 0, 0, 0  # Reset RGB values

                # Skip blurring pixels with alpha = 0 in 32-bit images (transparent pixels)
                if format_32 and bgr_array[i + 3] == 0:
                    continue

                # Pointer to the current pixel
                p1 = &bgr_array[i]
                p2 = &GAUSS_KERNEL[0]  # Pointer to the Gaussian kernel

                # Handle edge cases where pixels are near the boundary
                if i < 2 * bit_size:
                    p1 = &bgr_array[2 * bit_size]  # Shift to a safer region
                elif i > (length - 2 * bit_size):
                    p1 = &bgr_array[i - 2 * bit_size]  # Shift backward

                # Apply the Gaussian kernel to the red, green, and blue channels
                r = (p1 - 2 * bit_size)[0] * p2[0] + \
                    (p1 - bit_size)[0] * p2[1] + \
                    p1[0] * p2[2] + \
                    (p1 + bit_size)[0] * p2[3] + \
                    (p1 + 2 * bit_size)[0] * p2[4]

                g = (p1 - 2 * bit_size + 1)[0] * p2[0] + \
                    (p1 - bit_size + 1)[0] * p2[1] + \
                    (p1 + 1)[0] * p2[2] + \
                    (p1 + bit_size + 1)[0] * p2[3] + \
                    (p1 + 2 * bit_size + 1)[0] * p2[4]

                b = (p1 - 2 * bit_size + 2)[0] * p2[0] + \
                    (p1 - bit_size + 2)[0] * p2[1] + \
                    (p1 + 2)[0] * p2[2] + \
                    (p1 + bit_size + 2)[0] * p2[3] + \
                    (p1 + 2 * bit_size + 2)[0] * p2[4]

                # Store the computed values in the temporary buffer
                bgr_array[i]     = <unsigned char>r
                bgr_array[i + 1] = <unsigned char>g
                bgr_array[i + 2] = <unsigned char>b

            # Vertical convolution pass
            for i in prange(0, length, bit_size, schedule=METHOD, num_threads=THREADS):

                r, g, b = 0, 0, 0  # Reset RGB values

                if format_32 and bgr_array[i + 3] == 0:
                    continue  # Skip fully transparent pixels

                p1 = &bgr_array[i]  # Use the temporary buffer for the second pass
                p2 = &GAUSS_KERNEL[0]

                # Handle boundary cases for vertical blur
                if i - 2 * row < 0:
                    p1 = &bgr_array[i + 2 * row]
                elif i > length - 2 * row:
                    p1 = &bgr_array[i - 2 * row]

                r = (p1 - 2 * row)[ 0 ] * p2[ 0 ] + \
                    (p1 - row)[ 0 ] * p2[ 1 ] + \
                    p1[ 0 ] * p2[ 2 ] + \
                    (p1 + row)[ 0 ] * p2[ 3 ] + \
                    (p1 + 2 * row)[ 0 ] * p2[ 4 ]

                g = (p1 - 2 * row + 1)[ 0 ] * p2[ 0 ] + \
                    (p1 - row + 1)[ 0 ] * p2[ 1 ] + \
                    (p1 + 1)[ 0 ] * p2[ 2 ] + \
                    (p1 + row + 1)[ 0 ] * p2[ 3 ] + \
                    (p1 + 2 * row + 1)[ 0 ] * p2[ 4 ]

                b = (p1 - 2 * row + 2)[ 0 ] * p2[ 0 ] + \
                    (p1 - row + 2)[ 0 ] * p2[ 1 ] + \
                    (p1 + 2)[ 0 ] * p2[ 2 ] + \
                    (p1 + row + 2)[ 0 ] * p2[ 3 ] + \
                    (p1 + 2 * row + 2)[ 0 ] * p2[ 4 ]

                # Apply the vertical blur and store the final values in the original buffer
                bgr_array[ i ]    = <unsigned char>r
                bgr_array[ i + 1] = <unsigned char>g
                bgr_array[ i + 2] = <unsigned char>b




@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef unsigned char [::1] blur1d_cp_c(
        const unsigned char [::1] bgr_array,
        const unsigned int width,
        const unsigned int height,
        unsigned int npass = 1,
        bint format_32     = False):

    """
    Apply a blur effect to a 1D buffer (memoryview slice) and return a copy.

    This function takes a **1D buffer or memoryview slice** containing pixel data in **BGR(A)** format 
    (or any other compatible pixel format) and applies a **Gaussian blur**. The blurred result is 
    returned as a new array, leaving the original buffer unmodified.

    The function supports both **24-bit (BGR)** and **32-bit (BGRA)** pixel formats. The `format_32` 
    flag must be set accordingly:  
    - **`format_32=False`** → for **BGR (24-bit)** format.  
    - **`format_32=True`** → for **BGRA (32-bit)** format (includes an alpha channel).  

    ### Example Usage:
    **For a 24-bit image (BGR format):**
    ```python
    arr = blur1d_cp_c(
        image.get_buffer(),
        width=800, height=600,
        npass=1,
        format_32=False
    )
    image = pygame.image.frombuffer(arr, (800, 600), 'BGR')
    ```

    **For a 32-bit image (BGRA format):**
    ```python
    arr = blur1d_cp_c(
        source.get_buffer(),
        width=800, height=600,
        npass=5,
        format_32=True
    )
    image = pygame.image.frombuffer(arr, (800, 600), 'BGRA')
    ```

    ### Parameters:
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

    ### Returns:
    - **numpy.ndarray** or **memoryview slice** (`shape (w,)`, `dtype=uint8`):  
      A **blurred copy** of the input array, maintaining the original pixel format.

    ### Notes:
    - This function does **not modify the input buffer**; it returns a **new** blurred array.
    - The blur operation uses **Gaussian convolution** in multiple passes.
    - The **pixel format of the output array is identical** to the input buffer.
    - Providing an appropriate `format_32` flag ensures correct handling of **alpha transparency**.
    
    """

    cdef:
           Py_ssize_t length = 0
           Py_ssize_t length_ = 0

    try:
       length = len(bgr_array)

    except Exception as e:
       raise ValueError(
           "\nArray length is 'undefined'.\n%s " % e)


    # cdef unsigned char [::1] convolve
    cdef unsigned char [::1] convolved
    convolved = numpy.asarray(bgr_array, dtype = uint8)

    # kernel 5x5 separable
    cdef:
        float r, g, b  # Temporary variables for storing RGB values
        unsigned short int bit_size = 3  # Default to 3 bytes per pixel (RGB)
        int i, n
        unsigned char * p1
        float * p2

        # If format_32 is True, adjust bit_size to 4 bytes per pixel (RGBA)
    if format_32:
        bit_size = 4

        # Compute the row length in bytes (width * bytes per pixel)
    cdef Py_ssize_t row = width * bit_size
    with nogil:
        for n in range(npass):
            # horizontal convolution
            # goes through all RGB values of the buffer and apply the convolution
            for i in prange(0, length, bit_size, schedule=METHOD, num_threads=THREADS):

                r, g, b = 0, 0, 0  # Reset RGB values

                # Skip blurring pixels with alpha = 0 in 32-bit images (transparent pixels)
                if format_32 and convolved[ i + 3 ] == 0:
                    convolved[ i ] = 0
                    convolved[ i + 1 ] = 0
                    convolved[ i + 2 ] = 0
                    continue

                # Pointer to the current pixel
                p1 = &convolved[ i ]
                p2 = &GAUSS_KERNEL[ 0 ]  # Pointer to the Gaussian kernel

                # Handle edge cases where pixels are near the boundary
                if i < 2 * bit_size:
                    p1 = &convolved[ 2 * bit_size ]  # Shift to a safer region
                elif i > (length - 2 * bit_size):
                    p1 = &convolved[ i - 2 * bit_size ]  # Shift backward

                # Apply the Gaussian kernel to the red, green, and blue channels
                r = (p1 - 2 * bit_size)[ 0 ] * p2[ 0 ] + \
                    (p1 - bit_size)[ 0 ] * p2[ 1 ] + \
                    p1[ 0 ] * p2[ 2 ] + \
                    (p1 + bit_size)[ 0 ] * p2[ 3 ] + \
                    (p1 + 2 * bit_size)[ 0 ] * p2[ 4 ]

                g = (p1 - 2 * bit_size + 1)[ 0 ] * p2[ 0 ] + \
                    (p1 - bit_size + 1)[ 0 ] * p2[ 1 ] + \
                    (p1 + 1)[ 0 ] * p2[ 2 ] + \
                    (p1 + bit_size + 1)[ 0 ] * p2[ 3 ] + \
                    (p1 + 2 * bit_size + 1)[ 0 ] * p2[ 4 ]

                b = (p1 - 2 * bit_size + 2)[ 0 ] * p2[ 0 ] + \
                    (p1 - bit_size + 2)[ 0 ] * p2[ 1 ] + \
                    (p1 + 2)[ 0 ] * p2[ 2 ] + \
                    (p1 + bit_size + 2)[ 0 ] * p2[ 3 ] + \
                    (p1 + 2 * bit_size + 2)[ 0 ] * p2[ 4 ]

                # Store the computed values in the temporary buffer
                convolved[ i ] = <unsigned char> r
                convolved[ i + 1 ] = <unsigned char> g
                convolved[ i + 2 ] = <unsigned char> b

                # Vertical convolution pass
            for i in prange(0, length, bit_size, schedule = METHOD, num_threads = THREADS):

                r, g, b = 0, 0, 0  # Reset RGB values

                if format_32 and convolved[ i + 3 ] == 0:
                    continue  # Skip fully transparent pixels

                p1 = &convolved[ i ]  # Use the temporary buffer for the second pass
                p2 = &GAUSS_KERNEL[ 0 ]

                # Handle boundary cases for vertical blur
                if i - 2 * row < 0:
                    p1 = &convolved[ i + 2 * row ]
                elif i > length - 2 * row:
                    p1 = &convolved[ i - 2 * row ]

                r = (p1 - 2 * row)[ 0 ] * p2[ 0 ] + \
                    (p1 - row)[ 0 ] * p2[ 1 ] + \
                    p1[ 0 ] * p2[ 2 ] + \
                    (p1 + row)[ 0 ] * p2[ 3 ] + \
                    (p1 + 2 * row)[ 0 ] * p2[ 4 ]

                g = (p1 - 2 * row + 1)[ 0 ] * p2[ 0 ] + \
                    (p1 - row + 1)[ 0 ] * p2[ 1 ] + \
                    (p1 + 1)[ 0 ] * p2[ 2 ] + \
                    (p1 + row + 1)[ 0 ] * p2[ 3 ] + \
                    (p1 + 2 * row + 1)[ 0 ] * p2[ 4 ]

                b = (p1 - 2 * row + 2)[ 0 ] * p2[ 0 ] + \
                    (p1 - row + 2)[ 0 ] * p2[ 1 ] + \
                    (p1 + 2)[ 0 ] * p2[ 2 ] + \
                    (p1 + row + 2)[ 0 ] * p2[ 3 ] + \
                    (p1 + 2 * row + 2)[ 0 ] * p2[ 4 ]

                # Apply the vertical blur and store the final values in the original buffer
                convolved[ i ]     = <unsigned char> r
                convolved[ i + 1 ] = <unsigned char> g
                convolved[ i + 2 ] = <unsigned char> b

    return convolved


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef blur3d_cp(unsigned char [:, :, :] rgb_array):
    """
    Apply a blur effect to a 3D numpy array or memoryviewslice (returns a copy).

    This function performs a blur operation on a 3D array, such as an image, and returns a **blurred copy**.
    It supports **RGB(A)** or other pixel formats, maintaining the original format in the returned array.

    ### Parameters:
    - **rgb_array** (*numpy.ndarray* or *memoryviewslice*, shape *(w, h, n)*, dtype *uint8*):  
      - **3D array** containing pixel data in **RGB(A)** or any compatible pixel format (e.g., BGRA).  
      - The shape should be **(width, height, channels)** where channels typically represent color
       components (3 for RGB, 4 for RGBA, etc.).

    ### Returns:
    - **numpy.ndarray**: A **new array** (blurred copy) with the same shape and pixel format as `rgb_array`.  
      - The returned array will have the same **width**, **height**, and **pixel format** as the input,
       but with the blur effect applied.

    ### Notes:
    - The function **does not modify** the input array in-place; it returns a **new blurred array**.  
    - The output will preserve the same **color channels** (e.g., RGB or RGBA).
    """

    # Check that the input array is a valid type
    if not (isinstance(rgb_array, numpy.ndarray) or rgb_array.__class__.__name__ == '_memoryviewslice'):
        raise TypeError("Input rgb_array must be a numpy.ndarray or memoryviewslice.")

    if not is_uint8(rgb_array):
        raise TypeError(
            "\nrgb_array Expecting uint8 (unsigned char) data type got %s" % rgb_array.dtype)

    w, h, bit_size = rgb_array.shape[ :3 ]

    # Only RGB(A) array supported (w, h, 3|4)
    if bit_size not in (3, 4):
        raise ValueError('\nIncorrect bit_size, support only RGB(A)')

    return numpy.ndarray(
        shape=(w, h, bit_size),
        buffer=blur3d_cp_c(rgb_array),
        dtype=numpy.uint8,
        order='C'
    )


# todo simplify arrays
@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef inline unsigned char [:, :, :] blur3d_cp_c(unsigned char [:, :, :] rgb_array):
    """
    Apply a blur effect to a 3D numpy array or memoryviewslice (returns a copy).

    This function performs a blur operation on a 3D array, such as an image, and returns a **blurred copy**.
    It supports **RGB(A)** or other pixel formats, maintaining the original format in the returned array.

    ### Parameters:
    - **rgb_array** (*numpy.ndarray* or *memoryviewslice*, shape *(w, h, n)*, dtype *uint8*):  
      - **3D array** containing pixel data in **RGB(A)** or any compatible pixel format (e.g., BGRA).  
      - The shape should be **(width, height, channels)** where channels typically represent color
       components (3 for RGB, 4 for RGBA, etc.).

    ### Returns:
    - **numpy.ndarray**: A **new array** (blurred copy) with the same shape and pixel format as `rgb_array`.  
      - The returned array will have the same **width**, **height**, and **pixel format** as the input,
       but with the blur effect applied.

    ### Notes:
    - The function **does not modify** the input array in-place; it returns a **new blurred array**.  
    - The output will preserve the same **color channels** (e.g., RGB or RGBA).
    """


    cdef int w, h, bit_size

    try:
        w, h, bit_size = (<object>rgb_array).shape[:3]

    except (ValueError, pygame.error) as e:
        raise ValueError('\nArray shape not understood.')

    cdef:

        short int kernel_half = 2
        unsigned char [:, :, ::1] convolve = numpy.empty((w, h, bit_size), dtype=uint8)
        unsigned char [:, :, ::1] convolved = numpy.empty((h, w, bit_size), dtype=uint8)
        short int kernel_length = <int>len(GAUSS_KERNEL)
        int x, y, xx, yy
        float k, r, g, b
        char kernel_offset
        unsigned char red, green, blue

    with nogil:
        # horizontal convolution
        for y in prange(0, h, schedule=SCHEDULE, num_threads=THREADS):  # range [0..h-1)

            for x in range(0, w):  # range [0..w-1]

                r, g, b = 0, 0, 0

                if bit_size == 4 and rgb_array[x, y, 3 ] == 0:
                    convolve[ x, y, 0 ] = 0
                    convolve[ x, y, 1 ] = 0
                    convolve[ x, y, 2 ] = 0
                    convolve[ x, y, 3 ] = 0
                    continue

                for kernel_offset in range(-kernel_half, kernel_half + 1):

                    k = GAUSS_KERNEL[kernel_offset + kernel_half]

                    xx = x + kernel_offset

                    # check boundaries.
                    # Fetch the edge pixel for the convolution
                    if xx < 0:
                        red, green, blue = rgb_array[0, y, 0],\
                        rgb_array[0, y, 1], rgb_array[0, y, 2]

                    elif xx > (w - 1):
                        red, green, blue = rgb_array[w-1, y, 0],\
                        rgb_array[w-1, y, 1], rgb_array[w-1, y, 2]

                    else:
                        red, green, blue = rgb_array[xx, y, 0],\
                            rgb_array[xx, y, 1], rgb_array[xx, y, 2]

                    r = r + red * k
                    g = g + green * k
                    b = b + blue * k

                convolve[x, y, 0] = <unsigned char>r
                convolve[x, y, 1] = <unsigned char>g
                convolve[x, y, 2] = <unsigned char>b
                if bit_size == 4:
                    convolve[x, y, 3] = <unsigned char>rgb_array[x, y, 3]

        # Vertical convolution
        for x in prange(0,  w, schedule=SCHEDULE, num_threads=THREADS):

            for y in range(0, h):

                r, g, b = 0, 0, 0

                if bit_size == 4 and rgb_array[x, y, 3 ] == 0:
                    convolved[ y, x, 0 ] = 0
                    convolved[ y, x, 1 ] = 0
                    convolved[ y, x, 2 ] = 0
                    convolved[ y, x, 3 ] = 0
                    continue

                for kernel_offset in range(-kernel_half, kernel_half + 1):

                    k = GAUSS_KERNEL[kernel_offset + kernel_half]
                    yy = y + kernel_offset

                    if yy < 0:
                        red, green, blue = convolve[x, 0, 0],\
                        convolve[x, 0, 1], convolve[x, 0, 2]
                    elif yy > (h -1):
                        red, green, blue = convolve[x, h-1, 0],\
                        convolve[x, h-1, 1], convolve[x, h-1, 2]
                    else:
                        red, green, blue = convolve[x, yy, 0],\
                            convolve[x, yy, 1], convolve[x, yy, 2]

                    r = r + red * k
                    g = g + green * k
                    b = b + blue * k

                convolved[ y, x, 0 ] = <unsigned char> r
                convolved[ y, x, 1 ] = <unsigned char> g
                convolved[ y, x, 2 ] = <unsigned char> b
                if bit_size == 4:
                    convolved[ y, x, 3 ] = <unsigned char> rgb_array[ x, y, 3 ]

    return convolved