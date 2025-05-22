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
This library is a GPU-accelerated image processing library that leverages
CUDA (via CuPy) to perform various image transformations and filters efficiently on NVIDIA GPUs.
Its main purpose is to speed up computationally expensive image processing operations that 
would otherwise be slow on a CPU.

Key Features & Use Cases
1. GPU Memory & Hardware Information
Retrieves details about the available GPU memory, PCI bus ID, and compute capability.
Useful for dynamically managing memory allocation and optimizing parallel processing tasks.

2. Image Processing & Filters
Provides functions to apply common image filters such as:

Inversion (invert_gpu)
Grayscale & Sepia (grayscale_gpu, sepia_gpu)
Edge detection (Sobel, Prewitt, Canny)
Blur & smoothing filters (Gaussian, Median, Bilateral)
Embossing & Sharpening
Optimized for real-time video and image processing by executing these operations in parallel on the GPU.

3. Color Manipulation & Enhancements
Adjust brightness, contrast, saturation, HSL/HSV values, and perform color reduction.
Ideal for image enhancement, augmented reality (AR) applications, and photo editing software.

4. Special Effects & Transformations
Includes artistic transformations like:

Swirl, Fisheye, Wave, Ripple effects
Chromatic aberration & RGB splitting
Cartoon effect & Bloom filters
Dithering & Heatmap effects
Useful for game development, VFX, and computer graphics applications.

5. Geometric Transformations
Supports mirroring, zooming, downscaling, and blending of images.
Can be used for image compression, texture mapping, and real-time rendering.

6. Real-Time Video Processing
Functions like ripple_effect_gpu(), predator_gpu(), and heatmap_gpu() indicate support for real-time video effects.
Could be integrated into streaming software, video filters, or surveillance systems.

Potential Applications
✅ Real-time Image & Video Processing – Enhancing camera feeds in real-time.
✅ Computer Vision & AI Preprocessing – Applying fast filters before passing images to ML models.
✅ Game Development & Graphics – Special effects for gaming & simulation environments.
✅ Augmented & Virtual Reality (AR/VR) – Optimizing visuals for immersive experiences.
✅ High-performance Photography Tools – Editing large images efficiently using the GPU.

Summary
This library is designed to offload complex image processing tasks from the CPU to the GPU, 
making operations significantly faster and more efficient. 
It is ideal for applications that require real-time performance, 
such as games, AR/VR, video streaming, and AI-based image processing.
"""



from PygameShader.config import __VERSION__

import warnings

from PygameShader.shader cimport heatmap_c
from PygameShader.shader import shader_bloom_fast1
from PygameShader.misc cimport is_type_memoryview

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
    from cython.parallel cimport prange

except ImportError:
    raise ImportError("\n<cython> library is missing on your system."
          "\nTry: \n   C:\\pip install cython on a window command prompt.")

import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

# PYGAME IS REQUIRED
try:
    import pygame
    from pygame import Color, Surface, SRCALPHA, RLEACCEL, BufferProxy, HWACCEL, HWSURFACE, \
    QUIT, K_SPACE, BLEND_RGB_ADD, Rect, BLEND_RGB_MAX, BLEND_RGB_MIN
    from pygame.surfarray import pixels3d, array_alpha, pixels_alpha, array3d, \
        make_surface, blit_array, pixels_red, \
    pixels_green, pixels_blue
    from pygame.image import frombuffer, fromstring, tostring
    from pygame.math import Vector2
    from pygame import _freetype
    from pygame._freetype import STYLE_STRONG, STYLE_NORMAL
    from pygame.transform import scale, smoothscale, rotate, scale2x
    from pygame.pixelcopy import array_to_surface

except ImportError:
    raise ImportError("\n<Pygame> library is missing on your system."
          "\nTry: \n   C:\\pip install pygame on a window command prompt.")

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

try:
    import cupy
    import cupy as cp
    import cupyx.scipy.ndimage
    from cupyx.scipy import ndimage
except ImportError:
    raise ImportError("\n<cupy> library is missing on your system."
          "\nTry: \n   C:\\pip install cupy on a window command prompt.")

from libc.stdlib cimport malloc, free

from libc.math cimport sqrtf as sqrt, atan2f as atan2, sinf as sin,\
    cosf as cos, nearbyintf as nearbyint, expf as exp, powf as pow, floorf as floor, \
roundf as round_c, fminf as fmin, fmaxf as fmax
from libc.string cimport memcpy

DEF ONE_255 = 1.0/255.0

CP_VERSION = cupy.__version__
GPU_DEVICE = cupy.cuda.Device()

# https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities
# Maximum number of resident grids per device (Concurrent Kernel Execution)
COMPUTE_CAPABILITY = {
    '35':32,  '37':32,  '50':32,  '52':32,	'53':16,
    '60':128, '61':32,  '62':16,  '70':128, '72':16,
    '75':128, '80':128, '86':128, '87':128
}

# free dedicated memory
# The amount of free memory, in bytes. total: The total amount of memory, in bytes.
@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef long long int get_gpu_free_mem():
    """
    Retrieve the available free memory on the GPU.

    This function queries the GPU device for its current free memory and 
    returns the amount in bytes. It is useful for monitoring memory usage 
    when working with GPU-accelerated computations, ensuring that operations 
    do not exceed available memory.

    Returns:
    --------
    long long int
        The amount of free GPU memory in bytes.

    Example Usage:
    --------------
    free_mem = get_gpu_free_mem()
    print(f"Available GPU Memory: {free_mem / (1024**2):.2f} MB")
    """

    return GPU_DEVICE.mem_info[0]


# get max dedicated memory
# total: The total amount of memory, in bytes.
@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef long long int get_gpu_maxmem():
    """
    Retrieve the total memory capacity of the GPU.

    This function queries the GPU device to determine the maximum available 
    memory, which represents the total VRAM capacity of the GPU.

    Returns:
    --------
    long long int
        The total GPU memory in bytes.

    Example Usage:
    --------------
    total_mem = get_gpu_maxmem()
    print(f"Total GPU Memory: {total_mem / (1024**2):.2f} MB")
    """

    return GPU_DEVICE.mem_info[1]


# GPU pci bus id
# Returned identifier string for the device in the following
# format [domain]:[bus]:[device].[function] where domain, bus,
# device, and function are all hexadecimal values.
@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef str get_gpu_pci_bus_id():
    """
    Retrieve the PCI Bus ID of the GPU.

    This function queries the GPU device and returns its PCI Bus ID, 
    which uniquely identifies the GPU within the system. The PCI Bus ID 
    is useful for multi-GPU setups, debugging, and device management.

    Returns:
    --------
    str
        The PCI Bus ID of the GPU in the format "Domain:Bus:Device.Function"
        (e.g., "0000:01:00.0").

    Example Usage:
    --------------
    pci_id = get_gpu_pci_bus_id()
    print(f"GPU PCI Bus ID: {pci_id}")
    """

    return GPU_DEVICE.pci_bus_id


# Compute capability of this device.
# The capability is represented by a string containing the major
# index and the minor index. For example, compute capability 3.5
# is represented by the string ‘35’.
@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef str get_compute_capability():
    """
    Retrieve the compute capability of the GPU.

    Compute capability represents the GPU architecture version and determines 
    its compatibility with various CUDA features. Higher compute capability 
    values indicate support for more advanced features and optimizations.

    Returns:
    --------
    str
        The compute capability of the GPU as a string in the format "major.minor"
        (e.g., "7.5" for NVIDIA Turing GPUs).

    Example Usage:
    --------------
    capability = get_compute_capability()
    print(f"GPU Compute Capability: {capability}")
    """

    return GPU_DEVICE.compute_capability


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef unsigned int get_max_grid_per_block():
    """
    Retrieve the maximum number of grid blocks per multiprocessor.

    This function returns the maximum number of grid blocks that can be launched 
    per multiprocessor, based on the GPU's compute capability. This value is 
    essential for optimizing parallel workloads and ensuring efficient resource utilization.

    Returns:
    --------
    unsigned int
        The maximum number of grid blocks per multiprocessor.

    Example Usage:
    --------------
    max_grid_blocks = get_max_grid_per_block()
    print(f"Max grid blocks per multiprocessor: {max_grid_blocks}")

    Notes:
    ------
    - The returned value depends on the GPU's compute capability.
    - Refer to NVIDIA's CUDA programming guide for specific values per architecture.
    """

    return COMPUTE_CAPABILITY[get_compute_capability()]



# USED BY block_grid
@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef get_divisors(int n):
    """
    Compute the divisors of a given integer.

    This function finds all positive divisors of the given integer `n`, excluding `n` itself. 
    It iterates from 1 to `n/2`, as any divisor greater than `n/2` (except `n`) is redundant.

    Parameters:
    -----------
    n : int
        The integer for which to compute the divisors. Must be a positive integer.

    Returns:
    --------
    list[int]
        A list of all positive divisors of `n`, excluding `n` itself.

    Example Usage:
    --------------
    >>> get_divisors(12)
    [1, 2, 3, 4, 6]

    Notes:
    ------
    - If `n` is 1, the function returns an empty list since 1 has no proper divisors.
    - The function runs in O(n/2) time complexity.
    """
    l = []
    for i in range(1, int(n / <float>2.0) + 1):
        if n % i == 0:
            l.append(i)
    return l


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef block_grid(int w, int h):
    """
    Automatically computes optimal grid and block sizes for GPU execution.

    This function determines the best grid and block configuration for GPU execution based on 
    the dimensions of the display (or computational domain). It ensures that the computed grid 
    and block sizes are valid and compatible with the given width (`w`) and height (`h`).

    Parameters
    ----------
    w : int
        The width of the display or computational domain. Must be greater than 0.

    h : int
        The height of the display or computational domain. Must be greater than 0.

    Returns
    -------
    tuple
        A tuple containing:
        - `grid`: (y, x) - The computed grid size (number of blocks in each dimension).
        - `block`: (yy, xx) - The computed block size (number of threads per block in each dimension).

    Raises
    ------
    AssertionError
        - If `w` or `h` is less than or equal to 0.
        - If the computed grid and block sizes are not valid (i.e., they do not exactly cover the input dimensions).

    Notes
    -----
    - The function first determines possible divisors of `w` and `h`, then selects block sizes that 
      do not exceed 32 (a common maximum block size for GPU execution).
    - The final grid size is determined by dividing the input dimensions by the selected block sizes.
    - If the computed configuration is invalid (i.e., `yy * y != h` or `xx * x != w`), an assertion 
      error is raised, suggesting that manual configuration might be necessary.

    Example
    -------
    >>> grid, block = block_grid(128, 64)
    >>> print(grid, block)
    (4, 4) (16, 16)
    """


    assert w > 0, "Width (w) must be greater than zero."
    assert h > 0, "Height (h) must be greater than zero."

    # Get valid divisors of w and h (sorted in ascending order).
    divisors_w = numpy.array(get_divisors(w), dtype = numpy.int32)
    divisors_h = numpy.array(get_divisors(h), dtype = numpy.int32)

    # Compute possible block sizes by dividing the width and height by their divisors.
    possible_blocks_w = (w // divisors_w)
    possible_blocks_h = (h // divisors_h)

    # Filter out block sizes greater than 32 (since typical GPU block sizes max out at 32).
    valid_blocks_w = possible_blocks_w[ possible_blocks_w <= 32 ]
    valid_blocks_h = possible_blocks_h[ possible_blocks_h <= 32 ]

    # Select the smallest valid block size for efficiency.
    block_x = int(numpy.maximum(1, valid_blocks_w[ 0 ]))  # Ensure at least 1
    block_y = int(numpy.maximum(1, valid_blocks_h[ 0 ]))

    # Compute the corresponding grid sizes.
    grid_x = w // block_x
    grid_y = h // block_y

    # Ensure the computed grid and block sizes exactly cover the given width and height.
    assert block_y * grid_y == h, f"Invalid grid {grid_y, grid_x} or block {block_y, block_x}, adjust manually."
    assert block_x * grid_x == w, f"Invalid grid {grid_y, grid_x} or block {block_y, block_x}, adjust manually."

    return (grid_y, grid_x), (block_y, block_x)



volume = ["", "K", "M", "G", "T", "P", "E", "Z", "Y"]

def conv(v):
    """
    Convert a value to a human-readable format,
    scaling it by powers of 1024 (e.g., KB, MB, GB).

    This function divides the input value `v` by 1024 repeatedly until
    it is less than or equal to 1024.
    It then returns the value with its appropriate unit (e.g., KB, MB, GB)
    based on the number of divisions performed.

    :param v: float or int;
        the value to be converted (typically representing bytes or data size).

    :return: str;
        a human-readable string representing the value with an appropriate unit, rounded to 3 decimal places.

    The function uses the following units:
    - 1024 bytes = 1 KB
    - 1024 KB = 1 MB
    - 1024 MB = 1 GB
    - and so on...

    Example:
    - `conv(1048576)` would return `'1.0 MB'`
    - `conv(123456789)` would return `'117.74 MB'`

    Notes:
    - If the input value `v` is smaller than 1024, it will be returned as is with the first unit.
    - The function ensures that the appropriate unit is selected based on the scale of `v`.
    - The `volume` list is assumed to contain the units, starting from `KB` for values greater than or equal to 1024.

    """

    b = 0
    while v > 1024:
        b += 1
        v /= 1024
    if b > len(volume) - 1:
        b = len(volume)
    return str(round(v, 3))+volume[b]


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef block_and_grid_info(int w, int h):
    """
    Calculate and display the appropriate GPU block and grid dimensions for a given image size.

    This function computes the grid and block sizes required for optimal GPU parallelization.
    It utilizes the `block_grid` function to determine the best block and grid sizes based on the
    provided width (`w`) and height (`h`) of the image or display.

    The block and grid sizes are then validated to ensure they can correctly cover the entire area.
    If the validation fails, assertions are raised. Afterward, the function prints the resulting
    grid and block sizes.

    :param w: int; The width of the image or display (must be greater than 0).
    :param h: int; The height of the image or display (must be greater than 0).

    :raises AssertionError: If either `w` or `h` is non-positive, 
        or if the calculated grid and block sizes do not properly cover the image.

    :return: None; This function only prints the GPU grid and block information.
        It does not return any values.

    Example:
    ---------------
    >>> block_and_grid_info(1920, 1080)
    GPU GRID        : (grid_y=34.000000, grid_x=60.000000)
    GPU BLOCK       : (block_y=32.000000, block_x=32.000000)

    Notes:
    ---------------
    - The grid is determined by the number of blocks required to cover the image's height and width.
    - The block size refers to the size of each individual block of threads in the GPU.
    - This function assumes the GPU processing will be optimized for the calculated grid and block sizes.
    """

    assert w > 0, "Argument w cannot be < 0"
    assert h > 0, "Argument h cannot be < 0"
    grid, block = block_grid(w, h)

    assert block[0] * grid[0] == h, "\nInvalid grid or block values, you may want to set grid & block manually"
    assert block[1] * grid[1] == w, "\nInvalid grid or block values, you may want to set grid & block manually"

    print("GPU GRID        : (grid_y={grid_y:8f}, grid_x={grid_x:8f})".format(grid_y=grid[0], grid_x=grid[1]))
    print("GPU BLOCK       : (block_y={block_y:8f}, block_x={block_x:8f})".format(block_y=block[0], block_x=block[1]))


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef get_gpu_info():

    """
    Display detailed GPU information, including memory, compute capability, and PCI bus ID.

    This function retrieves and prints various details about the GPU, including:
    - The installed CuPy version.
    - The maximum grid size per block supported by the GPU.
    - The available and total GPU memory, both in bytes and human-readable formats.
    - The PCI Bus ID of the GPU.
    - The compute capability of the GPU.

    :return: None; This function prints GPU-related information to the console.

    Example output:
    ---------------
    >>> get_gpu_info()
    CUPY VERSION           : 11.2.0
    GPU MAX GRID PER BLOCK : 65535
    GPU FREE MEMORY        : (mem=8388608000.000000, (8.0 GB))
    GPU MAX MEMORY         : (mem=17179869184.000000, (16.0 GB))
    GPU PCI BUS ID         : (bus=0000:01:00.0)
    GPU CAPABILITY         : (capa=7.5)

    Notes:
    ---------------
    - The function relies on CuPy for GPU memory information.
    - The `conv` function is used to convert raw memory values into a more readable format (e.g., GB).
    - This function is useful for debugging and verifying GPU resource availability 
        before launching CUDA-based operations.
    """

    print("CUPY VERSION           : %s " % CP_VERSION)
    print("GPU MAX GRID PER BLOCK : %s" % get_max_grid_per_block())
    print("GPU FREE MEMORY : (mem={mem:8f}, ({v:5s}))".format(mem=get_gpu_free_mem(), v=conv(get_gpu_free_mem())))
    print("GPU MAX MEMORY  : (mem={mem:8f}, ({v:5s}))".format(mem=get_gpu_maxmem(), v=conv(get_gpu_maxmem())))
    print("GPU PCI BUS ID  : (bus={bus:12s})".format(bus=get_gpu_pci_bus_id()))
    print("GPU CAPABILITY  : (capa={capa:5s})".format(capa=get_compute_capability()))





@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef invert_gpu(surface_):
    """
    Perform GPU-accelerated color inversion on a Pygame surface (return a copy)

    This function transfers the image data from a `pygame.Surface` to the GPU,
    inverts the color values using CuPy, and then reconstructs a new surface 
    with the modified pixel data.

    Parameters
    ----------
    surface_ : pygame.Surface
        A Pygame surface object (24-bit or 32-bit) containing the image to be inverted.

    Returns
    -------
    pygame.Surface
        A new Pygame surface with inverted colors, preserving the original format (BGR or BGRA).

    Raises
    ------
    ValueError
        If the surface cannot be referenced as a 1D buffer or has zero length.
    TypeError
        If `surface_` is not a `pygame.Surface` object.

    Notes
    -----
    - Uses CuPy to perform the inversion directly on the GPU for optimal performance.
    - Automatically detects whether the surface is 24-bit (BGR) or 32-bit (BGRA).
    - Synchronizes the GPU to ensure completion before returning the processed image.

    Example
    -------
    ```python
    import pygame
    surface = pygame.image.load("image.png").convert()
    inverted_surface = invert_gpu(surface)
    pygame.image.save(inverted_surface, "inverted_image.png")
    ```
    """

    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)

    cdef Py_ssize_t bytesize, w, h
    bytesize = surface_.get_bytesize()
    w, h = surface_.get_size()

    cdef unsigned char [::1] cpu_array
    try:
        cpu_array = surface_.get_buffer()

    except Exception as e:
        raise ValueError("\nCannot reference source pixels into a 1d array.\n %s " % e)

    cdef:
        Py_ssize_t length = len(cpu_array)

    if length == 0:
        raise ValueError("Argument `surface_` cannot have null length.")

    gpu_array = <unsigned char> 255 - cp.asarray(cpu_array)
    cp.cuda.Device(0).synchronize()

    return pygame.image.frombuffer(
        gpu_array.tobytes(), (w, h), "BGR" if bytesize == 3 else "BGRA" )


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef void invert_gpu_inplace(surface_):
    """
    Perform in-place GPU-accelerated color inversion on a Pygame surface. 

    This function directly modifies the pixel data of the given `pygame.Surface` 
    by inverting its color values using GPU processing via CuPy. The inversion 
    is performed in-place, meaning the original surface is altered without 
    creating a new one.

    Parameters
    ----------
    surface_ : pygame.Surface
        The Pygame surface whose pixel colors will be inverted.
        Must be a valid 24-bit (BGR) or 32-bit (BGRA) surface.

    Raises
    ------
    TypeError
        If `surface_` is not a `pygame.Surface` object.
    ValueError
        If the surface buffer cannot be accessed or has zero length.

    Notes
    -----
    - The function transfers the pixel data to the GPU, performs the inversion, 
      and copies the modified pixels back to the original surface.
    - Uses CuPy for efficient GPU computation.

    Example
    -------
    ```python
    import pygame

    pygame.init()
    surface = pygame.image.load("image.png").convert()
    invert_gpu_inplace(surface)
    pygame.image.save(surface, "inverted_image.png")
    ```
    """

    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)

    cdef Py_ssize_t bytesize, w, h
    bytesize = surface_.get_bytesize()
    w, h = surface_.get_size()

    cdef unsigned char [::1] cpu_array
    try:
        cpu_array = surface_.get_buffer()

    except Exception as e:
        raise ValueError("\nCannot reference source pixels into a 1d array.\n %s " % e)

    cdef:
        Py_ssize_t length = len(cpu_array)

    if length == 0:
        raise ValueError("Argument `surface_` cannot have null length.")

    gpu_array = <unsigned char> 255 - cp.asarray(cpu_array)
    cp.cuda.Device(0).synchronize()

    cdef unsigned char [::1] arr = bytearray(gpu_array.get())

    memcpy(&cpu_array[0], &arr[0], length)



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef void invert_buffer_gpu(unsigned char [::1] bgr_array):

    """
    Performs an in-place inversion of a BGR pixel buffer using the GPU.

    This function takes a contiguous 1D BGR buffer (as a NumPy array or memoryview),
    transfers it to the GPU, inverts the pixel values (255 - pixel value), and 
    writes the result back into the original buffer.

    Parameters
    ----------
    bgr_array : numpy.ndarray or memoryview
        A contiguous 1D array representing the BGR pixel buffer. The buffer must 
        be mutable and non-empty.

    Raises
    ------
    TypeError
        If `bgr_array` is not a NumPy array or a memoryview.
    ValueError
        If `bgr_array` is empty (has zero length).

    Notes
    -----
    - This function modifies `bgr_array` in place.
    - Uses CUDA-enabled CuPy for GPU acceleration.
    - Suitable for high-performance image processing where in-place modification 
      is necessary.

    """

    if not (PyObject_IsInstance(bgr_array, numpy.ndarray) or is_type_memoryview(bgr_array)):
        raise TypeError("\nArgument bgr_array must be a "
                        "numpy.ndarray or a memoryviewtype, got %s " % type(bgr_array))

    cdef Py_ssize_t bytesize, w, h

    cdef:
        Py_ssize_t length = len(<object>bgr_array)

    if length == 0:
        raise ValueError("Argument `bgr_array` buffer cannot be null.")

    gpu_array = <unsigned char> 255 - cp.asarray(bgr_array)
    cp.cuda.Device(0).synchronize()

    cdef unsigned char [::1] temp = bytearray(gpu_array.get())

    memcpy(&bgr_array[0], &temp[0], length)


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef void invert_rgb_array_inplace(rgb_array):

    """
    Inverts the RGB values of an image (3D numpy array or memoryview) in place using GPU.

    This function performs an in-place inversion of the RGB channels in the input array.
    It operates on the assumption that the input `rgb_array` is a 3D numpy array or memoryview
    with shape (height, width, 3), where the last dimension represents the RGB color channels.

    **Note**: This operation is performed using GPU to speed up the computation.

    Parameters
    ----------
    rgb_array : numpy.ndarray or memoryview
        A 3D array or memoryview of shape (height, width, 3) containing the RGB values of the image.
        The dtype must be numpy.uint8. If `rgb_array` is a memoryview, it should be of compatible type.

    Raises
    ------
    TypeError
        If `rgb_array` is not a numpy.ndarray or memoryview.

    ValueError
        If the `rgb_array` is not of dtype numpy.uint8 or does not have 3 dimensions.
        If the width or height of the array is zero.

    Description
    -----------
    This function directly modifies the input `rgb_array` by inverting the RGB channels using
    GPU-accelerated operations. The inversion is done in place, meaning the original array will
    be updated. The GPU is used to perform the inversion efficiently.

    Example
    -------
    >>> import numpy as np
    >>> rgb_image = np.ones((100, 100, 3), dtype=np.uint8) * 255  # White image
    >>> invert_rgb_array_inplace(rgb_image)
    >>> print(rgb_image[0, 0])  # Should print [0, 0, 0] for a black pixel

    """

    if not (PyObject_IsInstance(rgb_array, numpy.ndarray) or is_type_memoryview(rgb_array)):
        raise TypeError(f"Argument rgb_array must be a numpy.ndarray "
                        f"or a memoryviewtype, got {type(rgb_array)}.")

    if PyObject_IsInstance(rgb_array, numpy.ndarray):
        if (<object>rgb_array).dtype != numpy.uint8:
            raise ValueError("\nArgument rgb_array datatype is invalid, "
                             "expecting numpy.uint8 got %s " % rgb_array.dtype)

        if (<object>rgb_array).ndim != 3:
            raise ValueError(f"Expecting 3 dimensions for rgb_array got {(<object>rgb_array).ndim}.")

    cdef:
        Py_ssize_t w, h, bytesize

    w, h, bytesize = (<object>rgb_array).shape

    if w == 0 or h == 0:
        raise ValueError("rgb_array width or height cannot be null!")

    gpu_array = cp.asarray(rgb_array)
    gpu_array = (<unsigned char>255 - gpu_array)
    cp.cuda.Device(0).synchronize()

    rgb_array[ :, :, 0 ] = gpu_array[ :, :, 0 ].get()
    rgb_array[ :, :, 1 ] = gpu_array[ :, :, 1 ].get()
    rgb_array[ :, :, 2 ] = gpu_array[ :, :, 2 ].get()




@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef sepia_gpu(surface_):
    """
    Apply a Sepia shader to a Pygame surface and return a new surface with the Sepia effect applied.

    This function works with Pygame surfaces of both 32-bit and 24-bit color depth. 
    It uses GPU-based computations to transform the pixel colors by applying a Sepia filter.

    The function first checks the type of the provided surface to ensure it's a valid 
    Pygame Surface. Then it extracts the pixel data from the surface, applies the Sepia 
    filter using a GPU-based kernel, and returns a new Pygame surface with the transformed image.

    :param surface_: pygame.Surface
        A Pygame Surface object containing the image to which the Sepia filter will be applied.
        The surface is expected to be in 32-bit or 24-bit format.

    :return: pygame.Surface
        A new Pygame Surface object with the same dimensions and the Sepia filter applied, 
        in the "RGB" format.
    """

    # Ensure the input is a valid Pygame surface object.
    if not PyObject_IsInstance(surface_, pygame.Surface):
        raise TypeError(
            "\nArgument surface_ must be a pygame.Surface "
            "type, got %s " % type(surface_))

    # Attempt to retrieve the pixel data from the surface as a 3D numpy-like array.
    cdef unsigned char [:, :, :] cpu_array_
    try:
        # Retrieve the surface pixel data as a 3D array with the '3' view (RGB channels).
        cpu_array_ = surface_.get_view('3')
    except Exception as e:
        # Raise an error if the pixel data could not be accessed or referenced.
        raise ValueError("\nCannot reference source pixels into a 3D array.\n %s " % e)

    # Extract the width and height of the surface from the array shape.
    cdef:
        Py_ssize_t w, h
    w, h = cpu_array_.shape[0], cpu_array_.shape[1]

    # Check if the surface has valid dimensions (both width and height must be greater than zero).
    if w == 0 or h == 0:
        raise ValueError("Surface width and height cannot be null!")

    # Convert the 3D pixel data array into a GPU-compatible array.
    gpu_array = cp.asarray(cpu_array_)

    # Apply the Sepia filter to the image data using the GPU-based function 'sepia_cupy'.
    sepia_cupy(gpu_array)

    # Synchronize the GPU stream to ensure that all GPU operations are completed before proceeding.
    cp.cuda.Stream.null.synchronize()

    # Convert the processed GPU array back into a Pygame surface.
    # The array is converted to bytes and then used to create a new surface with the same dimensions.
    return frombuffer(
        gpu_array.astype(dtype=cp.uint8)  # Convert the array back to uint8 type
        .transpose(1, 0, 2).get(),  # Transpose the array (swap width and height)
        (w, h),  # Dimensions of the new surface
        "RGB"  # Color format for the new surface
    )




# Define a kernel to apply the Sepia effect to RGB pixel values using element-wise operations.
sepia_kernel = cp.ElementwiseKernel(
    'float32 r, float32 g, float32 b',
    'float32 rr, float32 gg, float32 bb',

    '''
    // SEPIA RGB TRANSFORMATION
    // Apply the Sepia tone effect to the RGB channels by combining them with specific weights.
    // These formulas are based on a common Sepia transformation matrix.
    rr = (r * (float)0.393 + g * (float)0.769 + b * (float)0.189) * (float)255.0;  // New red channel
    gg = (r * (float)0.349 + g * (float)0.686 + b * (float)0.168) * (float)255.0;  // New green channel
    bb = (r * (float)0.272 + g * (float)0.534 + b * (float)0.131) * (float)255.0;  // New blue channel

    // CLAMPING RGB VALUES TO THE RANGE [0, 255]
    // Ensure that the transformed pixel values are clamped to the valid range for an 8-bit image.
    // This ensures that no value goes beyond the typical byte range (0 to 255).

    // For red channel:
    if ( rr > (float)255.0) { rr = (float)255.0; }  // If red is greater than 255, cap it to 255
    else if ( rr < 0 ) { rr = (float)0.0; }  // If red is less than 0, set it to 0

    // For green channel:
    if ( gg > (float)255.0) { gg = (float)255.0; }  // If green is greater than 255, cap it to 255
    else if ( gg < 0 ) { gg = (float)0.0; }  // If green is less than 0, set it to 0

    // For blue channel:
    if ( bb > (float)255.0) { bb = (float)255.0; }  // If blue is greater than 255, cap it to 255
    else if ( bb < 0 ) { bb = (float)0.0; }  // If blue is less than 0, set it to 0

    ''', 'sepia_kernel'
)


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef inline void sepia_cupy(gpu_array_):
    """
    Apply a Sepia filter to an image in-place using a GPU-based kernel.

    This function processes a given image array by applying a Sepia tone shader 
    to the RGB channels using elementwise operations. The input image is assumed 
    to be in the shape (width, height, 3) with uint8 data type. The function 
    utilizes a GPU kernel to perform the Sepia transformation and then returns the 
    processed image as a CPU-based numpy array with the same shape and data type.

    :param gpu_array_: cupy.ndarray
        A 3D array representing the image with shape (width, height, 3) and 
        dtype uint8. The array is expected to be located on the GPU.
        Assume the gpu_array to hold pixel format RGB 

    :return: numpy.ndarray
        A 3D array with the Sepia filter applied, shape (width, height, 3), 
        and dtype uint8, located on the CPU side.
    """
    # Normalize the GPU image array by dividing by 255.0 to convert values to float
    # and cast it to the 'cp.float32' type for GPU processing.
    gpu = (gpu_array_ / <float> 255.0).astype(dtype = cp.float32)

    # Apply the sepia filter by processing each color channel (R, G, B) using a GPU kernel
    # and updating the channels in the original image array.
    gpu_array_[ :, :, 0 ], \
    gpu_array_[ :, :, 1 ], \
    gpu_array_[ :, :, 2 ] = sepia_kernel(gpu[ :, :, 0 ], gpu[ :, :, 1 ], gpu[ :, :, 2 ])

    # The array 'gpu_array_' is modified in-place with the Sepia filter applied.
    # The output will be a modified image with Sepia tones, still on the GPU.


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef void sepia_rgb_array_gpu(rgb_array):
    """
    Apply a Sepia filter in-place to an RGB image array.

    This function processes a 32-bit or 24-bit image in-place by modifying the provided 
    `numpy.ndarray` of shape (w, h, 3) containing uint8 RGB values. The Sepia effect 
    is applied directly to the input array without creating a new array.

    The function expects the input array to have RGB values in the uint8 format. The image 
    data is transferred to the GPU, the Sepia filter is applied using a GPU kernel, 
    and the original array is updated with the transformed values.

    :param rgb_array : numpy.ndarray
        A 3D numpy array with shape (w, h, 3), representing an image with RGB pixel values 
        in the uint8 data type. The shape should represent the width (w), height (h), 
        and three channels (Red, Green, Blue).

    :return : void
        This function modifies the input `rgb_array` in-place and returns nothing.
    """

    # Check if the input is a numpy array.
    if not PyObject_IsInstance(rgb_array, numpy.ndarray):
        raise TypeError(
            "\nArgument rgb_array must be of numpy.ndarray type, but got %s " % type(rgb_array))

    # Ensure the input array has the correct datatype (uint8).
    if rgb_array.dtype != numpy.uint8:
        raise ValueError("\nArgument rgb_array must have dtype numpy.uint8, "
                         "but got %s " % rgb_array.dtype)

    # Extract the width and height of the image from the array shape.
    cdef Py_ssize_t w, h
    w, h = rgb_array.shape[ 0 ], rgb_array.shape[ 1 ]

    # Validate that the image has valid dimensions (width and height must not be zero).
    if w == 0 or h == 0:
        raise ValueError("Image width or height cannot be zero!")

    # Convert the image from CPU (numpy array) to GPU memory for faster processing.
    gpu_array = cp.asarray(rgb_array)

    # Normalize the pixel values to the range [0.0, 1.0] for the Sepia filter processing.
    # The values are converted to float32 for accurate computations.
    gpu_array = (gpu_array / <float> 255.0).astype(dtype = cp.float32)

    # Apply the Sepia kernel to the red, green, and blue channels of the image.
    # The sepia_kernel function is assumed to be a pre-defined GPU-based operation.
    rr, gg, bb = sepia_kernel(gpu_array[ :, :, 0 ], gpu_array[ :, :, 1 ], gpu_array[ :, :, 2 ])

    # Update the input rgb_array in-place with the transformed Sepia values.
    # Convert the results back to uint8 and copy them back into the original array.
    rgb_array[ :, :, 0 ] = rr.astype(cp.uint8).get()  # Red channel
    rgb_array[ :, :, 1 ] = gg.astype(cp.uint8).get()  # Green channel
    rgb_array[ :, :, 2 ] = bb.astype(cp.uint8).get()  # Blue channel



sepia_buffer_kernel = cp.RawKernel(
    r'''
    extern "C" __global__

    void sepia_buffer_kernel(
    const int w, 
    const int h, 
    unsigned char * bgr_buffer, 
    const unsigned short int bytesize
    )
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        int j = blockIdx.y * blockDim.y + threadIdx.y;

        // Max index for array shape (w, h)
        const int t_max  = w * h;    
        
        // Current index for array shape (w, h)
        const int index  = j * h + i;
        
        //Current index for array shape (w, h, 3)
        const int index1 = j * h * bytesize + i * bytesize;
        
        float rr, gg, bb;

        __syncthreads();

        if (index > 0 && index < t_max){            

            float r  = (float)bgr_buffer[index1 + 2] / 255.0f;
            float g  = (float)bgr_buffer[index1 + 1] / 255.0f;
            float b  = (float)bgr_buffer[index1  ] / 255.0f;

             // SEPIA RGB TRANSFORMATION
            // Apply the Sepia tone effect to the RGB channels by combining them with specific weights.
            // These formulas are based on a common Sepia transformation matrix.
            rr = (r * (float)0.393 + g * (float)0.769 + b * (float)0.189) * (float)255.0;  // New red channel
            gg = (r * (float)0.349 + g * (float)0.686 + b * (float)0.168) * (float)255.0;  // New green channel
            bb = (r * (float)0.272 + g * (float)0.534 + b * (float)0.131) * (float)255.0;  // New blue channel
        
            // CLAMPING RGB VALUES TO THE RANGE [0, 255]
            // Ensure that the transformed pixel values are clamped to the valid range for an 8-bit image.
            // This ensures that no value goes beyond the typical byte range (0 to 255).
        
            // For red channel:
            if ( rr > (float)255.0) { rr = (float)255.0; }  // If red is greater than 255, cap it to 255
            else if ( rr < 0 ) { rr = (float)0.0; }  // If red is less than 0, set it to 0
        
            // For green channel:
            if ( gg > (float)255.0) { gg = (float)255.0; }  // If green is greater than 255, cap it to 255
            else if ( gg < 0 ) { gg = (float)0.0; }  // If green is less than 0, set it to 0
        
            // For blue channel:
            if ( bb > (float)255.0) { bb = (float)255.0; }  // If blue is greater than 255, cap it to 255
            else if ( bb < 0 ) { bb = (float)0.0; }  // If blue is less than 0, set it to 0

            bgr_buffer[ index1 ] = (unsigned char)bb; 
            bgr_buffer[ index1 + 1 ] = (unsigned char)gg;
            bgr_buffer[ index1 + 2 ] = (unsigned char)rr;
            
            __syncthreads();

        }
    }
    ''',
    'sepia_buffer_kernel'
)

@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef void sepia_buffer_gpu(
    tuple grid_,
    tuple block_,
    const Py_ssize_t w,
    const Py_ssize_t h,
    unsigned char [::1] bgr_array,
    bint format_32=False
):
    """
    Apply a Sepia filter to a BGR or BGRA image buffer on the GPU (inplace)

    This function uses a GPU kernel to apply a Sepia effect to an image stored in either a 
    BGR or BGRA format. The operation is performed on the input image buffer (`bgr_array`), 
    which is a 1D array or memoryview representing the pixel data in either BGR or BGRA format.

    The function assumes the buffer is laid out as a 1D array, where each pixel consists of
    3 or 4 bytes (depending on whether it is in BGR or BGRA format). The Sepia effect is applied 
    in parallel on the GPU using a CUDA kernel, and the result is stored directly in the input 
    buffer.

    The kernel is launched with a specified grid and block configuration.

    :param grid_        : tuple
        The block grid dimensions to be used for the kernel launch, typically obtained 
        from a helper function such as `block_grid()`.

    :param block_       : tuple
        The block dimensions to be used for the kernel launch, typically obtained 
        from a helper function such as `block_grid()`.

    :param w            : Py_ssize_t
        The width of the image, in pixels. Must be greater than 0.

    :param h            : Py_ssize_t
        The height of the image, in pixels. Must be greater than 0.

    :param bgr_array    : unsigned char [::1]
        A 1D array or memoryview containing the image pixel data. This array represents
        the image in either BGR (3 channels) or BGRA (4 channels) format, with each pixel
        occupying 3 or 4 consecutive bytes.

    :param format_32    : bint, optional
        A boolean flag indicating whether the input array is in BGRA format (True) or BGR 
        format (False). Default is False (BGR format).

    :return            : void
        This function modifies the input `bgr_array` in-place and returns nothing.

    :raises TypeError  : If `bgr_array` is not a valid numpy.ndarray or memoryview.
    :raises ValueError : If the dimensions of the input image or the buffer length do not match
                         the expected values based on the width, height, and pixel format.
    """

    # Ensure the input image array is either a numpy ndarray or a memoryview.
    if not (PyObject_IsInstance(bgr_array, numpy.ndarray) or is_type_memoryview(bgr_array)):
        raise TypeError(
            "\nArgument bgr_array must be a numpy.ndarray or a memoryview type, got %s " % type(bgr_array))

    # Validate that the width and height are positive values.
    if w <= 0 or h <= 0:
        raise ValueError(f"Arguments `w` and `h` must be > 0, got w({w}), h({h})")

    # Ensure that the grid and block sizes are compatible with the given image dimensions.
    if grid_[0] * block_[0] != h or grid_[1] * block_[1] != w:
        raise ValueError(f"Invalid grid and block dimensions for image size ({w}, {h}). "
                         f"Ensure to use appropriate grid/block values. "
                         f"Call `grid, block = block_grid(surface.get_width(), surface.get_height())`")

    # Convert the input array (either numpy or memoryview) to a GPU array (CuPy array).
    gpu_array = cp.asarray(bgr_array)

    # Define variables for image length and byte size per pixel.
    cdef Py_ssize_t length, bytesize

    # Determine byte size based on whether it's a BGR (3 channels) or BGRA (4 channels) image.
    bytesize = 4 if format_32 else 3

    # Get the total length of the input image data.
    length = len(gpu_array)

    # Validate that the length of the input array matches the expected length for the image.
    if length != w * h * bytesize:
        raise ValueError(
            f"Expecting bgr_array length of {w * h * bytesize}, "
            f"but got {length}. "
            f"Use `format_32=True` for BGRA buffers and `format_32=False` for BGR buffers.")

    # Launch the GPU kernel to apply the Sepia effect to the image buffer.
    # The kernel is provided with grid/block configurations and image dimensions.
    sepia_buffer_kernel(
        grid_,
        block_,
        (w, h, gpu_array, bytesize)
    )

    # Synchronize the GPU device to ensure all kernel operations are completed before proceeding.
    cp.cuda.Device(0).synchronize()

    # Copy the processed image data from GPU back to the input buffer in-place.
    # The result is transferred to the `bgr_array` as a bytearray.
    cdef unsigned char [::1] temp = bytearray(gpu_array.get())

    memcpy(&bgr_array[0], &temp[0], w * h * bytesize)





grey_kernel = cp.ElementwiseKernel(
    'uint8 r, uint8 g, uint8 b',
    'uint8 rr, uint8 gg, uint8 bb',
    '''

    // ITU-R BT.601 luma coefficients
    float grey = (float)(r + g + b) / 3.0f ;  
    rr = (unsigned char)(grey); 
    gg = (unsigned char)(grey);
    bb = (unsigned char)(grey);   
   
    ''', 'grey_kernel'
)


grey_luminosity_kernel = cp.ElementwiseKernel(
    'uint8 r, uint8 g, uint8 b',
    'uint8 rr, uint8 gg, uint8 bb',
    '''

    // ITU-R BT.601 luma coefficients
    float luminosity = (unsigned char)(r * (float)0.2126 + g * (float)0.7152 + b * (float)0.072); 
    rr = (unsigned char)(luminosity); 
    gg = (unsigned char)(luminosity);
    bb = (unsigned char)(luminosity);   

    ''', 'grey_luminosity_kernel'
)


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef bpf_gpu(
        surface_,
        unsigned char threshold_ = 128
):
    """
    Apply a Bright Pass Filter (BPF) effect to a pygame.Surface using GPU.

    The function applies a Bright Pass Filter (elementwise kernel) to an image represented by
    a pygame.Surface. The filter keeps pixels with brightness values above a specified threshold 
    and sets others to black. The image is processed on the GPU to improve performance.

    :param surface_   : pygame.Surface
        The input surface (image) on which the Bright Pass Filter effect will be applied.

    :param threshold_ : integer (default: 128)
        The brightness threshold in the range [0...255]. Pixels with a brightness value 
        greater than or equal to the threshold will retain their color, while those below 
        will be set to black. The default threshold is 128.

    :return          : pygame.Surface
        A new pygame.Surface with the Bright Pass Filter effect applied.
        The result will be in RGB format, with the same width and height as the input surface.

    :raises TypeError : If `surface_` is not a pygame.Surface type.
    :raises ValueError : If there is an issue referencing the surface pixels into a 3D array,
                         or if the surface dimensions are invalid.
    """

    # Validate that the input is a pygame.Surface.
    if not PyObject_IsInstance(surface_, pygame.Surface):
        raise TypeError(
            "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_))

    # Initialize a 3D numpy array to hold the pixel data of the surface.
    cdef np.ndarray[ np.uint8_t, ndim=3 ] cpu_array_

    try:
        # Attempt to extract the surface pixel data as a 3D array (height, width, channels).
        cpu_array_ = pixels3d(surface_)  # Equivalent to surface_.get_view('3')
    except Exception as e:
        # Raise an error if the surface cannot be converted to a 3D array.
        raise ValueError(
            "\nCannot reference `surface_` pixels into a 3D array.\n %s " % e)

    # Extract the width and height of the image from the shape of the array.
    cdef Py_ssize_t w, h
    w, h = (<object> cpu_array_).shape[ :2 ]

    # Ensure that the surface dimensions are valid (non-zero width and height).
    if w <= 0 or h <= 0:
        raise ValueError("Surface width or height cannot be null!")

    # Convert the numpy array to a CuPy array for GPU processing.
    gpu_array = cp.asarray(cpu_array_)

    # Apply the Bright Pass Filter effect on the GPU. This function modifies the image based on the threshold.
    bpf_cupy(gpu_array, threshold_)

    # Transpose the image to change the order from (height, width, channels) to (width, height, channels)
    # so that it is compatible with the final return format.
    gpu_array = gpu_array.transpose(1, 0, 2)

    # Convert the GPU array back to a byte buffer and return a new pygame.Surface.
    return frombuffer(
        gpu_array.tobytes(),
        (w, h), "RGB"
    )




bpf_kernel = cp.ElementwiseKernel(
    'uint8 r, uint8 g, uint8 b, float32 threshold',
    'uint8 rr, uint8 gg, uint8 bb',
    '''

    // ITU-R BT.601 luma coefficients
    float lum = r * 0.299f + g * 0.587f + b * 0.114f;    
    if (lum > threshold) {
        float c = (float)(lum - threshold) / (lum+1.0f);
        rr = (unsigned char)(fmaxf(r * c, 0.0f));
        gg = (unsigned char)(fmaxf(g * c, 0.0f));
        bb = (unsigned char)(fmaxf(b * c, 0.0f));   
    } 
     else {
        rr = (unsigned char)0;
        gg = (unsigned char)0;
        bb = (unsigned char)0;                    
    }

    ''', 'bpf_kernel'
)



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef inline void bpf_cupy(gpu_array_, const unsigned char threshold_):
    """
    Apply a Bright Pass Filter (BPF) to a GPU array in-place.

    This function applies a Bright Pass Filter effect to an image represented by a 
    GPU array (using CuPy). The filter keeps pixels with a brightness value above 
    the specified threshold and sets other pixels to black. The operation is performed 
    element-wise and directly modifies the input GPU array.

    :param gpu_array_: cp.ndarray
        A 3D GPU array (shape: w, h, 3|4) representing the image in RGB (or RGBA) format.
        The array should be in the format of `uint8` with pixel values in the range [0, 255].
        The image will be modified in-place.

    :param threshold_: unsigned char (integer)
        The brightness threshold in the range [0, 255]. Pixels with brightness values 
        greater than or equal to the threshold will remain unchanged, while pixels with 
        values below the threshold will be set to black.

    :return: void
        The operation modifies the input `gpu_array_` in-place, so no value is returned.
    """

    # Apply the Bright Pass Filter kernel to each color channel (Red, Green, Blue)
    # The kernel checks the brightness value of each pixel and applies the filter accordingly.
    gpu_array_[ :, :, 0 ], \
    gpu_array_[ :, :, 1 ], \
    gpu_array_[ :, :, 2 ] = bpf_kernel(
        gpu_array_[ :, :, 0 ],  # Red channel
        gpu_array_[ :, :, 1 ],  # Green channel
        gpu_array_[ :, :, 2 ],  # Blue channel
        <float> threshold_  # Threshold value as a float
    )




@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef void bpf_inplace_gpu(
    tuple grid_,
    tuple block_,
    surface_,
    unsigned char threshold_ = 128
):
    """
    Apply a Bright Pass Filter (BPF) to a pygame.Surface in-place using GPU acceleration.

    This function applies a Bright Pass Filter effect to an image on the GPU. The filter retains
    pixels with brightness values greater than or equal to a specified threshold, setting others to black.
    The operation is done in-place, modifying the input surface directly.

    :param surface_   : pygame.Surface
        The input surface (image) on which the Bright Pass Filter effect will be applied. 
        The surface should be in RGB format.

    :param grid_      : tuple (grid_y, grid_x)
        Specifies the grid configuration for the GPU kernel. This determines how the work is divided 
        across multiple threads in the GPU. Grid dimensions should match the texture and array sizes.

    :param block_     : tuple (block_y, block_x)
        Specifies the block configuration for the GPU kernel. This defines the number of threads 
        within a block. The maximum number of threads per block is 1024. Block size should be chosen 
        to optimize performance on the hardware.

    :param threshold_ : unsigned char (default: 128)
        The brightness threshold (in the range [0, 255]) that determines which pixels are kept. 
        Pixels with a brightness value greater than or equal to the threshold remain unchanged, 
        while pixels below the threshold are set to black.

    :return          : void
        Transformation is applied in-place
        

    :raises TypeError : If `surface_` is not a pygame.Surface type.
    :raises ValueError : If there is an issue referencing the surface pixels into a 3D array, or 
                         if the surface dimensions are invalid.
    """

    # Validate that the input surface is of type pygame.Surface
    if not PyObject_IsInstance(surface_, pygame.Surface):
        raise TypeError("\nArgument surface_ must be a pygame.Surface, got %s " % type(surface_))

    # Attempt to extract the pixel data from the surface as a 3D numpy array (height, width, channels)
    try:
        rgb_array = pixels3d(surface_)  # Get the surface pixels as a 3D array
    except Exception as e:
        # Raise an error if surface pixels cannot be referenced as a 3D array
        raise ValueError("\nCannot reference `surface_` pixels into a 3D array.\n %s " % e)

    # Extract the width and height from the shape of the rgb_array
    cdef Py_ssize_t w, h
    w, h = (<object> rgb_array).shape[ :2 ]

    # Ensure the surface has valid dimensions (non-zero width and height)
    if w <= 0 or h <= 0:
        raise ValueError("Surface width or height cannot be null!")

    # Convert the RGB numpy array to a CuPy GPU array for efficient GPU processing
    gpu_array = cp.asarray(rgb_array)

    # Apply the Bright Pass Filter (BPF) on the GPU using the specified grid and block configuration
    # The BPF operation is done element-wise across the image with GPU acceleration
    bpf_inplace_cupy(grid_, block_, w, h, gpu_array, threshold_)

    # Copy the processed image back to the original `rgb_array` in-place
    # This updates the surface's pixel values with the modified GPU array.
    rgb_array[ :, :, 0 ] = gpu_array[ :, :, 0 ].get()  # Red channel
    rgb_array[ :, :, 1 ] = gpu_array[ :, :, 1 ].get()  # Green channel
    rgb_array[ :, :, 2 ] = gpu_array[ :, :, 2 ].get()  # Blue channel



bpf_rgb_kernel1 = cp.RawKernel(
    r'''
    extern "C" __global__
    
    // Kernel function for applying a Bright Pass Filter (BPF) effect
    // This kernel processes an image in parallel using CUDA threads on the GPU.
    void bpf_rgb_kernel1(
        unsigned char * rgb_array,
        const int w, 
        const int h, 
        const double threshold,
        const short int bytesize = 3
        )
    {
        // Calculate the 2D grid index for the current thread
        int i = blockIdx.x * blockDim.x + threadIdx.x;  // X-axis index
        int j = blockIdx.y * blockDim.y + threadIdx.y;  // Y-axis index
    
        // The total number of pixels in the image (width * height)
        const int t_max  = w * h;    
        // Calculate the index for accessing the 1D linear array representation of the image
        const int index  = j * h + i;  // Index for the pixel in the 1D array
        // Calculate the index for accessing the RGB components of the pixel (each pixel has 3 values: R, G, B)
        const int index1 = j * h * bytesize + i * bytesize;
    
        // Ensure all threads in the block synchronize before proceeding
        __syncthreads();
    
        // Check if the current thread index is within the valid image bounds
        if (index > 0 && index < t_max) {            
    
            // Retrieve the RGB values for the current pixel
            float r  = (float)rgb_array[index1    ];  // Red component
            float g  = (float)rgb_array[index1 + 1];  // Green component
            float b  = (float)rgb_array[index1 + 2];  // Blue component
    
            // Calculate the luminance (brightness) of the pixel using ITU-R BT.601 coefficients
            // These coefficients are used to convert RGB to grayscale (luma) values.
            float lum = r * 0.299f + g * 0.587f + b * 0.114f;    
    
            // If the pixel's luminance is greater than the threshold, retain the pixel with reduced intensity
            if (lum > threshold) {
                // Apply a contrast adjustment based on the difference from the threshold
                float c = (float)((lum - threshold) / (lum + 1));
    
                // Apply the contrast factor `c` to each of the RGB components of the pixel
                // This brightens the pixels with high luminance and darkens the ones with low luminance
                rgb_array[index1    ] = (unsigned char)(r * c);  // Adjust red
                rgb_array[index1 + 1] = (unsigned char)(g * c);  // Adjust green
                rgb_array[index1 + 2] = (unsigned char)(b * c);  // Adjust blue
            } 
            else {
                // If the luminance is below the threshold, set the pixel to black (0, 0, 0)
                rgb_array[index1    ] = (unsigned char)0;  // Set red to 0
                rgb_array[index1 + 1] = (unsigned char)0;  // Set green to 0
                rgb_array[index1 + 2] = (unsigned char)0;  // Set blue to 0
            }
    
            // Ensure all threads in the block synchronize before completing the kernel execution
            __syncthreads();
        }    
    }
    ''',
    'bpf_rgb_kernel1'
)

bpf_bgr_kernel1 = cp.RawKernel(
    r'''
    extern "C" __global__

    // Kernel function for applying a Bright Pass Filter (BPF) effect
    // This kernel processes an image in parallel using CUDA threads on the GPU.
    void bpf_bgr_kernel1(
        unsigned char * bgr_array,
        const int w, 
        const int h, 
        const double threshold,
        const short int bytesize = 3
        )
    {
        // Calculate the 2D grid index for the current thread
        int i = blockIdx.x * blockDim.x + threadIdx.x;  // X-axis index
        int j = blockIdx.y * blockDim.y + threadIdx.y;  // Y-axis index

        // The total number of pixels in the image (width * height)
        const int t_max  = w * h;    
        // Calculate the index for accessing the 1D linear array representation of the image
        const int index  = j * h + i;  // Index for the pixel in the 1D array
        // Calculate the index for accessing the BGR components of the pixel (each pixel has 3 values: R, G, B)
        const int index1 = j * h * bytesize + i * bytesize;

        // Ensure all threads in the block synchronize before proceeding
        __syncthreads();

        // Check if the current thread index is within the valid image bounds
        if (index > 0 && index < t_max) {            

            // Retrieve the BGR values for the current pixel
            float r  = (float)bgr_array[index1 + 2];  // Red component
            float g  = (float)bgr_array[index1 + 1];  // Green component
            float b  = (float)bgr_array[index1    ];  // Blue component

            // Calculate the luminance (brightness) of the pixel using ITU-R BT.601 coefficients
            // These coefficients are used to convert BGR to grayscale (luma) values.
            float lum = r * 0.299f + g * 0.587f + b * 0.114f;    

            // If the pixel's luminance is greater than the threshold, retain the pixel with reduced intensity
            if (lum > threshold) {
                // Apply a contrast adjustment based on the difference from the threshold
                float c = (float)((lum - threshold) / (lum + 1));

                // Apply the contrast factor `c` to each of the BGR components of the pixel
                // This brightens the pixels with high luminance and darkens the ones with low luminance
                bgr_array[index1    ] = (unsigned char)(b * c); 
                bgr_array[index1 + 1] = (unsigned char)(g * c);  
                bgr_array[index1 + 2] = (unsigned char)(r * c);  
            } 
            else {
                // If the luminance is below the threshold, set the pixel to black (0, 0, 0)
                bgr_array[index1    ] = (unsigned char)0;  
                bgr_array[index1 + 1] = (unsigned char)0; 
                bgr_array[index1 + 2] = (unsigned char)0;  
            }

            // Ensure all threads in the block synchronize before completing the kernel execution
            __syncthreads();
        }    
    }
    ''',
    'bpf_bgr_kernel1'
)



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef inline void bpf_inplace_cupy(
    tuple grid_,
    tuple block_,
    const unsigned int w,
    const unsigned int h,
    gpu_array_,
    unsigned char threshold_
):
    """
    Apply Bright Pass Filter (BPF) effect in-place on the GPU using a CUDA kernel.

    This function processes an image using a CUDA kernel to apply a bright pass filter (BPF) effect.
    It modifies the provided GPU array (in-place) based on a given luminance threshold. 
    Pixels with a luminance greater than the threshold are adjusted by a contrast factor, 
    while pixels with lower luminance are turned black.

    :param grid_        : tuple; 
        Grid size (grid_y, grid_x) defining the number of blocks in the grid.
        Example: (25, 25). The grid size should match the image dimensions.
        
    :param block_       : tuple; 
        Block size (block_y, block_x) defining the number of threads in a block.
        Example: (32, 32). Maximum threads per block is 1024.
        
    :param w            : unsigned int; 
        Width of the image.
        
    :param h            : unsigned int; 
        Height of the image.
        
    :param gpu_array_   : cp.ndarray; 
        GPU array with shape (w, h, 3|4) containing RGB or RGBA pixel data.
        This array will be modified in place.
        
    :param threshold_   : unsigned char; 
        Threshold value in range [0, 255], used to determine which pixels
        undergo the bright pass filter effect.

    :return            : void; 
        The operation is done in-place on the provided GPU array.
    """

    # Apply the BPF kernel using the specified grid and block configuration
    # The kernel processes the image array in-place using the provided threshold value.
    bpf_rgb_kernel1(
        grid_,
        block_,
        (gpu_array_, w, h, <float>threshold_)
    )

    # Synchronize the CUDA stream to ensure all operations are completed before proceeding.
    cp.cuda.Stream.null.synchronize()




@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef bpf_buffer_gpu(
        tuple grid,
        tuple block,
        const Py_ssize_t w,
        const Py_ssize_t h,
        unsigned char [::1] bgr_array,
        unsigned char threshold_ = 128,
        bint format_32 = False
):
    """
    Applies a band-pass filter (BPF) to an image buffer using GPU acceleration.

    Parameters:
    -----------
    grid : tuple
        Grid size for GPU kernel execution (blocks per grid).
        
    block : tuple
        Block size for GPU kernel execution (threads per block).
        
    w : Py_ssize_t
        Width of the image in pixels.
        
    h : Py_ssize_t
        Height of the image in pixels.
        
    bgr_array : memoryview or numpy.ndarray
        Input image buffer in BGR or BGRA format.
        
    threshold_ : unsigned char, optional
        Threshold value used for filtering (default: 128).
        
    format_32 : bool, optional
        If True, treats the input buffer as BGRA (4 bytes per pixel).
        If False, treats it as BGR (3 bytes per pixel).

    Raises:
    -------
    TypeError:
        If `bgr_array` is not a numpy.ndarray or memoryview.
    ValueError:
        If `bgr_array` is empty or does not match the expected size.

    Returns:
    --------
    None
        The function modifies `bgr_array` in-place with the filtered image.
    """

    # Ensure bgr_array is a valid numpy array or memoryview.
    if not (PyObject_IsInstance(bgr_array, numpy.ndarray) or is_type_memoryview(bgr_array)):
        raise TypeError(
            "\nArgument bgr_array must be a numpy.ndarray or a memoryview, "
            "but got %s." % type(bgr_array)
        )

    cdef:
        unsigned short int bytesize = 4 if format_32 else 3  # Determine bytes per pixel.
        Py_ssize_t length = len(<object> bgr_array)  # Get the buffer length.

    # Validate the input buffer size.
    if length == 0:
        raise ValueError("Argument `bgr_array` buffer cannot be empty.")

    expected_length = w * h * bytesize
    if length != expected_length:
        raise ValueError(
            f"Expecting bgr_array length of {expected_length}, but got {length}. "
            f"Use `format_32=True` for BGRA buffers and `format_32=False` for BGR buffers."
        )

    # Convert the numpy array to a CuPy array for GPU processing.
    gpu_array = cp.asarray(bgr_array)

    # Apply the BPF kernel on the GPU.
    # The kernel processes the image array in-place, using the given threshold.
    bpf_bgr_kernel1(
        grid,
        block,
        (gpu_array, w, h, <float> threshold_, bytesize)
    )

    # Copy the processed GPU buffer back to the original bgr_array.
    cdef unsigned char [::1] arr = bytearray(gpu_array.get())

    memcpy(&bgr_array[0], &arr[0], length)




@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef grayscale_gpu(surface_):
    """
    Converts a Pygame surface to grayscale using GPU acceleration.

    This function takes a Pygame surface in 24-bit or 32-bit format, processes
    it on the GPU to convert it to grayscale, and returns a new 24-bit grayscale surface.

    Parameters:
    -----------
    surface_ : pygame.Surface
        A Pygame surface with a 24-bit or 32-bit color format.

    Raises:
    -------
    TypeError:
        If `surface_` is not a Pygame surface.
    ValueError:
        If the surface cannot be converted into a 3D array view.
        If the surface width or height is zero.

    Returns:
    --------
    pygame.Surface
        A new 24-bit grayscale Pygame surface.
    """

    # Ensure the input is a valid Pygame surface.
    if not PyObject_IsInstance(surface_, pygame.Surface):
        raise TypeError(
            "\nArgument `surface_` must be a pygame.Surface, but got %s." % type(surface_)
        )

    cdef unsigned char [:, :, :] rgb_array

    # Try to get a 3D view of the surface's pixel data.
    try:
        rgb_array = surface_.get_view('3')  # Extract RGB channels as a 3D array.

    except Exception as e:
        raise ValueError(
            "\nCannot reference source pixels into a 3D array.\n%s" % e
        )

    cdef Py_ssize_t w, h

    # Extract width and height from the RGB array.
    w, h = (<object>rgb_array).shape[:2]

    # Validate surface dimensions.
    if w <= 0 or h <= 0:
        raise ValueError("Surface width or height cannot be zero!")

    # Convert the pixel array to a CuPy array for GPU processing.
    gpu_array = cp.asarray(rgb_array)

    # Apply grayscale conversion using the GPU.
    grayscale_cupy(gpu_array)

    # Convert the processed array back to a 24-bit grayscale Pygame surface.
    return frombuffer(
        gpu_array.astype(dtype=cp.uint8).transpose(1, 0, 2).tobytes(),
        (w, h),
        "RGB"
    )



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef inline void grayscale_cupy(gpu_array_):
    """
    Applies a grayscale transformation to an image array on the GPU.

    This function modifies the input CuPy array in-place, converting its RGB
    channels to grayscale using a GPU-accelerated kernel.

    Parameters:
    -----------
    gpu_array_ : cp.ndarray
        A CuPy 3D array representing an image with shape (height, width, 3).
        The array must contain RGB pixel values in the range [0, 255].

    Returns:
    --------
    None
        The function modifies `gpu_array_` in-place, replacing its RGB values
        with grayscale values.
    """

    # Apply the grayscale transformation using a GPU kernel.
    # The grey_kernel function takes the individual R, G, and B channels,
    # processes them, and returns new grayscale-adjusted channel values.
    gpu_array_[:, :, 0], \
    gpu_array_[:, :, 1], \
    gpu_array_[:, :, 2] = \
        grey_kernel(gpu_array_[:, :, 0], gpu_array_[:, :, 1], gpu_array_[:, :, 2])

    # Ensure all GPU operations are completed before returning.
    cp.cuda.Stream.null.synchronize()







@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef grayscale_lum_gpu(surface_):
    """
    Converts a Pygame surface to grayscale using GPU acceleration with a luminance-based method.

    This function takes a Pygame surface in 24-bit or 32-bit format, processes
    it on the GPU using a luminance-based grayscale conversion, and returns
    a new 24-bit grayscale Pygame surface.

    Parameters:
    -----------
    surface_ : pygame.Surface
        A Pygame surface with a 24-bit or 32-bit color format.

    Raises:
    -------
    TypeError:
        If `surface_` is not a Pygame surface.
    ValueError:
        If the surface cannot be converted into a 3D array view.
        If the surface width or height is zero.

    Returns:
    --------
    pygame.Surface
        A new 24-bit grayscale Pygame surface.
    """

    # Ensure the input is a valid Pygame surface.
    if not PyObject_IsInstance(surface_, pygame.Surface):
        raise TypeError(
            "\nArgument `surface_` must be a pygame.Surface, but got %s." % type(surface_)
        )

    cdef unsigned char [:, :, :] rgb_array

    # Attempt to get a 3D view of the surface's pixel data (RGB format).
    try:
        rgb_array = surface_.get_view('3')  # Extract RGB channels as a 3D array.

    except Exception as e:
        raise ValueError(
            "\nCannot reference `surface_` pixels into a 3D array.\n%s" % e
        )

    cdef Py_ssize_t w, h

    # Extract width and height from the RGB array.
    w, h = (<object>rgb_array).shape[:2]

    # Validate surface dimensions.
    if w <= 0 or h <= 0:
        raise ValueError("Surface width and height cannot be zero!")

    # Convert the pixel array to a CuPy array for GPU processing.
    gpu_array = cp.asarray(rgb_array)

    # Apply luminance-based grayscale conversion using the GPU.
    grayscale_lum_cupy(gpu_array)

    # Convert the processed GPU array back to a 24-bit grayscale Pygame surface.
    return frombuffer(
        gpu_array.astype(dtype=cp.uint8).transpose(1, 0, 2).tobytes(),
        (w, h),
        "RGB"
    )


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef inline void grayscale_lum_cupy(gpu_array_):
    """
    Applies a luminance-based grayscale transformation to an image array on the GPU.

    This function modifies the input CuPy array in-place, converting its RGB
    channels to grayscale using a GPU-accelerated luminosity method.

    The luminosity method calculates grayscale values using a weighted sum
    of the RGB channels, giving more importance to green, as it contributes
    more to perceived brightness.

    Parameters:
    -----------
    gpu_array_ : cp.ndarray
        A CuPy 3D array representing an image with shape (height, width, 3).
        The array must contain RGB pixel values in the range [0, 255].

    Returns:
    --------
    None
        The function modifies `gpu_array_` in-place, replacing its RGB values
        with grayscale values based on luminance.
    """

    # Apply the luminance-based grayscale transformation using a GPU kernel.
    # The `grey_luminosity_kernel` function computes the grayscale value
    # for each pixel using a weighted sum of the R, G, and B channels.
    gpu_array_[:, :, 0], \
    gpu_array_[:, :, 1], \
    gpu_array_[:, :, 2] = \
        grey_luminosity_kernel(
            gpu_array_[:, :, 0],  # Red channel
            gpu_array_[:, :, 1],  # Green channel
            gpu_array_[:, :, 2])


# todo cpdef grayscale_lum_gpu(unsigned char [::1] buffer):




median_kernel = cp.RawKernel(

    '''extern "C" __global__ void median_kernel(
        double* buffer, 
        const int filter_size,
        double* return_value
        )
    {
    
    int i, j;
    
    double temp = 0;
    for (i = 0; i < (filter_size - 1); ++i)
    {
        for (j = 0; j < filter_size - 1 - i; ++j )
        {
            if (buffer[j] > buffer[j+1])
            {
                temp = buffer[j+1];
                buffer[j+1] = buffer[j];
                buffer[j] = temp;
            }
        }
    }
     
    return_value[0] = (double)buffer[int(filter_size/2.0f)];
    }
    ''',
    'median_kernel'
)




@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef median_generic_filter_gpu(surface_, unsigned int size_ = 5):
    """
    Applies a median filter to a Pygame surface using GPU acceleration.

    This function utilizes a generic median filter, which processes each
    pixel based on neighboring values. The filter operates on 24-bit and
    32-bit surfaces by applying a kernel function.

    **Performance Consideration:**
    The median filter uses a kernel with a buffer of type `double`, which
    may result in lower performance compared to specialized GPU-based median
    filtering methods.

    Example of a compatible median filter kernel:
    ```c
    void median_kernel(double* buffer, int filter_size, double* return_value)
    ```

    Parameters:
    -----------
    surface_ : pygame.Surface
        The input Pygame surface, expected to be in 24-bit or 32-bit format.

    size_ : int, optional (default=5)
        The size of the neighborhood window used for the median filter.
        Must be greater than zero.

    Returns:
    --------
    pygame.Surface
        A new Pygame surface with the median filter applied.

    Raises:
    -------
    ValueError:
        - If `size_` is less than or equal to zero.
        - If the input surface has zero width or height.

    TypeError:
        - If `surface_` is not a valid `pygame.Surface`.

    Notes:
    ------
    - This function leverages CuPy (`cp.asarray`) for GPU-accelerated computation.
    - The filter operates on each color channel (R, G, B) separately.
    """

    # Ensure the kernel size is valid.
    if size_ <= 0:
        raise ValueError("Argument `size_` must be greater than 0.")

    # Validate the input surface type.
    if not PyObject_IsInstance(surface_, pygame.Surface):
        raise TypeError(f"Argument `surface_` must be a pygame.Surface, got {type(surface_)}.")

    cdef unsigned char [:, :, :] rgb_array

    # Attempt to extract the surface pixel data as a 3D RGB array.
    try:
        rgb_array = surface_.get_view('3')
    except Exception as e:
        raise ValueError(f"Cannot reference `surface_` pixels into a 3D array.\n{e}")

    # Get the width (w) and height (h) of the image.
    cdef Py_ssize_t w, h
    w, h = (<object>rgb_array).shape[:2]

    # Ensure the surface dimensions are valid.
    if w <= 0 or h <= 0:
        raise ValueError("Surface width or height cannot be zero!")

    # Convert the CPU-based numpy array to a GPU-compatible CuPy array.
    gpu_array = cp.asarray(rgb_array)

    # Apply the median filter on the GPU.
    median_generic_filter_cupy(gpu_array, size_)

    # Convert the processed GPU array back to a Pygame surface.
    return frombuffer(
        gpu_array.astype(cp.uint8).transpose(1, 0, 2).tobytes(),
        (w, h),
        "RGB"
    )



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef inline void median_generic_filter_cupy(gpu_array_, unsigned int size_=5):
    """
    Applies a median filter to an image using GPU-accelerated computation.

    This function performs median filtering on each color channel (R, G, B)
    of an image using CuPy's `generic_filter` function

    **Performance Consideration:**
    - `generic_filter` applies a custom median kernel directly, making it
      more efficient than standard median filtering for large images.
    - Each color channel is processed independently.

    Parameters:
    -----------
    gpu_array_ : cupy.ndarray
        A 3D CuPy array representing an image in RGB format.

    size_ : unsigned int, optional (default=5)
        The size of the kernel window used for the median filter.
        Must be greater than zero.

    Returns:
    --------
    None
        The function modifies `gpu_array_` in place.

    Notes:
    ------
    - The median filter replaces each pixel value with the median of
      neighboring pixels, reducing noise while preserving edges.
    - The operation is performed separately on the red, green, and blue channels.
    - `cp.cuda.Stream.null.synchronize()` ensures GPU operations are completed.
    """

    # Ensure the kernel size is valid
    if size_ <= 0:
        raise ValueError("Kernel size (`size_`) must be greater than 0.")

    r = cupyx.scipy.ndimage.generic_filter(
        gpu_array_[ :, :, 0 ], median_kernel, size_).astype(dtype = cp.uint8)  # Red channel
    g = cupyx.scipy.ndimage.generic_filter(
        gpu_array_[ :, :, 1 ], median_kernel, size_).astype(dtype = cp.uint8)  # Green channel
    b = cupyx.scipy.ndimage.generic_filter(
        gpu_array_[ :, :, 2 ], median_kernel, size_).astype(dtype = cp.uint8)  # Blue channel

    # Assign the filtered values back to the original array
    gpu_array_[ :, :, 0 ], gpu_array_[ :, :, 1 ], gpu_array_[ :, :, 2 ] = r, g, b

    # Ensure all CUDA operations are completed before continuing
    cp.cuda.Stream.null.synchronize()



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef median_filter_gpu(
        surface_,
        unsigned int size_ = 5
):
    """
    Applies a median filter to a Pygame surface using GPU acceleration.

    The median filter is a non-linear filter that reduces noise by replacing 
    each pixel with the median value of its surrounding neighbors.

    Compatible with 24-bit and 32-bit Pygame surfaces.

    Parameters:
    -----------
    surface_ : pygame.Surface
        The input Pygame surface to be processed.

    size_ : int, optional (default=5)
        The size of the neighborhood considered for median filtering.
        Must be greater than zero.

    Raises:
    -------
    AssertionError:
        If `size_` is not greater than zero.
        If `surface_` is not a valid Pygame surface.

    ValueError:
        If the surface cannot be referenced as a 3D array.
        If the surface width or height is zero.

    Returns:
    --------
    pygame.Surface
        A new Pygame surface with the median filter applied.
    """

    # Ensure the neighborhood size is valid.
    if size_ <= 0:
        raise ValueError(f"Argument `size_` must be > 0")

    # Ensure the input is a valid Pygame surface.
    if not PyObject_IsInstance(surface_, pygame.Surface):
        raise TypeError(f"Argument `surface_` must be a "
                        f"pygame.Surface, but got {type(surface_)}.")

    cdef unsigned char [:, :, :] rgb_array

    # Try to get a 3D view of the surface's pixel data.
    try:
        rgb_array = surface_.get_view('3')  # Extract RGB channels as a 3D array.

    except Exception as e:
        raise ValueError("\nCannot reference `surface_` pixels into a 3D array.\n%s" % e)

    cdef Py_ssize_t w, h

    # Extract width and height from the RGB array.
    w, h = (<object>rgb_array).shape[:2]

    # Validate surface dimensions.
    if w <= 0 or h <= 0:
        raise ValueError("Surface width or height cannot be zero!")

    # Convert the pixel array to a CuPy array for GPU processing.
    gpu_array = cp.asarray(rgb_array)

    # Apply the median filter on the GPU.
    median_filter_cupy(gpu_array, size_)

    # Convert the processed GPU array back to a 24-bit Pygame surface.
    return frombuffer(
        gpu_array.astype(cp.uint8).transpose(1, 0, 2).tobytes(),
        (w, h),
        "RGB"
    )




@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef inline void median_filter_cupy(gpu_array_, unsigned int size_=5):
    """
    Applies a median filter to an image array using GPU acceleration.

    This function processes each color channel (R, G, B) independently using
    CuPy's GPU-accelerated `median_filter` from `cupyx.scipy.ndimage`.
    The median filter is effective at reducing noise while preserving edges.

    Parameters:
    -----------
    gpu_array_ : cp.ndarray
        A CuPy 3D array representing an image with shape (height, width, 3).
        The array must contain RGB pixel values in the range [0, 255].

    size_ : int, optional (default=5)
        The size of the neighborhood window used to compute the median.
        Must be greater than zero.

    Returns:
    --------
    None
        The function modifies `gpu_array_` in-place, replacing each pixel with 
        the median value of its surrounding neighbors.

    Notes:
    ------
    - The median filter is applied separately to each color channel.
    - Synchronization ensures GPU computations are completed before returning.
    """

    # Apply the median filter to each color channel (R, G, B) independently.
    gpu_array_[:, :, 0] = cupyx.scipy.ndimage.median_filter(input=gpu_array_[:, :, 0], size=size_)  # Red channel
    gpu_array_[:, :, 1] = cupyx.scipy.ndimage.median_filter(input=gpu_array_[:, :, 1], size=size_)  # Green channel
    gpu_array_[:, :, 2] = cupyx.scipy.ndimage.median_filter(input=gpu_array_[:, :, 2], size=size_)  # Blue channel

    # Ensure all GPU operations complete before continuing execution.
    cp.cuda.Stream.null.synchronize()



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef gaussian_5x5_gpu(surface_):
    """
    Applies a 5x5 Gaussian blur to an image using GPU-accelerated processing.

    This function convolves each RGB channel of the given surface with a 
    5x5 Gaussian blur kernel to create a smoothing effect while preserving edges.

    **Performance Consideration:**
    - Processing is done on the GPU using CuPy for fast execution.
    - Each color channel (R, G, B) is processed separately.

    Parameters:
    -----------
    surface_ : pygame.Surface
        A 24-bit or 32-bit Pygame surface to be blurred.

    Returns:
    --------
    pygame.Surface
        A new Pygame surface with the applied Gaussian blur format 24-bit.

    Raises:
    -------
    TypeError:
        If `surface_` is not a valid Pygame surface.

    ValueError:
        If the surface dimensions are invalid or pixel data cannot be accessed.

    Notes:
    ------
    - The function uses `pixels3d(surface_)` to extract the pixel array.
    - The computation is performed on the GPU for efficiency.
    - `gaussian_5x5_cupy` is called to apply the blur effect.
    """

    # Ensure the input is a valid Pygame surface
    if not PyObject_IsInstance(surface_, pygame.Surface):
        raise TypeError(f"Argument `surface_` must be a pygame.Surface, got {type(surface_)}.")

    # Declare a NumPy array with unsigned 8-bit integer type for pixel data
    cdef np.ndarray[ np.uint8_t, ndim=3 ] rgb_array

    try:
        # Extract the pixel data from the surface into an RGB array
        rgb_array = pixels3d(surface_)
    except Exception as e:
        raise ValueError(f"Cannot reference `surface_` pixels into a 3D array.\n{e}")

    cdef Py_ssize_t w, h  # Declare width and height variables

    # Get the width and height of the surface
    w, h = (<object> rgb_array).shape[ :2 ]

    # Validate the surface dimensions
    if w <= 0 or h <= 0:
        raise ValueError("Surface width or height cannot be null!")

    # Convert the RGB array to a CuPy array for GPU-based processing
    gpu_array = cp.asarray(rgb_array, order = 'C')

    # Apply the Gaussian blur using a 5x5 kernel
    gaussian_5x5_cupy(gpu_array)

    # Convert the processed CuPy array back to a Pygame surface and return
    return frombuffer(
        gpu_array.astype(cp.uint8).transpose(1, 0, 2).tobytes(),
        (w, h),
        "RGB"
    )

# Gaussian kernel 5x5
gaussian_kernel_5x5 = cp.array([[1,   4,   6,   4,  1],
    [4,  16,  24,  16,  4],
    [6,  24,  36,  24,  6],
    [4,  16,  24,  16,  4],
    [1,  4,    6,   4,  1]], dtype=cp.float32) * <float>1.0/<float>256.0

@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef inline void gaussian_5x5_cupy(gpu_array_):
    """
    Applies a 5x5 Gaussian blur to an image using GPU acceleration with CuPy.

    This function convolves each color channel (R, G, B) of the given image 
    with a predefined 5x5 Gaussian kernel. The operation is performed 
    using `cupyx.scipy.ndimage.convolve` for efficient GPU-based processing.

    **Processing Details:**
    - The Gaussian blur smooths the image while preserving edges.
    - The convolution is applied separately to each channel.
    - `mode='constant', cval=0.0` ensures out-of-bound pixels are treated as zero.

    Parameters:
    -----------
    gpu_array_ : cupy.ndarray
        A CuPy GPU array representing an image with shape `(height, width, 3)`.
        Each channel (Red, Green, Blue) is processed separately.

    Returns:
    --------
    None
        The input `gpu_array_` is modified in-place.

    Notes:
    ------
    - This function assumes `gaussian_kernel_5x5` is a predefined 5x5 kernel.
    - `cp.cuda.Stream.null.synchronize()` ensures all GPU operations finish before returning.
    """

    cdef:
        Py_ssize_t w, h

    # Extract image dimensions
    w, h = gpu_array_.shape[0], gpu_array_.shape[1]

    # Get the convolution function from CuPy's scipy module
    conv = cupyx.scipy.ndimage.convolve

    # Apply the Gaussian blur separately to each color channel
    gpu_array_[:, :, 0] = conv(
        gpu_array_[:, :, 0], gaussian_kernel_5x5, mode='constant', cval=0.0)
    gpu_array_[:, :, 1] = conv(
        gpu_array_[:, :, 1], gaussian_kernel_5x5, mode='constant', cval=0.0)
    gpu_array_[:, :, 2] = conv(
        gpu_array_[:, :, 2], gaussian_kernel_5x5, mode='constant', cval=0.0)

    # Ensure all GPU operations complete before returning
    cp.cuda.Stream.null.synchronize()




@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef gaussian_3x3_gpu(surface_):
    """
    Applies a 3x3 Gaussian blur to a pygame surface using GPU acceleration.

    This function converts a given `pygame.Surface` to a GPU-compatible array, 
    applies a 3x3 Gaussian blur filter to each color channel (R, G, B), and 
    returns a new blurred `pygame.Surface`.

    **Processing Details:**
    - Uses `cupyx.scipy.ndimage.convolve` to apply a predefined Gaussian kernel.
    - The convolution is applied separately to each RGB channel.
    - The boundary mode is set to `'constant'` with `cval=0.0` to handle edges.

    Parameters:
    -----------
    surface_ : pygame.Surface
        A 24-bit or 32-bit `pygame.Surface` representing the input image.

    Returns:
    --------
    pygame.Surface
        A new `pygame.Surface` with the Gaussian blur effect applied.

    Raises:
    -------
    TypeError:
        If `surface_` is not a `pygame.Surface`.
    ValueError:
        If the surface has invalid dimensions or cannot be converted into an array.

    Notes:
    ------
    - The function assumes that `gaussian_kernel_3x3` is predefined.
    - `cp.cuda.Stream.null.synchronize()` ensures the GPU operations complete before returning.
    """

    if not PyObject_IsInstance(surface_, pygame.Surface):
        raise TypeError(f"Argument surface_ must be a pygame.Surface, got {type(surface_)}.")

    cdef unsigned char [:, :, :] rgb_array
    try:
        # Retrieve pixel data as a 3D array (H, W, RGB)
        rgb_array = surface_.get_view('3')
    except Exception as e:
        raise ValueError(f"\nCannot reference source pixels into a 3D array.\n{e}")

    cdef:
        Py_ssize_t w, h

    # Get surface dimensions
    w, h = (<object>rgb_array).shape[:2]

    if w <= 0 or h <= 0:
        raise ValueError("Surface width or height cannot be zero or negative!")

    # Transfer data to GPU (CuPy array)
    gpu_array = cp.asarray(rgb_array)

    # Apply the Gaussian blur
    gaussian_3x3_cupy(gpu_array)

    # Convert the processed GPU array back to a Pygame surface
    return pygame.image.frombuffer(
        gpu_array.astype(cp.uint8).transpose(1, 0, 2).tobytes(),
        (w, h),
        "RGB")



# Gaussian kernel 3x3
gaussian_kernel_3x3 = cp.array([[1, 2, 1 ],
              [2, 4, 2],
              [1, 2, 1]], dtype=cp.float32) * <float>1.0 / <float>16.0


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef inline void gaussian_3x3_cupy(gpu_array_):
    """
    Applies a 3x3 Gaussian blur to an image stored in a CuPy GPU array.

    **Processing Details:**
    - The function uses `cupyx.scipy.ndimage.convolve` to apply a Gaussian filter 
      to each color channel (R, G, B) separately.
    - The Gaussian kernel is predefined as `gaussian_kernel_3x3`.
    - The operation is performed on the GPU for faster execution.

    Parameters:
    -----------
    gpu_array_ : cupy.ndarray
        A CuPy GPU array representing an image with shape `(height, width, 3)`.
        Each channel (Red, Green, Blue) is processed independently.

    Returns:
    --------
    None
        The input `gpu_array_` is modified in-place.

    Notes:
    ------
    - Uses `'constant'` mode with `cval=0.0` to handle image boundaries.
    - `cp.cuda.Stream.null.synchronize()` ensures all GPU computations complete before returning.
    """

    cdef:
        Py_ssize_t w, h

    # Get image dimensions
    w, h = gpu_array_.shape[:2]

    # Extract separate R, G, and B channels
    r = gpu_array_[:, :, 0]
    g = gpu_array_[:, :, 1]
    b = gpu_array_[:, :, 2]

    # Apply Gaussian blur separately to each channel
    gpu_array_[:, :, 0] = cupyx.scipy.ndimage.convolve(r, gaussian_kernel_3x3, mode='constant', cval=0.0)
    gpu_array_[:, :, 1] = cupyx.scipy.ndimage.convolve(g, gaussian_kernel_3x3, mode='constant', cval=0.0)
    gpu_array_[:, :, 2] = cupyx.scipy.ndimage.convolve(b, gaussian_kernel_3x3, mode='constant', cval=0.0)

    # Ensure all GPU operations finish before proceeding
    cp.cuda.Stream.null.synchronize()





sobel_kernel = cp.RawKernel(
    '''   
    extern "C" __global__
    
    __constant__ double gx[9] = {1.0, 2.0, 1.0, 0.0, 0.0, 0.0, -1.0, -2.0, -1.0};
    __constant__ double gy[9] = {1.0, 0.0, -1.0, 2.0, 0.0, -2.0, 1.0, 0.0, -1.0};   
    
    void sobel_kernel(double* buffer, const int filter_size,
                     double* return_value)
    {
    double s_x=0;
    double s_y=0;
    double magnitude=0;
    const double threshold = 15.0;
      
    for (int i=0; i<filter_size; ++i){
        s_x += buffer[i] * gx[i];
        s_y += buffer[i] * gy[i];
    }
  
    magnitude = sqrtf(s_x * s_x + s_y * s_y);
    
    if (magnitude > 255.0f) {
        magnitude = 255.0f;
    } 
    if (magnitude > threshold){
        return_value[0] = magnitude;
    } else {
        return_value[0] = (double)0.0;
    }
    
          
    }
    ''', 'sobel_kernel'
)




prewitt_kernel = cp.RawKernel(
    '''extern "C" 
    
    __constant__ double gx[9] = {1.0, 0.0, -1.0, 1.0, 0.0, -1.0, 1.0, 0.0, -1.0};
    __constant__ double gy[9] = {1.0, 1.0, 1.0, 0.0, 0.0, 0.0, -1.0, -1.0, -1.0};
    
    __global__ void prewitt_kernel(double* buffer, int filter_size,
                     double* return_value)
    {
    double s_x=0;
    double s_y=0;
    double magnitude=0;
    const double threshold = 12.0;

    for (int i=0; i<filter_size; ++i){
        s_x += buffer[i] * gx[i];
        s_y += buffer[i] * gy[i];
    }

    magnitude = sqrtf(s_x * s_x + s_y * s_y);

    if (magnitude > 255.0f) {
        magnitude = 255.0f;
    } 
    if (magnitude > threshold){
        return_value[0] = magnitude;
    } else {
        return_value[0] = (double)0.0;
    }


    }
    ''',
    'prewitt_kernel'
)



canny_smooth = cp.RawKernel(
    '''
    
    extern "C" 
    
    __constant__ double kernel[5][5] = 
        {{2.0, 4.0,  5.0,  4.0,  2.0}, 
         {4.0, 9.0,  12.0, 9.0,  4.0}, 
         {5.0, 12.0, 15.0, 12.0, 5.0}, 
         {4.0, 9.0,  12.0, 9.0,  4.0}, 
         {2.0, 4.0,  5.0,  4.0,  2.0}};
    
    __global__ void canny_smooth(double* buffer, int filter_size,
                     double* return_value)
    {
    double color=0;
    
  
    for (int i=0; i<filter_size; ++i){
        for (int kx = 0; kx < 4; ++kx){
            for (int ky = 0; ky < 4; ++ky){
                color += buffer[i] * kernel[kx][ky]/159.0;
    
            }
        }
    }
    color /= 25.0;
    if (color > 255.0f) {color = 255.0f;} 
    else if (color < 0.0f) {color = 0.0;}   
       
    return_value[0] = color;
    }
    ''',
    'canny_smooth'
)


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef sobel_gpu(surface_):
    """
    Applies the Sobel edge detection filter to a grayscale image using GPU acceleration.

    This function assumes the input image is **grayscale**, meaning the same intensity 
    is present in all three RGB channels. It applies the Sobel filter using only the red 
    channel (`R`), though the green (`G`) or blue (`B`) channels could also be used. 

    If the input image is not truly grayscale, the Sobel effect may vary slightly because 
    the RGB channels might have different intensities.

    **Processing Details:**
    - Converts the `pygame.Surface` into a 3D array (`H, W, RGB`).
    - Transfers the image data to the GPU using `CuPy`.
    - Applies the Sobel filter using `sobel_cupy()`, which operates on the GPU.
    - Converts the processed GPU array back to a `pygame.Surface`.

    Parameters:
    -----------
    surface_ : pygame.Surface
        A 24-bit or 32-bit `pygame.Surface` representing the input image.

    Returns:
    --------
    pygame.Surface
        A new `pygame.Surface` with the Sobel edge detection effect applied format 24-bit

    Raises:
    -------
    TypeError:
        If `surface_` is not a `pygame.Surface`.
    ValueError:
        If the surface has invalid dimensions or cannot be converted into an array.

    Notes:
    ------
    - The function assumes that `sobel_cupy()` is implemented to process the GPU array.
    - The final image retains the `RGB` format, but only the red channel is modified.
    """

    if not PyObject_IsInstance(surface_, pygame.Surface):
        raise TypeError(f"Argument `surface_` must be a pygame.Surface, got {type(surface_)}.")

    cdef unsigned char [:, :, :] cpu_array_

    try:
        # Retrieve pixel data as a 3D array (H, W, RGB)
        cpu_array_ = surface_.get_view('3')
    except Exception as e:
        raise ValueError(f"\nCannot reference `surface_` pixels into a 3D array.\n{e}")

    cdef:
        Py_ssize_t w, h

    # Get image dimensions
    w, h = cpu_array_.shape[:2]

    if w <= 0 or h <= 0:
        raise ValueError("Surface width or height cannot be zero or negative!")

    # Transfer data to GPU (CuPy array)
    gpu_array = cp.asarray(cpu_array_)

    # Apply Sobel edge detection on the GPU
    sobel_cupy(w, h, gpu_array)

    # Convert the processed GPU array back to a Pygame surface
    return frombuffer(
        gpu_array.astype(cp.uint8).transpose(1, 0, 2).tobytes(),
        (w, h),
        "RGB")



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef inline void sobel_cupy(
    const Py_ssize_t w,
    const Py_ssize_t h,
    gpu_array_):

    """
    Applies Sobel edge detection to the red channel of the image using GPU acceleration.

    This function processes the red channel of the input image (`gpu_array_`) using 
    a Sobel filter. The filter is applied using the `cupyx.scipy.ndimage.generic_filter`
    method, which is GPU-accelerated for faster processing. The filtered result is 
    applied to all three color channels (R, G, B) of the image.

    **Processing Details:**
    - The Sobel filter is applied only to the red channel (`R`) of the image.
    - The result is then copied to the green (`G`) and blue (`B`) channels as well, 
      making the entire image look like a grayscale edge-detection result.
    - This approach is based on the assumption that the input image is already in grayscale.

    Parameters:
    -----------
    w : Py_ssize_t
        The width of the input image.

    h : Py_ssize_t
        The height of the input image.

    gpu_array_ : CuPy ndarray
        A 3D GPU array (H, W, 3) representing the input image with RGB channels.

    Returns:
    --------
    None
        The Sobel edge detection is applied directly to the `gpu_array_` in-place.

    Notes:
    ------
    - The function uses the `sobel_kernel` for edge detection and assumes that it is defined.
    - The result will modify all three channels (`R`, `G`, and `B`) to contain the edge-detection result.
    - The operation is performed in-place on the provided `gpu_array_`, and no new array is returned.
    """

    # Access the red channel of the image (assuming the input is in grayscale format)
    r = gpu_array_[:, :, 0]

    # Apply the Sobel filter to the red channel (3x3 kernel size)
    sobel2d_r = cupyx.scipy.ndimage.generic_filter(
        r, sobel_kernel, 3).astype(dtype=cp.uint8)

    # Synchronize the GPU stream to ensure the operation is complete
    cp.cuda.Stream.null.synchronize()

    # Apply the result of the Sobel filter to all three channels (R, G, B)
    gpu_array_[:, :, 0], \
    gpu_array_[:, :, 1], \
    gpu_array_[:, :, 2] = sobel2d_r, sobel2d_r, sobel2d_r





@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef prewitt_gpu( surface_):
    """
    Prewitt edge detection

    Applies the Prewitt edge detection algorithm to a grayscale image. The algorithm 
    is applied to the red channel of the input image, which is assumed to be grayscale. 
    The result is copied to all RGB channels. If the image is not grayscale, the result 
    may differ since the red, green, and blue channels may have different intensity values.

    This function requires the input image (`surface_`) to be in a format compatible 
    with 32-bit or 24-bit color depth.

    **Processing Details:**
    - The image should be in grayscale (all RGB channels should have similar intensity).
    - The Prewitt filter is applied to the red channel, and the resulting edge-detected 
      image is copied to all channels (R, G, B).

    Parameters:
    -----------
    surface_ : pygame.Surface
        The input image in Pygame surface format. It should be in grayscale for best results.

    Returns:
    --------
    pygame.Surface
        A new Pygame surface with the Prewitt edge detection effect applied format 24-bit

    Raises:
    -------
    TypeError
        If `surface_` is not a Pygame surface.

    ValueError
        If the input surface has invalid or null width/height.
    """

    # Ensure that the surface is a valid Pygame surface
    if not PyObject_IsInstance(surface_, pygame.Surface):
        raise TypeError(f"Argument `surface_` must be a pygame.Surface, got {type(surface_)}.")

    # Try to extract the 3D pixel array view from the surface
    cdef unsigned char [:, :, :] rgb_array
    try:
        rgb_array = surface_.get_view('3')
    except Exception as e:
        raise ValueError("\nCannot reference `surface_` pixels into a 3D array.\n %s " % e)

    # Extract the dimensions (width and height) of the surface
    cdef Py_ssize_t w, h
    w, h = (<object>rgb_array).shape[:2]

    # Validate that the surface dimensions are valid
    if w <= 0 or h <= 0:
        raise ValueError("Surface width or height cannot be null!")

    gpu_array = cp.asarray(rgb_array)
    prewitt_cupy(w, h, gpu_array)

    return frombuffer(
        gpu_array.astype(cp.uint8).transpose(1, 0, 2).tobytes(),
        (w, h),
        "RGB")



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef inline void prewitt_cupy(
    const Py_ssize_t w,
    const Py_ssize_t h,
    gpu_array_
):
    """
    Apply the Prewitt edge detection filter to a GPU-based image array.

    This function processes a given GPU-based image using the Prewitt operator to 
    detect edges. It applies the filter to the red channel and then copies the 
    resulting edge map to all three color channels.

    Parameters:
    -----------
    w : Py_ssize_t
        Width of the image.

    h : Py_ssize_t
        Height of the image.

    gpu_array_ : cupy.ndarray
        A 3D GPU array (height x width x 3) representing an RGB image.
        The processing is done in-place, modifying the input array.

    Notes:
    ------
    - The function assumes the input image is already allocated on the GPU.
    - The Prewitt filter is applied to the red channel of the image.
    - The edge-detected output is copied to all three channels.
    """

    # Extract the red channel from the input image
    r = gpu_array_[:, :, 0]

    # Apply the Prewitt filter to detect edges in the red channel
    sobel2d_r = cupyx.scipy.ndimage.generic_filter(r, prewitt_kernel, 3).astype(dtype=cp.uint8)

    # Ensure all GPU operations are completed before proceeding
    cp.cuda.Stream.null.synchronize()

    # Copy the filtered result to all three RGB channels
    gpu_array_[:, :, 0], \
    gpu_array_[:, :, 1], \
    gpu_array_[:, :, 2] = sobel2d_r, sobel2d_r, sobel2d_r





@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef canny_gpu(surface_):
    """
    CANNY EDGE DETECTION

    Applies the Canny edge detection algorithm to a grayscale image. This algorithm 
    works by first applying a Sobel filter to detect edges in the image. The image 
    is assumed to be grayscale (i.e., all RGB channels have the same intensity). The 
    algorithm uses the red channel, but it can be equally applied to the green or blue 
    channels as well. If the image is not grayscale, the result may differ, as the RGB 
    channels could have different intensity values.

    **Processing Details:**
    - The image should be in grayscale, with similar intensities across all RGB channels.
    - The Canny edge detection is typically applied after converting the image to grayscale.

    Parameters:
    -----------
    surface_ : pygame.Surface
        The input image in Pygame surface format. It should be a grayscale image for best results.

    Returns:
    --------
    pygame.Surface
        A new Pygame surface with the Canny edge detection effect applied format 24-bit

    Raises:
    -------
    TypeError
        If `surface_` is not a Pygame surface.

    ValueError
        If the input surface has invalid or null width/height.
    """

    # Validate the input surface type to ensure it is a valid Pygame surface
    if not PyObject_IsInstance(surface_, pygame.Surface):
        raise TypeError(f"Argument `surface_` must be a pygame.Surface, got {type(surface_)}.")

    # Try to get the 3D pixel array from the surface for processing
    cdef unsigned char [:, :, :] rgb_array
    try:
        rgb_array = surface_.get_view('3')
    except Exception as e:
        raise ValueError("\nCannot reference `surface_` pixels into a 3D array.\n %s " % e)

    # Get the width and height of the image from the 3D array shape
    cdef Py_ssize_t w, h
    w, h = (<object>rgb_array).shape[:2]

    # Ensure the surface has valid dimensions (non-zero width/height)
    if w <= 0 or h <= 0:
        raise ValueError("Surface width or height cannot be null!")

    # Convert the pixel data to a GPU array for processing
    gpu_array = cp.asarray(rgb_array)

    # Apply the Canny edge detection algorithm on the GPU
    canny_cupy(w, h, gpu_array)

    # Convert the processed GPU array back into a Pygame surface and return
    return frombuffer(
        gpu_array.astype(cp.uint8).transpose(1, 0, 2).tobytes(),
        (w, h),
        "RGB")


# canny kernel 5x5
canny_kernel_5x5 = \
    cp.array([[2, 4, 5, 4, 2, ],
              [4, 9, 12, 9, 4],
              [5, 12, 15, 12, 5],
              [4, 9, 12, 9, 4],
              [2, 4, 5, 4, 2]], dtype=cp.float32) * <float>1.0 / <float>256.0



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef inline void canny_cupy(
    const Py_ssize_t w,
    const Py_ssize_t h,
    gpu_array_):

    """
    Applies Canny edge detection on the GPU using convolution and filtering.

    This function processes the red channel of the input image, applies a convolution 
    with a predefined Canny kernel to detect edges, and then applies a Sobel filter 
    for further edge detection refinement. The result is stored back into the red, 
    green, and blue channels of the input image, since the image is assumed to be grayscale.

    **Processing Steps:**
    1. Convolve the red channel with a Canny kernel (5x5) to detect edges.
    2. Apply a Sobel filter to the convolved result for more refined edge detection.
    3. Update the red, green, and blue channels with the Sobel-filtered results.

    **Assumptions:**
    - The input image is assumed to be grayscale, meaning the RGB channels will have 
      the same intensity. Only the red channel is used for edge detection, but as the 
      image is grayscale, the green and blue channels will be identical.

    Parameters:
    -----------
    w : int
        Width of the image (in pixels).

    h : int
        Height of the image (in pixels).

    gpu_array_ : cupy.ndarray
        A 3D GPU array (shape: [height, width, 3]) representing the RGB image 
        to be processed.

    Returns:
    --------
    None
        The function modifies the `gpu_array_` in place. The processed image is 
        stored back in the same array.

    Notes:
    ------
    The operation is performed in-place, modifying the red, green, and blue 
    channels of the input `gpu_array_`.
    """

    # Apply convolution with the Canny kernel to the red channel for edge detection
    r = cupyx.scipy.ndimage.convolve(
        gpu_array_[ :, :, 0 ], canny_kernel_5x5, mode = 'constant', cval = 0.0)

    # Apply Sobel filter to the convolved result (refining edge detection)
    sobel2d_r = cupyx.scipy.ndimage.generic_filter(r, sobel_kernel, 3).astype(dtype = cp.uint8)

    # Update all RGB channels with the refined Sobel results (grayscale assumption)
    gpu_array_[ :, :, 0 ], \
    gpu_array_[ :, :, 1 ], \
    gpu_array_[ :, :, 2 ] = sobel2d_r, sobel2d_r, sobel2d_r

    # Synchronize GPU operations to ensure processing is complete
    cp.cuda.Stream.null.synchronize()





color_reduction_kernel = cp.ElementwiseKernel(
    'uint8 r, uint8 g, uint8 b, int16 color_number',
    'uint8 rr, uint8 gg, uint8 bb',
    '''
    
    const float f = 255.0f / (float)color_number;
    const float c1 = (float)color_number / 255.0f;
    
    rr = (unsigned char)((int)((float)roundf(c1 * (float)r) * f));
    gg = (unsigned char)((int)((float)roundf(c1 * (float)g) * f));
    bb = (unsigned char)((int)((float)roundf(c1 * (float)b) * f));
 
    ''', 'color_reduction_kernel'
)

@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef color_reduction_gpu(surface_, int color_number=8):
    """
    Color reduction effect

    This function applies a color reduction effect to a given image by reducing the number of unique 
    colors in the image to a specified palette size. Each pixel's color is replaced by the closest match 
    from a limited color palette.

    The reduction is achieved by quantization each RGB color channel. For example, if `color_number=8`, 
    each channel will have 8 distinct levels (0, 32, 64, ..., 255). This results in a total of 8^3 possible colors.

    This function works with 32-bit and 24-bit images (Pygame Surface format) and reduces the number of unique colors 
    in the image based on the provided `color_number` parameter.

    Parameters:
    -----------
    surface_ : pygame.Surface
        The input surface (image) to which the color reduction will be applied.

    color_number : int, optional
        The number of colors (levels) to reduce each RGB channel to. The default value is 8.
        The number of colors in the palette will be `color_number^2`.

    Returns:
    --------
    pygame.Surface
        A new surface with the color reduction effect applied, in 24-bit format.

    Raises:
    -------
    TypeError
        If the `surface_` argument is not a valid `pygame.Surface` instance.

    ValueError
        If the `color_number` is less than or equal to 0 or if the surface has invalid dimensions.
    """

    # Validate that the input is a valid Pygame surface
    if not PyObject_IsInstance(surface_, pygame.Surface):
        raise TypeError(f"Argument `surface_` must be a pygame.Surface, got {type(surface_)}.")

    cdef unsigned char [:, :, :] rgb_array

    # Try to obtain a 3D view of the surface pixels as an array (RGB channels)
    try:
        rgb_array = surface_.get_view('3')
    except Exception as e:
        raise ValueError("\nCannot reference `surface_` pixels into a 3d array.\n %s " % e)

    cdef Py_ssize_t w, h

    # Get the width and height of the surface
    w, h = (<object>rgb_array).shape[:2]

    # Validate that the surface has non-zero dimensions
    if w == 0 or h == 0:
        raise ValueError("Surface width or height cannot be null!")

    # Validate that color_number is a positive integer
    if color_number <= 0:
        raise ValueError("\nArgument color_number cannot be <= 0")

    # Convert the RGB array into a GPU array for processing
    gpu_array = cp.asarray(rgb_array)

    # Apply the color reduction effect on the GPU
    color_reduction_cupy(w, h, gpu_array, color_number)

    # Convert the processed GPU array back into a Pygame surface and return it
    return frombuffer(
        gpu_array.astype(cp.uint8).transpose(1, 0, 2).tobytes(),
        (w, h),
        "RGB"
    )


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef inline void color_reduction_cupy(
    const Py_ssize_t w,
    const Py_ssize_t h,
    gpu_array_,
    const int color_number
):
    """
    Applies the color reduction effect to each RGB channel on the GPU.

    This function reduces the number of colors in each RGB channel by quantizing them to a limited palette.
    The palette size is determined by `color_number`, which specifies the number of levels per channel. 
    Each color channel (R, G, B) is reduced independently.

    Parameters:
    -----------
    w : int
        Width of the image (in pixels).

    h : int
        Height of the image (in pixels).

    gpu_array_ : cupy.ndarray
        A 3D GPU array (shape: [height, width, 3]) representing the RGB image to be processed.

    color_number : int
        The number of colors (levels) to reduce each RGB channel to. 
        This determines the size of the color palette.

    Returns:
    --------
    None
        This function modifies the `gpu_array_` in place. 
        The processed image is stored back in the same array.

    Notes:
    ------
    - The function operates in-place and directly modifies the red, green, and blue channels of the input array.
    - The quantization of colors is achieved by the `color_reduction_kernel` function.
    """

    # Apply color reduction on each channel (R, G, B)
    gpu_array_[:, :, 0], \
    gpu_array_[:, :, 1], \
    gpu_array_[:, :, 2] = color_reduction_kernel(
        gpu_array_[:, :, 0], gpu_array_[:, :, 1], gpu_array_[:, :, 2], color_number
    )




rgb2hsv_cuda = r'''   
    extern "C"
    
    __global__ void rgb2hsv(float * r, float * g, float * b, int w, int h, double val_)
    {
    
    int xx = blockIdx.x * blockDim.x + threadIdx.x;     
    int yy = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Index value of the current_ pixel
    const int index = yy * h + xx;
    const int t_max = h * w;
     
    if (index> 0 && index < t_max){ 
     
    float h, s, v;
    int i=-1; 
    float f, p, q, t;
    float mx, mn;
    
    // Create a reference to RGB
    float rr = r[index];
    float gg = g[index];
    float bb = b[index];
    
    // Find max and min of RGB values 
    if (rr > gg){
		if (rr > bb){
			mx = rr;
			if (bb > gg){ mn = gg;}
			else mn = bb;
        }
		else{
			mx = bb;
			if (bb > gg){ mn = gg;}
			else mn = bb;
		}
    }
	else{
		if (gg > bb){
			mx = gg;
			if (bb > rr){ mn = rr;}
			else mn = bb;
		} 
		else{
			mx = bb;
			if (bb > rr) { mn = rr;}
			else  mn = bb;
		}
    }
    
    
    __syncthreads();
    
    // Convert RGB to HSV 
    float df = mx-mn;  
    float df_ = 1.0f/df;        
       
    if (mx == mn)
    { h = 0.0;}
  
    else if (mx == rr){
	    h = (float)fmodf(60.0f * ((gg-bb) * df_) + 360.0, 360);
	}
    else if (mx == gg){
	    h = (float)fmodf(60.0f * ((bb-rr) * df_) + 120.0, 360);
	}
    else if (mx == bb){
	    h = (float)fmodf(60.0f * ((rr-gg) * df_) + 240.0, 360);
    }
    
    if (mx == 0.0){
        s = 0.0;
    }
    else{
        s = df/mx;
    }
     
    v = mx;   
    h = h * 1.0f/360.0f;

    // Increment the hue 
    h = (float)fmodf(h + val_, (double)1.0);

    __syncthreads();
    
    // Convert HSV to RGB    
    if (s == 0.0){
         r[index] = v;
         g[index] = v;
         b[index] = v;         
         }
    else {
        i = (int)(h*6.0f);
        f = (h*6.0f) - i;
        p = v*(1.0f - s);
        q = v*(1.0f - s*f);
        t = v*(1.0f - s*(1.0f-f));
        i = i%6;
        
        switch(i) { 
            case 0:
                r[index] = v;
                g[index] = t;
                b[index] = p;
                break; 
            case 1: 
                r[index] = q; 
                g[index] = v;
                b[index] = p;
                break;
            case 2:
                r[index] = p;
                g[index] = v;
                b[index] = t;
                break;
            case 3:
                r[index] = p;
                g[index] = q;
                b[index] = v;
                break;
            case 4:
                r[index] = t;
                g[index] = p;
                b[index] = v;
                break;
            case 5: 
                r[index] = v;
                g[index] = p; 
                b[index] = q;
                break;
            default:
                ;
            
        }
    }
    
    }
    
    __syncthreads();
    
    
  }
'''

@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef hsv_gpu(
        surface_,
        float val_,
        tuple grid_,
        tuple block_
):
    """
    HSV (Hue Rotation with GPU Acceleration)
    
    Applies a hue rotation effect to an image using CUDA-powered GPU acceleration. 
    This function modifies the hue of each pixel in the image while preserving its 
    saturation and brightness, effectively shifting colors while maintaining overall 
    image structure.
    
    Compatible with 32-bit and 24-bit image formats.
    
    ### Parameters:
    - **surface_** (*pygame.Surface*):  
      The input surface (image) on which the hue rotation effect will be applied.
    
    - **val_** (*float*):  
      The hue shift value, which must be within the range **[0.0, 1.0]**.  
      - `0.0` corresponds to a **-180° shift** (full backward rotation).  
      - `0.5` represents **0° shift** (no change).  
      - `1.0` corresponds to a **+180° shift** (full forward rotation).  
    
    - **grid_** (*tuple*):  
      Specifies the CUDA grid dimensions for kernel execution.  
      - Example: `(grid_y, grid_x)`, e.g., `(25, 25)`.  
      - The grid size should be tuned according to the texture and array sizes.
    
    - **block_** (*tuple*):  
      Specifies the CUDA block dimensions for kernel execution.  
      - Example: `(block_y, block_x)`, e.g., `(32, 32)`.  
      - The total number of threads (`block_x * block_y`) **must not exceed 1024** 
        due to GPU hardware limitations.
    
    ### Returns:
    - **pygame.Surface**:  
      A new surface containing the hue-modified image, returned in **24-bit RGB format**.
    
    ### Raises:
    - **TypeError**:  
      If `surface_` is not a valid `pygame.Surface` instance.
    
    - **ValueError**:  
      - If the input surface has **zero width or height**.  
      - If `val_` is **out of the valid range [0.0, 1.0]**.  
      - If the surface pixels cannot be **referenced as a 3D array**.
    
    ### Notes:
    - The input image **must be in RGB format** before applying this transformation.  
    - The function performs hue rotation using **GPU acceleration with CuPy** for optimized performance.  
    - The transformation works by **converting RGB to HSV**, modifying the **H (hue) channel**,
         and then **converting it back to RGB**.  
    """


    # Validate that the input is a valid Pygame surface
    if not PyObject_IsInstance(surface_, pygame.Surface):
        raise TypeError(f"Argument `surface_` must be a pygame.Surface, got {type(surface_)}.")

    cdef unsigned char [:, :, :] rgb_array

    # Try to obtain a 3D view of the surface pixels (RGB channels)
    try:
        rgb_array = surface_.get_view('3')
    except Exception as e:
        raise ValueError("\nCannot reference `surface_` pixels into a 3D array.\n %s " % e)

    cdef Py_ssize_t w, h

    # Get the width and height of the surface
    w, h = (<object>rgb_array).shape[:2]

    # Validate that the surface has non-zero dimensions
    if w == 0 or h == 0:
        raise ValueError("Surface width or height cannot be null!")

    # Validate that val_ is in the correct range [0.0, 1.0]
    if val_< 0.0 or val_> 1.0:
        raise ValueError(f"Argument `val_` must be in range [0.0 ... 1.0], got {val_}.")

    # Convert the RGB array into a GPU array for processing, ensuring float32 for accuracy
    gpu_array = cp.asarray(rgb_array).astype(dtype=cp.float32)

    # Apply the hue rotation effect using a CUDA kernel
    hsv_cupy(gpu_array, grid_, block_, val_, w, h)

    # Convert the processed GPU array back into a Pygame surface and return it
    return frombuffer(
        gpu_array.astype(cp.uint8).transpose(1, 0, 2).tobytes(),
        (w, h),
        "RGB"
    )




@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef void hsv_cupy(
    gpu_array,
    tuple grid_,
    tuple block_,
    float val_,
    const Py_ssize_t w,
    const Py_ssize_t h
):
    """
    Applies a hue rotation transformation to an image using CUDA-based GPU acceleration.

    This function converts the image from RGB to HSV, modifies the hue channel,
    and then converts it back to RGB, all within the GPU using a CUDA kernel.

    Parameters:
    -----------
    gpu_array : CuPy array
        A 3D GPU array representing the image in RGB format.

    grid_ : tuple
        Tuple representing the grid dimensions for CUDA kernel execution.
        Example: (grid_y, grid_x), e.g., (25, 25). The grid size should be set 
        based on the texture and array sizes.

    block_ : tuple
        Tuple representing the block dimensions for CUDA kernel execution.
        Example: (block_y, block_x), e.g., (32, 32). The total number of threads 
        (`block_x * block_y`) should not exceed 1024 due to GPU limitations.

    val_ : float
         The hue shift value, which must be within the range **[0.0, 1.0]**.  
          - `0.0` corresponds to a **-180° shift** (full backward rotation).  
          - `0.5` represents **0° shift** (no change).  
          - `1.0` corresponds to a **+180° shift** (full forward rotation).  

    w : Py_ssize_t
        The width of the image.

    h : Py_ssize_t
        The height of the image.

    Returns:
    --------
    None
        The function modifies `gpu_array` in place, applying the hue transformation.

    Notes:
    ------
    - The image should be in RGB format before processing.
    - Uses a CUDA kernel (`rgb2hsv_cuda`) for efficient RGB-to-HSV conversion and hue modification.
    - Synchronization (`cp.cuda.Stream.null.synchronize()`) ensures all GPU operations are complete 
      before continuing execution.
    """

    # Load the CUDA kernel module containing the RGB-to-HSV conversion function
    module = cp.RawModule(code=rgb2hsv_cuda)

    # Get the function from the compiled CUDA module
    rgb_to_hsv_ = module.get_function("rgb2hsv")

    # Normalize RGB values to the range [0, 1] for HSV conversion
    r = (gpu_array[:, :, 0] * <float>ONE_255)  # Red channel normalized
    g = (gpu_array[:, :, 1] * <float>ONE_255)  # Green channel normalized
    b = (gpu_array[:, :, 2] * <float>ONE_255)  # Blue channel normalized

    # Execute the CUDA kernel for hue rotation
    rgb_to_hsv_(grid_, block_, (r, g, b, w, h, val_))

    # Convert the modified HSV values back to RGB and scale to [0, 255]
    gpu_array[:, :, 0] = cp.multiply(r, <float>255.0)  # Red channel scaled back
    gpu_array[:, :, 1] = cp.multiply(g, <float>255.0)  # Green channel scaled back
    gpu_array[:, :, 2] = cp.multiply(b, <float>255.0)  # Blue channel scaled back

    # Ensure GPU operations complete before returning
    cp.cuda.Stream.null.synchronize()




downscale_kernel = cp.RawKernel(
    r'''


    extern "C"
    __global__  void downscale_kernel(unsigned char* source, unsigned char * new_array,
    const double w1, const double h1, const double w2, const double h2)
    {

        int xx = blockIdx.x * blockDim.x + threadIdx.x;
        int yy = blockIdx.y * blockDim.y + threadIdx.y;
        int zz = blockIdx.z * blockDim.z + threadIdx.z;

        const int index = yy * h1 * 3 + xx * 3 + zz;
        const int index1 = (int)(yy * h1 * h2/h1 * 3 + xx * w2/w1 * 3  + zz);
        const int t_max = h1 * w1 * 3;
        const int t_max_ = h2 * w2 * 3;

        if (index>= 0 && index <= t_max){

        __syncthreads();

        const float fx = (float)(w2 / w1);
        const float fy = (float)(h2 / h1);


        __syncthreads();


        float ix = (float)index / 3.0f;
        int y = (int)(ix / h1);
        int x = (int)ix % (int)h1;

        int new_x = (int)(x * fx);
        int new_y = (int)(y * fy);

        const int new_index = (int)(new_y * 3 * h2) + new_x * 3 + zz;

        __syncthreads();

        new_array[index1] = source[index];

        __syncthreads();
        }
    }
    ''',
    'downscale_kernel'
)

@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef mult_downscale_gpu(gpu_array

):
    """
    Perform bloom downscaling by progressively reducing image resolution.

    This function downscales an input GPU-based image into four progressively 
    smaller sub-arrays with resolutions reduced by factors of 2, 4, 8, and 16.
    It is compatible with 24-bit and 32-bit images (RGB format, uint8). 

    Parameters:
    -----------
    gpu_array : cupy.ndarray
        A 3D CuPy array with shape (height, width, 3) and dtype uint8, 
        representing an RGB image stored on the GPU.

    Returns:
    --------
    tuple of cupy.ndarray
        A tuple containing four downscaled versions of the input image 
        with shapes (H/2, W/2, 3), (H/4, W/4, 3), (H/8, W/8, 3), and (H/16, W/16, 3).

    Notes:
    ------
    - The function performs downscaling using `cupyx.scipy.ndimage.zoom`, 
      applying nearest-neighbor interpolation (`order=0`) to preserve hard edges.
    - Pixels outside the boundaries are filled with zero (`mode='constant', cval=0.0`).
    - GPU synchronization is performed to ensure all CUDA operations complete before returning.
    """

    # Ensure input is of type uint8
    assert gpu_array.dtype == cupy.uint8, \
        f"\nArgument gpu_array datatype must be uint8, but got {gpu_array.dtype}"

    # Downscale the image by progressively reducing its size
    downscale_x2 = cupyx.scipy.ndimage.zoom(
        gpu_array, (<float>1.0 / <float>2.0, <float>1.0 / <float>2.0, 1), 
        order=0, mode='constant', cval=0.0)

    downscale_x4 = cupyx.scipy.ndimage.zoom(
        downscale_x2, (<float>1.0 / <float>2.0, <float>1.0 / <float>2.0, 1), 
        order=0, mode='constant', cval=0.0)

    downscale_x8 = cupyx.scipy.ndimage.zoom(
        downscale_x4, (<float>1.0 / <float>2.0, <float>1.0 / <float>2.0, 1), 
        order=0, mode='constant', cval=0.0)

    downscale_x16 = cupyx.scipy.ndimage.zoom(
        downscale_x8, (<float>1.0 / <float>2.0, <float>1.0 / <float>2.0, 1), 
        order=0, mode='constant', cval=0.0)

    # Synchronize GPU computations to ensure all operations complete
    cp.cuda.Stream.null.synchronize()

    return downscale_x2, downscale_x4, downscale_x8, downscale_x16

#
# @cython.boundscheck(False)
# @cython.wraparound(False)
# @cython.nonecheck(False)
# @cython.cdivision(True)
# cpdef object zoom_in_gpu(object surface_, int w, int h):
#
#     if not PyObject_IsInstance(surface_, pygame.Surface):
#         raise TypeError('Argument surface_ must be a pygame Surface type got %s' % type(surface_))
#
#     assert w > 0, "Argument w cannot be < 0"
#     assert h > 0, "Argument h cannot be < 0"
#
#     try:
#         gpu_array_ = cp.asarray(surface_.get_view('3'), dtype=cp.uint8)
#
#     except Exception as e:
#         raise ValueError(
#             "\nCannot reference source pixels into a 3d array.\n %s " % e)
#
#     cdef int w0, h0
#     w0, h0 = gpu_array_.shape[0], gpu_array_.shape[1]
#
#     downscale_ = cupyx.scipy.ndimage.zoom(
#         gpu_array_, (<float>w0/<float>w, <float>h0/<float>h, 1), order=0, modes='constant', cval=0.0)
#
#     cp.cuda.Stream.null.synchronize()
#
#     return frombuffer(downscale_.astype(
#         dtype=cp.uint8).transpose(1, 0, 2).tobytes(),
#                       (downscale_.shape[0], downscale_.shape[1]), "RGB")
#
#
# @cython.boundscheck(False)
# @cython.wraparound(False)
# @cython.nonecheck(False)
# @cython.cdivision(True)
# cpdef object upscale_gpu(object surface_, int w, int h):
#
#     if not PyObject_IsInstance(surface_, pygame.Surface):
#         raise TypeError('Argument surface_ must be a pygame Surface type got %s' % type(surface_))
#
#     assert w > 0, "Argument w cannot be < 0"
#     assert h > 0, "Argument h cannot be < 0"
#
#     try:
#         gpu_array_ = cp.asarray(surface_.get_view('3'), dtype=cp.uint8)
#
#     except Exception as e:
#         raise ValueError(
#             "\nCannot reference source pixels into a 3d array.\n %s " % e)
#
#     cdef int w0, h0
#     w0, h0 = gpu_array_.shape[0], gpu_array_.shape[1]
#
#     gpu_array = cp.asarray(gpu_array_, dtype=cp.uint8)
#
#     upscale_array = cupyx.scipy.ndimage.zoom(
#         gpu_array, (<float>w/<float>w0, <float>h/<float>h0, 1), order=0, modes='constant', cval=0.0)
#
#     cp.cuda.Stream.null.synchronize()
#
#     return frombuffer(gpu_array_.astype(dtype=cp.uint8).transpose(1, 0, 2).tobytes(), (w, h), "RGB")




upscale_x2 = cp.RawKernel(
    r'''
    extern "C" __global__
    void upscale_x2(unsigned char* source, unsigned char * new_array,
    const double w1, const double h1, const double w2, const double h2)
    {
        int xx = blockIdx.x * blockDim.x + threadIdx.x;
        int yy = blockIdx.y * blockDim.y + threadIdx.y;
        int zz = blockIdx.z * blockDim.z + threadIdx.z;

        __syncthreads();

        const float fx = (float)(w1 / w2);
        const float fy = (float)(h1 / h2);

        const int index = yy * h2 * 3 + xx * 3 + zz ;

        const unsigned int ix = (int)(index / 3.0f);
        const int y = (int)(ix/h2);
        const int x = ix % (int)h2;

        const int new_x = (int)(x * fx);
        const int new_y = (int)(y * fy);

        int new_index = (int)((int)(new_y * h1 *3) + (int)(new_x * 3) + zz);
        __syncthreads();

        new_array[index] = source[new_index];

        __syncthreads();

    }
    ''',
    'upscale_x2'
)


# ************************************ BLOOM ***************************************


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef inline void bpf_c(gpu_array_, unsigned int threshold_=128):
    """
    Apply a band-pass filter (BPF) to an RGB image on the GPU.

    This function processes a GPU-based image by applying a band-pass filter to 
    each color channel (red, green, and blue). The filtering operation is 
    performed using a custom kernel function `bpf_kernel`, which modifies the 
    pixel values based on a given threshold.

    Parameters:
    -----------
    gpu_array_ : cupy.ndarray
        A 3D GPU array (height x width x 3) representing an RGB image.
        The filtering operation is performed in-place.
    
    threshold_ : unsigned int, optional (default=128)
        A threshold value that controls the filtering effect in the `bpf_kernel`.

    Notes:
    ------
    - The function extracts the red, green, and blue channels separately.
    - The `bpf_kernel` function is applied to each channel to perform the filtering.
    - The modified channel values are stored back in the original GPU array.
    - GPU stream synchronization ensures all CUDA operations are completed 
      before proceeding.
    """

    # Extract the individual red, green, and blue color channels
    r = gpu_array_[:, :, 0]
    g = gpu_array_[:, :, 1]
    b = gpu_array_[:, :, 2]

    # Apply the band-pass filter kernel to each channel
    r, g, b = bpf_kernel(r, g, b, <float>threshold_)

    # Ensure all GPU computations complete before further execution
    cp.cuda.Stream.null.synchronize()




@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef inline void gaussian_3x3_c(gpu_array_):
    """
    Apply a 3x3 Gaussian blur filter to an RGB image on the GPU.

    This function performs a 3x3 Gaussian convolution on each color channel 
    (red, green, and blue) of a GPU-based image. The filtering smooths 
    the image by reducing noise and fine details.

    Parameters:
    -----------
    gpu_array_ : cupy.ndarray
        A 3D GPU array (height x width x 3) representing an RGB image.
        The filtering operation modifies the image in-place.

    Notes:
    ------
    - The function separately processes the red, green, and blue channels.
    - The convolution is performed using a predefined 3x3 Gaussian kernel (`gaussian_kernel_3x3`).
    - The `mode='constant', cval=0.0` ensures that pixels outside the boundaries 
      are treated as zero.
    - GPU synchronization is performed to ensure all operations are completed 
      before moving forward.
    """

    # Extract the individual red, green, and blue color channels
    r = gpu_array_[:, :, 0]
    g = gpu_array_[:, :, 1]
    b = gpu_array_[:, :, 2]

    # Apply 3x3 Gaussian convolution to each color channel
    r = cupyx.scipy.ndimage.convolve(r, gaussian_kernel_3x3, mode='constant', cval=0.0).astype(cp.uint8)
    g = cupyx.scipy.ndimage.convolve(g, gaussian_kernel_3x3, mode='constant', cval=0.0).astype(cp.uint8)
    b = cupyx.scipy.ndimage.convolve(b, gaussian_kernel_3x3, mode='constant', cval=0.0).astype(cp.uint8)

    # Ensure all GPU computations complete before further execution
    cp.cuda.Stream.null.synchronize()




@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef inline void gaussian_5x5_c(gpu_array_):
    """
    Apply a 5x5 Gaussian blur filter to an RGB image on the GPU.

    This function performs a 5x5 Gaussian convolution on each color channel 
    (red, green, and blue) of a GPU-based image. The Gaussian blur helps 
    smooth the image by reducing noise and fine details.

    Parameters:
    -----------
    gpu_array_ : cupy.ndarray
        A 3D GPU array (height x width x 3) representing an RGB image.
        The filtering operation modifies the image in-place.

    Notes:
    ------
    - The function applies a predefined 5x5 Gaussian kernel (`gaussian_kernel_5x5`) 
      to each of the RGB channels separately.
    - The `mode='constant', cval=0.0` ensures that pixels outside the boundaries 
      are treated as zero during convolution.
    - The result is cast to `uint8` to maintain consistency with typical image formats.
    - GPU synchronization (`cp.cuda.Stream.null.synchronize()`) ensures all operations 
      are completed before further execution.
    """

    # Extract the individual red, green, and blue color channels
    r = gpu_array_[:, :, 0]
    g = gpu_array_[:, :, 1]
    b = gpu_array_[:, :, 2]

    # Apply 5x5 Gaussian convolution to each color channel
    r = cupyx.scipy.ndimage.convolve(r, gaussian_kernel_5x5, mode='constant', cval=0.0).astype(cp.uint8)
    g = cupyx.scipy.ndimage.convolve(g, gaussian_kernel_5x5, mode='constant', cval=0.0).astype(cp.uint8)
    b = cupyx.scipy.ndimage.convolve(b, gaussian_kernel_5x5, mode='constant', cval=0.0).astype(cp.uint8)

    # Ensure all GPU computations complete before further execution
    cp.cuda.Stream.null.synchronize()



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef upscale_c(gpu_array_, int new_width, int new_height, int order_=0):
    """
    Upscale an image on the GPU using interpolation.

    This function resizes a GPU-based image to the specified `new_width` and `new_height` 
    using the specified interpolation order. The resizing is performed using 
    `cupyx.scipy.ndimage.zoom`, which supports different interpolation methods.

    Parameters:
    -----------
    gpu_array_ : cupy.ndarray
        A 3D GPU array (height x width x 3) representing an RGB image in `uint8` format.

    new_width : int
        The desired width of the upscaled image.

    new_height : int
        The desired height of the upscaled image.

    order_ : int, optional (default=0)
        The interpolation order for resizing:
        - 0 (nearest-neighbor)
        - 1 (bilinear)
        - 3 (bicubic)
        Higher orders provide smoother results but increase computation time.

    Returns:
    --------
    cupy.ndarray
        The upscaled image as a GPU-based array with shape (`new_height`, `new_width`, 3).

    Notes:
    ------
    - The function ensures that the input array is of type `uint8` before processing.
    - The `mode='constant', cval=0.0` ensures that pixels outside the boundaries 
      are treated as zero during interpolation.
    - GPU synchronization (`cp.cuda.Stream.null.synchronize()`) is performed to 
      ensure all operations complete before returning the result.
    """

    # Ensure input is of type uint8 to maintain consistency with image processing standards
    if gpu_array_.dtype != cupy.uint8:
        raise TypeError(f"Argument `gpu_array_` datatype must be uint8, got {gpu_array_.dtype}.")

    # Extract original image dimensions
    cdef int w1, h1
    w1, h1 = (<object>gpu_array_).shape[:2]

    # Ensure the array is properly formatted as a CuPy array
    gpu_array = cp.asarray(gpu_array_, dtype=cp.uint8)

    # Perform upscaling using specified interpolation order
    upscale_array = cupyx.scipy.ndimage.zoom(
        gpu_array, (new_width / w1, new_height / h1, 1), 
        order=order_, mode='constant', cval=0.0
    )

    # Ensure all GPU computations complete before returning the result
    cp.cuda.Stream.null.synchronize()

    return upscale_array



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef bloom_gpu(
        surface_,
        unsigned int threshold_ = 128,
        bint fast_ = True,
        int flag_ = pygame.BLEND_RGB_ADD,
        unsigned short int factor_ = 2
    ):
    """
    Apply a Bloom Effect to a Pygame Surface using GPU Acceleration.

    This function enhances bright regions of an image to create a glowing effect (bloom)
    using a multi-step process that involves:
    - Downscaling the image into progressively smaller sub-surfaces.
    - Applying a bright-pass filter to isolate bright areas.
    - Blurring the bright regions using a Gaussian filter.
    - Upscaling and blending the processed layers back into the original surface.

    Parameters:
    -----------
    surface_ : pygame.Surface
        The input surface to apply the bloom effect on.

    threshold_ : int, optional (default=128)
        The brightness threshold for the bright-pass filter.
        Pixels with intensity above this value contribute to the bloom effect.
        Must be in range [0, 255].

    fast_ : bool, optional (default=True)
        - When `True`, only the lowest-resolution downsample (x16) is used for bloom.
        - When `False`, multiple downsampled layers (x2, x4, x8, x16) are processed
          and blended for a more detailed effect.

    flag_ : int, optional (default=pygame.BLEND_RGB_ADD)
        The blending mode used when merging the bloom effect back onto the original surface.
        Common flags include:
        - `pygame.BLEND_RGB_ADD` (default)
        - `pygame.BLEND_RGB_MULT`, etc.

    factor_ : int, optional (default=2)
        Determines how much the texture is downscaled.
        Must be in the range [0, 4], corresponding to downscaling by:
        - 1 (no reduction)
        - 2 (half-size)
        - 4 (quarter-size)
        - 8 (eighth-size)

    Returns:
    --------
    pygame.Surface
        The input surface blended with the bloom effect.

    Raises:
    -------
    ValueError:
        If the input image is too small to be processed.
    """

    # Make a copy of the original surface
    surface_copy = surface_.copy()

    # Reduce surface size based on the downscaling factor
    surface_ = smoothscale(
        surface_, 
        (surface_.get_width() >> factor_, surface_.get_height() >> factor_)
    )

    # Convert the surface pixels to a GPU-compatible array
    try:
        gpu_array_ = cp.asarray(pixels3d(surface_), dtype=cp.uint8)
    except Exception as e:
        raise ValueError(f"\nCannot reference source pixels into a 3D array.\n {e}")

    # Define variables for image dimensions at different downscaling levels
    cdef int w1, h1, w2, h2, w4, h4, w8, h8, w16, h16
    cdef bint x2, x4, x8, x16 = False

    # Get original image dimensions
    w1, h1 = gpu_array_.shape[0], gpu_array_.shape[1]

    # Compute progressively smaller downscaling levels (halving each step)
    w2, h2 = w1 >> 1, h1 >> 1
    w4, h4 = w2 >> 1, h2 >> 1
    w8, h8 = w4 >> 1, h4 >> 1
    w16, h16 = w8 >> 1, h8 >> 1

    # Ensure the image is large enough to be processed
    if w16 == 0 or h16 == 0:
        raise ValueError(
            "\nImage too small to be processed.\n"
            "Increase the image size or decrease the downscaling factor.\n"
            f"Current factor: {factor_}"
        )

    # Allocate memory for the downscaled images on the GPU
    cdef:
        scale_x2 = cp.empty((w2, h2, 3), cp.uint8)
        scale_x4 = cp.empty((w4, h4, 3), cp.uint8)
        scale_x8 = cp.empty((w8, h8, 3), cp.uint8)
        scale_x16 = cp.empty((w16, h16, 3), cp.uint8)

    # Determine which downscaling levels are possible
    x2, x4, x8, x16 = w2 > 0, w4 > 0, w8 > 0, w16 > 0

    # Skip multi-step processing in fast mode
    if fast_:
        x2, x4, x8 = False, False, False

    # Perform multi-resolution downscaling
    scale_x2, scale_x4, scale_x8, scale_x16 = mult_downscale_gpu(gpu_array_)

    # Synchronize GPU processing before further operations
    cp.cuda.Stream.null.synchronize()

    # Process downscaled images and apply bloom at each level
    s2, s4, s8, s16 = None, None, None, None

    if x2:
        # Apply bright-pass filter and Gaussian blur
        bpf_c(scale_x2, threshold_=threshold_)
        gaussian_3x3_c(scale_x2)

        # Upscale back and create a pygame surface
        s2 = make_surface(upscale_c(scale_x2, w1, h1, order_=0).get())

    if x4:
        bpf_c(scale_x4, threshold_=threshold_)
        gaussian_3x3_c(scale_x4)
        s4 = make_surface(upscale_c(scale_x4, w1, h1, order_=0).get())

    if x8:
        bpf_c(scale_x8, threshold_=threshold_)
        gaussian_3x3_c(scale_x8)
        s8 = make_surface(upscale_c(scale_x8, w1, h1, order_=1).get())

    if x16:
        bpf_c(scale_x16, threshold_=threshold_)
        gaussian_3x3_c(scale_x16)
        s16 = make_surface(upscale_c(scale_x16, w1, h1, order_=1).get())

    # Blending the processed images back onto the original surface
    if fast_:
        # In fast mode, only the smallest resolution (x16) is blended
        s16 = smoothscale(s16, (w1 << factor_, h1 << factor_))
        surface_copy.blit(s16, (0, 0), special_flags=BLEND_RGB_ADD)
    else:
        # In normal mode, multiple layers are blended for a richer bloom effect
        s2.blit(s4, (0, 0), special_flags=BLEND_RGB_ADD)
        s2.blit(s8, (0, 0), special_flags=BLEND_RGB_ADD)
        s2.blit(s16, (0, 0), special_flags=BLEND_RGB_ADD)
        
        # Scale the blended effect back to the original size
        s2 = smoothscale(s2, (w1 << factor_, h1 << factor_))
        surface_copy.blit(s2, (0, 0), special_flags=BLEND_RGB_ADD)

    # Synchronize GPU operations before returning the final image
    cp.cuda.Stream.null.synchronize()

    return surface_copy




@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef bloom_array(
        gpu_array_,
        unsigned int threshold_ = 128,
        bint fast_              = True,
        int flag_               = pygame.BLEND_RGB_ADD,
        mask_                   = None
):
    """
    Apply a Bloom Effect to an Image Represented as a GPU Array.

    This function performs a bloom effect on a given `gpu_array_`, which represents an image on the GPU.
    The process involves:
    - Downscaling the image into progressively smaller versions (x2, x4, x8, x16).
    - Applying a bright-pass filter to isolate high-intensity pixels.
    - Blurring the bright regions to create a glow effect.
    - Upscaling and blending the processed images to achieve the final bloom effect.

    Parameters:
    -----------
    gpu_array_ : cupy.ndarray (dtype=uint8)
        A 3D array representing the image in GPU memory. Must have dtype `cupy.uint8`.

    threshold_ : int, optional (default=128)
        The brightness threshold for the bright-pass filter. Pixels above this value contribute to bloom.
        Must be within [0, 255].

    fast_ : bool, optional (default=True)
        - When `True`, only the lowest-resolution downsample (x16) is used, reducing processing time.
        - When `False`, multiple levels (x2, x4, x8, x16) are blended for a richer bloom effect.

    flag_ : int, optional (default=pygame.BLEND_RGB_ADD)
        The blending mode used when merging the bloom effect layers.

    mask_ : optional (default=None)
        Unused parameter, reserved for future implementations.

    Returns:
    --------
    tuple of pygame.Surface
        The processed bloom effect surfaces at different downscaling levels (s2, s4, s8, s16).
        If `fast_` is enabled, only `s16` is used.

    """

    # Ensure the input array is of type uint8
    assert gpu_array_.dtype == cupy.uint8, (
        "\nArgument gpu_array_ datatype must be uint8, got %s " % gpu_array_.dtype
    )

    # Initialize image dimensions for different downscaling levels
    cdef:
        int w1, h1, w2, h2, w4, h4, w8, h8, w16, h16
        bint x2, x4, x8, x16 = False

    # Get the original image dimensions
    w1, h1 = gpu_array_.shape[0], gpu_array_.shape[1]

    # Compute the dimensions for progressively downsampled versions
    w2, h2 = w1 >> 1, h1 >> 1
    w4, h4 = w2 >> 1, h2 >> 1
    w8, h8 = w4 >> 1, h4 >> 1
    w16, h16 = w8 >> 1, h8 >> 1

    # Ensure the image is large enough for processing
    if w16 == 0 or h16 == 0:
        raise ValueError(
            "\nImage cannot be processed. Increase the size of the image."
        )

    # Allocate GPU memory for downsampled images
    cdef:
        scale_x2 = cp.empty((w2, h2, 3), cp.uint8)
        scale_x4 = cp.empty((w4, h4, 3), cp.uint8)
        scale_x8 = cp.empty((w8, h8, 3), cp.uint8)
        scale_x16 = cp.empty((w16, h16, 3), cp.uint8)

    # Determine which downscaling levels are available
    x2, x4, x8, x16 = w2 > 0, w4 > 0, w8 > 0, w16 > 0

    # If the smallest required downscale cannot be performed, return
    if not x2:
        return

    # In fast mode, only the x16 level is processed
    if fast_:
        x2, x4, x8 = False, False, False

    # Perform multi-resolution downscaling on the GPU
    scale_x2, scale_x4, scale_x8, scale_x16 = mult_downscale_gpu(gpu_array_)

    # Ensure GPU processing is completed before proceeding
    cp.cuda.Stream.null.synchronize()

    s2, s4, s8, s16 = None, None, None, None

    # Process downscaled images to apply bright-pass filter and Gaussian blur
    if x2:
        bpf_c(scale_x2, threshold_=threshold_)
        gaussian_3x3_c(scale_x2)
        s2 = make_surface(upscale_c(scale_x2, w1, h1, order_=0).get())

    if x4:
        bpf_c(scale_x4, threshold_=threshold_)
        gaussian_3x3_c(scale_x4)
        s4 = make_surface(upscale_c(scale_x4, w1, h1, order_=0).get())

    if x8:
        bpf_c(scale_x8, threshold_=threshold_)
        gaussian_3x3_c(scale_x8)
        s8 = make_surface(upscale_c(scale_x8, w1, h1, order_=1).get())

    if x16:
        bpf_c(scale_x16, threshold_=threshold_)
        gaussian_3x3_c(scale_x16)
        s16 = make_surface(upscale_c(scale_x16, w1, h1, order_=1).get())

    # Synchronize GPU operations before returning
    cp.cuda.Stream.null.synchronize()

    return s2, s4, s8, s16

# --------------------------------------- CARTOON EFFECT


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef cartoon_gpu(
        surface_,
        int sobel_threshold_ = 128,
        int median_kernel_   = 2,
        unsigned char color_ = 8,
        bint contour_        = False,
        unsigned char flag_  = BLEND_RGB_ADD
):
    """
    Apply a cartoon effect to a given surface using GPU acceleration.

    This function processes an image to generate a cartoon-like effect by applying 
    edge detection, median filtering, and color quantization. The effect can also 
    include contour outlining if enabled.

    Compatible with 24-bit and 32-bit image formats.

    Parameters:
    -----------
    :param surface_: pygame.Surface
        The input image surface to be processed.

    :param sobel_threshold_: int, optional (default=128)
        The threshold value for the Sobel edge detection filter.

    :param median_kernel_: int, optional (default=2)
        Kernel size for median filtering (used to reduce noise while preserving edges).

    :param color_: int, optional (default=8)
        Number of maximum colors to be used in the cartoon effect (color reduction).

    :param contour_: bool, optional (default=False)
        Whether to draw contours on the edges detected in the image.

    :param flag_: int, optional (default=BLEND_RGB_ADD)
        Blending mode used to combine the effect with the original image.
        Options include:
        - `BLEND_RGB_ADD` (default)
        - `BLEND_RGB_SUB`
        - `BLEND_RGB_MULT`
        - `BLEND_RGB_MAX`
        - `BLEND_RGB_MIN`
    
    Returns:
    --------
    :return: pygame.Surface
        The transformed surface with the cartoon effect applied.

    """

    # Call the CuPy-based implementation for GPU-accelerated processing
    return cartoon_cupy(surface_, sobel_threshold_, median_kernel_, color_, contour_, flag_)



# Gaussian kernel 5x5
k = cp.array([[2, 4, 5, 4, 2, ],
              [4, 9, 12, 9, 4],
              [5, 12, 15, 12, 5],
              [4, 9, 12, 9, 4],
              [2, 4, 5, 4, 2]], dtype=cp.float32) * <float>1.0 / <float>256.0


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef inline void canny_cupy_c(gpu_array_):
    """
    Applies the Canny edge detection filter to the input GPU array using Sobel operators
    for edge detection and convolution for gradient calculation. The input image is
    processed in the GPU using CuPy and cupyx functions for efficient computation.

    Parameters:
    gpu_array_ (cupy.ndarray): A 3D GPU array representing the image (height, width, 3 channels).
    
    The first channel (index 0) of the image is used for gradient calculation. The result
    is stored back into all three channels of the input array (`gpu_array_[:, :, 0]`, 
    `gpu_array_[:, :, 1]`, `gpu_array_[:, :, 2]`), effectively converting the image 
    into grayscale based on the edge information.
    """
    
    # Apply convolution to the first channel of the input array using the predefined kernel 'k'
    r = cupyx.scipy.ndimage.convolve(gpu_array_[:, :, 0], k, mode='constant', cval=0.0)

    # Perform 2D Sobel edge detection filtering to calculate the gradient magnitude
    sobel2d_r = cupyx.scipy.ndimage.generic_filter(r, sobel_kernel, 3).astype(dtype=cp.uint8)

    # Assign the processed Sobel result to all three channels (R, G, B) of the input array
    # This effectively creates a grayscale image showing the edge magnitudes.
    gpu_array_[:, :, 0], \
    gpu_array_[:, :, 1], \
    gpu_array_[:, :, 2] = sobel2d_r, sobel2d_r, sobel2d_r

    # Synchronize the CUDA stream to ensure the computation is complete before proceeding
    cp.cuda.Stream.null.synchronize()


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef inline void sobel_cupy_c(gpu_array_):

    """
    Applies the Sobel edge detection filter to the input GPU array using a 2D Sobel kernel
    on the first channel (usually the grayscale channel) of the input image. The resulting
    gradient magnitude is then applied to all three channels (R, G, B) of the input array 
    to convert it to a grayscale edge-detected image.

    Parameters:
    gpu_array_ (cupy.ndarray): A 3D GPU array representing an image (height, width, 3 channels).
    The first channel (index 0) of the array is used for the edge detection.

    This function does not return a value but modifies the input array (`gpu_array_`) in-place 
    by applying the Sobel edge filter to all three color channels.
    """
    
    # Apply 2D Sobel edge detection filter to the first channel of the input array
    # sobel_kernel is assumed to be predefined. The result is a gradient magnitude.
    sobel2d_r = cupyx.scipy.ndimage.generic_filter(
        gpu_array_[:, :, 0], sobel_kernel, 3).astype(dtype=cp.uint8)

    # Synchronize the CUDA stream to ensure the computation completes before continuing
    cp.cuda.Stream.null.synchronize()

    # Assign the processed Sobel result to all three channels (R, G, B) of the input array
    # This creates a grayscale image of edge-detected features.
    gpu_array_[:, :, 0], \
    gpu_array_[:, :, 1], \
    gpu_array_[:, :, 2] = sobel2d_r, sobel2d_r, sobel2d_r



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef void median_cupy_c(gpu_array_, unsigned int size_=5):
    """
    Applies a median filter to each color channel (R, G, B) of the input GPU array.
    The median filter is applied to each channel individually to reduce noise in the image
    while preserving edges. The size of the filter kernel can be customized via the `size_` parameter.

    Parameters:
    gpu_array_ (cupy.ndarray): A 3D GPU array representing an image (height, width, 3 channels).
                               The filter is applied independently to the R, G, and B channels.
    w (int): The width of the input image (not directly used in this function but passed for consistency).
    h (int): The height of the input image (not directly used in this function but passed for consistency).
    size_ (unsigned int, optional): The size of the median filter kernel. Default is 5.

    The function does not return a value. It modifies the input array (`gpu_array_`) in place 
    by applying the median filter to all three color channels.
    """
    
    # Apply median filter to the Red (R) channel of the image
    gpu_array_[:, :, 0] = cupyx.scipy.ndimage.median_filter(gpu_array_[:, :, 0], size_)

    # Apply median filter to the Green (G) channel of the image
    gpu_array_[:, :, 1] = cupyx.scipy.ndimage.median_filter(gpu_array_[:, :, 1], size_)

    # Apply median filter to the Blue (B) channel of the image
    gpu_array_[:, :, 2] = cupyx.scipy.ndimage.median_filter(gpu_array_[:, :, 2], size_)

    # Synchronize the CUDA stream to ensure that the filtering operation completes
    cp.cuda.Stream.null.synchronize()


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef color_reduction_cupy_c(
    gpu_array_,  # Input: A CuPy GPU array with shape (h, w, 3), representing an image in RGB format
    int color_number   # Input: The desired number of colors for color reduction
):
    """
    Reduce the number of unique colors in an image represented as a GPU array.

    This function applies a color reduction algorithm to an image (3-channel RGB) using a GPU array.
    Each of the RGB channels is processed independently by the color_reduction_kernel to reduce the 
    number of unique colors to the specified color_number. The image array is processed on the GPU 
    to ensure efficient handling of large images.

    Parameters:
    gpu_array_ (object): A CuPy GPU array of shape (h, w, 3), representing an image in RGB format.
    color_number (int): The number of colors to reduce the image to.

    Returns:
    cupy.ndarray 3d array shape (w,h, 3): The processed GPU array with reduced colors.
    """
    
    # Reduce the color information by applying the color reduction kernel
    # The kernel is applied independently to each RGB channel of the image
    gpu_array_[:, :, 0], \
    gpu_array_[:, :, 1], \
    gpu_array_[:, :, 2] = color_reduction_kernel(
        gpu_array_[:, :, 0],  # Red channel of the image
        gpu_array_[:, :, 1],  # Green channel of the image
        gpu_array_[:, :, 2],  # Blue channel of the image
        color_number,         # Desired number of colors after reduction
        block_size=1024       # Define the block size for GPU kernel execution
    )
    
    return gpu_array_  # Return the image with reduced colors



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef cartoon_cupy(
        surface_,         # Input: pygame.Surface compatible (24 - 32 bit surface)
        int sobel_threshold_,    # Input: Integer value for Sobel edge detection threshold
        int median_kernel_,      # Input: Integer specifying the size of the median filter neighborhood
        int color_,              # Input: Integer for color reduction value (maximum color levels)
        bint contour_,           # Input: Boolean to control whether to apply contour (edge detection)
        int flag_                # Input: Integer for blending flag (e.g., BLEND_RGB_ADD, BLEND_RGB_SUB, etc.)
):
    """
    Apply a cartoon effect to a pygame.Surface image by performing edge detection, 
    median filtering, and color reduction.

    The function processes the input surface with the following steps:
    - Optional edge detection using Sobel.
    - Median filter application for smoothing.
    - Color reduction to limit the number of colors.
    - Blending of edges and smoothed colors.
    
    Parameters:
    surface_ (object): A pygame.Surface compatible object (24 - 32 bit image).
    sobel_threshold_ (int): Threshold value for Sobel edge detection.
    median_kernel_ (int): Size of the median filter kernel (defines the neighborhood for smoothing).
    color_ (int): Maximum number of colors for color reduction.
    contour_ (bint): Boolean flag to apply contour (edge detection).
    flag_ (int): Flag for blending modes (e.g., BLEND_RGB_ADD, BLEND_RGB_SUB, BLEND_RGB_MULT, etc.).

    Returns:
    object: The resulting pygame.Surface object with the cartoon effect applied.
    """
    
    cdef unsigned char [:, :, :] cpu_array_

    try:
        # Convert the pygame surface to a 3D array (for RGB channels).
        cpu_array_ = surface_.get_view('3')
    except Exception as e:
        raise ValueError("\nCannot reference source pixels into a 3D array.\n %s " % e)

    try:
        # Convert the CPU array to a GPU array using CuPy.
        gpu_array_ = cp.asarray(cpu_array_)
    except Exception as e:
        raise ValueError("\nCannot reference source pixels into a 3D array.\n %s " % e)

    # Create a copy of the GPU array for processing
    gpu_array_copy = gpu_array_.copy()

    # Get the width and height of the image from the GPU array
    cdef:
        Py_ssize_t w, h
    w, h = gpu_array_.shape[0], gpu_array_.shape[1]

    # FIRST BRANCH: Apply edge detection and possibly Sobel filter

    # Perform Canny edge detection (in-place modification of the GPU array)
    canny_cupy_c(gpu_array_)

    # Optionally apply Sobel edge detection if contour_ is True
    if contour_:
        sobel_cupy_c(gpu_array_)

    # SECOND BRANCH: Apply median filter and color reduction

    # Apply a median filter to smooth the image (in-place modification on the copy)
    median_cupy_c(gpu_array_copy, median_kernel_)

    # Reduce the number of unique colors in the image (in-place modification)
    color_reduction_cupy_c(gpu_array_copy, color_)

    # Convert the processed GPU array back into a pygame surface and transpose the channels for correct orientation
    surface_ = frombuffer(gpu_array_copy.astype(dtype=cp.uint8).transpose(1, 0, 2).tobytes(), (w, h), "RGB")

    # BLEND both the edge-detected and smoothed color-reduced versions of the image
    surf = frombuffer(gpu_array_.astype(dtype=cp.uint8).transpose(1, 0, 2).tobytes(), (w, h), "RGB")

    # Perform the blending based on the specified flag (e.g., add, subtract, multiply, etc.)
    surface_.blit(surf, (0, 0), special_flags=flag_)

    # Return the final surface with the cartoon effect applied
    return surface_




alpha_blending_kernel = cp.ElementwiseKernel(
    'float32 r0, float32 g0, float32 b0, float32 a0, float32 r1, float32 g1, float32 b1, float32 a1',
    'uint8 rr, uint8 gg, uint8 bb, uint8 aa',
    '''
    float n = (1.0f - a0);
    
    rr = (unsigned char)((r0 + r1 * n) * 255.0f);
    gg = (unsigned char)((g0 + g1 * n) * 255.0f);
    bb = (unsigned char)((b0 + b1 * n) * 255.0f);
    aa = (unsigned char)((a0 + a1 * n) * 255.0f);
    __syncthreads();
    if (rr > 255) {rr = 255;}
    if (gg > 255) {gg = 255;}
    if (bb > 255) {bb = 255;}
    if (aa > 255) {aa = 255;}
   
    ''', 'alpha_blending_kernel'
)


# TODO INVESTIGATE
#  Can only create pre-multiplied alpha bytes if the surface has per-pixel alpha
@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef object blending_gpu(object source_, object destination_, float percentage_):
    """
    BLEND A SOURCE TEXTURE TOWARD A DESTINATION TEXTURE (TRANSITION EFFECT)

    * Video system must be initialised 
    * source & destination Textures must be same sizes
    * Compatible with 24 - 32 bit surface
    * Output create a new surface
    * Image returned is converted for fast blit (convert())

    # ***********************************
    # Calculations for alpha & RGB values
    # outA = SrcA + DstA(1 - SrcA)
    # outRGB = SrcRGB + DstRGB(1 - SrcA)
    # ***********************************
    
    :param source_     : pygame.Surface (Source)
    :param destination_: pygame.Surface (Destination)
    :param percentage_ : float; Percentage value between [0.0 ... 100.0]
    :return: return    : Return a 24 bit pygame.Surface and blended with a percentage
                         of the destination texture.
    """

    cdef:
        Py_ssize_t w, h
    w, h = source_.get_width(), source_.get_height()


    try:

        source_array = numpy.frombuffer(
            tostring(source_, "RGBA_PREMULT"), dtype=numpy.uint8)
        source_array = cp.asarray(source_array, dtype=cp.uint8)
        source_array = (source_array.reshape(w, h, 4)/<float>255.0).astype(dtype=float32)

    except Exception as e:
        raise ValueError("\nCannot reference source pixels into a 3d array.\n %s " % e)

    try:

        destination_array = numpy.frombuffer(
            tostring(destination_, "RGBA_PREMULT"), dtype=numpy.uint8)
        destination_array = cp.asarray(destination_array, dtype=cp.uint8)
        destination_array = (destination_array.reshape(w, h, 4) / <float>255.0).astype(dtype=float32)
    except Exception as e:
        raise ValueError("\nCannot reference destination pixels into a 3d array.\n %s " % e)
    cdef:
        out = cp.empty((w, h, 4), cp.uint8)


    r0, g0, b0, a0 = source_array[:, :, 0], source_array[:, :, 1], source_array[:, :, 2], source_array[:, :, 3]
    r1, g1, b1, a1 = destination_array[:, :, 0], destination_array[:, :, 1], \
                     destination_array[:, :, 2], destination_array[:, :, 3]


    out[:, :, 0], out[:, :, 1], out[:, :, 2], out[:, :, 3] = \
        alpha_blending_kernel(r0.astype(cupy.float32), g0.astype(cupy.float32),
                              b0.astype(cupy.float32), a0,
                              r1.astype(cupy.float32), g1.astype(cupy.float32), b1.astype(cupy.float32), a1)

    return frombuffer(out.astype(cp.uint8).tobytes(), (w, h), "RGBA").convert()




sharpen_kernel = cp.RawKernel(
    '''
    extern "C" 
    
    __constant__ double kernel[9]  = {0, -1, 0,-1, 5, -1, 0, -1, 0};
    
    __global__ void sharpen_kernel(double* buffer, int filter_size,
                     double* return_value)
    {
    double color=0;
    
    for (int i=0; i<filter_size; ++i){            
        color += buffer[i] * kernel[i];
        
    }

    if (color > 255.0f) {color = 255.0f;} 
    else if (color < 0.0f) {color = 0.0;}   
       
    return_value[0] = color;
   
    }
    ''',
    'sharpen_kernel'
)


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef sharpen_gpu(surface_):
    """
    Apply a sharpening filter to the given pygame.Surface image using a generic filter kernel.

    The function sharpens the input image by applying a sharpening kernel to each RGB channel
    (red, green, blue) independently. The sharpened image is returned as a 24-bit pygame.Surface 
    format.

    Parameters:
    surface_ (pygame.Surface): A pygame surface object that represents the input image to be sharpened.

    Returns:
    pygame.Surface: A new pygame surface with the sharpened image in 24-bit format (RGB).
    
    Raises:
    ValueError: If the input surface is not a valid pygame.Surface or has zero dimensions.
    """
    
    # Ensure the input is a valid pygame.Surface object
    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)

    # Try to convert the surface pixels into a 3D array (RGB channels)
    cdef unsigned char [:, :, :] cpu_array_
    try:
        cpu_array_ = surface_.get_view('3')  # Access the 3D view of the surface data (RGB channels)
    except Exception as e:
        raise ValueError("\nCannot reference source pixels into a 3d array.\n %s " % e)

    # Get the width and height of the image from the CPU array
    cdef:
        Py_ssize_t w, h
    w, h = cpu_array_.shape[0], cpu_array_.shape[1]

    # Validate that the surface has non-zero dimensions
    if w == 0 or h == 0:
        raise ValueError("Surface width and height cannot be null!")

    # Convert the CPU array into a GPU array for processing using CuPy
    gpu_array_ = cp.asarray(cpu_array_)

    # Separate the RGB channels for individual processing
    r = gpu_array_[:, :, 0]  # Red channel
    g = gpu_array_[:, :, 1]  # Green channel
    b = gpu_array_[:, :, 2]  # Blue channel

    # Apply the sharpening filter to each channel using a generic filter with a 3x3 kernel
    r = cupyx.scipy.ndimage.generic_filter(r, sharpen_kernel, 3).astype(dtype=cp.uint8)
    g = cupyx.scipy.ndimage.generic_filter(g, sharpen_kernel, 3).astype(dtype=cp.uint8)
    b = cupyx.scipy.ndimage.generic_filter(b, sharpen_kernel, 3).astype(dtype=cp.uint8)

    # Synchronize the GPU stream to ensure all operations are completed
    cp.cuda.Stream.null.synchronize()

    # Convert the processed GPU array back to a pygame.Surface object, transposing channels as needed
    return frombuffer(gpu_array_.transpose(1, 0, 2).tobytes(), (w, h), "RGB")




ripple_kernel = cp.RawKernel(
    r'''
    extern "C" __global__
    void ripple_kernel(
        float * current_, 
        const float * previous_, 
        unsigned char * bck_array, 
        const unsigned char * texture_array_,    
        const int w, const int h)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        int j = blockIdx.y * blockDim.y + threadIdx.y;
        
        // index for array shape (w, h)
        const int index = j * h + i;
        
        // index for array shape (w, h, 3) e.g texture & background 
        const int index1 = j * h * 3 + i * 3;
        
        // Maximum index value for array shape (w, h) 
        const int t_max = w * h;
        
        // Maximum index value for array shape (w, h, 3
        const int t_max_ = w * h * 3;
        
        __syncthreads();
        
        float left = 0.0f;
        float right = 0.0f; 
        float top = 0.0f; 
        float bottom = 0.0f;
        float data = 0.0f;
        
        // Limit the loop to the valid indexes 
        if (index> 0 && index < t_max){
            
            /*
            float data = (previous_[fmaxf((index + 1) % t_max, 0)] +                 // left  
                          previous_[fmaxf((index - 1) % t_max, 0)] +                 // right 
                          previous_[fmaxf((index - h) % t_max, 0)] +                 // top 
                          previous_[fmaxf((index + h) % t_max, 0)]) * (float)0.5;    // bottom
            */
            
            
            
            if ((index - h) < 0) {
                top = 0.0f;
            }
            else {
                top = previous_[index - h];
            }
            
            if ((index + h) > t_max) {
                bottom = 0.0f; 
            } 
              else {
                bottom = previous_[index + h];
            }
            
            
            if ((index - 1) < 0) {
                right = 0.0f;
            }
            else {
                right = previous_[index - 1];
            }
            
            if ((index + 1) > t_max) {
                left = 0.0f;
            } 
            else {
                left = previous_[index + 1];
            }
            
            
            data = (left + right + top + bottom) * 0.5f; 
            
            
              
            data = data - current_[index];
            data = data - (data * 0.01125f);   // Attenuation
             
            __syncthreads();
            
            current_[index] = data;
            
                
            data = 1.0f - data * 1.0f/1024.0f;
            const int w2 = w >> 1;
            const int h2 = h >> 1;
            const int a = fmaxf((int)(((i - w2) * data) + w2) % h, 0);              // texture index (x)
            const int b = fmaxf((int)(((j - h2) * data) + h2) % w, 0);              // texture index (y)
            // int ind = a * h * 3 + b * 3;   // inverse texture
            const int ind = b * h * 3 + a * 3;
            bck_array[index1       ] = texture_array_[ind       ];    // red
            bck_array[(index1 + 1) ] = texture_array_[(ind + 1) ];    // green 
            bck_array[(index1 + 2) ] = texture_array_[(ind + 2) ];    // blue
            
            __syncthreads();
        }
    }
    ''',
    'ripple_kernel'
)


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef ripple_effect_gpu(
       tuple grid,               # Grid dimensions for CUDA kernel (grid_y, grid_x)
       tuple block,              # Block dimensions for CUDA kernel (block_y, block_x)
       const Py_ssize_t w,       # Width of the texture and arrays
       const Py_ssize_t h,       # Height of the texture and arrays
       previous,                 # CuPy array (w, h) of float32; represents the previous state of the ripple effect
       current,                  # CuPy array (w, h) of float32; represents the current state of the ripple effect
       texture_array,            # CuPy array (w, h, 3) of uint8; contains RGB texture pixels (source texture)
       background_array          # CuPy array (w, h, 3) of uint8; destination texture to apply the ripple effect
):
    """
    Apply a water drop (ripple) effect to a texture using GPU acceleration.

    This function uses a CUDA kernel (`ripple_kernel`) to compute the ripple effect on the texture. 
    It works by manipulating the `previous` and `current` state of the ripple effect and applying 
    it to the texture. The effect is computed in parallel on the GPU, which makes it efficient for 
    handling large textures or simulations.

    The `grid` and `block` tuples specify the number of blocks and threads per block for the CUDA kernel.
    This function is compatible with 24-bit or 32-bit textures (which should be converted to 24-bit using `pygame.convert()`).

    Parameters:
    grid (tuple): Tuple defining the grid size for CUDA kernel execution, e.g., (25, 25).
                  The grid size should match the texture size for efficient parallel processing.
    block (tuple): Tuple defining the block size for CUDA kernel execution, e.g., (32, 32).
                   The maximum number of threads is 1024 (block_x * block_y).
    w (int): The width of the texture and arrays.
    h (int): The height of the texture and arrays.
    previous (cupy.ndarray): A CuPy array of shape (w, h), containing the previous state of the ripple effect (float32).
    current (cupy.ndarray): A CuPy array of shape (w, h), containing the current state of the ripple effect (float32).
    texture_array (cupy.ndarray): A CuPy array of shape (w, h, 3), containing the source texture in RGB (uint8).
    background_array (cupy.ndarray): A CuPy array of shape (w, h, 3), representing the background texture to apply 
    the effect to (uint8).
    
    Returns:
    tuple: A tuple containing two CuPy arrays, the updated `previous` and `current` states of the ripple effect.
    
    Example:
    result_previous, result_current = ripple_effect_gpu(grid, block, w, h, previous, current, texture_array, background_array)
    
    Notes:
    - The grid and block sizes must be chosen appropriately based on the texture size and available GPU resources.
    - The kernel computation is done on the GPU using CuPy for faster execution.
    """
    
    # Execute the ripple effect computation on the GPU using the CUDA kernel
    ripple_kernel(
        grid,                # Grid dimensions for the kernel
        block,               # Block dimensions for the kernel
        (current, previous, background_array, texture_array, w, h)  # Arguments for the kernel
    )
    
    # Synchronize the CUDA stream to ensure all GPU operations are completed before returning
    cp.cuda.Stream.null.synchronize()

    # Return the updated previous and current arrays
    return previous, current




# Sharpen kernel (different method)
sharpen1_kernel = cp.RawKernel(
    r'''
    extern "C" __global__
    void sharpen1_kernel(float * current_, float * previous_, 
    const int w, const int h)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        int j = blockIdx.y * blockDim.y + threadIdx.y;

        const int col = h * 3;

        // index for array shape (w, h)
        const int index = j * h + i;

        // index for array shape (w, h, 3) e.g texture & background 
        const int index1 = j * col + i * 3;

        // Maximum index value for array shape (w, h) 
        const int t_max = w * h;

        // Maximum index value for array shape (w, h, 3
        const int t_max_ = w * col;
        
        
        __syncthreads();
        
        float red   = 0;
        float green = 0;
        float blue  = 0;

        // Limit the loop to the valid indexes 
        if (index > 0 && index < t_max){
            
            if ((index1 - col> 0) && (index1 + col < t_max_)) {
            
            red = 
                        -previous_[index1 - col          ] +                               
                        -previous_[index1 - 3            ] +          
                         previous_[index1                ] * 5.0f  +          
                        -previous_[index1 + 3            ] +                           
                        -previous_[index1 + col          ];          
                         
            green =  
                        -previous_[index1 - col  + 1     ] +                                 
                        -previous_[index1 - 2            ] +         
                         previous_[index1 + 1            ] * 5.0f  +          
                        -previous_[index1 + 4            ] +                           
                        -previous_[index1 + col + 1      ];        
                                             
            blue =         
                        -previous_[index1 - col + 2     ] +                                  
                        -previous_[index1 - 1           ] +          
                         previous_[index1 + 2           ] * 5.0f  +          
                        -previous_[index1 + 5           ] +                         
                        -previous_[index1 + col + 2     ];                     
            }
            
            __syncthreads();     
                       
            if (red > 255) { red = 255; } 
            if (red < 0) { red = 0; }
            
            if (green > 255) { green = 255; } 
            if (green < 0) { green = 0; }
            
            if (blue > 255) { blue = 255; } 
            if (blue < 0) { blue = 0; }
                  
            current_[ index1     ] = red;
            current_[ index1 + 1 ] = green;
            current_[ index1 + 2 ] = blue;
            
            
            __syncthreads();
            
        }
    }
    ''',
    'sharpen1_kernel'
)


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef object sharpen1_gpu(object surface_, grid_, block_):
    """
    SHARPEN AN IMAGE (RAWKERNEL)
    
    Different method, use a raw kernel to sharp the image
    The borders are not compute with the kernel (value =0)    
    
    :param surface_          : pygame.Surface  
    :param grid_             : tuple; grid values (grid_y, grid_x) e.g (25, 25). The grid values and block values must 
        match the texture and array sizes. 
    :param block_            : tuple; block values (block_y, block_x) e.g (32, 32). Maximum threads is 1024.
        Max threads = block_x * block_y
    :return           : pygame.Surface
    """

    if not PyObject_IsInstance(surface_, pygame.Surface):
        raise TypeError(f"\nArgument `surface_` must be a pygame.Surface type, got {type(surface_)}.")

    cdef unsigned char [:, :, :] cpu_array_
    try:
        cpu_array_ = surface_.get_view('3')

    except Exception as e:
        raise ValueError("\nCannot reference source pixels into a 3d array.\n %s " % e)

    cdef:
        Py_ssize_t w, h
    w, h = (<object>cpu_array_).shape[:2]

    if w == 0 or h == 0:
        raise ValueError("Surface `w` or `h` cannot be null!")

    gpu_array_ = cp.asarray(cpu_array_)

    cdef destination = cupy.empty((w, h, 3), cp.float32)

    sharpen1_kernel(
        grid_,
        block_,
        (destination, gpu_array_.astype(dtype=cp.float32), w, h))

    cp.cuda.Stream.null.synchronize()

    return frombuffer(destination.astype(
        dtype=cupy.uint8).transpose(1, 0, 2).tobytes(), (w, h), "RGB")


# todo format_=False trigger image pixels variation/displacement
@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef mirroring_gpu(
    surface_,
    tuple grid_,
    tuple block_,
    bint format_ = 0
):
    """
    Applies a mirror effect to an image using GPU acceleration.
    
    This function creates a mirrored version of an image represented as a `pygame.Surface`.  
    It supports 32-bit and 24-bit image formats and outputs a 24-bit format image.  
    The mirror orientation (horizontal or vertical) is controlled using the `format_` parameter.
    
    ### Parameters:
    - **surface_** (*pygame.Surface*):  
      The input image as a `pygame.Surface` object.
    
    - **grid_** (*tuple*):  
      Defines the grid dimensions as `(grid_y, grid_x)`, e.g., `(25, 25)`.  
      The grid and block dimensions must match the texture and array sizes.
    
    - **block_** (*tuple*):  
      Specifies the block dimensions as `(block_y, block_x)`, e.g., `(32, 32)`.  
      The maximum number of threads per block is 1024, following the condition:  
      `Max threads = block_x * block_y`.
    
    - **format_** (*bool*, optional, default=`False`):  
      Determines the mirror orientation:  
      - `False` (0) → Horizontal mirroring (default).  
      - `True` (1) → Vertical mirroring.  
    
    ### Returns:
    - (*pygame.Surface*):  
      A 24-bit `pygame.Surface` with the applied mirror effect.
    
    ### Raises:
    - **TypeError**: If `surface_` is not a `pygame.Surface` object.  
    - **ValueError**: If `surface_` has an invalid format or dimensions (`w` or `h` is 0).  
    - **ValueError**: If `surface_` cannot be referenced as a 3D array.  
    
    ### Notes:
    - The function extracts pixel data from `surface_` and converts it into a 3D array.  
    - The `mirroring_cupy` function performs GPU-accelerated mirroring and returns the processed image.  
    """


    if not PyObject_IsInstance(surface_, pygame.Surface):
        raise TypeError(f"\nArgument `surface_` must be a pygame.Surface type, got {type(surface_)}.")

    cdef unsigned char [:, :, :] cpu_array_

    try:
        cpu_array_ = surface_.get_view('3')

    except Exception as e:
        raise ValueError("\nCannot reference `surface_` pixels into a 3d array.\n %s " % e)

    cdef:
        Py_ssize_t w, h

    w, h = (<object>cpu_array_).shape[:2]

    if w == 0 or h == 0:
        raise ValueError("Surface `w` or `h` cannot be null!")

    return mirroring_cupy(w, h, cp.asarray(cpu_array_), grid_, block_, format_)




mirror_kernel = cp.RawKernel(
    r'''
    extern "C" __global__
    void mirror_kernel(
        float * current_,
        const float * previous_,
        const int w,
        const int h,
        bool format
        )
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        int j = blockIdx.y * blockDim.y + threadIdx.y;
        

        const int col = h * 3;

        // index for array shape (w, h, 3) e.g texture & background
        const int index1 = j * col + i * 3;

        

        float red   = 0;
        float green = 0;
        float blue  = 0;
        int x2, x3;
        
        red   = previous_[index1    ];
        green = previous_[index1 + 1];
        blue  = previous_[index1 + 2];
        
        __syncthreads();
          
       
        if (format == 1){

        x2 = i >> 1;
        current_[j * col + x2 * 3    ] = red;
        current_[j * col + x2 * 3 + 1] = green;
        current_[j * col + x2 * 3 + 2] = blue;

        x3 = h - x2 - 1;
        current_[j * col + x3 * 3    ] = red;
        current_[j * col + x3 * 3 + 1] = green;
        current_[j * col + x3 * 3 + 2] = blue;
        }
        
        if (format == 0){
        x2 = j >> 1;
        current_[x2 * h * 3 + i * 3    ] = red;
        current_[x2 * h * 3 + i * 3 + 1] = green;
        current_[x2 * h * 3 + i * 3 + 2] = blue;

        x3 = w - x2 - 1;
        current_[x3 * h * 3 + i * 3    ] = red;
        current_[x3 * h * 3 + i * 3 + 1] = green;
        current_[x3 * h * 3 + i * 3 + 2] = blue;
        }
        
        __syncthreads();
    }
    ''',
    'mirror_kernel'
)


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef inline mirroring_cupy(
    const Py_ssize_t w,
    const Py_ssize_t h,
    gpu_array_,
    tuple grid_,
    tuple block_,
    bint format_=0):
    """
    Performs GPU-accelerated mirroring of an image using CuPy.

    This function applies a mirror effect (horizontal or vertical) to an image stored  
    as a CuPy array. The processing is handled by a CUDA kernel for high efficiency.

    ### Parameters:
    - **w** (*Py_ssize_t*):  
      Width of the image.

    - **h** (*Py_ssize_t*):  
      Height of the image.

    - **gpu_array_** (*cupy.ndarray*):  
      Input image as a CuPy array with shape `(h, w, 3)` and dtype `uint8`.

    - **grid_** (*tuple*):  
      Defines the grid dimensions as `(grid_y, grid_x)`, e.g., `(25, 25)`.  
      The grid and block dimensions must match the texture and array sizes.

    - **block_** (*tuple*):  
      Specifies the block dimensions as `(block_y, block_x)`, e.g., `(32, 32)`.  
      The maximum number of threads per block is 1024, following the condition:  
      `Max threads = block_x * block_y`.

    - **format_** (*bool*, optional, default=`False`):  
      Determines the mirror orientation:  
      - `False` (0) → Horizontal mirroring (default).  
      - `True` (1) → Vertical mirroring.  

    ### Returns:
    - (*tuple*):  
      A buffer containing the processed image data in 24-bit RGB format with dimensions `(w, h)`.  

    ### Notes:
    - A new CuPy array (`destination`) is allocated to store the mirrored image in `float32` format.  
    - The `mirror_kernel` CUDA function is invoked with the provided grid and block sizes.  
    - The function ensures synchronization with `cp.cuda.Stream.null.synchronize()` and  
      `cp.cuda.Device(0).synchronize()` to complete all GPU operations before returning.  
    - The output is converted back to an 8-bit RGB format before returning.  
    """

    destination = cupy.empty((w, h, 3), cupy.float32)

    mirror_kernel(
        grid_,
        block_,
        (destination, gpu_array_.astype(dtype=cp.float32), w, h, format_))

    cp.cuda.Stream.null.synchronize()
    cp.cuda.Device(0).synchronize()

    return frombuffer(
        destination.astype(dtype=cupy.uint8).transpose(1, 0, 2).tobytes(),
        (w, h),
        "RGB"
    )






@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef saturation_gpu(
    surface_,
    tuple grid_,
    tuple block_,
    float val_ = 1.0
):

    """
    Adjusts the saturation level of a given image using GPU acceleration.
    
    This function modifies the saturation of an image represented as a `pygame.Surface`.  
    It supports 32-bit and 24-bit image formats, producing an output in 24-bit format.  
    The saturation adjustment value must be within the range [-1.0, 1.0].  
    
    ### Parameters:
    - **surface_** (*pygame.Surface*):  
      The input image as a `pygame.Surface` object.
      
    - **grid_** (*tuple*):  
      Defines the grid dimensions as `(grid_y, grid_x)`, e.g., `(25, 25)`.  
      The grid and block dimensions must match the texture and array sizes.
    
    - **block_** (*tuple*):  
      Specifies the block dimensions as `(block_y, block_x)`, e.g., `(32, 32)`.  
      The maximum number of threads per block is 1024, following the condition:  
      `Max threads = block_x * block_y`.
    
    - **val_** (*float*, optional, default=`1.0`):  
      Saturation adjustment level in the range `[-1.0, 1.0]`.  
      - `-1.0` → Completely desaturated (grayscale).  
      - `0.0` → No change in saturation.  
      - `1.0` → Fully saturated.  
    
    ### Returns:
    - (*pygame.Surface*):  
      A 24-bit `pygame.Surface` with the adjusted saturation level.
    
    ### Raises:
    - **TypeError**: If `surface_` is not a `pygame.Surface` object.  
    - **ValueError**: If `surface_` has an invalid format or dimensions (`w` or `h` is 0).  
    - **ValueError**: If `val_` is outside the range `[-1.0, 1.0]`.  
    """


    if not PyObject_IsInstance(surface_, pygame.Surface):
        raise TypeError(f"\nArgument `surface_` must be a pygame.Surface type, got {type(surface_)}.")

    cdef unsigned char [:, :, :] cpu_array_

    try:
        cpu_array_ = surface_.get_view('3')

    except Exception as e:
        raise ValueError("\nCannot reference `surface_` pixels into a 3d array.\n %s " % e)

    cdef:
        Py_ssize_t w, h

    w, h = (<object>cpu_array_).shape[:2]

    if w == 0 or h == 0:
        raise ValueError("Surface `w` or `h` cannot be null!")

    if val_ < -1.0 or val_ > 1.0:
        raise ValueError(f"\nArgument `val_` must be in range [-1.0 ... 1.0] got {val_}.")

    return saturation_cupy(w, h, cp.asarray(cpu_array_), grid_, block_, val_)




saturation_kernel = cp.RawKernel(

    r'''
    extern "C" __global__
    void saturation_kernel(
        const float * source, 
        float * destination, 
        const int width, 
        const int height, 
        const double val_)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        int j = blockIdx.y * blockDim.y + threadIdx.y;

        const int index  = j * height + i;
        const int index1 = j * height * 3 + i * 3;     
        const int t_max  = width * height;
        
        float h, s, v;
        int ii = 0; 
        float f, p, q, t;
        float mx, mn;

        __syncthreads();
        
        float red   = 0.0f;
        float green = 0.0f;
        float blue  = 0.0f;

        if (index > 0 && index < t_max) {
       

        red   = source[index1    ] / 255.0f;       
        green = source[index1 + 1] / 255.0f;
        blue  = source[index1 + 2] / 255.0f;    
        
        
    
        // Find max and min of RGB values 
        if (red > green){
            if (red > blue){
                mx = red;
                if (blue > green){ mn = green;}
                else mn = blue;
            }
            else{
                mx = blue;
                if (blue > green){ mn = green;}
                else mn = blue;
            }
        }
        else{
            if (green > blue){
                mx = green;
                if (blue > red){ mn = red;}
                else mn = blue;
            } 
            else{
                mx = blue;
                if (blue > red) { mn = red;}
                else  mn = blue;
            }
        }  
            
            
        // Convert RGB to HSV 
        float df = mx-mn;  
        float df_ = 1.0f/df;        
    
        if (mx == mn)
        { h = 0.0;}
    
        else if (mx == red){
            h = (float)fmodf(60.0f * ((green-blue) * df_) + 360.0, 360);
        }
        else if (mx == green){
            h = (float)fmodf(60.0f * ((blue-red) * df_) + 120.0, 360);
        }
        else if (mx == blue){
            h = (float)fmodf(60.0f * ((red-green) * df_) + 240.0, 360);
        }
    
        if (mx == 0.0){
            s = 0.0;
        }
        else{
            s = df/mx;
        }
    
        v = mx;   
        h = h * 1.0f/360.0f;
    
    
        s = fmaxf(s + (float)val_, 0.0f);
        s = fminf(s, 1.0f);    
          
        __syncthreads();
    
    
        // Convert HSV to RGB    
        if (s == 0.0){
             destination[index1    ] = v;
             destination[index1 + 1] = v;
             destination[index1 + 2] = v;         
             }
        else {
            ii = (int)(h*6.0f);
            f = (h * 6.0f) - ii;
            p = v*(1.0f - s);
            q = v*(1.0f - s * f);
            t = v*(1.0f - s * (1.0f - f));
            ii = ii%6;
    
            switch(ii) { 
                case 0:
                    destination[index1    ] = v;
                    destination[index1 + 1] = t;
                    destination[index1 + 2] = p;
                    break; 
                case 1: 
                    destination[index1    ] = q; 
                    destination[index1 + 1] = v;
                    destination[index1 + 2] = p;
                    break;
                case 2:
                    destination[index1    ] = p;
                    destination[index1 + 1] = v;
                    destination[index1 + 2] = t;
                    break;
                case 3:
                    destination[index1    ] = p;
                    destination[index1 + 1] = q;
                    destination[index1 + 2] = v;
                    break;
                case 4:
                    destination[index1    ] = t;
                    destination[index1 + 1] = p;
                    destination[index1 + 2] = v;
                    break;
                case 5: 
                    destination[index1    ] = v;
                    destination[index1 + 1] = p; 
                    destination[index1 + 2] = q;
                    break;
                default: 
                    destination[index1    ] = red ;
                    destination[index1 + 1] = green; 
                    destination[index1 + 2] = blue;
        } //switch
        } //else
        __syncthreads();     
    
    
        destination[index1    ] = fminf(destination[index1    ] * 255.0f, 255.0f);
        destination[index1 + 1] = fminf(destination[index1 + 1] * 255.0f, 255.0f);
        destination[index1 + 2] = fminf(destination[index1 + 2] * 255.0f, 255.0f);
  
        } // if (index >0
        
        
        
    } // main
    ''',
    'saturation_kernel'
)



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef inline saturation_cupy(
    const Py_ssize_t w,
    const Py_ssize_t h,
    cupy_array,
    tuple grid_,
    tuple block_,
    float val_ = 1.0
):
    """
    Performs GPU-accelerated saturation adjustment using CuPy.

    This function applies a saturation transformation to an image represented as a CuPy array.  
    It uses a CUDA kernel to process the image in parallel and returns a new image buffer  
    with the adjusted saturation level.

    ### Parameters:
    - **w** (*Py_ssize_t*):  
      Width of the image.

    - **h** (*Py_ssize_t*):  
      Height of the image.

    - **cupy_array** (*cupy.ndarray*):  
      Input image as a CuPy array with shape `(h, w, 3)` and dtype `uint8`.  

    - **grid_** (*tuple*):  
      Defines the grid dimensions as `(grid_y, grid_x)`, e.g., `(25, 25)`.  
      The grid and block dimensions must match the texture and array sizes.

    - **block_** (*tuple*):  
      Specifies the block dimensions as `(block_y, block_x)`, e.g., `(32, 32)`.  
      The maximum number of threads per block is 1024, following the condition:  
      `Max threads = block_x * block_y`.

    - **val_** (*float*, optional, default=`1.0`):  
      Saturation adjustment level in the range `[-1.0, 1.0]`.  
      - `-1.0` → Completely desaturated (grayscale).  
      - `0.0` → No change in saturation.  
      - `1.0` → Fully saturated.  

    ### Returns:
    - (*tuple*):  
      A buffer containing the processed image data in 24-bit RGB format with dimensions `(w, h)`.  

    ### Notes:
    - The function creates an empty destination array of dtype `float32` to store the result.  
    - The `saturation_kernel` CUDA function is invoked with the provided grid and block sizes.  
    - The computation is synchronized using `cp.cuda.Stream.null.synchronize()` to ensure  
      that GPU operations complete before returning.  
    - The final output is transposed and converted back to an 8-bit RGB format.  
    """

    destination = cupy.empty((w, h, 3), dtype=cupy.float32)

    saturation_kernel(
        grid_,
        block_,
        (cupy_array.astype(cupy.float32), destination, w, h, val_))

    cp.cuda.Stream.null.synchronize()

    return frombuffer(
        destination.astype(cp.uint8).transpose(1, 0, 2).tobytes(),
        (w, h),
        "RGB"
    )





@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef bilateral_gpu(surface_, const unsigned int kernel_size_):

    """
    Apply a bilateral filter to a 32-bit or 24-bit image using the GPU.

    A bilateral filter is a non-linear, edge-preserving, and noise-reducing 
    smoothing filter. It replaces the intensity of each pixel with a weighted 
    average of nearby pixels, where the weights are determined by both spatial 
    proximity and intensity differences, following a Gaussian distribution.

    The filter parameters (sigma_r & sigma_s) are pre-defined within the GPU kernel.
    This function is compatible with 32-bit and 24-bit images.

    :param surface_: pygame.Surface
        The input image as a pygame surface.
        
    :param kernel_size_: int
        The kernel size, determining the number of neighboring pixels included in the calculation.
    
    :return: pygame.Surface
        A new 24-bit pygame.Surface with the bilateral filtering effect applied.

    :raises TypeError: If `surface_` is not an instance of `pygame.Surface`.
    :raises ValueError: If the surface dimensions are zero or if `kernel_size_` is negative.
    
    """

    if not PyObject_IsInstance(surface_, pygame.Surface):
        raise TypeError(f"\nArgument `surface_` must be a pygame.Surface type, got {type(surface_)}.")

    cdef unsigned char [:, :, :] cpu_array_

    try:
        cpu_array_ = surface_.get_view('3')

    except Exception as e:
        raise ValueError("\nCannot reference `surface_` pixels into a 3d array.\n %s " % e)

    cdef:
        Py_ssize_t w, h

    w, h = (<object>cpu_array_).shape[:2]

    if w == 0 or h == 0:
        raise ValueError("Surface `w` or `h` cannot be null!")

    if kernel_size_ <= 0:
        raise ValueError("\nArgument `kernel_size_` cannot be <= 0")

    return bilateral_cupy(w, h, cp.asarray(cpu_array_), kernel_size_)




bilateral_kernel = cp.RawKernel(
    '''
    extern "C" __global__
    void bilateral_kernel(const double* source, int filter_size, double* destination)
    {

    // float sigma_i (also call sigma_r) range kernel, 
    // minimum amplitude of an edge.
    const float sigma_i2 = 80.0f * 80.0f * 2.0f;   

    // float sigma_s : Spatial extent of the kernel, size of the 
    // considered neighborhood
    const float sigma_s2 = 16.0f * 16.0f * 2.0f;  

    double ir = 0.0; 
    float wpr = 0.0;  
    double r=0.0; 
    float dist=0.0;
    float gs=0.0;  
    float vr=0.0;
    float wr=0.0;  
    const float p = 3.14159265;
    const int k2 = (int)sqrtf((float)filter_size);

    int a = 0;

    __syncthreads();

    for (int ky = 0; ky < k2; ++ky)
    {
        for (int kx = 0; kx < k2; ++kx)
        {    

            dist = (float)sqrtf((float)kx * (float)kx + (float)ky * (float)ky);
            gs = ((float)1.0 / (p * sigma_s2)) * (float)exp(-(dist * dist ) / sigma_s2);             
            r = source[a];
            vr = r - (double)source[filter_size >> 1];
            wr = ((float)1.0 / (p * sigma_i2)) * (float)exp(-(vr * vr ) / sigma_i2);
            wr = wr *  gs;

            ir = ir + r * wr;          
            wpr = wpr + wr;
            a += 1;

        } // for
    } // for

    __syncthreads();

    ir = ir / wpr;
    if (ir > 255.0) {ir = 255.0;}
    if (ir < 0.0) {ir = 0.0;} 
    destination[0] = ir;

    __syncthreads();

    } //main

    ''',
    'bilateral_kernel'
)
    
    
    

@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef inline bilateral_cupy(
    const Py_ssize_t w,
    const Py_ssize_t h,
    gpu_array_,
    const unsigned int kernel_size_
):
    """
    Apply a bilateral filter to an image using CuPy.

    This function applies a bilateral filter to an image stored as a GPU-based CuPy array. 
    A bilateral filter smooths an image while preserving edges by considering both 
    spatial distance and intensity differences when averaging neighboring pixels.

    :param w: int
        The width of the image.
        
    :param h: int
        The height of the image.
        
    :param gpu_array_: cupy.ndarray
        A 3D CuPy array of shape (height, width, 3), representing an RGB image.
        
    :param kernel_size_: int
        The size of the kernel (defines the neighborhood of pixels included in the calculation).
    
    :return: pygame.Surface
        A new 24-bit pygame.Surface containing the filtered image.

    :raises ValueError: If the input array has an invalid shape.
    """

    r = gpu_array_[:, :, 0].astype(dtype=cupy.float32)
    g = gpu_array_[:, :, 1].astype(dtype=cupy.float32)
    b = gpu_array_[:, :, 2].astype(dtype=cupy.float32)

    bilateral_r = cupyx.scipy.ndimage.generic_filter(
        r, bilateral_kernel, kernel_size_).astype(dtype=cp.uint8)

    bilateral_g = cupyx.scipy.ndimage.generic_filter(
        g, bilateral_kernel, kernel_size_).astype(dtype=cp.uint8)

    bilateral_b = cupyx.scipy.ndimage.generic_filter(
        b, bilateral_kernel, kernel_size_).astype(dtype=cp.uint8)

    gpu_array_[:, :, 0], \
    gpu_array_[:, :, 1], \
    gpu_array_[:, :, 2] = bilateral_r, bilateral_g, bilateral_b

    cp.cuda.Stream.null.synchronize()

    return frombuffer(
        gpu_array_.astype(cp.uint8).transpose(1, 0, 2).tobytes(),
        (w, h),
        "RGB"
    )





@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef emboss5x5_gpu(surface_):
    """
    Apply an emboss effect to a 32-bit or 24-bit image using a 3x3 convolution kernel.

    The embossing kernel used:
        [-2.0, -1.0,  0.0]
        [-1.0,  1.0,  1.0]
        [ 0.0,  1.0,  2.0]

    Each RGB channel of the input image is processed independently using convolution.
    The input image must be in 32-bit or 24-bit format, and the output will be a 24-bit image.

    :param surface_: pygame.Surface
        A pygame surface representing the input image.
        
    :return: pygame.Surface
        A new 24-bit pygame.Surface with the emboss effect applied.

    :raises TypeError: If `surface_` is not an instance of `pygame.Surface`.
    :raises ValueError: If the surface's pixel data cannot be accessed or if its width/height is zero.
    
    """

    if not PyObject_IsInstance(surface_, pygame.Surface):
        raise TypeError(f"\nArgument `surface_` must be a pygame.Surface type, got {type(surface_)}.")

    cdef unsigned char [:, :, :] cpu_array_

    try:
        cpu_array_ = surface_.get_view('3')

    except Exception as e:
        raise ValueError("\nCannot reference `surface_` pixels into a 3d array.\n %s " % e)

    cdef:
        Py_ssize_t w, h
    w, h = (<object>cpu_array_).shape[:2]

    if w == 0 or h == 0:
        raise ValueError("Surface `w` and `h` cannot be null!")

    return emboss5x5_cupy(w, h, cp.asarray(cpu_array_))




emboss_kernel = cp.RawKernel(
    '''   
    extern "C" __global__
    
    __constant__ double k[9] = {-2.0, -1.0, 0.0, -1.0, 1.0, 1.0, 0.0, 1.0, 2.0};

    void sobel_kernel(
        const double* buffer, 
        int filter_size,
        double* return_value)
    {

    double color = 0.0; 

    for (int i=0; i<9; ++i){
        color += buffer[i] * k[i];
    }

    if (color > 255.0) {color = 255.0;}
    if (color < 0.0) {color = 0.0;}
    return_value[0] = color;

    }
    ''',
    'sobel_kernel'
)


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef inline emboss5x5_cupy(
    const Py_ssize_t w,
    const Py_ssize_t h,
    gpu_array_):

    """
    Apply an emboss effect to an image using a 3x3 convolution kernel with CuPy.

    This function performs an embossing effect by applying the following 3x3 kernel 
    to each RGB channel of a GPU-based image array:

        [-2.0, -1.0,  0.0]
        [-1.0,  1.0,  1.0]
        [ 0.0,  1.0,  2.0]

    The input image must be stored as a CuPy array in 32-bit or 24-bit format.
    The output is a 24-bit image.
    
    :param w: int
        The width of the image.
        
    :param h: int
        The height of the image.
        
    :param gpu_array_: cupy.ndarray
        A 3D CuPy array representing the image in (height, width, RGB) format.

    :return: pygame.Surface
        A new 24-bit pygame.Surface with the emboss effect applied.

    :raises ValueError: If the input array has an invalid shape.
    """

    r = gpu_array_[:, :, 0].astype(dtype=cupy.float32)
    g = gpu_array_[:, :, 1].astype(dtype=cupy.float32)
    b = gpu_array_[:, :, 2].astype(dtype=cupy.float32)

    emboss_r = cupyx.scipy.ndimage.generic_filter(
        r, emboss_kernel, 3).astype(dtype=cp.uint8)

    emboss_g = cupyx.scipy.ndimage.generic_filter(
        g, emboss_kernel, 3).astype(dtype=cp.uint8)

    emboss_b = cupyx.scipy.ndimage.generic_filter(
        b, emboss_kernel, 3).astype(dtype=cp.uint8)

    gpu_array_[:, :, 0], \
    gpu_array_[:, :, 1], \
    gpu_array_[:, :, 2] = emboss_r, emboss_g, emboss_b

    cp.cuda.Stream.null.synchronize()

    return frombuffer(
        gpu_array_.astype(cp.uint8).transpose(1, 0, 2).tobytes(),
        (w, h),
        "RGB"
    )


# ---------------------------------------- LIGHT ---------------------------------------------------


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef tuple area24_gpu(
        const int x,
        const int y,
        rgb_array,
        mask_alpha,
        float intensity=<float>1.0,
        color=cupy.asarray([128.0, 128.0, 128.0], dtype=cupy.float32)
):
    """
    Applies a color overlay to a specified area in a 24-bit RGB image using GPU acceleration.

    This function modifies the color of a region in the `rgb_array` at coordinates `(x, y)`,  
    blending it with the given `color` based on the `mask_alpha` and `intensity` parameters.  
    It utilizes CuPy for fast GPU-based computation.

    :param x:  
        int; X-coordinate of the target pixel in the image.

    :param y:  
        int; Y-coordinate of the target pixel in the image.

    :param rgb_array:  
        CuPy ndarray; A 3D array representing the RGB image stored in GPU memory.  
        Must have a shape of `(height, width, 3)`.

    :param mask_alpha:  
        CuPy ndarray; Alpha mask determining the transparency effect applied to the color overlay.  
        Must match the shape of `rgb_array`.

    :param intensity:  
        float; Scaling factor for the overlay effect.  
        Determines how strongly the `color` blends with the original image.  
        Must be in the range `[0.0, 1.0]`. Default is `1.0`.

    :param color:  
        CuPy ndarray; A 3-element array representing the RGB color to be blended.  
        Default is `[128.0, 128.0, 128.0]` (mid-gray).

    :return:  
        tuple; The modified RGB values at `(x, y)` after applying the effect.

    :raises ValueError: If `x` or `y` is out of bounds.  
    :raises ValueError: If `intensity` is not within `[0.0, 1.0]`.  
    :raises ValueError: If `rgb_array` or `mask_alpha` have incompatible shapes.
    """

    if not isinstance(rgb_array, cupy.ndarray):
        raise TypeError(
            f"Argument `rgb_array` must be type cupy.ndarray got {type(rgb_array)}.")

    if not isinstance(mask_alpha, cupy.ndarray):
        raise TypeError(
            f"Argument `mask_alpha` must be type cupy.ndarray got {type(mask_alpha)}.")

    if intensity <= 0.0:
        raise ValueError(f"\nIntensity value cannot be < 0.0 got {intensity}")


    cdef unsigned int w, h, lx, ly, ax, ay

    try:
        w, h = rgb_array.shape[:2]

    except (ValueError, AttributeError) as e:
        raise ValueError('\nArray `rgb_array` shapes not understood.')

    try:
        ax, ay = (<object>mask_alpha).shape[:2]

    except (ValueError, pygame.error) as e:
        raise ValueError('\nArray `mask_alpha` shapes not understood.')

    # Return an empty surface if the x or y are not within the normal range.
    if (x < 0) or (x > w - 1) or (y < 0) or (y > h - 1):
        return Surface((ax, ay), SRCALPHA), ax, ay

    # return an empty Surface when intensity == 0.0
    if intensity == 0.0:
        return Surface((ax, ay), SRCALPHA), ax, ay

    lx = ax >> 1
    ly = ay >> 1

    cdef:
        unsigned int w_low = lx
        unsigned int w_high = lx
        unsigned int h_low = ly
        unsigned int h_high = ly

    rgb = cupy.empty((ax, ay, 3), cupy.uint8, order='C')
    alpha = cupy.empty((ax, ay), cupy.uint8, order='C')


    if x < lx:
        w_low = x
    elif x > w - lx:
        w_high = w - x

    if y < ly:
        h_low = y
    elif y >  h - ly:
        h_high = h - y


    rgb = rgb_array[ y - h_low:y + h_high, x - w_low:x + w_high, :]
    alpha = mask_alpha[ly - h_low:ly + h_high, lx - w_low:lx + w_high]

    ax, ay = rgb.shape[:2]

    new_array = cupy.empty((ax, ay, 3), cupy.uint8)

    f = cupy.multiply(alpha, <float>ONE_255 * intensity, dtype=cupy.float32)

    new_array[:, :, 0] = cupy.minimum(rgb[:, :, 0] * f, 255).astype(dtype=cupy.uint8)
    new_array[:, :, 1] = cupy.minimum(rgb[:, :, 1] * f, 255).astype(dtype=cupy.uint8)
    new_array[:, :, 2] = cupy.minimum(rgb[:, :, 2] * f, 255).astype(dtype=cupy.uint8)


    surface = pygame.image.frombuffer(new_array.tobytes(), (ay, ax), "RGB")

    return surface, ay, ax





@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef brightness_gpu(
        surface_,
        const float brightness_factor,
        tuple grid_ ,
        tuple block_
):
    """
    Adjusts the brightness of a given image using GPU acceleration.

    This function modifies the brightness of a `pygame.Surface` by applying a  
    GPU-accelerated transformation. It efficiently processes 24-bit and 32-bit  
    images, leveraging CUDA for parallel computation.

    :param surface_:  
        pygame.Surface; The input image to be processed.  
        Must be in 24-bit or 32-bit format.

    :param brightness_factor:  
        float; The brightness scaling factor.  
        - Values greater than `1.0` increase brightness.  
        - Values between `0.0` and `1.0` decrease brightness.  
        - `1.0` keeps the original brightness.  

    :param grid_:  
        tuple (int, int); Grid dimensions `(grid_y, grid_x)`, e.g., `(25, 25)`.  
        Defines how the image is divided for parallel GPU execution.  
        Must match the texture and array sizes.

    :param block_:  
        tuple (int, int); Block dimensions `(block_y, block_x)`, e.g., `(32, 32)`.  
        Determines the number of threads per block for CUDA execution.  
        The product `block_x * block_y` must not exceed `1024` (CUDA's max thread limit per block).

    :return:  
        pygame.Surface; A new surface with modified brightness format 24-bit. 

    :raises TypeError: If `surface_` is not of type `pygame.Surface`.  
    :raises ValueError: If `brightness_factor` is negative.  
    """

    if not PyObject_IsInstance(surface_, pygame.Surface):
        raise TypeError(f"\nArgument `surface_` must be a pygame.Surface type, got {type(surface_)}.")

    cdef unsigned char [:, :, :] cpu_array_

    try:
        cpu_array_ = surface_.get_view('3')

    except Exception as e:
        raise ValueError("\nCannot reference `surface_` pixels into a 3d array.\n %s " % e)

    cdef:
        Py_ssize_t w, h

    w, h = cpu_array_.shape[:2]

    if w == 0 or h == 0:
        raise ValueError("Surface `w` or `h` cannot be null!")

    if brightness_factor<-1 or brightness_factor>1.0:
        raise ValueError(f"\nArgument `brightness_factor` "
                         f"must be in range [-1.0 ... 1.0] got %{brightness_factor}.")

    return brightness_cupy(cp.asarray(cpu_array_).astype(
        dtype=cp.float32), grid_, block_, brightness_factor, w, h)


brightness_cuda = r'''

  
    struct hsl{
        float h;    // hue
        float s;    // saturation
        float l;    // value
    };
    
    struct rgb{
    float r;
    float g;
    float b;
    };
    
    __device__ struct rgb struct_hsl_to_rgb(const float h, const float s, const float l);
    __device__ struct hsl struct_rgb_to_hsl(const float r, const float g, const float b);
    __device__ float hue_to_rgb(const float m1, const float m2, float h);
    __device__ float fmin_rgb_value(const float red, const float green, const float blue);
    __device__ float fmax_rgb_value(const float red, const float green, const float blue);
  
    
    __device__ float fmax_rgb_value(const float red, const float green, const float blue)
    {
        if (red>green){
            if (red>blue) {
                return red;
        }
            else {
                return blue;
            }
        }
        else if (green>blue){
            return green;
        }
        else {
            return blue;
        }
    }
    

    __device__ float fmin_rgb_value(const float red, const float green, const float blue)
    {
        if (red<green){
            if (red<blue){
                return red;
            }
        else{
            return blue;}
        }
        else if (green<blue){
            return green;
        }
        else{
            return blue;
        }
    }
    
    
    __device__ float hue_to_rgb(const float m1, const float m2, float h)
        {
            if ((fabsf(h) > 1.0f) && (h > 0.0f)) {
              h = (float)fmodf(h, 1.0f);
            }
            else if (h < 0.0f){
            h = 1.0f - (float)fabsf(h);
            }
        
            if (h < 1.0f/6.0f){
                return m1 + (m2 - m1) * h * 6.0f;
            }
            if (h < 0.5f){
                return m2;
            }
            if (h < 2.0f/3.0f){
                return m1 + ( m2 - m1 ) * (float)((float)2.0f/3.0f - h) * 6.0f;
            }
            return m1;
        }
    
    __device__ struct hsl struct_rgb_to_hsl(const float r, const float g, const float b)
    {
    // check if all inputs are normalized
    assert ((0.0<= r) <= 1.0);
    assert ((0.0<= g) <= 1.0);
    assert ((0.0<= b) <= 1.0);

    struct hsl hsl_;

    float cmax=0.0f, cmin=0.0f, delta=0.0f, t;

    cmax = fmax_rgb_value(r, g, b);
    cmin = fmin_rgb_value(r, g, b);
    delta = (cmax - cmin);


    float h, l, s;
    l = (cmax + cmin) / 2.0f;

    if (delta == 0) {
    h = 0.0f;
    s = 0.0f;
    }
    else {
    	  if (cmax == r){
    	        t = (g - b) / delta;
    	        if ((fabsf(t) > 6.0f) && (t > 0.0f)) {
                  t = (float)fmodf(t, 6.0f);
                }
                else if (t < 0.0f){
                t = 6.0f - (float)fabsf(t);
                }

	            h = 60.0f * t;
          }
    	  else if (cmax == g){
                h = 60.0f * (((b - r) / delta) + 2.0f);
          }

    	  else if (cmax == b){
    	        h = 60.0f * (((r - g) / delta) + 4.0f);
          }

    	  if (l <=0.5f) {
	            s=(delta/(cmax + cmin));
	      }
	  else {
	        s=(delta/(2.0f - cmax - cmin));
	  }
    }

    hsl_.h = (float)(h * (float)1.0f/360.0f);
    hsl_.s = s;
    hsl_.l = l;
    return hsl_;
    }

    
    
    __device__ struct rgb struct_hsl_to_rgb(const float h, const float s, const float l)
    {
    
        struct rgb rgb_;
    
        float m2=0.0f, m1=0.0f;
    
        if (s == 0.0){
            rgb_.r = l;
            rgb_.g = l;
            rgb_.b = l;
            return rgb_;
        }
        if (l <= 0.5f){
            m2 = l * (1.0f + s);
        }
        else{
            m2 = l + s - (l * s);
        }
        m1 = 2.0f * l - m2;
    
        rgb_.r = hue_to_rgb(m1, m2, (float)(h + 1.0f/3.0f));
        rgb_.g = hue_to_rgb(m1, m2, h);
        rgb_.b = hue_to_rgb(m1, m2, (float)(h - 1.0f/3.0f));
        return rgb_;
    }
    
    extern "C"  __global__ void brightness(float * r, float * g, float * b, int w, int h, const double val_)
    { 
        int xx = blockIdx.x * blockDim.x + threadIdx.x;     
        int yy = blockIdx.y * blockDim.y + threadIdx.y;
    
        // Index value of the current_ pixel
        const int index = yy * h + xx;
        const int t_max = h * w;
       
        struct hsl hsl_; 
        struct rgb rgb_;
        
        if (index > 0 && index < t_max) { 
            
            float rr = r[index];
            float gg = g[index];
            float bb = b[index];
            
            hsl_ = struct_rgb_to_hsl(rr, gg, bb);
            hsl_.l += val_;
            
            hsl_.l = max(hsl_.l, -1.0f);
            hsl_.l = min(hsl_.l, 1.0f);
                
            rgb_ = struct_hsl_to_rgb(hsl_.h, hsl_.s, hsl_.l); 
            
            r[index] = rgb_.r;
            g[index] = rgb_.g;
            b[index] = rgb_.b;
            
        } 
        
        
    }
    
    
    
'''




@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef inline brightness_cupy(
        cupy_array,
        tuple grid_,
        tuple block_,
        const float brightness_factor,
        const Py_ssize_t w,
        const Py_ssize_t h
):
    """
    Applies a brightness adjustment to an RGB image using GPU acceleration.

    This function modifies the brightness of the input `cupy_array` using CUDA,  
    allowing for fast parallel processing. The brightness is adjusted by multiplying  
    each pixel value by the `brightness_factor` factor.

    :param cupy_array:  
        cupy.ndarray; A 3D array of shape `(w, h, 3)`, representing an RGB image.  
        Each pixel should have values in the range `[0, 255]`.

    :param grid_:  
        tuple (int, int); Grid dimensions `(grid_y, grid_x)`, e.g., `(25, 25)`.  
        Defines how the image is divided for parallel GPU execution.  
        Must match the texture and array sizes.

    :param block_:  
        tuple (int, int); Block dimensions `(block_y, block_x)`, e.g., `(32, 32)`.  
        Determines the number of threads per block for CUDA processing.  
        The product `block_x * block_y` must not exceed `1024` (CUDA's max thread limit per block).

    :param brightness_factor:  
        float; Brightness scaling factor.  
        - `brightness_factor > 1.0` increases brightness.  
        - `brightness_factor < 1.0` decreases brightness.  
        - `brightness_factor = 1.0` keeps the original brightness.  

    :param w:  
        int; The width of the image.

    :param h:  
        int; The height of the image.

    :return:  
        pygame.Surface; A new surface with adjusted brightness.

    :raises ValueError: If `w` or `h` is non-positive.  
    :raises ValueError: If `brightness_factor` is outside a reasonable brightness range (e.g., `brightness_factor < 0`).  
    """

    module = cp.RawModule(code=brightness_cuda)
    bright = module.get_function("brightness")

    cdef:
        r = cp.zeros((w, h), dtype=cp.float32)
        g = cp.zeros((w, h), dtype=cp.float32)
        b = cp.zeros((w, h), dtype=cp.float32)


    r = (cupy_array[:, :, 0] * <float>ONE_255)
    g = (cupy_array[:, :, 1] * <float>ONE_255)
    b = (cupy_array[:, :, 2] * <float>ONE_255)

    bright(grid_, block_, (r, g, b, w, h, brightness_factor))

    cupy_array[:, :, 0] = cp.multiply(r, <float>255.0)
    cupy_array[:, :, 1] = cp.multiply(g, <float>255.0)
    cupy_array[:, :, 2] = cp.multiply(b, <float>255.0)

    cp.cuda.Stream.null.synchronize()

    return frombuffer(
        cupy_array.astype(cp.uint8).transpose(1, 0, 2).tobytes(),
        (w, h),
        "RGB"
    )



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef hsl_gpu(
        object surface_,
        const float val_,
        tuple grid_  = None,
        tuple block_ = None
):
    """
    HSL ROTATION 

    Compatible with image format 32 - 24 bit
    Rotate the pixels color of an image/texture

    :param surface_  : pygame.Surface 
    :param grid_     : tuple; grid values (grid_y, grid_x) e.g (25, 25). The grid values and block values must 
        match the texture and array sizes. 
    :param block_    : tuple; block values (block_y, block_x) e.g (32, 32). Maximum threads is 1024.
        Max threads = block_x * block_y
    :param val_      : float; Float values representing the next hue value   
    :return          : Return a pygame.Surface with a modified HUE  
    """

    if not PyObject_IsInstance(surface_, pygame.Surface):
        raise TypeError(f"Argument `surface_` must be a pygame.Surface, got {type(surface_)}.")

    cdef unsigned char [:, :, :] cpu_array_
    try:
        cpu_array_ = surface_.get_view('3')

    except Exception as e:
        raise ValueError("\nCannot reference `surface_` pixels into a 3d array.\n %s " % e)

    cdef:
        Py_ssize_t w, h

    w, h = (<object>cpu_array_).shape[:2]

    if w == 0 or h == 0:
        raise ValueError("Surface `w` or `h` cannot be null!")

    assert 0.0 <= val_ <= 1.0, "\nArgument `val_` must be in range [0.0 ... 1.0] got %s " % val_

    return hsl_cupy(cp.asarray(cpu_array_).astype(
        dtype=cp.float32), grid_, block_, val_, w, h)

rgb2hsl_cuda = r'''


    struct hsl{
        float h;    // hue
        float s;    // saturation
        float l;    // value
    };

    struct rgb{
    float r;
    float g;
    float b;
    };

    __device__ struct rgb struct_hsl_to_rgb(const float h, const float s, const float l);
    __device__ struct hsl struct_rgb_to_hsl(const float r, const float g, const float b);
    __device__ float hue_to_rgb(const float m1, const float m2, float h);
    __device__ float fmin_rgb_value(const float red, const float green, const float blue);
    __device__ float fmax_rgb_value(const float red, const float green, const float blue);

    __device__ float fmax_rgb_value(const float red, const float green, const float blue)
    {
        if (red>green){
            if (red>blue) {
                return red;
        }
            else {
                return blue;
            }
        }
        else if (green>blue){
            return green;
        }
        else {
            return blue;
        }
    }

    __device__ float fmin_rgb_value(const float red, const float green, const float blue)
    {
        if (red<green){
            if (red<blue){
                return red;
            }
        else{
            return blue;}
        }
        else if (green<blue){
            return green;
        }
        else{
            return blue;
        }
    }

    __device__ float hue_to_rgb(const float m1, const float m2, float h)
        {
            if ((fabsf(h) > 1.0f) && (h > 0.0f)) {
              h = (float)fmodf(h, 1.0f);
            }
            else if (h < 0.0f){
            h = 1.0f - (float)fabsf(h);
            }

            if (h < 1.0f/6.0f){
                return m1 + (m2 - m1) * h * 6.0f;
            }
            if (h < 0.5f){
                return m2;
            }
            if (h < 2.0f/3.0f){
                return m1 + ( m2 - m1 ) * (float)((float)2.0f/3.0f - h) * 6.0f;
            }
            return m1;
        }

    __device__ struct hsl struct_rgb_to_hsl(const float r, const float g, const float b)
    {
    // check if all inputs are normalized
    assert ((0.0<= r) <= 1.0);
    assert ((0.0<= g) <= 1.0);
    assert ((0.0<= b) <= 1.0);

    struct hsl hsl_;

    float cmax=0.0f, cmin=0.0f, delta=0.0f, t;

    cmax = fmax_rgb_value(r, g, b);
    cmin = fmin_rgb_value(r, g, b);
    delta = (cmax - cmin);


    float h, l, s;
    l = (cmax + cmin) / 2.0f;

    if (delta == 0) {
    h = 0.0f;
    s = 0.0f;
    }
    else {
    	  if (cmax == r){
    	        t = (g - b) / delta;
    	        if ((fabsf(t) > 6.0f) && (t > 0.0f)) {
                  t = (float)fmodf(t, 6.0f);
                }
                else if (t < 0.0f){
                t = 6.0f - (float)fabsf(t);
                }

	            h = 60.0f * t;
          }
    	  else if (cmax == g){
                h = 60.0f * (((b - r) / delta) + 2.0f);
          }

    	  else if (cmax == b){
    	        h = 60.0f * (((r - g) / delta) + 4.0f);
          }

    	  if (l <=0.5f) {
	            s=(delta/(cmax + cmin));
	      }
	  else {
	        s=(delta/(2.0f - cmax - cmin));
	  }
    }

    hsl_.h = (float)(h * (float)1.0f/360.0f);
    hsl_.s = s;
    hsl_.l = l;
    return hsl_;
    }



    __device__ struct rgb struct_hsl_to_rgb(const float h, const float s, const float l)
    {

        struct rgb rgb_;

        float m2=0.0f, m1=0.0f;

        if (s == 0.0){
            rgb_.r = l;
            rgb_.g = l;
            rgb_.b = l;
            return rgb_;
        }
        if (l <= 0.5f){
            m2 = l * (1.0f + s);
        }
        else{
            m2 = l + s - (l * s);
        }
        m1 = 2.0f * l - m2;

        rgb_.r = hue_to_rgb(m1, m2, (float)(h + 1.0f/3.0f));
        rgb_.g = hue_to_rgb(m1, m2, h);
        rgb_.b = hue_to_rgb(m1, m2, (float)(h - 1.0f/3.0f));
        return rgb_;
    }

    extern "C"  __global__ void rgb2hsl(float * r, float * g, float * b, const int w, const int h, const double val_)
    { 
        int xx = blockIdx.x * blockDim.x + threadIdx.x;     
        int yy = blockIdx.y * blockDim.y + threadIdx.y;

        // Index value of the current_ pixel
        const int index = yy * h + xx;
        const int t_max = h * w;

        struct hsl hsl_; 
        struct rgb rgb_;
        float hh; 
        if (index > 0 && index < t_max) { 

            float rr = r[index] ;
            float gg = g[index] ;
            float bb = b[index] ;

            hsl_ = struct_rgb_to_hsl(rr, gg, bb);
            hh += hsl_.h + val_;
            rgb_ = struct_hsl_to_rgb(hh, hsl_.s, hsl_.l); 

            r[index] = rgb_.r ;
            g[index] = rgb_.g ;
            b[index] = rgb_.b ;
        } 
    }
'''


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef hsl_cupy(
        cupy_array,        # CuPy ndarray (w, h, 3); input image array in RGB format
        tuple grid_,       # Tuple; CUDA grid dimensions (grid_y, grid_x)
        tuple block_,      # Tuple; CUDA block dimensions (block_y, block_x)
        const float val_,  # Float; HSL adjustment value (e.g., for hue shift, saturation, or lightness modification)
        const Py_ssize_t w,  # Integer; Width of the image
        const Py_ssize_t h   # Integer; Height of the image
):
    """
    Convert an RGB image to HSL color space using a CUDA kernel and apply an HSL adjustment.

    This function uses CUDA to efficiently process an image by converting its RGB 
    channels to HSL (Hue, Saturation, Lightness).
    It modifies the HSL values based on the provided adjustment factor (`val_`), 
    then converts the image back to RGB format.

    Parameters:
    cupy_array (cupy.ndarray): A CuPy array of shape (w, h, 3) containing the image data in RGB format.
    grid_ (tuple): CUDA grid size, typically (grid_y, grid_x). Defines how the computation is distributed across the GPU.
    block_ (tuple): CUDA block size, typically (block_y, block_x). The number of threads per block should not exceed 1024.
    val_ (float): Adjustment factor for modifying HSL values (e.g., hue rotation, saturation boost, or brightness change).
    w (int): Width of the image.
    h (int): Height of the image.

    Returns:
    pygame.Surface: A new pygame surface containing the processed image in RGB format.

    Notes:
    - The function uses a CUDA kernel (`rgb2hsl_cuda`) for conversion and modification.
    - The adjustment factor `val_` determines the transformation applied to the HSL values.
    - The operation is performed entirely on the GPU for high performance.
    """

    # Load the CUDA kernel for RGB to HSL conversion
    module = cp.RawModule(code=rgb2hsl_cuda)
    rgb_to_hsl_ = module.get_function("rgb2hsl")

    # Normalize the RGB channels to the range [0, 1] by multiplying by (1/255)
    r = (cupy_array[:, :, 0] * <float>ONE_255)  # Red channel normalized
    g = (cupy_array[:, :, 1] * <float>ONE_255)  # Green channel normalized
    b = (cupy_array[:, :, 2] * <float>ONE_255)  # Blue channel normalized

    # Execute the CUDA kernel to convert RGB to HSL and apply the transformation
    rgb_to_hsl_(grid_, block_, (r, g, b, w, h, val_))

    # Convert the processed HSL values back to RGB by scaling them back to [0, 255]
    cupy_array[:, :, 0] = cp.multiply(r, <float>255.0)  # Red channel
    cupy_array[:, :, 1] = cp.multiply(g, <float>255.0)  # Green channel
    cupy_array[:, :, 2] = cp.multiply(b, <float>255.0)  # Blue channel

    # Synchronize CUDA operations to ensure all GPU computations are completed
    cp.cuda.Stream.null.synchronize()

    # Convert the processed CuPy array into a pygame.Surface object
    return frombuffer(
        cupy_array.astype(cp.uint8).transpose(1, 0, 2).tobytes(), 
        (w, h), 
        "RGB"
    )




dithering_kernel = cp.RawKernel(
r'''
    extern "C"

    __global__ void dithering_kernel(float * destination, const int w, const int h)
    {

        int i = blockIdx.x * blockDim.x + threadIdx.x;
        int j = blockIdx.y * blockDim.y + threadIdx.y;
        
        const int index  = j * h + i;          // (2d)  
        const int index1 = j * h * 3 + i * 3;  // (3d)
        const int t_max1 = w * h * 3;
        
        const float c1 = (float)(7.0f / 16.0f);
        const float c2 = (float)(3.0f / 16.0f);
        const float c3 = (float)(5.0f / 16.0f);
        const float c4 = (float)(1.0f / 16.0f);
        
        const int col = h * 3;
        
        float old_red   = (float)destination[index1 % t_max1      ];
        float old_green = (float)destination[(index1 + 1) % t_max1];
        float old_blue  = (float)destination[(index1 + 2) % t_max1];
        //__syncthreads();            
              
        float new_red   = (float)roundf(old_red);   
        float new_green = (float)roundf(old_green);    
        float new_blue  = (float)roundf(old_blue);    

        
        destination[index1 % t_max1       ] = (float)new_red;
        destination[(index1 + 1) % t_max1 ] = (float)new_green;
        destination[(index1 + 2) % t_max1 ] = (float)new_blue;
        //__syncthreads();
        
        float quantization_error_red   = (float)(old_red   - new_red);
        float quantization_error_green = (float)(old_green - new_green);
        float quantization_error_blue  = (float)(old_blue  - new_blue);

        
        destination[(index1 + 3)%t_max1] = (float)(destination[(index1 + 3)% t_max1] + quantization_error_red   * c1);
        destination[(index1 + 4)%t_max1] = (float)(destination[(index1 + 4)% t_max1] + quantization_error_green * c1);
        destination[(index1 + 5)%t_max1] = (float)(destination[(index1 + 5)% t_max1] + quantization_error_blue  * c1);
        
        
        destination[(index1 + col - 3)% t_max1] = (float)(destination[(index1 + col - 3)% t_max1] + quantization_error_red   * c2);
        destination[(index1 + col - 2)% t_max1] = (float)(destination[(index1 + col - 2)% t_max1] + quantization_error_green * c2);
        destination[(index1 + col - 1)% t_max1] = (float)(destination[(index1 + col - 1)% t_max1] + quantization_error_blue  * c2);
        
        
        destination[(index1 + col    )% t_max1] = (float)(destination[(index1 + col    )% t_max1] + quantization_error_red   * c3);
        destination[(index1 + col + 1)% t_max1] = (float)(destination[(index1 + col + 1)% t_max1] + quantization_error_green * c3);
        destination[(index1 + col + 2)% t_max1] = (float)(destination[(index1 + col + 2)% t_max1] + quantization_error_blue  * c3);
         
        
        destination[(index1 + col + 3)% t_max1] = (float)(destination[(index1 + col + 3)% t_max1] + quantization_error_red   * c4);
        destination[(index1 + col + 4)% t_max1] = (float)(destination[(index1 + col + 4)% t_max1] + quantization_error_green * c4); 
        destination[(index1 + col + 5)% t_max1] = (float)(destination[(index1 + col + 5)% t_max1] + quantization_error_blue  * c4); 
        
    __syncthreads();        
    }
    ''',
    'dithering_kernel'
)


# TODO algo is now working and the threads are overlapping

@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef dithering_gpu(
        gpu_array_,    # CuPy ndarray (w, h, 3); input image array in RGB format
        tuple grid_,   # Tuple; CUDA grid dimensions (grid_y, grid_x)
        tuple block_,  # Tuple; CUDA block dimensions (block_y, block_x)
        float factor_ = 1.0  # Float; Dithering intensity factor (default = 1.0)
):
    """
    Apply a dithering effect to an image using GPU acceleration.

    This function applies a dithering effect to an RGB image using a CUDA kernel. The operation is performed
    on the GPU for fast processing. The image is first normalized, processed using the dithering kernel,
    and then converted back to 8-bit format.

    Parameters:
    gpu_array_ (cupy.ndarray): A CuPy array of shape (w, h, 3) containing the image data in RGB format.
                               Must be of dtype `cupy.uint8`.
    grid_ (tuple): CUDA grid size, typically (grid_y, grid_x). Defines how the computation is distributed across the GPU.
    block_ (tuple): CUDA block size, typically (block_y, block_x). The number of threads per block should not exceed 1024.
    factor_ (float, optional): A factor controlling the dithering intensity. Default is 1.0.

    Returns:
    pygame.Surface: A new pygame surface containing the processed image in RGB format.

    Raises:
    TypeError: If `gpu_array_` is not a CuPy ndarray.
    ValueError: If `gpu_array_` is not of type `cupy.uint8`.

    Notes:
    - The function uses a CUDA kernel (`dithering_kernel`) for high-performance processing.
    - The input image is normalized to [0, 1] before processing and then converted back to [0, 255].
    - The grid and block sizes must be chosen appropriately based on the image dimensions and GPU resources.
    """

    # Ensure the input is a CuPy ndarray
    if not PyObject_IsInstance(gpu_array_, cp.ndarray):
        raise TypeError(f"\nArgument `gpu_array_` must be a cupy ndarray, got {type(gpu_array_)}.")

    # Validate that the input array has the correct data type
    if gpu_array_.dtype != cp.uint8:
        raise ValueError("\nArgument `gpu_array_` datatype is invalid, "
                         "expecting cupy.uint8 got %s " % gpu_array_.dtype)

    cdef:
        Py_ssize_t w, h  # Image width and height

    # Extract image dimensions (height, width)
    w, h = (<object>gpu_array_).shape[:2]

    # Normalize pixel values to the range [0, 1] for processing
    gpu_array_ = (gpu_array_ / <float>255.0).astype(cp.float32)

    # Execute the CUDA dithering kernel
    dithering_kernel(
        (grid_[0], grid_[1]),  # Grid size
        (block_[0], block_[1]),  # Block size
        (gpu_array_, w, h)  # Arguments passed to the kernel
    )

    # Synchronize CUDA operations to ensure the computation is completed
    cp.cuda.Stream.null.synchronize()

    # Convert the processed image back to 8-bit format
    gpu_array_ = (gpu_array_ * <float>255.0).astype(cp.uint8)

    # Convert the CuPy array into a pygame.Surface object and return it
    return frombuffer(
        gpu_array_.transpose(1, 0, 2).tobytes(), 
        (w, h), 
        "RGB"
    )



# -------------------------------------------------------------------------------------------------------------------

@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef inline fisheye_gpu(
        surface_,       # pygame.Surface; input surface to apply the fisheye effect
        float focal,    # Float; focal length of the fisheye effect
        float focal_texture,  # Float; texture focal length for intensity control
        tuple grid_,    # Tuple; CUDA grid dimensions (grid_y, grid_x)
        tuple block_    # Tuple; CUDA block dimensions (block_y, block_x)
):
    """
    Apply a fisheye lens effect to a pygame.Surface using GPU acceleration.

    This function applies a fisheye distortion effect to a given surface using CUDA-based processing.
    It is optimized for real-time rendering and can be used to display game scenes through a lens effect.

    Parameters:
    surface_ (pygame.Surface): Input pygame surface in 24-bit or 32-bit format.
    focal (float): Focal length of the fisheye effect, controlling the strength of the distortion.
    focal_texture (float): Texture focal length, further adjusting the fisheye intensity.
    grid_ (tuple): CUDA grid size, typically (grid_y, grid_x). Defines how computation is distributed across the GPU.
    block_ (tuple): CUDA block size, typically (block_y, block_x). Maximum threads per block should not exceed 1024.

    Returns:
    pygame.Surface: A new pygame surface with the applied fisheye lens effect.

    Raises:
    TypeError: If `surface_` is not a pygame.Surface instance.
    ValueError: If the function cannot access the pixel data of `surface_`.

    Notes:
    - This function leverages CUDA for high-performance image processing.
    - The effect is applied directly to the input surface and returned as a transformed surface.
    - Ensure that the grid and block values are chosen to match the texture and array sizes for optimal performance.
    """

    # Validate input type
    if not PyObject_IsInstance(surface_, pygame.Surface):
        raise TypeError(f"\nArgument `surface_` must be a pygame.Surface type, got {type(surface_)}.")

    cdef unsigned char [:, :, :] gpu_array  # Define a 3D array for pixel data

    try:
        # Attempt to obtain a 3D pixel view of the surface
        gpu_array = surface_.get_view('3')

    except Exception as e:
        raise ValueError("\nCannot reference `surface_` pixels into a 3D array.\n %s " % e)

    # Convert the pixel data into a CuPy array and apply the fisheye effect using CUDA
    return fisheye_cupy(cupy.asarray(gpu_array), focal, focal_texture, grid_, block_)



fisheye_kernel = cp.RawKernel(
    r'''
    
    extern "C" __global__
    
    void fisheye_kernel(unsigned char * destination, const unsigned char * source,    
    const int w, const int h, const double focal, const double focal_texture)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        int j = blockIdx.y * blockDim.y + threadIdx.y;
    
        const int index  = j * h + i;          // (2d)  
        const int index1 = j * h * 3 + i * 3;  // (3d)
        const int t_max  = w * h;
        
        const float c1 = 2.0f / (float)h;
        const float c2 = 2.0f / (float)w;
        const float w2 = (float)w * 0.5f;
        const float h2 = (float)h * 0.5f;
        
        __syncthreads();            
        
        
        if (index> 0 && index < t_max){
                   
            float nx = j * c2 - 1.0f;
            float nx2 = nx * nx;
            
            float ny = i * c1 - 1.0f;
            float ny2 = ny * ny;
            float r = (float)sqrtf(nx2 + ny2);
            //if (0.0f <= r && r <= 1.0f){
            float nr = (r * focal + 1.0f  - (float)sqrtf(1.0f - (nx2 + ny2))) * focal_texture;
            if (nr <= 1.0f){
                float theta = (float)atan2f(ny, nx);
                float nxn = nr * (float)cosf(theta);
                float nyn = nr * (float)sinf(theta);
                int x2 = (int)(nxn * w2 + w2);
                int y2 = (int)(nyn * h2 + h2);
                // int v  = (int)(y2 * w + x2);
                int index2 = x2  * h * 3 + y2 * 3;
                destination[index1 + 0] = source[index2 + 0];
                destination[index1 + 1] = source[index2 + 1];
                destination[index1 + 2] = source[index2 + 2];
            }
            //}            
        }
        __syncthreads();
    }
    ''',
    'fisheye_kernel'
)

@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef inline fisheye_cupy(
        gpu_array,        # cupy.ndarray; input image as a GPU array
        const float focal,      # Float; focal length controlling the strength of the fisheye effect
        const float focal_texture,  # Float; additional texture focal length for finer adjustments
        tuple grid_,      # Tuple; CUDA grid size (grid_y, grid_x)
        tuple block_      # Tuple; CUDA block size (block_y, block_x)
):
    """
    Apply a fisheye lens effect to a GPU-based image using CUDA.

    This function distorts an input image using a CUDA fisheye shader, simulating a real-world 
    lens distortion effect. The transformation is applied directly on a CuPy GPU array for 
    high-performance processing.

    Parameters:
    gpu_array (cupy.ndarray): Input image stored as a CuPy array (on the GPU).
    focal (float): Controls the strength of the fisheye distortion.
    focal_texture (float): Additional texture-based focal adjustment.
    grid_ (tuple): CUDA grid size, typically (grid_y, grid_x). Defines how computation is 
        distributed across the GPU.
    block_ (tuple): CUDA block size, typically (block_y, block_x). Maximum threads per block 
        should not exceed 1024.

    Returns:
    pygame.Surface: A new pygame surface with the applied fisheye effect.

    Raises:
    ValueError: If the `gpu_array` shape is invalid or the operation fails.

    Notes:
    - The function operates entirely on the GPU using CuPy and CUDA for fast execution.
    - Ensure that `grid_` and `block_` are set appropriately for optimal performance.
    - The function synchronizes the CUDA stream to ensure all operations are completed before returning.
    """

    cdef:
        Py_ssize_t w, h  # Define width and height variables

    # Extract the dimensions of the input GPU array
    w, h = (<object>gpu_array).shape[:2]

    # Create a copy of the input array to store the transformed output
    destination = gpu_array.copy()

    # Launch the CUDA kernel to apply the fisheye effect
    fisheye_kernel(
        grid_,  # CUDA grid configuration
        block_,  # CUDA block configuration
        (destination, gpu_array, w, h, focal, focal_texture)  # Kernel arguments
    )

    # Ensure all GPU operations are completed before proceeding
    cp.cuda.Stream.null.synchronize()

    # Convert the processed GPU array back into a pygame surface format
    return frombuffer(destination.transpose(1, 0, 2).tobytes(), (w, h), "RGB")




# ---------------------------------------------------------------------------------------------------

@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef inline swirl_gpu(
        surface_,           # pygame.Surface; input image
        const float rad,    # Float; rotation angle in radians
        tuple grid_,        # Tuple; CUDA grid size (grid_y, grid_x)
        tuple block_,       # Tuple; CUDA block size (block_y, block_x)
        const unsigned int centre_x,  # Integer; x-coordinate of the swirl center
        const unsigned int centre_y   # Integer; y-coordinate of the swirl center
):
    """
    Apply a swirl distortion effect to an image using GPU acceleration.

    This function creates a swirl effect on a given `pygame.Surface` by applying a CUDA-based 
    transformation. The effect distorts pixels radially around a specified center point.

    Parameters:
    surface_ (pygame.Surface): Input surface, must be in 24-bit or 32-bit format.
    rad (float): Rotation angle in radians, controlling the intensity of the swirl.
    grid_ (tuple): CUDA grid size, typically (grid_y, grid_x). Defines how computation is 
        distributed across the GPU.
    block_ (tuple): CUDA block size, typically (block_y, block_x). Maximum threads per block 
        should not exceed 1024.
    centre_x (int): X-coordinate of the swirl center (must be greater than 0).
    centre_y (int): Y-coordinate of the swirl center (must be greater than 0).

    Returns:
    pygame.Surface: A new pygame surface with the applied swirl effect.

    Raises:
    ValueError: If the input surface cannot be referenced as a 3D array.
    TypeError: If `surface_` is not a valid `pygame.Surface`.

    Notes:
    - The function operates directly on the GPU using CuPy for high performance.
    - Ensure `grid_` and `block_` are correctly configured for optimal execution.
    - The function calls `swirl_cupy`, which handles the actual GPU-based processing.
    """

    # Validate that the input is a pygame.Surface
    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument `surface_` must be a pygame.Surface type, got %s " % type(surface_)

    cdef unsigned char [:, :, :] gpu_array  # Define a GPU-compatible array

    try:
        # Obtain a direct reference to the pixel data as a 3D array
        gpu_array = surface_.get_view('3')
    except Exception as e:
        raise ValueError("\nCannot reference `surface_` pixels into a 3D array.\n %s " % e)

    # Convert the CPU surface to a CuPy GPU array and apply the swirl effect
    return swirl_cupy(cupy.asarray(gpu_array), rad, grid_, block_, centre_x, centre_y)



swirl_kernel = cp.RawKernel(
    r'''

    extern "C" __global__

    void swirl_kernel(unsigned char * destination, const unsigned char * source, const double rad,   
    const int w, const int h, const double x, const double y)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        int j = blockIdx.y * blockDim.y + threadIdx.y;

        const int index  = j * h + i;          // (2d)  
        const int index1 = j * h * 3 + i * 3;  // (3d)
        const int t_max  = w * h;


        __syncthreads();            

        if (index> 0 && index < t_max){   
            
            // 3 constant can be passed instead  
            float columns = x * w;
            float rows    = y * h;
            float r_max   = (float)sqrtf(columns * columns + rows * rows);

            float di = (float)j - columns;
            float dj = (float)i - rows;

            float r = (float)sqrtf(di * di + dj * dj);

            float an = rad * r/r_max;
                
            float c1 = (float)cosf(an);
            float c2 = (float)sinf(an);

            int diffx = (int)(di * c1 - dj * c2 + columns);
            int diffy = (int)(di * c2 + dj * c1 + rows);

            if ((diffx >-1 && diffx < w) && (diffy >-1 && diffy < h)){

                int index2 = diffx * h * 3 + diffy * 3;
                __syncthreads();
                destination[index1 + 0] = source[index2 + 0];
                destination[index1 + 1] = source[index2 + 1];
                destination[index1 + 2] = source[index2 + 2];
            }       
        }
        __syncthreads();
    }
    ''',
    'swirl_kernel'
)

@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef inline swirl_cupy(
        gpu_array,  # CuPy ndarray; input image stored in GPU memory
        const float rad,         # Float; rotation angle in radians
        tuple grid_,      # CUDA grid size (grid_y, grid_x)
        tuple block_,     # CUDA block size (block_y, block_x)
        const unsigned int centre_x,  # Integer; x-coordinate of the swirl center
        const unsigned int centre_y   # Integer; y-coordinate of the swirl center
):
    """
    Apply a swirl distortion effect to an image using GPU acceleration.

    This function warps an image by applying a swirl effect centered at `(centre_x, centre_y)`, 
    using a specified radial distortion (`rad`). The computation is performed on the GPU using CuPy 
    and a custom CUDA kernel (`swirl_kernel`).

    Parameters:
    gpu_array (cupy.ndarray): Input image stored in GPU memory (shape: [width, height, 3], dtype=uint8).
    rad (float): Rotation angle in radians, determining the intensity of the swirl effect.
    grid_ (tuple or cupy object): CUDA grid size, typically (grid_y, grid_x). Controls work distribution on GPU.
    block_ (tuple or cupy object): CUDA block size, typically (block_y, block_x). Defines the number of threads per block.
    centre_x (int): X-coordinate of the swirl center, should be within image bounds.
    centre_y (int): Y-coordinate of the swirl center, should be within image bounds.

    Returns:
    pygame.Surface: A new surface containing the image with the applied swirl effect.

    Raises:
    ValueError: If the input image has a width (`w`) or height (`h`) of zero.

    Notes:
    - The swirl transformation distorts pixel positions radially around `(centre_x, centre_y)`.
    - Ensure `grid_` and `block_` are optimized for the best performance.
    - The kernel operates directly on the GPU to achieve real-time processing.
    """

    cdef:
        Py_ssize_t w, h  # Variables for image width and height
    
    # Extract dimensions from the input GPU array
    w, h = (<object>gpu_array).shape[:2]

    # Validate dimensions to prevent invalid operations
    if w == 0 or h == 0:
        raise ValueError("Surface `w` or `h` cannot be zero!")

    # Create an empty destination image on the GPU with the same shape as the input
    destination = cupy.zeros((w, h, 3), dtype=cupy.uint8)  

    # Launch the swirl effect CUDA kernel on the GPU
    swirl_kernel(
        grid_,
        block_,
        (destination, gpu_array, rad, w, h, <float>centre_x/<float>w, <float>centre_y/<float>h)
    )

    # Synchronize GPU stream to ensure computation is complete before proceeding
    cp.cuda.Stream.null.synchronize()

    # Convert the processed GPU image back to a pygame-compatible format
    return frombuffer(destination.transpose(1, 0, 2).tobytes(), (w, h), "RGB")


#--------------------------------------------------------------------------------------------------

@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef inline wave_gpu(
    surface_, 
    const float rad_, 
    const int size_, 
    tuple grid_, 
    tuple block_):
    """
    CREATE A WAVE EFFECT
     
    e.g
    IMAGE = wave_gpu(IMAGE, 8 * math.pi/180.0 + FRAME/10, 8, grid, block)
    IMAGE = scale(IMAGE, (WIDTH + 16, HEIGHT + 16))  # Hide the left and bottom borders 
    
    :param surface_ : pygame.Surface compatible 24 - 32 bit 
    :param rad_     : float; representing a variable angle in radians
    :param size_    : integer; block size (for a realistic wave effect, keep the size below 15)
    :param grid_    : tuple; grid values (grid_y, grid_x) e.g (25, 25). The grid values and block values must 
        match the texture and array sizes. 
    :param block_   : tuple; block values (block_y, block_x) e.g (32, 32). Maximum threads is 1024.
        Max threads = block_x * block_y
    :return         : Return a pygame.Surface with a wave effect. Re-scale the final image if you can 
        see the left and bottom side with a texture wrap around effect. 
        Enlarging the final image will hide this effect when blit from the screen origin (0, 0)
    """

    if not PyObject_IsInstance(surface_, pygame.Surface):
        raise TypeError(f"\nArgument `surface_` must be a pygame.Surface type, got {type(surface_)}.")

    cdef unsigned char [:, :, :] gpu_array
    try:
        gpu_array = surface_.get_view('3')

    except Exception as e:
        raise ValueError("\nCannot reference `surface_` pixels into a 3d array.\n %s " % e)

    return wave_cupy(cupy.asarray(gpu_array), rad_, size_, grid_, block_)


wave_kernel = cp.RawKernel(

    '''
    
    extern "C" __global__
    
    void wave_kernel(unsigned char * destination, unsigned char * source, 
        double rad, int size, const int w, const int h)
{
    
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;

    const int index1 = j * h * 3 + i * 3; 
    const int t_max1 = w * h * 3;
    
    const float c1 = 1.0f / (float)(size * size);
    
    if (i < h && j < w) {
    
        unsigned int y_pos = (unsigned int) (j + size + (int) ((float) sinf(rad + (float) j * c1) * (float) size));
        unsigned int x_pos = (unsigned int) (i + size + (int) ((float) sinf(rad + (float) i * c1) * (float) size));
     
        // % t_max1 help wrap around the image when index is overflow in the texture 
        unsigned int index2 = (unsigned int) (y_pos * h * 3 + x_pos * 3) % t_max1;  
        __syncthreads();
        destination[index1 + 0] = source[index2 + 0];
        destination[index1 + 1] = source[index2 + 1];
        destination[index1 + 2] = source[index2 + 2];       
    } 
    
}
    ''',
    'wave_kernel'
)


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef inline wave_cupy(
        gpu_array,      # CuPy ndarray; input image stored in GPU memory
        const float rad_,  # Float; wave intensity in radians
        const int size_,   # Integer; wave frequency size (pixel distance between waves)
        tuple grid_,       # CUDA grid size (grid_y, grid_x)
        tuple block_       # CUDA block size (block_y, block_x)
):
    """
    Apply a wave distortion effect to an image using GPU acceleration.

    This function distorts an image by applying a sinusoidal wave effect. The transformation 
    is performed entirely on the GPU using CuPy and a custom CUDA kernel (`wave_kernel`).

    Parameters:
    gpu_array (cupy.ndarray): Input image stored in GPU memory (shape: [width, height, 3], dtype=uint8).
    rad_ (float): Amplitude of the wave effect, represented as a rotation in radians.
    size_ (int): Frequency of the wave, determining the distance between successive waves in pixels.
    grid_ (tuple): CUDA grid size (grid_y, grid_x), which controls the distribution of work on the GPU.
    block_ (tuple): CUDA block size (block_y, block_x), defining the number of threads per block.

    Returns:
    pygame.Surface: A new surface containing the image with the applied wave effect.

    Raises:
    ValueError: If the input image has a width (`w`) or height (`h`) of zero.

    Notes:
    - The wave transformation displaces pixels sinusoidally along the x-axis or y-axis.
    - Ensure `grid_` and `block_` values are optimized for efficient GPU performance.
    - The CUDA kernel performs the wave effect directly on the GPU to enable real-time rendering.
    """

    cdef:
        Py_ssize_t w, h  # Variables for image width and height

    # Extract dimensions from the input GPU array
    w, h = (<object>gpu_array).shape[:2]

    # Create an empty output image on the GPU with the same shape as the input
    destination = cupy.empty((w, h, 3), dtype=cupy.uint8)

    # Launch the wave effect CUDA kernel on the GPU
    wave_kernel(
        grid_,
        block_,
        (destination, gpu_array, rad_, size_, w, h)
    )

    # Synchronize GPU stream to ensure computation is complete before proceeding
    cp.cuda.Stream.null.synchronize()

    # Convert the processed GPU image back to a pygame-compatible format
    return frombuffer(destination.transpose(1, 0, 2).tobytes(), (w, h), "RGB")


# ---------------------------------------------------------------------------------------------------------------



# ---------------------------------------------------------------------------------------------------------------

@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef inline chromatic_gpu(
        surface_,
        unsigned int delta_x,
        unsigned int delta_y,
        tuple grid_,
        tuple block_,
        float zoom = 0.999,
        float fx = 0.05
):
    """
    Applies a chromatic displacement effect to a given surface using GPU acceleration.  
    This effect creates a color separation effect by shifting RGB channels based on
     the specified displacement values.  
    The function is optimized for 24-bit and 32-bit image formats.

    :param surface_:  
        pygame.Surface; The input surface to be processed.  
        Must be in 24-bit or 32-bit format.

    :param delta_x:  
        int; Horizontal displacement for the chromatic shift.  
        Affects the red and blue channels, creating a color separation effect.  
        A value of zero results in no horizontal displacement (if fx is low).

    :param delta_y:  
        int; Vertical displacement for the chromatic shift.  
        Functions similarly to `delta_x`, but applies to vertical movement.  
        A value of zero results in no vertical displacement (if fx is low).

    :param grid_:  
        tuple (int, int); Grid dimensions (grid_y, grid_x), e.g., (25, 25).  
        These values define the division of the image for GPU processing  
        and must match the texture and array sizes.

    :param block_:  
        tuple (int, int); Block dimensions (block_y, block_x), e.g., (32, 32).  
        Determines the number of threads per block in GPU processing.  
        The product `block_x * block_y` must not exceed 1024 (the CUDA limit for threads per block).

    :param zoom:  
        float; Zoom factor, controlling the image scale after processing.  
        Must be in the range (0.0, 1.0). Default is `1.0` (no zoom). 
        
    :param fx:  
        float; Intensity of the chromatic effect, controlling the color separation strength.  
        Must be within the range [0.0, 0.2]. Default is `0.05`.

    :return:  
        CuPy array representing the processed image with the chromatic displacement effect applied.  
        The output format is a 24-bit image.

    :raises TypeError: If `surface_` is not of type `pygame.Surface`.  
    :raises ValueError: If `surface_` has zero width or height.  
    :raises ValueError: If `delta_x` or `delta_y` is negative.  
    """

    # Ensure the input surface is a valid pygame.Surface instance
    if not PyObject_IsInstance(surface_, pygame.Surface):
        raise TypeError(f"\nArgument `surface_` must be a pygame.Surface type, got {type(surface_)}.")

    # Retrieve the dimensions of the surface
    cdef Py_ssize_t w, h
    w, h = surface_.get_size()

    # Validate dimensions of the image
    if w == 0:
        raise ValueError("Image width `w` cannot be zero!")
    if h == 0:
        raise ValueError("Image height `h` cannot be zero!")

    # Validate delta_x and delta_y are non-negative
    if delta_x < 0 or delta_y < 0:
        raise ValueError("Arguments `delta_x` and `delta_y` must be >= 0")

    # Apply modulo operation to ensure delta_x and delta_y stay within bounds
    delta_x %= w
    delta_y %= h

    # Validate zoom factor
    if zoom <= 0.0 or zoom > 1.0:
        raise ValueError("Argument `zoom` must be in the range ]0.0, 1.0]")

    # Validate fx intensity factor
    if not (0.0 <= fx <= 0.2):
        raise ValueError("Argument `fx` must be in range [0.0, 0.2]")

    # Attempt to obtain a 3D array reference to the surface's pixel data
    cdef unsigned char [:, :, :] gpu_array

    try:
        gpu_array = surface_.get_view('3')
    except Exception as e:
        raise ValueError("\nCannot reference source pixels into a 3D array.\n %s " % e)

    # Convert the 3D pixel data to a CuPy array and apply the chromatic effect
    return chromatic_cupy(cupy.asarray(gpu_array),
                          grid_, block_, delta_x, delta_y, zoom, fx)



chromatic_kernel = cp.RawKernel(

    '''

    extern "C" 
       
    __global__  void chromatic_kernel(
        unsigned char * destination, 
        const unsigned char * source,
        const int w, 
        const int h, 
        const int delta_x, 
        const int delta_y, 
        const double zoom, 
        const double fx
        )
{

    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;

    int index1 = j * h * 3 + i * 3;   
    const unsigned int t_max = w * h * 3;
    unsigned char local_memory[3]; 
    const float dw = (float)delta_x/(float)w;
    const float dh = (float)delta_y/(float)h;
    
    float nx = float((float)i / (float)h) - dh;  
    float ny = float((float)j / (float)w) - dw;
    
    float theta_rad = (float)atan2f (ny,nx);
    float cos_ = (float)cosf((float)theta_rad);
    float sin_ = (float)sinf((float)theta_rad);
    
    float dist = (float)sqrtf(nx * nx + ny * ny); 
    
    float new_dist = dist * (zoom - fx);
    
    float new_ii = (float)cos_ * new_dist ;
    float new_jj = (float)sin_ * new_dist;
    
    int new_j = (int)((new_jj + dw) * (float)w); 
    int new_i = (int)((new_ii + dh) * (float)h); 
        
    if (i < h && j < w ) {
        __syncthreads();
        const int r = new_j * h * 3 + new_i * 3 + 0;      
        // local_memory[0] = source[r];   
        destination[index1 + 0] = source[r]; 

    }
    new_dist = dist * (zoom  - fx * 2); 
    
    new_j = (int)(((float)sin_ * new_dist + dw) * (float)w); 
    new_i = (int)(((float)cos_ * new_dist + dh) * (float)h); 
        
    if (i < h && j < w ) {
        __syncthreads();
        const int g = new_j * h * 3 + new_i * 3 + 1;      
        // local_memory[1] = source[g];   
        destination[index1 + 1] = source[g];  

    }
    
    new_dist = dist * (zoom  - fx * 3); 
    
    new_ii = (float)cos_ * new_dist;
    new_jj = (float)sin_ * new_dist;
    
    new_j = (int)((new_jj + dw) * (float)w); 
    new_i = (int)((new_ii + dh) * (float)h); 
    
    if (i < h && j < w) {
        __syncthreads();
        const int b = new_j * h * 3 + new_i * 3 + 2;      
        // local_memory[2] = source[b];   
        destination[index1 + 2] = source[b];
    }    
}
    ''',
    'chromatic_kernel'
)

@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef inline chromatic_cupy(
        object gpu_array,
        object grid_,
        object block_,
        unsigned int delta_x,
        unsigned int delta_y,
        float zoom = <float>1.0,
        float fx = <float>0.05
):
    """
    Applies a chromatic displacement effect to a CuPy-based GPU array using CUDA acceleration.  
    This function executes a GPU kernel to shift RGB channels based on the specified displacement values.

    :param gpu_array:  
        CuPy ndarray; The input image data in GPU memory, with shape (height, width, channels).  
        Must be a 3D array representing an RGB image.

    :param grid_:  
        tuple (int, int); Grid dimensions (grid_y, grid_x), e.g., (25, 25).  
        Defines the division of work for CUDA processing.  
        Must be compatible with the texture and array sizes.

    :param block_:  
        tuple (int, int); Block dimensions (block_y, block_x), e.g., (32, 32).  
        Determines the number of threads per block for GPU execution.  
        The product `block_x * block_y` must not exceed 1024 (CUDA's thread limit per block).

    :param delta_x:  
        unsigned int; Horizontal displacement for the chromatic shift.  
        Affects the red and blue channels, creating a horizontal color separation effect.  
        A value of `0` results in no horizontal displacement.

    :param delta_y:  
        unsigned int; Vertical displacement for the chromatic shift.  
        Similar to `delta_x`, but applies displacement vertically.  
        A value of `0` results in no vertical displacement.

    :param zoom:  
        float; Scaling factor for the output image.  
        Must be in the range `(0.0, 1.0]`. Default is `1.0` (no scaling).

    :param fx:  
        float; Intensity of the chromatic effect, controlling the amount of color separation.  
        Must be within the range `[0.0, 0.2]`. Default is `0.05`.

    :return:  
        A 2D RGB image in 24-bit format, reconstructed from the processed GPU array.

    :raises ValueError: If `gpu_array` does not have a valid shape.  
    :raises ValueError: If `delta_x` or `delta_y` is negative.  
    :raises ValueError: If `zoom` is not within the valid range `(0.0, 1.0]`.  
    :raises ValueError: If `fx` is not within the valid range `[0.0, 0.2]`.
    """

    cdef:
        Py_ssize_t w, h
    w, h = (<object>gpu_array).shape[:2]

    destination = cupy.zeros((w, h, 3), dtype=cupy.uint8, order='C')

    chromatic_kernel(
        grid_,
        block_,
        (destination, gpu_array, w, h, delta_x, delta_y, zoom, fx)
    )

    cp.cuda.Stream.null.synchronize()

    return frombuffer(
        destination.transpose(1, 0, 2).tobytes(),
        (w, h),
        "RGB"
        )


# ---------------------------------------------------------------------------------------------------------------

@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef inline rgb_split_gpu(
        surface_,   # Pygame surface (input image)
        float delta_x,  # Horizontal displacement for color channels
        float delta_y,  # Vertical displacement for color channels
        tuple grid_,    # CUDA grid size (grid_y, grid_x)
        tuple block_    # CUDA block size (block_y, block_x)
):
    """
    Apply an RGB split effect to an image using GPU acceleration.

    This function shifts the red, green, and blue channels of an image by `delta_x` and `delta_y`
    to create a chromatic aberration (glitch) effect. The transformation is performed entirely on 
    the GPU using CuPy.

    Parameters:
    surface_ (pygame.Surface): Input image in a 24-bit or 32-bit format.
    delta_x (float): The horizontal shift applied to the RGB channels.
    delta_y (float): The vertical shift applied to the RGB channels.
    grid_ (tuple): CUDA grid size (grid_y, grid_x), which defines the distribution of computation.
    block_ (tuple): CUDA block size (block_y, block_x), defining the number of threads per block.

    Returns:
    pygame.Surface: A new surface with the RGB split effect applied.

    Raises:
    TypeError: If `surface_` is not a pygame.Surface.
    ValueError: If `delta_x` or `delta_y` are non-positive values.
    ValueError: If the function cannot access pixel data from `surface_`.

    Notes:
    - The RGB split effect creates a chromatic distortion, commonly used in glitch effects.
    - `delta_x` and `delta_y` define how far the color channels are shifted.
    - Ensure `grid_` and `block_` values are optimized for GPU execution.
    """

    # Ensure the input is a pygame.Surface
    if not PyObject_IsInstance(surface_, pygame.Surface):
        raise TypeError(f"\nArgument `surface_` must be a pygame.Surface type, got {type(surface_)}.")

    # Get original surface dimensions
    cdef Py_ssize_t prev_w, prev_h
    prev_w, prev_h = surface_.get_size()

    # Scale the image to accommodate the RGB split effect
    new_surface = scale(surface_, (prev_w + delta_x * <float>3.0, prev_h + delta_y * <float>3.0)).convert()

    # Validate image dimensions
    assert prev_w != 0, "Image width (`w`) cannot be zero!"
    assert prev_h != 0, "Image height (`h`) cannot be zero!"

    # Validate displacement values
    if delta_x <= 0 or delta_y <= 0:
        raise ValueError("Arguments `delta_x` and `delta_y` must be greater than 0.")

    # Apply modular arithmetic to ensure the displacement stays within bounds
    delta_x %= prev_w
    delta_y %= prev_h

    # Extract pixel data as a 3D array (height, width, color channels)
    cdef unsigned char [:, :, :] gpu_array
    try:
        gpu_array = surface_.get_view('3')
    except Exception as e:
        raise ValueError(f"\nCannot reference `surface_` pixels into a 3D array.\n {e}")

    # Convert to CuPy array and apply the RGB split effect using a CUDA kernel
    return rgb_split_cupy(
        cupy.asarray(gpu_array),  # Convert to CuPy array
        prev_w, prev_h, grid_, block_, delta_x, delta_y
    )





rgb_split_kernel = cp.RawKernel(

    '''

    extern "C" __global__

    void rgb_split_kernel(unsigned char * destination, const unsigned char * source,
        const int w, const int h, const int ww, const int hh, 
        double delta_x, double delta_y)
{

    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;

    int index1 = j * h * 3 + i * 3;
    int index2 = j * hh * 3  + i * 3;
    // unsigned char local_memory[3]; 

    if (i < hh && j < ww) {
        
        const int r = (int)(j + delta_x * 1.0f) * hh * 3.0f  + (int)(i + delta_y * 1.0f) * 3.0f + 0.0f; 
        const int g = (int)(j + delta_x * 2.0f) * hh * 3.0f  + (int)(i + delta_y * 2.0f) * 3.0f + 1.0f; 
        const int b = (int)(j + delta_x * 3.0f) * hh * 3.0f  + (int)(i + delta_y * 3.0f) * 3.0f + 2.0f; 
        
        /*
        local_memory[0] = source[r];
        local_memory[1] = source[g];
        local_memory[2] = source[b];
        __syncthreads();
        destination[index1 + 0] = local_memory[0];
        destination[index1 + 1] = local_memory[1];
        destination[index1 + 2] = local_memory[2];
        */ 
        
        __syncthreads();
        
        destination[index1 + 0] = source[r];
        destination[index1 + 1] = source[g];
        destination[index1 + 2] = source[b];
        
    }

}
    ''',
    'rgb_split_kernel'
)



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef inline rgb_split_cupy(
        gpu_array,      # CuPy array representing the image
        const Py_ssize_t prev_w,  # Original image width before processing
        const Py_ssize_t prev_h,  # Original image height before processing
        tuple grid_,    # CUDA grid size (grid_y, grid_x)
        tuple block_,   # CUDA block size (block_y, block_x)
        const float delta_x,  # Horizontal displacement for RGB channels
        const float delta_y   # Vertical displacement for RGB channels
):
    """
    Apply an RGB split effect on a GPU-accelerated image using CuPy.

    This function shifts the red, green, and blue channels of an image by `delta_x` and `delta_y`
    to create a chromatic aberration (glitch) effect. The computation is performed on the GPU
    using a CUDA kernel.

    Parameters:
    gpu_array (cupy.ndarray): Input image as a CuPy array in shape (h, w, 3), dtype=uint8.
    prev_w (int): Original width of the image.
    prev_h (int): Original height of the image.
    grid_ (tuple): CUDA grid size (grid_y, grid_x), defining thread block distribution.
    block_ (tuple): CUDA block size (block_y, block_x), defining threads per block.
    delta_x (float): The horizontal shift applied to the RGB channels.
    delta_y (float): The vertical shift applied to the RGB channels.

    Returns:
    cupy.ndarray: A new CuPy array with the RGB split effect applied.

    Notes:
    - This function is optimized for performance using GPU parallelization.
    - The image data is stored in row-major order (`order='C'` for better performance).
    - `delta_x` and `delta_y` determine the degree of chromatic shift.

    Raises:
    ValueError: If the input array has zero width or height.
    """

    cdef:
        Py_ssize_t w, h  # Dimensions of the working image

    # Extract the width and height of the input image
    w, h = (<object>gpu_array).shape[:2]

    # Validate input dimensions
    if w == 0 or h == 0:
        raise ValueError("Input image width (`w`) or height (`h`) cannot be zero!")

    # Create an empty destination array with the original dimensions
    # dtype=cupy.uint8 ensures proper image format
    destination = cupy.empty((prev_w, prev_h, 3), dtype=cupy.uint8, order='C')

    # Execute the CUDA kernel for RGB splitting
    rgb_split_kernel(
        grid_,
        block_,
        (destination, gpu_array, prev_w, prev_h, w, h, delta_x, delta_y)
    )

    # Ensure CUDA operations are completed before returning the result
    cp.cuda.Stream.null.synchronize()

    # Convert back to a format that can be used in a standard RGB representation
    return frombuffer(destination.transpose(1, 0, 2).tobytes(), (prev_w, prev_h), "RGB")


# ---------------------------------------------------------------------------------------------------------------

@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef inline zoom_gpu(
        surface_,
        unsigned int centre_x,
        unsigned int centre_y,
        tuple grid_,
        tuple block_,
        float zoom = <float>1.0
):
    """
    Applies a zoom effect using GPU acceleration.
    
    This function processes a `pygame.Surface` by mapping its pixel data to GPU memory 
    and applying a zoom transformation. 
    It utilizes CUDA acceleration for high-performance image processing.
    
    :param surface_:  
        pygame.Surface; The input surface to be processed.  
        Must be a 24-bit or 32-bit format image.
    
    :param centre_x:  
        unsigned int; zoom centre x 
        Must be `>= 0`.  
    
    :param centre_y:  
        unsigned int; zoom centre y 
        Must be `>= 0`.  
    
    :param grid_:  
        tuple (int, int); Grid dimensions `(grid_y, grid_x)`, e.g., `(25, 25)`.  
        Defines the division of the image for parallel GPU processing.  
        The values must match the texture and array sizes.
    
    :param block_:  
        tuple (int, int); Block dimensions `(block_y, block_x)`, e.g., `(32, 32)`.  
        Determines the number of threads per block in GPU execution.  
        The product `block_x * block_y` must not exceed `1024` (CUDA limit).
    
    :param zoom:  
        float; Zoom factor controlling the scaling of the image.  
        Must be within the range `(0.0, 1.0]`. Default is `1.0` (no scaling).
    
    :return:  
        A CuPy array containing the transformed image with zoom and chromatic displacement applied
        image format 24-bit.
    
    :raises TypeError: If `surface_` is not of type `pygame.Surface`.  
    :raises ValueError: If the width or height of `surface_` is `0`.  
    :raises ValueError: If `zoom` is not within the range `(0.0, 1.0]`.  
    :raises ValueError: If `centre_x` or `centre_y` is negative.  
    :raises ValueError: If `surface_` cannot be mapped to a 3D array.
    """

    # Ensure `surface_` is a valid pygame.Surface object
    if not PyObject_IsInstance(surface_, pygame.Surface):
        raise TypeError(f"Argument `surface_` must be a pygame.Surface, got {type(surface_)}.")

    # Retrieve the width and height of the surface
    cdef Py_ssize_t w, h
    w, h = surface_.get_size()

    # Validate image dimensions
    if w == 0:
        raise ValueError("Image `width` cannot be zero!")
    if h == 0:
        raise ValueError("Image `height` cannot be zero!")

    # Validate zoom factor
    if zoom <= 0.0 or zoom > 1.0:
        raise ValueError("Argument `zoom` must be in range (0.0, 1.0].")

    # Ensure displacement values are non-negative
    if centre_x < 0 or centre_y < 0:
        raise ValueError("Arguments `centre_x` and `centre_y` must be >= 0.")

    # Normalize displacement values to ensure they wrap around the image dimensions
    centre_x %= w
    centre_y %= h

    # Declare a GPU-accessible 3D array
    cdef unsigned char [:, :, :] gpu_array

    try:
        # Obtain a reference to the surface pixel data as a 3D array
        gpu_array = surface_.get_view('3')
    except Exception as e:
        raise ValueError(f"Cannot reference `surface_` pixels into a 3D array.\n{e}")

    # Convert the surface data to a CuPy array and apply the zoom effect using the GPU
    return zoom_cupy(cupy.asarray(gpu_array), grid_, block_, centre_x, centre_y, zoom)




zoom_kernel = cp.RawKernel(

    '''

    extern "C" 
       
    __global__  void zoom_kernel(unsigned char * destination, const unsigned char * source,
        const int w, const int h, const int centre_x, const int centre_y, const double zoom)
{

    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;

    int index1 = j * h * 3 + i * 3;   
    const unsigned int t_max = w * h * 3;  
   
    const float dw = (float)centre_x/(float)w;
    const float dh = (float)centre_y/(float)h;
    
    float nx = float((float)i / (float)h) - dh;  
    float ny = float((float)j / (float)w) - dw;
    
    float theta = (float)atan2f (ny,nx);
        
    float nx2 = nx * nx;
    float ny2 = ny * ny; 
    
    float dist = (float)sqrtf(nx2 + ny2); 
    float new_dist = dist * zoom;
    
    int new_j = (int)(((float)sinf((float)theta) * new_dist + dw) * (float)w); 
    int new_i = (int)(((float)cosf((float)theta) * new_dist + dh) * (float)h); 
        
    if (i < h && j < w) {

        const int ind = new_j * h * 3 + new_i * 3;
        
        __syncthreads();
        destination[index1 + 0] = source[ind];
        destination[index1 + 1] = source[ind+1];
        destination[index1 + 2] = source[ind+2];
    }
     
}
    ''',
    'zoom_kernel'
)


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef inline zoom_cupy(
        gpu_array,
        tuple grid_,
        tuple block_,
        const unsigned int centre_x,
        const unsigned int centre_y,
        float zoom = <float>0.99999
):
    """
    Applies a zoom transformation to a GPU-accelerated image using CUDA.

    This function performs a zoom effect centered at `(centre_x, centre_y)` using 
    GPU acceleration. The zoom factor determines how much the image is scaled, 
    with values closer to `1.0` representing minimal zoom.

    :param gpu_array:  
        CuPy ndarray; The input image stored in GPU memory.  
        Must be a 3D array of shape `(height, width, channels)` representing an RGB image.

    :param grid_:  
        tuple (int, int); Grid dimensions `(grid_y, grid_x)`, e.g., `(25, 25)`.  
        Defines how the image is divided for parallel processing.  
        Must match the texture and array sizes.

    :param block_:  
        tuple (int, int); Block dimensions `(block_y, block_x)`, e.g., `(32, 32)`.  
        Determines the number of threads per block for CUDA execution.  
        The product `block_x * block_y` must not exceed `1024` (CUDA's thread limit per block).

    :param centre_x:  
        unsigned int; X-coordinate of the zoom center.  
        Defines the focal point around which the image is scaled.

    :param centre_y:  
        unsigned int; Y-coordinate of the zoom center.  
        Functions similarly to `centre_x`, but applies to the vertical axis.

    :param zoom:  
        float; Zoom factor controlling the scaling effect.  
        Must be in the range `(0.0, 1.0]`, where values closer to `1.0` result in a near-original scale.  
        Default is `0.99999`, which applies a very subtle zoom effect.

    :return:  
        A 2D RGB image in 24-bit format, reconstructed from the processed GPU array.

    :raises ValueError: If `gpu_array` does not have a valid shape.  
    :raises ValueError: If `centre_x` or `centre_y` is out of image bounds.  
    :raises ValueError: If `zoom` is not within the range `(0.0, 1.0]`.
    """

    cdef:
        Py_ssize_t w, h
    w, h = (<object>gpu_array).shape[:2]

    destination = cupy.zeros((w, h, 3), dtype=cupy.uint8)

    zoom_kernel(
        grid_,
        block_,
        (destination, gpu_array, w, h, centre_x, centre_y, zoom)
    )

    cp.cuda.Stream.null.synchronize()

    return frombuffer(
        destination.transpose(1, 0, 2).tobytes(),
        (w, h),
        "RGB"
    )

# ---------------------------------------------------------------------------------------------------------------



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef tuple wavelength_mapper(
    const unsigned int wavelength_min,  # Minimum wavelength value (in nanometers)
    const unsigned int wavelength_max   # Maximum wavelength value (in nanometers)
):
    """
    Generate an RGB color mapping for a range of wavelengths.

    This function maps wavelengths in the visible spectrum to their corresponding RGB values.
    It creates a lookup table for converting specific wavelengths into their approximate colors.

    Parameters:
    wavelength_min (unsigned int): The minimum wavelength (e.g., 380 nm for violet).
    wavelength_max (unsigned int): The maximum wavelength (e.g., 700 nm for red).

    Returns:
    tuple:
        - redmap_array (numpy.ndarray): A (wavelength_range, 3) array of uint8 values,
          where each row represents an (R, G, B) color corresponding to a wavelength.
        - f_redmap (float): A scaling factor used for mapping color intensity.

    Notes:
    - Wavelengths are expected in the visible light spectrum (approximately 380–700 nm).
    - Uses the `wavelength_to_rgb` function to determine RGB values per wavelength.
    - The returned array can be used for visualization or color mapping in simulations.
    """

    cdef:
        rgb_color_int rgb_c  # Struct to hold RGB values for a wavelength
        int i  # Loop variable

    # Calculate the range of wavelengths
    cdef Py_ssize_t wavelength_range = wavelength_max - wavelength_min

    # Pre-allocate the redmap_array as a NumPy array of shape (wavelength_range, 3)
    REDMAP = numpy.zeros((wavelength_range, 3), dtype=numpy.uint8)

    # Initialize an empty list to store RGB values for each wavelength
    REDMAP = [x for x in range(wavelength_min, wavelength_max)]

    cdef int n = 0
    for i in range(wavelength_min, wavelength_max):
        # Convert the wavelength to an RGB color
        rgb_c = wavelength_to_rgb(i, 1.0)  # 1.0 represents full intensity
        REDMAP[n] = (rgb_c.r, rgb_c.g, rgb_c.b)
        n += 1

    # Compute a normalization factor for mapping wavelength range to RGB values
    cdef float f_redmap = (wavelength_range - 1.0) / (255.0 * 3.0)

    # Create a NumPy array to store the RGB values
    cdef unsigned char[:, ::1] redmap_array = numpy.zeros((wavelength_range, 3), numpy.uint8)

    # Populate the array with the calculated RGB values
    i = 0
    for t in REDMAP:
        redmap_array[i, 0] = t[0]  # Red channel
        redmap_array[i, 1] = t[1]  # Green channel
        redmap_array[i, 2] = t[2]  # Blue channel
        i += 1

    # Return the generated RGB color map and the normalization factor
    return redmap_array, f_redmap





"""
- Violet: 380–450 nm (688–789 THz)
- Blue: 450–495 nm
- Green: 495–570 nm
- Yellow: 570–590 nm
- Orange: 590–620 nm
- Red: 620–750 nm (400–484 THz)
"""

redmap_array,  f_redmap = wavelength_mapper(620, 750)
greenmap_array,  f_greenmap = wavelength_mapper(495, 570)
bluemap_array, f_bluemap = wavelength_mapper(450, 495)
heatmap_array, f_heatmap = wavelength_mapper(450, 720)


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef inline wavelength_map_gpu(
        surface_,
        tuple grid_,
        tuple block_,
        unsigned short int layer_=0
):
    """
    APPLY A CHANNEL-SPECIFIC COLOR EFFECT (REDSCALE, GREEN SCALE, OR BLUESCALE) TO AN IMAGE.

    This function applies a redscale, greenscale, or bluescale effect to a given pygame.Surface image.
    It isolates and emphasizes one of the color channels (Red, Green, or Blue) depending on the 
    provided `layer_` parameter.
    The operation is performed using GPU acceleration for better performance.

    :param surface_: pygame.Surface object; The input image to which the effect will be applied.
    :param grid_: tuple; Grid configuration (grid_y, grid_x), where the grid values should match the image and array sizes.
    :param block_: tuple; Block configuration (block_y, block_x), where the block values determine how many threads are launched per block.
    :param layer_: unsigned short integer; Specifies which color channel to isolate:
        0 for Red channel, 1 for Green channel, and 2 for Blue channel.
    :return: pygame.Surface; The image with the selected channel effect applied (redscale, greenscale, or bluescale).
    """

    # Validate that the input is a pygame.Surface object.
    if not PyObject_IsInstance(surface_, pygame.Surface):
        raise TypeError(f"\nArgument `surface_` must be a pygame.Surface type, got {type(surface_)}.")

    # Get the width and height of the surface (image).
    cdef Py_ssize_t w, h
    w, h = surface_.get_size()

    # Ensure the image dimensions are valid.
    assert w != 0, "Image `w` cannot be null!"
    assert h != 0, "Image `h` cannot be null!"

    # Try to get a view of the surface as a 3D array (for accessing RGB values).
    cdef unsigned char [:, :, :] gpu_array
    try:
        gpu_array = surface_.get_view('3')
    except Exception as e:
        raise ValueError("\nCannot reference `surface_` pixels into a 3d array.\n %s " % e)

    # Call the GPU-accelerated function (wavelength_map_cupy) to apply the effect.
    return wavelength_map_cupy(cupy.asarray(gpu_array), grid_, block_, layer_)




wavelength_map_kernel = cp.RawKernel(

    '''
    extern "C" 

    __global__  void wavelength_map_kernel(
    unsigned char * destination, 
    const unsigned char * source, 
    const unsigned char * map,
    const double map_c, 
    const int w, 
    const int h)
{

    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;

    int index1 = j * h * 3 + i * 3;    
    
    unsigned int s = source[index1 + 0] + source[index1 + 1] + source[index1 + 2];
    unsigned int index = (unsigned int)((float)s * map_c);  

    if (i < h && j < w) {
        
        __syncthreads();
        
        destination[index1 + 0] = (unsigned char)map[index * 3 + 0];
        destination[index1 + 1] = (unsigned char)map[index * 3 + 1];
        destination[index1 + 2] = (unsigned char)map[index * 3 + 2];       
    }
}
    ''',
    'wavelength_map_kernel'
)



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef inline wavelength_map_cupy(
        gpu_array,
        tuple grid_,
        tuple block_,
        unsigned short int layer_
):
    cdef:
        Py_ssize_t w, h
    w, h = (<object>gpu_array).shape[:2]

    destination = cupy.zeros((w, h, 3), dtype=cupy.uint8)

    cdef float f_map


    if layer_ == 0:
        array_, f_map = redmap_array, f_redmap

    if layer_ == 1:
        array_, f_map = greenmap_array, f_greenmap

    if layer_ == 2:
        array_, f_map = bluemap_array, f_bluemap

    wavelength_map_kernel(
        grid_,
        block_,
        (destination, gpu_array, cupy.asarray(array_), f_map, w, h)
    )

    cp.cuda.Stream.null.synchronize()

    return frombuffer(destination.transpose(1, 0, 2).tobytes(), (w, h), "RGB")




@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef inline wavelength_map_cupy_inplace(
        gpu_array,
        cpu_array,
        tuple grid_,
        tuple block_,
        unsigned short int layer_
):
    cdef:
        Py_ssize_t w, h
    w, h = (<object>gpu_array).shape[:2]

    destination = cupy.zeros((w, h, 3), dtype=cupy.uint8)

    cdef float f_map


    if layer_ == 0:
        array_, f_map = redmap_array, f_redmap

    if layer_ == 1:
        array_, f_map = greenmap_array, f_greenmap

    if layer_ == 2:
        array_, f_map = bluemap_array, f_bluemap

    wavelength_map_kernel(
        grid_,
        block_,
        (destination, gpu_array, cupy.asarray(array_), f_map, w, h)
    )

    cp.cuda.Stream.null.synchronize()

    cpu_array[:, :, 0] = destination[:, :, 0].get()
    cpu_array[:, :, 1] = destination[:, :, 1].get()
    cpu_array[:, :, 2] = destination[:, :, 2].get()

# ---------------------------------------------------------------------------------------------------------------


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef inline heatmap_gpu(
        surface_,
        tuple grid_,
        tuple block_,
        bint invert_ = False
):
    """
    APPLY A HEATMAP EFFECT TO AN IMAGE WITH OPTIONAL INVERSION.

    This function applies a heatmap effect to an image using GPU acceleration. The heatmap effect is typically
    used to visualize intensity or magnitude across a 2D surface. The effect can be inverted based on the `invert_`
    parameter, changing the color representation.

    :param surface_: pygame.Surface; The input image to which the heatmap effect will be applied.
    :param grid_: tuple; Grid configuration (grid_y, grid_x) used for parallel execution on the GPU.
                  The grid values should match the texture and array sizes.
    :param block_: tuple; Block configuration (block_y, block_x) defines the number of threads per block for GPU execution.
                   Maximum threads = block_x * block_y, with a limit of 1024 threads.
    :param invert_: boolean; Optional parameter that inverts the heatmap effect. If `True`, the color range is reversed.
    :return: pygame.Surface; The image with the heatmap effect applied.
    """

    # Validate that the input is a pygame.Surface object.
    if not PyObject_IsInstance(surface_, pygame.Surface):
        raise TypeError(f"\nArgument `surface_` must be a pygame.Surface type, got {type(surface_)}.")

    # Get the width and height of the surface (image).
    cdef Py_ssize_t w, h
    w, h = surface_.get_size()

    # Ensure the image dimensions are valid.
    assert w != 0, "Image `w` cannot be null!"
    assert h != 0, "Image `h` cannot be null!"

    # Attempt to get a 3D view of the surface pixels for further processing.
    cdef unsigned char [:, :, :] rgb_array
    try:
        rgb_array = surface_.get_view('3')  # Access surface as a 3D array (RGB channels).
    except Exception as e:
        raise ValueError("\nCannot reference `surface_` pixels into a 3D array.\n %s " % e)

    # Call the GPU-accelerated function (heatmap_cupy) to apply the heatmap effect to the image.
    return heatmap_cupy(cupy.asarray(rgb_array), rgb_array, grid_, block_, invert_)




heatmap_kernel = cp.RawKernel(

    '''
    extern "C" 

    __global__  void heatmap_kernel(
    unsigned char * destination, 
    const unsigned char * source, 
    const unsigned char * map,
    const double map_c, 
    const bool invert, 
    const int w, 
    const int h)
{

    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;

    int index1 = j * h * 3 + i * 3;    

    unsigned int s = source[index1 + 0] + source[index1 + 1] + source[index1 + 2];
    unsigned int index = (unsigned int)((float)s * map_c);  

    if (i < h && j < w) {

        __syncthreads();
        if (invert == 1){
        destination[index1 + 0] = (unsigned char)map[index * 3 + 0];
        destination[index1 + 1] = (unsigned char)map[index * 3 + 1];
        destination[index1 + 2] = (unsigned char)map[index * 3 + 2];
        }
        else {
            destination[index1 + 0] = (unsigned char)map[index * 3 + 2];
            destination[index1 + 1] = (unsigned char)map[index * 3 + 1];
            destination[index1 + 2] = (unsigned char)map[index * 3 + 0];
        }       
    }
}
    ''',
    'heatmap_kernel'
)

@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef inline heatmap_cupy(
        gpu_array,
        rgb_array,
        tuple grid_,
        tuple block_,
        const bint invert_
):
    """
    APPLY HEATMAP EFFECT TO AN IMAGE USING GPU ACCELERATION.

    This function applies a heatmap effect to the given image using GPU-based computations. The effect visualizes 
    intensity or magnitude by coloring the image based on predefined color mappings. Optionally, the color range can 
    be inverted using the `invert_` parameter.

    :param gpu_array: cupy.ndarray; The input image as a 3D array (height x width x 3), representing the RGB values.
    :param rgb_array: unsigned char array; The raw RGB pixel data of the image.
    :param grid_: tuple; Grid configuration (grid_y, grid_x) used for parallel execution on the GPU.
                  The grid values should match the texture and array sizes.
    :param block_: tuple; Block configuration (block_y, block_x) determines the number of threads per block for GPU execution.
                   Maximum threads = block_x * block_y, with a limit of 1024 threads.
    :param invert_: boolean; If `True`, inverts the heatmap effect, reversing the color range.
    :return: pygame.Surface; The modified image with the heatmap effect applied.

    """

    # Retrieve the dimensions (width and height) of the input image (gpu_array).
    cdef Py_ssize_t w, h
    w, h = (<object>gpu_array).shape[:2]

    # Initialize the destination image with the same dimensions and RGB channels.
    destination = cp.zeros((w, h, 3), dtype=cupy.uint8)

    # Set up the factor for the heatmap (not used directly here but could be used for scaling purposes).
    cdef float f_map

    # Call the heatmap kernel to apply the heatmap effect. The kernel is executed in parallel on the GPU.
    heatmap_kernel(
        grid_,
        block_,
        (destination, gpu_array, cupy.asarray(heatmap_array), f_heatmap, int(invert_), w, h)
    )

    # Synchronize the CUDA stream to ensure that all operations are completed before proceeding.
    cp.cuda.Stream.null.synchronize()

    # Convert the resulting image (destination) to bytes, transpose its dimensions for the expected format,
    # and return the image as a pygame surface in RGB format.
    return frombuffer(destination.transpose(1, 0, 2).tobytes(), (w, h), "RGB")




@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef void heatmap_gpu_inplace(
        surface_,
        tuple grid_,
        tuple block_,
        bint invert_ = False
):
    """
    APPLY HEATMAP EFFECT INPLACE TO A PYGAME SURFACE.

    This function applies a heatmap effect to the input surface in place, modifying the pixel data directly 
    without returning a new surface. The heatmap visualization enhances the intensity of the image based on a color 
    mapping, with the option to invert the color effect.

    :param surface_: pygame.Surface; The surface on which the heatmap effect will be applied. The surface must 
                     be a valid pygame surface containing pixel data in a 3D array (RGB channels).
    :param grid_: tuple; Grid configuration (grid_y, grid_x) for parallel GPU execution. This defines how many 
                  blocks will be used for processing the surface, and must match the array and texture sizes.
    :param block_: tuple; Block configuration (block_y, block_x) defines the number of threads per block for 
                   GPU computation. The total number of threads should not exceed 1024 (block_x * block_y).
    :param invert_: boolean; If `True`, the heatmap effect will be inverted, reversing the color range. 
                    Defaults to `False` (no inversion).
    :return: void; This function modifies the `surface_` directly and does not return a value.
    """

    # Check that the input surface is of type pygame.Surface.
    if not PyObject_IsInstance(surface_, pygame.Surface):
        raise TypeError(f"\nArgument `surface_` must be a pygame.Surface type, got {type(surface_)}.")

    # Get the width and height of the surface.
    cdef Py_ssize_t w, h
    w, h = surface_.get_size()

    # Ensure the image dimensions are valid.
    assert w != 0, "Image `w` cannot be null!"
    assert h != 0, "Image `h` cannot be null!"

    # Retrieve the RGB pixel data from the surface and store it in a 3D array.
    cdef unsigned char [:, :, :] rgb_array
    try:
        rgb_array = pixels3d(surface_)  # Convert the surface pixels into a 3D array format (height x width x 3).
    except Exception as e:
        raise ValueError("\nCannot reference `surface_` pixels into a 3D array.\n %s " % e)

    # Apply the heatmap effect in place using GPU-based computation.
    heatmap_cupy_inplace(cupy.asarray(rgb_array), rgb_array, grid_, block_, invert_)


    

@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef inline void heatmap_cupy_inplace(
        gpu_array,
        rgb_array,
        tuple grid_,
        tuple block_,
        bint invert_
):
    """
    APPLY HEATMAP EFFECT INPLACE USING GPU ACCELERATION.

    This function applies the heatmap effect directly to an image (represented by a 3D array) in place using 
    GPU computation. The color intensity is modified based on a heatmap, and the effect can be inverted 
    by passing the `invert_` parameter. This function modifies the input `rgb_array` directly and does not 
    return a new array.

    :param gpu_array : ndarray; 3D array representing the input image on the GPU. The shape of the array 
                        should be (height, width, 3) corresponding to RGB channels.
    :param rgb_array : ndarray; 3D array representing the original RGB image data, which will be modified 
                        in place. The image is processed directly in memory without creating a new output array.
    :param grid_     : tuple; Grid configuration for the GPU kernel, defining the number of blocks in each dimension 
                        (grid_y, grid_x). This must align with the texture and array sizes for correct processing.
    :param block_    : tuple; Block configuration for the GPU kernel, defining the number of threads per block 
                        (block_y, block_x). The maximum number of threads should not exceed 1024.
    :param invert_   : boolean; If `True`, the heatmap effect will be inverted (i.e., the color range will be reversed). 
                        Defaults to `False`, meaning no inversion.
    :return: void; This function modifies the `rgb_array` in place and does not return any value.
    """

    # Extract the width and height of the image from the gpu_array (height, width, 3).
    cdef Py_ssize_t w, h
    w, h = (<object>gpu_array).shape[:2]

    # Initialize an empty destination array (same dimensions as gpu_array) for the processed image.
    destination = cp.zeros((w, h, 3), dtype=cupy.uint8)

    cdef float f_map  # This variable can be used for a scaling factor or heatmap parameter if needed.

    # Launch the heatmap kernel on the GPU with the provided grid and block configuration.
    # The kernel will modify the 'destination' array in place with the heatmap effect.
    heatmap_kernel(
        grid_,
        block_,
        (destination, gpu_array, cupy.asarray(heatmap_array), f_heatmap, int(invert_), w, h)
    )

    # Synchronize the GPU to ensure that all computations are completed before proceeding.
    cp.cuda.Stream.null.synchronize()

    # After the heatmap effect is applied, copy the resulting data from 'destination' back into 'rgb_array' in place.
    rgb_array[:, :, 0] = destination[:, :, 0].get()  # Update the red channel.
    rgb_array[:, :, 1] = destination[:, :, 1].get()  # Update the green channel.
    rgb_array[:, :, 2] = destination[:, :, 2].get()  # Update the blue channel.




# --------------------------------------------------------------------------------------------------------------------

@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef inline predator_gpu(
        surface_,
        tuple grid_,
        tuple block_,
        unsigned int bloom_smooth    = 50,
        unsigned int bloom_threshold = 50,
        bint inv_colormap            = False,
        int blend                    = pygame.BLEND_RGB_ADD,
        bint bloom_flag              = pygame.BLEND_RGB_ADD
):
    """
    Applies a predator vision effect using GPU acceleration.

    Parameters:
    surface_ (pygame.Surface): The input surface to process.
    grid_ (tuple): Grid dimensions for CUDA kernel execution.
    block_ (tuple): Block dimensions for CUDA kernel execution.
    bloom_smooth (unsigned int, optional): Smoothing factor for bloom effect. Default is 50.
    bloom_threshold (unsigned int, optional): Intensity threshold for bloom effect. Default is 50.
    inv_colormap (bint, optional): Whether to invert the colormap. Default is False.
    blend (int, optional): Blending mode for final rendering. Default is pygame.BLEND_RGB_ADD.
    bloom_flag (bint, optional): Flag to enable bloom effect. Default is pygame.BLEND_RGB_ADD.

    Returns:
    pygame.Surface: Processed surface with the applied predator vision effect.
    """

    # Apply bloom effect for enhanced lighting and visibility
    shader_bloom_fast1(surface_, smooth_=bloom_smooth, threshold_=bloom_threshold, flag_=bloom_flag, saturation_=True)

    cdef unsigned char [:, :, :] cpu_array
    try:
        # Obtain a 3D pixel view of the surface
        cpu_array = surface_.get_view('3')
        gpu_array = cupy.asarray(cpu_array)  # Convert to a CuPy GPU array
    except Exception as e:
        raise ValueError("\nCannot reference source pixels into a 3D array.\n %s " % e)

    cdef Py_ssize_t w, h
    w, h = surface_.get_size()  # Get surface dimensions

    # Create a copy of the GPU array to preserve the original data
    gpu_array_copy = gpu_array.copy()
    
    # Apply Sobel edge detection on the copied array
    # sobel_cupy(w, h, gpu_array_copy)

    # Convert the original GPU array to grayscale for further processing
    grayscale_lum_cupy(gpu_array)

    # Apply a heatmap effect to highlight edges and features
    heatmap_cupy_inplace(gpu_array, cpu_array, grid_, block_, invert_=inv_colormap)
    
    # Convert processed GPU array back to a pygame surface
    surf = frombuffer(gpu_array_copy.transpose(1, 0, 2).tobytes(), (w, h), "RGB")
    
    # Clean up allocated GPU memory
    del gpu_array_copy
    del cpu_array

    # Blend the processed surface onto the original
    surface_.blit(surf, (0, 0), special_flags=blend)
    
    return surface_.convert()


# -------------------------------------------------------------------------------------------------------


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef inline downscale_surface_gpu(
        surface_,
        tuple grid_,
        tuple block_,
        const float zoom,
        const int w2,
        const int h2
):
    """
    Downscale a pygame surface using CUDA on CuPy.

    This function extracts pixel data from a pygame surface, transfers it to the GPU,
    applies a downscaling operation using a CUDA kernel, and returns the processed image.

    Parameters:
    surface_ : pygame.Surface
        The input surface to be downscaled.
    grid_ : tuple
        CUDA grid dimensions for kernel execution.
    block_ : tuple
        CUDA block dimensions for kernel execution.
    zoom : float
        Scaling factor for downscaling (must be in the range [0.0, 0.9999]).
    w2 : int
        Target width of the downscaled image.
    h2 : int
        Target height of the downscaled image.

    Returns:
    bytes
        The downscaled image in RGB format as a byte buffer.
    """

    # Ensure the input is a pygame.Surface instance
    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument `surface_` must be a pygame.Surface type, got %s " % type(surface_)

    cdef Py_ssize_t w, h
    w, h = surface_.get_size()

    # Validate that the image dimensions are not zero
    assert w != 0, "Image `w` cannot be null!"
    assert h != 0, "Image `h` cannot be null!"

    # Ensure the zoom factor is within the valid range
    if zoom < 0 or <float> floor(zoom) > <float> 0.9999:
        raise ValueError("Argument zoom must be in range [0.0 ... 0.9999]")

    cdef unsigned char [:, :, :] gpu_array
    try:
        # Obtain a 3D pixel reference from the surface
        gpu_array = surface_.get_view('3')
    except Exception as e:
        raise ValueError("\nCannot reference `surface_` pixels into a 3D array.\n %s " % e)

    # Convert the surface pixel data to a CuPy array and call the GPU downscaling function
    return downscale_surface_cupy(cupy.asarray(gpu_array),
                     grid_, block_, zoom, w2, h2)





downscale_surface_kernel = cp.RawKernel(

    '''

    extern "C" 

    __global__  void downscale_surface_kernel(unsigned char * destination, const unsigned char * source,
        const int w, const int h, const int w2, const int h2)
{

    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    float rj = (float)h2/(float)h; 

    int index1 = j * h * 3 + i * 3;   
    int index2 = (int)(j * rj) * h2 * 3 + (int)(i * rj) * 3;
    
    const unsigned int t_max = w * h * 3;  
    unsigned char local_memory[3]; 
   
 
    if (i < h && j < w) {

        const int r = index2 + 0; 
        const int g = index2 + 1;
        const int b = index2 + 2;

        local_memory[0] = source[r];
        local_memory[1] = source[g];
        local_memory[2] = source[b];
        
        __syncthreads();
        destination[index1 + 0] = local_memory[0];
        destination[index1 + 1] = local_memory[1];
        destination[index1 + 2] = local_memory[2]; 
    }


}
    ''',
    'downscale_surface_kernel'
)




@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef inline downscale_surface_cupy(
        gpu_array,
        tuple grid_,
        tuple block_,
        const float zoom,
        const int w2,
        const int h2
):
    """
    Downscale an image using CUDA on CuPy.

    This function takes an input image stored on the GPU, applies a downscaling operation
    using a CUDA kernel, and returns the downscaled image in RGB format.

    Parameters:
    gpu_array : cupy.ndarray
        Input image stored on the GPU (CuPy array) with shape (height, width, channels).
    grid_ : tuple
        CUDA grid dimensions for kernel execution.
    block_ : tuple
        CUDA block dimensions for kernel execution.
    zoom : float
        Scaling factor for downscaling.
    w2 : int
        Target width of the downscaled image.
    h2 : int
        Target height of the downscaled image.

    Returns:
    bytes
        The downscaled image in RGB format as a byte buffer.
    """
    cdef:
        Py_ssize_t w, h

    # Get the width and height of the input image
    w, h = gpu_array.shape[:2]

    # Create an empty destination array on the GPU for the downscaled image
    destination = cupy.zeros((w2, h2, 3), dtype=cupy.uint8)

    # Launch the CUDA kernel to perform the downscaling operation
    downscale_surface_kernel(
        grid_,
        block_,
        (destination, gpu_array, w2, h2, w, h, zoom)
    )

    # Ensure all CUDA operations are synchronized before proceeding
    cp.cuda.Stream.null.synchronize()

    # Convert the resulting GPU array to an RGB byte buffer and return it
    return frombuffer(destination.transpose(1, 0, 2).tobytes(), (w2, h2), "RGB")














