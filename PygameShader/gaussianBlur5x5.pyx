# cython: binding=False, boundscheck=False, wraparound=False, nonecheck=False, cdivision=True,
# cython: optimize.use_switch=True
# encoding: utf-8

"""
                 GNU GENERAL PUBLIC LICENSE
                       Version 3, 29 June 2007

 Copyright (C) 2007 Free Software Foundation, Inc. <https://fsf.org/>
 Everyone is permitted to copy and distribute verbatim copies
 of this license document, but changing it is not allowed.

Copyright Yoann Berenguer
"""

# NUMPY IS REQUIRED
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



DEF SCHEDULE = 'static'

DEF OPENMP = True
# num_threads – The num_threads argument indicates how many threads the team should consist of.
# If not given, OpenMP will decide how many threads to use.
# Typically this is the number of cores available on the machine. However,
# this may be controlled through the omp_set_num_threads() function,
# or through the OMP_NUM_THREADS environment variable.
DEF THREAD_NUMBER = 1
if OPENMP is True:
    DEF THREAD_NUMBER = 8


cpdef blur5x5_array24_inplace_c(unsigned char [:, :, :] rgb_array_):
    """
    # Gaussian kernel 5x5
        # |1   4   6   4  1|
        # |4  16  24  16  4|
        # |6  24  36  24  6|  x 1/256
        # |4  16  24  16  4|
        # |1  4    6   4  1|
    This method is using convolution property and process the image in two passes,
    first the horizontal convolution and last the vertical convolution
    pixels convoluted outside image edges will be set to adjacent edge value

    :param rgb_array_: numpy.ndarray type (w, h, 3) uint8 
    :return: Return 24-bit a numpy.ndarray type (w, h, 3) uint8
    """

    cdef int w, h, dim
    try:
        w, h, dim = (<object>rgb_array_).shape[:3]

    except Exception as e:
        raise ValueError('\nArray shape not understood.')

    # kernel 5x5 separable
    cdef:
        float[5] kernel = [1.0/16.0, 4.0/16.0, 6.0/16.0, 4.0/16.0, 1.0/16.0]
        short int kernel_half = <short int>2
        unsigned char [:, :, ::1] convolve = numpy.empty((w, h, 3), dtype=uint8)
        unsigned char [:, :, ::1] convolved = numpy.empty((w, h, 3), dtype=uint8)
        short int kernel_length = <short int>len(kernel)
        int x, y, xx, yy
        float k, r, g, b, s
        char kernel_offset
        unsigned char red, green, blue

    with nogil:
        # horizontal convolution
        for y in prange(0, h, schedule=SCHEDULE, num_threads=THREAD_NUMBER):  # range [0..h-1)

            for x in range(0, w):  # range [0..w-1]

                r, g, b = 0, 0, 0

                for kernel_offset in range(-kernel_half, kernel_half + 1):

                    k = kernel[kernel_offset + kernel_half]

                    xx = x + kernel_offset

                    # check boundaries.
                    # Fetch the edge pixel for the convolution
                    if xx < 0:
                        red, green, blue = rgb_array_[0, y, 0],\
                        rgb_array_[0, y, 1], rgb_array_[0, y, 2]
                    elif xx > (w - 1):
                        red, green, blue = rgb_array_[w-1, y, 0],\
                        rgb_array_[w-1, y, 1], rgb_array_[w-1, y, 2]
                    else:
                        red, green, blue = rgb_array_[xx, y, 0],\
                            rgb_array_[xx, y, 1], rgb_array_[xx, y, 2]

                    r = r + red * k
                    g = g + green * k
                    b = b + blue * k

                convolve[x, y, 0], convolve[x, y, 1], convolve[x, y, 2] = <unsigned char>r,\
                    <unsigned char>g, <unsigned char>b

        # Vertical convolution
        for x in prange(0,  w, schedule=SCHEDULE, num_threads=THREAD_NUMBER):

            for y in range(0, h):
                r, g, b = 0, 0, 0

                for kernel_offset in range(-kernel_half, kernel_half + 1):

                    k = kernel[kernel_offset + kernel_half]
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

                rgb_array_[x, y, 0], rgb_array_[x, y, 1], rgb_array_[x, y, 2] = \
                    <unsigned char>r, <unsigned char>g, <unsigned char>b



cpdef blur5x5_surface24_inplace_c(surface_):
    """
    # Gaussian kernel 5x5
        # |1   4   6   4  1|
        # |4  16  24  16  4|
        # |6  24  36  24  6|  x 1/256
        # |4  16  24  16  4|
        # |1  4    6   4  1|
    This method is using convolution property and process the image in two passes,
    first the horizontal convolution and last the vertical convolution
    pixels convoluted outside image edges will be set to adjacent edge value

    :param surface_: numpy.ndarray type (w, h, 3) uint8 
    :return: Return 24-bit a numpy.ndarray type (w, h, 3) uint8
    """
    cdef unsigned char [:, :, :] rgb_array_
    try:
        rgb_array_ = pixels3d(surface_)
    except:
        raise ValueError(
            'Invalid pygame surface, compatible with 24bit only got %s ' % surface_.get_bitsize())

    cdef int w, h, dim
    try:
        w, h, dim = (<object>rgb_array_).shape[:3]

    except Exception as e:
        raise ValueError('\nArray shape not understood.')

    # kernel 5x5 separable
    cdef:
        float[5] kernel = [1.0/16.0, 4.0/16.0, 6.0/16.0, 4.0/16.0, 1.0/16.0]
        short int kernel_half = <short int>2
        unsigned char [:, :, ::1] convolve = numpy.empty((w, h, 3), dtype=uint8)
        unsigned char [:, :, ::1] convolved = numpy.empty((w, h, 3), dtype=uint8)
        short int kernel_length = <short int>len(kernel)
        int x, y, xx, yy
        float k, r, g, b, s
        char kernel_offset
        unsigned char red, green, blue

    with nogil:
        # horizontal convolution
        for y in prange(0, h, schedule=SCHEDULE, num_threads=THREAD_NUMBER):  # range [0..h-1)

            for x in range(0, w):  # range [0..w-1]

                r, g, b = 0, 0, 0

                for kernel_offset in range(-kernel_half, kernel_half + 1):

                    k = kernel[kernel_offset + kernel_half]

                    xx = x + kernel_offset

                    # check boundaries.
                    # Fetch the edge pixel for the convolution
                    if xx < 0:
                        red, green, blue = rgb_array_[0, y, 0],\
                        rgb_array_[0, y, 1], rgb_array_[0, y, 2]
                    elif xx > (w - 1):
                        red, green, blue = rgb_array_[w-1, y, 0],\
                        rgb_array_[w-1, y, 1], rgb_array_[w-1, y, 2]
                    else:
                        red, green, blue = rgb_array_[xx, y, 0],\
                            rgb_array_[xx, y, 1], rgb_array_[xx, y, 2]

                    r = r + red * k
                    g = g + green * k
                    b = b + blue * k

                convolve[x, y, 0], convolve[x, y, 1], convolve[x, y, 2] = <unsigned char>r,\
                    <unsigned char>g, <unsigned char>b

        # Vertical convolution
        for x in prange(0,  w, schedule=SCHEDULE, num_threads=THREAD_NUMBER):

            for y in range(0, h):
                r, g, b = 0, 0, 0

                for kernel_offset in range(-kernel_half, kernel_half + 1):

                    k = kernel[kernel_offset + kernel_half]
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

                rgb_array_[x, y, 0], rgb_array_[x, y, 1], rgb_array_[x, y, 2] = \
                    <unsigned char>r, <unsigned char>g, <unsigned char>b


cpdef canny_blur5x5_surface24_c(surface_):
    """
    # Gaussian kernel 5x5
        # |2   4   5   4  2|
        # |4   9  12   9  4|
        # |5  12  15  12  5|  x 1/159
        # |4   9  12   9  4|
        # |2   4   5   4  2|
    pixels convoluted outside image edges will be set to adjacent edge value

    :param surface_: Surface, 8, 24-32 bit format
    :return: return a numpy.ndarray (w, h, 3) uint8 with RGB values
    """

    assert isinstance(surface_, Surface),\
        'Argument image must be a valid Surface, got %s ' % type(surface_)

    # kernel definition
    kernel = numpy.array(([2.0,  4.0,  5.0,  4.0,  2.0],
                          [4.0,  9.0, 12.0,  9.0,  4.0],
                          [5.0, 12.0, 15.0, 12.0,  5.0],
                          [4.0,  9.0, 12.0,  9.0,  4.0],
                          [2.0,  4.0,  5.0,  4.0,  2.0])).astype(dtype=float32, order='C')

    cdef int w, h
    w, h = surface_.get_size()

    try:
        rgb_array_ = pixels3d(surface_)
    except (surface_.error, ValueError):
        raise ValueError('\nTexture/image is not compatible.')

    assert w != 0 or h !=0, 'image with incorrect dimensions (w>0, h>0) got (%s, %s) ' % (w, h)

    cdef:
        float kernel_weight = sum(kernel)
        float [:, :] canny_kernel = divide(kernel, 159.0, dtype=float32)
        unsigned char [:, :, :] rgb_array = rgb_array_
        short kernel_half = <short>(len(kernel) >> 1)
        float [:, :, ::1] output_array = empty((w, h, 3), order='C', dtype=float32)
        int x, y, xx, yy
        unsigned short red, green, blue,
        short kernel_offset_y, kernel_offset_x
        float r, g, b, k

    with nogil:

        for x in prange(0, w, schedule=SCHEDULE, num_threads=THREAD_NUMBER):

            for y in range(0, h):

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

                        k = canny_kernel[kernel_offset_y + kernel_half,
                                         kernel_offset_x + kernel_half]

                        r += red * k
                        g += green * k
                        b += blue * k

                if r > 255.0:
                    r = 255.0
                if g > 255.0:
                    g = 255.0
                if b > 255.0:
                    b = 255.0

                output_array[x, y, 0] = r
                output_array[x, y, 1] = g
                output_array[x, y, 2] = b

    return asarray(output_array).astype(dtype=uint8)



cpdef canny_blur5x5_surface32_c(surface_):
    """
    # Gaussian kernel 5x5
        # |2   4   5   4  2|
        # |4   9  12   9  4|
        # |5  12  15  12  5|  x 1/159
        # |4   9  12   9  4|
        # |2   4   5   4  2|
    pixels convoluted outside image edges will be set to adjacent edge value

    :param surface_: Surface, 8, 24-32 bit format
    :return: return a numpy.ndarray (w, h, 3) uint8 with RGB values
    """

    assert isinstance(surface_, Surface), \
        'Argument image must be a valid Surface, got %s ' % type(surface_)

    # kernel definition
    kernel = numpy.array(([2.0,  4.0,  5.0,  4.0,  2.0],
                          [4.0,  9.0, 12.0,  9.0,  4.0],
                          [5.0, 12.0, 15.0, 12.0,  5.0],
                          [4.0,  9.0, 12.0,  9.0,  4.0],
                          [2.0,  4.0,  5.0,  4.0,  2.0])).astype(dtype=float32, order='C')

    cdef int w, h
    w, h = surface_.get_size()

    try:
        rgb_array_ = pixels3d(surface_)
        array_alpha_ = array_alpha(surface_)

    except (surface_.error, ValueError):
        raise ValueError('\nInvalid texture or image. '
                         'This version is compatible with 32-bit image format '
                         'with per-pixel transparency.')

    assert w != 0 or h !=0, 'image with incorrect dimensions (w>0, h>0) got (%s, %s) ' % (w, h)

    cdef:
        float kernel_weight = sum(kernel)
        float [:, :] canny_kernel = divide(kernel, 159.0, dtype=float32)
        unsigned char [:, :, :] rgb_array = rgb_array_
        short kernel_half = <short>(len(kernel) >> 1)
        unsigned char [:, :, :] output_array = empty((h, w, 4), dtype=uint8)
        unsigned char [:, :] alpha = array_alpha_
        int x, y, xx, yy
        unsigned char red, green, blue,
        short kernel_offset_y, kernel_offset_x
        float r, g, b, k

    with nogil:

        for x in prange(0, w, schedule=SCHEDULE, num_threads=THREAD_NUMBER):

            for y in range(0, h):

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

                        k = canny_kernel[kernel_offset_y + kernel_half,
                                         kernel_offset_x + kernel_half]

                        r += red * k
                        g += green * k
                        b += blue * k

                if r > 255.0:
                    r = 255.0
                if g > 255.0:
                    g = 255.0
                if b > 255.0:
                    b = 255.0

                output_array[y, x, 0] = <unsigned char>r
                output_array[y, x, 1] = <unsigned char>g
                output_array[y, x, 2] = <unsigned char>b
                output_array[y, x, 3] = <unsigned char>alpha[x, y]

    # return asarray(output_array).astype(dtype=uint8)
    return frombuffer(output_array, (h, w), "RGBA")
