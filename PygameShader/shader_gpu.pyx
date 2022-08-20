# cython: binding=False, boundscheck=False, wraparound=False, nonecheck=False, cdivision=True,
# cython: optimize.use_switch=True
# cython: warn.maybe_uninitialized=False
# cython: warn.unused=False
# cython: warn.unused_result=False
# cython: warn.unused_arg=False
# cython: language_level=3
# encoding: utf-8



"""
                 GNU GENERAL PUBLIC LICENSE
                       Version 3, 29 June 2007

 Copyright (C) 2007 Free Software Foundation, Inc. <https://fsf.org/>
 Everyone is permitted to copy and distribute verbatim copies
 of this license document, but changing it is not allowed.

Copyright Yoann Berenguer
"""


import warnings

from PygameShader import array2d_normalized_c, filtering24_c, heatmap_convert, shader_rgb_to_yiq_i_comp_inplace, \
    shader_rgb_to_yiq_i_comp_inplace_c, shader_rgb_to_yiq_q_comp_inplace_c
from PygameShader.shader import shader_bloom_fast1, shader_rgb_to_yiq_inplace_c

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
from libc.math cimport sqrt, floor


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
cpdef long long int get_gpu_free_mem():
    return GPU_DEVICE.mem_info[0]

# get max dedicated memory
# total: The total amount of memory, in bytes.
cpdef long long int get_gpu_maxmem():
    return GPU_DEVICE.mem_info[1]

# GPU pci bus id
# Returned identifier string for the device in the following
# format [domain]:[bus]:[device].[function] where domain, bus,
# device, and function are all hexadecimal values.
cpdef str get_gpu_pci_bus_id():
    return GPU_DEVICE.pci_bus_id

# Compute capability of this device.
# The capability is represented by a string containing the major
# index and the minor index. For example, compute capability 3.5
# is represented by the string ‘35’.
cpdef str get_compute_capability():
    return GPU_DEVICE.compute_capability

cpdef unsigned int get_max_grid_per_block():
    return COMPUTE_CAPABILITY[get_compute_capability()]


# USED BY block_grid
cdef get_divisors(int n):
    l = []
    for i in range(1, int(n / 2.0) + 1):
        if n % i == 0:
            l.append(i)
    return l


cpdef block_grid(int w, int h):
    """
    AUTO GRID AND BLOCK FOR GPU 

    :param w: integer; with of the display 
    :param h: integer; height of the display
    :return: tuples; tuple grid (y, x) and tuple block (yy, xx) 
    """

    assert w > 0, "Argument w cannot be < 0"
    assert h > 0, "Argument h cannot be < 0"

    cdef int x, y, xx, yy
    cdef unsigned int max_grid

    a = get_divisors(w)
    b = get_divisors(h)

    a = (w / numpy.array(list(a))).astype(dtype=numpy.int32)
    b = (h / numpy.array(list(b))).astype(dtype=numpy.int32)
    a = numpy.delete(a, numpy.where(a > 32))
    b = numpy.delete(b, numpy.where(b > 32))
    xx = int(a[0])
    yy = int(b[0])
    x = w // xx
    y = h // yy

    assert yy * y == h, \
        "\nInvalid grid %s or block %s values, you may want to set grid & block manually" % ((y, x), (yy, xx))
    assert xx * x == w, \
        "\nInvalid grid %s or block %s values, you may want to set grid & block manually" % ((y, x), (yy, xx))

    return (y, x), (yy, xx)

volume = ["", "K", "M", "G", "T", "P", "E", "Z", "Y"]

def conv(v):
    b = 0
    while v > 1024:
        b += 1
        v /= 1024
    if b > len(volume) - 1:
        b = len(volume)
    return str(round(v, 3))+volume[b]


cpdef block_and_grid_info(int w, int h):
    assert w > 0, "Argument w cannot be < 0"
    assert h > 0, "Argument h cannot be < 0"
    grid, block = block_grid(w, h)

    assert block[0] * grid[0] == h, "\nInvalid grid or block values, you may want to set grid & block manually"
    assert block[1] * grid[1] == w, "\nInvalid grid or block values, you may want to set grid & block manually"

    print("GPU GRID        : (grid_y={grid_y:8f}, grid_x={grid_x:8f})".format(grid_y=grid[0], grid_x=grid[1]))
    print("GPU BLOCK       : (block_y={block_y:8f}, block_x={block_x:8f})".format(block_y=block[0], block_x=block[1]))

cpdef get_gpu_info():
    print("CUPY VERSION           : %s " % CP_VERSION)
    print("GPU MAX GRID PER BLOCK : %s" % get_max_grid_per_block())
    print("GPU FREE MEMORY : (mem={mem:8f}, ({v:5s}))".format(mem=get_gpu_free_mem(), v=conv(get_gpu_free_mem())))
    print("GPU MAX MEMORY  : (mem={mem:8f}, ({v:5s}))".format(mem=get_gpu_maxmem(), v=conv(get_gpu_maxmem())))
    print("GPU PCI BUS ID  : (bus={bus:12s})".format(bus=get_gpu_pci_bus_id()))
    print("GPU CAPABILITY  : (capa={capa:5s})".format(capa=get_compute_capability()))

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef object invert_gpu(surface_):
    """
    SHADER INVERT, 
    
    This shader invert a 32 - 24 bit texture/image  
       
    :param surface_  : pygame.Surface
    :return          : Return a pygame surface (inverted) 
    """

    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)

    try:
        cpu_array = pixels3d(surface_)

    except Exception as e:
        raise ValueError("\nCannot reference source pixels into a 3d array.\n %s " % e)

    cdef:
        Py_ssize_t w, h
    w, h = cpu_array.shape[0], cpu_array.shape[1]

    if w == 0 or h == 0:
        raise ValueError("Surface width and height cannot be null!")

    return invert_cupy(cp.asarray(cpu_array))


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef object invert_cupy(gpu_array_):

    cdef:
        Py_ssize_t w, h
    w, h = gpu_array_.shape[0], gpu_array_.shape[1]

    if w == 0 or h == 0:
        raise ValueError("Surface width and height cannot be null!")

    gpu_array_ = 255 - gpu_array_

    cp.cuda.Stream.null.synchronize()
    surface = frombuffer(gpu_array_.transpose(1, 0, 2).tobytes(), (w, h), "RGB")

    return surface.convert()



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef void invert_inplace_cupy(cpu_array_):

    """
    SHADER INVERT INPLACE, 
    
    This shader invert a 32 - 24 bit texture/image inplace  
       
    :param cpu_array_: numpy.ndarray; numpy array shape (w, h, 3) of type uint8, 
        referencing the surface pixels RGB 
    :return          : void 
    """


    assert PyObject_IsInstance(cpu_array_, numpy.ndarray), \
        "\nArgument gpu_array must be a numpy.ndarray type, got %s " % type(cpu_array_)

    if cpu_array_.dtype != cp.uint8:
        raise ValueError("\nArgument cpu_array_ datatype is invalid, "
                         "expecting numpy.uint8 got %s " % cpu_array_.dtype)

    cdef:
        Py_ssize_t w, h
    w, h = cpu_array_.shape[0], cpu_array_.shape[1]

    if w == 0 or h == 0:
        raise ValueError("Surface width and height cannot be null!")

    gpu_array = cp.asarray(cpu_array_, dtype=cp.uint8)

    gpu_array = (255 - gpu_array).astype(dtype=cp.uint8)

    cp.cuda.Stream.null.synchronize()

    cpu_array_[:, :, 0] = gpu_array[:, :, 0].get()
    cpu_array_[:, :, 1] = gpu_array[:, :, 1].get()
    cpu_array_[:, :, 2] = gpu_array[:, :, 2].get()



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef object sepia_gpu(object surface_):
    """
    SEPIA SHADER, 
    
    Compatible with 32 - 24 bit pygame surface 
    
    :param surface_  : pygame.Surface
    :return          :  Return a pygame.Surface shape (w, h, 3) 
    """

    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)

    try:
        cpu_array_ = pixels3d(surface_)

    except Exception as e:
        raise ValueError("\nCannot reference source pixels into a 3d array.\n %s " % e)

    cdef:
        Py_ssize_t w,h
    w, h = cpu_array_.shape[0], cpu_array_.shape[1]

    if w == 0 or h == 0:
        raise ValueError("Surface width and height cannot be null!")

    gpu_array = cp.asarray(cpu_array_)
    sepia_cupy(gpu_array)

    cp.cuda.Stream.null.synchronize()

    return frombuffer(
        gpu_array.astype(dtype=cp.uint8).transpose(1, 0, 2).tobytes(), (w,h), "RGB").convert()



sepia_kernel = cp.ElementwiseKernel(
    'float32 r, float32 g, float32 b',
    'float32 rr, float32 gg, float32 bb',
    '''

    // SEPIA RGB 
    rr = (r * (float)0.393 + g * (float)0.769 + b * (float)0.189) * (float)255.0;
    gg = (r * (float)0.349 + g * (float)0.686 + b * (float)0.168) * (float)255.0;
    bb = (r * (float)0.272 + g * (float)0.534 + b * (float)0.131) * (float)255.0;

    // CAP all the values in range [0...255] (uint8)
    if ( rr > (float)255.0) {rr = (float)255.0;} else if ( rr < 0 ) {rr = (float)0.0;}
    if ( gg > (float)255.0) {gg = (float)255.0;} else if ( gg < 0 ) {gg = (float)0.0;}
    if ( bb > (float)255.0) {bb = (float)255.0;} else if ( bb < 0 ) {bb = (float)0.0;}

    ''', 'sepia_kernel'
)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef object sepia_cupy(gpu_array_):
    """
    
    :param gpu_array_: cupy.ndarray; Array shape (w, h, 3) type uint8 
    :return          : a numpy.ndarray shape (w, h, 3) of type uint8 (located on the CPU side)
    """

    rr = (gpu_array_[:, :, 0] / <float>255.0).astype(dtype=cp.float32)
    gg = (gpu_array_[:, :, 1] / <float>255.0).astype(dtype=cp.float32)
    bb = (gpu_array_[:, :, 2] / <float>255.0).astype(dtype=cp.float32)

    gpu_array_[:, :, 0], \
    gpu_array_[:, :, 1], \
    gpu_array_[:, :, 2] = sepia_kernel(rr, gg, bb)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef void sepia_inplace_cupy(cpu_array_):
    """
    SEPIA INPLACE  
    
    Compatible with 32 - 24 bit image format 
    
    :param cpu_array_ : numpy.ndarray; Array shape (w, h, 3) type uint8 containing RGB pixels
    :return           : void 
    """

    assert PyObject_IsInstance(cpu_array_, numpy.ndarray), \
        "\nArgument a numpy.ndarray type, got %s " % type(cpu_array_)

    if cpu_array_.dtype != cp.uint8:
        raise ValueError("\nArgument cpu_array_ datatype is invalid, "
                         "expecting numpy.uint8 got %s " % cpu_array_.dtype)

    cdef Py_ssize_t w, h
    w, h = cpu_array_.shape[0], cpu_array_.shape[1]

    if w == 0 or h == 0:
        raise ValueError("Surface width and height cannot be null!")

    gpu_array = cp.asarray(cpu_array_, dtype=cp.float32)

    rr = gpu_array[:, :, 0] / <float>255.0
    gg = gpu_array[:, :, 1] / <float>255.0
    bb = gpu_array[:, :, 2] / <float>255.0

    rr, gg, bb = sepia_kernel(rr, gg, bb)

    cpu_array_[:, :, 0] = rr.astype(cp.uint8).get()
    cpu_array_[:, :, 1] = gg.astype(cp.uint8).get()
    cpu_array_[:, :, 2] = bb.astype(cp.uint8).get()





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


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef object bpf_gpu(
        object surface_,
        unsigned int threshold_ = 128
):
    """
    BRIGHT PASS FILTER (ELEMENTWISEKERNEL)
    
    :param surface_  : pygame.Surface
    :param threshold_: integer; Threshold value in range [0...255]. 
    :return: Return a pygame.Surface with PBF effect
    """
    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)

    try:
        cpu_array_ = pixels3d(surface_)

    except Exception as e:
        raise ValueError("\nCannot reference source pixels into a 3d array.\n %s " % e)

    cdef:
        Py_ssize_t w, h
    w, h = cpu_array_.shape[0], cpu_array_.shape[1]

    if w == 0 or h == 0:
        raise ValueError("Surface width and height cannot be null!")

    assert 0 <= threshold_ <= 255, "Argument threshold must be in range [0 ... 255]"

    return frombuffer(bpf_cupy(
        cp.asarray(cpu_array_), threshold_).astype(dtype=cp.uint8).tobytes(), (w, h), "RGB").convert()



bpf_kernel = cp.ElementwiseKernel(
    'uint8 r, uint8 g, uint8 b, float32 threshold_',
    'uint8 rr, uint8 gg, uint8 bb',
    '''

    // ITU-R BT.601 luma coefficients
    float lum = r * 0.299f + g * 0.587f + b * 0.114f;    
    if (lum > threshold_) {
        float c = (float)(lum - threshold_) / (lum+1.0f);
        rr = (unsigned char)(max(r * c, 0.0f));
        gg = (unsigned char)(max(g * c, 0.0f));
        bb = (unsigned char)(max(b * c, 0.0f));   
    } 
     else {
        rr = (unsigned char)0;
        gg = (unsigned char)0;
        bb = (unsigned char)0;                    
    }

    ''', 'bpf_kernel'
)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef object bpf_cupy(gpu_array_, unsigned int threshold_):



    gpu_array_[:, :, 0], gpu_array_[:, :, 1], gpu_array_[:, :, 2] = \
        bpf_kernel(gpu_array_[:, :, 0], gpu_array_[:, :, 1], gpu_array_[:, :, 2], <float>threshold_)

    cp.cuda.Stream.null.synchronize()

    return gpu_array_.transpose(1, 0, 2)



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef object bpf1_gpu(
        object surface_,
        grid_,
        block_,
        unsigned int threshold_ = 128
):
    """
    BRIGHT PASS FILTER (RAWKERNEL)
    This version is faster than bpf_gpu

    :param surface_          : pygame.Surface
    :param grid_             : tuple; grid values (grid_y, grid_x) e.g (25, 25). The grid values and block values must 
        match the texture and array sizes. 
    :param block_            : tuple; block values (block_y, block_x) e.g (32, 32). Maximum threads is 1024.
        Max threads = block_x * block_y
    :param threshold_: integer; Threshold value in range [0...255].
    :return: Return a pygame.Surface with PBF effect
    """
    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface, got %s " % type(surface_)

    assert 0 <= threshold_ <= 255, "Argument threshold must be in range [0 ... 255]"

    try:
        cpu_array_ = pixels3d(surface_)

    except Exception as e:
        raise ValueError("\nCannot reference source pixels into a 3d array.\n %s " % e)

    cdef:
        Py_ssize_t w, h
    w, h = cpu_array_.shape[0], cpu_array_.shape[1]

    if w == 0 or h == 0:
        raise ValueError("Surface width and height cannot be null!")

    return bpf1_cupy(cp.asarray(cpu_array_), threshold_, grid_, block_)

bpf_kernel1 = cp.RawKernel(
    r'''
    extern "C" __global__
    
    void bpf_kernel1(unsigned char * current, unsigned char * previous,
        const int w, const int h, const double threshold)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        int j = blockIdx.y * blockDim.y + threadIdx.y;

        const int t_max  = w * h;    
        const int index  = j * h + i;
        const int index1 = j * h * 3 + i * 3;

        __syncthreads();

        if (index > 0 && index < t_max){            

            float r  = (float)previous[index1    ];
            float g  = (float)previous[index1 + 1];
            float b  = (float)previous[index1 + 2];

             // ITU-R BT.601 luma coefficients
            float lum = r * 0.299f + g * 0.587f + b * 0.114f;    
            if (lum > threshold) {
                float c = (float)((lum - threshold) / (lum +1));
                current[index1    ] = (unsigned char)(r * c);
                current[index1 + 1] = (unsigned char)(g * c);
                current[index1 + 2] = (unsigned char)(b * c);   
            } 
             else {
                current[index1    ] = (unsigned char)0;
                current[index1 + 1] = (unsigned char)0;
                current[index1 + 2] = (unsigned char)0;                    
            }

            __syncthreads();

        }
    }
    ''',
    'bpf_kernel1'
)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef object bpf1_cupy(object gpu_array_, unsigned int threshold_, object grid_, object block_):

    cdef:
        Py_ssize_t w, h
    w, h = gpu_array_.shape[0], gpu_array_.shape[1]

    cdef:
        current = cupy.empty((w, h, 3), dtype=cupy.uint8)

    bpf_kernel1(
        grid_,
        block_,
        (current, gpu_array_, w, h, <float>threshold_))

    cp.cuda.Stream.null.synchronize()

    return frombuffer(current.transpose(1, 0, 2).tobytes(), (w, h), "RGB").convert()





@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef object grayscale_gpu(object surface_):
    """
    GRAYSCALE  
    
    Compatible with format 32 - 24 bit 

    :param surface_ : pygame.Surface
    :return         : Return a grayscale surface
    """
    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface, got %s " % type(surface_)

    try:
        cpu_array_ = pixels3d(surface_)

    except Exception as e:
        raise ValueError("\nCannot reference source pixels into a 3d array.\n %s " % e)

    cdef:
        Py_ssize_t w, h
    w, h = cpu_array_.shape[0], cpu_array_.shape[1]

    if w == 0 or h == 0:
        raise ValueError("Surface width and height cannot be null!")

    return frombuffer(grayscale_cupy(cp.asarray(cpu_array_)).astype(
        dtype=cp.uint8).transpose(1, 0, 2).tobytes(), (w, h), "RGB").convert()


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef object grayscale_cupy(gpu_array_):

    gpu_array_[:, :, 0], gpu_array_[:, :, 1], gpu_array_[:, :, 2] = \
        grey_kernel(gpu_array_[:, :, 0], gpu_array_[:, :, 1], gpu_array_[:, :, 2])

    cp.cuda.Stream.null.synchronize()

    return gpu_array_




@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef object grayscale_lum_gpu(object surface_):
    """
    GRAYSCALE  

    Compatible with format 32 - 24 bit
    
    :param surface_  : pygame.Surface
    :return          : Return a pygame.Surface with grayscale effect
    """
    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface, got %s " % type(surface_)

    try:
        cpu_array_ = pixels3d(surface_)

    except Exception as e:
        raise ValueError("\nCannot reference source pixels into a 3d array.\n %s " % e)

    cdef:
        Py_ssize_t w, h
    w, h = cpu_array_.shape[0], cpu_array_.shape[1]

    if w == 0 or h == 0:
        raise ValueError("Surface width and height cannot be null!")

    return frombuffer(grayscale__lum_cupy(cp.asarray(cpu_array_)).astype(
        dtype=cp.uint8).transpose(1, 0, 2).tobytes(), (w, h), "RGB").convert()

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef object grayscale__lum_cupy(gpu_array_):

    gpu_array_[:, :, 0], gpu_array_[:, :, 1], gpu_array_[:, :, 2] = \
        grey_luminosity_kernel(gpu_array_[:, :, 0], gpu_array_[:, :, 1], gpu_array_[:, :, 2])

    cp.cuda.Stream.null.synchronize()

    return gpu_array_



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef object median_gpu(
        object surface_,
        unsigned int size_ = 5
):
    """

    MEDIAN FILTER (MEDIAN_FILTER)
    
    Compatible with format 32 - 24 bit
    Create a median filter effect using the method median_filter

    :param surface_  : pygame.Surface
    :param size_     : integer; Neighbours included in the median calculation.
    :return          : Return a pygame.Surface with a median effect
    """

    assert size_ > 0, "\nArgument size_ must be >0"

    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface, got %s " % type(surface_)

    try:
        cpu_array_ = pixels3d(surface_)

    except Exception as e:
        raise ValueError("\nCannot reference source pixels into a 3d array.\n %s " % e)

    cdef:
        Py_ssize_t w, h
    w, h = cpu_array_.shape[0], cpu_array_.shape[1]

    if w == 0 or h == 0:
        raise ValueError("Surface width and height cannot be null!")

    return frombuffer(median_cupy(cp.asarray(cpu_array_), size_).astype(
        cp.uint8).transpose(1, 0, 2).tobytes(), (w, h), "RGB").convert()



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef object median_cupy(gpu_array_, unsigned int size_=5):

    gpu_array_[:, :, 0] = cupyx.scipy.ndimage.median_filter(gpu_array_[:, :, 0], size_)
    gpu_array_[:, :, 1] = cupyx.scipy.ndimage.median_filter(gpu_array_[:, :, 1], size_)
    gpu_array_[:, :, 2] = cupyx.scipy.ndimage.median_filter(gpu_array_[:, :, 2], size_)

    cp.cuda.Stream.null.synchronize()

    return gpu_array_


median_kernel = cp.RawKernel(
    '''extern "C" __global__
    void median_kernel(double* buffer, int filter_size,
                     double* return_value)
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

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef object median1_gpu(
        object surface_,
        unsigned int size_ = 5
):
    """
    MEDIAN FILTER (GENERIC_FILTER)

    Compatible with format 32 - 24 bit
    Create a median filter effect using the method generic_filter.
    The generic filter accept a kernel with buffer type double (low performance) e.g :
    
    void median_kernel(double* buffer, int filter_size,
                     double* return_value)
    
    :param surface_  : pygame.Surface
    :param size_     : integer; Neighbours included in the median calculation.
    :return          : Return a pygame.Surface with a median effect

    """
    assert size_ > 0, "\nArgument size_ must be >0"

    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface, got %s " % type(surface_)

    try:
        cpu_array_ = pixels3d(surface_)

    except Exception as e:
        raise ValueError("\nCannot reference source pixels into a 3d array.\n %s " % e)

    cdef:
        Py_ssize_t w, h
    w, h = cpu_array_.shape[0], cpu_array_.shape[1]

    if w == 0 or h == 0:
        raise ValueError("Surface width and height cannot be null!")

    return frombuffer(median1_cupy(cp.asarray(cpu_array_).astype(dtype=cp.uint8), size_).astype(
        cp.uint8).transpose(1, 0, 2).tobytes(), (w, h), "RGB").convert()

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef object median1_cupy(gpu_array_, unsigned int size_=5):

    r = cupyx.scipy.ndimage.generic_filter(
        gpu_array_[:, :, 0], median_kernel, size_).astype(dtype=cp.uint8)
    g = cupyx.scipy.ndimage.generic_filter(
        gpu_array_[:, :, 1], median_kernel, size_).astype(dtype=cp.uint8)
    b = cupyx.scipy.ndimage.generic_filter(
        gpu_array_[:, :, 2], median_kernel, size_).astype(dtype=cp.uint8)

    gpu_array_[:, :, 0],\
    gpu_array_[:, :, 1],\
    gpu_array_[:, :, 2] = r, g, b

    cp.cuda.Stream.null.synchronize()

    return gpu_array_



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef object gaussian_5x5_gpu(object surface_):
    """
    GAUSSIAN BLUR KERNEL 5x5
    
    Compatible with format 32 - 24 bit
    Convolve RGB channels separately with kernel 5x5
    
    :param surface_  : pygame.Surface
    :return          : Return a pygame.Surface with the gaussian effect
    """

    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface, got %s " % type(surface_)

    try:
        cpu_array_ = pixels3d(surface_)

    except Exception as e:
        raise ValueError("\nCannot reference source pixels into a 3d array.\n %s " % e)

    cdef:
        Py_ssize_t w, h
    w, h = cpu_array_.shape[0], cpu_array_.shape[1]

    if w == 0 or h == 0:
        raise ValueError("Surface width and height cannot be null!")

    return gaussian_5x5_cupy(cp.asarray(cpu_array_))


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef object gaussian_5x5_cupy(gpu_array_):

    cdef:
        Py_ssize_t w, h
    w, h = gpu_array_.shape[0], gpu_array_.shape[1]

    cdef:
        r = cp.empty((w, h), dtype=cp.uint8)
        g = cp.empty((w, h), dtype=cp.uint8)
        b = cp.empty((w, h), dtype=cp.uint8)

    r = gpu_array_[:, :, 0]
    g = gpu_array_[:, :, 1]
    b = gpu_array_[:, :, 2]

    # Gaussian kernel 5x5
    k = cp.array([[1,   4,   6,   4,  1],
    [4,  16,  24,  16,  4],
    [6,  24,  36,  24,  6],
    [4,  16,  24,  16,  4],
    [1,  4,    6,   4,  1]], dtype=cp.float32) * <float>1.0/<float>256.0

    r = cupyx.scipy.ndimage.convolve(r, k, mode='constant', cval=0.0)
    g = cupyx.scipy.ndimage.convolve(g, k, mode='constant', cval=0.0)
    b = cupyx.scipy.ndimage.convolve(b, k, mode='constant', cval=0.0)

    gpu_array_[:, :, 0], \
    gpu_array_[:, :, 1], \
    gpu_array_[:, :, 2] = r, g, b

    cp.cuda.Stream.null.synchronize()

    return frombuffer(gpu_array_.astype(
        cp.uint8).transpose(1, 0, 2).tobytes(), (w, h), "RGB").convert()


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef object gaussian_3x3_gpu(object surface_):
    """
    GAUSSIAN BLUR KERNEL 3x3
    
    Compatible with format 32 - 24 bit
    Convolve RGB channels separately with kernel 3x3

    :param surface_  : pygame.Surface
    :return          : pygame.Surface with the gaussian effect
    """

    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface, got %s " % type(surface_)

    try:
        cpu_array_ = pixels3d(surface_)

    except Exception as e:
        raise ValueError("\nCannot reference source pixels into a 3d array.\n %s " % e)

    cdef:
        Py_ssize_t w, h
    w, h = cpu_array_.shape[0], cpu_array_.shape[1]

    if w == 0 or h == 0:
        raise ValueError("Surface width and height cannot be null!")

    return gaussian_3x3_cupy(cp.asarray(cpu_array_))


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef object gaussian_3x3_cupy(gpu_array_):

    cdef:
        Py_ssize_t w, h
    w, h = gpu_array_.shape[0], gpu_array_.shape[1]

    cdef:
        r = cp.empty((w, h), dtype=cp.uint8)
        g = cp.empty((w, h), dtype=cp.uint8)
        b = cp.empty((w, h), dtype=cp.uint8)

    r = gpu_array_[:, :, 0]
    g = gpu_array_[:, :, 1]
    b = gpu_array_[:, :, 2]

    # Gaussian kernel 3x3
    k = cp.array([[1, 2, 1 ],
                  [2, 4, 2],
                  [1, 2, 1]], dtype=cp.float32) * <float>1.0 / <float>16.0

    r = cupyx.scipy.ndimage.convolve(r, k, mode='constant', cval=0.0)
    g = cupyx.scipy.ndimage.convolve(g, k, mode='constant', cval=0.0)
    b = cupyx.scipy.ndimage.convolve(b, k, mode='constant', cval=0.0)

    gpu_array_[:, :, 0], \
    gpu_array_[:, :, 1], \
    gpu_array_[:, :, 2] = r, g, b

    cp.cuda.Stream.null.synchronize()

    return frombuffer(gpu_array_.astype(
        cp.uint8).transpose(1, 0, 2).tobytes(), (w, h), "RGB").convert()


sobel_kernel = cp.RawKernel(
    '''   d
    extern "C" __global__
    
    __constant__ double gx[9] = {1.0, 2.0, 1.0, 0.0, 0.0, 0.0, -1.0, -2.0, -1.0};
    __constant__ double gy[9] = {1.0, 0.0, -1.0, 2.0, 0.0, -2.0, 1.0, 0.0, -1.0};   
    
    void sobel_kernel(double* buffer, int filter_size,
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
  
    magnitude = sqrt(s_x * s_x + s_y * s_y);
    
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
    'sobel_kernel'
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

    magnitude = sqrt(s_x * s_x + s_y * s_y);

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


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef object sobel_gpu(object surface_):
    """
    SOBEL EDGE DETECTION 
    
    Compatible with format 32 - 24 bit
    The image must be grayscale before creating the gpu_array_
    This algorithm used the red channel only to generate the sobel effect, as 
    the image is grayscale we could also used the green or blue channel to 
    create the effect. 
    If the image is not grayscale, the result might be slightly different from 
    the grayscale model as RGB channels can have different intensity. 

    
    :param surface_  : pygame.Surface
    :return          : Return a pygame.Surface with the sobel effect
    """

    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface, got %s " % type(surface_)

    try:
        cpu_array_ = pixels3d(surface_)

    except Exception as e:
        raise ValueError("\nCannot reference source pixels into a 3d array.\n %s " % e)

    cdef:
        Py_ssize_t w, h
    w, h = cpu_array_.shape[0], cpu_array_.shape[1]

    if w == 0 or h == 0:
        raise ValueError("Surface width and height cannot be null!")

    return sobel_cupy(cp.asarray(cpu_array_))


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef object sobel_cupy(gpu_array_):

    cdef:
        Py_ssize_t w, h
    w, h = gpu_array_.shape[0], gpu_array_.shape[1]

    cdef:
        r = cp.empty((w, h), dtype=cp.uint8)

    r = gpu_array_[:, :, 0]

    sobel2d_r = cupyx.scipy.ndimage.generic_filter(
        r, sobel_kernel, 3).astype(dtype=cp.uint8)

    cp.cuda.Stream.null.synchronize()

    gpu_array_[:, :, 0], \
    gpu_array_[:, :, 1], \
    gpu_array_[:, :, 2] = sobel2d_r, sobel2d_r, sobel2d_r

    return frombuffer(gpu_array_.astype(
        cp.uint8).transpose(1, 0, 2).tobytes(), (w, h), "RGB").convert()


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef object prewitt_gpu(object surface_):
    """
    PREWITT EDGE DETECTION
    
    Compatible with format 32 - 24 bit
    The image must be grayscale before creating the gpu_array_
    This algorithm used the red channel only to generate the sobel effect, as 
    the image is grayscale we could also used the green or blue channel to 
    create the prewitt image. 
    If the image is not grayscale, the result might be slightly different from 
    the grayscale model as RGB channels can have different intensity. 
    
    
    :param surface_  : pygame.Surface
    :return          : Return a pygame.Surface with the prewitt effect
    """

    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface, got %s " % type(surface_)

    try:
        cpu_array_ = pixels3d(surface_)

    except Exception as e:
        raise ValueError("\nCannot reference source pixels into a 3d array.\n %s " % e)

    cdef:
        Py_ssize_t w, h
    w, h = cpu_array_.shape[0], cpu_array_.shape[1]

    if w == 0 or h == 0:
        raise ValueError("Surface width and height cannot be null!")

    return prewitt_cupy(cp.asarray(cpu_array_))


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef object prewitt_cupy(gpu_array_):
    cdef:
        Py_ssize_t w, h
    w, h = gpu_array_.shape[0], gpu_array_.shape[1]

    cdef:
        r = cp.empty((w, h), dtype=cp.float32)

    r = gpu_array_[:, :, 0]

    sobel2d_r = cupyx.scipy.ndimage.generic_filter(r, prewitt_kernel, 3).astype(dtype=cp.uint8)

    cp.cuda.Stream.null.synchronize()

    gpu_array_[:, :, 0], \
    gpu_array_[:, :, 1], \
    gpu_array_[:, :, 2] = sobel2d_r, sobel2d_r, sobel2d_r

    return frombuffer(gpu_array_.astype(
        cp.uint8).transpose(1, 0, 2).tobytes(), (w, h), "RGB").convert()


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef object canny_gpu(object surface_):
    """
    CANNY EDGE DETECTION 
    
    Compatible with format 32 - 24 bit
    The image must be grayscale before creating the gpu_array_
    This algorithm used the red channel only to generate the sobel effect, as 
    the image is grayscale we could also used the green or blue channel to 
    create the prewitt image. 
    If the image is not grayscale, the result might be slightly different from 
    the grayscale model as RGB channels can have different intensity. 
    
    
    :param surface_  : pygame.Surface
    :return          : Return a pygame.Surface with the canny effect
    """

    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface, got %s " % type(surface_)

    try:
        cpu_array_ = pixels3d(surface_)

    except Exception as e:
        raise ValueError("\nCannot reference source pixels into a 3d array.\n %s " % e)

    cdef:
        Py_ssize_t w, h
    w, h = cpu_array_.shape[0], cpu_array_.shape[1]

    if w == 0 or h == 0:
        raise ValueError("Surface width and height cannot be null!")

    return canny_cupy(cp.asarray(cpu_array_))


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef object canny_cupy(gpu_array_):

    cdef:
        Py_ssize_t w, h
    w, h = gpu_array_.shape[0], gpu_array_.shape[1]

    cdef:
        r = cp.empty((w, h), dtype=cp.float32)

    r = gpu_array_[:, :, 0]

    # Gaussian kernel 5x5
    k = cp.array([[2, 4, 5, 4, 2, ],
                  [4, 9, 12, 9, 4],
                  [5, 12, 15, 12, 5],
                  [4, 9, 12, 9, 4],
                  [2, 4, 5, 4, 2]], dtype=cp.float32) * <float>1.0 / <float>256.0

    r = cupyx.scipy.ndimage.convolve(r, k, mode='constant', cval=0.0)

    sobel2d_r = cupyx.scipy.ndimage.generic_filter(r, sobel_kernel, 3).astype(dtype=cp.uint8)

    gpu_array_[:, :, 0], \
    gpu_array_[:, :, 1], \
    gpu_array_[:, :, 2] = sobel2d_r, sobel2d_r, sobel2d_r

    cp.cuda.Stream.null.synchronize()

    return frombuffer(gpu_array_.astype(
        cp.uint8).transpose(1, 0, 2).tobytes(), (w, h), "RGB").convert()




color_reduction_kernel = cp.ElementwiseKernel(
    'uint8 r, uint8 g, uint8 b, int16 color_number',
    'uint8 rr, uint8 gg, uint8 bb',
    '''
    
    const float f = 255.0f / (float)color_number;
    const float c1 = (float)color_number / 255.0f;
    
    rr = (unsigned char)((int)((float)round(c1 * (float)r) * f));
    gg = (unsigned char)((int)((float)round(c1 * (float)g) * f));
    bb = (unsigned char)((int)((float)round(c1 * (float)b) * f));
 
    ''', 'color_reduction_kernel'
)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef object color_reduction_gpu(
        object surface_,
        int color_number = 8):
    """
    COLOR REDUCTION SHADER
    
    Compatible with format 32 - 24 bit
    Decrease the amount of colors in the display or texture.
    The method of color reduction is very simple: every color of the original picture is replaced
    by an appropriate color from the limited palette that is accessible.
 
    :param surface_    : pygame.Surface
    :param color_number: integer; Number of colors ^2
    :return            : Return a pygame.Surface with color reduction effect
    """

    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface, got %s " % type(surface_)

    try:
        cpu_array_ = pixels3d(surface_)

    except Exception as e:
        raise ValueError("\nCannot reference source pixels into a 3d array.\n %s " % e)

    cdef:
        Py_ssize_t w, h
    w, h = cpu_array_.shape[0], cpu_array_.shape[1]

    if w == 0 or h == 0:
        raise ValueError("Surface width and height cannot be null!")

    assert color_number > 0, "\nArgument color_number cannot be < 0"

    return color_reduction_cupy(cp.asarray(cpu_array_), color_number)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef object color_reduction_cupy(
        object gpu_array_,
        int color_number
):
    cdef:
        Py_ssize_t w, h
    w, h = gpu_array_.shape[0], gpu_array_.shape[1]

    gpu_array_[:, :, 0], \
    gpu_array_[:, :, 1], \
    gpu_array_[:, :, 2] = color_reduction_kernel(
        gpu_array_[:, :, 0], gpu_array_[:, :, 1], gpu_array_[:, :, 2], color_number)

    return frombuffer(gpu_array_.astype(
        cp.uint8).transpose(1, 0, 2).tobytes(), (w, h), "RGB").convert()


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef object color_reduction_cupy_inplace(
        object gpu_array_,
        cpu_array,
        int color_number
):
    cdef:
        Py_ssize_t w, h
    w, h = gpu_array_.shape[0], gpu_array_.shape[1]

    gpu_array_[:, :, 0], \
    gpu_array_[:, :, 1], \
    gpu_array_[:, :, 2] = color_reduction_kernel(
        gpu_array_[:, :, 0], gpu_array_[:, :, 1], gpu_array_[:, :, 2], color_number)

    cpu_array[:, :, 0] = gpu_array_[:, :, 0].get()
    cpu_array[:, :, 1] = gpu_array_[:, :, 1].get()
    cpu_array[:, :, 2] = gpu_array_[:, :, 2].get()


rgb2hsv_cuda = r'''   
    extern "C"
    
    __global__ void rgb2hsv(float * r, float * g, float * b, int width, int height, double val_)
    {
    
    int xx = blockIdx.x * blockDim.x + threadIdx.x;     
    int yy = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Index value of the current pixel
    const int index = yy * height + xx;
    const int t_max = height * width;
     
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
	    h = (float)fmod(60.0f * ((gg-bb) * df_) + 360.0, 360);
	}
    else if (mx == gg){
	    h = (float)fmod(60.0f * ((bb-rr) * df_) + 120.0, 360);
	}
    else if (mx == bb){
	    h = (float)fmod(60.0f * ((rr-gg) * df_) + 240.0, 360);
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
    h = (float)fmod(h + val_, (double)1.0);

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

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef object hsv_gpu(
        object surface_,
        float val_,
        object grid_  = None,
        object block_ = None
):
    """
    HSV (HUE ROTATION)
    
    Compatible with image format 32 - 24 bit
    Rotate the pixels color of an image/texture
    
    :param surface_          : pygame.surface
    :param grid_             : tuple; grid values (grid_y, grid_x) e.g (25, 25). The grid values and block values must 
        match the texture and array sizes. 
    :param block_            : tuple; block values (block_y, block_x) e.g (32, 32). Maximum threads is 1024.
        Max threads = block_x * block_y
    :param val_              : float; Float values representing the next hue value   
    :return                  : Return a pygame.Surface with a modified HUE  
    """

    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface, got %s " % type(surface_)

    try:
        cpu_array_ = pixels3d(surface_)

    except Exception as e:
        raise ValueError("\nCannot reference source pixels into a 3d array.\n %s " % e)

    cdef:
        Py_ssize_t w, h
    w, h = cpu_array_.shape[0], cpu_array_.shape[1]

    if w == 0 or h == 0:
        raise ValueError("Surface width and height cannot be null!")

    assert 0.0 <= val_ <= 1.0, "\nArgument val_ must be in range [0.0 ... 1.0] got %s " % val_
    assert PyObject_IsInstance(grid_, tuple), \
        "\nArgument grid_ must be a tuple (gridy, gridx)  got %s " % type(grid_)
    assert PyObject_IsInstance(block_, tuple), \
        "\nArgument block_ must be a tuple (blocky, blockx) got %s " % type(block_)

    return hsv_cupy(cp.asarray(cpu_array_).astype(
        dtype=cp.float32), grid_, block_, val_, w, h)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef object hsv_cupy(
        object cupy_array,
        object grid_,
        object block_,
        float val_,
        w, h
):

    module = cp.RawModule(code=rgb2hsv_cuda)
    rgb_to_hsv_ = module.get_function("rgb2hsv")

    cdef:
        r = cp.zeros((w, h), dtype=cp.float32)
        g = cp.zeros((w, h), dtype=cp.float32)
        b = cp.zeros((w, h), dtype=cp.float32)


    r = (cupy_array[:, :, 0] * ONE_255)
    g = (cupy_array[:, :, 1] * ONE_255)
    b = (cupy_array[:, :, 2] * ONE_255)

    rgb_to_hsv_(grid_, block_, (r, g, b, w, h, val_))

    cupy_array[:, :, 0] = cp.multiply(r, 255.0)
    cupy_array[:, :, 1] = cp.multiply(g, 255.0)
    cupy_array[:, :, 2] = cp.multiply(b, 255.0)

    cp.cuda.Stream.null.synchronize()

    return frombuffer(cupy_array.astype(cp.uint8).transpose(1, 0, 2).tobytes(), (w, h), "RGB").convert()


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

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef object mult_downscale_gpu(
        object gpu_array

):
    """
    BLOOM DOWNSCALE SUB-METHOD 
    
    compatible with image format 32 - 24 bit
    Downscale an array into 4 sub-arrays with shapes (div 2, div 4, div 8, div 16)   
    
    :param gpu_array: cupy array type (w, h, 3) uint8
    :return         : Returns 4 sub-arrays with shapes (div 2, div 4, div 8, div 16)
    """

    assert gpu_array.dtype == cupy.uint8, \
        "\nArgument gpu_array_ datatype must be uint8 got %s " % gpu_array.dtype

    downscale_x2 = cupyx.scipy.ndimage.zoom(
        gpu_array, (1.0 / 2.0, 1.0 / 2.0, 1), order=0, mode='constant', cval=0.0)
    downscale_x4 = cupyx.scipy.ndimage.zoom(
        downscale_x2, (1.0 / 2.0, 1.0 / 2.0, 1), order=0, mode='constant', cval=0.0)
    downscale_x8 = cupyx.scipy.ndimage.zoom(
        downscale_x4, (1.0 / 2.0, 1.0 / 2.0, 1), order=0, mode='constant', cval=0.0)
    downscale_x16 = cupyx.scipy.ndimage.zoom(
        downscale_x8, (1.0 / 2.0, 1.0 / 2.0, 1), order=0, mode='constant', cval=0.0)

    cp.cuda.Stream.null.synchronize()


    return downscale_x2, \
           downscale_x4,\
           downscale_x8, \
           downscale_x16

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
#         gpu_array_ = cp.asarray(pixels3d(surface_), dtype=cp.uint8)
#
#     except Exception as e:
#         raise ValueError(
#             "\nCannot reference source pixels into a 3d array.\n %s " % e)
#
#     cdef int w0, h0
#     w0, h0 = gpu_array_.shape[0], gpu_array_.shape[1]
#
#     downscale_ = cupyx.scipy.ndimage.zoom(
#         gpu_array_, (<float>w0/<float>w, <float>h0/<float>h, 1), order=0, mode='constant', cval=0.0)
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
#         gpu_array_ = cp.asarray(pixels3d(surface_), dtype=cp.uint8)
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
#         gpu_array, (<float>w/<float>w0, <float>h/<float>h0, 1), order=0, mode='constant', cval=0.0)
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

# BRIGHT PASS FILTER FOR BLOOM EFFECT
cdef void bpf_c(object gpu_array_, int w, int h, unsigned int threshold_=128):

    cdef:
        r = cp.empty((w, h), dtype=cp.uint8)
        g = cp.empty((w, h), dtype=cp.uint8)
        b = cp.empty((w, h), dtype=cp.uint8)

    r = gpu_array_[:, :, 0]
    g = gpu_array_[:, :, 1]
    b = gpu_array_[:, :, 2]

    gpu_array_[:, :, 0], gpu_array_[:, :, 1], gpu_array_[:, :, 2] = \
        bpf_kernel(r, g, b, <float>threshold_)

    cp.cuda.Stream.null.synchronize()



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
# BLUR GAUSSIAN 3x3 FOR BLOOM EFFECT
cdef gaussian_3x3_c(gpu_array_, int w, int h):

    cdef:
        r = cp.empty((w, h), dtype=cp.float32)
        g = cp.empty((w, h), dtype=cp.float32)
        b = cp.empty((w, h), dtype=cp.float32)

    r = gpu_array_[:, :, 0]
    g = gpu_array_[:, :, 1]
    b = gpu_array_[:, :, 2]

    # Gaussian kernel 3x3
    k = cp.array([[1, 2, 1],
                  [2, 4, 2],
                  [1, 2, 1]], dtype=cp.float32) * <float>1.0 / <float>16.0

    r = cupyx.scipy.ndimage.convolve(r, k, mode='constant', cval=0.0)
    g = cupyx.scipy.ndimage.convolve(g, k, mode='constant', cval=0.0)
    b = cupyx.scipy.ndimage.convolve(b, k, mode='constant', cval=0.0)

    gpu_array_[:, :, 0], \
    gpu_array_[:, :, 1], \
    gpu_array_[:, :, 2] = r.astype(cp.uint8), g.astype(cp.uint8), b.astype(cp.uint8)
    cp.cuda.Stream.null.synchronize()


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
#BLUR GAUSSIAN 5x5 FOR BLOOM
cdef gaussian_5x5_c(gpu_array_, int w, int h):

    cdef:
        r = cp.empty((w, h), dtype=cp.float32)
        g = cp.empty((w, h), dtype=cp.float32)
        b = cp.empty((w, h), dtype=cp.float32)

    r = gpu_array_[:, :, 0]
    g = gpu_array_[:, :, 1]
    b = gpu_array_[:, :, 2]

    # Gaussian kernel 5x5
    k = cp.array([[1,   4,   6,   4,  1],
    [4,  16,  24,  16,  4],
    [6,  24,  36,  24,  6],
    [4,  16,  24,  16,  4],
    [1,  4,    6,   4,  1]], dtype=cp.float32) * <float>1.0 / <float>256.0

    r = cupyx.scipy.ndimage.convolve(r, k, mode='constant', cval=0.0)
    g = cupyx.scipy.ndimage.convolve(g, k, mode='constant', cval=0.0)
    b = cupyx.scipy.ndimage.convolve(b, k, mode='constant', cval=0.0)

    gpu_array_[:, :, 0], \
    gpu_array_[:, :, 1], \
    gpu_array_[:, :, 2] = r.astype(cp.uint8), g.astype(cp.uint8), b.astype(cp.uint8)

    cp.cuda.Stream.null.synchronize()


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
# ARRAY UPSCALE FOR BLOOM EFFECT
cpdef object upscale_c(object gpu_array_, int new_width, int new_height, int order_=0):

    assert gpu_array_.dtype == cupy.uint8, \
        "\nArgument gpu_array_ datatype must be uint8 got %s " % gpu_array_.dtype

    cdef int w1, h1
    w1, h1 = gpu_array_.shape[0], gpu_array_.shape[1]

    gpu_array = cp.asarray(gpu_array_, dtype=cp.uint8)

    upscale_array = cupyx.scipy.ndimage.zoom(
        gpu_array, (new_width / w1, new_height/ h1, 1), order=order_, mode='constant', cval=0.0)

    cp.cuda.Stream.null.synchronize()

    return upscale_array


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef object bloom_gpu(
        object surface_,
        unsigned int threshold_=128,
        bint fast_=True,
        int flag_ = pygame.BLEND_RGB_ADD,
        unsigned short int factor_ = 2
        ):
    """
    CREATE A BLOOM EFFECT (USING A PYGAME SURFACE) 
    
    Compatible with image format 32 - 24 bit
    * Take a pygame.Surface for input
    
    :param surface_  : pygame.Surface; Surface to bloom  
    :param threshold_: integer; Threshold value used by the bright pass filter. 
        Value must be in range [0..255]. Default is 128 
    :param fast_     : boolean; True | False; When True -> improve the algorithm performances
        (using only the sub-surface x16). Default is True (fast)
    :param flag_     : integer; Pygame flags such as pygame.BLEND_RGB_ADD, pygame.BLEND_RGB_MULT etc to 
        blend the bloom effect to the original surface. default is BLEND_RGB_ADD 
    :param factor_   : integer; Texture reduction value, must be in range [0, 4] and correspond to the dividing texture
        factor (div 1, div 2, div 4, div 8)
    :return          : Return a pygame.Surface; Argument surface_ blended with the bloom effect
    """

    surface_copy = surface_.copy()
    surface_ = smoothscale(surface_,
                           (surface_.get_width() >> factor_, surface_.get_height() >> factor_))

    try:
        gpu_array_ = cp.asarray(pixels3d(surface_), dtype=cp.uint8)

    except Exception as e:
        raise ValueError("\nCannot reference source pixels into a 3d array.\n %s " % e)

    cdef:
        int w1, h1, w2, h2, w4, h4, w8, h8, w16, h16
        bint x2, x4, x8, x16 = False

    # Original size (width and height)
    w1, h1 = gpu_array_.shape[0], gpu_array_.shape[1]

    w2 = w1 >> 1
    h2 = h1 >> 1
    w4 = w2 >> 1
    h4 = h2 >> 1
    w8 = w4 >> 1
    h8 = h4 >> 1
    w16 = w8 >> 1
    h16 = h8 >> 1

    if w16 == 0 or h16 == 0:
        raise ValueError(
            "\nImage too small and cannot be processed.\n"
            "Try to increase the size of the image or decrease the factor_ value (default 2).\n"
            "Current value %s " % factor_)

    cdef:
        scale_x2 = cp.empty((w2, h2, 3), cp.uint8)
        scale_x4 = cp.empty((w4, h4, 3), cp.uint8)
        scale_x8 = cp.empty((w8, h8, 3), cp.uint8)
        scale_x16 = cp.empty((w16, h16, 3), cp.uint8)

    if w2 > 0 and h2 > 0:
        x2 = True
    else:
        x2 = False

    if w4 > 0 and h4 > 0:
        x4 = True
    else:
        x4 = False

    if w8 > 0 and h8 > 0:
        x8 = True
    else:
        x8 = False

    if w16 > 0 and h16 > 0:
        x16 = True
    else:
        x16 = False

    s2, s4, s8, s16 = None, None, None, None

    # SUBSURFACE DOWNSCALE CANNOT
    # BE PERFORMED AND WILL RAISE AN EXCEPTION
    if not x2:
        return

    if fast_:
        x2, x4, x8 = False, False, False


    scale_x2, scale_x4, scale_x8, scale_x16 = mult_downscale_gpu(gpu_array_)

    cp.cuda.Stream.null.synchronize()

    # FIRST SUBSURFACE DOWNSCALE x2
    # THIS IS THE MOST EXPENSIVE IN TERM OF PROCESSING TIME
    if x2:

        bpf_c(scale_x2, w2, h2, threshold_=threshold_)
        gaussian_3x3_c(scale_x2, w2, h2)
        s2 = make_surface(upscale_c(scale_x2, w1, h1, order_=0).get())
        # surface_.blit(s2, (0, 0), special_flags=flag_)


    # SECOND SUBSURFACE DOWNSCALE x4
    # THIS IS THE SECOND MOST EXPENSIVE IN TERM OF PROCESSING TIME
    if x4:
        bpf_c(scale_x4, w4, h4, threshold_=threshold_)
        gaussian_3x3_c(scale_x4, w4, h4)
        s4 = make_surface(upscale_c(scale_x4, w1, h1, order_=0).get())
        # surface_.blit(s4, (0, 0), special_flags=flag_)

    # THIRD SUBSURFACE DOWNSCALE x8
    if x8:

        bpf_c(scale_x8, w8, h8, threshold_=threshold_)
        gaussian_3x3_c(scale_x8, w8, h8)
        s8 = make_surface(upscale_c(scale_x8, w1, h1, order_=1).get())
        # surface_.blit(s8, (0, 0), special_flags=flag_)

    # FOURTH SUBSURFACE DOWNSCALE x16
    # LEAST SIGNIFICANT IN TERMS OF RENDERING AND PROCESSING TIME
    if x16:

        bpf_c(scale_x16, w16, h16, threshold_=threshold_)
        gaussian_3x3_c(scale_x16, w16, h16)
        s16 = make_surface(upscale_c(scale_x16, w1, h1, order_=1).get())
        # surface_.blit(s16, (0, 0), special_flags=flag_)


    if fast_:
        s16 = smoothscale(s16, (w1 << factor_, h1 << factor_))
        surface_copy.blit(s16, (0, 0), special_flags=BLEND_RGB_ADD)

    else:
        s2.blit(s4, (0, 0), special_flags=BLEND_RGB_ADD)
        s2.blit(s8, (0, 0), special_flags=BLEND_RGB_ADD)
        s2.blit(s16, (0, 0), special_flags=BLEND_RGB_ADD)
        s2 = smoothscale(s2, (w1 << factor_, h1 << factor_))
        surface_copy.blit(s2, (0, 0), special_flags=BLEND_RGB_ADD)

    cp.cuda.Stream.null.synchronize()

    # if mask_ is not None:
    #     surface_ = filtering24_c(surface_, mask_)
    #     return surface_

    return surface_copy




@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef object bloom_array(
        object gpu_array_,
        unsigned int threshold_=128,
        bint fast_=True,
        int flag_ = pygame.BLEND_RGB_ADD,
        mask_ = None
):
    assert gpu_array_.dtype == cupy.uint8, \
        "\nArgument gpu_array_ datatype must be uint8 got %s " % gpu_array_.dtype

    cdef:
        int w1, h1, w2, h2, w4, h4, w8, h8, w16, h16
        bint x2, x4, x8, x16 = False


    # Original size (width and height)
    w1, h1 = gpu_array_.shape[0], gpu_array_.shape[1]

    w2 = w1 >> 1
    h2 = h1 >> 1
    w4 = w2 >> 1
    h4 = h2 >> 1
    w8 = w4 >> 1
    h8 = h4 >> 1
    w16 = w8 >> 1
    h16 = h8 >> 1

    if w16 == 0 or h16 == 0:
        raise ValueError(
            "\nImage cannot be processed\n."
            " Try increase the size of the image.")

    cdef:
        scale_x2 = cp.empty((w2, h2, 3), cp.uint8)
        scale_x4 = cp.empty((w4, h4, 3), cp.uint8)
        scale_x8 = cp.empty((w8, h8, 3), cp.uint8)
        scale_x16 = cp.empty((w16, h16, 3), cp.uint8)

    if w2 > 0 and h2 > 0:
        x2 = True
    else:
        x2 = False

    if w4 > 0 and h4 > 0:
        x4 = True
    else:
        x4 = False

    if w8 > 0 and h8 > 0:
        x8 = True
    else:
        x8 = False

    if w16 > 0 and h16 > 0:
        x16 = True
    else:
        x16 = False

    # SUBSURFACE DOWNSCALE CANNOT
    # BE PERFORMED AND WILL RAISE AN EXCEPTION
    if not x2:
        return

    if fast_:
        x2, x4, x8 = False, False, False

    scale_x2, scale_x4, scale_x8, scale_x16 = mult_downscale_gpu(gpu_array_)

    cp.cuda.Stream.null.synchronize()

    s2, s4, s8, s16 = None, None, None, None
    # FIRST SUBSURFACE DOWNSCALE x2
    # THIS IS THE MOST EXPENSIVE IN TERM OF PROCESSING TIME
    if x2:

        bpf_c(scale_x2, w2, h2, threshold_=threshold_)
        gaussian_3x3_c(scale_x2, w2, h2)
        s2 = make_surface(upscale_c(scale_x2, w1, h1, order_=0).get())


    # SECOND SUBSURFACE DOWNSCALE x4
    # THIS IS THE SECOND MOST EXPENSIVE IN TERM OF PROCESSING TIME
    if x4:

        bpf_c(scale_x4, w4, h4, threshold_=threshold_)
        gaussian_3x3_c(scale_x4, w4, h4)
        s4 = make_surface(upscale_c(scale_x4, w1, h1, order_=0).get())


    # THIRD SUBSURFACE DOWNSCALE x8
    if x8:

        bpf_c(scale_x8, w8, h8, threshold_=threshold_)
        gaussian_3x3_c(scale_x8, w8, h8)
        s8 = make_surface(upscale_c(scale_x8, w1, h1, order_=1).get())


    # FOURTH SUBSURFACE DOWNSCALE x16
    # LEAST SIGNIFICANT IN TERMS OF RENDERING AND PROCESSING TIME
    if x16:

        bpf_c(scale_x16, w16, h16, threshold_=threshold_)
        gaussian_3x3_c(scale_x16, w16, h16)
        s16 = make_surface(upscale_c(scale_x16, w1, h1, order_=1).get())


    cp.cuda.Stream.null.synchronize()

    # if mask_ is not None:
    #     surface_ = filtering24_c(surface_, mask_)
    #     return surface_
    return s2, s4, s8, s16

# --------------------------------------- CARTOON EFFECT


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef object cartoon_gpu(
        object surface_,
        int sobel_threshold_ = 128,
        int median_kernel_   = 2,
        unsigned char color_ = 8,
        bint contour_        = False,
        unsigned char flag_  = BLEND_RGB_ADD
):
    """
    CREATE A CARTOON EFFECT FROM A GIVEN SURFACE 

    Compatible with image format 32 - 24 bit
     
    :param surface_        : pygame.Surface compatible 24 - 32 bit 
    :param sobel_threshold_: integer sobel threshold
    :param median_kernel_  : integer median kernel  
    :param color_          : integer; color reduction value (max color)
    :param contour_        : boolean; Draw the contour
    :param flag_           : integer; Blend flag e.g (BLEND_RGB_ADD, BLEND_RGB_SUB, 
                             BLEND_RGB_MULT, BLEND_RGB_MAX, BLEND_RGB_MIN  
    :return                : Return a pygame Surface with the cartoon effect 
    """

    return cartoon_cupy(surface_, sobel_threshold_, median_kernel_, color_, contour_, flag_)

# Gaussian kernel 5x5
k = cp.array([[2, 4, 5, 4, 2, ],
              [4, 9, 12, 9, 4],
              [5, 12, 15, 12, 5],
              [4, 9, 12, 9, 4],
              [2, 4, 5, 4, 2]], dtype=cp.float32) * <float>1.0 / <float>256.0

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef object canny_cupy_c(gpu_array_, int w, int h):

    r = cupyx.scipy.ndimage.convolve(gpu_array_[:, :, 0], k, mode='constant', cval=0.0)

    sobel2d_r = cupyx.scipy.ndimage.generic_filter(r, sobel_kernel, 3).astype(dtype=cp.uint8)

    gpu_array_[:, :, 0], \
    gpu_array_[:, :, 1], \
    gpu_array_[:, :, 2] = sobel2d_r, sobel2d_r, sobel2d_r

    cp.cuda.Stream.null.synchronize()

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef object sobel_cupy_c(gpu_array_, int w, int h):

    sobel2d_r = cupyx.scipy.ndimage.generic_filter(
        gpu_array_[:, :, 0], sobel_kernel, 3).astype(dtype=cp.uint8)

    cp.cuda.Stream.null.synchronize()

    gpu_array_[:, :, 0], \
    gpu_array_[:, :, 1], \
    gpu_array_[:, :, 2] = sobel2d_r, sobel2d_r, sobel2d_r


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef object median_cupy_c(gpu_array_, int w, int h, unsigned int size_=5):

    gpu_array_[:, :, 0] = cupyx.scipy.ndimage.median_filter(gpu_array_[:, :, 0], size_)
    gpu_array_[:, :, 1] = cupyx.scipy.ndimage.median_filter(gpu_array_[:, :, 1], size_)
    gpu_array_[:, :, 2] = cupyx.scipy.ndimage.median_filter(gpu_array_[:, :, 2], size_)

    cp.cuda.Stream.null.synchronize()


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef object color_reduction_cupy_c(
        object gpu_array_,
        int color_number,
        int w, int h
):

    gpu_array_[:, :, 0], \
    gpu_array_[:, :, 1], \
    gpu_array_[:, :, 2] = color_reduction_kernel(
        gpu_array_[:, :, 0], gpu_array_[:, :, 1], gpu_array_[:, :, 2], color_number, block_size=1024)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef object cartoon_cupy(
        object surface_,
        int sobel_threshold_,
        int median_kernel_,
        int color_,
        bint contour_,
        int flag_):
    """

    :param surface_        : pygame.Surface compatible 24 - 32 bit 
    :param sobel_threshold_: integer sobel threshold
    :param median_kernel_  : integer median kernel (size of the median neighbourhood) 
    :param color_          : integer; color reduction value (max color)
    :param contour_        : boolean; Draw the contour_ 
    :param flag_           : integer; Blend flag e.g (BLEND_RGB_ADD, BLEND_RGB_SUB, 
        BLEND_RGB_MULT, BLEND_RGB_MAX, BLEND_RGB_MIN  
    :return                : Return a pygame Surface with the cartoon effect 
    """

    try:
        gpu_array_ = cp.asarray(pixels3d(surface_))
    except Exception as e:
        raise ValueError("\nCannot reference source pixels into a 3d array.\n %s " % e)

    gpu_array_copy = gpu_array_.copy()

    cdef:
        Py_ssize_t w, h
    w, h = gpu_array_.shape[0], gpu_array_.shape[1]

    # FIRST BRANCH
    # APPLY A CANNY EDGE DETECTION INPLACE
    canny_cupy_c(gpu_array_, w, h)

    # APPLY A GRAYSCALE INPLACE
    # grayscale_cupy(gpu_array_)

    # APPLY AN OPTIONAL SOBEL EFFECT INPLACE
    if contour_:
        sobel_cupy_c(gpu_array_, w, h)


    # SECOND BRANCH

    # APPLY MEDIAN FILTER INPLACE
    median_cupy_c(gpu_array_copy, w, h, median_kernel_)

    # APPLY COLOR REDUCTION INPLACE
    color_reduction_cupy_c(gpu_array_copy, color_, w, h)

    surface_ = frombuffer(gpu_array_copy.astype(dtype=cp.uint8).transpose(1, 0, 2).tobytes(), (w, h), "RGB")

    # BLEND BOTH BRANCHES
    surf = frombuffer(gpu_array_.astype(dtype=cp.uint8).transpose(1, 0, 2).tobytes(), (w, h), "RGB")

    surface_.blit(surf, (0, 0), special_flags=flag_)

    return surface_.convert()



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



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)

cpdef object blending_gpu(object source_, object destination_, float percentage_):
    """
    BLEND A SOURCE TEXTURE TOWARD A DESTINATION TEXTURE (TRANSITION EFFECT)

    * Video system must be initialised 
    * source_ & destination_ Textures must be same sizes
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
        source_array = (source_array.reshape(w, h, 4)/255.0).astype(dtype=float32)

    except Exception as e:
        raise ValueError("\nCannot reference source pixels into a 3d array.\n %s " % e)

    try:

        destination_array = numpy.frombuffer(
            tostring(destination_, "RGBA_PREMULT"), dtype=numpy.uint8)
        destination_array = cp.asarray(destination_array, dtype=cp.uint8)
        destination_array = (destination_array.reshape(w, h, 4) / 255.0).astype(dtype=float32)
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


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef object sharpen_gpu(object surface_):
    """
    SHARPEN FILTER (GENERIC_FILTER)
    
    :param surface_  : pygame.Surface
    :return          : pygame.Surface format 24 bit 
    """

    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)

    try:
        cpu_array_ = pixels3d(surface_)

    except Exception as e:
        raise ValueError("\nCannot reference source pixels into a 3d array.\n %s " % e)

    cdef:
        Py_ssize_t w, h
    w, h = cpu_array_.shape[0], cpu_array_.shape[1]

    if w == 0 or h == 0:
        raise ValueError("Surface width and height cannot be null!")

    gpu_array_ = cp.asarray(cpu_array_)

    cdef:
        r = cp.empty((w, h), dtype=cp.float32)
        g = cp.empty((w, h), dtype=cp.float32)
        b = cp.empty((w, h), dtype=cp.float32)

    r = gpu_array_[:, :, 0]
    g = gpu_array_[:, :, 1]
    b = gpu_array_[:, :, 2]


    rr = cupyx.scipy.ndimage.generic_filter(r, sharpen_kernel, 3)
    gg = cupyx.scipy.ndimage.generic_filter(g, sharpen_kernel, 3)
    bb = cupyx.scipy.ndimage.generic_filter(b, sharpen_kernel, 3)


    gpu_array_[:, :, 0], \
    gpu_array_[:, :, 1], \
    gpu_array_[:, :, 2] = rr.astype(dtype=cp.uint8), gg.astype(dtype=cp.uint8), bb.astype(dtype=cp.uint8)

    cp.cuda.Stream.null.synchronize()

    return frombuffer(gpu_array_.transpose(1, 0, 2).tobytes(), (w, h), "RGB")



ripple_kernel = cp.RawKernel(
    r'''
    extern "C" __global__
    void ripple_kernel(float * current, float * previous, 
    unsigned char * background_array, unsigned char * texture_array,    
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
            float data = (previous[max((index + 1) % t_max, 0)] +                 // left  
                          previous[max((index - 1) % t_max, 0)] +                 // right 
                          previous[max((index - h) % t_max, 0)] +                 // top 
                          previous[max((index + h) % t_max, 0)]) * (float)0.5;    // bottom
            */
            
            
            
            if ((index - h) < 0) {
                top = 0.0f;
            }
            else {
                top = previous[index - h];
            }
            
            if ((index + h) > t_max) {
                bottom = 0.0f; 
            } 
              else {
                bottom = previous[index + h];
            }
            
            
            if ((index - 1) < 0) {
                right = 0.0f;
            }
            else {
                right = previous[index - 1];
            }
            
            if ((index + 1) > t_max) {
                left = 0.0f;
            } 
            else {
                left = previous[index + 1];
            }
            
            
            data = (left + right + top + bottom) * 0.5f; 
            
            
              
            data = data - current[index];
            data = data - (data * 0.01125f);   // Attenuation
             
            __syncthreads();
            
            current[index] = data;
            
                
            data = 1.0f - data * 1.0f/1024.0f;
            const int w2 = w >> 1;
            const int h2 = h >> 1;
            const int a = max((int)(((i - w2) * data) + w2) % h, 0);              // texture index (x)
            const int b = max((int)(((j - h2) * data) + h2) % w, 0);              // texture index (y)
            // int ind = a * h * 3 + b * 3;   // inverse texture
            const int ind = b * h * 3 + a * 3;
            background_array[index1       ] = texture_array[ind       ];    // red
            background_array[(index1 + 1) ] = texture_array[(ind + 1) ];    // green 
            background_array[(index1 + 2) ] = texture_array[(ind + 2) ];    // blue
            
            __syncthreads();
        }
    }
    ''',
    'ripple_kernel'
)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef ripple_effect_gpu(
       object grid,
       object block,
       int w, int h,
       previous,            # type cupy.float32 (w, h)
       current,             # type cupy.float32 (w, h)
       texture_array,       # type cupy.ndarray (w, h, 3) uint8
       background_array     # type cupy.ndarray (w, h, 3) uint8
       ):
    """
    WATER DROP / RIPPLE EFFECT (USING GPU)
    
    Compatible with 24-bit or 32-bit textures (converted to 24 bit with pygame.convert()) 
    This method call cuda kernel ripple_kernel to do the math with the GPU
    This version is compatible with textures & array with identical width and height  
    
    :param grid             : tuple; grid values (grid_y, grid_x) e.g (25, 25). The grid values and block values must 
        match the texture and array sizes. 
    :param block            : tuple; block values (block_y, block_x) e.g (32, 32). Maximum threads is 1024.
        Max threads = block_x * block_y
    :param w                : integer; Width of the textures and arrays
    :param h                : integer; height of the textures and arrays
    :param previous         : cupy.array; Array shape (w, h) float32 containing the water 
        ripple effect (previous status)
    :param current          : cupy.array; Array shape (w, h) float32 containing the water ripple
        effect (current status)
    :param texture_array    : cupy.array; Array shape (w, h, 3) of uint8 containing the RGB pixels (source)  
    :param background_array : cupy.array; Array shape (w, h, 3) of uint8 (destination texture)
    :return                 : tuple; Return two cupy arrays (previous & current) 
    """

    ripple_kernel(
        grid,
        block,
        (current, previous, background_array, texture_array, w, h))
    cp.cuda.Stream.null.synchronize()
    return previous, current



# Sharpen kernel (different method)
sharpen1_kernel = cp.RawKernel(
    r'''
    extern "C" __global__
    void sharpen1_kernel(float * current, float * previous, 
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
                        -previous[index1 - col          ] +                               
                        -previous[index1 - 3            ] +          
                         previous[index1                ] * 5.0f  +          
                        -previous[index1 + 3            ] +                           
                        -previous[index1 + col          ];          
                         
            green =  
                        -previous[index1 - col  + 1     ] +                                 
                        -previous[index1 - 2            ] +         
                         previous[index1 + 1            ] * 5.0f  +          
                        -previous[index1 + 4            ] +                           
                        -previous[index1 + col + 1      ];        
                                             
            blue =         
                        -previous[index1 - col + 2     ] +                                  
                        -previous[index1 - 1           ] +          
                         previous[index1 + 2           ] * 5.0f  +          
                        -previous[index1 + 5           ] +                         
                        -previous[index1 + col + 2     ];                     
            }
            
            __syncthreads();     
                       
            if (red > 255) { red = 255; } 
            if (red < 0) { red = 0; }
            
            if (green > 255) { green = 255; } 
            if (green < 0) { green = 0; }
            
            if (blue > 255) { blue = 255; } 
            if (blue < 0) { blue = 0; }
                  
            current[ index1     ] = red;
            current[ index1 + 1 ] = green;
            current[ index1 + 2 ] = blue;
            
            
            __syncthreads();
            
        }
    }
    ''',
    'sharpen1_kernel'
)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
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

    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)

    try:
        cpu_array_ = pixels3d(surface_)

    except Exception as e:
        raise ValueError("\nCannot reference source pixels into a 3d array.\n %s " % e)

    cdef:
        Py_ssize_t w, h
    w, h = cpu_array_.shape[0], cpu_array_.shape[1]

    if w == 0 or h == 0:
        raise ValueError("Surface width and height cannot be null!")

    gpu_array_ = cp.asarray(cpu_array_)

    cdef destination = cupy.empty((w, h, 3), cp.float32)

    sharpen1_kernel(
        grid_,
        block_,
        (destination, gpu_array_.astype(dtype=cp.float32), w, h))

    cp.cuda.Stream.null.synchronize()

    return frombuffer(destination.astype(
        dtype=cupy.uint8).transpose(1, 0, 2).tobytes(), (w, h), "RGB").convert()



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef object mirroring_gpu(
        object surface_,
        object grid_,
        object block_,
        bint format_ = 0
):
    """
    MIRROR EFFECT 
    
    Create a mirror effect (lateral or vertical), adjust the variable format_ to change the 
    mirror orientation
    This algorithm is compatible with image format 32 - 24 bit
    The output image format is 24-bit  
    
    :param surface_     : pygame.Surface
    :param grid_        : tuple; grid values (grid_y, grid_x) e.g (25, 25). The grid values and block values must 
        match the texture and array sizes. 
    :param block_       : tuple; block values (block_y, block_x) e.g (32, 32). Maximum threads is 1024.
        Max threads = block_x * block_y
    :param format_      : bool; Flip the mirror orientation (vertical or lateral)
    :return             : Return a 24-bit pygame.Surface with a mirror effect 
    """

    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)

    try:
        cpu_array_ = pixels3d(surface_)

    except Exception as e:
        raise ValueError("\nCannot reference source pixels into a 3d array.\n %s " % e)

    cdef:
        Py_ssize_t w, h
    w, h = cpu_array_.shape[0], cpu_array_.shape[1]

    if w == 0 or h == 0:
        raise ValueError("Surface width and height cannot be null!")

    if len(grid_) != 2 or len(block_) != 2:
        raise ValueError("\nArgument grid_, block_ must be tuples (y, x)")

    return mirroring_cupy(cp.asarray(cpu_array_), grid_, block_, format_)


mirror_kernel = cp.RawKernel(
    r'''
    extern "C" __global__
    void mirror_kernel(float * current, float * previous, 
    const int w, const int h, bool format)
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

        // Maximum index value for array shape (w, h, 3)
        const int t_max_ = w * col;

        __syncthreads();

        float red   = 0;
        float green = 0;
        float blue  = 0;
        int x2, x3;

        // Limit the loop to the valid indexes 
        if (index > 0 && index < t_max){

            red   = previous[index1    ];       
            green = previous[index1 + 1];
            blue  = previous[index1 + 2];     

            __syncthreads(); 

            if (format == 1){

            x2 = i >> 1;

            current[j * col + x2 * 3    ] = red;
            current[j * col + x2 * 3 + 1] = green;
            current[j * col + x2 * 3 + 2] = blue;
            __syncthreads(); 

            x3 = h - x2 - 1;

            current[j * col + x3 * 3    ] = red;
            current[j * col + x3 * 3 + 1] = green;
            current[j * col + x3 * 3 + 2] = blue;
            }
            else{

            x2 = j >> 1;

            current[x2 * h * 3 + i * 3    ] = red;
            current[x2 * h * 3 + i * 3 + 1] = green;
            current[x2 * h * 3 + i * 3 + 2] = blue;

            __syncthreads();   

            x3 = w - x2 - 1;

            current[x3 * h * 3 + i * 3    ] = red;
            current[x3 * h * 3 + i * 3 + 1] = green;
            current[x3 * h * 3 + i * 3 + 2] = blue;
            }

            __syncthreads();     

        }
    }
    ''',
    'mirror_kernel'
)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline mirroring_cupy(object gpu_array_, object grid_, object block_, bint format_=0):

    cdef:
        Py_ssize_t w, h


    w, h = gpu_array_.shape[:2]

    destination = cupy.empty((w, h, 3), cupy.float32)

    mirror_kernel(
        grid_,
        block_,
        (destination, gpu_array_.astype(dtype=cp.float32), w, h, format_))

    cp.cuda.Stream.null.synchronize()

    return frombuffer(destination.astype(
        dtype=cupy.uint8).transpose(1, 0, 2).tobytes(), (w, h), "RGB").convert()






@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef object saturation_gpu(
        object surface_,
        object grid_,
        object block_,
        float val_ = 1.0
):

    """
    SATURATION 
    
    Compatible with image format 32 - 24 bit 
    Change the saturation level of an image, the saturation value must be in 
    range [-1.0 ... 1.0]. 
    Output image is 24 bit format 
    
    :param surface_     : pygame.Surface 
    :param grid_        : tuple; grid values (grid_y, grid_x) e.g (25, 25). The grid values and block values must 
        match the texture and array sizes. 
    :param block_       : tuple; block values (block_y, block_x) e.g (32, 32). Maximum threads is 1024.
        Max threads = block_x * block_y
    :param val_      : float; saturation level in range [-1.0 ... 1.0] 
    :return          : a 24-bit pygame.Surface with a defined saturation level
    """
    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)

    try:
        cpu_array_ = pixels3d(surface_)

    except Exception as e:
        raise ValueError("\nCannot reference source pixels into a 3d array.\n %s " % e)

    cdef:
        Py_ssize_t w, h
    w, h = cpu_array_.shape[0], cpu_array_.shape[1]

    if w == 0 or h == 0:
        raise ValueError("Surface width and height cannot be null!")

    if not -1.0 < val_ < 1.0:
        raise ValueError("\nArgument val_ must be in range [-1.0 ... 1.0]")

    if len(grid_) !=2 or len(block_) != 2:
        raise ValueError("\nArgument grid_, block_ must be tuples (y, x)")

    return saturation_cupy(cp.asarray(cpu_array_), grid_, block_, val_)


saturation_kernel = cp.RawKernel(
    r'''
    extern "C" __global__
    void saturation_kernel(float * source, float * destination, int width, int height, double val_)
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
            h = (float)fmod(60.0f * ((green-blue) * df_) + 360.0, 360);
        }
        else if (mx == green){
            h = (float)fmod(60.0f * ((blue-red) * df_) + 120.0, 360);
        }
        else if (mx == blue){
            h = (float)fmod(60.0f * ((red-green) * df_) + 240.0, 360);
        }
    
        if (mx == 0.0){
            s = 0.0;
        }
        else{
            s = df/mx;
        }
    
        v = mx;   
        h = h * 1.0f/360.0f;
    
    
        s = max(s + (float)val_, 0.0f);
        s = min(s, 1.0f);    
          
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
    
    
        destination[index1    ] = min(destination[index1    ] * 255.0f, 255.0f);
        destination[index1 + 1] = min(destination[index1 + 1] * 255.0f, 255.0f);
        destination[index1 + 2] = min(destination[index1 + 2] * 255.0f, 255.0f);
  
        } // if (index >0
        
        
        
    } // main
    ''',
    'saturation_kernel'
)



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef object saturation_cupy(
        object cupy_array,
        object grid_,
        object block_,
        float val_ = 1.0
):


    cdef:
        Py_ssize_t w, h

    w, h = cupy_array.shape[0], cupy_array.shape[1]

    destination = cupy.empty((w, h, 3), dtype=cupy.float32)

    saturation_kernel(
        grid_,
        block_,
        (cupy_array.astype(cupy.float32), destination, w, h, val_))

    cp.cuda.Stream.null.synchronize()

    return frombuffer(destination.astype(
        cp.uint8).transpose(1, 0, 2).tobytes(), (w, h), "RGB").convert()





@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef object bilateral_gpu(surface_, unsigned int kernel_size_):
    """
    BILATERAL FILTER 

    A bilateral filter is a non-linear, edge-preserving, and noise-reducing
    smoothing filter for images. It replaces the intensity of each pixel with a
    weighted average of intensity values from nearby pixels. This weight can be
    based on a Gaussian distribution.

    sigma_r & sigma_s are hard encoded in the GPU kernel
    Compatible with 32 - 24 bit image 

    :param surface_     : pygame.Surface 
    :param kernel_size_ : int; Kernel size (or neighbours pixels to be included in the calculation) 
    :return             : Return a 24 bit pygame.Surface with bilateral effect
    """
    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)

    try:
        cpu_array_ = pixels3d(surface_)

    except Exception as e:
        raise ValueError("\nCannot reference source pixels into a 3d array.\n %s " % e)

    cdef:
        Py_ssize_t w, h
    w, h = cpu_array_.shape[0], cpu_array_.shape[1]

    if w == 0 or h == 0:
        raise ValueError("Surface width and height cannot be null!")

    if kernel_size_ < 0:
        raise ValueError("\nArgument kernel_size_ cannot be < 0")

    return bilateral_cupy(cp.asarray(cpu_array_), kernel_size_)


bilateral_kernel = cp.RawKernel(
    '''
    extern "C" __global__
    void bilateral_kernel(double* source, int filter_size, double* destination)
    {

    // float sigma_i (also call sigma_r) range kernel, 
    // minimum amplitude of an edge.
    const double sigma_i2 = 80 * 80 * 2;   
    // float sigma_s : Spatial extent of the kernel, size of the 
    // considered neighborhood
    const double sigma_s2 = 16 * 16 * 2;  

    double ir = 0.0; 
    double wpr = 0.0;  
    double r=0.0, dist=0.0, gs=0.0;  
    double vr=0.0, wr=0.0;  
    const double p = 3.14159265;
    const int k2 = (int)sqrt((float)filter_size);

    int a = 0;

    __syncthreads();

    for (int ky = 0; ky < k2; ++ky)
    {
        for (int kx = 0; kx < k2; ++kx)
        {    

            dist = (double)sqrt((double)kx * (double)kx + (double)ky * (double)ky);
            gs = ((double)1.0 / (p * sigma_s2)) * (double)exp(-(dist * dist ) / sigma_s2);             
            r = source[a];
            vr = r - source[filter_size >> 1];
            wr = ((double)1.0 / (p * sigma_i2)) * (double)exp(-(vr * vr ) / sigma_i2);
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

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef bilateral_cupy(gpu_array_, unsigned int kernel_size_):
    """
    :param gpu_array_   : cupy.ndarray containing the RGB pixels 
    :param kernel_size_ : int; Kernel size (or neighbours pixels to be included in the calculation) 
    :return             : Return a pygame.Surface with bilateral effect
    """

    cdef Py_ssize_t w, h
    w, h = gpu_array_.shape[:2]

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

    return frombuffer(gpu_array_.astype(
        cp.uint8).transpose(1, 0, 2).tobytes(), (w, h), "RGB").convert()





@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef object emboss5x5_gpu(surface_):
    """
    EMBOSS 
    
    Emboss a 32 - 24 bit image using the kernel  {-2.0, -1.0, 0.0, -1.0, 1.0, 1.0, 0.0, 1.0, 2.0};
    Each channels (RGB) will be convoluted.
    This algorithm works for image format 32 - 24 bits. 
    The output image will be format 24 bit 
    
    
    :param surface_  : pygame.Surface      
    :return          : Return a 24-bit pygame.Surface with the emboss effect
    """
    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)

    try:
        cpu_array_ = pixels3d(surface_)

    except Exception as e:
        raise ValueError("\nCannot reference source pixels into a 3d array.\n %s " % e)

    cdef:
        Py_ssize_t w, h
    w, h = cpu_array_.shape[0], cpu_array_.shape[1]

    if w == 0 or h == 0:
        raise ValueError("Surface width and height cannot be null!")

    return emboss5x5_cupy(cp.asarray(cpu_array_))

emboss_kernel = cp.RawKernel(
    '''   
    extern "C" __global__
    
    __constant__ double k[9] = {-2.0, -1.0, 0.0, -1.0, 1.0, 1.0, 0.0, 1.0, 2.0};

    
    void sobel_kernel(double* buffer, int filter_size,
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


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef object emboss5x5_cupy(gpu_array_):
    """
    
    :param gpu_array_: cupy.ndarray; shape (w, h, 3) of uint8 containing the RGB values      
    :return          : Return a 24-bit pygame.Surface with the emboss effect
    """
    # texture sizes
    cdef Py_ssize_t w, h
    w = <object>gpu_array_.shape[0]
    h = <object>gpu_array_.shape[1]

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

    return frombuffer(gpu_array_.astype(
        cp.uint8).transpose(1, 0, 2).tobytes(), (w, h), "RGB").convert()


# ---------------------------------------- LIGHT ---------------------------------------------------


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef area24_gpu(int x, int y, object background_rgb, object mask_alpha, float intensity=1.0,
              color=cupy.array([128.0, 128.0, 128.0], dtype=cupy.float32, copy=False)):


    assert intensity >= 0.0, '\nIntensity value cannot be > 0.0'


    cdef int w, h, lx, ly, ax, ay

    try:
        w, h = background_rgb.shape[:2]
    except (ValueError, pygame.error) as e:
        raise ValueError('\nArray shape not understood.')

    try:
        ax, ay = (<object>mask_alpha).shape[:2]
    except (ValueError, pygame.error) as e:
        raise ValueError('\nArray shape not understood.')

    # Return an empty surface if the x or y are not within the normal range.
    if (x < 0) or (x > w - 1) or (y < 0) or (y > h - 1):
        return Surface((ax, ay), SRCALPHA), ax, ay

    # return an empty Surface when intensity = 0.0
    if intensity == 0.0:
        return Surface((ax, ay), SRCALPHA), ax, ay

    lx = ax >> 1
    ly = ay >> 1

    cdef:

        int i=0, j=0
        int w_low = lx
        int w_high = lx
        int h_low = ly
        int h_high = ly

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

    # rgb   = background_rgb[x - w_low:x + w_high, y - h_low:y + h_high, :]
    # rgb = rgb.transpose(1, 0, 2)
    # alpha = mask_alpha[lx - w_low:lx + w_high, ly - h_low:ly + h_high]
    # alpha = alpha.transpose(1, 0)

    # 175 FPS
    rgb = background_rgb[ y - h_low:y + h_high, x - w_low:x + w_high, :]
    alpha = mask_alpha[ly - h_low:ly + h_high, lx - w_low:lx + w_high]

    ax, ay = rgb.shape[:2]
    new_array = cupy.empty((ax, ay, 3), cupy.uint8)

    f = cupy.multiply(alpha, ONE_255 * intensity, dtype=cupy.float32)

    new_array[:, :, 0] = cupy.minimum(rgb[:, :, 0] * f, 255).astype(dtype=cupy.uint8)
    new_array[:, :, 1] = cupy.minimum(rgb[:, :, 1] * f, 255).astype(dtype=cupy.uint8)
    new_array[:, :, 2] = cupy.minimum(rgb[:, :, 2] * f, 255).astype(dtype=cupy.uint8)


    surface = pygame.image.frombuffer(new_array.tobytes(), (ay, ax), "RGB").convert()

    return surface, ay, ax





@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef object brightness_gpu(
        object surface_,
        float val_,
        object grid_  = None,
        object block_ = None
):
    """
    BRIGHTNESS 

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

    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)

    try:
        cpu_array_ = pixels3d(surface_)

    except Exception as e:
        raise ValueError("\nCannot reference source pixels into a 3d array.\n %s " % e)

    cdef:
        Py_ssize_t w, h
    w, h = cpu_array_.shape[0], cpu_array_.shape[1]

    if w == 0 or h == 0:
        raise ValueError("Surface width and height cannot be null!")

    assert -1.0 <= val_ <= 1.0, "\nArgument val_ must be in range [-1.0 ... 1.0] got %s " % val_
    assert PyObject_IsInstance(grid_, tuple), \
        "\nArgument grid_ must be a tuple (gridy, gridx)  got %s " % type(grid_)
    assert PyObject_IsInstance(block_, tuple), \
        "\nArgument block_ must be a tuple (blocky, blockx) got %s " % type(block_)


    return brightness_cupy(cp.asarray(cpu_array_).astype(
        dtype=cp.float32), grid_, block_, val_, w, h)


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
    
    __device__ struct rgb struct_hsl_to_rgb(float h, float s, float l);
    __device__ struct hsl struct_rgb_to_hsl(float r, float g, float b);
    __device__ float hue_to_rgb(float m1, float m2, float h);
    __device__ float fmin_rgb_value(float red, float green, float blue);
    __device__ float fmax_rgb_value(float red, float green, float blue);
  
  
    
    
    
    __device__ float fmax_rgb_value(float red, float green, float blue)
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
    

    __device__ float fmin_rgb_value(float red, float green, float blue)
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
    
    
    __device__ float hue_to_rgb(float m1, float m2, float h)
        {
            if ((fabs(h) > 1.0f) && (h > 0.0f)) {
              h = (float)fmod(h, 1.0f);
            }
            else if (h < 0.0f){
            h = 1.0f - (float)fabs(h);
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
    
    __device__ struct hsl struct_rgb_to_hsl(float r, float g, float b)
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
    	        if ((fabs(t) > 6.0f) && (t > 0.0f)) {
                  t = (float)fmod(t, 6.0f);
                }
                else if (t < 0.0f){
                t = 6.0f - (float)fabs(t);
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

    
    
    __device__ struct rgb struct_hsl_to_rgb(float h, float s, float l)
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
    
    extern "C"  __global__ void brightness(float * r, float * g, float * b, int width, int height, double val_)
    { 
        int xx = blockIdx.x * blockDim.x + threadIdx.x;     
        int yy = blockIdx.y * blockDim.y + threadIdx.y;
    
        // Index value of the current pixel
        const int index = yy * height + xx;
        const int t_max = height * width;
       
        struct hsl hsl_; 
        struct rgb rgb_;
        
        if (index > 0 && index < t_max) { 
            
            float rr = r[index] ;
            float gg = g[index] ;
            float bb = b[index] ;
            
            hsl_ = struct_rgb_to_hsl(rr, gg, bb);
            hsl_.l += val_;
            hsl_.l = max(hsl_.l, -1.0f);
            hsl_.l = min(hsl_.l, 1.0f);
            rgb_ = struct_hsl_to_rgb(hsl_.h, hsl_.s, hsl_.l); 
            
            r[index] = rgb_.r ;
            g[index] = rgb_.g ;
            b[index] = rgb_.b ;
            
        } 
        
        
    }
    
    
    
'''




@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef object brightness_cupy(
        object cupy_array,
        object grid_,
        object block_,
        float val_,
        w, h
):

    """
    BRIGHTNESS 
    
    
    :param cupy_array: cupy.ndarray; shape (w, h, 3) containing RGB pixel values 
    :param grid_      : tuple; grid values (grid_y, grid_x) e.g (25, 25). The grid values and block values must 
        match the texture and array sizes. 
    :param block_    : tuple; block values (block_y, block_x) e.g (32, 32). Maximum threads is 1024.
        Max threads = block_x * block_y
    :param val_      : float; Float values representing the next hue value   
    :param w         : integer;
    :param h         : integer;
    :return          : pygame.Surface with adjusted brightness
    """
    module = cp.RawModule(code=brightness_cuda)
    bright = module.get_function("brightness")

    cdef:
        r = cp.zeros((w, h), dtype=cp.float32)
        g = cp.zeros((w, h), dtype=cp.float32)
        b = cp.zeros((w, h), dtype=cp.float32)


    r = (cupy_array[:, :, 0] * ONE_255)
    g = (cupy_array[:, :, 1] * ONE_255)
    b = (cupy_array[:, :, 2] * ONE_255)

    bright(grid_, block_, (r, g, b, w, h, val_))

    cupy_array[:, :, 0] = cp.multiply(r, 255.0)
    cupy_array[:, :, 1] = cp.multiply(g, 255.0)
    cupy_array[:, :, 2] = cp.multiply(b, 255.0)

    cp.cuda.Stream.null.synchronize()

    return frombuffer(cupy_array.astype(cp.uint8).transpose(1, 0, 2).tobytes(), (w, h), "RGB").convert()



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef object hsl_gpu(
        object surface_,
        float val_,
        object grid_  = None,
        object block_ = None
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

    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface, got %s " % type(surface_)

    try:
        cpu_array_ = pixels3d(surface_)

    except Exception as e:
        raise ValueError("\nCannot reference source pixels into a 3d array.\n %s " % e)

    cdef:
        Py_ssize_t w, h
    w, h = cpu_array_.shape[0], cpu_array_.shape[1]

    if w == 0 or h == 0:
        raise ValueError("Surface width and height cannot be null!")

    assert 0.0 <= val_ <= 1.0, "\nArgument val_ must be in range [0.0 ... 1.0] got %s " % val_
    assert PyObject_IsInstance(grid_, tuple), \
        "\nArgument grid_ must be a tuple (gridy, gridx)  got %s " % type(grid_)
    assert PyObject_IsInstance(block_, tuple), \
        "\nArgument block_ must be a tuple (blocky, blockx) got %s " % type(block_)

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

    __device__ struct rgb struct_hsl_to_rgb(float h, float s, float l);
    __device__ struct hsl struct_rgb_to_hsl(float r, float g, float b);
    __device__ float hue_to_rgb(float m1, float m2, float h);
    __device__ float fmin_rgb_value(float red, float green, float blue);
    __device__ float fmax_rgb_value(float red, float green, float blue);

    __device__ float fmax_rgb_value(float red, float green, float blue)
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

    __device__ float fmin_rgb_value(float red, float green, float blue)
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

    __device__ float hue_to_rgb(float m1, float m2, float h)
        {
            if ((fabs(h) > 1.0f) && (h > 0.0f)) {
              h = (float)fmod(h, 1.0f);
            }
            else if (h < 0.0f){
            h = 1.0f - (float)fabs(h);
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

    __device__ struct hsl struct_rgb_to_hsl(float r, float g, float b)
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
    	        if ((fabs(t) > 6.0f) && (t > 0.0f)) {
                  t = (float)fmod(t, 6.0f);
                }
                else if (t < 0.0f){
                t = 6.0f - (float)fabs(t);
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



    __device__ struct rgb struct_hsl_to_rgb(float h, float s, float l)
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

    extern "C"  __global__ void rgb2hsl(float * r, float * g, float * b, int width, int height, double val_)
    { 
        int xx = blockIdx.x * blockDim.x + threadIdx.x;     
        int yy = blockIdx.y * blockDim.y + threadIdx.y;

        // Index value of the current pixel
        const int index = yy * height + xx;
        const int t_max = height * width;

        struct hsl hsl_; 
        struct rgb rgb_;
        float h; 
        if (index > 0 && index < t_max) { 

            float rr = r[index] ;
            float gg = g[index] ;
            float bb = b[index] ;

            hsl_ = struct_rgb_to_hsl(rr, gg, bb);
            h += hsl_.h + val_;
            rgb_ = struct_hsl_to_rgb(h, hsl_.s, hsl_.l); 

            r[index] = rgb_.r ;
            g[index] = rgb_.g ;
            b[index] = rgb_.b ;
        } 
    }
'''


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef object hsl_cupy(
        object cupy_array,
        object grid_,
        object block_,
        float val_,
        w, h
):

    module = cp.RawModule(code=rgb2hsl_cuda)
    rgb_to_hsl_ = module.get_function("rgb2hsl")

    cdef:
        r = cp.zeros((w, h), dtype=cp.float32)
        g = cp.zeros((w, h), dtype=cp.float32)
        b = cp.zeros((w, h), dtype=cp.float32)


    r = (cupy_array[:, :, 0] * ONE_255)
    g = (cupy_array[:, :, 1] * ONE_255)
    b = (cupy_array[:, :, 2] * ONE_255)

    rgb_to_hsl_(grid_, block_, (r, g, b, w, h, val_))

    cupy_array[:, :, 0] = cp.multiply(r, 255.0)
    cupy_array[:, :, 1] = cp.multiply(g, 255.0)
    cupy_array[:, :, 2] = cp.multiply(b, 255.0)

    cp.cuda.Stream.null.synchronize()

    return frombuffer(cupy_array.astype(cp.uint8).transpose(1, 0, 2).tobytes(), (w, h), "RGB").convert()





dithering_kernel = cp.RawKernel(
r'''
    extern "C"

    __global__ void dithering_kernel(unsigned char * source, 
    float * destination, int w, int h, double factor_)
    {

        int i = blockIdx.x * blockDim.x + threadIdx.x;
        int j = blockIdx.y * blockDim.y + threadIdx.y;
        
        const int index  = j * h + i;          // (2d)  
        const int index1 = j * h * 3 + i * 3;  // (3d)
        const int t_max  = w * h;
        const int t_max1 = w * h * 3;
        
        const float c1 = (float)(7.0f / 16.0f);
        const float c2 = (float)(3.0f / 16.0f);
        const float c3 = (float)(5.0f / 16.0f);
        const float c4 = (float)(1.0f / 16.0f);
        
        const int col = h * 3;
        
        
        float old_red   = (float)source[index1 % t_max1      ];
        float old_green = (float)source[(index1 + 1) % t_max1];
        float old_blue  = (float)source[(index1 + 2) % t_max1];
        __syncthreads();            
              
        float new_red   = (float)roundf(old_red   * factor_/255.0f) * 255.0f/factor_;   
        float new_green = (float)roundf(old_green * factor_/255.0f) * 255.0f/factor_;    
        float new_blue  = (float)roundf(old_blue  * factor_/255.0f) * 255.0f/factor_;    

        
        source[index1 % t_max1       ] = (unsigned char)new_red;
        source[(index1 + 1) % t_max1 ] = (unsigned char)new_green;
        source[(index1 + 2) % t_max1 ] = (unsigned char)new_blue;
        __syncthreads();
        
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


# TODO NOT WORKING
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef object dithering_gpu(
        object gpu_array_,
        object grid_,
        object block_,
        float factor_=1.0
):

    assert PyObject_IsInstance(gpu_array_, cp.ndarray), \
        "\nArgument gpu_array_ must be a cupy ndarray, got %s " % type(gpu_array_)

    if gpu_array_.dtype != cp.uint8:
        raise ValueError("\nArgument gpu_array_ datatype is invalid, "
                         "expecting cupy.uint8 got %s " % gpu_array_.dtype)

    cdef:
        Py_ssize_t w, h
    w, h = gpu_array_.shape[0], gpu_array_.shape[1]

    cdef destination = cupy.empty((w, h, 3), cp.float32)
    #destination = gpu_array_.copy()

    dithering_kernel(
        (grid_[0], grid_[1]),
        (block_[0], block_[1]),
        (gpu_array_, destination.astype(dtype=cp.float32), w, h, <float>factor_)
    )

    cp.cuda.Stream.null.synchronize()


    return frombuffer(destination.astype(dtype=cupy.uint8).transpose(1, 0, 2).tobytes(), (w, h), "RGB").convert()


# -------------------------------------------------------------------------------------------------------------------

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef inline object fisheye_gpu(
        object surface_,
        float focal,
        float focal_texture,
        object grid_,
        object block_
):
    """
    THIS SHADER CAN BE USE TO DISPLAY THE GAME THROUGH A LENS EFFECT

    Display a fisheye effect in real time given a pygame.Surface
    This shader can be applied directly to the pygame display


    :param surface_       : pygame.Surface 24-32 bit format 
    :param grid_             : tuple; grid values (grid_y, grid_x) e.g (25, 25). The grid values and block values must 
        match the texture and array sizes. 
    :param block_            : tuple; block values (block_y, block_x) e.g (32, 32). Maximum threads is 1024.
        Max threads = block_x * block_y
    :return               : pygame.Surface with a lens effect
    """

    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)

    try:
        gpu_array = pixels3d(surface_)

    except Exception as e:
        raise ValueError("\nCannot reference source pixels into a 3d array.\n %s " % e)

    return fisheye_cupy(cupy.asarray(gpu_array), focal, focal_texture, grid_, block_)

fisheye_kernel = cp.RawKernel(
    r'''
    
    extern "C" __global__
    
    void fisheye_kernel(unsigned char * destination, unsigned char * source,    
    const int w, const int h, double focal, double focal_texture)
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
            float r = (float)sqrt(nx2 + ny2);
            if (0.0f <= r && r <= 1.0f){
                float nr = (r * focal + 1.0f  - (float)sqrt(1.0f - (nx2 + ny2))) * focal_texture;
                if (nr <= 1.0f){
                    float theta = (float)atan2(ny, nx);
                    float nxn = nr * (float)cos(theta);
                    float nyn = nr * (float)sin(theta);
                    int x2 = (int)(nxn * w2 + w2);
                    int y2 = (int)(nyn * h2 + h2);
                    int v  = (int)(y2 * w + x2);
                    int index2 = x2  * h * 3 + y2 * 3;
                    destination[index1 + 0] = source[index2 + 0];
                    destination[index1 + 1] = source[index2 + 1];
                    destination[index1 + 2] = source[index2 + 2];
                }
            }            
        }
        __syncthreads();
    }
    ''',
    'fisheye_kernel'
)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)

cdef inline fisheye_cupy(
        object gpu_array, float focal, float focal_texture,
        object grid_, object block_
):


    cdef:
        Py_ssize_t w, h
    w, h = gpu_array.shape[:2]

    destination = gpu_array.copy()

    fisheye_kernel(
        grid_,
        block_,
        (destination, gpu_array, w, h, focal, focal_texture))

    cp.cuda.Stream.null.synchronize()

    return frombuffer(destination.transpose(1, 0, 2).tobytes(), (w, h), "RGB").convert()



# ---------------------------------------------------------------------------------------------------

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef inline object swirl_gpu(
        object surface_,
        float rad,
        object grid_,
        object block_,
        unsigned int centre_x,
        unsigned int centre_y

):


    """
    SWIRL AN IMAGE 
  
    Compatible 24 - 32 bit with or without alpha layer
    
    :param surface_ : Pygame.Surface compatible 24-32 bit  
    :param rad      : float; Float value representing a variable angle in radians
    :param grid_    : tuple; grid values (grid_y, grid_x) e.g (25, 25). The grid values and block values must 
        match the texture and array sizes. 
    :param block_   : tuple; block values (block_y, block_x) e.g (32, 32). Maximum threads is 1024.
        Max threads = block_x * block_y
    :return         : pygame.Surface with a swirl effect
    """

    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)

    try:
        gpu_array = pixels3d(surface_)

    except Exception as e:
        raise ValueError("\nCannot reference source pixels into a 3d array.\n %s " % e)

    return swirl_cupy(cupy.asarray(gpu_array), rad, grid_, block_, centre_x, centre_y)


swirl_kernel = cp.RawKernel(
    r'''

    extern "C" __global__

    void swirl_kernel(unsigned char * destination, unsigned char * source, double rad,   
    const int w, const int h, double x, double y)
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
            float r_max   = (float)sqrt(columns * columns + rows * rows);

            float di = (float)j - columns;
            float dj = (float)i - rows;

            float r = (float)sqrt(di * di + dj * dj);

            float c1 = (float)cos(rad * r/r_max);
            float c2 = (float)sin(rad * r/r_max);

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

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline swirl_cupy(
        object gpu_array,
        float rad,
        object grid_,
        object block_,
        unsigned int centre_x,
        unsigned int centre_y
):
    cdef:
        Py_ssize_t w, h
    w, h = gpu_array.shape[:2]

    if w == 0 or h == 0:
        raise ValueError("Surface width or height cannot be null!")

    destination = cupy.zeros((w, h, 3), dtype=cupy.uint8) # gpu_array.copy()

    swirl_kernel(
        grid_,
        block_,
        (destination, gpu_array, rad, w, h, <float>centre_x/<float>w, <float>centre_y/<float>h))

    cp.cuda.Stream.null.synchronize()

    return frombuffer(destination.transpose(1, 0, 2).tobytes(), (w, h), "RGB").convert()

#--------------------------------------------------------------------------------------------------

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef inline object wave_gpu(object surface_, float rad_, int size_, object grid_, object block_):
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

    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)

    try:
        gpu_array = pixels3d(surface_)

    except Exception as e:
        raise ValueError("\nCannot reference source pixels into a 3d array.\n %s " % e)

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
    
        unsigned int y_pos = (unsigned int) (j + size + (int) ((float) sin(rad + (float) j * c1) * (float) size));
        unsigned int x_pos = (unsigned int) (i + size + (int) ((float) sin(rad + (float) i * c1) * (float) size));
     
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


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline wave_cupy(
        object gpu_array, float rad_, int size_,
        object grid_, object block_
):
    cdef:
        Py_ssize_t w, h
    w, h = gpu_array.shape[:2]

    destination = cupy.empty((w, h, 3), dtype=cupy.uint8)

    wave_kernel(
        grid_,
        block_,
        (destination, gpu_array, rad_, size_, w, h )
    )

    cp.cuda.Stream.null.synchronize()

    return frombuffer(destination.transpose(1, 0, 2).tobytes(), (w, h), "RGB").convert()

# ---------------------------------------------------------------------------------------------------------------



# ---------------------------------------------------------------------------------------------------------------

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef inline object chromatic_gpu(
        object surface_,
        unsigned int delta_x,
        unsigned int delta_y,
        object grid_,
        object block_,
        float zoom = 1.0,
        float fx = 0.05
):
    """
    
    :param surface_: 
    :param delta_x: 
    :param delta_y: 
    :param grid_    : tuple; grid values (grid_y, grid_x) e.g (25, 25). The grid values and block values must 
        match the texture and array sizes. 
    :param block_   : tuple; block values (block_y, block_x) e.g (32, 32). Maximum threads is 1024.
        Max threads = block_x * block_y
    :param zoom: 
    :param fx: 
    :return: 
    """

    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)

    cdef  Py_ssize_t w, h
    w, h = surface_.get_size()

    assert w != 0, "Image width cannot be null !"
    assert h != 0, "Image height cannot be null !"

    if zoom < 0 or <float> floor(zoom) > <float> 0.9999:
        raise ValueError("Argument zoom must be in range [0.0 ... 0.9999]")

    if delta_x < 0 or delta_y < 0:
        raise ValueError("Arguments delta_x and delta_y must be > 0")

    delta_x %= w
    delta_y %= h

    if zoom < 0 or floor(zoom) > <float>0.99999999:
        raise ValueError("Argument zoom must be in range [0.0 ... 0.999]")

    if 0 > fx > 0.2:
        raise ValueError("Argument fx must be in range [0.0 ... 0.2]")

    try:
        gpu_array = pixels3d(surface_)

    except Exception as e:
        raise ValueError("\nCannot reference source pixels into a 3d array.\n %s " % e)

    return chromatic_cupy(cupy.asarray(gpu_array),
                          grid_, block_, delta_x, delta_y, zoom, fx)


chromatic_kernel = cp.RawKernel(

    '''

    extern "C" 
       
    __global__  void chromatic_kernel(unsigned char * destination, unsigned char * source,
        const int w, const int h, int delta_x, int delta_y, double zoom, double fx)
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
    
    float theta_rad = (float)atan2 (ny,nx);
        
    float nx2 = nx * nx;
    float ny2 = ny * ny; 
    
    float dist = (float)sqrt(nx2 + ny2); 
    
    float new_dist = dist * (zoom - fx);
    
    float new_ii = (float)cos((float)theta_rad) * new_dist ;
    float new_jj = (float)sin((float)theta_rad) * new_dist;
    
    int new_j = (int)((new_jj + dw) * (float)w); 
    int new_i = (int)((new_ii + dh) * (float)h); 
        
    if (i < h && j < w ) {
        __syncthreads();
        const int r = new_j * h * 3 + new_i * 3 + 0;      
        local_memory[0] = source[r];   
        destination[index1 + 0] = local_memory[0];

    }
    new_dist = dist * (zoom  - fx * 2); 
    
    new_j = (int)(((float)sin((float)theta_rad) * new_dist + dw) * (float)w); 
    new_i = (int)(((float)cos((float)theta_rad) * new_dist + dh) * (float)h); 
        
    if (i < h && j < w ) {
        __syncthreads();
        const int g = new_j * h * 3 + new_i * 3 + 1;      
        local_memory[1] = source[g];   
        destination[index1 + 1] = local_memory[1];

    }
    
    new_dist = dist * (zoom  - fx * 3); 
    
    new_ii = (float)cos((float)theta_rad) * new_dist;
    new_jj = (float)sin((float)theta_rad) * new_dist;
    
    new_j = (int)((new_jj + dw) * (float)w); 
    new_i = (int)((new_ii + dh) * (float)h); 
    
    if (i < h && j < w) {
        __syncthreads();
        const int b = new_j * h * 3 + new_i * 3 + 2;      
        local_memory[2] = source[b];   
        destination[index1 + 2] = local_memory[2];
    }    
}
    ''',
    'chromatic_kernel'
)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline chromatic_cupy(
        object gpu_array,
        object grid_,
        object block_,
        unsigned int delta_x,
        unsigned int delta_y,
        float zoom = 1.0,
        float fx = 0.05
):
    cdef:
        Py_ssize_t w, h
    w, h = gpu_array.shape[:2]

    destination = cupy.zeros((w, h, 3), dtype=cupy.uint8, order='C')

    chromatic_kernel(
        grid_,
        block_,
        (destination, gpu_array, w, h, delta_x, delta_y, zoom, fx)
    )

    cp.cuda.Stream.null.synchronize()

    return frombuffer(destination.transpose(1, 0, 2).tobytes(), (w, h), "RGB").convert()

# ---------------------------------------------------------------------------------------------------------------

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef inline object rgb_split_gpu(
        object surface_,
        float delta_x,
        float delta_y,
        object grid_,
        object block_
):
    """
    
    :param surface_: 
    :param delta_x: 
    :param delta_y: 
    :param grid_    : tuple; grid values (grid_y, grid_x) e.g (25, 25). The grid values and block values must 
        match the texture and array sizes. 
    :param block_   : tuple; block values (block_y, block_x) e.g (32, 32). Maximum threads is 1024.
        Max threads = block_x * block_y 
    :return: 
    """
    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)

    cdef Py_ssize_t prev_w, prev_h
    prev_w, prev_h = surface_.get_size()

    new_surface = scale(surface_, (prev_w + delta_x * <float>3.0, prev_h + delta_y * <float>3.0)).convert()

    assert prev_w != 0, "Image width cannot be null !"
    assert prev_h != 0, "Image height cannot be null !"

    if delta_x < 0 or delta_y < 0:
        raise ValueError("Arguments delta_x and delta_y must be > 0")

    delta_x %= prev_w
    delta_y %= prev_h

    try:
        gpu_array = pixels3d(surface_)

    except Exception as e:
        raise ValueError("\nCannot reference source pixels into a 3d array.\n %s " % e)

    return rgb_split_cupy(cupy.asarray(gpu_array),
                          prev_w, prev_h, grid_, block_, delta_x, delta_y)

rgb_split_kernel = cp.RawKernel(

    '''

    extern "C" __global__

    void rgb_split_kernel(unsigned char * destination, unsigned char * source,
        const int w, const int h, const int ww, const int hh, 
        double delta_x, double delta_y)
{

    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;

    int index1 = j * h * 3 + i * 3;
    int index2 = j * hh * 3  + i * 3;
    unsigned char local_memory[3]; 

    if (i < hh && j < ww) {
        
        const int r = (int)(j + delta_x * 1.0f) * hh * 3  + (int)(i + delta_y * 1.0f) * 3 + 0; 
        const int g = (int)(j + delta_x * 2.0f) * hh * 3  + (int)(i + delta_y * 2.0f) * 3 + 1; 
        const int b = (int)(j + delta_x * 3.0f) * hh * 3  + (int)(i + delta_y * 3.0f) * 3 + 2; 
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
    'rgb_split_kernel'
)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline rgb_split_cupy(
        object gpu_array,
        Py_ssize_t prev_w,
        Py_ssize_t prev_h,
        object grid_,
        object block_,
        float delta_x,
        float delta_y
):
    cdef:
        Py_ssize_t w, h
    w, h = gpu_array.shape[:2]

    destination = cupy.empty((prev_w, prev_h, 3), dtype=cupy.uint8, order='C')

    rgb_split_kernel(
        grid_,
        block_,
        (destination, gpu_array, prev_w, prev_h, w, h, delta_x, delta_y)
    )

    cp.cuda.Stream.null.synchronize()

    return frombuffer(destination.transpose(1, 0, 2).tobytes(), (prev_w, prev_h), "RGB").convert()

# ---------------------------------------------------------------------------------------------------------------

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef inline object zoom_gpu(
        object surface_,
        unsigned int delta_x,
        unsigned int delta_y,
        object grid_,
        object block_,
        float zoom = 0.9999
):
    """
    
    :param surface_: 
    :param delta_x: 
    :param delta_y: 
    :param grid_    : tuple; grid values (grid_y, grid_x) e.g (25, 25). The grid values and block values must 
        match the texture and array sizes. 
    :param block_   : tuple; block values (block_y, block_x) e.g (32, 32). Maximum threads is 1024.
        Max threads = block_x * block_y
    :param zoom: 
    :return: 
    """
    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)

    cdef Py_ssize_t w, h
    w, h = surface_.get_size()

    assert w != 0, "Image width cannot be null !"
    assert h != 0, "Image height cannot be null !"

    if zoom < 0 or <float>floor(zoom) > <float>0.9999:
        raise ValueError("Argument zoom must be in range [0.0 ... 0.9999]")

    if delta_x < 0 or delta_y < 0:
        raise ValueError("Arguments delta_x and delta_y must be > 0")

    delta_x %= w
    delta_y %= h

    try:
        gpu_array = pixels3d(surface_)

    except Exception as e:
        raise ValueError("\nCannot reference source pixels into a 3d array.\n %s " % e)

    return zoom_cupy(cupy.asarray(gpu_array),
                          grid_, block_, delta_x, delta_y, zoom)

zoom_kernel = cp.RawKernel(

    '''

    extern "C" 
       
    __global__  void zoom_kernel(unsigned char * destination, unsigned char * source,
        const int w, const int h, int delta_x, int delta_y, double zoom)
{

    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;

    int index1 = j * h * 3 + i * 3;   
    const unsigned int t_max = w * h * 3;  
    unsigned char local_memory[3]; 
    const float dw = float(delta_x)/(float)w;
    const float dh = (float)delta_y/(float)h;
    
    float nx = float((float)i / (float)h) - dh;  
    float ny = float((float)j / (float)w) - dw;
    
    float theta = (float)atan2 (ny,nx);
        
    float nx2 = nx * nx;
    float ny2 = ny * ny; 
    
    float dist = (float)sqrt(nx2 + ny2); 
    float new_dist = dist * zoom;
    
    int new_j = (int)(((float)sin((float)theta) * new_dist + dw) * (float)w); 
    int new_i = (int)(((float)cos((float)theta) * new_dist + dh) * (float)h); 
        
    if (i < h && j < w) {

        const int r = new_j * h * 3 + new_i * 3 + 0; 
        const int g = new_j * h * 3 + new_i * 3 + 1;
        const int b = new_j * h * 3 + new_i * 3 + 2;
        
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
    'zoom_kernel'
)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline zoom_cupy(
        object gpu_array,
        object grid_,
        object block_,
        unsigned int delta_x,
        unsigned int delta_y,
        float zoom = 0.99999
):
    cdef:
        Py_ssize_t w, h
    w, h = gpu_array.shape[:2]

    destination = cupy.zeros((w, h, 3), dtype=cupy.uint8)

    zoom_kernel(
        grid_,
        block_,
        (destination, gpu_array, w, h, delta_x, delta_y, zoom)
    )

    cp.cuda.Stream.null.synchronize()

    return frombuffer(destination.transpose(1, 0, 2).tobytes(), (w, h), "RGB").convert()

# ---------------------------------------------------------------------------------------------------------------



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef wavelength_mapper(unsigned int wavelength_min, unsigned int wavelength_max):
    """
    RETURN AN RGB COLOR VALUE MATCHING A SPECIFIC WAVELENGTH

    """
    cdef:
        rgb_color_int rgb_c
        int i

    REDMAP = [x for x in range(wavelength_min, wavelength_max)]

    cdef int n = 0
    for i in range(wavelength_min, wavelength_max):
        rgb_c = wavelength_to_rgb(i, 1.0)
        REDMAP[n] = (rgb_c.r, rgb_c.g, rgb_c.b)
        n += 1

    cdef float f_redmap = (<float> wavelength_max -
                           <float> wavelength_min - <float> 1.0) / (<float> 255.0 * <float> 3.0)
    cdef unsigned char[:, ::1] redmap_array = numpy.zeros((wavelength_max - wavelength_min, 3), numpy.uint8)

    i = 0
    for t in REDMAP:
        redmap_array[i, 0] = t[0]
        redmap_array[i, 1] = t[1]
        redmap_array[i, 2] = t[2]
        i += 1
    return redmap_array, f_redmap

"""
Violet: 380–450 nm (688–789 THz frequency)
Blue: 450–495 nm
Green: 495–570 nm
Yellow: 570–590 nm
Orange: 590–620 nm
Red: 620–750 nm (400–484 THz frequency)
"""
redmap_array,  f_redmap = wavelength_mapper(620, 750)
greenmap_array,   f_greenmap = wavelength_mapper(495, 570)
bluemap_array, f_bluemap = wavelength_mapper(450, 495)
heatmap_array, f_heatmap = wavelength_mapper(450, 720)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef inline object wavelength_map_gpu(
        object surface_,
        object grid_,
        object block_,
        unsigned short int layer_=0
):
    """
    CONVERT AN IMAGE TO AN EQUIVALENT REDSCALE | GREENSCALE | BLUESCALE EFFECT
    
    
    :param layer_   : integer; 0:red channel, 1:green, 2:blue 
    :param surface_ : pygame.Surface;
    :param grid_    : tuple; grid values (grid_y, grid_x) e.g (25, 25). The grid values and block values must 
        match the texture and array sizes. 
    :param block_   : tuple; block values (block_y, block_x) e.g (32, 32). Maximum threads is 1024.
        Max threads = block_x * block_y
    :return         : pygame.Surface with a redscale | greenscale | bluescale effect
    """

    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)

    cdef Py_ssize_t w, h
    w, h = surface_.get_size()

    assert w != 0, "Image width cannot be null !"
    assert h != 0, "Image height cannot be null !"

    try:
        gpu_array = pixels3d(surface_)

    except Exception as e:
        raise ValueError("\nCannot reference source pixels into a 3d array.\n %s " % e)

    return wavelength_map_cupy(cupy.asarray(gpu_array), grid_, block_, layer_)


wavelength_map_kernel = cp.RawKernel(

    '''
    extern "C" 

    __global__  void wavelength_map_kernel(unsigned char * destination, unsigned char * source, unsigned char * map,
        double map_c, const int w, const int h)
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

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline wavelength_map_cupy(
        object gpu_array,
        object grid_,
        object block_,
        unsigned short int layer_
):
    cdef:
        Py_ssize_t w, h
    w, h = gpu_array.shape[:2]

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

    return frombuffer(destination.transpose(1, 0, 2).tobytes(), (w, h), "RGB").convert()


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline wavelength_map_cupy_inplace(
        object gpu_array,
        cpu_array,
        object grid_,
        object block_,
        unsigned short int layer_
):
    cdef:
        Py_ssize_t w, h
    w, h = gpu_array.shape[:2]

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


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef inline object heatmap_gpu(
        object surface_,
        object grid_,
        object block_,
        bint invert_ = False
):
    """
    HEATMAP EffECT
     
    :param invert_ : boolean; Invert the effect  
    :param surface_: pygame.Surface; 
    :param grid_    : tuple; grid values (grid_y, grid_x) e.g (25, 25). The grid values and block values must 
        match the texture and array sizes. 
    :param block_   : tuple; block values (block_y, block_x) e.g (32, 32). Maximum threads is 1024.
        Max threads = block_x * block_y
    :return        : pygame.Surface
    """

    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)

    cdef Py_ssize_t w, h
    w, h = surface_.get_size()

    assert w != 0, "Image width cannot be null !"
    assert h != 0, "Image height cannot be null !"

    try:
        rgb_array = pixels3d(surface_)

    except Exception as e:
        raise ValueError("\nCannot reference source pixels into a 3d array.\n %s " % e)

    return heatmap_cupy(cupy.asarray(rgb_array), rgb_array, grid_, block_, invert_)

heatmap_kernel = cp.RawKernel(

    '''
    extern "C" 

    __global__  void heatmap_kernel(unsigned char * destination, unsigned char * source, unsigned char * map,
        double map_c, bool invert, const int w, const int h)
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

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline heatmap_cupy(
        object gpu_array,
        rgb_array,
        object grid_,
        object block_,
        bint invert_
):
    cdef:
        Py_ssize_t w, h
    w, h = gpu_array.shape[:2]

    destination = cp.zeros((w, h, 3), dtype=cupy.uint8)

    cdef float f_map

    heatmap_kernel(
        grid_,
        block_,
        (destination, gpu_array, cupy.asarray(heatmap_array), f_heatmap, int(invert_), w, h)
    )

    cp.cuda.Stream.null.synchronize()

    return frombuffer(destination.transpose(1, 0, 2).tobytes(), (w, h), "RGB").convert()




@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef void heatmap_gpu_inplace(
        object surface_,
        object grid_,
        object block_,
        bint invert_ = False
):
    """
    HEATMAP EFFECT 
    
    * Apply effect inplace

    :param invert_  : boolean; Invert the effect
    :param surface_ : pygame.Surface
    :param grid_    : tuple; grid values (grid_y, grid_x) e.g (25, 25). The grid values and block values must 
        match the texture and array sizes. 
    :param block_   : tuple; block values (block_y, block_x) e.g (32, 32). Maximum threads is 1024.
        Max threads = block_x * block_y
    :return         : void 
    """

    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)

    cdef Py_ssize_t w, h
    w, h = surface_.get_size()

    assert w != 0, "Image width cannot be null !"
    assert h != 0, "Image height cannot be null !"

    try:
        rgb_array = pixels3d(surface_)

    except Exception as e:
        raise ValueError("\nCannot reference source pixels into a 3d array.\n %s " % e)

    heatmap_cupy_inplace(cupy.asarray(rgb_array), rgb_array, grid_, block_, invert_)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline heatmap_cupy_inplace(
        object gpu_array,
        rgb_array,
        object grid_,
        object block_,
        bint invert_
):
    cdef:
        Py_ssize_t w, h
    w, h = gpu_array.shape[:2]

    destination = cp.zeros((w, h, 3), dtype=cupy.uint8)

    cdef float f_map

    heatmap_kernel(
        grid_,
        block_,
        (destination, gpu_array, cupy.asarray(heatmap_array), f_heatmap, int(invert_), w, h)
    )

    cp.cuda.Stream.null.synchronize()

    rgb_array[:, :, 0] = destination[:, :, 0].get()
    rgb_array[:, :, 1] = destination[:, :, 1].get()
    rgb_array[:, :, 2] = destination[:, :, 2].get()



# --------------------------------------------------------------------------------------------------------------------

cpdef inline predator_gpu(
        object surface_,
        grid_,
        block_,
        unsigned int bloom_smooth    = 50,
        unsigned int bloom_threshold = 50,
        bint inv_colormap            = False,
        int blend                    = pygame.BLEND_RGB_ADD,
        bint bloom_flag              = pygame.BLEND_RGB_ADD

):
    shader_bloom_fast1(surface_, smooth_=bloom_smooth, threshold_=bloom_threshold, flag_=bloom_flag, saturation_=True)

    try:
        cpu_array = pixels3d(surface_)
        gpu_array = cupy.asarray(cpu_array)

    except Exception as e:
        raise ValueError("\nCannot reference source pixels into a 3d array.\n %s " % e)

    cdef Py_ssize_t w, h
    w, h = surface_.get_size()

    gpu_array_copy = gpu_array.copy()
    sobel_cupy(gpu_array_copy)

    gpu_array = grayscale__lum_cupy(gpu_array)

    # cpu_arr = cp.asnumpy(gpu_array)
    # shader_rgb_to_yiq_q_comp_inplace_c(cpu_arr)
    # gpu_array = cp.asarray(cpu_arr)

    heatmap_cupy_inplace(gpu_array, cpu_array, grid_, block_, invert_=inv_colormap)
    surf = frombuffer(gpu_array_copy.transpose(1, 0, 2).tobytes(), (w, h), "RGB").convert()

    del gpu_array_copy
    del cpu_array

    surface_.blit(surf, (0, 0), special_flags=blend)


    return surface_.convert()


# -------------------------------------------------------------------------------------------------------


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef inline object downscale_surface_gpu(
        object surface_,
        object grid_,
        object block_,
        float zoom,
        int w2,
        int h2
):

    assert PyObject_IsInstance(surface_, pygame.Surface), \
        "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)

    cdef Py_ssize_t w, h
    w, h = surface_.get_size()

    assert w != 0, "Image width cannot be null !"
    assert h != 0, "Image height cannot be null !"

    if zoom < 0 or <float> floor(zoom) > <float> 0.9999:
        raise ValueError("Argument zoom must be in range [0.0 ... 0.9999]")

    try:
        gpu_array = pixels3d(surface_)

    except Exception as e:
        raise ValueError("\nCannot reference source pixels into a 3d array.\n %s " % e)

    return downscale_surface_cupy(cupy.asarray(gpu_array),
                     grid_, block_, zoom, w2, h2)

downscale_surface_kernel = cp.RawKernel(

    '''

    extern "C" 

    __global__  void downscale_surface_kernel(unsigned char * destination, unsigned char * source,
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

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline downscale_surface_cupy(
        object gpu_array,
        object grid_,
        object block_,
        float zoom,
        int w2,
        int h2
):
    cdef:
        Py_ssize_t w, h
    w, h = gpu_array.shape[:2]

    destination = cupy.zeros((w2, h2, 3), dtype=cupy.uint8)

    downscale_surface_kernel(
        grid_,
        block_,
        (destination, gpu_array, w2, h2, w, h, zoom)
    )

    cp.cuda.Stream.null.synchronize()

    return frombuffer(destination.transpose(1, 0, 2).tobytes(), (w2, h2), "RGB").convert()




















#
#
# cpdef inline object pixelation_gpu(
#         object surface_,
#         unsigned int kernel_size,
#         object grid_,
#         object block_
# ):
#
#
#     assert PyObject_IsInstance(surface_, pygame.Surface), \
#         "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)
#
#     try:
#         gpu_array = pixels3d(surface_)
#
#     except Exception as e:
#         raise ValueError("\nCannot reference source pixels into a 3d array.\n %s " % e)
#
#     return pixelation_cupy(cupy.asarray(gpu_array), kernel_size, grid_, block_)
#
#
#
# pixelation_kernel = cp.RawKernel(
#     '''
#     extern "C"
#
#     __global__ void pixelation_kernel(double* buffer, int filter_size,
#                      double* return_value)
#     {
#
#     float sum_ = 0.0;
#
#     for (int i=0; i<filter_size; ++i){
#         sum_ += buffer[i];
#     }
#
#     return_value[0] = sum_/filter_size;
#
#     }
#     ''',
#     'pixelation_kernel'
# )
#
#
# @cython.boundscheck(False)
# @cython.wraparound(False)
# @cython.nonecheck(False)
# @cython.cdivision(True)
# cdef inline pixelation_cupy(
#         object gpu_array, unsigned int kernel_size, object grid_, object block_):
#     cdef:
#         Py_ssize_t w, h
#     w, h = gpu_array.shape[:2]
#
#     destination = cupy.empty((w, h, 3), dtype=cupy.uint8)
#
#     r = gpu_array[:, :, 0]
#
#     px_r = cupyx.scipy.ndimage.generic_filter(
#         r, pixelation_kernel, (10, 10)).astype(dtype=cp.uint8)
#
#     destination[:, :, 0] = px_r
#     destination[:, :, 1] = px_r
#     destination[:, :, 2] = px_r
#
#     cp.cuda.Stream.null.synchronize()
#
#     return frombuffer(destination.transpose(1, 0, 2).tobytes(), (w, h), "RGB").convert()
#
#
#
#










#
# cpdef inline object glitch_gpu(object surface_, float angle_, float amplitude_,object grid_, object block_):
#
#
#     assert PyObject_IsInstance(surface_, pygame.Surface), \
#         "\nArgument surface_ must be a pygame.Surface type, got %s " % type(surface_)
#
#     try:
#         gpu_array = pixels3d(surface_)
#
#     except Exception as e:
#         raise ValueError("\nCannot reference source pixels into a 3d array.\n %s " % e)
#
#     return glitch_cupy(cupy.asarray(gpu_array), angle_, amplitude_, grid_, block_)
#
#
# glitch_kernel = cp.RawKernel(
#
#     '''
#     #include <cstdlib.h>
#
#     extern "C" __global__
#
#     void glitch_kernel(unsigned char * destination, unsigned char * source,
#         double angle_, double amplitude_, const int w, const int h)
# {
#
#     unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
#     unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
#
#     const int index1 = j * h * 3 + i * 3;
#
#
#     if (i < h && j < w) {
#
#         angle_ = angle_ + rand() * i;
#
#         int ii = (i + (int)((float)cos(angle_) * amplitude_));
#                 if (ii > (int)w - 1){
#                     ii = (int)w - 1;
#                     }
#                 if (ii < 0){
#                     ii = 0;
#                     }
#
#         int index2 = j * h * 3 + ii * 3;
#
#         destination[index1 + 0] = source[index2 + 0];
#         destination[index1 + 1] = source[index2 + 1];
#         destination[index1 + 2] = source[index2 + 2];
#
#     }
#
# }
#     ''',
#     'glitch_kernel'
# )
#
# @cython.boundscheck(False)
# @cython.wraparound(False)
# @cython.nonecheck(False)
# @cython.cdivision(True)
# cdef inline glitch_cupy(
#         object gpu_array, float angle_, float amplitude_,
#         object grid_, object block_
# ):
#     cdef:
#         Py_ssize_t w, h
#     w, h = gpu_array.shape[:2]
#
#     destination = cupy.empty((w, h, 3), dtype=cupy.uint8)
#
#
#
#     glitch_kernel(
#         grid_,
#         block_,
#         (destination, gpu_array, angle_, amplitude_, w, h)
#     )
#
#     cp.cuda.Stream.null.synchronize()
#
#     return frombuffer(destination.transpose(1, 0, 2).tobytes(), (w, h), "RGB").convert()
#
# # ---------------------------------------------------------------------------------------------------------------
#


#
#
#
# source=r'''
# extern "C"{
# __global__ void copyKernel(float* output,
#                            cudaTextureObject_t texObj,
#                            int width, int height)
# {
#     unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
#     unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
#     unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;
#
#     // Read from texture and write to global memory
#     float u = x;
#     float v = y;
#     float w = z;
#     if (x < width && y < height && z < 3)
#         output[z * width * height + y * width + x] = tex3D<float>(texObj, u, v, w);
# }
# }
# '''
#
# cpdef test(image, grid, block):
#
#     width, height = image.get_size()
#     depth = 3
#
#     ch = cp.cuda.texture.ChannelFormatDescriptor(32, 0, 0, 0, cp.cuda.runtime.cudaChannelFormatKindFloat)
#     arr2 = cp.cuda.texture.CUDAarray(ch, width, height, depth)
#     res = cp.cuda.texture.ResourceDescriptor(cp.cuda.runtime.cudaResourceTypeArray, cuArr=arr2)
#     tex = cp.cuda.texture.TextureDescriptor((cp.cuda.runtime.cudaAddressModeClamp, cp.cuda.runtime.cudaAddressModeClamp),
#                                             cp.cuda.runtime.cudaFilterModePoint,
#                                             cp.cuda.runtime.cudaReadModeElementType)
#     texobj = cp.cuda.texture.TextureObject(res, tex)
#
#     tex_data = cupy.asarray(pixels3d(image)).astype(dtype=cupy.float32).reshape(3, height, width)
#
#     real_output = cp.zeros_like(tex_data)
#     expected_output = cp.zeros_like(tex_data)
#
#     arr2.copy_from(tex_data)
#     arr2.copy_to(expected_output)
#
#     ker = cp.RawKernel(source, 'copyKernel')
#
#
#     ker((grid[1], grid[0], 3), (block[1], block[0], 1), (real_output, texobj, width, height))
#
#     # return make_surface(cupy.asnumpy(real_output).reshape((800, 800, 3)).astype(dtype=cupy.uint8))
#     return frombuffer(real_output.astype(
#         dtype=cupy.uint8).reshape((width, height, 3)).transpose(1, 0, 2).tobytes(), (width, height), "RGB").convert()
#
# source_surfobj = r"""
# extern "C" {
# __global__ void writeKernel3D(cudaSurfaceObject_t surf,
#                               int width, int height, int depth)
# {
#     unsigned int w = blockIdx.x * blockDim.x + threadIdx.x;
#     unsigned int h = blockIdx.y * blockDim.y + threadIdx.y;
#     unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;
#     if (w < width && h < height && z < depth)
#     {
#         float value = z * width * height + h * width + w;
#         value *= 3.0;
#         surf3Dwrite(value, surf, w*4, h, z);
#     }
# }
# }
# """
#
# from cupy.cuda import runtime
# from cupy.cuda.texture import (ChannelFormatDescriptor, CUDAarray,
#                                ResourceDescriptor, TextureDescriptor,
#                                TextureObject, TextureReference,
#                                SurfaceObject)
#
#
# cpdef test_write_float_surface(image):
#
#
#         width, height, depth = 800, 800, 3
#
#         shape = (depth, height, width)
#
#         real_output = cupy.zeros(shape, dtype=cupy.float32)
#
#         ch = ChannelFormatDescriptor(32, 0, 0, 0,
#                                      runtime.cudaChannelFormatKindFloat)
#
#         # expected_output = cupy.arange(numpy.prod(shape), dtype=cupy.float32)
#         # expected_output = expected_output.reshape(shape) * 3.0
#
#         expected_output = cupy.asarray(pixels3d(image), dtype=cupy.float32).reshape(shape)
#         expected_output *= 3.0
#
#         arr = CUDAarray(ch, width, height, depth,
#                         runtime.cudaArraySurfaceLoadStore)
#
#         arr.copy_from(real_output)
#         res = ResourceDescriptor(runtime.cudaResourceTypeArray, cuArr=arr)
#
#         surfobj = SurfaceObject(res)
#         mod = cupy.RawModule(code=source_surfobj)
#
#
#         ker = mod.get_function("writeKernel3D")
#         block = (25, 25, 3)
#         grid = (32, 32, 1)
#
#         ker(block,
#             grid,
#             (surfobj, width, height, depth))
#
#         arr.copy_to(real_output)
#         print(real_output.shape, expected_output.shape)
#         # return make_surface(cupy.asnumpy(real_output.reshape((800, 800, 3))).astype(dtype=numpy.uint8))
#         assert (real_output == expected_output).all()
