# cython: binding=False, boundscheck=False, wraparound=False, nonecheck=False, cdivision=True,
# profile=False, initializedcheck=False
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


"""
# **GPU Image Processing Functions Summary**

## **1. Memory & GPU Information**
- `get_gpu_free_mem()`
- `get_gpu_maxmem()`
- `get_gpu_pci_bus_id()`
- `get_compute_capability()`
- `get_max_grid_per_block()`
- `get_divisors(n)`

## **2. Grid & Block Management**
- `block_grid(w, h)`

---

## **3. Image Processing Effects**

### **Image Inversion**
- `invert_gpu(surface_)`
- `invert_gpu_inplace(surface_)`
- `invert_buffer_gpu(bgr_array)`
- `invert_rgb_array_inplace(rgb_array)`

### **Sepia Effect**
- `sepia_gpu(surface_)`
- `sepia_cupy(gpu_array_)`
- `sepia_rgb_array_gpu(rgb_array)`
- `sepia_buffer_gpu(grid_, block_, w, h, bgr_array, format_32=*)`

### **Grayscale Conversion**
- `grayscale_gpu(surface_)`
- `grayscale_cupy(gpu_array_)`
- `grayscale_lum_gpu(surface_)`
- `grayscale_lum_cupy(gpu_array_)`

### **Median Filtering**
- `median_filter_gpu(surface_, size_=*)`
- `median_filter_cupy(gpu_array_, size_=*)`
- `median_generic_filter_gpu(surface_, size_=*)`
- `median_generic_filter_cupy(gpu_array_, size_=*)`

### **Gaussian Filtering**
- `gaussian_5x5_gpu(surface_)`
- `gaussian_5x5_cupy(gpu_array_)`
- `gaussian_3x3_gpu(surface_)`
- `gaussian_3x3_cupy(gpu_array_)`

### **Edge Detection**
- `sobel_gpu(surface_)`
- `sobel_cupy(w, h, gpu_array_)`
- `prewitt_gpu(surface_)`
- `prewitt_cupy(w, h, gpu_array_)`
- `canny_gpu(surface_)`
- `canny_cupy(w, h, gpu_array_)`

### **Color Reduction**
- `color_reduction_gpu(surface_, color_number=*)`
- `color_reduction_cupy(w, h, gpu_array_, color_number)`

### **HSV & HSL Adjustments**
- `hsv_gpu(surface_, val_, grid_, block_)`
- `hsv_cupy(gpu_array, grid_, block_, val_, w, h)`
- `hsl_gpu(surface_, val_, grid_=*, block_=*)`
- `hsl_cupy(cupy_array, grid_, block_, val_, w, h)`

### **Brightness Adjustment**
- `brightness_gpu(surface_, brightness_factor, grid_, block_)`
- `brightness_cupy(cupy_array, grid_, block_, brightness_factor, w, h)`

### **Saturation Adjustment**
- `saturation_gpu(surface_, grid_, block_, val_=*)`
- `saturation_cupy(w, h, cupy_array, grid_, block_, val_=*)`

### **Bilateral Filtering**
- `bilateral_gpu(surface_, kernel_size_)`
- `bilateral_cupy(w, h, gpu_array_, kernel_size_)`

### **Embossing**
- `emboss5x5_gpu(surface_)`
- `emboss5x5_cupy(w, h, gpu_array_)`

---

## **4. Image Transformations**

### **Blending & Sharpening**
- `blending_gpu(source_, destination_, percentage_)`
- `sharpen_gpu(surface_)`
- `sharpen1_gpu(surface_, grid_, block_)`

### **Mirroring**
- `mirroring_gpu(surface_, grid_, block_, format_=*)`
- `mirroring_cupy(w, h, gpu_array_, grid_, block_, format_=*)`

### **Zoom Effects**
- `zoom_gpu(surface_, delta_x, delta_y, grid_, block_, zoom=*)`
- `zoom_cupy(gpu_array, grid_, block_, centre_x, centre_y, zoom=*)`

### **Chromatic Aberration**
- `chromatic_gpu(surface_, delta_x, delta_y, grid_, block_, zoom=*, fx=*)`

### **RGB Split Effect**
- `rgb_split_gpu(surface_, delta_x, delta_y, grid_, block_)`
- `rgb_split_cupy(gpu_array, prev_w, prev_h, grid_, block_, delta_x, delta_y)`

### **Wave & Ripple Effects**
- `wave_gpu(surface_, rad_, size_, grid_, block_)`
- `ripple_effect_gpu(grid, block, w, h, previous, current, texture_array, background_array)`

### **Swirl Effect**
- `swirl_gpu(surface_, rad, grid_, block_, centre_x, centre_y)`
- `swirl_cupy(gpu_array, rad, grid_, block_, centre_x, centre_y)`

### **Fisheye Effect**
- `fisheye_gpu(surface_, focal, focal_texture, grid_, block_)`
- `fisheye_cupy(gpu_array, focal, focal_texture, grid_, block_)`

---

## **5. Filters & Enhancements**

### **Binary Processing Filter (BPF)**
- `bpf_gpu(surface_, threshold_=*)`
- `bpf_cupy(gpu_array_, threshold_)`
- `bpf_inplace_gpu(grid_, block_, surface_, threshold_=*)`
- `bpf_inplace_cupy(grid_, block_, w, h, gpu_array_, threshold_)`
- `bpf_buffer_gpu(grid, block, w, h, bgr_array, threshold_=*, format_32=*)`

### **Cartoon Effect**
- `cartoon_gpu(surface_, sobel_threshold_=*, median_kernel_=*, color_=*, contour_=*, flag_=*)`
- `cartoon_cupy(surface_, sobel_threshold_, median_kernel_, color_, contour_, flag_)`

### **Bloom Effect**
- `bloom_gpu(surface_, threshold_=*, fast_=*, flag_=*, factor_=*)`
- `bloom_array(gpu_array_, threshold_=*, fast_=*, flag_=*, mask_=*)`

### **Dithering**
- `dithering_gpu(gpu_array_, grid_, block_, factor_=*)`

---

## **6. Heatmap & Wavelength Processing**
- `heatmap_gpu(surface_, grid_, block_, invert_=*)`
- `heatmap_cupy(gpu_array, rgb_array, grid_, block_, invert_)`
- `heatmap_gpu_inplace(surface_, grid_, block_, invert_=*)`
- `heatmap_cupy_inplace(gpu_array, rgb_array, grid_, block_, invert_)`

- `wavelength_map_gpu(surface_, grid_, block_, layer_=*)`
- `wavelength_mapper(wavelength_min, wavelength_max)`
- `wavelength_map_cupy(gpu_array, grid_, block_, layer_)`
- `wavelength_map_cupy_inplace(gpu_array, cpu_array, grid_, block_, layer_)`

---

## **7. Miscellaneous**
- `predator_gpu(surface_, grid_, block_, bloom_smooth=*, bloom_threshold=*, inv_colormap=*, blend=*, bloom_flag=*)`
- `area24_gpu(x, y, background_rgb, mask_alpha, intensity=*, color=*)`
- `downscale_surface_gpu(surface_, grid_, block_, zoom, w2, h2)`
- `downscale_surface_cupy(gpu_array, grid_, block_, zoom, w2, h2)`

---

"""


try:
    cimport numpy as np

except ImportError:
    raise ImportError("\n<numpy> library is missing on your system."
          "\nTry: \n   C:\\pip install numpy on a window command prompt.")


except ImportError:
    raise ImportError("\n<cupy> library is missing on your system."
          "\nTry: \n   C:\\pip install cupy on a window command prompt.")


cdef extern from 'Include/Shaderlib.c':

    float fmax_rgb_value(float red, float green, float blue)nogil;
    float fmin_rgb_value(float red, float green, float blue)nogil;
    float hue_to_rgb(float m1, float m2, float hue)nogil;
    int *quickSort(int arr[], int low, int high)nogil;
    double *my_sort(double buffer[], int filter_size)nogil;
    rgb_color_int wavelength_to_rgb(int wavelength, float gamma)nogil;

    struct rgb_color_int:
        int r;
        int g;
        int b;

    struct hsl:
        float h
        float s
        float l


cpdef long long int get_gpu_free_mem()
cpdef long long int get_gpu_maxmem()
cpdef str get_gpu_pci_bus_id()
cpdef str get_compute_capability()
cpdef unsigned int get_max_grid_per_block()
cdef get_divisors(int n)
cpdef block_grid(int w, int h)

cpdef invert_gpu(surface_)
cpdef void invert_gpu_inplace(surface_)
cpdef void invert_buffer_gpu(unsigned char [::1] bgr_array)
cpdef void invert_rgb_array_inplace(rgb_array)

cpdef sepia_gpu(surface_)
cdef void sepia_cupy(gpu_array_)
cpdef void sepia_rgb_array_gpu(rgb_array)
cpdef void sepia_buffer_gpu(
    tuple grid_,
    tuple block_,
    const Py_ssize_t w,
    const Py_ssize_t h,
    unsigned char [::1] bgr_array,
    bint format_32 = *
)


cpdef bpf_gpu(surface_, unsigned char threshold_ = *)
cdef void bpf_cupy(gpu_array_, unsigned char threshold_)

cpdef void bpf_inplace_gpu(
    tuple grid_,
    tuple block_,
    surface_,
    unsigned char threshold_ = *
)
cdef void bpf_inplace_cupy(
    tuple grid_,
    tuple block_,
    const unsigned int w,
    const unsigned int h,
    gpu_array_,
    unsigned char threshold_
)

cpdef bpf_buffer_gpu(
    tuple grid,
    tuple block,
    const Py_ssize_t w,
    const Py_ssize_t h,
    unsigned char [::1] bgr_array,
    unsigned char threshold_ = *,
    bint format_32 = *
)


cpdef grayscale_gpu(surface_)
cdef void grayscale_cupy(gpu_array_)

cpdef grayscale_lum_gpu(surface_)
cdef void grayscale_lum_cupy(gpu_array_)

cpdef median_filter_gpu(surface_, unsigned int size_ = *)
cdef void median_filter_cupy(gpu_array_, unsigned int size_ = *)

cpdef median_generic_filter_gpu(surface_, unsigned int size_ = *)
cdef void median_generic_filter_cupy(gpu_array_, unsigned int size_ = *)

cpdef gaussian_5x5_gpu(surface_)
cdef void gaussian_5x5_cupy(gpu_array_)

cpdef gaussian_3x3_gpu(surface_)
cdef void gaussian_3x3_cupy(gpu_array_)


cpdef sobel_gpu(surface_)
cdef void sobel_cupy(const Py_ssize_t w, const Py_ssize_t h, gpu_array_)

cpdef prewitt_gpu( surface_)
cdef void prewitt_cupy(
    const Py_ssize_t w,
    const Py_ssize_t h,
    gpu_array_)

cpdef canny_gpu(surface_)
cdef void canny_cupy(
    const Py_ssize_t w,
    const Py_ssize_t h,
    gpu_array_)

cpdef color_reduction_gpu(surface_, int color_number = *)
cdef void color_reduction_cupy(
    const Py_ssize_t w,
    const Py_ssize_t h,
    gpu_array_,
    const int color_number
)


cpdef hsv_gpu(
    surface_,
    float val_,
    tuple grid_,
    tuple block_
)
cdef void hsv_cupy(
    gpu_array,
    tuple grid_,
    tuple block_,
    float val_,
    const Py_ssize_t w,
    const Py_ssize_t h
)

cpdef mult_downscale_gpu(gpu_array)

# cpdef object zoom_in_gpu(object surface, int w, int h)
# cpdef object upscale_gpu(object gpu_array_, int w, int h)

# ------------------ BLOOM ------------------
cdef void bpf_c(gpu_array_, unsigned int threshold_=*)
cdef void gaussian_3x3_c(gpu_array_)
cdef void gaussian_5x5_c(gpu_array_)
cpdef upscale_c(object gpu_array_, int new_width, int new_height, int order_=*)

cpdef bloom_gpu(surface_,
        unsigned int threshold_ = *,
        bint fast_ = *,
        int flag_ = *,
        unsigned short int factor_ = *
        )

cpdef bloom_array(
        gpu_array_,
        unsigned int threshold_ = *,
        bint fast_ = *,
        int flag_  = *,
        mask_ = *
)
# ----------------------------------------------

cpdef cartoon_gpu(
        surface_,
        int sobel_threshold_ = *,
        int median_kernel_   = *,
        unsigned char color_ = *,
        bint contour_        = *,
        unsigned char flag_  = *
)
cdef void canny_cupy_c(gpu_array_)
cdef void sobel_cupy_c(gpu_array_)
cdef void median_cupy_c(gpu_array_, unsigned int size_=*)
cdef color_reduction_cupy_c(gpu_array_, int color_number)
cdef cartoon_cupy(
        surface_,
        int sobel_threshold_,
        int median_kernel_,
        int color_,
        bint contour_,
        int flag_)


cpdef object blending_gpu(object source_, object destination_, float percentage_)

cpdef object sharpen_gpu(object surface_)
cpdef object sharpen1_gpu(object surfaace_, grid_, block_)


cpdef ripple_effect_gpu(
       tuple grid,
       tuple block,
       const Py_ssize_t w, 
       const Py_ssize_t h,
       previous,
       current,
       texture_array,
       background_array
       )

cpdef mirroring_gpu(
    surface_,
    tuple grid_,
    tuple block_,
    bint format_ = *
)
cdef mirroring_cupy(
    const Py_ssize_t w,
    const Py_ssize_t h,
    gpu_array_,
    tuple grid_,
    tuple block_,
    bint format_=*)


cpdef saturation_gpu(
    surface_,
    tuple grid_,
    tuple block_,
    float val_ = *
)
cdef saturation_cupy(
    const Py_ssize_t w,
    const Py_ssize_t h,
    cupy_array,
    tuple grid_,
    tuple block_,
    float val_ = *
)


cpdef bilateral_gpu(surface_, const unsigned int kernel_size_)
cdef bilateral_cupy(const Py_ssize_t w,
    const Py_ssize_t h,
    gpu_array_,
    const unsigned int kernel_size_)


cpdef emboss5x5_gpu(surface_)
cdef emboss5x5_cupy(
    const Py_ssize_t w,
    const Py_ssize_t h,
    gpu_array_)

cpdef tuple area24_gpu(
    const int x, 
    const int y,
    background_rgb,
    mask_alpha,
    float intensity=*,
    color=*
)

cpdef brightness_gpu(
    surface_,
    const float brightness_factor,
    tuple grid_,
    tuple block_
)
cdef brightness_cupy(
    cupy_array,
    tuple grid_,
    tuple block_,
    const float brightness_factor,
    const Py_ssize_t w,
    const Py_ssize_t h
)


cpdef hsl_gpu(
    surface_,
    const float val_,
    tuple grid_  = *,
    tuple block_ = *
)
cdef hsl_cupy(
    cupy_array,
    tuple grid_,
    tuple block_,
    const float val_,
    const Py_ssize_t w, 
    const Py_ssize_t h
)

cpdef dithering_gpu(
    gpu_array_,
    tuple grid_,
    tuple block_,
    float factor_ = *
)


cpdef fisheye_gpu(surface_, float focal, float focal_texture, tuple grid_, tuple block_)

cdef fisheye_cupy(
    gpu_array, 
    const float focal, 
    const float focal_texture,
    tuple grid_, 
    tuple block_
)


cpdef swirl_gpu(
    surface_,
    const float rad,
    tuple grid_,
    tuple block_,
    const unsigned int centre_x,
    const unsigned int centre_y
)

cdef swirl_cupy(
    gpu_array,  
    const float rad,         
    tuple grid_,      
    tuple block_,     
    const unsigned int centre_x,  
    const unsigned int centre_y
)

cpdef wave_gpu(
    surface_, 
    const float rad_, 
    const int size_, 
    tuple grid_, 
    tuple block_
    )


cpdef rgb_split_gpu(
    surface_,
    float delta_x,
    float delta_y,
    tuple grid_,
    tuple block_
)

cdef rgb_split_cupy(
    gpu_array,
    const Py_ssize_t prev_w,
    const Py_ssize_t prev_h,
    tuple grid_,
    tuple block_,
    const float delta_x,
    const float delta_y
)


cpdef chromatic_gpu(
    surface_,
    unsigned int delta_x,
    unsigned int delta_y,
    tuple grid_,
    tuple block_,
    float zoom = *,
    float fx = *
)


cpdef zoom_gpu(
    surface_,
    unsigned int delta_x,
    unsigned int delta_y,
    tuple grid_,
    tuple block_,
    float zoom = *
)
cdef zoom_cupy(
    gpu_array,
    tuple grid_,
    tuple block_,
    const unsigned int centre_x,
    const unsigned int centre_y,
    float zoom = *
)


cpdef wavelength_map_gpu(
    surface_,
    tuple grid_,
    tuple block_,
    unsigned short int layer_=*
)

cdef tuple wavelength_mapper(
    const unsigned int wavelength_min, 
    const unsigned int wavelength_max
    )

cdef wavelength_map_cupy(
    gpu_array,
    tuple grid_,
    tuple block_,
    unsigned short int layer_
)

cdef wavelength_map_cupy_inplace(
    gpu_array,
    cpu_array,
    tuple grid_,
    tuple block_,
    unsigned short int layer_
)

cpdef heatmap_gpu(
    surface_,
    tuple grid_,
    tuple block_,
    bint invert_ = *
)

cdef heatmap_cupy(
    gpu_array,
    rgb_array,
    tuple grid_,
    tuple block_,
    const bint invert_
)

cpdef void heatmap_gpu_inplace(
    surface_,
    tuple grid_,
    tuple block_,
    bint invert_ = *
)

cdef void heatmap_cupy_inplace(
    gpu_array,
    rgb_array,
    tuple grid_,
    tuple block_,
    bint invert_
)

cpdef predator_gpu(
    surface_,
    tuple grid_,
    tuple block_,
    unsigned int bloom_smooth    = *,
    unsigned int bloom_threshold = *,
    bint inv_colormap            = *,
    int blend                    = *,
    bint bloom_flag              = *

)

cpdef downscale_surface_gpu(
    surface_,
    tuple grid_,
    tuple block_,
    const float zoom,
    const int w2,
    const int h2
)


cdef downscale_surface_cupy(
    gpu_array,
    tuple grid_,
    tuple block_,
    const float zoom,
    const int w2,
    const int h2
)