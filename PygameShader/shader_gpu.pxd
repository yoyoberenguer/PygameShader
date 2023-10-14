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

cpdef object invert_gpu(surface_)
cpdef void invert_inplace_cupy(cpu_array_)

cpdef object sepia_gpu(object surface_)
cdef object sepia_cupy(gpu_array_)
cpdef void sepia_inplace_cupy(cpu_array_)


cpdef object bpf_gpu(object surface_, unsigned int threshold_ = *)
cdef object bpf_cupy(gpu_array_, unsigned int threshold_)
cpdef object bpf1_gpu(object surface_, grid_, block_, unsigned int threshold_ = *)
cdef object bpf1_cupy(object gpu_array_, unsigned int threshold_, object grid_, object block_)


cpdef object grayscale_gpu(object surface_)
cdef object grayscale_cupy(gpu_array_)

cpdef object grayscale_lum_gpu(object surface_)
cdef object grayscale__lum_cupy(gpu_array_)

cpdef object median_gpu(object surface_, unsigned int size_ = *)
cdef object median_cupy(gpu_array_, unsigned int size_ = *)

cpdef object median1_gpu(object surface_, unsigned int size_ = *)
cdef object median1_cupy(gpu_array_, unsigned int size_ = *)

cpdef object gaussian_5x5_gpu(object surface_)
cdef object gaussian_5x5_cupy(gpu_array_)

cpdef object gaussian_3x3_gpu(object surface_)
cdef object gaussian_3x3_cupy(gpu_array_)


cpdef object sobel_gpu(object surface_)
cdef object sobel_cupy(gpu_array_)

cpdef object prewitt_gpu(object surface_)
cdef object prewitt_cupy(gpu_array_)

cpdef object canny_gpu(object surface_)
cdef object canny_cupy(gpu_array_)

cpdef object color_reduction_gpu(object surface_, int color_number = *)
cdef object color_reduction_cupy(object gpu_array_, int color_number)


cpdef object hsv_gpu(object surface_, float val_, object grid_ = *, object block_ = *)
cdef object hsv_cupy(object cupy_array, object grid_, object block_, float val_, w, h)

cpdef object mult_downscale_gpu(object gpu_array)

# cpdef object zoom_in_gpu(object surface, int w, int h)
# cpdef object upscale_gpu(object gpu_array_, int w, int h)

# ------------------ BLOOM ------------------
cdef void bpf_c(object gpu_array_, int w, int h, unsigned int threshold_=*)
cdef gaussian_3x3_c(gpu_array_, int w, int h)
cdef gaussian_5x5_c(gpu_array_, int w, int h)
cpdef object upscale_c(object gpu_array_, int new_width, int new_height, int order_=*)

cpdef object bloom_gpu(object surface_,
        unsigned int threshold_ = *,
        bint fast_ = *,
        int flag_ = *,
        unsigned short int factor_ = *
        )

cpdef object bloom_array(
        object gpu_array_,
        unsigned int threshold_ = *,
        bint fast_ = *,
        int flag_  = *,
        mask_ = *
)
# ----------------------------------------------

cpdef object cartoon_gpu(
        object surface_,
        int sobel_threshold_ = *,
        int median_kernel_   = *,
        unsigned char color_ = *,
        bint contour_        = *,
        unsigned char flag_  = *
)
cdef object canny_cupy_c(gpu_array_, int w, int h)
cdef object sobel_cupy_c(gpu_array_, int w, int h)
cdef object median_cupy_c(gpu_array_, int w, int h, unsigned int size_=*)
cdef object color_reduction_cupy_c(
        object gpu_array_,
        int color_number,
        int w, int h
)
cdef object cartoon_cupy(
        object surface_,
        int sobel_threshold_,
        int median_kernel_,
        int color_,
        bint contour_,
        int flag_)


cpdef object blending_gpu(object source_, object destination_, float percentage_)

cpdef object sharpen_gpu(object surface_)
cpdef object sharpen1_gpu(object surfaace_, grid_, block_)


cpdef ripple_effect_gpu(
       object grid,
       object block,
       int w, int h,
       previous,
       current,
       texture_array,
       background_array
       )

cpdef object mirroring_gpu(
        object surface_,
        object grid_,
        object block_,
        bint format_ = *
)
cdef mirroring_cupy(object gpu_array_, object grid_, object block_, bint format_=*)


cpdef object saturation_gpu(
        object surface_,
        object grid_,
        object block_,
        float val_ = *
)
cdef object saturation_cupy(
        object cupy_array,
        object grid_,
        object block_,
        float val_ = *
)


cpdef object bilateral_gpu(surface_, unsigned int kernel_size_)
cdef bilateral_cupy(gpu_array_, unsigned int kernel_size_)


cpdef object emboss5x5_gpu(surface_)
cdef object emboss5x5_cupy(gpu_array_)

cpdef area24_gpu(int x, int y,
                 object background_rgb,
                 object mask_alpha,
                 float intensity=*,
                 color=*)

cpdef object brightness_gpu(
        object surface_,
        float val_,
        object grid_  = *,
        object block_ = *
)

cpdef object hsl_gpu(
        object surface_,
        float val_,
        object grid_  = *,
        object block_ = *
)

cpdef object fisheye_gpu(object surface_, float focal, float focal_texture, object grid_, object block_)

cpdef object swirl_gpu(
        object surface_,
        float rad,
        object grid_,
        object block_,
        unsigned int centre_x,
        unsigned int centre_y
)

cpdef object wave_gpu(object surface_, float rad_, int size_, object grid_, object block_)


cpdef object rgb_split_gpu(
        object surface_,
        float delta_x,
        float delta_y,
        object grid_,
        object block_
)

cpdef object chromatic_gpu(
        object surface_,
        unsigned int delta_x,
        unsigned int delta_y,
        object grid_,
        object block_,
        float zoom = *,
        float fx = *
)


cpdef object zoom_gpu(
        object surface_,
        unsigned int delta_x,
        unsigned int delta_y,
        object grid_,
        object block_,
        float zoom = *
)

cpdef object wavelength_map_gpu(
        object surface_,
        object grid_,
        object block_,
        unsigned short int layer_=*
)
cpdef object heatmap_gpu(
        object surface_,
        object grid_,
        object block_,
        bint invert_ = *
)
