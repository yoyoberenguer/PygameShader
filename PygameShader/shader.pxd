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
#
# BUILD THIS PROJECT WITHOUT OPENMP
# setup(
#     name='COBRA',
#     ext_modules=cythonize(Extension(
#             "*", ['*.pyx'], extra_compile_args=["/Qpar", "/fp:fast", "/O2", "/Oy", "/Ot"],
#             language="c",
#         )),    include_dirs=[numpy.get_include()])
#
# BUILD THIS PROJECT WITH OPENMP
# setup(
#     name='COBRA',
#     ext_modules=cythonize(Extension(
#             "*", ['*.pyx'], extra_compile_args=["/openmp", "/Qpar", "/fp:fast", "/O2", "/Oy",
#             "/Ot"], language="c",
#         )),    include_dirs=[numpy.get_include()])



import warnings


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=ImportWarning)

cdef extern from 'Include/Shaderlib.c':

    struct hsl:
        float h
        float s
        float l

    struct rgb:
        float r
        float g
        float b

    struct rgb_color_int:
        int r;
        int g;
        int b;

    struct extremum:
        int min;
        int max;

    hsl struct_rgb_to_hsl(float r, float g, float b)nogil;
    rgb struct_hsl_to_rgb(float h, float s, float l)nogil;
    int *quickSort(int arr[], int low, int high)nogil;
    float Q_inv_sqrt(float number)nogil;
    rgb_color_int wavelength_to_rgb(int wavelength, float gamma)nogil;
    rgb_color_int wavelength_to_rgb_custom(int wavelength, int arr[], float gamma)nogil;
    float perlin(float x, float y)nogil;
    float randRangeFloat(float lower, float upper)nogil;
    int randRange(int lower, int upper)nogil;
    int get_largest(int arr[], int n)nogil;
    int get_lowest(int arr[], int n)nogil;
    int get_max_and_min(int arr[], int n)nogil;



cdef float M_PI = 3.14159265358979323846
cdef float M_PI2 =3.14159265358979323846/2.0
cdef float M_2PI =2 * 3.14159265358979323846
cdef float RAD_TO_DEG=180.0/M_PI
cdef float DEG_TO_RAD=M_PI/180.0

cdef int THREADS = 6

cpdef void shader_rgb_to_bgr_inplace(object surface_)
cpdef void shader_rgb_to_brg_inplace(object surface_)
cpdef void shader_greyscale_luminosity24_inplace(object surface_)
cpdef void shader_sepia24_inplace(object surface_)

cdef void median_fast(object surface_, unsigned short int kernel_size_=*,
                      unsigned short int reduce_factor_=*)
cpdef void shader_median_filter24_inplace(
        object surface_,
        unsigned short int kernel_size_=*,
        bint fast_=*,
        unsigned short int reduce_factor_=*)except *
cpdef void shader_median_grayscale_filter24_inplace(object surface_, int kernel_size_=*)
cpdef void shader_median_filter24_avg_inplace(object surface_, int kernel_size_=*)

cpdef void shader_color_reduction24_inplace(object surface_, int color_=*)

cpdef void shader_sobel24_inplace(object surface_, int threshold_=*)
cpdef void shader_sobel24_fast_inplace(object surface_, int threshold_=*,
                                       unsigned short factor_=*)

cpdef void shader_invert_surface_24bit_inplace(object surface_)

cpdef void shader_hsl_surface24bit_inplace(object surface_, float shift_)
cpdef void shader_hsl_surface24bit_fast_inplace(
        object surface_, float shift_, float [:, :, :, ::1] hsl_model_,
        unsigned char [:, :, :, ::1] rgb_model_)

cpdef void shader_blur5x5_array24_inplace(object surface_)
cpdef void shader_wave24bit_inplace(object surface_, float rad, int size)

cpdef void shader_swirl24bit_inplace(object surface_, float degrees)
cpdef void shader_swirl24bit_inplace1(object surface_, float degrees)

cpdef void shader_plasma24bit_inplace(object surface_, int frame,
                                      float hue_=*, float sat_=*, float value_=*,
                                      float a_=*, float b_=*, float c_=*)
cpdef void shader_plasma(surface_, float frame, unsigned int [::1] palette_)
cpdef float [:, :, :, ::1] rgb_to_hsl_model()
cpdef unsigned char [:, :, :, ::1] hsl_to_rgb_model()

cpdef void shader_brightness24_inplace(object surface_, float shift_)
cpdef void shader_brightness24_exclude_inplace(object surface_, float shift_, color_=*)
cpdef void shader_brightness_24_inplace1(object surface_, float shift_,
                                         float [:, :, :, :] rgb_to_hsl_model)

cpdef void shader_saturation_array24_inplace(object surface_, float shift_)
cpdef void shader_heatwave24_vertical_inplace(
        object surface_, unsigned char [:, :] mask, float factor_, float center_,
        float sigma_, float mu_)
cpdef void shader_horizontal_glitch24_inplace(object surface_, float rad1_, float frequency_,
                                              float amplitude_)
cpdef void shader_bpf24_inplace(object surface_, int threshold = *)
cpdef void shader_bloom_effect_array24(object surface_, int threshold_, bint fast_=*)

cpdef shader_fisheye24_footprint_inplace(int w, int h)
cpdef void shader_fisheye24_inplace(object surface_, unsigned int [:, :, :] fisheye_model)

cpdef tuple shader_rain_footprint_inplace(int w, int h)
cpdef void shader_rain_fisheye24_inplace(object surface_,
                                         unsigned int [:, :, ::1] rain_fisheye_model)

cpdef void shader_tv_scanline_inplace(object surface_, int space=*)
cpdef void shader_rgb_split_inplace(object surface_, int offset_=*)
cpdef tuple shader_ripple(int rows_, int cols_, float [:, ::1] previous_, float [:, ::1] current_,
           unsigned char [:, :, ::1] array_)

cpdef tunnel_modeling32(Py_ssize_t screen_width, Py_ssize_t screen_height)
cpdef tunnel_render32(int t,
                    Py_ssize_t screen_width,
                    Py_ssize_t screen_height,
                    int screen_w2,
                    int screen_h2,
                    int [::1] distances,
                    int [::1] angles,
                    int [::1] shades,
                    unsigned char [::1] scr_data,
                    unsigned char [::1] dest_array)

cpdef void heatmap_surface24_conv_inplace(object surface_, bint rgb_=*)

cpdef predator_vision_mode(object surface_, unsigned int sobel_threshold=*,
                           unsigned int bpf_threshold=*, unsigned int bloom_threshold=*,
                           inv_colormap=*, fast=*)
cpdef shader_blood_inplace(object surface_, float [:, :] mask_, float perc_)
cpdef object make_palette(int width, float fh, float fs, float fl)

cpdef void bluemap_surface24_inplace_c(object surface_)
cpdef void redmap_surface24_inplace_c(object surface_)

cpdef shader_fire_surface24(
        int width, int height, float factor, unsigned int [::1] palette, float [:, ::1] fire)

cpdef shader_fire_effect(
        int width_,
        int height_,
        float factor_,
        unsigned int [::1] palette_,
        float [:, ::1] fire_,
        unsigned short int reduce_factor_ = *,
        unsigned short int fire_intensity_= *,
        bint smooth_                      = *,
        bint bloom_                       = *,
        bint fast_bloom_                  = *,
        unsigned char bpf_threshold_      = *,
        unsigned int low_                 = *,
        unsigned int high_                = *,
        bint brightness_                  = *,
        float brightness_intensity_       = *,
        object surface_                   = *,
        bint adjust_palette_              = *,
        tuple hsl_                        = *,
        bint transpose_                   = *,
        bint border_                      = *,
        bint blur_                        = *
        )

cpdef shader_cloud_effect(
        int width_,
        int height_,
        float factor_,
        unsigned int [::1] palette_,
        float [:, ::1] cloud_,
        unsigned short int reduce_factor_ = *,
        unsigned short int cloud_intensity_= *,
        bint smooth_                      = *,
        bint bloom_                       = *,
        bint fast_bloom_                  = *,
        unsigned char bpf_threshold_      = *,
        unsigned int low_                 = *,
        unsigned int high_                = *,
        bint brightness_                  = *,
        float brightness_intensity_       = *,
        object surface_                   = *,
        bint transpose_                   = *,
        bint blur_                        = *
        )

cpdef mirroring(object surface_)
cpdef void mirroring_inplace(object surface_)

cpdef float lateral_dampening_effect(int frame_, float amplitude_=*, int duration_=*,
                                     float freq_=*)
cpdef tuple dampening_effect(
        object surface_, int frame_, int display_width, int display_height_,
        float amplitude_=*, int duration_=*, float freq_=*)

# Added version 1.0.1
cpdef cartoon(
        object surface_,
        int sobel_threshold_ = *,
        int median_kernel_   = *,
        color_               = *,
        flag_                = *
)