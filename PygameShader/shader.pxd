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

    struct yiq:
        float y;
        float i;
        float q;


    struct hsv:
        float h;
        float s;
        float v;

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

    rgb struct_hsv_to_rgb(float h, float s, float v)nogil;
    hsv struct_rgb_to_hsv(float r, float g, float b)nogil;

    yiq rgb_to_yiq(float r, float g, float b)nogil;
    rgb yiq_to_rgb(float y, float i, float q)nogil;

    int * quickSort(int arr[], int low, int high)nogil;
    float Q_inv_sqrt(float number)nogil;
    rgb_color_int wavelength_to_rgb(int wavelength, float gamma)nogil;
    rgb_color_int wavelength_to_rgb_custom(int wavelength, int arr[], float gamma)nogil;
    float randRangeFloat(float lower, float upper)nogil;
    int randRange(int lower, int upper)nogil;
    int get_largest(int arr[], int n)nogil;
    int min_c(int arr[], int n)nogil;
    float minf(float arr[ ], int n)nogil;




cdef float M_PI = 3.14159265358979323846
cdef float M_PI2 =3.14159265358979323846/2.0
cdef float M_2PI =2 * 3.14159265358979323846
cdef float RAD_TO_DEG=180.0/M_PI
cdef float DEG_TO_RAD=M_PI/180.0

cdef int THREADS = 8

cpdef tuple yiq_2_rgb(float y, float i, float q)
cpdef tuple rgb_2_yiq(unsigned char r, unsigned char g, unsigned char b)

cpdef void rgb_to_bgr(object surface_)
cpdef void rgb_to_brg(object surface_)
cpdef void greyscale(object surface_)
cpdef void sepia(object surface_)

cdef void median_fast(object surface_, unsigned short int kernel_size_=*,
                      unsigned short int reduce_factor_=*)
cpdef void median(
        object surface_,
        unsigned short int kernel_size_=*,
        bint fast_=*,
        unsigned short int reduce_factor_=*)except *

cpdef void median_grayscale(object surface_, int kernel_size_=*)

cpdef void color_reduction(object surface_, int color_=*)

cpdef void sobel(object surface_, int threshold_=*)
cpdef void sobel_fast(object surface_, int threshold_=*,
                                       unsigned short factor_=*)

cpdef void invert(object surface_)

cpdef void hsl_effect(object surface_, float shift_)
cpdef void hsl_fast(
        object surface_, float shift_, float [:, :, :, ::1] hsl_model_,
        unsigned char [:, :, :, ::1] rgb_model_)

cpdef void blur(object surface_, t_=*)
cpdef void wave(object surface_, float rad, int size)

cpdef void swirl(object surface_, float degrees)
cpdef void swirl2(object surface_, float degrees)

cpdef void plasma_config(object surface_, int frame,
                                      float hue_=*, float sat_=*, float value_=*,
                                      float a_=*, float b_=*, float c_=*)
cpdef void plasma(surface_, float frame, unsigned int [::1] palette_)
cpdef float [:, :, :, ::1] rgb_to_hsl_model()
cpdef unsigned char [:, :, :, ::1] hsl_to_rgb_model()

cpdef void brightness(object surface_, float shift_)
cpdef void brightness_exclude(object surface_, float shift_, color_=*)
cpdef void brightness_bpf(
        object surface_,
        float shift_,
        unsigned char bpf_threshold = *)
cpdef void brightness_model(object surface_, float shift_,
                                         float [:, :, :, :] rgb_to_hsl_model)

cpdef void saturation(object surface_, float shift_)
cpdef void heatwave_vertical(
        object surface_, unsigned char [:, :] mask, float factor_, float center_,
        float sigma_, float mu_)
cpdef void horizontal_glitch(object surface_, float rad1_, float frequency_,
                                              float amplitude_)

cpdef void bpf(object surface_, int threshold = *)

cpdef void bloom(object surface_, int threshold_, bint fast_=*, object mask_=*)
cpdef object shader_bloom_fast(surface_, int threshold_, bint fast_ = *, unsigned short int factor_ = *)
cpdef void shader_bloom_fast1(
        object surface_,
        unsigned short int smooth_= *,
        unsigned int threshold_   = *,
        unsigned short int flag_  = *,
        bint saturation_          = *
)

cpdef fisheye_footprint(int w, int h, unsigned int centre_x, unsigned int centre_y)
cpdef void fisheye(object surface_, unsigned int [:, :, :] fisheye_model)

cpdef tuple rain_footprint(int w, int h)
cpdef void rain_fisheye(object surface_,
                                         unsigned int [:, :, ::1] rain_fisheye_model)

cpdef void tv_scan(object surface_, int space=*)

cpdef void rgb_split(object surface_, int offset_=*)
cpdef object rgb_split_clean(object surface_, int offset_=*)

cpdef tuple ripple(int rows_, int cols_, float [:, ::1] previous_, float [:, ::1] current_,
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

cpdef void heatmap(object surface_, bint rgb_=*)

cpdef predator_vision(object surface_, unsigned int sobel_threshold=*,
                           unsigned int bpf_threshold=*, unsigned int bloom_threshold=*,
                           bint inv_colormap=*, bint fast=*, int blend=*)


cpdef blood(object surface_, float [:, :] mask_, float perc_)
cpdef object make_palette(int width, float fh, float fs, float fl)

cpdef void bluescale(object surface_)
cpdef void redscale(object surface_)

cpdef fire_sub(
        int width, int height, float factor, unsigned int [::1] palette, float [:, ::1] fire)

cpdef fire_effect(
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

cpdef cloud_effect(
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

cpdef mirroring_array(object surface_)
cpdef void mirroring(object surface_)

cpdef float lateral_dampening(int frame_, float amplitude_=*, int duration_=*,
                                     float freq_=*)
cpdef tuple dampening(
        object surface_, int frame_, int display_width, int display_height_,
        float amplitude_=*, int duration_=*, float freq_=*)

# Added version 1.0.2
cpdef object blend(object source_, object destination_, float percentage_)

# Added version 1.0.2
cpdef cartoon(
        object surface_,
        int sobel_threshold_ = *,
        int median_kernel_   = *,
        color_               = *,
        flag_                = *
)

cpdef void dirt_lens(
        object surface_,
        object lens_model_,
        int flag_=*,
        float light_ = *
)

cpdef object dithering(object surface_, int factor_=*)
cpdef void dithering_int(object surface_, int factor_=*)

cpdef object spectrum(int width, int height, float gamma=*)

cpdef void convert_27colors(object surface_)

cpdef void palette_change(
        object surface_,
        object palette_)

cpdef object bilateral(object image, float sigma_s, float sigma_i)

cpdef object emboss(object surface_, unsigned int flag=*)

cpdef object pixelation(object surface_)

cpdef object bilinear(object surface_, int new_width, int new_height, fx=*, fy=*)

cpdef object alpha_blending(object source_, object destination_)
cpdef void alpha_blending_inplace(object source_, object destination_)

cpdef float [:, :] array2d_normalized_c(unsigned char [:, :] array)
cpdef filtering24_c(object surface_, mask_)
cpdef void heatmap_convert(object surface_, bint rgb_=*)


cpdef object chromatic(
        surface_,
        unsigned int delta_x,
        unsigned int delta_y,
        float zoom=*,
        float fx=*
)

cpdef object zoom(
        surface_,
        unsigned int delta_x,
        unsigned int delta_y,
        float zx=*
)

cpdef void shader_rgb_to_yiq_inplace(object surface_)
cpdef void shader_rgb_to_yiq_inplace_c(unsigned char [:, :, :] rgb_array)
cpdef void shader_rgb_to_yiq_i_comp_inplace(object surface_)
cpdef void shader_rgb_to_yiq_q_comp_inplace(object surface_)

