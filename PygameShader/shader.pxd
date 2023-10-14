# cython: binding=False, boundscheck=False, wraparound=False, nonecheck=False, cdivision=True,
# profile=False, initializedcheck=False
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


import warnings
cimport numpy as np

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
    unsigned char * new_quickSort(unsigned char arr[ ], int low, int high)nogil;
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


cpdef float _randf(float lower, float upper)nogil
cpdef int _randi(int lower, int upper)nogil

cpdef rgb _hsl_to_rgb(float h, float s, float l)nogil
cpdef hsl _rgb_to_hsl(unsigned char r, unsigned char g, unsigned char b)nogil

cpdef rgb _hsv_to_rgb(float h, float s, float v)nogil
cpdef hsv _rgb_to_hsv(unsigned char r, unsigned char g, unsigned char b)nogil

cpdef tuple yiq_2_rgb(float y, float i, float q)
cpdef tuple rgb_2_yiq(unsigned char r, unsigned char g, unsigned char b)

cpdef void rgb_to_bgr(object surface_)
cdef void rgb_to_bgr_inplace_c(unsigned char [:, :, :] rgb_array)

cpdef void rgb_to_brg(object surface_)
cdef void rgb_to_brg_inplace_c(unsigned char [:, :, :] rgb_array)

cpdef void greyscale(object surface_)
cdef void greyscale_luminosity24_inplace_c(unsigned char [:, :, :] rgb_array)

cpdef void sepia(object surface_)
cdef void sepia_inplace_c(unsigned char [:, :, :] rgb_array)

cpdef void painting(object surface_) except *
cpdef void pixels(object surface_) except *

cpdef void median(
        object surface_,
        unsigned short int kernel_size_=*,
        bint fast_=*,
        unsigned short int reduce_factor_=*)except *

cdef void median_inplace_c(
        unsigned char [:, :, :] rgb_array_,
        int kernel_size_=*
)
cdef void median_fast(object surface_, unsigned short int kernel_size_=*,
                      unsigned short int reduce_factor_=*)

cpdef void median_grayscale(object surface_, int kernel_size_=*)

cpdef void color_reduction(object surface_, int color_=*)

cpdef void sobel(object surface_, int threshold_=*)
cdef void sobel_inplace_c(unsigned char [:, :, :] rgb_array, float threshold=*)

cpdef void sobel_fast(object surface_, int threshold_=*,
                                       unsigned short factor_=*)
cdef void sobel_fast_inplace_c(
        surface_, int threshold_=*, unsigned short factor_=*)

cpdef void invert(object surface_)
cdef void invert_inplace_c(
        unsigned char [:, :, :] rgb_array)

cpdef void hsl_effect(object surface_, float shift_)
cdef void hsl_inplace_c(unsigned char [:, :, :] rgb_array, float shift_)

cpdef void hsv_effect(object surface_, float shift_)
cdef void hsv_inplace_c(unsigned char [:, :, :] rgb_array, float shift_)


cpdef void hsl_fast(
        object surface_, float shift_, float [:, :, :, ::1] rgb_to_hsl_,
        unsigned char [:, :, :, ::1] hsl_to_rgb_)
cdef void hsl_fast_inplace_c(
        unsigned char [:, :, :] rgb_array,
        float shift_,
        float [:, :, :, ::1] rgb_to_hsl_,
        unsigned char [:, :, :, ::1] hsl_to_rgb_)

cpdef void blur(object surface_, t_=*)
cpdef void blur5x5_array24_inplace(rgb_array_, mask_=*, t_=*)
cdef void blur_array_inplace_c(
        unsigned char [:, :, :] rgb_array_, mask=*, t=*)

cpdef void wave(object surface_, float rad, int size=*)
cdef void wave_inplace_c(
        unsigned char [:, :, :] rgb_array_,
        float rad,
        int size)

cpdef void wave_static(object surface_, array_, float rad, int size=*)
cdef void wave_static_inplace_c(
        unsigned char [:, :, :] rgb_array_,
        unsigned char [:, :, :] array_,
        float rad,
        int size)

cpdef void swirl(object surface_, float degrees)
cdef void swirl_inplace_c(
        unsigned char [:, :, :] rgb_array_,
        float degrees
)

cpdef void swirl_inplace(object surface_, float degrees)
cdef void swirl_inplace_c1(unsigned char [:, :, :] rgb_array_, float degrees)


cpdef void swirl_static(object surface_, array_, float degrees)
cdef void swirl_static_inplace_c(
        unsigned char [:, :, :] rgb_array_,
        unsigned char [:, :, :] array_,
        float degrees
)

cpdef void swirl2_static(object surface_, array_, float degrees)
cdef void swirl2_static_inplace_c1(
        unsigned char [:, :, :] rgb_array_,
        unsigned char [:, :, :] array_,
        float degrees)

cpdef void plasma_config(object surface_, int frame,
                                      float hue_=*, float sat_=*, float value_=*,
                                      float a_=*, float b_=*, float c_=*)
cdef void plasma_inplace_c(
        unsigned char [:, :, :] rgb_array_,
        int frame,
        float hue_=*,
        float sat_=*,
        float value_=*,
        float a_=*,
        float b_=*,
        float c_=*)


cpdef void plasma(surface_, float frame, unsigned int [::1] palette_)
cdef void plasma_c(surface_, float frame, unsigned int [::1] palette_)


cpdef float [:, :, :, ::1] rgb_to_hsl_model()
cdef float [:, :, :, ::1] rgb_to_hsl_model_c()

cpdef unsigned char [:, :, :, ::1] hsl_to_rgb_model()
cdef unsigned char [:, :, :, ::1] hsl_to_rgb_model_c()

cpdef void brightness(object surface_, float shift_)
cdef void brightness_inplace_c(unsigned char [:, :, :] rgb_array_, float shift_=*)

cpdef brightness_(object surface_, float shift_)
cdef object brightness_c(
        unsigned char [:, :, :] rgb_array_, float shift_=*)

cpdef void brightness_exclude(object surface_, float shift_, color_=*)
cdef void brightness_exclude_inplace_c(
        unsigned char [:, :, :] rgb_array_, float shift_=*, color_=*)

cpdef void brightness_bpf(
        object surface_,
        float shift_,
        unsigned char bpf_threshold = *)
cdef void brightness_bpf_c(
        unsigned char [:, :, :] rgb_array_,
        float shift_=*,
        unsigned char bpf_threshold_=*)


cpdef void brightness_model(
        object surface_, float shift_,
        float [:, :, :, :] rgb_to_hsl_model)
cdef void brightness_inplace1_c(
        unsigned char [:, :, :] rgb_array_,
        float shift_, float [:, :, :, :] rgb_to_hsl_model)


cpdef void saturation(object surface_, float shift_)
cdef void saturation_inplace_c(
        unsigned char [:, :, :] rgb_array_, float shift_)


cdef saturation_array_mask(
        unsigned char [:, :, :] array_,
        float shift_, unsigned char [:, :] mask_array,
        bint swap_row_column
)
cdef void saturation_array_mask_inplace(
        unsigned char [:, :, :] array_,
        float shift_,
        unsigned char [:, :] mask_array,
        bint swap_row_column
)

cpdef void heatwave_vertical(
        object surface_, unsigned char [:, :] mask, float factor_, float center_,
        float sigma_, float mu_)
cdef void heatwave24_vertical_inplace_c(
        unsigned char [:, :, :] rgb_array_,
        unsigned char [:, :] mask,
        float amplitude_,
        float center_,
        float sigma_,
        float mu_)


cpdef void horizontal_glitch(object surface_, float rad1_, float frequency_, float amplitude_)
cdef void horizontal_glitch_inplace_c(
        unsigned char [:, :, :] rgb_array_,
        float rad1_,
        float frequency_,
        float amplitude_)


cpdef void horizontal_static_glitch(
        object surface_,
        object array_,
        float rad1_,
        float frequency_,
        float amplitude_
)
cdef void horizontal_glitch_static_inplace_c(
        unsigned char [:, :, :] rgb_array_,
        unsigned char [:, :, :] array_,
        float rad1_,
        float frequency_,
        float amplitude_)


cpdef void bpf(object surface_, int threshold = *)
cdef void bpf24_inplace_c(
        unsigned char [:, :, :] rgb_array_, int threshold = *)
cdef bpf24_c(
        unsigned char [:, :, :] input_array_,
        int threshold = *
        )

cpdef void bloom(object surface_, int threshold_, bint fast_=*, object mask_=*)
cdef void bloom_array24_c(
        surface_,
        int threshold_,
        bint fast_ = *,
        object mask_ = *
)


cpdef void bloom_array24(surface_, threshold_, fast_, mask_)
cpdef object shader_bloom_fast(surface_, int threshold_,
                               bint fast_ = *, unsigned short int factor_ = *)
cpdef void shader_bloom_fast1(
        object surface_,
        unsigned short int smooth_= *,
        unsigned int threshold_   = *,
        unsigned short int flag_  = *,
        bint saturation_          = *
)

cpdef fisheye_footprint(
    int w,
    int h,
    unsigned int centre_x,
    unsigned int centre_y
    )
cpdef void fisheye(object surface_, unsigned int [:, :, ::1] fisheye_model)
cdef void fisheye_inplace_c(
        unsigned char [:, :, :] rgb_array_, unsigned int [:, :, ::1] fisheye_model)

cpdef void fisheye_footprint_param(
        tmp_array_,
        float centre_x,
        float centre_y,
        float param1_,
        float focal_length,
        )
cdef void fisheye_footprint_param_c(
        unsigned int [:, :, :] tmp_array_,
        float centre_x,
        float centre_y,
        float param1,
        float focal_length,
)

cpdef void tv_scan(object surface_, int space=*)
cdef void tv_scanline_inplace_c(
        unsigned char [:, :, :] rgb_array_,
        int frame_)

cpdef void rgb_split(object surface_, int offset_=*)
cdef void rgb_split_inplace_c(object surface_, int offset_)


cpdef object rgb_split_clean(object surface_, int offset_=*)
cdef rgb_split_c(object surface_, int offset_)


cpdef tuple ripple(
    int rows_, int cols_,
    float [:, ::1] previous_,
    float [:, ::1] current_,
    unsigned char [:, :, :] array_,
    float dispersion_ = *)
cdef tuple ripple_c(
       Py_ssize_t rows_, Py_ssize_t cols_,
       float [:, ::1] previous,
       float [:, ::1] current,
       unsigned char [:, :, :] array,
       float dispersion_ = *
       )


cpdef tuple ripple_seabed(
    int cols_, int rows_,
    float [:, ::1] previous_,
    float [:, ::1] current_,
    unsigned char [:, :, ::1] texture_array_,
    unsigned char [:, :, :] background_array_,
    float dispersion_ = *
)
cdef ripple_seabed_c(
           int cols_, int rows_,
           float [:, ::1] previous,
           float [:, ::1] current,
           unsigned char [:, :, ::1] texture_array,
           unsigned char [:, :, :] background_array,
           float dispersion_ = *
           )



cpdef void heatmap(object surface_, bint rgb_=*)
cpdef void heatmap_convert(object surface_, bint rgb_=*)


cpdef predator_vision(object surface_, unsigned int sobel_threshold=*,
                           unsigned int bpf_threshold=*, unsigned int bloom_threshold=*,
                           bint inv_colormap=*, bint fast=*, int blend=*)

cpdef void blood(object surface_, float [:, :] mask_, float perc_)
cdef void blood_inplace_c(
        unsigned char [:, :, :] rgb_array_, float [:, :] mask_, float perc_)


cpdef object make_palette(int width, float fh, float fs, float fl)
cdef unsigned int [::1] make_palette_c(int width, float fh, float fs, float fl)
cpdef object palette_to_surface(unsigned int [::1] palette_c)


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

cdef fire_effect_c(
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
cdef mirroring_c(unsigned char[:, :, :] rgb_array_)

cpdef void mirroring(object surface_)
cdef mirroring_inplace_c(unsigned char[:, :, :] rgb_array_)

cpdef void sharpen(object surface_)
cdef void sharpen_inplace_c(unsigned char [:, :, :] rgb_array_)

cpdef void dirt_lens(
        object surface_,
        object lens_model_,
        int flag_=*,
        float light_ = *
)
cdef void dirt_lens_c(
        object source_,
        object lens_model_,
        int flag_,
        float light_=*)


cpdef object dithering(object surface_)
cdef object dithering_c(float [:, :, :] rgb_array_)

cpdef void dithering_inplace(object surface_)
cdef void dithering_inplace_c(float [:, :, :] rgb_array_, unsigned char[:, :, :] tmp)

cpdef object dithering_atkinson(object surface_)
cdef object dithering_atkinson_c(float [:, :, :] rgb_array_)

cpdef object pixelation(object surface_)


cpdef cartoon(
        object surface_,
        unsigned int sobel_threshold_ = *,
        unsigned int median_kernel_   = *,
        unsigned int color_           = *,
        unsigned int flag_            = *
)
cdef cartoon_c(
        object surface_,
        unsigned int sobel_threshold_,
        unsigned int median_kernel_,
        unsigned int color_,
        unsigned int flag_)


cpdef object spectrum(int width, int height, float gamma=*)
cdef object spectrum_c(int width, int height, float gamma=*)

cpdef void convert_27colors(object surface_)
cdef void convert_27colors_c(
        unsigned char [:, :, :] rgb_array)

cdef float distance_ (float x, float y)nogil
cdef float gaussian_ (float v, float sigma2)nogil
cpdef object bilateral(object image, float sigma_s, float sigma_i)
cdef bilateral_c(
        unsigned char [:, :, :] rgb_array_,
        float sigma_s_,
        float sigma_i_,
        unsigned int kernel_size = *
)


cpdef object emboss(object surface_, unsigned short int flag_=*)
cdef object emboss5x5_c(unsigned char [:, :, :] rgb_array_)



cpdef void palette_change(
        object surface_,
        object palette_,
        object tmp_v
)
cdef void palette_change_c(
        unsigned char [:, :, :] rgb_array,
        float [:, :] palette_,
        float [:, ::1] tmp_v_
)


cpdef object bilinear(
    object surface_,
    tuple size_,
    fx=*,
    fy=*
    )

cdef object bilinear_c(
        unsigned char [:, :, :] rgb_array_,
        tuple size_,
        fx=*, fy=*)



cpdef tunnel_modeling24(int screen_width, int screen_height, object surface_)
cpdef tunnel_render24(int t,
                    int screen_width,
                    int screen_height,
                    int screen_w2,
                    int screen_h2,
                    int [::1] distances,
                    int [::1] angles,
                    int [::1] shades,
                    unsigned char [::1] scr_data,
                    unsigned char [::1] dest_array)

cpdef tunnel_modeling32(Py_ssize_t screen_width, Py_ssize_t screen_height, object surface_)
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


cpdef unsigned char[::1] stack_buffer_c(rgb_array_, alpha_, int w, int h, bint transpose=*)

cpdef float [:, :] array2d_normalized_c(unsigned char [:, :] array)

cpdef tuple area24_c(int x, int y, np.ndarray[np.uint8_t, ndim=3] background_rgb,
              np.ndarray[np.uint8_t, ndim=2] mask_alpha, float intensity=*,
              float [:] color=*,
              bint smooth=*, bint saturation=*, float sat_value=*, bint bloom=*,
              unsigned int threshold=*, bint heat=*, float frequency=*)

cpdef object bpf24_c2(image, int threshold = *, bint transpose=*)


cpdef object bloom_effect_array24_c2(
        object surface_, unsigned char threshold_, int smooth_=*, mask_=*, bint fast_ = *)

cpdef area24_cc(int x, int y, np.ndarray[np.uint8_t, ndim=3] background_rgb,
              np.ndarray[np.uint8_t, ndim=2] mask_alpha, float intensity=*,
              float [::1] color=*,
              bint smooth=*, bint saturation=*, float sat_value=*, bint bloom=*,
              unsigned char bloom_threshold=*)


cpdef object chromatic(
        surface_,
        unsigned int delta_x,
        unsigned int delta_y,
        float zoom=*,
        float fx=*
)

cpdef object chromatic_inplace(
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

cpdef void Luma_GreyScale(object surface_)
cdef void Luma_GreyScale_c(unsigned char [:, :, :] rgb_array)

cpdef void RGB_TO_YIQ_Q0_inplace(object surface_)
cdef void RGB_TO_YIQ_Q0_inplace_c(unsigned char [:, :, :] rgb_array)

cpdef void RGB_TO_YIQ_I0_inplace(object surface_)
cdef void RGB_TO_YIQ_I0_inplace_c(unsigned char [:, :, :] rgb_array)

cpdef void RGB_TO_YIQ_Y0_inplace(object surface_)
cdef void RGB_TO_YIQ_Y0_inplace_c(unsigned char [:, :, :] rgb_array)

cpdef void bluescale(object surface_)
cpdef void redscale(object surface_)

cpdef tuple blue_map(int wavelength, float gamma=*)
cpdef tuple red_map(int wavelength, float gamma=*)


cpdef float lateral_dampening(int frame_, float amplitude_=*, int duration_=*,
                                     float freq_=*)
cpdef tuple dampening(
        object surface_, int frame_, int display_width, int display_height_,
        float amplitude_=*, int duration_=*, float freq_=*)


cpdef object blend(object source_, object destination_, float percentage_)
cdef object blending(object source_, unsigned char[:, :, :] destination_, float percentage_)



cpdef void blend_inplace(
        object source_,
        object destination_,
        float percentage_
        )
cdef void blend_inplace_c(
        object source_,
        unsigned char[:, :, :] destination_,
        float percentage_
       )

# todo review the method below for improvement
cpdef object alpha_blending(object source_, object destination_)
cpdef void alpha_blending_inplace(object source_, object destination_)


cpdef filtering24_c(object surface_, mask_)
cpdef void filtering_inplace_c(object surface_, mask_)


cdef heatwave_array24_horiz_c(unsigned char [:, :, :] rgb_array,
                            unsigned char [:, :] mask_array,
                            float frequency, float amplitude, float attenuation=*,
                            unsigned char threshold=*)


cdef unsigned char [:, :, ::1] blur5x5_array24_c2(
        unsigned char [:, :, :] rgb_array_, mask=*)