# cython: binding=False, boundscheck=False, wraparound=False, nonecheck=False, cdivision=True,
# profile=False, initializedcheck=False, exceptval=False)
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

"""
# **Image Processing Library in Cython**

## **Purpose**
This Cython library is designed for **high-performance image processing and manipulation**. 
It provides a wide range of functions to transform, filter, and enhance images, leveraging Cython's 
ability to combine Python's ease of use with C-like performance. The library is optimized for
real-time or computationally intensive tasks, making it suitable for applications in game development,
computer vision, artistic effects, and scientific visualization.

---

## **Key Functionalities**

### **1. Color Manipulation**
- **Color Space Conversions**:
  - `bgr`, `brg`: Convert images between BGR and BRG color formats.
  - `grey`: Convert images to grayscale.
  - `sepia`: Apply a sepia tone effect.
  - `hsl_effect`, `hsv_effect`: Transform images using HSL (Hue, Saturation, Lightness) and 
        HSV (Hue, Saturation, Value) color spaces.
- **Brightness and Saturation**:
  - `brightness`: Adjust the brightness of an image.
  - `saturation`: Adjust the saturation of an image.
  - `brightness_exclude`, `brightness_bpf`: Apply brightness adjustments with exclusions 
        or based on specific thresholds.
- **Inversion**:
  - `invert`: Invert the colors of an image.

---

### **2. Visual Effects**
- **Filters**:
  - `median`: Apply median filtering for noise reduction.
  - `sobel`: Perform edge detection using the Sobel operator.
  - `bloom`: Add a bloom effect to highlight bright areas.
  - `posterize_surface`: Reduce the number of colors for a posterized effect.
- **Distortions**:
  - `wave`, `swirl`, `fisheye`: Apply wave, swirl, and fisheye distortions.
  - `horizontal_glitch`, `horizontal_sglitch`: Simulate horizontal glitch effects.
- **Artistic Effects**:
  - `painting`, `cartoon`: Apply artistic effects to mimic painting or cartoon styles.
  - `dithering`, `dithering_atkinson`: Apply dithering algorithms to reduce color 
        depth while preserving visual quality.

---

### **3. Advanced Image Processing**
- **Edge Detection and Sharpening**:
  - `sobel`, `sobel_fast`: Detect edges using the Sobel operator.
  - `sharpen`, `sharpen32`: Sharpen images to enhance details.
- **Blending and Compositing**:
  - `blend`, `alpha_blending`: Blend two images with adjustable opacity or blending modes.
- **Heatmap and Predator Vision**:
  - `heatmap`: Generate heatmap visualizations.
  - `predator_vision`: Simulate a predator-like vision effect with edge detection and color mapping.

---

### **4. Physical Simulations**
- **Ripple Effects**:
  - `ripple`, `ripple_seabed`: Simulate ripple effects on water surfaces.
- **Heat Convection**:
  - `heatconvection`: Simulate heat convection effects on images.
- **Plasma Effects**:
  - `plasma`, `plasma_config`: Generate dynamic plasma effects.

---

### **5. Utility Functions**
- **Image Transformations**:
  - `mirroring`: Mirror images horizontally or vertically.
  - `pixelation`: Pixelate images by reducing resolution.
  - `bilinear`: Perform bilinear interpolation for resizing images.
- **Color Mapping**:
  - `wavelength2rgb`, `custom_map`: Convert wavelengths to RGB colors or apply custom color mappings.
- **Performance-Oriented Functions**:
  - Many functions (e.g., `bgr_1d`, `grey_1d`) are optimized for 1D or 3D pixel arrays, 
        enabling efficient processing of large datasets.

---

### **6. Special Effects**
- **TV Scanlines**:
  - `tv_scan`: Simulate old TV scanline effects.
- **Blood Effects**:
  - `blood`: Apply blood-like effects to images.
- **Dirt and Lens Effects**:
  - `dirt_lens`: Simulate dirt or lens distortion effects.

---

### **7. Performance and Optimization**
- **Memory Efficiency**:
  - Functions like `bgr_1d`, `grey_1d`, and `invert1d` are designed to work with 1D or 
        3D arrays, ensuring efficient memory usage.
- **Parallel Processing**:
  - Many functions use `nogil` to release the Global Interpreter Lock (GIL), 
        enabling multi-threaded execution for faster processing.

---

## **Target Use Cases**
This library is ideal for:
- **Game Development**: Real-time image effects (e.g., distortions, filters, blending).
- **Computer Vision**: Preprocessing images for machine learning or analysis.
- **Artistic Applications**: Applying creative effects to images or videos.
- **Scientific Visualization**: Simulating physical phenomena (e.g., heat convection, ripples).

---

## **Summary**
The library is a **powerful and versatile toolkit** for image processing, 
combining **performance optimization** with a **wide range of visual effects and transformations**. 
It is well-suited for applications requiring real-time or high-performance image manipulation, 
such as games, simulations, and computer vision tasks.


"""



import warnings
cimport numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=ImportWarning)

cdef extern from 'Include/Shaderlib.c':

    packed struct yiq:
        float y;
        float i;
        float q;


    packed struct hsv:
        float h;
        float s;
        float v;

    packed struct hsl:
        float h
        float s
        float l

    packed struct rgb:
        float r
        float g
        float b

    packed struct rgb_color_int:
        int r;
        int g;
        int b;

    # struct extremum:
    #     int min;
    #     int max;

    hsl struct_rgb_to_hsl(const float r, const float g, const float b)nogil;
    rgb struct_hsl_to_rgb(const float h, const float s, const float l)nogil;

    rgb struct_hsv_to_rgb(const float h, const float s, const float v)nogil;
    hsv struct_rgb_to_hsv(const float r, const float g, const float b)nogil;

    yiq rgb_to_yiq(const float r, const float g, const float b)nogil;
    rgb yiq_to_rgb(const float y, const float i, const float q)nogil;


    unsigned char * new_quickSort(unsigned char arr[ ],  int low, int high)nogil;

    rgb_color_int wavelength_to_rgb(int wavelength, float gamma)nogil;
    rgb_color_int wavelength_to_rgb_custom(int wavelength, int arr[], float gamma)nogil;

    int randRange(int lower, int upper)nogil;





cdef float M_PI = 3.14159265358979323846
cdef float M_PI2 =3.14159265358979323846/2.0
cdef float M_2PI =2 * 3.14159265358979323846
cdef float RAD_TO_DEG=180.0/M_PI
cdef float DEG_TO_RAD=M_PI/180.0

cdef int THREADS = 8

cpdef void bgr(object surface_)
cpdef bgr_copy(object surface_)
cpdef void bgr_3d(unsigned char [:, :, :] rgb_array)
cpdef void bgr_1d(unsigned char [::1] rgb_array, bint format_32=*)

cpdef np.ndarray[np.uint8_t, ndim=1] bgr_1d_cp(
        unsigned char [::1] rgb_array, bint format_32=*)
cpdef void brg(object surface_)
cpdef brg_copy(object surface_)
cpdef void brg_3d(unsigned char [:, :, :] rgb_array)
cpdef void brg_1d(unsigned char [::1] rgb_array, bint format_32=*)
cpdef np.ndarray[np.uint8_t, ndim=1] brg_1d_cp(
        const unsigned char [::1] bgr_array, bint format_32=*)
cpdef void grey(object surface_)
cpdef grey_copy(object surface_)
cpdef np.ndarray[np.uint8_t, ndim=2] grey_2d(surface_)
cpdef void grey_3d(unsigned char [:, :, :] rgb_array)
cpdef void grey_1d(unsigned char [:] rgb_array, bint format_32=*)
cpdef np.ndarray[np.uint8_t, ndim=1]\
        grey_1d_cp(const unsigned char [::1] bgr_array, bint format_32=*)
cpdef void sepia(object surface_)
cpdef sepia_copy(object surface_)
cpdef void sepia_3d(unsigned char [:, :, : ] rgb_array)
cpdef void sepia_1d(unsigned char [:] rgb_array, bint format_32=*)
cdef np.ndarray[np.uint8_t, ndim=1, mode='c'] sepia_1d_cp(
        const unsigned char [::1] rgb_array, bint format_32=*)
cpdef void median(
        object surface_,
        unsigned short int kernel_size_=*,
        bint fast_=*,
        unsigned short int reduce_factor_=*
) except *
cpdef void painting(object surface_) except *
cpdef void pixels(object surface_) except *
cdef void median_fast(
        object surface_,
        unsigned short int kernel_size_=*,
        unsigned short int reduce_factor_=*
)
cpdef void median_grayscale(object surface_, int kernel_size_=*)
cpdef void posterize_surface(
        object surface_,
        int color_=*
)
cpdef void sobel(
        object surface_,
        int threshold_ = *
)
cpdef void sobel_1d(
        Py_ssize_t w,
        Py_ssize_t h,
        unsigned char [::1] bgr_array,
        tmp_array = *,
        int threshold = *,
        bint format_32 = *,
        bint greyscale = *
        )
cpdef void sobel_fast(
        object surface_,
        int threshold_ = *,
        unsigned short factor_ = *
)
cpdef void invert(object surface_)
cpdef invert_copy(object surface_)
cpdef void invert3d(unsigned char [:, :, :] rgb_array)
cpdef void invert1d(unsigned char [:] rgb_array, bint format_32=*)
cpdef np.ndarray[np.uint8_t, ndim=1] invert1d_cp(const unsigned char [:] rgb_array, bint format_32=*)
cpdef void hsl_effect(object surface_, const float shift)
cpdef void hsl3d(unsigned char [:, :, :] rgb_array, const float shift)
cpdef void hsl1d(unsigned char [::1] bgr_array, const float shift, bint format_32=*)
cpdef np.ndarray[np.uint8_t, ndim=1] hsl1d_cp(
    const unsigned char [::1] bgr_array, const float shift, bint format_32=*)
cpdef void hsv_effect(object surface_, const float shift)
cpdef void hsv3d(unsigned char [:, :, :] rgb_array, const float shift)
cpdef void hsv1d(unsigned char [::1] bgr_array, const float shift, bint format_32=*)
cpdef np.ndarray[np.uint8_t, ndim=1] hsv1d_cp(
        const unsigned char [::1] bgr_array, const float shift, bint format_32=*)
cpdef void wave(object surface_, const float rad, int size=*)
cpdef void wave32(object surface_, const float rad, int size=*)
cpdef void wave_static(object surface_, array_, const float rad, int size=*)
cpdef void swirl(object surface_, rgb_array_cp, float degrees)
cpdef void swirl32(object surface_, float degrees)

cpdef void swirlf(object surface_, float degrees)

cpdef void plasma_config(
        object surface_,
        int frame,
        float hue_=*,
        float sat_=*,
        float value_=*,
        float a_=*,
        float b_=*,
        float c_=*
)
cpdef void plasma(surface_, float frame, unsigned int [::1] palette_)
cpdef void brightness(object surface_, float shift_)
cpdef void brightness3d(unsigned char [:, :, :] rgb_array, float shift)
cpdef void brightness1d(unsigned char [:] bgr_array, const float shift, bint format_32=*)
cpdef np.ndarray[np.uint8_t, ndim=1] brightness1d_copy(
        unsigned char [:] bgr_array, const float shift, bint format_32=*)
cpdef object brightness_copy(object surface_, const float shift)
cpdef void brightness_exclude(
        object surface_,
        const float shift_,
        color_=*
)
cpdef void brightness_bpf(
        object surface_,
        const float shift_,
        unsigned char bpf_threshold = *)
cpdef void saturation(object surface_, float shift_)
cpdef void saturation3d(unsigned char [:, :, :] rgb_array, float shift)
cpdef void saturation1d(unsigned char [:] buffer, const float shift, bint format_32=*)
cpdef np.ndarray[np.uint8_t, ndim=1]  saturation1d_cp(
        const unsigned char [:] buffer,
        const float shift,
        bint format_32=*
)
cpdef void heatconvection(
        object surface_,
        float amplitude,
        float center = *,
        float sigma = *,
        float mu = *)
cpdef void horizontal_glitch(
        object surface_,
        const float deformation,
        const float frequency,
        const float amplitude
)
cpdef void horizontal_sglitch(
        object surface_,
        object array_,
        const float deformation,
        const float frequency,
        const float amplitude
)
cpdef void bpf(object surface_, int threshold = *)
cpdef void bloom(
        object surface_,
        int threshold_,
        bint fast_=*,
        object mask_=*
)

cpdef np.ndarray[np.uint32_t, ndim=2] fisheye_footprint(
        const int w,
        const int h,
        const unsigned int centre_x,
        const unsigned int centre_y
)
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
cpdef void fisheye(
        object surface_,
        unsigned int [:, :, ::1] fisheye_model
)
cpdef void tv_scan(surface_, int space=*)
cpdef tuple ripple(
        int rows_,
        int cols_,
        const float [:, ::1] previous_,
        float [:, ::1] current_,
        unsigned char [:, :, :] array_,
        float dispersion_ = *
)
cpdef tuple ripple_seabed(
    int cols_, int rows_,
    const float [:, ::1] previous_,                 # type numpy.float32 (w, h)
    float [:, ::1] current_,                        # type numpy.float32 (w, h)
    const unsigned char [:, :, ::1] texture_array_, # type numpy.ndarray (w, h, 3)
    unsigned char [:, :, :] background_array_,      # type numpy.ndarray (w, h, 3)
    float dispersion_ = *
)
cpdef void heatmap(object surface_, bint rgb_=*)
cpdef predator_vision(
        object surface_,
        unsigned int sobel_threshold = *,
        unsigned int bpf_threshold   = *,
        unsigned int bloom_threshold = *,
        bint inv_colormap            = *,
        bint fast                    = *,
        int blend                    = *
)
cpdef  void blood(
        object surface_,
        const float [:, :] mask_,
        float percentage_
)
cpdef  np.ndarray[np.uint8_t, ndim=3] mirroring_array(const unsigned char [:, :, :] rgb_array)
cpdef  void mirroring(object surface_)
cpdef  void sharpen(object surface_)
cpdef  void sharpen_1d(
        const Py_ssize_t w,
        const Py_ssize_t h,
        unsigned char [::1] bgr_array,
        bint format_32=*)
cdef  void sharpen_1d_c(
        const Py_ssize_t w,
        const Py_ssize_t l,
        unsigned char [::1] bgr_array,
        const unsigned char [::1] bgr_array_cp,
        bint format_32=*)nogil
cpdef  np.ndarray[np.uint8_t, ndim=1] sharpen_1d_cp(
        const Py_ssize_t w,
        const Py_ssize_t h,
        unsigned char [::1] bgr_array,
        bint format_32=*
)
cdef  unsigned char [::1] sharpen_1d_cp_c(
        const Py_ssize_t w,
        const Py_ssize_t l,
        unsigned char [::1] bgr_array,
        unsigned char [::1] bgr_array_cp,
        bint format_32=*)nogil
cpdef  void sharpen32(object surface_)
cpdef  void dirt_lens(
        object surface_,
        object lens_model_,
        int flag_=*,
        float light_ = *
)
cpdef  void end_game(object surface)
cpdef  void level_clear(object surface)
cpdef object dithering(object surface_)
cpdef  void dithering_inplace(object surface_)
cpdef  void dithering1d(
        Py_ssize_t w,
        Py_ssize_t h,
        unsigned char [::1] bgr_array,
        bint format_32=*
)
cdef  void dithering1d_c(
        const Py_ssize_t w,
        const Py_ssize_t h,
        const Py_ssize_t l,
        unsigned char [::1] bgr_array,
        float [::1] tmp_array,
        bint format_32=*
)nogil
cpdef  dithering1d_cp(
        Py_ssize_t w,
        Py_ssize_t h,
        rgb_array,
        bint format_32 = *
)
cdef dithering1d_cp_c(
        const Py_ssize_t w,
        const Py_ssize_t h,
        const Py_ssize_t l,
        float [::1] bgr_array,
        unsigned char [::1] tmp_array,
        bint format_32=*)
cpdef object dithering_atkinson(object surface_)
cpdef  void dithering_atkinson1d(
    Py_ssize_t w,
    Py_ssize_t h,
    unsigned char [::1] c_buffer,
    bint format_32=*
)
cdef  void dithering_atkinson1d_c(
        const Py_ssize_t w,
        const Py_ssize_t h,
        const Py_ssize_t l,
        unsigned char [::1] c_buffer,
        float [::1] tmp_buffer,
        bint format_32=*
)nogil
cpdef object pixelation(object surface_, unsigned int blocksize_=*)
cpdef object blend(object source, object destination, float percentage)
cpdef blend1d(
        const Py_ssize_t w,
        const Py_ssize_t h,
        const unsigned char [::1] source,
        const unsigned char [::1] destination,
        float percentage,
        modes,
        bint format_32 = *)

cdef blend1d_c(
        const Py_ssize_t w,
        const Py_ssize_t h,
        const Py_ssize_t l,
        const unsigned char[::1] source_array,
        const unsigned char[::1] destination_array,
        float percentage,
        modes,
        bint format_32 = *)



cpdef void blend_inplace(
        object destination,
        object source,
        float percentage
        )
cpdef cartoon(
        object surface_,
        unsigned int sobel_threshold = *,
        unsigned int median_kernel   = *,
        unsigned int color           = *,
        unsigned int flag            = *
)
cpdef  void convert_27(object surface_)
cpdef object bilateral(object image, const float sigma_s, const float sigma_i, unsigned int kernel_size = *)
cpdef object emboss(object surface_, unsigned short int flag_=*)
cpdef void emboss_inplace(object surface_, copy=*)
cpdef  void emboss1d(
        const Py_ssize_t w,
        const Py_ssize_t h,
        unsigned char [:] bgr_array,
        tmp_array = *,
        bint format_32 = *
)
cpdef object emboss_gray(object surface_)
cpdef object bilinear(
    object surface_,
    tuple size_,
    fx=*,
    fy=*
    )
cpdef tuple tunnel_modeling24(
        const int screen_width,
        const int screen_height,
        object surface_
)
cpdef tunnel_render24(
        int t,
        const int screen_width,
        const int screen_height,
        const int screen_w2,
        const int screen_h2,
        const int [::1] distances,
        const int [::1] angles,
        const int [::1] shades,
        unsigned char [::1] scr_data,
        unsigned char [::1] dest_array)
cpdef tuple tunnel_modeling32(
        const Py_ssize_t screen_width,
        const Py_ssize_t screen_height,
        object surface_
)
cpdef tunnel_render32(
        int t,
        const Py_ssize_t screen_width,
        const Py_ssize_t screen_height,
        const int screen_w2,
        const int screen_h2,
        const int [::1] distances,
        const int [::1] angles,
        const int [::1] shades,
        unsigned char [::1] scr_data,
        unsigned char [::1] dest_array)
cdef  float damped_oscillation(float t)nogil
cdef  float gauss(float x, float c, float sigma=*, float mu=*)nogil
cdef  void bgr_c(unsigned char [:, :, :] rgb_array)nogil
cdef  void bgr_1d_c(unsigned char [::1] rgb_array, bint format_32=*)nogil
cdef  np.ndarray[np.uint8_t, ndim=1] bgr_1d_cp_c(
        unsigned char [::1] rgb_array, bint format_32=*)
cdef  void brg_c(unsigned char [:, :, :] rgb_array)
cdef  void brg_1d_c(
        unsigned char [::1] bgr_array, bint format_32=*)
cdef  np.ndarray[np.uint8_t, ndim=1] brg_1d_cp_c(
        const unsigned char [::1] bgr_array, bint format_32=*)
cdef  void grey_c(unsigned char [:, :, :] rgb_array)
cdef np.ndarray[np.uint8_t, ndim=2] grey_2d_c(unsigned char [:, :, :] rgb_array)
cdef  void grey_1d_c(
        unsigned char [:] rgb_array, bint format_32=*)
cdef  np.ndarray[np.uint8_t, ndim=1] \
        grey_1d_cp_c(const unsigned char [::1] bgr_array,
                     bint format_32=*)
cdef  void sepia_c(unsigned char [:, :, :] rgb_array)
cdef void sepia_1d_c(unsigned char [:] rgb_array, bint format_32=*)
cdef unsigned char [::1] sepia_1d_cp_c(
        const unsigned char [::1] bgr_array,
        unsigned char [::1] destination_array,
        bint format_32=*)nogil
cdef  void median_inplace_c(
        unsigned char [:, :, :] rgb_array_, int kernel_size_=*)
cdef  void median_grayscale_inplace_c(
        unsigned char [:, :, :] rgb_array_, int kernel_size_=*)
cdef  void posterize_surface_c(
        unsigned char [:, :, :] rgb_array, int color_number)
cdef  void sobel_inplace_c(
        unsigned char [:, :, :] rgb_array,
        float threshold=*
)
cdef  void sobel_1d_c(
        const Py_ssize_t w,
        const Py_ssize_t l,
        unsigned char [::1] bgr_array,
        const unsigned char [::1] bgr_array_cp,
        float threshold=*,
        bint format_32 = *,
        bint greyscale = *)nogil
cdef  void sobel_fast_inplace_c(
        surface_, int threshold_=*, unsigned short factor_=*)
cdef  void invert_inplace_c(
        unsigned char [:, :, :] rgb_array)
cdef void invert3d_c(unsigned char [:, :, :] rgb_array)
cdef void invert1d_c(unsigned char [:] rgb_array, bint format_32=*)
cdef np.ndarray[np.uint8_t, ndim=1] invert1d_cp_c(
        const unsigned char [:] rgb_array,
        bint format_32=*)
cdef  void wave_inplace_c(
        unsigned char [:, :, :] rgb_array,
        float rad,
        int size = *)
cdef  void wave32_c(
        unsigned char [:, :, ::1] rgba_array,
        float rad,
        int size)
cdef  void wave_static_c(
        unsigned char [:, :, :] rgb_array,
        const unsigned char [:, :, :] rgb_array_cp,
        const float rad,
        const int size)
cdef  void swirl_c(
        const Py_ssize_t w,
        const Py_ssize_t h,
        unsigned char [:, :, :] rgb_array,
        const unsigned char [:, :, :] rgb_array_cp,
        float degrees
)nogil
cdef void swirl32_c(
        Py_ssize_t w,
        Py_ssize_t h,
        unsigned char [:, :, ::1] rgb_array_,
        const unsigned char [:, :, :] rgb,
        const float degrees
)

cdef  void swirlf_c(
    const Py_ssize_t w,
    const Py_ssize_t h,
    unsigned char [:, :, :] rgb_array,
    const unsigned char [::1, :, :] rgb,
    const float degrees
)nogil
cdef  void plasma_inplace_c(
        unsigned char [:, :, :] rgb_array_,
        int frame,
        float hue_=*,
        float sat_=*,
        float value_=*,
        float a_=*,
        float b_=*,
        float c_=*)
cdef  void plasma_c(surface_, float frame, unsigned int [::1] palette_)
cdef void hsl_c(unsigned char [:, :, :] rgb_array, const float shift)
cdef  void hsl1d_c(unsigned char [::1] bgr_array, const float shift, bint format_32=*)
cdef np.ndarray[np.uint8_t, ndim=1] hsl1d_cp_c(
        const unsigned char [::1] bgr_array,
        const float shift,
        bint format_32=*
)
cdef  void hsv3d_c(unsigned char [:, :, :] rgb_array, const float shift_)
cdef  void hsv1d_c(
        unsigned char [::1] bgr_array, const float shift, bint format_32=*)
cdef np.ndarray[np.uint8_t, ndim=1] hsv1d_cp_c(
        const unsigned char [::1] bgr_array,
        const float shift,
        bint format_32=*)
cdef  void brightness_c(
        unsigned char [:, :, :] rgb_array,
        float shift=*)
cdef  object brightness_copy_c(
        unsigned char [:, :, :] rgb_array, float shift=*)
cdef  void brightness1d_c(unsigned char [:] bgr_array, const float shift, bint format_32=*)nogil
cdef  np.ndarray[np.uint8_t, ndim=1] brightness1d_copy_c(
        unsigned char [:] bgr_array, const float shift, bint format_32=*)
cdef  void brightness_ex_c(
        unsigned char [:, :, :] rgb_array_, float shift_=*, color_=*)
cdef  void brightness_bpf_c(
        unsigned char [:, :, :] rgb_array_,
        float shift_=*,
        unsigned char bpf_threshold_=*)
cdef  void saturation_c(unsigned char [:, :, :] rgb_array_, const float shift_)
cdef  void saturation1d_c(
        unsigned char [:] buffer,
        const float shift,
        bint format_32=*
)nogil
cdef np.ndarray[np.uint8_t, ndim=1] saturation1d_cp_c(
        const unsigned char [:] buffer,
        const float shift,
        bint format_32=*
)
cdef  saturation_mask(
        const unsigned char [:, :, :] rgb_array,
        const float shift,
        const unsigned char [:, :] mask,
)
cdef  void saturation_mask_inplace(
        unsigned char [:, :, :] rgb_array,
        const float shift,
        const unsigned char [:, :] mask,
        int w, int h
)nogil
cdef  void heatconvection_inplace_c(
        unsigned char [:, :, :] rgb_array,
        float amplitude,
        float center=*,
        float sigma=*,
        float mu=*)
cdef  void horizontal_glitch_c(
        unsigned char [:, :, :] rgb_array,
        const float deformation,
        const float frequency,
        const float amplitude)
cdef  void horizontal_sglitch_c(
        unsigned char [:, :, :] bck_array,
        unsigned char [:, :, :] array,
        const float deformation,
        const float frequency,
        const float amplitude)
cdef  void bpf_inplace_c(
        unsigned char [:, :, :] rgb_array_, int w, int h, int threshold = *)nogil

cdef  void bpf_c(
        object surface_,
        int threshold = *)
cdef  bpf24_c(
        unsigned char [:, :, :] rgb_array,
        int threshold = *,
        )
cdef void filtering24_c(object surface_, mask_)
cdef void filtering_inplace_c(object surface_, mask_)
cdef void filtering1d_inplace_c(object surface_, mask_)
cdef  void bloom_c(
        surface_,
        int threshold_,
        bint fast_ = *,
        object mask_ = *
)
cpdef  object shader_bloom_fast(
        surface_,
        int threshold,
        bint fast = *,
        unsigned short int factor = *
)
cpdef void shader_bloom_fast1(
        object surface_,
        unsigned short int smooth_ = *,
        unsigned int threshold_ = *,
        unsigned short int flag_ = *,
        bint saturation_ = *,
        mask_ = *
)
cdef  np.ndarray[np.uint32_t, ndim=2] fisheye_footprint_c(
        const Py_ssize_t w,
        const Py_ssize_t h,
        const unsigned int centre_x,
        const unsigned int centre_y
)
cdef  void fisheye_inplace_c(
        unsigned char [:, :, :] rgb_array,
        const unsigned int [:, :, ::1] fisheye_model
)

cdef  void tv_scanline_c(
        unsigned char [:, :, :] rgb_array,
        int space)
cpdef object split_channels(
        object surface_,
        char offset_,
        array_ = *)
cdef object split_channels_c(
        object surface_,
        char offset_,
        array_ = *)

cpdef void split_channels_inplace(
        object surface_,
        char offset_,
        array_=*)
cdef void split_channels_inplace_c(
        object surface_,
        char offset_,
        array_=*)


cdef  tuple ripple_c(
       Py_ssize_t rows_, Py_ssize_t cols_,
       const float [:, ::1] previous_,
       float [:, ::1] current_,
       unsigned char [:, :, :] array_,
       float dispersion_ = *
       )

cdef  ripple_seabed_c(
           int rows_, int cols_,
           const float [:, ::1] previous_,                 # type numpy.float32 (w, h)
           float [:, ::1] current_,                        # type numpy.float32 (w, h)
           const unsigned char [:, :, ::1] texture_array_, # type numpy.ndarray (w, h, 3)
           unsigned char [:, :, :] background_array_,      # type numpy.ndarray (w, h, 3)
           float dispersion_ = *
           )
cpdef (int, int, int) wavelength2rgb(int wavelength, float gamma=*)

cpdef tuple custom_map(int wavelength, int [::1] color_array, float gamma=*)
cdef void heatmap_c(object surface_, bint rgb_=*)
cpdef (int, int, int) blue_map(int wavelength, float gamma=*)
cpdef  void bluescale(object surface_)
cpdef (int, int, int) red_map(int wavelength, float gamma=*)
cpdef  void redscale(object surface_)
cdef  void blood_inplace_c(
        Py_ssize_t w,
        Py_ssize_t h,
        unsigned char [:, :, :] rgb_array,
        const float [:, :] mask,
        float percentage)nogil
cdef  unsigned char [:, :, :] mirroring_c(
        Py_ssize_t w,
        Py_ssize_t h,
        const unsigned char[:, :, :] rgb_array,
        unsigned char [:, :, :] new_array
)nogil
cdef  void mirroring_inplace_c(
        const Py_ssize_t w,
        const Py_ssize_t h,
        unsigned char[:, :, :] rgb_array,
        const unsigned char [:, :, :] rgb_array_cp
)nogil

cpdef tuple dampening(
        object surface_,
        int frame,
        int display_width,
        int display_height,
        float amplitude = *,
        int duration    = *,
        float freq      = *)
cpdef  float lateral_dampening(
        int frame,
        float amplitude = *,
        int duration    = *,
        float freq      = *)
cdef  void sharpen_inplace_c(unsigned char [:, :, :] rgb_array)
cdef  void sharpen32_inplace_c(unsigned char [:, :, :] rgba_array_)
cdef cartoon_c(
        object surface_,
        unsigned int sobel_threshold = *,
        unsigned int median_kernel   = *,
        unsigned int color           = *,
        unsigned int flag            = *)

cdef object blend_c(
        unsigned char[:, :, :] source_array,
        unsigned char[:, :, :] destination_array,
        float percentage)
cdef  void blend_inplace_c(
        unsigned char[:, :, :] destination,
        const unsigned char[:, :, :] source,
        const float percentage
       )

cpdef object alpha_blending(source, destination)
cpdef void alpha_blending_inplace(object image1, object image2)

cdef  void dirt_lens_c(
        object source_,
        object lens_model_,
        int flag_,
        float light_=*)
cdef object dithering_c(float [:, :, :] rgb_array_)
cdef  void dithering_inplace_c(float [:, :, :] rgb_array_, unsigned char[:, :, :] tmp)

cdef object dithering_atkinson_c(float [:, :, :] rgb_array_)

cdef dithering1D_atkinson_c(
        const Py_ssize_t w,
        const Py_ssize_t h,
        unsigned char [::1] bgr_array,
        bint format_32 = *
)

cdef  void convert_27_c(
        const Py_ssize_t w,
        const Py_ssize_t h,
        unsigned char [:, :, :] rgb_array)nogil


cdef bilateral_c(
        const unsigned char [:, :, :] rgb_array_,
        const float sigma_s_,
        const float sigma_i_,
        unsigned int kernel_size = *
)
cdef object emboss3d_c(const unsigned char [:, :, :] rgb_array_)

cdef object emboss3d_gray_c(const unsigned char [:, :, :] rgb_array_)

cdef  void emboss3d_inplace_c(unsigned char [:, :, :] rgb_array, copy = *)
cdef  void emboss1d_c(
        const Py_ssize_t w,
        const Py_ssize_t l,
        unsigned char [:] bgr_array,
        const unsigned char [:] bgr_array_cp,
        bint format_32 = *)nogil
cdef object bilinear_c(
        unsigned char [:, :, :] rgb_array_,
        tuple size_,
        fx=*,
        fy=*
)

cpdef tuple render_light_effect24(
        int x,
        int y,
        np.ndarray[np.uint8_t, ndim=3] background_rgb,
        np.ndarray[np.uint8_t, ndim=2] mask_alpha,
        float intensity=*,
        float [:] color=*,
        bint smooth=*,
        bint saturation=*,
        float sat_value=*,
        bint bloom=*,
        unsigned int threshold=*,
        bint heat=*,
        float frequency=*)

cdef tuple bpf24_c2(image, int threshold = *, bint transpose=*)

cpdef  object bloom_effect_array24_c2(
        object surface_,
        const unsigned char threshold_,
        int smooth_=*,
        mask_=*,
        bint fast_ = *
)
cpdef area24_cc(
        int x, int y,
        np.ndarray[np.uint8_t, ndim=3] background_rgb,
        np.ndarray[np.uint8_t, ndim=2] mask_alpha, float intensity=*,
        float [::1] color=*,
        bint smooth=*, bint saturation=*, float sat_value=*, bint bloom=*,
        unsigned char bloom_threshold=*
)

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

cpdef object zoom(surface_, unsigned int delta_x, unsigned int delta_y, float zx=*)
cpdef void zoom_inplace(
        surface_,
        unsigned int delta_x,
        unsigned int delta_y,
        float zx=*
)

cpdef void Luma_GreyScale(object surface_)
cdef heatwave_array24_horiz_c(unsigned char [:, :, :] rgb_array,
                            unsigned char [:, :] mask_array,
                            float frequency, float amplitude, float attenuation=*,
                            unsigned char threshold=*)
cdef  void Luma_GreyScale_c(unsigned char [:, :, :] rgb_array)

