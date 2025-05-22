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

"""

This library appears to be focused on **image blending operations**,
particularly for **BGRA/BGR and alpha channel manipulations**.
The functions suggest that it is optimized for **fast, parallelized image processing**
using **Cython** and **OpenMP**.

**Key Features & Purpose**
--------------------------

1. **Image Blitting (``blit_s``)**

   - Handles copying an image (``image``) onto a destination surface (``destination``)
     at a specified position.
   - Likely a wrapper for efficient low-level blitting.

2. **Blending Operations**

   - Functions with names like ``blend_add_*``, ``blend_sub_*``, ``blend_min_*``,
     and ``blend_max_*`` indicate support for **various blending modes**:

     - **Additive blending (``blend_add_*``)** → Increases pixel brightness.
     - **Subtractive blending (``blend_sub_*``)** → Decreases pixel brightness.
     - **Minimum blending (``blend_min_*``)** → Keeps the darkest pixels.
     - **Maximum blending (``blend_max_*``)** → Keeps the brightest pixels.

3. **Support for Different Data Representations**

   - **Surface-based functions (``blend_*_surface``)**: Operate on whole images
     (possibly a wrapper for SDL surfaces or similar structures).
   - **Array-based functions (``blend_*_array``)**: Operate on NumPy-style 3D arrays
     (``[height, width, channels]``).
   - **In-place operations (``blend_*_bgra_inplace``)**: Modify a 1D BGRA/BGR array
     directly, avoiding extra memory allocation.

4. **Alpha Channel Operations**

   - Functions like ``blend_add_alpha``, ``blend_max_alpha``, and ``blend_min_alpha``
     specifically target **alpha blending**, which is crucial for transparency effects.

5. **Parallelization & Optimization**

   - Use of **Cython** (``cdef`` functions) and **parallelization** (``prange`` from OpenMP)
     suggests that the library is designed for **high-performance image processing**,
     likely intended for real-time applications.

**Possible Use Cases**
----------------------

- **Game development & rendering** (e.g., blending sprites, effects)
- **Image processing applications** (e.g., compositing, transparency adjustments)
- **Graphics frameworks** needing fast pixel operations (e.g., custom shaders, filters)


"""



import warnings
cimport numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=ImportWarning)

cpdef void blit_s(destination, image, tuple position=*)

# EQU TO BLEND_RGB_ADD
cpdef void blend_add_surface(image1, image2)
cdef void blend_add_surface_c(image1, image2)


cpdef void blend_add_array(
    unsigned char [:, :, :] rgb_array1,
    const unsigned char [:, :, :] rgb_array2)
cdef void blend_add_array_c(
    unsigned char [:, :, :] rgb_array1,
    const unsigned char [:, :, :] rgb_array2)

cpdef void blend_add_bgra_inplace(
    unsigned char [::1] bgra_array1,
    const unsigned char [::1] bgra_array2,
    bint format_32 = *
    )
cdef void blend_add_bgra_inplace_c(
    unsigned char [::1] bgra_array1,
    const unsigned char [::1] bgra_array2,
    bint format_32 = *
    )



# BLEND_RGB_ADD FOR ALPHA (INPLACE)
cpdef void blend_add_alpha(
    unsigned char [:, :] alpha_array1,
    const unsigned char [:, :] alpha_array2
    )
cdef void blend_add_alpha_c(
    unsigned char [:, :] alpha_array1,
    const unsigned char [:, :] alpha_array2
    )



# EQU BLEND_RGB_SUB
cpdef void blend_sub_surface(image1, image2)
cpdef void blend_sub_array(
    const Py_ssize_t w,
    const Py_ssize_t h,
    unsigned char [:, :, :] rgb_array1,
    const unsigned char [:, :, :] rgb_array2
    )
cdef void blend_sub_array_c(
    const Py_ssize_t w,
    const Py_ssize_t h,
    unsigned char [:, :, :] rgb_array1,
    const unsigned char [:, :, :] rgb_array2
    )

cpdef void blend_sub_bgra_inplace(
    unsigned char [::1] bgra_array1,
    const unsigned char [::1] bgra_array2,
    bint format_32 = *
    )
cdef void blend_sub_bgra_inplace_c(
    unsigned char [::1] bgra_array1,
    const unsigned char [::1] bgra_array2,
    bint format_32 = *
    )



# BLEND_RGB_MIN
cpdef void blend_min_surface(image1, image2)
cpdef void blend_min_array(
    const Py_ssize_t w,
    const Py_ssize_t h,
    unsigned char [:, :, :] rgb_array1,
    const unsigned char [:, :, :] rgb_array2
    )
cdef void blend_min_array_c(
    const Py_ssize_t w,
    const Py_ssize_t h,
    unsigned char [:, :, :] rgb_array1,
    const unsigned char [:, :, :] rgb_array2
    )


cpdef void blend_min_bgra_inplace(
    unsigned char [::1] bgra_array1,
    const unsigned char [::1] bgra_array2,
    bint format_32 = *
    )
cdef void blend_min_bgra_inplace_c(
    unsigned char [::1] bgra_array1,
    const unsigned char [::1] bgra_array2,
    bint format_32 = *
    )




# BLEND_RGB_MAX
cpdef void blend_max_surface(image1, image2)
cpdef void blend_max_array(
    const Py_ssize_t w,
    const Py_ssize_t h,
    unsigned char [:, :, :] rgb_array1,
    const unsigned char [:, :, :] rgb_array2
    )
cdef void blend_max_array_c(
    const Py_ssize_t w,
    const Py_ssize_t h,
    unsigned char [:, :, :] rgb_array1,
    const unsigned char [:, :, :] rgb_array2
    )
cpdef void blend_max_bgra_inplace(
    unsigned char [::1] bgra_array1,
    const unsigned char [::1] bgra_array2,
    bint format_32 = *
)
cdef void blend_max_bgra_inplace_c(
    unsigned char [::1] bgra_array1,
    const unsigned char [::1] bgra_array2,
    bint format_32 = *
    )


cpdef void blend_max_alpha(
    unsigned char [:, :] alpha1_array,
    unsigned char [:, :] alpha2_array
    )

cdef void blend_max_alpha_c(
    unsigned char [:, :] alpha1_array,
    unsigned char [:, :] alpha2_array
    )


cpdef void blend_min_alpha(
    unsigned char [:, :] alpha1_array,
    unsigned char [:, :] alpha2_array
    )

cdef void blend_min_alpha_c(
    unsigned char [:, :] alpha1_array,
    unsigned char [:, :] alpha2_array
    )