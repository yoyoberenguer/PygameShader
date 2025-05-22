BlendFlags
========================================

:mod:`BlendFlags.pyx`

=====================


.. currentmodule:: BlendFlags

|


This library is focusing on **image blending operations**,
particularly for **BGRA/BGR and alpha channel manipulations**.
The functions are optimized for **fast, parallelized image processing**
using **Cython** and **OpenMP**.

1. Key Features & Purpose
-------------------------

**Image Blitting (``blit_s``)**

   - Handles copying an image (``image``) onto a destination surface (``destination``)
     at a specified position.

**Blending Operations**

   - Functions with names like ``blend_add_*``, ``blend_sub_*``, ``blend_min_*``,
     and ``blend_max_*`` indicate support for **various blending modes**:

     - Additive blending → Increases pixel brightness.
     - Subtractive blending → Decreases pixel brightness.
     - Minimum blending → Keeps the darkest pixels.
     - Maximum blending → Keeps the brightest pixels.

**Support for Different Data Representations**

   - **Surface-based functions (``blend_*_surface``)**: Operate on whole images
     (wrapper for SDL surfaces or similar structures).
   - **Array-based functions (``blend_*_array``)**: Operate on NumPy-style 3D arrays
     (``[height, width, channels]``).
   - **In-place operations (``blend_*_bgra_inplace``)**: Modify a 1D BGRA/BGR array
     directly, avoiding extra memory allocation.

**Alpha Channel Operations**

   - Functions like ``blend_add_alpha``, ``blend_max_alpha``, and ``blend_min_alpha``
     specifically target **alpha blending**, which is crucial for transparency effects.

**Parallelization & Optimization**

   - Use of **Cython** (``cdef`` functions) and **parallelization** (``prange`` from OpenMP)
     suggests that the library is designed for **high-performance image processing**,
     likely intended for real-time applications.

2. Possible Use Cases
---------------------

- **Game development & rendering** (e.g., blending sprites, effects)
- **Image processing applications** (e.g., compositing, transparency adjustments)
- **Graphics frameworks** needing fast pixel operations (e.g., custom shaders, filters)
