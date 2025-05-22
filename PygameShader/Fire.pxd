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
# **Fire and Cloud Effects Library**

This library is designed to generate **procedural fire and cloud effects** for use in games
 or other graphical applications. It uses a combination of noise textures, color palettes,
  and post-processing effects to create realistic and customizable fire and cloud animations.

---

## **Purpose of the Library**
The library provides tools to generate dynamic fire and cloud effects that can be rendered 
in real-time. These effects are highly customizable, allowing developers to control aspects such as:
- **Size and shape** of the effect (e.g., width, height, intensity).
- **Color palettes** for the fire or cloud.
- **Post-processing effects** like bloom, blur, brightness adjustment, and smoothing.
- **Performance optimization** through texture resolution reduction and fast algorithms.

The library is optimized for performance, making it suitable for real-time applications like games.
 It uses **Cython** for speed and integrates with **Pygame** for rendering.

---

## **Key Functions**

### **1. Fire Effect Functions**
These functions generate procedural fire effects.

#### **`fire_surface24_c_border`**
- Generates a fire effect with an optional border.
- **Inputs**:
  - `width`, `height`: Dimensions of the fire texture.
  - `factor`: Controls the vertical size of the fire.
  - `palette`: Color palette for the fire.
  - `fire`: 2D array of fire intensity values.
  - `intensity`, `low`, `high`: Control the fire's intensity and horizontal limits.

#### **`fire_sub`**
- A helper function for generating fire effects. It processes the fire intensity array and 
        applies the color palette.

#### **`fire_surface24_c`**
- Similar to `fire_surface24_c_border` but without the border option. It generates a fire effect with
        customizable intensity and horizontal limits.

#### **`fire_effect`**
- The main Python-callable function for generating fire effects.
- Includes options for post-processing (bloom, blur, brightness adjustment) and 
        performance optimization (`reduce_factor_`).

#### **`fire_effect_c`**
- The Cython-optimized core function for generating fire effects. It is called internally by `fire_effect`.

---

### **2. Cloud Effect Functions**
These functions generate procedural cloud or smoke effects.

#### **`cloud_surface24_c`**
- Generates a cloud effect as a 24-bit RGB pixel array.
- **Inputs**:
  - `width`, `height`: Dimensions of the cloud texture.
  - `factor`: Controls the vertical size of the cloud.
  - `palette`: Color palette for the cloud.
  - `cloud_`: 2D array of cloud intensity values.
  - `intensity`, `low`, `high`: Control the cloud's intensity and horizontal limits.

#### **`cloud_effect`**
- The main Python-callable function for generating cloud effects.
- Includes options for post-processing (bloom, blur, brightness adjustment) and 
        performance optimization (`reduce_factor_`).

#### **`cloud_effect_c`**
- The Cython-optimized core function for generating cloud effects. It is called internally by `cloud_effect`.

---

## **Core Features**

### **1. Customizable Effects**
- Control the size, intensity, and shape of fire and cloud effects using parameters 
        like `factor`, `intensity`, `low`, and `high`.

### **2. Color Palettes**
- Use custom color palettes (`palette_`) to define the colors of the fire or cloud.

### **3. Post-Processing**
- **Bloom**: Adds a glowing effect to bright areas.
- **Blur**: Softens the edges of the effect for a more realistic appearance.
- **Brightness Adjustment**: Increases or decreases the brightness of the effect.
- **Smoothing**: Applies bi-linear filtering for smoother textures.

### **4. Performance Optimization**
- **Reduce Factor**: Reduces the resolution of the texture for faster processing.
- **Fast Algorithms**: Options like `fast_bloom_` provide a balance between performance and visual quality.

### **5. Flexible Rendering**
- Supports optional Pygame surfaces (`surface_`) for reusing textures and improving performance.
- Allows transposing the effect (`transpose_`) to change its orientation.

---

## **Typical Workflow**

1. **Initialize Parameters**:
   - Set the dimensions (`width_`, `height_`), intensity, and other parameters for the effect.
   - Define a color palette (`palette_`) and intensity array (`fire_` or `cloud_`).

2. **Generate the Effect**:
   - Call the main functions (`fire_effect` or `cloud_effect`) to generate the effect.
   - Use optional parameters to customize the appearance and performance.

3. **Render the Effect**:
   - The function returns a Pygame surface that can be blitted directly to the game display.

---

## **Example Use Cases**

### **1. Game Effects**
- Create realistic fire effects for torches, explosions, or burning objects.
- Generate dynamic cloud or smoke effects for weather, magic spells, or environmental ambiance.

### **2. Simulations**
- Use the library to simulate fire and smoke in scientific or educational applications.

### **3. Art and Animation**
- Generate procedural fire and cloud textures for use in animations or digital art.

---

## **Summary**
This library is a powerful tool for generating procedural fire and cloud effects
in real-time applications. It combines flexibility, performance, and ease of use,
making it suitable for game development, simulations, and artistic projects.
The functions are highly customizable, allowing developers to fine-tune the appearance 
and behavior of the effects while maintaining optimal performance.

---

"""

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=ImportWarning)

cdef fire_surface24_c_border(
        const int width,
        const int height,
        const float factor,
        const unsigned int [::1] palette,
        float [:, ::1] fire,
        int intensity = *,
        int low       = *,
        int high      = *,
)

cpdef fire_sub(
        const int width,
        const int height,
        const float factor,
        const unsigned int [::1] palette,
        float [:, ::1] fire)

cdef fire_surface24_c(
        const int width,
        const int height,
        const float factor,
        const unsigned int [::1] palette,
        float [:, ::1] fire,
        unsigned int intensity = *,
        unsigned int low       = *,
        unsigned int high      = *,
)



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

cdef cloud_surface24_c(
        int width,
        int height,
        float factor,
        unsigned int [::1] palette,
        float [:, ::1] cloud_,
        int intensity = *,
        int low       = *,
        int high      = *,
    )

cdef cloud_effect_c(
        int width_,
        int height_,
        float factor_,
        unsigned int [::1] palette_,
        float [:, ::1] cloud_,

        # OPTIONAL
        unsigned short int reduce_factor_   = *,
        unsigned short int cloud_intensity_ = *,
        bint smooth_                        = *,
        bint bloom_                         = *,
        bint fast_bloom_                    = *,
        unsigned short int bpf_threshold_   = *,
        unsigned int low_                   = *,
        unsigned int high_                  = *,
        bint brightness_                    = *,
        float brightness_intensity_         = *,
        object surface_                     = *,
        bint transpose_                     = *,
        bint blur_                          = *
        )