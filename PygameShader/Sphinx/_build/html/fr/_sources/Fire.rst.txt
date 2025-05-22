

Procedural Fire and Cloud Effects Library
=========================================

:mod:`Fire.pyx`

=====================

.. currentmodule:: Fire

|

1. Purpose of the Library
-------------------------

The library provides tools to generate dynamic fire and cloud effects that can be rendered
in real-time. These effects are highly customizable, allowing developers to control aspects such as:

- **Size and shape** of the effect (e.g., width, height, intensity).
- **Color palettes** for the fire or cloud.
- **Post-processing effects** like bloom, blur, brightness adjustment, and smoothing.
- **Performance optimization** through texture resolution reduction and fast algorithms.

The library is optimized for performance, making it suitable for real-time applications like games.
It uses **Cython** for speed and integrates with **Pygame** for rendering.

2. Key Functions
----------------

Fire Effect Functions
^^^^^^^^^^^^^^^^^^^^^

These functions generate procedural fire effects.

.. function:: fire_surface24_c_border(width, height, factor, palette, fire, intensity, low, high)

   Generates a fire effect with an optional border.

.. function:: fire_sub()

   A helper function for generating fire effects. It processes the fire intensity array and applies the color palette.

.. function:: fire_surface24_c()

   Similar to ``fire_surface24_c_border`` but without the border option. It generates a fire effect with
   customizable intensity and horizontal limits.

.. function:: fire_effect()

   The main Python-callable function for generating fire effects.

   Includes options for post-processing (bloom, blur, brightness adjustment) and
   performance optimization (``reduce_factor_``).

.. function:: fire_effect_c()

   The Cython-optimized core function for generating fire effects. It is called internally by ``fire_effect``.

Cloud Effect Functions
^^^^^^^^^^^^^^^^^^^^^^

These functions generate procedural cloud or smoke effects.

.. function:: cloud_surface24_c(width, height, factor, palette, cloud_, intensity, low, high)

   Generates a cloud effect as a 24-bit RGB pixel array.

.. function:: cloud_effect()

   The main Python-callable function for generating cloud effects.

   Includes options for post-processing (bloom, blur, brightness adjustment) and
   performance optimization (``reduce_factor_``).

.. function:: cloud_effect_c()

   The Cython-optimized core function for generating cloud effects. It is called internally by ``cloud_effect``.

3. Core Features
----------------

Customizable Effects
^^^^^^^^^^^^^^^^^^^^

- Control the size, intensity, and shape of fire and cloud effects using parameters
  like ``factor``, ``intensity``, ``low``, and ``high``.

Color Palettes
^^^^^^^^^^^^^^

- Use custom color palettes (``palette_``) to define the colors of the fire or cloud.

Post-Processing
^^^^^^^^^^^^^^^

- **Bloom**: Adds a glowing effect to bright areas.
- **Blur**: Softens the edges of the effect for a more realistic appearance.
- **Brightness Adjustment**: Increases or decreases the brightness of the effect.
- **Smoothing**: Applies bi-linear filtering for smoother textures.

Performance Optimization
^^^^^^^^^^^^^^^^^^^^^^^^

- **Reduce Factor**: Reduces the resolution of the texture for faster processing.
- **Fast Algorithms**: Options like ``fast_bloom_`` provide a balance between performance and visual quality.

Flexible Rendering
^^^^^^^^^^^^^^^^^^

- Supports optional Pygame surfaces (``surface_``) for reusing textures and improving performance.
- Allows transposing the effect (``transpose_``) to change its orientation.

Typical Workflow
----------------

1. **Initialize Parameters**:
   - Set the dimensions (``width_``, ``height_``), intensity, and other parameters for the effect.
   - Define a color palette (``palette_``) and intensity array (``fire_`` or ``cloud_``).

2. **Generate the Effect**:
   - Call the main functions (``fire_effect`` or ``cloud_effect``) to generate the effect.
   - Use optional parameters to customize the appearance and performance.

3. **Render the Effect**:
   - The function returns a Pygame surface that can be blitted directly to the game display.

4. Example Use Cases
--------------------

Game Effects
^^^^^^^^^^^^

- Create realistic fire effects for torches, explosions, or burning objects.
- Generate dynamic cloud or smoke effects for weather, magic spells, or environmental ambiance.

Simulations
^^^^^^^^^^^

- Use the library to simulate fire and smoke in scientific or educational applications.

Art and Animation
^^^^^^^^^^^^^^^^^

- Generate procedural fire and cloud textures for use in animations or digital art.

5. Summary
----------

This library is a powerful tool for generating procedural fire and cloud effects
in real-time applications. It combines flexibility, performance, and ease of use,
making it suitable for game development, simulations, and artistic projects.
The functions are highly customizable, allowing developers to fine-tune the appearance
and behavior of the effects while maintaining optimal performance.


6. Video samples
-----------------

.. video:: _static/Demo.mp4
   :width: 560
   :height: 315
