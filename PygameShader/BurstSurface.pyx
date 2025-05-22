# cython: binding=False, boundscheck=False, wraparound=False, nonecheck=False, cdivision=True,
# profile=False, initializedcheck=False, exceptval(check=False)
# cython: optimize.use_switch=True
# cython: warn.maybe_uninitialized=False
# cython: warn.unused=False
# cython: warn.unused_result=False
# cython: warn.unused_arg=False
# cython: language_level=3
# cython: write_stub_file = True
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

Cython Graphics Library
=======================

Key Features
------------
This library provides advanced tools for image manipulation and rendering in graphical applications.
Built on pygame and other low-level rendering techniques, it enables efficient handling of surfaces,
sprites, and pixel-based effects. The key features include:

- Splitting images into smaller blocks for sprite sheets and tile maps.
- Applying dynamic effects like bursts and controlled rebuilding of images.
- Storing and manipulating graphical elements in memory for optimized rendering.
- Enhancing performance in real-time applications through optimized functions.
- Supporting experimental features for advanced graphical effects and debugging.

Library Functional Overview
---------------------------
The library consists of a set of functions that facilitate various graphical transformations
and effects. It focuses on the efficient handling of pixel-based surfaces, allowing developers
to create complex visual effects such as explosions, dynamic surface reconstruction, and optimized
sprite manipulations.

Target Applications
-------------------
This library is particularly useful for:

- Game development: Enabling real-time effects like explosions, animations, and sprite transformations.
- Graphics simulations: Creating dynamic visual effects that require image manipulation.
- Image processing: Splitting, reconstructing, and modifying images for artistic or technical purposes.
- Research and experimentation: Testing new graphical rendering techniques and optimization strategies.

Summary
-------
This library is designed to enhance the capabilities of graphical applications, particularly
in game development and advanced image manipulation. By offering optimized functions for handling
surfaces, splitting images, applying burst effects, and rebuilding images, it provides a flexible
and efficient toolset for developers. Experimental functions add further possibilities for exploring
novel rendering techniques. The library is a valuable resource for those looking to implement complex
graphical transformations efficiently.

"""

# EXPERIMENTAL
import numpy
from pygame._sdl2.video import Image, Texture

try:
    cimport cython
    from cython.parallel cimport prange
    from cpython cimport PyObject_CallFunctionObjArgs, PyObject, \
        PyList_SetSlice, PyObject_HasAttr, PyObject_IsInstance, \
        PyObject_CallMethod, PyObject_CallObject
    from cpython.dict cimport PyDict_DelItem, PyDict_Clear, PyDict_GetItem, PyDict_SetItem, \
        PyDict_Values, PyDict_Keys, PyDict_Items
    from cpython.list cimport PyList_Append, PyList_GetItem, PyList_Size
    from cpython.object cimport PyObject_SetAttr
except ImportError:
    raise ImportError("\n<cython> library is missing on your system."
        "\nTry: \n   C:\\pip install cython on a window command prompt.")

from numpy import empty, uint8, asarray, ascontiguousarray

cdef extern from 'Include/Shaderlib.c':
    float randRangeFloat(float lower, float upper)nogil;
    int randRange(int lower, int upper)nogil;

from libc.stdio cimport printf

cdef float M_PI = <float>3.14159265358979323846

import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
try:
    import pygame
    from pygame import freetype, Color, BLEND_RGB_ADD, RLEACCEL, Surface, Rect
    from pygame import freetype
    from pygame.freetype import STYLE_STRONG
    from pygame.transform import rotate, smoothscale
    from pygame.surfarray import pixels3d
    from pygame.image import frombuffer
    from pygame.math import Vector2
    from pygame.sprite import Group

except ImportError:
    print("\n<Pygame> library is missing on your system."
          "\nTry: \n   C:\\pip install pygame on a window command prompt.")
    raise SystemExit

from pygame import sprite
from libc.math cimport atan2f, sinf, cosf, sqrtf

from PygameShader.config import OPENMP, THREAD_NUMBER, __VERSION__

cdef int THREADS = 1

if OPENMP:
     THREADS = THREAD_NUMBER

DEF SCHEDULE = 'static'

# -------------------------------------- INTERFACE -------------------------------------

@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef unsigned char [:, :, ::1] pixel_block_rgb(
        unsigned char [:, :, :] array_, int start_x, int start_y,
        int w, int h, unsigned char [:, :, ::1] block) nogil:
    """
    Extract a block of pixels from a larger 2D array representing an image.

    This function retrieves a rectangular block of pixels from a larger image 
    (or sprite sheet) and stores it into a smaller block array. The function 
    delegates the actual pixel extraction to the `pixel_block_rgb_c` helper 
    function for performance optimization.

    :param array_:
        unsigned char; A 3D array representing the full image (height x width x 3). 
        This is the source from which the pixel block will be extracted.

    :param start_x:
        int; The x-coordinate of the top-left corner of the block to extract. 
        This defines where the block starts in the source image.

    :param start_y:
        int; The y-coordinate of the top-left corner of the block to extract. 
        This defines where the block starts in the source image.

    :param w:
        int; The width of the block to extract. The block will be `w` pixels wide.

    :param h:
        int; The height of the block to extract. The block will be `h` pixels tall.

    :param block:
        unsigned char; A pre-allocated empty block of size `w x h x 3`. 
        This array will be filled with the pixel data from the source image.

    :return:
        unsigned char[:,:,:]; A 3D array (height x width x 3) representing the 
        extracted pixel block. The block is stored in the `block` parameter.
    """
    return pixel_block_rgb_c(array_, start_x, start_y, w, h, block)


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef surface_split(object surface_, int size_, int rows_, int columns_):
    """
    Split a pygame.Surface into smaller subsurfaces.

    This function divides a larger surface into smaller subsurfaces (blocks) 
    based on the specified size, number of rows, and columns. It delegates 
    the actual splitting to the `surface_split_c` helper function for efficient processing.

    :param surface_:
        pygame.Surface; The surface to be split into smaller subsurfaces.

    :param size_:
        int; The size (width and height) of each subsurface. The resulting 
        subsurfaces will be square with dimensions `size_ x size_`.

    :param rows_:
        int; The number of rows of subsurfaces to create. This determines 
        how many horizontal blocks will be extracted from the original surface.

    :param columns_:
        int; The number of columns of subsurfaces to create. This determines 
        how many vertical blocks will be extracted from the original surface.

    :return:
        list; A list of pygame.Surface objects, each representing a smaller 
        block of the original surface.
    """
    return surface_split_c(surface_, size_, rows_, columns_)



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef void burst(
        object image_,
        list vertex_array_,
        int block_size_,
        int rows_,
        int columns_,
        int x_,
        int y_,
        int max_angle_=0
):
    """
    Simulate an explosion effect on a surface by splitting it into smaller subsurfaces or pixels.

    This function breaks a given surface into multiple smaller blocks or pixels and applies 
    an explosion effect, projecting them in random directions. The subsurfaces or pixels 
    are placed at specified positions and can be projected at different angles.

    :param image_:
        pygame.Surface; The surface to be split into smaller subsurfaces or pixels.

    :param vertex_array_:
        list; A list that will be populated with the created sprites, each representing a 
        smaller subsurface or pixel. These sprites will have attributes like position, 
        velocity (vector), and other states related to the explosion.

    :param block_size_:
        int; The size of each block (subsurface). If set to 1, each block will be a single pixel.

    :param rows_:
        int; The number of rows of subsurfaces or pixels to create. Determines how many 
        horizontal blocks will be created.

    :param columns_:
        int; The number of columns of subsurfaces or pixels to create. Determines how many 
        vertical blocks will be created.

    :param x_:
        int; The x-coordinate of the top-left corner of the surface before the explosion effect.

    :param y_:
        int; The y-coordinate of the top-left corner of the surface before the explosion effect.

    :param max_angle_:
        int, optional; The maximum angle (in degrees) at which each subsurface/pixel will be 
        projected. If not provided, defaults to 0 (no angle applied). The projection angle 
        is random within the given range.

    :return:
        void; The function modifies the `vertex_array_` in place by appending the created 
        sprite objects.
    """
    burst_c(image_, vertex_array_, block_size_, rows_, columns_, x_, y_, max_angle_)


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef void display_burst(
        object screen_,
        list vertex_array_,
        unsigned char blend_=0
):
    """
    Display an explosion effect on the screen by rendering the burst of sprites.

    This function iterates through the list of sprites (created by the burst effect), 
    rendering each sprite onto the provided screen surface. The sprites are drawn 
    with optional blending effects for smoother transitions, such as additive blending.

    :param screen_:
        pygame.Surface; The screen (or rendering surface) where the explosion effect 
        (burst of sprites) will be drawn.

    :param vertex_array_:
        list; A list of sprite objects that represent the subsurfaces or pixels created 
        during the explosion effect. These sprites will be rendered on the screen.

    :param blend_:
        unsigned char, optional; The blending mode to apply when drawing the sprites.
        Default is 0 (no blending). A non-zero value may specify an additive or 
        other blending mode depending on how it's defined.

    :return:
        void; The function draws each sprite from `vertex_array_` to the `screen_` surface.
    """
    display_burst_c(screen_, vertex_array_, blend_)


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef void rebuild_from_frame(
        object screen_,
        unsigned int current_frame_,
        unsigned int start_frame,
        list vertex_array_,
        unsigned char blend_ = 0
):
    """
    Rebuild and render an explosion effect from a specific frame.

    This function rebuilds and renders an explosion effect (created from sprites) on the 
    screen starting from a specified frame number. The sprites are progressively 
    re-rendered from their original positions and updated based on the `current_frame_`. 
    The function also supports blending for smoother visual transitions during the effect.

    :param screen_:
        pygame.Surface; The surface (or screen) where the rebuilt explosion effect will 
        be drawn.

    :param current_frame_:
        unsigned int; The current frame number, used to control the timing and progression 
        of the explosion effect.

    :param start_frame:
        unsigned int; The frame number at which to start rebuilding the explosion effect. 
        The process will only begin after the `current_frame_` exceeds this value.

    :param vertex_array_:
        list; A list of sprite objects representing the subsurfaces or pixels of the 
        explosion. These sprites will be rendered onto the screen.

    :param blend_:
        unsigned char, optional; The blending mode to use when rendering the sprites. 
        The default is 0 (no blending). A non-zero value will apply the specified blend mode 
        (e.g., additive blending).
        
    :return:
        void; This function updates the screen by rendering the sprites based on the 
        specified frame and blending settings.
    """
    rebuild_from_frame_c(screen_, current_frame_, start_frame, vertex_array_, blend_)


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef void burst_into_memory(
        unsigned int n_,
        object sg_,
        object screenrect,
        bint warn_ = False,
        bint auto_n_ = False
):
    """
    Burst a sprite group into memory, creating an explosion effect.

    This function processes a sprite group (`sg_`) and generates an explosion effect by 
    bursting it into memory. The function creates a set of sprites or subsurfaces that 
    are distributed across a specified region of the screen defined by `screenrect`. 
    The number of sprites (`n_`) determines how many parts the sprite group will be 
    divided into. It also supports warnings and automatic configuration for the number 
    of elements to process.

    :param n_:
        unsigned int; The number of elements to burst into memory. This controls 
        how many parts the sprite group will be split into for the explosion effect.

    :param sg_:
        object; A pygame.sprite.Group containing the sprites to be processed and 
        exploded into memory.

    :param screenrect:
        object; The pygame.Rect object that defines the area of the screen where 
        the explosion effect will be displayed. This helps control the positioning 
        of the sprites within the screen area.

    :param warn_:
        bint, optional; A flag to trigger warnings if certain conditions are met 
        (e.g., if the sprite group is too large to process efficiently). Defaults to False.

    :param auto_n_:
        bint, optional; A flag that, if set to True, automatically determines the 
        number of elements to burst based on the sprite group’s size. Defaults to False.

    :return:
        void; This function modifies the memory by creating and storing the explosion 
        effect, which can then be displayed on the screen.
    """
    burst_into_memory_c(n_, sg_, screenrect, warn_, auto_n_)


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef void rebuild_from_memory(
        object screen_,
        list vertex_array_,
        unsigned char blend_ = 0
):
    """
    Rebuild and render an image from stored memory data.

    This function takes a list of vertex data (`vertex_array_`) and reconstructs 
    an image or animation by rendering the stored memory information onto the 
    provided screen. The image is rendered using the specified blending mode 
    (`blend_`). This can be used for reconstructing exploded or fragmented 
    visual effects from previously stored memory.

    :param screen_:
        object; The surface or screen where the rebuilt image will be rendered. 
        Typically, this would be a pygame.Surface representing the display window.

    :param vertex_array_:
        list; A list containing the vertex data (e.g., pixels, sprite blocks, 
        or fragmented image parts) used to reconstruct the image. This data 
        is used to render the final visual output on the screen.

    :param blend_:
        unsigned char, optional; The blend mode to be used when rendering the 
        image. A blend mode of 0 typically represents no blending (normal mode), 
        while other values may represent different additive or subtractive blending 
        effects. Defaults to 0 (no blending).
        
    :return:
        void; This function modifies the screen object by rendering the rebuilt 
        image from the provided vertex data without returning a value.
    """
    rebuild_from_memory_c(screen_, vertex_array_, blend_)



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef void burst_experimental(
        object render_,
        object image_,
        object group_,
        int block_size_,
        int rows_,
        int columns_,
        int x_,
        int y_,
        int max_angle_ = 359
):
    """
    Explode a surface into multiple subsurfaces or pixels and apply random projections.

    This function takes a single surface (`image_`) and splits it into multiple blocks 
    or individual pixels, based on the specified block size, number of rows, and columns. 
    Each resulting block or pixel is assigned a random angle of projection, giving the effect 
    of the surface being 'exploded' into separate pieces. The pieces are added to the specified 
    sprite group (`group_`) for further manipulation or rendering.

    :param render_:
        object; The 2D rendering context used for rendering the exploded pieces. 
        Typically a pygame.Surface or similar rendering context.

    :param image_:
        object; The pygame.Surface to be split into smaller blocks or pixels. 
        This image will be 'exploded' into multiple parts based on the block size.

    :param group_:
        object; A pygame.sprite.Group that will contain the resulting pieces 
        (subsurfaces or pixels) after the explosion. Each part of the exploded image 
        will be added as a sprite to this group.

    :param block_size_:
        int; The size of each block or pixel in the explosion. This defines 
        the width and height of each subsurface or pixel entity. A block size of 1 
        will result in individual pixels being created.

    :param rows_:
        int; The number of rows into which the surface will be divided. This, combined 
        with the block size, determines how many blocks are created along the vertical axis.

    :param columns_:
        int; The number of columns into which the surface will be divided. This, combined 
        with the block size, determines how many blocks are created along the horizontal axis.

    :param x_:
        int; The starting x-coordinate of the surface, representing the top-left corner 
        before the explosion.

    :param y_:
        int; The starting y-coordinate of the surface, representing the top-left corner 
        before the explosion.

    :param max_angle_:
        int, optional; The maximum angle (in degrees) for the projection of each exploded 
        piece. Each piece will be projected at a random angle between 0 and `max_angle_`. 
        Defaults to 359 degrees (full circle).

    :return:
        void; This function modifies the `group_` object by adding the exploded pieces as 
        sprites, but does not return any value.
    """
    burst_experimental_c(
        render_,
        image_,
        group_,
        block_size_,
        rows_,
        columns_,
        x_, y_,
        max_angle_
    )


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef void db_experimental(
        object screenrect_,
        object render_,
        object group_,
        bint clean_ = False,
        unsigned char blend_=0):
    db_experimental_c(
        screenrect_,
        render_,
        group_,
        clean_,
        blend_)

@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef void rff_experimental(
        object render_,
        object screenrect_,
        unsigned int current_frame_,
        unsigned int start_frame,
        object group_,
        unsigned char blend_ = 0):
    """
    Rebuild an exploded image from a specific frame number.

    This function starts rebuilding an exploded image from a given start frame 
    number and continues to modify the image until it reaches the current frame.
    It delegates the core logic to the `rff_experimental_c` function for performance.

    The rebuild process involves inverting the direction of each pixel or pixel block's 
    vector and drawing them to the render surface, while respecting the screen boundaries.

    :param render_:
        Pygame.Surface; The surface where the image will be rendered. This is usually 
        the game screen or the target rendering surface.

    :param screenrect_:
        Pygame.Rect; The rectangle defining the screen's area. Used to determine 
        if a sprite is within the visible area of the screen.

    :param current_frame_:
        unsigned int; The current frame number. The image will start rebuilding 
        from the specified `start_frame` until the `current_frame` is reached.

    :param start_frame:
        unsigned int; The frame number at which the rebuild process starts. 
        The image will not be modified before this frame.

    :param group_:
        Pygame.sprite.Group; A group of sprites representing the blocks or 
        pixels of the image to be rebuilt. Each sprite is manipulated and 
        rendered during the rebuild process.

    :param blend_:
        unsigned char (optional, default 0); The blend mode to use when rendering 
        the sprites. Blend modes control how the sprite colors are combined 
        with the existing surface.

    :return:
        None; This function modifies the `group_` sprites and the surface 
        directly. No value is returned.
    """
    rff_experimental_c(
        render_,
        screenrect_,
        current_frame_,
        start_frame,
        group_,
        blend_)


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef void rfm_experimental(
        object screenrect_,
        object render_,
        object group_,
        unsigned char blend_=0):
    """
    Rebuild an exploded image with pixels or pixel blocks outside the screen boundaries.

    This function applies an experimental process to rebuild an image using pixels or 
    pixel blocks that are outside the boundaries of the screen. It delegates the core 
    logic to the `rfm_experimental_c` function for better performance. This process 
    is often used for creating effects where the image "explodes" or pieces of it move 
    outside the screen area.

    :param screenrect_:
        Pygame.Rect; The rectangle representing the boundaries of the screen or 
        the game display. Used to check if a sprite is within the screen's area.

    :param render_:
        Pygame.Surface; The surface where the rebuilt image will be rendered. This is 
        typically the game display or render target.

    :param group_:
        Pygame.sprite.Group; A group of sprites that represent the pixels or blocks 
        being used to rebuild the image. Each sprite in the group will be processed 
        and blitted onto the surface.

    :param blend_:
        unsigned char (optional, default 0); A blend mode to be used when rendering 
        the sprites. Blend modes control how the colors of the sprites are combined 
        with the existing colors on the surface. 

    :return:
        None; This function modifies the surface directly and does not return any 
        value. It also modifies the group of sprites in place.
    """
    rfm_experimental_c(
        screenrect_,
        render_,
        group_,
        blend_)


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef void build_surface_inplace(
        object surface_,
        object group_,
        unsigned int block_width,
        unsigned int block_height
):
    """
    Rebuild a Pygame surface from a group of sprites (inplace).

    This function takes a group of sprites and blits their images onto the specified 
    surface at their respective positions, updating the surface in place. The process 
    modifies the surface directly without creating a new one.

    It delegates the task to the `build_surface_inplace_c` function, which is written 
    in C for improved performance.

    :param surface_:
        Pygame.Surface; The surface that will be updated by blitting the sprites onto it. 
        The original surface is modified directly, and no new surface is created.

    :param group_:
        Pygame.sprite.Group; A group of sprites to be blitted onto the surface. Each sprite's 
        position is determined by its `rect` attribute.

    :param block_width:
        unsigned int; The width of the block used for the tiling or reconstruction process. 
        It can influence how the surface is rebuilt.

    :param block_height:
        unsigned int; The height of the block used for the tiling or reconstruction process. 
        It works in conjunction with `block_width` to determine the surface layout.

    :return:
        None; This function modifies the surface directly and does not return any value.
    """
    build_surface_inplace_c(surface_, group_, block_width, block_height)



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef void build_surface_inplace_fast(
        object surface_,
        object group_,
        unsigned int block_width,
        unsigned int block_height
):
    """
    Efficiently rebuild a Pygame surface from a group of sprites (inplace).

    This function rapidly constructs a surface by blitting individual sprite images 
    onto the given surface at their respective positions. The process is performed 
    in place, meaning the surface is directly updated without creating a new surface.

    It delegates the actual work to the optimized `build_surface_inplace_fast_c` 
    function for performance.

    :param surface_:
        Pygame.Surface; The surface to which the sprites will be blitted. 
        This surface is modified directly.

    :param group_:
        Pygame.sprite.Group; The group of sprites whose images will be blitted onto 
        the surface. Each sprite's position is taken from its `rect` attribute.

    :param block_width:
        unsigned int; The width of the blocks that will be used to build the surface. 
        This is important for any tiling logic and might impact performance depending 
        on the block size.

    :param block_height:
        unsigned int; The height of the blocks that will be used to build the surface. 
        Similar to `block_width`, it affects how the surface is reconstructed.

    :return:
        None; The surface is modified directly, so there is no return value.
    """
    build_surface_inplace_fast_c(surface_, group_, block_width, block_height)


# -------------------------------------- INTERFACE -------------------------------------

@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef inline unsigned char [:, :, ::1] pixel_block_rgb_c(
        unsigned char [:, :, :] array_, int start_x, int start_y,
        int w, int h, unsigned char [:, :, ::1] block) nogil:
    """
    Extract a rectangular block of pixels (RGB) from a larger image (sprite sheet).

    This method extracts a rectangular region from the provided image array and 
    stores it into the provided empty block array. The block is filled with the 
    pixel values from the source image starting from the coordinates `start_x` and 
    `start_y`, with the specified width `w` and height `h`.

    This method is typically used to extract individual sprites from a sprite sheet.

    :param array_: 
        unsigned char; A 3D array (height x width x 3) representing the source image 
        (sprite sheet) from which the block will be extracted.

    :param start_x: 
        int; The x-coordinate of the starting point of the block to be extracted.

    :param start_y: 
        int; The y-coordinate of the starting point of the block to be extracted.

    :param w: 
        int; The width (in pixels) of the block to extract.

    :param h: 
        int; The height (in pixels) of the block to extract.

    :param block: 
        unsigned char; A 3D array (height x width x 3) that will be populated with 
        the extracted pixel data.

    :return: 
        unsigned char; The filled block array (w x h x 3) containing the extracted 
        RGB pixel data from the sprite sheet.
    """

    cdef:
        int x, y, xx, yy

    # Iterate over the width (w) of the block
    for x in prange(w, schedule=SCHEDULE, num_threads=THREADS):
        xx = start_x + x  # Calculate the x-coordinate in the source image
        # Iterate over the height (h) of the block
        for y in range(h):
            yy = start_y + y  # Calculate the y-coordinate in the source image

            # Assign the RGB values from the source image to the block
            block[y, x, 0] = array_[xx, yy, 0]  # Red channel
            block[y, x, 1] = array_[xx, yy, 1]  # Green channel
            block[y, x, 2] = array_[xx, yy, 2]  # Blue channel

    return block


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef surface_split_c(surface_, int size_, int rows_, int columns_):
    """
    Split a surface into smaller subsurfaces or blocks of pixels.

    This function takes a Pygame surface and splits it into smaller square 
    blocks (subsurfaces) of a specified size. The number of rows and columns 
    defines how many blocks the surface will be divided into.

    :param surface_: 
        Pygame.Surface; The surface to split into smaller subsurfaces.
        
    :param size_: 
        int; The size of each block (subsurface). The block will be a square of 
        dimensions `size_ x size_`.

    :param rows_: 
        int; The number of rows of blocks to generate.

    :param columns_: 
        int; The number of columns of blocks to generate.

    :return: 
        list; A list of smaller surfaces (subsurfaces), each of size `size_ x size_`. 
        These blocks are derived from the original surface.
        
    This function processes the surface using NumPy to efficiently split the surface 
    into the specified number of rows and columns, creating subsurfaces for each block.
    """
    
    cdef:
        unsigned int w, h

    w, h = surface_.get_size()

    cdef:
        unsigned char [:, :, :] rgb_array = surface_.get_view('3')
        list subsurface = []
        int rows, columns
        int start_x, end_x, start_y, end_y
        int width = <object>rgb_array.shape[1]
        int height = <object>rgb_array.shape[0]

    cdef:
        unsigned char [:, :, ::1] empty_array = empty((size_, size_, 3), uint8)
        unsigned char [:, :, ::1] block_array = empty((size_, size_, 3), uint8)

    with nogil:
        # Iterate over rows and columns to split the surface into blocks
        for rows in range(rows_):
            start_y = rows * size_
            end_y   = (rows + 1) * size_
            for columns in range(columns_):
                start_x = columns * size_
                end_x   = start_x + size_
                
                # Extract a block of pixels from the surface
                block_array = pixel_block_rgb_c(
                    rgb_array, start_x, start_y, size_, size_, empty_array)
                
                with gil:
                    # Create a subsurface from the block and append it to the list
                    sub_surface = frombuffer(
                        asarray(block_array).copy(), (size_, size_), 'RGB')
                    PyList_Append(subsurface, sub_surface)
    
    return subsurface



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef bint is_power_of_two(int n) nogil:
    """
    Check if a given integer is a power of two.

    This function determines if the integer `n` is a power of two 
    by using bitwise operations. A number is a power of two if it 
    has exactly one '1' bit in its binary representation (e.g., 1, 2, 4, 8, 16, ...).

    :param n: 
        int; The number to check.

    :return: 
        bint; Returns True if `n` is a power of two, False otherwise.
        
    The function returns False for values of `n` that are 0 or less, as they cannot be powers of two.
    """
    return (n != 0) and (n & (n - 1) == 0)

"""
<< ----------------------------------- EXPERIMENTAL ---------------------------------------- >>
Using pygame._sdl2.video library

** Blending process is currently not working with _sdl2 (PYGAME 2.4.0)
"""


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef void burst_experimental_c(
        object render_,
        object image_,
        object group_,
        int block_size_,
        int rows_,
        int columns_,
        int x_, int y_,
        int max_angle_=0):
    """
    Explode a surface into multiple subsurfaces or pixels, creating an explosion effect.
    
    This function breaks a surface into smaller pieces (either subsurfaces or pixels) 
    and applies a random velocity vector to each piece, making it move in random directions 
    from the original position. The pieces are created by splitting the surface into 
    a grid of blocks or pixels, and each block/pixel is assigned a random angle and velocity 
    for movement.

    :param render_:
        Render context, which is the _sdl 2D rendering context for a window (experimental). 
        Used to render the textures.

    :param image_:
        pygame.Surface; The surface to transform into multiple subsurfaces or pixels.

    :param group_:
        pygame.sprite.Group; A sprite group that will hold all the subsurface or pixel sprites.

    :param block_size_:
        int; The size of each block (subsurface). If set to 1, each block will be a single pixel. 
        Otherwise, it's the width and height of the subsurface blocks.

    :param rows_:
        int; The number of rows of subsurfaces or pixels to generate.

    :param columns_:
        int; The number of columns of subsurfaces or pixels to generate.

    :param x_:
        int; The x-coordinate of the top-left corner of the surface before the explosion effect.

    :param y_:
        int; The y-coordinate of the top-left corner of the surface before the explosion effect.

    :param max_angle_:
        int; Optional; The maximum angle in degrees for the velocity direction of each particle. 
        If not specified, defaults to 0 (no angle applied).

    :return:
        None; This function adds the generated sprite objects to the provided sprite group.
    """

    # Validate input parameters
    assert PyObject_IsInstance(image_, Surface), \
        "\nAttribute image_ must be a pygame.Surface type, got %s" % type(image_)
    assert PyObject_IsInstance(group_, Group), \
        "\nAttribute group_ must be a pygame.sprite.Group type, got %s" % type(group_)
    assert is_power_of_two(block_size_) and block_size_ > 0, \
        "\nAttribute block_size_ must be a power of two and greater than 0"
    assert rows_ > 0, "\nAttribute rows_ must be > 0"
    assert columns_ > 0, "\nAttribute columns_ must be > 0"

    # Normalize max_angle to be between 0 and 360 degrees
    max_angle_ %= 360

    cdef:
        list subsurface = surface_split_c(image_, block_size_, rows_, columns_)
        int n = 0
        float random_angle

    # Define the rectangular area where the exploded image will be placed
    r = Rect(x_, y_, columns_ * block_size_, rows_ * block_size_)

    with cython.cdivision(True):
        # Iterate through each subsurface and create a corresponding sprite
        for surf in subsurface:
            s = pygame.sprite.Sprite()
            s.image = Image(Texture.from_surface(render_, surf))  # Create an image from the surface

            # Calculate the position of each block/pixel in the grid
            s.rect = surf.get_rect(
                topleft=((n % columns_) * block_size_ + x_,
                         <int>(n / columns_) * block_size_ + y_))

            # Generate a random angle for the velocity vector
            random_angle = randRangeFloat(<float>0.0, max_angle_)
            s.vector = Vector2(<float>cosf(random_angle), <float>sinf(random_angle)) * \
                       randRangeFloat(<float>5.0, <float>10.0)  # Random speed between 5 and 10

            n += 1  # Increment the counter for the next block

            # Store the original position of the sprite
            s.org = (s.rect.topleft[0], s.rect.topleft[1])
            s.counter = 0  # Initialize counter (could be used for animation or effects)
            s.stop = False  # Initially, the sprite is not stopped
            s.rebuild_state = False  # Sprite has not been rebuilt yet

            # Add the sprite to the sprite group for rendering and updating
            group_.add(s)



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef void db_experimental_c(
        object screenrect_,
        object render_,
        object group_,
        bint clean_ = False,
        unsigned char blend_=0):
    """
    DB_EXPERIMENTAL (DISPLAY BURST) 
    
    Module using the _sdl pygame library
    """

    cdef:
        screen_blit = render_.blit

    for s in group_:

        if s.rect.colliderect(screenrect_):

            s_rect          = s.rect
            s_vector        = s.vector
            s_rect.centerx += s_vector.x
            s_rect.centery += s_vector.y

            screen_blit(s.image, s.rect)

        else:
            if clean_:
                group_.remove(s)

@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef void rff_experimental_c(
        object render_,
        object screenrect_,
        unsigned int current_frame_,
        unsigned int start_frame,
        object group_,
        unsigned char blend_ = 0
):
    """
    (RFF_EXPERIMENTAL) Rebuild an exploded image from a specific frame number.

    This function rebuilds an image (exploded pixels or blocks) starting from a given frame number. 
    The process involves iterating over the sprite group and using vectors to control the position 
    of each pixel or block as the image is reconstructed. Once the specified start frame is reached, 
    the rebuild process begins, and the image is rebuilt from that frame onward.

    :param render_:
        pygame.Surface; The game display surface to render the exploded image.

    :param screenrect_:
        pygame.Rect; A rectangle that defines the boundaries of the game screen.

    :param current_frame_:
        unsigned int; The current frame number in the animation or game loop.

    :param start_frame:
        unsigned int; The frame number at which the image rebuilding should begin. 
        If `start_frame` is 0, a ValueError is raised.

    :param group_:
        pygame.sprite.Group; A group of sprites, each representing a pixel or block of pixels 
        in the exploded image. Each sprite has attributes like `rect`, `org`, `vector`, and `rebuild_state`.

    :param blend_:
        unsigned char; A blend mode flag used when drawing the pixels, for example, additive blending modes 
        (e.g., `BLEND_RGB_ADD`).
        
    :return: 
        void; This function does not return any value, it modifies the sprites' positions and renders them on the screen.
    """

    # Ensure the start_frame is valid (not 0)
    if start_frame == 0:
        raise ValueError("\nYou must specify a start frame number != 0")

    cdef:
        render_blit = render_.blit  # Reference to the blit function for rendering sprites
        int s_org0, s_org1  # Variables to hold the original positions of the sprite

    screenrect = screenrect_  # Define the screen boundaries for rendering

    # Iterate through each sprite in the group
    for s in group_:

        # Rebuild the image by inverting the vector after the start_frame has been reached
        if current_frame_ > start_frame and not s.rebuild_state:
            s.vector = -s.vector  # Invert the vector to start rebuilding the image
            s.rebuild_state = True  # Mark the sprite as having started the rebuild process

        s_rect = s.rect  # Get the current position (rect) of the sprite
        s_vector = s.vector  # Get the current movement vector of the sprite
        s_rect.topleft += s_vector  # Move the sprite based on its vector

        # Only start checking distance and stopping the sprite once the frame is beyond the start_frame
        if current_frame_ > start_frame and not s.stop:
            s_org0, s_org1 = s.org[0], s.org[1]  # Get the original position of the sprite

            # Calculate the distance squared from the original position (avoiding square root for performance)
            if <float> sqrtf((s_org0 - s.rect.topleft[0]) ** 2 + (s_org1 - s.rect.topleft[1]) ** 2) < <float> 2.0:
                # If the sprite has reached its original position, stop its movement
                s.vector = Vector2(<float> 0.0, <float> 0.0)  # Set the movement vector to zero
                s.rect.topleft = (s_org0, s_org1)  # Reset the position to the original one
                s.stop = True  # Mark the sprite as stopped

        # If the sprite is within the screen boundaries, blit it to the screen
        if screenrect.contains(s_rect):
            render_blit(s.image, s.rect, special_flags=blend_)  # Render the sprite using the specified blend mode




@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef void rfm_experimental_c(
        object screenrect_,
        object render_,
        object group_,
        unsigned char blend_=0
):
    """
    (RFM_EXPERIMENTAL REBUILD FROM MEMORY) 
    Rebuild an image with exploded pixels or pixel blocks, some of which may be outside screen boundaries.

    This function uses the _SDL Pygame library to handle the rendering of exploded pixel blocks or pixels. 
    Each sprite in the group is checked for its position relative to the screen boundaries, and pixels are 
    blitted (drawn) on the screen while considering their movement based on a predefined vector.

    :param screenrect_: 
        pygame.Surface; The game display surface that acts as the boundary for the rendering.
        
    :param render_: 
        list; A Python list containing objects (pixels or blocks of pixels), each with predefined attributes and values. 
        These objects are used to render the exploded image.

    :param group_: 
        pygame.SpriteGroup; A group of sprite objects, where each sprite represents a pixel or block of pixels. 
        Each sprite has attributes such as `rect`, `org`, `vector`, and `stop` to control its movement and status.

    :param blend_: 
        unsigned char; A blend mode flag used when drawing the pixels, for example, 
        additive blending modes (e.g., BLEND_RGB_ADD).
    """
    cdef:
        render_blit = render_.blit  # Reference to the blit function for rendering
        int s_org0, s_org1, s_0, s_1  # Coordinates for the original position and current position of the block
        float s_r  # Distance from the original position
        bint s_stop  # Flag to stop the movement of the block

    screenrect = screenrect_
    screenrect_contains = screenrect.contains  # Function to check if the block is within the screen boundaries

    # Iterate over each sprite (pixel/block) in the sprite group
    for s in group_:
        # Update the sprite's position based on its movement vector
        s.rect.topleft -= s.vector
        s_rect = s.rect  # Get the current position (rect) of the sprite
        s_stop = s.stop  # Get the current status of the sprite (whether it should stop moving)

        # Check if the block is still moving
        if not s_stop:
            # Get the original and current positions of the sprite
            s_org0, s_org1 = s.org[0], s.org[1]
            s_0, s_1 = s_rect.topleft[0], s_rect.topleft[1]

            # Calculate the distance squared from the original position (avoiding square root for performance)
            s_r = ((s_org0 - s_0) * (s_org0 - s_0)
                        + (s_org1 - s_1) * (s_org1 - s_1))

            # If the sprite is close enough to its original position, stop its movement
            if s_r < <float>8.0:
                s.vector = Vector2(<float>0.0, <float>0.0)  # Set the movement vector to zero
                s_0, s_1 = s_org0, s_org1  # Reset the position to the original
                s_stop = True  # Mark the sprite as stopped
                # group_.remove(s)  # Optionally remove the sprite from the group (commented out)

        # Check if the sprite's rectangle is within the screen boundaries
        if screenrect_contains(s_rect):
            # If the sprite is inside the screen boundaries, blit it to the screen
            render_blit(s.image, s_rect, special_flags=blend_)  # Render the sprite with the specified blend mode



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef void build_surface_inplace_c(
        object surface_,
        object group_,
        unsigned int block_width,
        unsigned int block_height
):
    """
    Build a Pygame surface from a sprite group (in-place) with block-based pixel manipulation.

    This function takes an existing Pygame surface and a sprite group, then draws (blits) 
    each sprite from the group onto the surface at its appropriate position. Additionally, 
    it manipulates individual pixel blocks of the surface, potentially altering the pixel 
    data directly within the provided surface. 

    The function uses a block-based approach to handle pixel manipulation and transfers 
    pixel data from the sprite's image to the surface in block-sized chunks.

    :param surface_:
        pygame.Surface; The target surface to which the sprites and pixel blocks will be drawn. 
        This surface is modified in place.

    :param group_:
        object; A Pygame sprite group (or any iterable) containing sprite objects. Each sprite 
        should have an `image` attribute (the visual representation) and a `rect` attribute 
        (the position on the surface).

    :param block_width:
        unsigned int; The width of each pixel block used for manipulating and drawing onto the surface.

    :param block_height:
        unsigned int; The height of each pixel block used for manipulating and drawing onto the surface.
    """

    # Prepare the surface view for manipulation and create a temporary block for pixel transfer
    cdef:
        unsigned char [:, :, :] array_ = surface_.get_view('3')  # Access the surface as a 3D numpy array
        unsigned char [:, :, :] block_ = \
            numpy.empty((block_width, block_height, 3), dtype=numpy.uint8)  # Temporary block for pixel data
        int i, j
        int x_, y_
        int w_, h_
        unsigned char x = 0
        unsigned char ii = 0, jj = 0
        unsigned int ww = surface_.get_width() - block_width  # Width limit for block position
        unsigned int hh = surface_.get_height() - block_height  # Height limit for block position

    # Iterate through each sprite in the sprite group
    for s in group_:
        # Get the top-left corner coordinates of the sprite
        x_ = s.rect[0]
        y_ = s.rect[1]

        # Blit the sprite onto the surface at the specified rect
        surface_.blit(s.image, s.rect)

        # Ensure coordinates are within the valid range of the surface
        with nogil:
            x_ = 0 if x_ < 0 else x_
            y_ = 0 if y_ < 0 else y_
            if x_ >= ww: x_ = ww
            if y_ >= hh: y_ = hh

            # Define the bottom-right corner for the block
            w_ = x_ + block_width
            h_ = y_ + block_height

            # Transfer the pixel data from the temporary block to the surface in blocks
            for i in prange(x_, w_, schedule=SCHEDULE, num_threads=THREADS):
                jj = 0  # Reset vertical block counter
                x = i - x_  # Calculate the offset within the block
                for j in range(y_, h_):
                    # Copy the pixel data from the block to the surface
                    array_[i, j, 0] = block_[x, jj, 0]
                    array_[i, j, 1] = block_[x, jj, 1]
                    array_[i, j, 2] = block_[x, jj, 2]
                    jj = jj + 1  # Increment the vertical counter for the block transfer



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef void build_surface_inplace_fast_c(
        object surface_,
        object group_,
        unsigned int block_width,
        unsigned int block_height
):
    """
    Build a Pygame surface from a sprite group (in-place).

    This function takes an existing Pygame surface and a sprite group, then draws (blits) 
    each sprite from the group onto the surface at the appropriate position. The drawing 
    is done directly onto the provided surface, so no new surface is created.

    :param surface_:
        pygame.Surface; The target surface where the sprites will be drawn. This surface is modified in place.

    :param group_:
        object; A Pygame sprite group (or any iterable) containing sprite objects. Each sprite should 
        have an `image` attribute (the visual representation) and a `rect` attribute (the position on the surface).

    :param block_width:
        unsigned int; The width of the blocks if needed for context, though this isn't currently utilized in the code.
        It may be used for future extension (e.g., to fit sprites into a grid or constrain the block size).

    :param block
    """
    for s in group_:
        surface_.blit(s.image, s.rect)



"""
<< ----------------------------------- EXPERIMENTAL ---------------------------------------- >>
"""

@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(False)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef void burst_c(
    image_,
    list vertex_array_,
    int block_size_,
    int rows_,
    int columns_,
    int x_,
    int y_,
    int max_angle_=0):

    """
    Explode a surface into multiple subsurfaces or pixels with an explosion effect.

    This function splits a `pygame.Surface` into smaller subsurfaces or individual 
    pixels (depending on the specified `block_size_`) and applies an explosion-like effect 
    by projecting them in random directions. The subsurfaces or pixels are projected from 
    the initial position in a 2D Cartesian plane. The optional `max_angle_` parameter defines 
    the maximum angle for each subsurface's movement direction.

    :param image_: 
        pygame.Surface; The surface to be split into multiple subsurfaces or pixels.
        
    :param vertex_array_: 
        list; A list of vertices or coordinates where the subsurfaces/pixels will be projected. 
        This will be populated with sprite objects representing the exploded parts.
        
    :param block_size_: 
        int; The size (width x height) of each subsurface. If set to 1, the subsurfaces will be individual pixels.
        
    :param rows_: 
        int; The number of rows of subsurfaces or pixels to create.
        
    :param columns_: 
        int; The number of columns of subsurfaces or pixels to create.
        
    :param x_: 
        int; The x-coordinate of the top-left corner of the surface before the explosion effect.
        
    :param y_: 
        int; The y-coordinate of the top-left corner of the surface before the explosion effect.
        
    :param max_angle_: 
        int, optional; The maximum angle (in degrees) for the projection of each pixel/subsurface 
        in the 2D Cartesian plane. A value of 0 means no angle variation. Defaults to 0.
    """
    
    # Ensure the image_ is a pygame.Surface object
    assert PyObject_IsInstance(image_, Surface), \
        "\nAttribute image_ must be a pygame.Surface, got %s" % type(image_)

    # Ensure block_size_ is a positive power of two
    assert is_power_of_two(block_size_) and block_size_ > 0, \
        "\nAttribute block_size_ must be a power of two and greater than 0"

    # Ensure that rows_ and columns_ are positive integers
    assert rows_ > 0, "\nAttribute rows_ must be > 0"
    assert columns_ > 0, "\nAttribute columns_ must be > 0"

    # Normalize max_angle_ to ensure it's within 0-360 degrees
    max_angle_ %= 360

    cdef:
        # Split the image into smaller subsurfaces based on the block_size_
        list subsurface = surface_split_c(
            image_, block_size_, rows_, columns_)
        
        int n = 0
        float random_angle  # Random angle for movement direction

    # Define the area of the surface to be split (Rect defines the region)
    r = Rect(x_, y_, columns_ * block_size_, rows_ * block_size_)

    # Loop through each subsurface, assigning movement and position
    with cython.cdivision(True):
        for surf in subsurface:
            # Create a sprite for each subsurface
            s = sprite.Sprite()
            s.image = surf

            # Calculate the position of the subsurface within the grid
            s.rect = surf.get_rect(
                topleft=((n % columns_) * block_size_ + x_,
                         <int>(n / columns_) * block_size_ + y_))

            # Generate a random angle for the direction of movement
            random_angle = randRangeFloat(<float>0.0, max_angle_)

            # Assign a vector for movement direction, scaled randomly
            s.vector = Vector2(<float>cosf(random_angle),
                               <float>sinf(random_angle)) * randRangeFloat(<float>5.0, <float>10.0)
            
            n += 1  # Increment to move to the next subsurface
            s.org = (s.rect.topleft[0], s.rect.topleft[1])  # Original position
            s.counter = 0  # Initialize counter for animation effect
            s.stop = False  # Flag to stop movement
            s.rebuild_state = False  # Flag to rebuild the state if needed
            vertex_array_.append(s)  # Add the sprite to the vertex array



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef void display_burst_c(object screen_, list vertex_array_, unsigned char blend_=0):
    """
    Display an exploded image effect by decreasing the alpha value of each pixel/block every frame.
    
    This function simulates an "exploding" effect by decrementing the alpha value of the pixels 
    or blocks of pixels in `vertex_array_` each frame. It also supports different blend modes 
    through the `blend_` parameter. Note that using certain blend modes (like BLEND_RGB_ADD) 
    may prevent the alpha channel from being modified.

    :param screen_: Surface; the game display surface on which to render the pixels/blocks.
    :param vertex_array_: list; a Python list containing objects (pixels or blocks) which have 
                          attributes like `rect` (for position), `vector` (for movement), 
                          `image` (for appearance), and `counter` (for alpha decay).
    :param blend_: unsigned char; specifies the blend mode to be used (e.g., BLEND_RGB_ADD). 
                   A value of 0 defaults to no blending mode.
    """
    
    cdef:
        # Get references to blit and screen rectangle methods for performance
        screen_blit = screen_.blit
        screenrect  = screen_.get_rect()

    # Iterate through all the elements in the vertex array
    for s in vertex_array_:
        # Check if the pixel/block is within the screen boundaries
        if s.rect.colliderect(screenrect):

            # Update the position of the pixel/block based on its movement vector
            s_rect = s.rect
            s_vector = s.vector
            s_rect.centerx += s_vector.x
            s_rect.centery += s_vector.y

            # Blit the image onto the screen with the specified blending mode
            screen_blit(s.image, s.rect, special_flags=blend_)

            # Decrease the alpha value of the image to create the "explosion" effect
            # The alpha value is reduced by `counter` each frame (incremented by 2)
            s.image.set_alpha(max(<unsigned char>255 - s.counter, 0), RLEACCEL)
            s.counter += 2  # Increase the counter to progressively reduce alpha

        else:
            # Remove the pixel/block if it is outside the screen boundaries
            vertex_array_.remove(s)




@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef void rebuild_from_frame_c(
        object screen_,
        unsigned int current_frame_,
        unsigned int start_frame,
        list vertex_array_,
        unsigned char blend_ = 0
):
    """
    Rebuilds an exploded image from a specified game frame.

    This function reverses the explosion effect by moving pixel blocks back to their 
    original positions, starting from a given frame number.

    :param screen_: pygame Surface; The game display surface.
    :param current_frame_: unsigned int; The current frame count.
    :param start_frame: unsigned int; The frame number at which the image starts rebuilding.
    :param vertex_array_: List of objects representing pixel blocks, each with predefined 
                          attributes such as `rect`, `vector`, and `org` (original position).
    :param blend_: unsigned char; Blend mode for rendering (e.g., additive blending).
    :return: None (void function).
    """
    if start_frame == 0:
        raise ValueError("\nYou must specify a start frame number != 0")

    cdef:
        screen_blit = screen_.blit  # Optimized function reference for blitting
        int s_org0, s_org1  # Variables to store original coordinates

    screenrect = screen_.get_rect()  # Get screen boundaries

    for s in vertex_array_:
        
        # If past the start frame and the block hasn't started rebuilding, invert its vector
        if current_frame_ > start_frame and not s.rebuild_state:
            s.vector = -s.vector
            s.rebuild_state = True

        # Move the pixel block according to its vector
        s_rect = s.rect
        s_vector = s.vector
        s_rect.topleft += s_vector

        # Only check distance once the frame count exceeds start_frame
        if current_frame_ > start_frame and not s.stop:
            s_org0, s_org1 = s.org[0], s.org[1]

            # If the block is close enough to its original position, stop its movement
            if <float>sqrtf((s_org0 - s.rect.topleft[0]) ** 2
                    + (s_org1 - s.rect.topleft[1]) ** 2) < <float>2.0:
                s.vector = Vector2(<float>0.0, <float>0.0)
                s.rect.topleft = (s_org0, s_org1)
                s.stop = True

        # Render the block if it's still within screen boundaries
        if screenrect.contains(s_rect):
            screen_blit(s.image, s.rect, special_flags=blend_)




@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef void burst_into_memory_c(
        unsigned int n_,
        object sg_,
        object screenrect,
        bint warn_ = False,
        bint auto_n_ = False
):
    """
    Moves image pixel blocks into memory and tracks their locations over iterations.

    This function moves a set of pixels or blocks of pixels across the screen by adjusting 
    their position incrementally. It can be used in conjunction with `rebuild_from_memory`
    to create the effect of an image exploding and then reforming from scattered pixel blocks.

    :param n_: Number of iterations. In each iteration, pixels or blocks move to a new 
               location determined by their respective vectors.
    :param sg_: List or pygame Sprite group containing objects with `rect` and `vector` attributes.
                - `rect`: Defines the position and boundaries of the pixel block.
                - `vector`: Determines the direction and movement of the block.
    :param screenrect: A `pygame.Rect` object representing the display area.
    :param warn_: If True, raises a warning if any pixels are still visible after the movement.
                  Ignored if `auto_n_` is True.
    :param auto_n_: If True, overrides `n_` and automatically determines the necessary iterations
                    to move all pixels/blocks outside the screen bounds recursively.
    :return: None (void function).
    """
    assert n_ > 0, "\nArgument n_ must be > 0"

    cdef bint contain_rect

    # If auto_n_ is enabled, initialize with a single iteration and disable warnings
    if auto_n_:
        n_ = 1
        warn_ = False

    # Iterate through `n_` frames to move all pixel blocks
    for _ in range(n_):
        contain_rect = False  # Reset flag to check if any pixel block is still within bounds

        # Move each pixel block according to its vector
        for s in sg_:
            s.rect.topleft += s.vector  # Update position
            
            # Check if the block is still within the screen bounds
            if screenrect.contains(s.rect):
                contain_rect = True

    # If any pixel block remains within bounds and auto_n_ is enabled, recurse
    if contain_rect and auto_n_:
        burst_into_memory_c(
            n_,
            sg_,
            screenrect,
            warn_=warn_,
            auto_n_=auto_n_
        )

    # If warnings are enabled and some pixels remain visible, raise an error
    if warn_ and contain_rect:
        raise ValueError("\nburst_into_memory - Some pixels are still visible, increase n_.")



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef void rebuild_from_memory_c(object screen_, list vertex_array_, unsigned char blend_=0):
    """
    Rebuilds an image by repositioning pixel blocks that have moved outside the screen boundaries.
    This function ensures that displaced pixels or pixel blocks return to their original positions
    when they have reached a certain threshold distance from their origin.

    :param screen_: Pygame.Surface object representing the game display.
    :param vertex_array_: List of objects (pixels or blocks of pixels) containing pre-defined attributes
        such as position, movement vector, and image.
    :param blend_: Unsigned char specifying the blend mode for rendering (e.g., additive blending).
    """

    cdef:
        screen_blit = screen_.blit  # Optimized local reference to the blit method for performance
        int s_org0, s_org1, s_0, s_1  # Integer variables to store original and current positions
        float s_r  # Squared distance from the original position (avoiding sqrt for performance)
        bint s_stop  # Boolean flag to check if a block has stopped moving

    screenrect = screen_.get_rect()
    screenrect_contains = screenrect.contains  # Optimized local reference for boundary checking

    # Iterate over each pixel block in the vertex array
    for s in vertex_array_:
        s.rect.topleft -= s.vector  # Apply movement vector to update position
        s_rect = s.rect
        s_stop = s.stop

        # Check if the block is still in motion
        if not s_stop:
            s_org0, s_org1 = s.org[0], s.org[1]  # Original coordinates
            s_0, s_1 = s_rect.topleft[0], s_rect.topleft[1]  # Current coordinates

            # Calculate squared distance from origin to avoid performance-heavy sqrt operation
            s_r = ((s_org0 - s_0) * (s_org0 - s_0) + (s_org1 - s_1) * (s_org1 - s_1))

            # If the block is close to its original position, stop its movement
            if s_r < <float>8.0:
                s.vector = Vector2(<float>0.0, <float>0.0)  # Reset movement
                s_0, s_1 = s_org0, s_org1  # Snap to original position
                s_stop = True  # Mark as stopped

        # Only draw the block if it is within the screen boundaries
        if screenrect_contains(s_rect):
            screen_blit(s.image, s_rect, special_flags=blend_)  # Render the pixel block


