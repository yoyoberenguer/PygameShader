BurstSurface
========================================

:mod:`BurstSurface.pyx`

=====================


.. currentmodule:: BurstSurface

|

1. Cython Burst Surface Library
-------------------------------

2. Key Features
---------------

This library provides advanced tools for image manipulation and rendering in graphical applications.
Built on pygame and other low-level rendering techniques, it enables efficient handling of surfaces,
sprites, and pixel-based effects. The key features include:

- Splitting images into smaller blocks for sprite sheets and tile maps.
- Applying dynamic effects like bursts and controlled rebuilding of images.
- Storing and manipulating graphical elements in memory for optimized rendering.
- Enhancing performance in real-time applications through optimized functions.
- Supporting experimental features for advanced graphical effects and debugging.

3. Library Functional Overview
------------------------------

The library consists of a set of functions that facilitate various graphical transformations
and effects. It focuses on the efficient handling of pixel-based surfaces, allowing developers
to create complex visual effects such as explosions, dynamic surface reconstruction, and optimized
sprite manipulations.

4. Target Applications
----------------------

This library is particularly useful for:

- Game development: Enabling real-time effects like explosions, animations, and sprite transformations.
- Graphics simulations: Creating dynamic visual effects that require image manipulation.
- Image processing: Splitting, reconstructing, and modifying images for artistic or technical purposes.
- Research and experimentation: Testing new graphical rendering techniques and optimization strategies.

5. Cython Functions
-------------------

.. code-block:: python

   cpdef pixel_block_rgb_c():
       """
       Extracts a block of RGB pixels from a surface.
       Efficiently retrieves a section of an image as a contiguous memory view.
       Useful for sprite sheets and image manipulation.
       """

.. code-block:: python

   cpdef surface_split_c():
       """
       Splits a surface into multiple subsurfaces.
       Used for working with sprite sheets or dividing an image for animation.
       """

.. code-block:: python

   cpdef burst_c():
       """
       Creates an explosion effect by splitting a surface into blocks and moving them randomly.
       Ideal for animations where objects break apart dynamically.
       """

.. code-block:: python

   cpdef display_burst_c():
       """
       Renders the exploded effect created by `burst_c` onto the screen.
       """

.. code-block:: python

   cpdef rebuild_from_frame_c():
       """
       Reassembles an exploded surface back to its original form over a sequence of frames.
       Useful for animation effects where an object reconstructs over time.
       """

.. code-block:: python

   cpdef burst_into_memory_c():
       """
       Stores the burst effect in memory for later use.
       Helps in precomputing effects for better performance.
       """

.. code-block:: python

   cpdef rebuild_from_memory_c():
       """
       Rebuilds a surface from previously stored exploded pieces.
       Useful for reusing precomputed effects efficiently.
       """

.. code-block:: python

   cpdef burst_experimental_c():
       """
       A variant of `burst_c` with additional or alternative rendering techniques.
       """

.. code-block:: python

   cpdef db_experimental_c():
       """
       An experimental function that likely applies blending or cleaning effects to a surface.
       """

.. code-block:: python

   cpdef rff_experimental_c():
       """
       Rebuilds an exploded surface from a specific frame with potential experimental logic.
       """

.. code-block:: python

   cpdef rfm_experimental_c():
       """
       Another experimental function focusing on surface restoration and rendering techniques.
       """

.. code-block:: python

   cpdef build_surface_inplace():
       """
       Splits a surface into smaller blocks within a sprite group.
       Used for managing sprites at a granular level.
       """

.. code-block:: python

   cpdef build_surface_inplace_fast():
       """
       An optimized version of `build_surface_inplace` with improved performance.
       Suitable for real-time applications requiring minimal processing time.
       """

6. Summary
----------

This library is designed to enhance the capabilities of graphical applications, particularly
in game development and advanced image manipulation. By offering optimized functions for handling
surfaces, splitting images, applying burst effects, and rebuilding images, it provides a flexible
and efficient toolset for developers. Experimental functions add further possibilities for exploring
novel rendering techniques. The library is a valuable resource for those looking to implement complex
graphical transformations efficiently.

7. Cython method list
---------------------

|

.. function:: pixel_block_rgb(array_, start_x, start_y, w, h, block)

   |

   **Extract a block of pixels from a larger 2D array representing an image.**

   This function retrieves a rectangular block of pixels from a larger image
   (or sprite sheet) and stores it into a smaller block array. The function
   delegates the actual pixel extraction to the `pixel_block_rgb_c` helper
   function for performance optimization.

   **Parameters**

   **array_**
       unsigned char; A 3D array representing the full image (height x width x 3).
       This is the source from which the pixel block will be extracted.

   **start_x**
       int; The x-coordinate of the top-left corner of the block to extract.
       This defines where the block starts in the source image.

   **start_y**
       int; The y-coordinate of the top-left corner of the block to extract.
       This defines where the block starts in the source image.

   **w**
       int; The width of the block to extract. The block will be `w` pixels wide.

   **h**
       int; The height of the block to extract. The block will be `h` pixels tall.

   **block**
       unsigned char; A pre-allocated empty block of size `w x h x 3`.
       This array will be filled with the pixel data from the source image.

   **Returns**

   **unsigned char[:, :, :]**
       A 3D array (height x width x 3) representing the extracted pixel block.
       The block is stored in the `block` parameter.


.. function:: surface_split(surface_, size_, rows_, columns_)

   |

   **Split a pygame.Surface into smaller subsurfaces.**

   This function divides a larger surface into smaller subsurfaces (blocks)
   based on the specified size, number of rows, and columns. It delegates
   the actual splitting to the `surface_split_c` helper function for efficient processing.

   **Parameters**

   **surface_**
       pygame.Surface; The surface to be split into smaller subsurfaces.

   **size_**
       int; The size (width and height) of each subsurface. The resulting
       subsurfaces will be square with dimensions `size_ x size_`.

   **rows_**
       int; The number of rows of subsurfaces to create. This determines
       how many horizontal blocks will be extracted from the original surface.

   **columns_**
       int; The number of columns of subsurfaces to create. This determines
       how many vertical blocks will be extracted from the original surface.

   **Returns**

   **list**
       A list of pygame.Surface objects, each representing a smaller
       block of the original surface.


.. function:: burst(image_, vertex_array_, block_size_, rows_, columns_, x_, y_, max_angle_=0)

   |

   **Simulate an explosion effect on a surface by splitting it into smaller subsurfaces or pixels.**

   This function breaks a given surface into multiple smaller blocks or pixels and applies
   an explosion effect, projecting them in random directions. The subsurfaces or pixels
   are placed at specified positions and can be projected at different angles.

   **Parameters**

   **image_**
       pygame.Surface; The surface to be split into smaller subsurfaces or pixels.

   **vertex_array_**
       list; A list that will be populated with the created sprites, each representing a
       smaller subsurface or pixel. These sprites will have attributes like position,
       velocity (vector), and other states related to the explosion.

   **block_size_**
       int; The size of each block (subsurface). If set to 1, each block will be a single pixel.

   **rows_**
       int; The number of rows of subsurfaces or pixels to create. Determines how many
       horizontal blocks will be created.

   **columns_**
       int; The number of columns of subsurfaces or pixels to create. Determines how many
       vertical blocks will be created.

   **x_**
       int; The x-coordinate of the top-left corner of the surface before the explosion effect.

   **y_**
       int; The y-coordinate of the top-left corner of the surface before the explosion effect.

   **max_angle_**
       int, optional; The maximum angle (in degrees) at which each subsurface/pixel will be
       projected. If not provided, defaults to 0 (no angle applied)

.. py:method:: display_burst(object screen_, list vertex_array_, unsigned char blend_=0)

    |

    Display an explosion effect on the screen by rendering the burst of sprites.

    This function iterates through the list of sprites (created by the burst effect),
    rendering each sprite onto the provided screen surface. The sprites are drawn
    with optional blending effects for smoother transitions, such as additive blending.

    **Parameters**

    **screen_**:
        pygame.Surface; The screen (or rendering surface) where the explosion effect
        (burst of sprites) will be drawn.

    **vertex_array_**:
        list; A list of sprite objects that represent the subsurfaces or pixels created
        during the explosion effect. These sprites will be rendered on the screen.

    **blend_**:
        unsigned char, optional; The blending mode to apply when drawing the sprites.
        Default is 0 (no blending). A non-zero value may specify an additive or
        other blending mode depending on how it's defined.

    **Returns**
    void; The function draws each sprite from `vertex_array_` to the `screen_` surface.

.. py:method:: rebuild_from_frame(object screen_, unsigned int current_frame_, unsigned int start_frame, list vertex_array_, unsigned char blend_ = 0)

    |

    Rebuild and render an explosion effect from a specific frame.

    This function rebuilds and renders an explosion effect (created from sprites) on the
    screen starting from a specified frame number. The sprites are progressively
    re-rendered from their original positions and updated based on the `current_frame_`.
    The function also supports blending for smoother visual transitions during the effect.

    **Parameters**

    **screen_**:
        pygame.Surface; The surface (or screen) where the rebuilt explosion effect will
        be drawn.

    **current_frame_**:
        unsigned int; The current frame number, used to control the timing and progression
        of the explosion effect.

    **start_frame**:
        unsigned int; The frame number at which to start rebuilding the explosion effect.
        The process will only begin after the `current_frame_` exceeds this value.

    **vertex_array_**:
        list; A list of sprite objects representing the subsurfaces or pixels of the
        explosion. These sprites will be rendered onto the screen.

    **blend_**:
        unsigned char, optional; The blending mode to use when rendering the sprites.
        The default is 0 (no blending). A non-zero value will apply the specified blend mode
        (e.g., additive blending).

    **Returns**
    void; This function updates the screen by rendering the sprites based on the
    specified frame and blending settings.

.. py:method:: burst_into_memory(unsigned int n_, object sg_, object screenrect, bint warn_ = False, bint auto_n_ = False)

    |

    Burst a sprite group into memory, creating an explosion effect.

    This function processes a sprite group (`sg_`) and generates an explosion effect by
    bursting it into memory. The function creates a set of sprites or subsurfaces that
    are distributed across a specified region of the screen defined by `screenrect`.
    The number of sprites (`n_`) determines how many parts the sprite group will be
    divided into. It also supports warnings and automatic configuration for the number
    of elements to process.

    **Parameters**

    **n_**:
        unsigned int; The number of elements to burst into memory. This controls
        how many parts the sprite group will be split into for the explosion effect.

    **sg_**:
        object; A pygame.sprite.Group containing the sprites to be processed and
        exploded into memory.

    **screenrect**:
        object; The pygame.Rect object that defines the area of the screen where
        the explosion effect will be displayed. This helps control the positioning
        of the sprites within the screen area.

    **warn_**:
        bint, optional; A flag to trigger warnings if certain conditions are met
        (e.g., if the sprite group is too large to process efficiently). Defaults to False.

    **auto_n_**:
        bint, optional; A flag that, if set to True, automatically determines the
        number of elements to burst based on the sprite group’s size. Defaults to False.

    **Returns**
    void; This function modifies the memory by creating and storing the explosion
    effect, which can then be displayed on the screen.

.. py:method:: rebuild_from_memory(object screen_, list vertex_array_, unsigned char blend_ = 0)

    |

    Rebuild and render an image from stored memory data.

    This function takes a list of vertex data (`vertex_array_`) and reconstructs
    an image or animation by rendering the stored memory information onto the
    provided screen. The image is rendered using the specified blending mode
    (`blend_`). This can be used for reconstructing exploded or fragmented
    visual effects from previously stored memory.

    **Parameters**

    **screen_**:
        object; The surface or screen where the rebuilt image will be rendered.
        Typically, this would be a pygame.Surface representing the display window.

    **vertex_array_**:
        list; A list containing the vertex data (e.g., pixels, sprite blocks,
        or fragmented image parts) used to reconstruct the image. This data
        is used to render the final visual output on the screen.

    **blend_**:
        unsigned char, optional; The blend mode to be used when rendering the
        image. A blend mode of 0 typically represents no blending (normal mode),
        while other values may represent different additive or subtractive blending
        effects. Defaults to 0 (no blending).

    **Returns**
    void; This function modifies the screen object by rendering the rebuilt
    image from the provided vertex data without returning a value.

.. py:method:: burst_experimental(object render_, object image_, object group_, int block_size_, int rows_, int columns_, int x_, int y_, int max_angle_ = 359)

    |

    Explode a surface into multiple subsurfaces or pixels and apply random projections.

    This function takes a single surface (`image_`) and splits it into multiple blocks
    or individual pixels, based on the specified block size, number of rows, and columns.
    Each resulting block or pixel is assigned a random angle of projection, giving the effect
    of the surface being 'exploded' into separate pieces. The pieces are added to the specified
    sprite group (`group_`) for further manipulation or rendering.

    **Parameters**

    **render_**:
        object; The 2D rendering context used for rendering the exploded pieces.
        Typically a pygame.Surface or similar rendering context.

    **image_**:
        object; The pygame.Surface to be split into smaller blocks or pixels.
        This image will be 'exploded' into multiple parts based on the block size.

    **group_**:
        object; A pygame.sprite.Group that will contain the resulting pieces
        (subsurfaces or pixels) after the explosion. Each part of the exploded image
        will be added as a sprite to this group.

    **block_size_**:
        int; The size of each block or pixel in the explosion. This defines
        the width and height of each subsurface or pixel entity. A block size of 1
        will result in individual pixels being created.

    **rows_**:
        int; The number of rows into which the surface will be divided. This, combined
        with the block size, determines how many blocks are created along the vertical axis.

    **columns_**:
        int; The number of columns into which the surface will be divided. This, combined
        with the block size, determines how many blocks are created along the horizontal axis.

    **x_**:
        int; The starting x-coordinate of the surface, representing the top-left corner
        before the explosion.

    **y_**:
        int; The starting y-coordinate of the surface, representing the top-left corner
        before the explosion.

    **max_angle_**:
        int, optional; The maximum angle (in degrees) for the projection of each exploded
        piece. Each piece will be projected at a random angle between 0 and `max_angle_`.
        Defaults to 359 degrees (full circle).

    **Returns**
    void; This function modifies the `group_` object by adding the exploded pieces as
    sprites, but does not return any value.


.. py:method:: db_experimental(object screenrect_, object render_, object group_, bint clean_ = False, unsigned char blend_ = 0)

    |

    Perform an experimental database effect with a sprite group and optional blending.

    This function applies an experimental effect to a sprite group (`group_`), rendering
    it within a specified screen region (`screenrect_`). The effect is optionally blended
    based on the `blend_` value. The function can also optionally clean up the group (`clean_`),
    removing or resetting elements as necessary.

    **Parameters**

    **screenrect_**:
        object; A pygame.Rect that defines the region of the screen to render the effect.

    **render_**:
        object; The 2D rendering context (usually a pygame.Surface) where the effect will be rendered.

    **group_**:
        object; A pygame.sprite.Group that contains the sprites to be affected by the experimental effect.

    **clean_**:
        bint, optional; A flag that determines whether the group will be cleaned (reset or cleared) before rendering. Defaults to False.

    **blend_**:
        unsigned char, optional; The blending mode to be used when rendering the effect. A blend mode of 0 typically represents no blending (normal mode). Defaults to 0.

    **Returns**
    void; This function modifies the `group_` object by rendering or cleaning the sprites, but does not return any value.

.. py:method:: rff_experimental(object render_, object screenrect_, unsigned int current_frame_, unsigned int start_frame, object group_, unsigned char blend_ = 0)

    |

    Rebuild an exploded image from a specific frame number.

    This function starts rebuilding an exploded image from a given start frame
    number and continues to modify the image until it reaches the current frame.
    It delegates the core logic to the `rff_experimental_c` function for performance.

    The rebuild process involves inverting the direction of each pixel or pixel block's
    vector and drawing them to the render surface, while respecting the screen boundaries.

    **Parameters**

    **render_**:
        Pygame.Surface; The surface where the image will be rendered. This is usually
        the game screen or the target rendering surface.

    **screenrect_**:
        Pygame.Rect; The rectangle defining the screen's area. Used to determine
        if a sprite is within the visible area of the screen.

    **current_frame_**:
        unsigned int; The current frame number. The image will start rebuilding
        from the specified `start_frame` until the `current_frame` is reached.

    **start_frame**:
        unsigned int; The frame number at which the rebuild process starts.
        The image will not be modified before this frame.

    **group_**:
        Pygame.sprite.Group; A group of sprites representing the blocks or
        pixels of the image to be rebuilt. Each sprite is manipulated and
        rendered during the rebuild process.

    **blend_**:
        unsigned char, optional; The blend mode to use when rendering
        the sprites. Blend modes control how the sprite colors are combined
        with the existing surface. Defaults to 0 (no blending).

    **Returns**
    None; This function modifies the `group_` sprites and the surface
    directly. No value is returned.

.. py:method:: rfm_experimental(object screenrect_, object render_, object group_, unsigned char blend_ = 0)

    |

    Rebuild an exploded image with pixels or pixel blocks outside the screen boundaries.

    This function applies an experimental process to rebuild an image using pixels or
    pixel blocks that are outside the boundaries of the screen. It delegates the core
    logic to the `rfm_experimental_c` function for better performance. This process
    is often used for creating effects where the image "explodes" or pieces of it move
    outside the screen area.

    **Parameters**

    **screenrect_**:
        Pygame.Rect; The rectangle representing the boundaries of the screen or
        the game display. Used to check if a sprite is within the screen's area.

    **render_**:
        Pygame.Surface; The surface where the rebuilt image will be rendered. This is
        typically the game display or render target.

    **group_**:
        Pygame.sprite.Group; A group of sprites that represent the pixels or blocks
        being used to rebuild the image. Each sprite in the group will be processed
        and blitted onto the surface.

    **blend_**:
        unsigned char, optional; A blend mode to be used when rendering
        the sprites. Blend modes control how the colors of the sprites are combined
        with the existing colors on the surface. Defaults to 0 (no blending).

    **Returns**
    None; This function modifies the surface directly and does not return any
    value. It also modifies the group of sprites in place.

.. py:method:: build_surface_inplace(object surface_, object group_, unsigned int block_width, unsigned int block_height)

    |

    Rebuild a Pygame surface from a group of sprites (inplace).

    This function takes a group of sprites and blits their images onto the specified
    surface at their respective positions, updating the surface in place. The process
    modifies the surface directly without creating a new one.

    It delegates the task to the `build_surface_inplace_c` function, which is written
    in C for improved performance.

    **Parameters**

    **surface_**:
        Pygame.Surface; The surface that will be updated by blitting the sprites onto it.
        The original surface is modified directly, and no new surface is created.

    **group_**:
        Pygame.sprite.Group; A group of sprites to be blitted onto the surface. Each sprite's
        position is determined by its `rect` attribute.

    **block_width**:
        unsigned int; The width of the block used for the tiling or reconstruction process.
        It can influence how the surface is rebuilt.

    **block_height**:
        unsigned int; The height of the block used for the tiling or reconstruction process.
        It works in conjunction with `block_width` to determine the surface layout.

    **Returns**
    None; This function modifies the surface directly and does not return any value.

.. py:method:: build_surface_inplace_fast(object surface_, object group_, unsigned int block_width, unsigned int block_height)

    |

    Efficiently rebuild a Pygame surface from a group of sprites (inplace).

    This function rapidly constructs a surface by blitting individual sprite images
    onto the given surface at their respective positions. The process is performed
    in place, meaning the surface is directly updated without creating a new surface.

    It delegates the actual work to the optimized `build_surface_inplace_fast_c`
    function for performance.

    **Parameters**

    **surface_**:
        Pygame.Surface; The surface to which the sprites will be blitted.
        This surface is modified directly.

    **group_**:
        Pygame.sprite.Group; The group of sprites whose images will be blitted onto
        the surface. Each sprite's position is taken from its `rect` attribute.

    **block_width**:
        unsigned int; The width of the blocks that will be used to build the surface.
        This is important for any tiling logic and might impact performance depending
        on the block size.

    **block_height**:
        unsigned int; The height of the blocks that will be used to build the surface.
        Similar to `block_width`, it affects how the surface is reconstructed.

    **Returns**
    None; The surface is modified directly, so there is no return value.
