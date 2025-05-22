Sprites
========================================

:mod:`Sprites.pyx`

=====================

.. currentmodule:: Sprites

|


1. Sprite and Group Management
------------------------------

This library provides an efficient Cython implementation of **sprite management, collision detection, and rendering**
similar to `pygame.sprite`, but optimized for performance. This functionality already exists in the **Pygame project**,
but has been **Cythonized** to significantly improve performance by using C-level optimizations for common operations.
It includes:

- **Sprite Objects** (`Sprite` class)
- **Group Handling** (`AbstractGroup`, `GroupSingle`, `LayeredUpdates`)
- **Collision Detection** (`collide_rect`, `collide_circle`, `collide_mask`)
- **Layered Sprite Management** (`LayeredUpdates`, `LayeredUpdatesModified`)

2. Purpose
----------

This module enhances **Pygame's sprite system** by providing:
- Faster **group operations** with optimized internal structures.
- Efficient **collision detection** using various methods (rectangles, circles, masks).
- Advanced **layer-based rendering** for handling depth ordering.
- Support for **single-object sprite groups** (e.g., `GroupSingle`).
- **In-place updates** to minimize memory allocations.

The core functionality is inspired by the **Pygame sprite module** but is **Cythonized** for better performance. It takes advantage of
Cython's ability to compile code into C to achieve faster execution and reduced memory usage, making it suitable for performance-critical games.


3. Sprite Class
---------------

**`Sprite`** represents an individual game object that can belong to multiple groups.

.. code-block:: python

    sprite = Sprite()
    group = AbstractGroup()
    sprite.add_internal(group)

**Methods**
- `add_internal(group)`: Adds the sprite to a group.
- `remove_internal(group)`: Removes the sprite from a group.
- `update(args=*)`: Updates the sprite state.
- `kill()`: Removes the sprite from all groups.
- `groups() -> list`: Returns all groups containing this sprite.
- `alive() -> bool`: Checks if the sprite is still in any group.


**AbstractGroup**

**`AbstractGroup`** is the base class for managing sprite groups.

**Attributes**
- `_spritegroup (bool)`: Indicates whether this is a sprite group.
- `spritedict (dict)`: Stores sprites and their data.
- `lostsprites (list)`: Tracks removed sprites.

**Methods**
- `sprites() -> list`: Returns a list of all sprites in the group.
- `add_internal(sprite)`: Adds a sprite to the group.
- `remove_internal(sprite)`: Removes a sprite from the group.
- `has_internal(sprite) -> bool`: Checks if a sprite is in the group.
- `copy()`: Creates a copy of the group.
- `update(args=*)`: Calls the `update()` method on all sprites.
- `draw(surface)`: Draws all sprites onto the given surface.
- `clear(surface, bgd)`: Clears the group from the screen.
- `empty()`: Removes all sprites from the group.


**GroupSingle**

**`GroupSingle`** is a specialized group that holds only a **single sprite**.

**Methods**
- `sprites() -> list`: Returns a list containing the single sprite.
- `add_internal(sprite)`: Sets the sprite for this group.
- `remove_internal(sprite)`: Removes the sprite.
- `has_internal(sprite) -> bool`: Checks if a sprite exists in the group.



**Collision Detection**

**Collision functions** allow efficient detection between sprites and groups.

- `collide_rect(left, right)`: Rectangular collision detection.
- `collide_circle(left, right)`: Circular collision detection.
- `collide_mask(left, right)`: Pixel-perfect collision detection.
- `groupcollide(groupa, groupb, dokilla, dokillb, collided=*) -> dict`:
  - Checks collisions between two groups, optionally removing colliding sprites.
- `spritecollideany(sprite, group, collided=*)`:
  - Checks if a sprite collides with any sprite in a group.
- `spritecollide(sprite, group, dokill, collided=*) -> list`:
  - Returns a list of sprites that collide with `sprite`.



**Layered Sprite Management**

**`LayeredUpdates`** extends `AbstractGroup` to support **layer-based rendering**.

**Attributes**

- `_spritelayers (dict)`: Stores sprite-layer mappings.
- `_spritelist (list)`: Ordered list of sprites.
- `_default_layer (int)`: Default layer for new sprites.

**Methods**

- `add_internal(sprite, layer=*)`: Adds a sprite to a specified layer.
- `remove_internal(sprite)`: Removes a sprite.
- `sprites() -> list`: Returns all sprites.
- `get_sprites_at(pos)`: Retrieves sprites at a given position.
- `get_sprite(idx)`: Returns a sprite by index.
- `remove_sprites_of_layer(layer)`: Removes all sprites from a specific layer.
- `layers()`: Returns a list of all layers.
- `change_layer(sprite, new_layer)`: Moves a sprite to a new layer.
- `get_layer_of_sprite(sprite)`: Returns the layer of a given sprite.
- `get_top_layer() / get_bottom_layer()`: Returns the highest or lowest layer.
- `move_to_front(sprite) / move_to_back(sprite)`: Changes sprite depth ordering.
- `get_top_sprite()`: Retrieves the topmost sprite.
- `get_sprites_from_layer(layer)`: Returns all sprites in a given layer.
- `switch_layer(layer1, layer2)`: Swaps two layers.

**LayeredUpdatesModified**

Extends `LayeredUpdates` with additional drawing and update functionality.

- `update(args=*)`: Updates all sprites.
- `draw(surface)`: Draws sprites onto a surface.


**Helper Functions**

- `int_min(a, b) -> int`: Returns the smaller of two integers.
- `truth(a) -> bool`: Converts a value to boolean.



4. Summary
----------

This library optimizes **sprite handling** in Pygame by:
- **Enhancing performance** with Cython memoryviews.
- **Supporting efficient collision detection** (rect, circle, mask).
- **Managing layered rendering** with advanced depth ordering.
- **Providing various group structures** (standard, single, layered).

Although this functionality is available in the **Pygame project**, this library has been **Cythonized** to provide **significant performance improvements**, making it ideal for **performance-critical games**.

**Ideal for Pygame developers needing fast and efficient sprite management.** 🚀
