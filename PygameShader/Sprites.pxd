
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, optimize.use_switch=True, initializedcheck=False
# encoding: utf-8

"""
=============================
Sprite and Group Management
=============================

This library provides an efficient Cython implementation of **sprite management, collision detection, and rendering**
similar to `pygame.sprite`, but optimized for performance. This functionality already exists in the **Pygame project**,
but has been **Cythonized** to significantly improve performance by using C-level optimizations for common operations.
It includes:

- **Sprite Objects** (`Sprite` class)
- **Group Handling** (`AbstractGroup`, `GroupSingle`, `LayeredUpdates`)
- **Collision Detection** (`collide_rect`, `collide_circle`, `collide_mask`)
- **Layered Sprite Management** (`LayeredUpdates`, `LayeredUpdatesModified`)

**Purpose**
-----------
This module enhances **Pygame's sprite system** by providing:
- Faster **group operations** with optimized internal structures.
- Efficient **collision detection** using various methods (rectangles, circles, masks).
- Advanced **layer-based rendering** for handling depth ordering.
- Support for **single-object sprite groups** (e.g., `GroupSingle`).
- **In-place updates** to minimize memory allocations.

The core functionality is inspired by the **Pygame sprite module** but is **Cythonized** for better performance. It takes advantage of
Cython's ability to compile code into C to achieve faster execution and reduced memory usage, making it suitable for performance-critical games.

---

Sprite Class
------------
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

---

AbstractGroup
-------------
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

---

GroupSingle
-----------
**`GroupSingle`** is a specialized group that holds only a **single sprite**.

**Methods**
- `sprites() -> list`: Returns a list containing the single sprite.
- `add_internal(sprite)`: Sets the sprite for this group.
- `remove_internal(sprite)`: Removes the sprite.
- `has_internal(sprite) -> bool`: Checks if a sprite exists in the group.

---

Collision Detection
-------------------
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

---

Layered Sprite Management
-------------------------
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

---

LayeredUpdatesModified
----------------------
Extends `LayeredUpdates` with additional drawing and update functionality.

- `update(args=*)`: Updates all sprites.
- `draw(surface)`: Draws sprites onto a surface.

---

Helper Functions
----------------
- `int_min(a, b) -> int`: Returns the smaller of two integers.
- `truth(a) -> bool`: Converts a value to boolean.

---

**Summary**
-----------
This library optimizes **sprite handling** in Pygame by:
- **Enhancing performance** with Cython memoryviews.
- **Supporting efficient collision detection** (rect, circle, mask).
- **Managing layered rendering** with advanced depth ordering.
- **Providing various group structures** (standard, single, layered).

Although this functionality is available in the **Pygame project**, this library has been **Cythonized** to provide **significant performance improvements**, making it ideal for **performance-critical games**.

**Ideal for Pygame developers needing fast and efficient sprite management.** 🚀


"""

cdef int int_min(int a, int b)


cdef truth(bint a)

cdef class Sprite(object):

    cdef dict __dict__
    cdef dict __g
    cpdef add_internal(self, object group)

    cpdef remove_internal(self, object group)

    cpdef update(self, args=*)

    cpdef kill(self)

    cpdef list groups(self)

    cpdef bint alive(self)


cdef class AbstractGroup(object):

    cdef:
        public bint _spritegroup
        public dict spritedict
        public list lostsprites

    cpdef list sprites(self)

    cpdef void add_internal(self, sprite)

    cpdef void remove_internal(self, sprite)

    cpdef bint has_internal(self, sprite)

    cpdef copy(self)

    cdef void update(self, args=*)

    cpdef draw(self, object surface)

    cpdef void clear(self, object surface, object bgd)

    cpdef void empty(self)


cdef class GroupSingle(AbstractGroup):

    cdef public object __sprite

    cpdef copy(self)

    cpdef list sprites(self)

    cpdef void add_internal(self, sprite)

    cpdef _get_sprite(self)

    cpdef _set_sprite(self, sprite)

    cpdef void remove_internal(self, sprite)

    cpdef bint has_internal(self, sprite)



cdef collide_rect(left, right)

cdef class collide_rect_ratio(object):
    cdef float ratio


cdef collide_circle(left, right)


cdef dict groupcollide(groupa, groupb, bint dokilla, bint dokillb, collided=*)
cdef spritecollideany(sprite, group, collided=*)
cdef list spritecollide(sprite, group, bint dokill, collided=*)

cdef bint collide_mask(left, right)

cdef class collide_circle_ratio(object):
    cdef float ratio


cdef class LayeredUpdates(AbstractGroup):

    cdef:
        public dict _spritelayers
        public list _spritelist
        public int _default_layer
        public object _init_rect

    cpdef void add_internal(self, sprite, layer_=*)
    cpdef void remove_internal(self, sprite)
    cpdef list sprites(self)
    cpdef get_sprites_at(self, pos)
    cpdef get_sprite(self, int idx)
    cpdef remove_sprites_of_layer(self, int layer_nr)
    cpdef layers(self)
    cpdef change_layer(self, sprite, new_layer)
    cpdef get_layer_of_sprite(self, sprite)
    cpdef get_top_layer(self)
    cpdef get_bottom_layer(self)
    cpdef move_to_front(self, sprite)
    cpdef move_to_back(self, sprite)
    cpdef get_top_sprite(self)
    cpdef get_sprites_from_layer(self, layer)
    cpdef switch_layer(self, layer1_nr, layer2_nr)

cdef class LayeredUpdatesModified(LayeredUpdates):

    cpdef void update(self, args=*)
    cpdef draw(self, surface_)

