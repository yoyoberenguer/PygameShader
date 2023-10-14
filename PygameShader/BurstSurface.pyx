# cython: binding=False, boundscheck=False, wraparound=False, nonecheck=False, cdivision=True,
# profile=False, initializedcheck=False
# cython: optimize.use_switch=True
# cython: warn.maybe_uninitialized=False
# cython: warn.unused=False
# cython: warn.unused_result=False
# cython: warn.unused_arg=False
# cython: language_level=3
# encoding: utf-8

"""
                 GNU GENERAL PUBLIC LICENSE
                       Version 3, 29 June 2007

 Copyright (C) 2007 Free Software Foundation, Inc. <https://fsf.org/>
 Everyone is permitted to copy and distribute verbatim copies
 of this license document, but changing it is not allowed.

Copyright Yoann Berenguer
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
cpdef unsigned char [:, :, ::1] pixel_block_rgb(
        unsigned char [:, :, :] array_, int start_x, int start_y,
        int w, int h, unsigned char [:, :, ::1] block) nogil:

        return pixel_block_rgb_c(array_, start_x, start_y, w, h, block)

@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cpdef surface_split(object surface_, int size_, int rows_, int columns_):
    return surface_split_c(surface_, size_, rows_, columns_)


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cpdef void burst(image_, list vertex_array_, int block_size_, int rows_,
          int columns_, int x_, int y_, int max_angle_=0):
    burst_c(image_, vertex_array_, block_size_, rows_,
          columns_, x_, y_, max_angle_)

@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cpdef void display_burst(object screen_, list vertex_array_, unsigned char blend_=0):
    display_burst_c(screen_, vertex_array_, blend_)

@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cpdef void rebuild_from_frame(
        object screen_,
        unsigned int current_frame_,
        unsigned int start_frame,
        list vertex_array_,
        unsigned char blend_ = 0):

    rebuild_from_frame_c(screen_, current_frame_, start_frame, vertex_array_, blend_)

@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cpdef void burst_into_memory(
        unsigned int n_,
        object sg_,
        object screenrect,
        bint warn_ = False,
        bint auto_n_ = False):
    burst_into_memory_c(n_, sg_, screenrect, warn_, auto_n_)

@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cpdef void rebuild_from_memory(object screen_, list vertex_array_, unsigned char blend_=0):
    rebuild_from_memory_c(screen_, vertex_array_, blend_)


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cpdef void burst_experimental(
        object render_,
        object image_,
        object group_,
        int block_size_,
        int rows_,
        int columns_,
        int x_, int y_,
        int
        max_angle_=359):

    burst_experimental_c(
        render_,
        image_,
        group_,
        block_size_,
        rows_,
        columns_,
        x_, y_,
        max_angle_)

@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
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
cpdef void rff_experimental(
        object render_,
        object screenrect_,
        unsigned int current_frame_,
        unsigned int start_frame,
        object group_,
        unsigned char blend_ = 0):
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
cpdef void rfm_experimental(
        object screenrect_,
        object render_,
        object group_,
        unsigned char blend_=0):

    rfm_experimental_c(
        screenrect_,
        render_,
        group_,
        blend_)


cpdef void build_surface_inplace(
        object surface_,
        object group_,
        unsigned int block_width,
        unsigned int block_height
):
    build_surface_inplace_c(surface_, group_, block_width, block_height)

cpdef void build_surface_inplace_fast(
        object surface_,
        object group_,
        unsigned int block_width,
        unsigned int block_height
):
    build_surface_inplace_fast_c(surface_, group_, block_width, block_height)

# -------------------------------------- INTERFACE -------------------------------------

@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cdef inline unsigned char [:, :, ::1] pixel_block_rgb_c(
        unsigned char [:, :, :] array_, int start_x, int start_y,
        int w, int h, unsigned char [:, :, ::1] block) nogil:
    """
    EXTRACT A SPRITE FROM A SPRITE SHEET 

    * Method used by Sprite_Sheet_Uniform_RGB in order to extract all the sprites from 
    the sprite sheet
    * This method returns a memoryview type [:, :, ::1] contiguous of unsigned char 
    (sprite of size w x h)

    :param array_ : unsigned char; array of size w x h x 3 to parse into sub blocks
     (non contiguous)
    :param start_x: int; start of the block (x value) 
    :param start_y: int; start of the block (y value)
    :param w      : int; width of the block
    :param h      : int; height of the block
    :param block  : unsigned char; empty block of size w_n x h_n x 3 to fill up 
    :return       : Return 3d array of size (w_n x h_n x 3) of RGB pixels 
    """

    cdef:
        int x, y, xx, yy

    for x in prange(w, schedule=SCHEDULE, num_threads=THREADS):
        xx = start_x + x
        for y in range(h):
            yy = start_y + y

            block[y, x, 0] = array_[xx, yy, 0]
            block[y, x, 1] = array_[xx, yy, 1]
            block[y, x, 2] = array_[xx, yy, 2]

    return block

@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cdef surface_split_c(surface_, int size_, int rows_, int columns_):

    cdef:
        unsigned int w, h

    w, h = surface_.get_size()

    cdef:

        unsigned char [:, :, :] rgb_array = surface_.get_view('3')
        list subsurface = []
        int rows, columns
        int start_x, end_x, start_y, end_y;
        int width = <object>rgb_array.shape[1]
        int height = <object>rgb_array.shape[0]

    cdef:
        unsigned char [:, :, ::1] empty_array = empty((size_, size_, 3), uint8)
        unsigned char [:, :, ::1] block_array = empty((size_, size_, 3), uint8)

    with nogil:
        for rows in range(rows_):
            start_y = rows * size_
            end_y   = (rows + 1) * size_
            for columns in range(columns_):
                start_x = columns * size_
                end_x   = start_x + size_
                block_array = pixel_block_rgb_c(
                    rgb_array, start_x, start_y, size_, size_, empty_array)
                with gil:
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
cdef bint is_power_of_two(int n)nogil:
    return (n != 0) and (n & (n-1) == 0)

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
cdef void burst_experimental_c(
        object render_,
        object image_,
        object group_,
        int block_size_,
        int rows_,
        int columns_,
        int x_, int y_,
        int
        max_angle_=0):
    """
    EXPLODE A SURFACE IN MULTIPLE SUBSURFACE OR PIXELS
    Module using the _sdl pygame library
    
    :param render_    : Render: _sdl 2D rendering context for a window (experimental)
    :param image_     : pygame.Surface; Surface to transform into multiple subsurface or pixels
    :param group_     :python list; 
    :param block_size_: integer; Size of the subsurface (square entity w x h) or size 1 for 
    a pixel entity
    :param rows_      : integer; number of rows
    :param columns_   : integer; number of columns
    :param x_         : integer; position x of the surface (top left corner) before the blast
    :param y_         : integer; position y of the surface (top left corner)
    :param max_angle_ : integer; Angle projection of the particle/pixel/subsurface in the
     2d cartesian plan
    """

    assert PyObject_IsInstance(image_, Surface), \
        "\nAttribute image_ must be a pygame.Surface type got %s " % type(image_)
    assert PyObject_IsInstance(group_, Group), \
        "\nAttribute group_ must be a pygame.sprite.Group type got %s " % type(image_)
    assert is_power_of_two(block_size_) and block_size_>0, \
        "\nAttribute block_size_ must be a power of two"
    assert rows_ > 0, "\nAttribute row_ must be > 0"
    assert columns_ > 0,"\nAttribute columns_ must be > 0"

    max_angle_ %=360

    cdef:
        list subsurface = surface_split_c(
            image_, block_size_, rows_, columns_)
        int n = 0
        float random_angle

    r = Rect(x_, y_, columns_ * block_size_, rows_ * block_size_)

    with cython.cdivision(True):

        for surf in subsurface:
            s = pygame.sprite.Sprite()
            s.image = Image(Texture.from_surface(render_, surf))

            s.rect = surf.get_rect(
                topleft=((n % columns_) * block_size_ + x_,
                         <int>(n / columns_) * block_size_ + y_))

            random_angle = randRangeFloat(<float>0.0, max_angle_)
            s.vector = Vector2(<float>cosf(random_angle),
                               <float>sinf(random_angle)) * \
                       randRangeFloat(<float>5.0, <float>10.0)
            n += 1
            s.org = (s.rect.topleft[0], s.rect.topleft[1])
            s.counter = 0
            s.stop = False
            s.rebuild_state = False
            group_.add(s)



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
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
cdef void rff_experimental_c(
        object render_,
        object screenrect_,
        unsigned int current_frame_,
        unsigned int start_frame,
        object group_,
        unsigned char blend_ = 0
):
    """
    (RFF_EXPERIMENTAL) REBUILD FROM FRAME (REBUILD AN EXPLODED IMAGE FROM A SPECIFIC FRAME NUMBER) 

    The build process will start from a given frame number

    :param render_ : Surface; game display
    :param screenrect_: Pygame Rect (display)
    :param current_frame_: unsigned int; Current frame value
    :param start_frame : unsigned int; Frame number when image is starting rebuilding
    :param group_: python Sprite Group containing objects (pixels or blocks of pixels) set with 
       pre-defined attributes and values.
    :param blend_: unsigned char; blend mode (additive mode)
    :return: void
    """
    if start_frame == 0:
        raise ValueError("\nYou must specify a start frame number != 0")


    cdef:
        render_blit = render_.blit
        int s_org0, s_org1

    screenrect = screenrect_

    for s in group_:

        if current_frame_ > start_frame and not s.rebuild_state:
            # INVERT VECTOR TO REBUILD IMAGE
            s.vector = -s.vector
            s.rebuild_state = True

        s_rect = s.rect
        s_vector = s.vector
        s_rect.topleft += s_vector

        # START TO CHECK DISTANCE ONLY WHEN THE FRAME IS > START_FRAME
        # OTHERWISE THE DISTANCE WILL BE <2 WHEN THE PROCESS BEGIN
        if current_frame_ > start_frame and not s.stop:

            s_org0, s_org1 = s.org[ 0 ], s.org[ 1 ]

            if <float> sqrtf((s_org0 - s.rect.topleft[ 0 ]) ** 2
                            + (s_org1 - s.rect.topleft[ 1 ]) ** 2) < <float> 2.0:
                s.vector = Vector2(<float> 0.0, <float> 0.0)
                s.rect.topleft = (s_org0, s_org1)
                s.stop = True

        if screenrect.contains(s_rect):
            render_blit(s.image, s.rect, special_flags=blend_)

@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cdef void rfm_experimental_c(
        object screenrect_,
        object render_,
        object group_,
        unsigned char blend_=0
):
    """
    (RFM_EXPERIMENTAL REBUILD FROM MEMORY) 
    REBUILD AN IMAGE (EXPLODED PIXELS OR PIXEL'S BLOCK BEING OUTSIDE SCREEN BOUNDARIES)
    
    Module using the _sdl pygame library

    :param screenrect_: Pygame.Surface object; Game display 
    :param render_: list; python list containing objects (pixels or blocks of pixels) 
        set with pre-defined attributes and values.
    :param group_ : pygame Sprite group containing the sprites
    :param blend_: unsigned char; blend mode (additive mode)
    """

    cdef:
        render_blit = render_.blit
        int s_org0, s_org1, s_0, s_1
        float s_r
        bint s_stop


    screenrect = screenrect_
    screenrect_contains = screenrect.contains

    # ITERATE OVER EVERY BLOCKS
    for s in group_:

        s.rect.topleft -= s.vector
        s_rect = s.rect
        s_stop = s.stop

        # CHECK THE BLOCK STATUS,
        if not s_stop:

            s_org0, s_org1 = s.org[0], s.org[1]
            s_0, s_1 = s_rect.topleft[0], s_rect.topleft[1]

            # DETERMINE THE DISTANCE FROM ORIGIN
            # AVOIDING SQUARE ROOT TO INCREASE PERFS

            s_r = ((s_org0 - s_0) * (s_org0 - s_0)
                        + (s_org1 - s_1) * (s_org1 - s_1))

            if s_r < <float>8.0:
                s.vector = Vector2(<float>0.0, <float>0.0)
                s_0, s_1 = s_org0, s_org1
                s_stop = True
                # group_.remove(s)

        if screenrect_contains(s_rect):

            # DRAW THE PIXEL BLOCKs
            render_blit(s.image, s_rect, special_flags=blend_)


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cdef void build_surface_inplace_c(
        object surface_,
        object group_,
        unsigned int block_width,
        unsigned int block_height
):
    """
    BUILD A PYGAME SURFACE FROM A SPRITE GROUP (INPLACE)
    """
    cdef:

        unsigned char [:, :, :] array_ = surface_.get_view('3')
        unsigned char [:, :, :] block_ = \
            numpy.empty((block_width, block_height, 3), dtype=numpy.uint8)
        int i, j
        int x_, y_
        int w_, h_
        unsigned char x = 0
        unsigned char ii=0, jj=0
        unsigned int ww = surface_.get_width() - block_width
        unsigned int hh = surface_.get_height() - block_height

    for s in group_:
        # block_ = pixels3d(s.image)
        x_ = s.rect[0]
        y_ = s.rect[1]
        surface_.blit(s.image, s.rect)
        with nogil:
            x_ = 0 if x_ < 0 else x_
            y_ = 0 if y_ < 0 else y_
            if x_>=ww: x_=ww
            if y_>=ww: y_=hh
            # Transfer the pixel block
            w_ = x_ + block_width
            h_ = y_ + block_height
            for i in prange(x_, w_, schedule=SCHEDULE, num_threads=THREADS):
                jj = 0
                x = i - x_
                for j in range(y_, h_):
                    array_[i, j, 0] = block_[x, jj, 0]
                    array_[i, j, 1] = block_[x, jj, 1]
                    array_[i, j, 2] = block_[x, jj, 2]
                    jj = jj + 1


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cdef void build_surface_inplace_fast_c(
        object surface_,
        object group_,
        unsigned int block_width,
        unsigned int block_height
):
    """
    BUILD A PYGAME SURFACE FROM A SPRITE GROUP (INPLACE)
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
cdef void burst_c(image_, list vertex_array_, int block_size_, int rows_,
          int columns_, int x_, int y_, int max_angle_=0):
    """
    EXPLODE A SURFACE IN MULTIPLE SUBSURFACE OR PIXELS

    :param image_     : pygame.Surface; Surface to transform into multiple subsurface or pixels
    :param vertex_array_:python list; 
    :param block_size_: integer; Size of the subsurface (square entity w x h) or size 1 for 
    a pixel entity
    :param rows_      : integer; number of rows
    :param columns_   : integer; number of columns
    :param x_         : integer; position x of the surface (top left corner) before the blast
    :param y_         : integer; position y of the surface (top left corner)
    :param max_angle_ : integer; Angle projection of the particle/pixel/subsurface in the
     2d cartesian plan
    """
    # TODO calculate width x height and rows_ x columns_
    assert PyObject_IsInstance(image_, Surface), \
        "\nAttribute image_ must be a pygame.Surface got %s " % type(image_)
    assert is_power_of_two(block_size_) and block_size_>0, \
        "\nAttribute block_size_ must be a power of two"
    assert rows_ > 0, "\nAttribute row_ must be > 0"
    assert columns_ > 0,"\nAttribute columns_ must be > 0"

    max_angle_ %=360

    cdef:
        list subsurface = surface_split_c(
            image_, block_size_, rows_, columns_)
        int n = 0
        float random_angle

    r = Rect(x_, y_, columns_ * block_size_, rows_ * block_size_)

    with cython.cdivision(True):
        for surf in subsurface:
            s = sprite.Sprite()
            s.image = surf

            s.rect = surf.get_rect(
                topleft=((n % columns_) * block_size_ + x_,
                         <int>(n / columns_) * block_size_ + y_))

            random_angle = randRangeFloat(<float>0.0, max_angle_)
            s.vector = Vector2(<float>cosf(random_angle),
                               <float>sinf(random_angle)) * randRangeFloat(<float>5.0, <float>10.0)
            n += 1
            s.org = (s.rect.topleft[0], s.rect.topleft[1])
            s.counter = 0
            s.stop = False
            s.rebuild_state = False
            vertex_array_.append(s)



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cdef void display_burst_c(object screen_, list vertex_array_, unsigned char blend_=0):
    """
    DISPLAY AN EXPLODED IMAGE 

    This method decrement the pixels/blocks alpha value each frame
    Be aware that using blend_ (BLEND_RGB_ADD etc) will prevent the alpha channel to be modified

    :param screen_: Surface; game display  
    :param vertex_array_: python list containing objects (pixels or blocks of pixels) set with 
       pre-defined attributes and values.
    :param blend_: unsigned char; blend mode (additive mode)
    """

    cdef:
        screen_blit = screen_.blit
        screenrect  = screen_.get_rect()

    for s in vertex_array_:

        if s.rect.colliderect(screenrect):

            s_rect          = s.rect
            s_vector        = s.vector
            s_rect.centerx += s_vector.x
            s_rect.centery += s_vector.y

            screen_blit(s.image, s.rect, special_flags=blend_)

            s.image.set_alpha(max(<unsigned char>255 - s.counter, 0), RLEACCEL)
            s.counter += 2

        else:
            vertex_array_.remove(s)



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cdef void rebuild_from_frame_c(
        object screen_,
        unsigned int current_frame_,
        unsigned int start_frame,
        list vertex_array_,
        unsigned char blend_ = 0
):
    """
    REBUILD AN EXPLODED IMAGE 
    
    The build process will start from a given game frame number

    :param screen_ : Surface; game display
    :param current_frame_: unsigned int; Current frame value
    :param start_frame : unsigned int; Frame number when image is starting rebuilding
    :param vertex_array_: python list containing objects (pixels or blocks of pixels) set with 
       pre-defined attributes and values.
    :param blend_: unsigned char; blend mode (additive mode)
    :return: void
    """
    if start_frame == 0:
        raise ValueError("\nYou must specify a start frame number != 0")


    cdef:
        screen_blit = screen_.blit
        int s_org0, s_org1

    screenrect = screen_.get_rect()

    for s in vertex_array_:

        if current_frame_ > start_frame and not s.rebuild_state:
            # INVERT VECTOR TO REBUILD IMAGE
            s.vector = -s.vector
            s.rebuild_state = True

        s_rect          = s.rect
        s_vector        = s.vector
        s_rect.topleft += s_vector

        # START TO CHECK DISTANCE ONLY WHEN THE FRAME IS > START_FRAME
        # OTHERWISE THE DISTANCE WILL BE <2 WHEN THE PROCESS BEGIN
        if current_frame_ > start_frame and not s.stop:

            s_org0, s_org1 = s.org[0], s.org[1]

            if <float>sqrtf((s_org0 - s.rect.topleft[0]) ** 2
                    + (s_org1 - s.rect.topleft[1]) ** 2) < <float>2.0:
                s.vector = Vector2(<float>0.0, <float>0.0)
                s.rect.topleft = (s_org0, s_org1)
                s.stop = True

        if screenrect.contains(s_rect):
            screen_blit(s.image, s.rect, special_flags=blend_)



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cdef void burst_into_memory_c(
        unsigned int n_,
        object sg_,
        object screenrect,
        bint warn_ = False,
        bint auto_n_ = False
):
    """
    MOVE IMAGE PIXEL BLOCK INTO MEMORY AND MEMORIZED THEIR LOCATIONS AFTER N ITERATIONS

    This method can be used in addition to the method rebuild_from_memory 
    to give the illusion of an image being rebuilt from exploded pixel's blocks.
    
    :param n_: Number of iterations. Each iteration, the pixels or the block of pixels 
       will be moved to a new locations given by the vector value. We are moving the pixels by
       incrementing the top left corner of the block. 

    :param sg_: python list or Sprite group
     containing objects (pixels or blocks of pixels) set with pre-defined attributes and values. 
     Here we are using only two attributes, s.rect and 
       s.vector. The block rectangle limitations and the vector direction.

    :param screenrect: pygame.Rect object; Represent the display rectangle     

    :param warn_: bool; Default False, do not display a warning if one or more pixels are still 
        visible within the game display. This setting will be ignored if auto_n_ is True 

    :param auto_n_: bool; n_ value will be ignore. Auto check if pixels or pixels blocks are 
        still visible within the game space and increment n_ automatically (recursive), this 
        method is the best if you have no clue how many frames will be required to burst an image
        until all the pixels/blocks are outside the screen boundaries. It will also found the most
        effective frame number (lowest frame)

    :return: void
    """
    assert n_ > 0, "\nArgument n_ must be > 0"

    cdef bint contain_rect

    if auto_n_:
        n_ = 1
        warn_ = False

    # N ITERATIONS, MOVE THE PIXELS OR BLOCKS
    for _ in range(n_):

        # RESET THE VALUE
        contain_rect = False

        for s in sg_:
            s.rect.topleft += s.vector
            if screenrect.contains(s.rect):
                contain_rect = True

    if contain_rect:
        if auto_n_:
            burst_into_memory_c(
                n_,
                sg_,
                screenrect,
                warn_=warn_,
                auto_n_=auto_n_
            )

    # THROW AN ERROR MSG WHEN PIXELS ARE STILL VISIBLE
    # WITHIN THE GAME DISPLAY
    if warn_:
        if contain_rect:
            raise ValueError("\nburst_into_memory - At least "
                             "one or more pixels are still visible, increase n_.")


@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
cdef void rebuild_from_memory_c(object screen_, list vertex_array_, unsigned char blend_=0):
    """
    REBUILD AN IMAGE (EXPLODED PIXELS OR PIXEL'S BLOCK BEING OUTSIDE SCREEN BOUNDARIES)

    :param screen_: Pygame.Surface object; Game display 
    :param vertex_array_: list; python list containing objects (pixels or blocks of pixels) 
        set with pre-defined attributes and values.
    :param blend_: unsigned char; blend mode (additive mode)
    """

    cdef:
        screen_blit = screen_.blit
        int s_org0, s_org1, s_0, s_1
        float s_r
        bint s_stop


    screenrect = screen_.get_rect()
    screenrect_contains = screenrect.contains

    # ITERATE OVER EVERY BLOCKS
    for s in vertex_array_:

        s.rect.topleft -= s.vector
        s_rect = s.rect
        s_stop = s.stop

        # CHECK THE BLOCK STATUS,
        if not s_stop:

            s_org0, s_org1 = s.org[0], s.org[1]
            s_0, s_1 = s_rect.topleft[0], s_rect.topleft[1]

            # DETERMINE THE DISTANCE FROM ORIGIN
            # AVOIDING SQUARE ROOT TO INCREASE PERFS

            s_r = ((s_org0 - s_0) * (s_org0 - s_0)
                        + (s_org1 - s_1) * (s_org1 - s_1))

            if s_r < <float>8.0:
                s.vector = Vector2(<float>0.0, <float>0.0)
                s_0, s_1 = s_org0, s_org1
                s_stop = True

        if screenrect_contains(s_rect):

            # DRAW THE PIXEL BLOCKs
            screen_blit(s.image, s_rect, special_flags=blend_)


