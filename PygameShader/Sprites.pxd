
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, optimize.use_switch=True, initializedcheck=False
# encoding: utf-8

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

