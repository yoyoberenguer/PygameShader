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
Library Overview
The functions appear to be part of a library designed for image manipulation and palette management, 
for applications such as graphical rendering, image processing, or creating custom color palettes.
The library may be used in graphics software or visual applications where images need to be rendered 
with specific color palettes, or where images need to be modified dynamically based on palette data.

Image Surface: refers to an object representing an image, where individual pixels can be accessed
and manipulated. Palette: Refers to a collection of colors that can be applied to images, often used in
visual design, digital art, or simulations.Temporary Buffer: This could be used to store intermediate 
results during computations, reducing the need to perform expensive operations multiple times.
Overall, the purpose of the library is to provide tools for manipulating images at the color level, 
either by changing palettes or creating new ones from scratch, optimizing image handling for graphical applications.
"""

from PygameShader.config import __VERSION__


import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=ImportWarning)

import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

try:
    cimport cython
    from cython.parallel cimport prange, parallel

except ImportError:
    raise ImportError("\n<cython> library is missing on your system."
          "\nTry: \n   C:\\pip install cython on a window command prompt.")


try:
    import pygame
    from pygame import Surface
    from pygame.surfarray import pixels3d, array_alpha, pixels_alpha, array3d, \
        make_surface, blit_array

except ImportError:
    raise ImportError("\n<Pygame> library is missing on your system."
          "\nTry: \n   C:\\pip install pygame on a window command prompt.")

try:
    from numpy import empty, uint8, float32, asarray, \
        ascontiguousarray, zeros, uint32, int32, int8, array, ndarray
except ImportError:
    raise ImportError("\n<numpy> library is missing on your system."
          "\nTry: \n   C:\\pip install numpy on a window command prompt.")

from PygameShader.misc cimport rgb_to_int_c, int_to_rgb_c
from PygameShader.misc cimport is_type_memoryview

from PygameShader.config import OPENMP, THREAD_NUMBER, __VERSION__
from libc.string cimport memcpy


cdef int THREADS = 1

if OPENMP:
     THREADS = THREAD_NUMBER

DEF SCHEDULE = 'static'

# todo tidy up unused DEF
DEF HALF         = 1.0/2.0
DEF ONE_THIRD    = 1.0/3.0
DEF ONE_FOURTH   = 1.0/4.0
DEF ONE_FIFTH    = 1.0/5.0
DEF ONE_SIXTH    = 1.0/6.0
DEF ONE_SEVENTH  = 1.0/7.0
DEF ONE_HEIGHT   = 1.0/8.0
DEF ONE_NINTH    = 1.0/9.0
DEF ONE_TENTH    = 1.0/10.0
DEF ONE_ELEVENTH = 1.0/11.0
DEF ONE_TWELVE   = 1.0/12.0
DEF ONE_32       = 1.0/32.0
DEF ONE_64       = 1.0/64.0
DEF ONE_128      = 1.0/128.0
DEF ONE_255      = 1.0/255.0
DEF ONE_360      = 1.0/360.0
DEF TWO_THIRD    = 2.0/3.0
DEF ONE_1024     = 1.0/1024

"""
Enum of common palettes based on https://en.wikipedia.org/wiki/List_of_color_palettes
All palette definition taken from
https://en.wikipedia.org/wiki/List_of_8-bit_computer_hardware_graphics
"""


# THIS LIBRARY CONTAINS ALL THE PALETTES FOR THE PROJECT
# INCLUDES :
# + TELETEXT
# + BBC_MICRO
# + CGA_MODE4_PAL1_LOW
# + CGA_MODE4_PAL1_HIGH
# + CGA_MODE4_PAL2_LOW
# + CGA_MODE4_PAL2_HIGH
# + CGA_MODE5_LOW
# + CGA_MODE5_HIGH
# + ZX_SPECTRUM_LOW
# + ZX_SPECTRUM_HIGH
# + APPLE_II_LOW
# + APPLE_II_HIGH
# + APPLE_II
# + COMMODORE_64
# + AMSTRAD_CPC
# + MSX
# + TO7
# + PICO_8
# + MICROSOFT_WINDOWS_16
# + MICROSOFT_WINDOWS_20
# + MICROSOFT_WINDOWS_PAINT
# + MONO_PHOSPHOR_AMBER
# + MONO_PHOSPHOR_LTAMBER
# + MONO_PHOSPHOR_GREEN1
# + MONO_PHOSPHOR_GREEN2
# + MONO_PHOSPHOR_GREEN3
# + AAP64
# + APOLLO
# + FUNKYFUTURE
# + VINIK24
# + TWILIOQUEST76
# + IRIDESCENTCRYSTAL
# + AAPSPLENDOR128
# + LOSPEC500
# + FAMICUBE
# SILVER

_CGA_TEST = "The Color Graphics Adapter (CGA) outputs what IBM called digital RGB[1] " \
           "(that is, the R, G, B (and I) signals from the graphics card to the monitor " \
           "can each only have two states: on or off). CGA supports a maximum of 16 colors." \
           " However, its 320×200 graphics modes is restricted to fixed palettes containing " \
           "only four colors, and the 640×200 graphic modes is only two colors. 16 colors are" \
           " only available in text modes or the tweaked text 160×100 modes. A different set" \
           " of 16 colors is available in composite modes with an NTSC composite monitor. " \
           "(Independent groups have also demonstrated much larger composite color sets—over" \
           " 256 colors—on a composite monitor by the use of artifact color techniques." \
           " See Color Graphics Adapter#High color depth.) The full standard RGBI palette is a " \
           "variant of the 4-bit RGBI schema. Although the RGBI signals each have only two " \
           "states, the CGA color monitor decodes them as if RGB signals had four levels. " \
           "Darker colors are the basic RGB 2nd level signals except for brown, which is dark" \
           " yellow with the level for the green component halved (1st level). Brighter colors" \
           " are made by adding a uniform intensity one-level signal to every RGB signal of the" \
           " dark ones, reaching the 3rd level (except dark gray which reaches only the 1st " \
           "level), and in this case yellow is produced as if the brown were ordinary dark " \
           "yellow."

LIST_PALETTES = {
    "TELETEXT" :
        "World System Teletext Level 1 (1976) uses a 3-bit RGB, 8-color palette."
        " Teletext has 40×25 characters per page of which the first row is reserved"
        " for a page header. Every character cell has a background color and a text color."
        " These attributes along with others are set through control codes which each "
        "occupy one character position. Graphics characters consisting of 2×3 cells can "
        "used following a graphics color attribute.",

    "BBC_MICRO":
        "BBC Micro has 8 display modes, with resolutions like 640×256 (max. 2 colors),"
        " 320×256 (max. 4 colors) and 160×256 (max. 16 logical colors). No display modes"
        " have cell attribute clashes. The palette available has only 8 physical colors,"
        " plus a further 8 flashing colors (each being one of the eight non-flashing colors"
        " alternating with its physical complement every second), and the display modes can"
        " have 16, 4 or 2 simultaneous colors.",


    "CGA_MODE4_PAL1_LOW" : _CGA_TEST,
    "CGA_MODE4_PAL1_HIGH": _CGA_TEST,
    "CGA_MODE4_PAL2_LOW" : _CGA_TEST,
    "CGA_MODE4_PAL2_HIGH": _CGA_TEST,
    "CGA_MODE5_LOW"      : _CGA_TEST,
    "CGA_MODE5_HIGH"     : _CGA_TEST,
    "ZX_SPECTRUM_LOW"    :
        "The ZX Spectrum (and compatible) computers use a variation of the 4-bit RGBI palette"
        " philosophy. This results in each of the colors of the 3-bit palette to have a basic"
        " and bright variant, with the exception of black. This was accomplished by having a "
        "maximum voltage level for the bright variant, and a lower voltage level for the basic"
        " variant. Due to this, black is the same in both variants.",
    "ZX_SPECTRUM_HIGH"   :
        "The ZX Spectrum (and compatible) computers use a variation of the 4-bit RGBI palette"
        " philosophy. This results in each of the colors of the 3-bit palette to have a basic"
        " and bright variant, with the exception of black. This was accomplished by having a "
        "maximum voltage level for the bright variant, and a lower voltage level for the basic"
        " variant. Due to this, black is the same in both variants.",

    "APPLE_II_LOW"       :
        "The Apple II series features a 16-color composite video palette, based on the YIQ"
        " color space used by the NTSC color TV system.[10][11]."
        "The Apple II features low-res and hi-res modes. The 40x48 pixel lo-res modes allowed"
        " 15 different colors plus a duplicate gray(**).[12] * The majority of Apple graphic"
        " applications used the hi-res modes, which had 280×192 pixels (effectively 140x192"
        " on a color monitor). The hi-res modes allowed six colors: black, white, blue, orange,"
        "green and purple.",
    "APPLE_II_HIGH"      :
        "The Apple II series features a 16-color composite video palette, based on the YIQ"
        " color space used by the NTSC color TV system.[10][11]."
        "The Apple II features low-res and hi-res modes. The 40x48 pixel lo-res modes allowed"
        " 15 different colors plus a duplicate gray(**).[12] * The majority of Apple graphic"
        " applications used the hi-res modes, which had 280×192 pixels (effectively 140x192"
        " on a color monitor). The hi-res modes allowed six colors: black, white, blue, orange,"
        "green and purple.",
    "APPLE_II"           :
        "The Apple II series features a 16-color composite video palette, based on the YIQ"
        " color space used by the NTSC color TV system.[10][11]."
        "The Apple II features low-res and hi-res modes. The 40x48 pixel lo-res modes allowed"
        " 15 different colors plus a duplicate gray(**).[12] * The majority of Apple graphic"
        " applications used the hi-res modes, which had 280×192 pixels (effectively 140x192"
        " on a color monitor). The hi-res modes allowed six colors: black, white, blue, orange,"
        "green and purple.",
    "COMMODORE_64"       :
        "The Commodore 64 has two graphic modes: Multicolor and High Resolution."
        "In the Multicolor 160×200, 16-color modes, every cell of 4×8, 2:1 aspect ratio "
        "pixels can have one of four colors: one shared with the entire screen, the two"
        " background and foreground colors of the corresponding text modes character, and"
        " one more color also stored in the color RAM area, all of them freely selectable"
        " among the entire palette.In the High Resolution 320×200, 16-color modes, every "
        "cell of 8×8 pixels can have one of the two background and foreground colors of the"
        " correspondent text modes character, both freely selectable among the entire palette.",
    "AMSTRAD_CPC"        :
        "The Amstrad CPC 464/664/6128 series of computers generates the available palette with"
        " 3 levels (not bits) for every RGB primary. Thus, there are 27 different RGB "
        "combinations, from which 16 can be simultaneously displayed in low resolution modes,"
        " four in medium resolution modes and two in high resolution modes.[7]",
    "MSX"                :
        "The MSX series has two graphic modes. The MSX BASIC Screen 3 modes is a low-resolution"
        " modes with 15 colors, in which every pixel can be any of the 15 available colors. "
        "Screen modes 2 is a 256×192 high-resolution modes with 15 colors, in which each of "
        "every eight consecutive pixels can only use 2 colors.",
    "TO7"                :
        "For Thomson computers, a popular brand in France, the most common display modes "
        "are 320×200, with 8×1 attribute cells with 2 colors. Here the intensity byte "
        "affects saturation and not only brightness variations.Thomson MO5 and TO7 "
        "The Thomson MO5 generated graphics based on a EFGJ03L gate array.[2] The palette "
        "was based on 8 RGB primary colors, with an intensity bit that could change its "
        "brightness or saturation.[3] This allowed the machine to generate a 320×200 16 "
        "color display, but subject to proximity constraints (only two colors for a 8x1 pixel"
        " area). The Thomson TO7 had similar graphic abilities and palette.[4]",

    "PICO_8"             :
        "Every pixel on PICO-8 is stored as a 4-bit value in memory. Because a 4-bit value"
        " can only hold the values 0-15, this means pixels can only choose from a list of 16"
        " colors. This list is referred to as the palette. PICO-8 has three layers of palettes. "
        "Only the first two are configurable. The first is used during each draw call, "
        "re-mapping the requested 4-bit indices to the 4-bit indices that are actually written"
        " to screen data memory. The second is used when the frame is presented to the viewer,"
        " re-mapping the 4-bit indices in the screen data to 8-bit system color indices. "
        "The third maps the 8-bit system color indices to pre-defined R,G,B values.",

    "PICO_8_CUSTOM"     :
        "Every pixel on PICO-8 is stored as a 4-bit value in memory. Because a 4-bit value"
        " can only hold the values 0-15, this means pixels can only choose from a list of 16"
        " colors. This list is referred to as the palette. PICO-8 has three layers of palettes. "
        "Only the first two are configurable. The first is used during each draw call, "
        "re-mapping the requested 4-bit indices to the 4-bit indices that are actually written"
        " to screen data memory. The second is used when the frame is presented to the viewer,"
        " re-mapping the 4-bit indices in the screen data to 8-bit system color indices. "
        "The third maps the 8-bit system color indices to pre-defined R,G,B values.",

    "MICROSOFT_WINDOWS_16" : "Microsoft Windows and IBM OS/2 default 16-color palette",


    "MICROSOFT_WINDOWS_20":
        "Microsoft Windows default 20-color palette.In 256-color modes,"
        " there are four additional standard Windows colors, twenty system reserved colors"
        " in total;[1][2] thus the system leaves 236 palette indexes free for applications "
        "to use. The system color entries inside a 256-color palette table are the first ten"
        " plus the last ten. In any case, the additional system colors do not seem to add a"
        " sharp color richness: they are only some intermediate shades of grayish colors.",

    "MICROSOFT_WINDOWS_PAINT": "Microsoft paint color palette",
    "MONO_PHOSPHOR_AMBER"    : "MONOCHROME 2 amber colors",
    "MONO_PHOSPHOR_LTAMBER"  : "MONOCHROME 2 amber colors",
    "MONO_PHOSPHOR_GREEN1"   : "MONOCHROME 2 green colors",
    "MONO_PHOSPHOR_GREEN2"   : "MONOCHROME 2 green colors",
    "MONO_PHOSPHOR_GREEN3"   : "MONOCHROME 2 green colors",
    "AAP64"                  : "64 colors palette by Adigun Polack",
    "APOLLO"                 : "46 colors palette by AdamCYounis",
    "FUNKYFUTURE"            : "8 colors palette Created by Shamaboy",
    "VINIK24"                : "24 colors palette Created by Vinik",
    "TWILIOQUEST76"          : "76 colors palette Created by Kerrie Lake",
    "IRIDESCENTCRYSTAL"      : "Created by LostInIndigo",
    "AAPSPLENDOR128"         : "128 colors Created by Adigun A. Polack",
    "LOSPEC500"              : "https://lospec.com/palette-list/lospec500 - Collaboration",
    "FAMICUBE"               : "Created by Arne as part of his Famicube Project.",
    "SILVER"                 : "Created by Okee"
}


_TELETEXT_NORMALIZED = array([
    [ 0., 0., 0. ],
    [ 0., 0., 255.0/255.0 ],
    [ 255.0/255.0, 0.0, 0.0 ],
    [ 255.0/255.0, 0.0, 255.0/255.0 ],
    [ 0., 255.0/255.0, 0.0 ],
    [ 0.0, 255.0/255.0, 255.0/255.0 ],
    [ 255.0/255.0, 255.0/255.0, 0.0 ],
    [ 255.0/255.0, 255.0/255.0, 255.0/255.0]
], dtype=float32)
TELETEXT =_TELETEXT_NORMALIZED * <float>255.0

BBC_MICRO = TELETEXT

_CGA_MODE4_PAL1_NORMALIZED_LOW = array([
    [ 0., 0., 0. ],
    [ 0., 0.66666, 0.6666 ],
    [ 0.66666, 0., 0.6666 ],
    [ 0.66666, 0.6666, 0.66666 ]
], dtype=float32)
CGA_MODE4_PAL1_LOW =_CGA_MODE4_PAL1_NORMALIZED_LOW * <float>255.0

_CGA_MODE4_PAL1_NORMALIZED_HIGH = array([
    [ 0., 0., 0. ],
    [ 85.0/255.0, 255.0/255.0, 255.0/255.0 ],
    [ 255.0/255.0, 85.0/255.0, 255.0/255.0 ],
    [ 255.0/255.0, 255.0/255.0, 255.0/255.0 ]
], dtype=float32)
CGA_MODE4_PAL1_HIGH =_CGA_MODE4_PAL1_NORMALIZED_HIGH * <float>255.0


_CGA_MODE4_PAL2_NORMALIZED_LOW = array([
    [ 0., 0., 0. ],
    [ 0.0, 170.0/255.0, 0.0 ],
    [ 170.0/255.0, 0.0, 0.0 ],
    [ 170.0/255.0, 85.0/255.0, 0.0 ]
], dtype=float32)
CGA_MODE4_PAL2_LOW =_CGA_MODE4_PAL2_NORMALIZED_LOW * <float>255.0

_CGA_MODE4_PAL2_NORMALIZED_HIGH = array([
    [ 0., 0., 0. ],
    [ 85.0/255.0, 255.0/255.0, 85.0/255.0],
    [ 255.0/255.0, 85.0/255.0, 85.0/255.0 ],
    [ 255.0/255.0, 255.0/255.0, 85.0/255.0 ]
], dtype=float32)
CGA_MODE4_PAL2_HIGH =_CGA_MODE4_PAL2_NORMALIZED_HIGH * <float>255.0


_CGA_MODE5_NORMALIZED_LOW = array([
    [ 0., 0., 0. ],
    [ 0.0/255.0, 170.0/255.0, 170./255.0 ],
    [ 170.0/255.0, 0.0, 0.0 ],
    [ 170.0/255.0, 170.0/255.0, 170.0/255.0 ]
], dtype=float32)
CGA_MODE5_LOW =_CGA_MODE5_NORMALIZED_LOW * <float>255.0

_CGA_MODE5_NORMALIZED_HIGH = array([
    [ 0., 0., 0. ],
    [ 85.0/255.0, 255.0/255.0, 255.0/255.0],
    [ 255.0/255.0, 85.0/255.0, 85.0/255.0 ],
    [ 255.0/255.0, 255.0/255.0, 255.0/255.0 ]
], dtype=float32)
CGA_MODE5_HIGH =_CGA_MODE5_NORMALIZED_HIGH * <float>255.0


_ZX_SPECTRUM_NORMALIZED_LOW = array([
     [ 0., 0., 0. ] ,
     [ 0., 0.0, 192.0/255.0 ] ,
     [ 192.0/255.0, 0.0, 0.0 ] ,
     [ 192.0/255.0, 0.0, 192.0/255.0 ] ,
     [ 0., 192.0/255.0, 0.0 ],
     [ 0., 192.0/255.0, 192.0/255.0 ],
     [ 192.0/255.0, 192.0/255.0, 0.0 ],
     [ 192.0/255.0, 192.0/255.0, 192.0/255.0 ]
], dtype=float32)
ZX_SPECTRUM_LOW =_ZX_SPECTRUM_NORMALIZED_LOW * <float>255.0


_ZX_SPECTRUM_NORMALIZED_HIGH = array([
     [ 0., 0., 0. ] ,
     [ 0., 0.0, 255.0/255.0 ] ,
     [ 255.0/255.0, 0.0, 0.0 ] ,
     [ 255.0/255.0, 0.0, 255.0/255.0 ] ,
     [ 0., 255.0/255.0, 0.0 ],
     [ 0., 255.0/255.0, 255.0/255.0 ],
     [ 255.0/255.0, 255.0/255.0, 0.0 ],
     [ 255.0/255.0, 255.0/255.0, 255.0/255.0 ]
], dtype=float32)
ZX_SPECTRUM_HIGH =_ZX_SPECTRUM_NORMALIZED_HIGH * <float>255.0


_APPLE_II_LOW_NORMALIZED = array([
    [ 0., 0., 0. ],
    [ 108.0/255.0, 41.0/255.0, 64.0/255.0 ],
    [ 64.0/255.0, 53.0/255.0, 120.0/255.0 ],
    [ 217.0/255.0, 60.0/255.0, 240.0/255.0 ],
    [ 19.0/255.0, 87.0/255.0, 64.0/255.0 ],
    [ 128.0/255.0, 128.0/255.0, 128.0/255.0],
    [ 38.0/255.0, 151.0/255.0, 240.0/255.0 ],
    [ 191.0/255.0, 180.0/255.0, 248.0/255.0]
], dtype=float32)
APPLE_II_LOW =_APPLE_II_LOW_NORMALIZED * <float>255.0

_APPLE_II_HIGH_NORMALIZED = array([
    [ 64.0/255.0, 75.0/255.0, 7.0/255.0 ],
    [ 217.0/255.0, 104.0/255.0, 15.0/255.0 ],
    [ 128.0/255.0, 128.0/255.0, 128.0/255.0 ],
    [ 236.0/255.0, 168.0/255.0, 191.0/255.0 ],
    [ 38.0/255.0, 195.0/255.0, 15.0/255.0],
    [ 191.0/255.0, 202.0/255.0, 135.0/255.0],
    [ 147.0/255.0, 214.0/255.0, 191.0/255.0 ],
    [ 255.0/255.0, 255.0/255.0, 255.0/255.0]
], dtype=float32)
APPLE_II_HIGH =_APPLE_II_HIGH_NORMALIZED * <float>255.0

_APPLE_II_NORMALIZED = array([
    [ 0., 0., 0. ],
    [ 108.0/255.0, 41.0/255.0, 64.0/255.0 ],
    [ 64.0/255.0, 53.0/255.0, 120.0/255.0 ],
    [ 217.0/255.0, 60.0/255.0, 240.0/255.0 ],
    [ 19.0/255.0, 87.0/255.0, 64.0/255.0 ],
    [ 128.0/255.0, 128.0/255.0, 128.0/255.0],
    [ 38.0/255.0, 151.0/255.0, 240.0/255.0 ],
    [ 191.0/255.0, 180.0/255.0, 248.0/255.0],
    [ 64.0/255.0, 75.0/255.0, 7.0/255.0 ],
    [ 217.0/255.0, 104.0/255.0, 15.0/255.0 ],
    [ 236.0/255.0, 168.0/255.0, 191.0/255.0 ],
    [ 38.0/255.0, 195.0/255.0, 15.0/255.0],
    [ 191.0/255.0, 202.0/255.0, 135.0/255.0],
    [ 147.0/255.0, 214.0/255.0, 191.0/255.0 ],
    [ 255.0/255.0, 255.0/255.0, 255.0/255.0]
], dtype=float32)
APPLE_II =_APPLE_II_NORMALIZED * <float>255.0


_COMMODORE_64_NORMALIZED = array([
    [ 0., 0., 0. ],
    [ 255.0/255.0, 255.0/255.0, 255.0/255.0 ],
    [ 136.0/255.0, 0.0, 0.0],
    [ 170.0/255.0, 255.0/255.0, 238.0/255.0 ],
    [ 204.0/255.0, 68.0/255.0, 204.0/255.0],
    [ 0.0, 204.0/255.0, 85.0/255.0 ],
    [ 0.0, 0.0, 170.0/255.0 ],
    [ 238.0/255.0, 238.0/255.0, 119.0/255.0 ],
    [ 221.0/255.0, 136.0/255.0, 85.0/255.0],
    [ 102.0/255.0, 68.0/255.0, 0.0],
    [ 255.0/255.0, 119.0/255.0, 119.0/255.0 ],
    [ 51.0/255.0, 51.0/255.0, 51.0/255.0 ],
    [ 119.0/255.0, 119.0/255.0, 119.0/255.0 ],
    [ 170.0/255.0, 255.0/255.0, 102.0/255.0 ],
    [ 0.0, 136.0/255.0, 255.0/255.0 ],
    [ 187.0/255.0, 187.0/255.0, 187.0/255.0 ]
], dtype=float32)
COMMODORE_64 =_COMMODORE_64_NORMALIZED * <float>255.0


_AMSTRAD_CPC_NORMALIZED = array(
    [[0, 0, 0],
    [0, 0, 128.0/255.0],
    [0, 0, 255/255.0],
    [128/255.0, 0, 0],
    [128/255.0, 0, 128/255.0],
    [128/255.0, 0, 255/255.0],
    [255/255.0, 0, 0],
    [255/255.0, 0, 128/255.0],
    [255/255.0, 0, 255/255.0],
    [0, 128/255.0, 0],
    [0, 128/255.0, 128/255.0],
    [0, 128/255.0, 255/255.0],
    [128/255.0, 128/255.0, 0],
    [128/255.0, 128/255.0, 128/255.0],
    [128/255.0, 128/255.0, 255/255.0],
    [255/255.0, 128/255.0, 0],
    [255/255.0, 128/255.0, 128/255.0],
    [255/255.0, 128/255.0, 255/255.0],
    [0, 255/255.0, 0],
    [0, 255/255.0, 128/255.0],
    [0, 255/255.0, 255/255.0],
    [128/255.0, 255/255.0, 0],
    [128/255.0, 255/255.0, 128/255.0],
    [128/255.0, 255/255.0, 255/255.0],
    [255/255.0, 255/255.0, 0],
    [255/255.0, 255/255.0, 128/255.0],
    [255/255.0, 255/255.0, 255/255.0]], dtype=float32)

AMSTRAD_CPC =_AMSTRAD_CPC_NORMALIZED * <float>255.0


_MSX_NORMALIZED = array([
    [ 0., 0., 0. ],
    [ 1.0/255.0, 1.0/255.0, 1.0/255.0 ],
    [ 64/255.0, 184/255.0, 73/255.0 ],
    [ 116/255.0, 208/255.0, 125/255.0 ],
    [ 89/255.0, 85/255.0, 224/255.0 ],
    [ 128/255.0, 118/255.0, 241/255.0],
    [ 185/255.0, 94/255.0, 81/255.0],
    [  101/255.0, 219/255.0, 239/255.0],
    [  219/255.0, 101/255.0, 89/255.0],
    [ 255/255.0, 137/255.0, 125/255.0],
    [  204/255.0, 195/255.0, 94/255.0],
    [  222/255.0, 208/255.0, 135/255.0],
    [  58/255.0, 162/255.0, 65/255.0],
    [  183/255.0, 102/255.0, 181/255.0],
    [ 204/255.0, 204/255.0, 204/255.0],
    [ 255.0/255.0, 255.0/255.0, 255.0/255.0 ]
], dtype=float32)
MSX =_MSX_NORMALIZED * <float>255.0



_TO7_NORMALIZED = array([
    [ 0., 0., 0. ],
    [ 0.0/255.0, 0.0/255.0, 255.0/255.0 ],
    [ 255.0/255.0, 0.0/255.0, 0.0/255.0 ],
    [ 255.0/255.0, 0.0/255.0, 255.0/255.0 ],
    [ 0.0/255.0, 255.0/255.0, 0.0/255.0 ],
    [ 0.0/255.0, 255.0/255.0, 255.0/255.0],
    [ 255.0/255.0, 255.0/255.0, 0.0/255.0],
    [  255.0/255.0, 255.0/255.0, 255.0/255.0],
    [  221.0/255.0, 119.0/255.0, 119.0/255.0],
    [ 239.0/255.0, 187.0/255.0, 0.0/255.0],
    [  221.0/255.0, 221.0/255.0, 119.0/255.0],
    [  119.0/255.0, 221.0/255.0, 119.0/255.0],
    [  187.0/255.0, 255.0/255.0, 255.0/255.0],
    [  119.0/255.0, 119.0/255.0, 221.0/255.0],
    [ 221.0/255.0, 119.0/255.0, 239.0/255.0],
    [ 187.0/255.0, 187.0/255.0, 187.0/255.0 ]
], dtype=float32)
TO7 =_TO7_NORMALIZED * <float>255.0

_PICO_8_NORMALIZED = array([
    [ 0., 0., 0. ],
    [ 29.0/255.0, 43.0/255.0, 83.0/255.0 ],
    [ 126.0/255.0, 37.0/255.0, 83.0/255.0],
    [ 0.0, 135.0/255.0, 81.0/255.0],
    [ 171.0/255.0, 82.0/255.0, 54.0/255.0],
    [ 95/255.0, 87/255.0, 79/255.0],
    [  194/255.0, 195/255.0, 199/255.0],
    [  255/255.0, 241/255.0, 232/255.0],
    [  255/255.0, 0/255.0, 77/255.0],
    [  255/255.0, 163/255.0, 0/255.0],
    [  255/255.0, 236/255.0, 39/255.0],
    [  0/255.0, 228/255.0, 54/255.0],
    [  41/255.0, 173/255.0, 255/255.0],
    [  131/255.0, 118/255.0, 156/255.0],
    [  255/255.0, 119/255.0, 168/255.0],
    [  255/255.0, 204/255.0, 170/255.0]
], dtype=float32)
PICO_8 =_PICO_8_NORMALIZED * <float>255.0


_PICO_8_CUSTOM_NORMALIZED = array([
    [ 41/255.0, 24/255.0, 20/255.0 ],
    [ 17/255.0, 29/255.0, 53/255.0 ],
    [ 66/255.0, 33/255.0, 54/255.0],
    [ 18/255.0, 83/255.0, 89/255.0],
    [ 116/255.0, 47/255.0, 41/255.0],
    [ 73/255.0, 51/255.0, 59/255.0],
    [  162/255.0, 136/255.0, 121/255.0],
    [  243/255.0, 239/255.0, 125/255.0],
    [  190/255.0, 18/255.0, 80/255.0],
    [  255/255.0, 108/255.0, 36/255.0],
    [  168/255.0, 231/255.0, 46/255.0],
    [  0/255.0, 181/255.0, 67/255.0],
    [  6/255.0, 90/255.0, 181/255.0],
    [  117/255.0, 70/255.0, 101/255.0],
    [  255/255.0, 110/255.0, 89/255.0],
    [  255/255.0, 157/255.0, 129/255.0]
], dtype=float32)
PICO_8_CUSTOM =_PICO_8_CUSTOM_NORMALIZED * <float>255.0



_MICROSOFT_WINDOWS_16_NORMALIZED = array([
    [ 0., 0., 0. ],
    [ 0.50196078, 0., 0. ],
    [ 0., 0.50196078, 0. ],
    [ 0.50196078, 0.50196078, 0. ],
    [ 0., 0., 0.50196078 ],
    [ 0.50196078, 0., 0.50196078 ],
    [ 0., 0.50196078, 0.50196078 ],
    [ 0.75294118, 0.75294118, 0.75294118 ],
    [ 0.50196078, 0.50196078, 0.50196078 ],
    [ 1., 0., 0. ],
    [ 0., 1., 0. ],
    [ 1., 1., 0. ],
    [ 0., 0., 1. ],
    [ 1., 0., 1. ],
    [ 0., 1., 1. ],
    [ 1., 1., 1. ]
], dtype=float32)
MICROSOFT_WINDOWS_16 =_MICROSOFT_WINDOWS_16_NORMALIZED * <float>255.0

_MICROSOFT_WINDOWS_20_NORMALIZED = array([
    [ 0., 0., 0. ],
    [ 0.50196078, 0., 0. ],
    [ 0., 0.50196078, 0. ],
    [ 0.50196078, 0.50196078, 0. ],
    [ 0., 0., 0.50196078 ],
    [ 0.50196078, 0., 0.50196078 ],
    [ 0., 0.50196078, 0.50196078 ],
    [ 0.75294118, 0.75294118, 0.75294118 ],
    [ 0.75294118, 0.8627451, 0.75294118 ],
    [ 0.65098039, 0.79215686, 0.94117647 ],
    [ 1., 0.98431373, 0.94117647 ],
    [ 0.62745098, 0.62745098, 0.64313725 ],
    [ 0.50196078, 0.50196078, 0.50196078 ],
    [ 1., 0., 0. ],
    [ 0., 1., 0. ],
    [ 1., 1., 0. ],
    [ 0., 0., 1. ],
    [ 1., 0., 1. ],
    [ 0., 1., 1. ],
    [ 1., 1., 1. ]
], dtype=float32)
MICROSOFT_WINDOWS_20 =_MICROSOFT_WINDOWS_20_NORMALIZED * <float>255.0

_MICROSOFT_WINDOWS_PAINT_NORMALIZED = array([
    [ 0., 0., 0. ],
    [ 1., 1., 1. ],
    [ 0.48235294, 0.48235294, 0.48235294 ],
    [ 0.74117647, 0.74117647, 0.74117647 ],
    [ 0.48235294, 0.04705882, 0.00784314 ],
    [ 1., 0.14509804, 0. ],
    [ 0.48235294, 0.48235294, 0.00392157 ],
    [ 1., 0.98431373, 0.00392157 ],
    [ 0., 0.48235294, 0.00784314 ],
    [ 0.00784314, 0.97647059, 0.00392157 ],
    [ 0., 0.48235294, 0.47843137 ],
    [ 0.00784314, 0.99215686, 0.99607843 ],
    [ 0.00392157, 0.0745098, 0.47843137 ],
    [ 0.01568627, 0.19607843, 1. ],
    [ 0.48235294, 0.09803922, 0.47843137 ],
    [ 1., 0.25098039, 0.99607843 ],
    [ 0.47843137, 0.22352941, 0.00392157 ],
    [ 1., 0.47843137, 0.22352941 ],
    [ 0.48235294, 0.48235294, 0.21960784 ],
    [ 1., 0.98823529, 0.47843137 ],
    [ 0.00784314, 0.22352941, 0.22352941 ],
    [ 0.01176471, 0.98039216, 0.48235294 ],
    [ 0., 0.48235294, 1. ],
    [ 1., 0.17254902, 0.48235294 ]
], dtype=float32)
MICROSOFT_WINDOWS_PAINT =_MICROSOFT_WINDOWS_PAINT_NORMALIZED * <float>255.0

# https://superuser.com/questions/361297/what-colour-is-t
# he-dark-green-on-old-fashioned-green-screen-computer-displays/1206781#1206781
_MONO_PHOSPHOR_AMBER_NORMALIZED = array([
    [ 0.15686275, 0.15686275, 0.15686275 ],
    [ 1., 0.69019608, 0. ]
], dtype=float32)
MONO_PHOSPHOR_AMBER =_MONO_PHOSPHOR_AMBER_NORMALIZED * <float>255.0

_MONO_PHOSPHOR_LTAMBER_NORMALIZED = array([
    [ 0.15686275, 0.15686275, 0.15686275 ],
    [ 1., 0.8, 0. ]
], dtype=float32)
MONO_PHOSPHOR_LTAMBER =_MONO_PHOSPHOR_LTAMBER_NORMALIZED * <float>255.0

_MONO_PHOSPHOR_GREEN1_NORMALIZED = array([
    [ 0.15686275, 0.15686275, 0.15686275 ],
    [ 0.2, 1., 0. ]
], dtype=float32)
MONO_PHOSPHOR_GREEN1 =_MONO_PHOSPHOR_GREEN1_NORMALIZED * <float>255.0

_MONO_PHOSPHOR_GREEN2_NORMALIZED = array([
    [ 0.15686275, 0.15686275, 0.15686275 ],
    [ 0, 1., 0.2 ]
], dtype=float32)
MONO_PHOSPHOR_GREEN2 =_MONO_PHOSPHOR_GREEN2_NORMALIZED * <float>255.0

_MONO_PHOSPHOR_GREEN3_NORMALIZED = array([
    [ 0.15686275, 0.15686275, 0.15686275 ],
    [ 0, 1., 0.4 ]
], dtype=float32)
MONO_PHOSPHOR_GREEN3 =_MONO_PHOSPHOR_GREEN3_NORMALIZED * <float>255.0

# Created by Adigun A. Polack
# https://lospec.com/palette-list/aap-64
_AAP64_NORMALIZED = array([
    [0.023529411764705882, 0.023529411764705882, 0.03137254901960784],
    [0.0784313725490196, 0.06274509803921569, 0.07450980392156863],
    [0.23137254901960785, 0.09019607843137255, 0.1450980392156863],
    [0.45098039215686275, 0.09019607843137255, 0.17647058823529413],
    [0.7058823529411765, 0.12549019607843137, 0.16470588235294117],
    [0.8745098039215686, 0.24313725490196078, 0.13725490196078433],
    [0.9803921568627451, 0.41568627450980394, 0.0392156862745098],
    [0.9764705882352941, 0.6392156862745098, 0.10588235294117647],
    [1.0, 0.8352941176470589, 0.2549019607843137],
    [1.0, 0.9882352941176471, 0.25098039215686274],
    [0.8392156862745098, 0.9490196078431372, 0.39215686274509803],
    [0.611764705882353, 0.8588235294117647, 0.2627450980392157],
    [0.34901960784313724, 0.7568627450980392, 0.20784313725490197],
    [0.0784313725490196, 0.6274509803921569, 0.1803921568627451],
    [0.10196078431372549, 0.47843137254901963, 0.24313725490196078],
    [0.1411764705882353, 0.3215686274509804, 0.23137254901960785],
    [0.07058823529411765, 0.12549019607843137, 0.12549019607843137],
    [0.0784313725490196, 0.20392156862745098, 0.39215686274509803],
    [0.1568627450980392, 0.3607843137254902, 0.7686274509803922],
    [0.1411764705882353, 0.6235294117647059, 0.8705882352941177],
    [0.12549019607843137, 0.8392156862745098, 0.7803921568627451],
    [0.6509803921568628, 0.9882352941176471, 0.8588235294117647],
    [1.0, 1.0, 1.0],
    [0.996078431372549, 0.9529411764705882, 0.7529411764705882],
    [0.9803921568627451, 0.8392156862745098, 0.7215686274509804],
    [0.9607843137254902, 0.6274509803921569, 0.592156862745098],
    [0.9098039215686274, 0.41568627450980394, 0.45098039215686275],
    [0.7372549019607844, 0.2901960784313726, 0.6078431372549019],
    [0.4745098039215686, 0.22745098039215686, 0.5019607843137255],
    [0.25098039215686274, 0.2, 0.3254901960784314],
    [0.1411764705882353, 0.13333333333333333, 0.20392156862745098],
    [0.13333333333333333, 0.10980392156862745, 0.10196078431372549],
    [0.19607843137254902, 0.16862745098039217, 0.1568627450980392],
    [0.44313725490196076, 0.2549019607843137, 0.23137254901960785],
    [0.7333333333333333, 0.4588235294117647, 0.2784313725490196],
    [0.8588235294117647, 0.6431372549019608, 0.38823529411764707],
    [0.9568627450980393, 0.8235294117647058, 0.611764705882353],
    [0.8549019607843137, 0.8784313725490196, 0.9176470588235294],
    [0.7019607843137254, 0.7254901960784313, 0.8196078431372549],
    [0.5450980392156862, 0.5764705882352941, 0.6862745098039216],
    [0.42745098039215684, 0.4588235294117647, 0.5529411764705883],
    [0.2901960784313726, 0.32941176470588235, 0.3843137254901961],
    [0.2, 0.2235294117647059, 0.2549019607843137],
    [0.25882352941176473, 0.1411764705882353, 0.2],
    [0.3568627450980392, 0.19215686274509805, 0.2196078431372549],
    [0.5568627450980392, 0.3215686274509804, 0.3215686274509804],
    [0.7294117647058823, 0.4588235294117647, 0.41568627450980394],
    [0.9137254901960784, 0.7098039215686275, 0.6392156862745098],
    [0.8901960784313725, 0.9019607843137255, 1.0],
    [0.7254901960784313, 0.7490196078431373, 0.984313725490196],
    [0.5176470588235295, 0.6078431372549019, 0.8941176470588236],
    [0.34509803921568627, 0.5529411764705883, 0.7450980392156863],
    [0.2784313725490196, 0.49019607843137253, 0.5215686274509804],
    [0.13725490196078433, 0.403921568627451, 0.3058823529411765],
    [0.19607843137254902, 0.5176470588235295, 0.39215686274509803],
    [0.36470588235294116, 0.6862745098039216, 0.5529411764705883],
    [0.5725490196078431, 0.8627450980392157, 0.7294117647058823],
    [0.803921568627451, 0.9686274509803922, 0.8862745098039215],
    [0.8941176470588236, 0.8235294117647058, 0.6666666666666666],
    [0.7803921568627451, 0.6901960784313725, 0.5450980392156862],
    [0.6274509803921569, 0.5254901960784314, 0.3843137254901961],
    [0.4745098039215686, 0.403921568627451, 0.3333333333333333],
    [0.35294117647058826, 0.3058823529411765, 0.26666666666666666],
    [0.25882352941176473, 0.2235294117647059, 0.20392156862745098]
], dtype=float32)
AAP64 =_AAP64_NORMALIZED * <float>255.0

# https://lospec.com/palette-list/apollo
# Created by AdamCYounis
_APOLLO_NORMALIZED = array([
[0.09019607843137255, 0.12549019607843137, 0.2196078431372549],
[0.1450980392156863, 0.22745098039215686, 0.3686274509803922],
[0.23529411764705882, 0.3686274509803922, 0.5450980392156862],
[0.30980392156862746, 0.5607843137254902, 0.7294117647058823],
[0.45098039215686275, 0.7450980392156863, 0.8274509803921568],
[0.6431372549019608, 0.8666666666666667, 0.8588235294117647],
[0.09803921568627451, 0.2, 0.17647058823529413],
[0.1450980392156863, 0.33725490196078434, 0.1803921568627451],
[0.27450980392156865, 0.5098039215686274, 0.19607843137254902],
[0.4588235294117647, 0.6549019607843137, 0.2627450980392157],
[0.6588235294117647, 0.792156862745098, 0.34509803921568627],
[0.8156862745098039, 0.8549019607843137, 0.5686274509803921],
[0.30196078431372547, 0.16862745098039217, 0.19607843137254902],
[0.47843137254901963, 0.2823529411764706, 0.2549019607843137],
[0.6784313725490196, 0.4666666666666667, 0.3411764705882353],
[0.7529411764705882, 0.5803921568627451, 0.45098039215686275],
[0.8431372549019608, 0.7098039215686275, 0.5803921568627451],
[0.9058823529411765, 0.8352941176470589, 0.7019607843137254],
[0.20392156862745098, 0.10980392156862745, 0.15294117647058825],
[0.3764705882352941, 0.17254901960784313, 0.17254901960784313],
[0.5333333333333333, 0.29411764705882354, 0.16862745098039217],
[0.7450980392156863, 0.4666666666666667, 0.16862745098039217],
[0.8705882352941177, 0.6196078431372549, 0.2549019607843137],
[0.9098039215686274, 0.7568627450980392, 0.4392156862745098],
[0.1411764705882353, 0.08235294117647059, 0.15294117647058825],
[0.2549019607843137, 0.11372549019607843, 0.19215686274509805],
[0.4588235294117647, 0.1411764705882353, 0.2196078431372549],
[0.6470588235294118, 0.18823529411764706, 0.18823529411764706],
[0.8117647058823529, 0.3411764705882353, 0.23529411764705882],
[0.8549019607843137, 0.5254901960784314, 0.24313725490196078],
[0.11764705882352941, 0.11372549019607843, 0.2235294117647059],
[0.25098039215686274, 0.15294117647058825, 0.3176470588235294],
[0.47843137254901963, 0.21176470588235294, 0.4823529411764706],
[0.6352941176470588, 0.24313725490196078, 0.5490196078431373],
[0.7764705882352941, 0.3176470588235294, 0.592156862745098],
[0.8745098039215686, 0.5176470588235295, 0.6470588235294118],
[0.03529411764705882, 0.0392156862745098, 0.0784313725490196],
[0.06274509803921569, 0.0784313725490196, 0.12156862745098039],
[0.08235294117647059, 0.11372549019607843, 0.1568627450980392],
[0.12549019607843137, 0.1803921568627451, 0.21568627450980393],
[0.2235294117647059, 0.2901960784313726, 0.3137254901960784],
[0.3411764705882353, 0.4470588235294118, 0.4666666666666667],
[0.5058823529411764, 0.592156862745098, 0.5882352941176471],
[0.6588235294117647, 0.7098039215686275, 0.6980392156862745],
[0.7803921568627451, 0.8117647058823529, 0.8],
[0.9215686274509803, 0.9294117647058824, 0.9137254901960784]
], dtype=float32)
APOLLO =_APOLLO_NORMALIZED * <float>255.0

#https://lospec.com/palette-list/funkyfuture-8
# Created by Shamaboy
_FUNKYFUTURE_NORMALIZED = array([
[0.16862745098039217,0.058823529411764705,0.32941176470588235],
[0.6705882352941176,0.12156862745098039,0.396078431372549],
[1.0,0.30980392156862746,0.4117647058823529],
[1.0,0.9686274509803922,0.9725490196078431],
[1.0,0.5058823529411764,0.25882352941176473],
[1.0,0.8549019607843137,0.27058823529411763],
[0.2,0.40784313725490196,0.8627450980392157],
[0.28627450980392155,0.9058823529411765,0.9254901960784314]
], dtype=float32)
FUNKYFUTURE =_FUNKYFUTURE_NORMALIZED * <float>255.0

# https://lospec.com/palette-list/vinik24
# Created by Vinik
_VINIK24_NORMALIZED = array([
[0.0,0.0,0.0],
[0.43529411764705883,0.403921568627451,0.4627450980392157],
[0.6039215686274509,0.6039215686274509,0.592156862745098],
[0.7725490196078432,0.8,0.7215686274509804],
[0.5450980392156862,0.3333333333333333,0.5019607843137255],
[0.7647058823529411,0.5333333333333333,0.5647058823529412],
[0.6470588235294118,0.5764705882352941,0.6470588235294118],
[0.4,0.3764705882352941,0.5725490196078431],
[0.6039215686274509,0.30980392156862746,0.3137254901960784],
[0.7607843137254902,0.5529411764705883,0.4588235294117647],
[0.48627450980392156,0.6313725490196078,0.7529411764705882],
[0.2549019607843137,0.41568627450980394,0.6392156862745098],
[0.5529411764705883,0.3843137254901961,0.40784313725490196],
[0.7450980392156863,0.5843137254901961,0.3607843137254902],
[0.40784313725490196,0.6745098039215687,0.6627450980392157],
[0.2196078431372549,0.4392156862745098,0.5019607843137255],
[0.43137254901960786,0.4117647058823529,0.3843137254901961],
[0.5764705882352941,0.6313725490196078,0.403921568627451],
[0.43137254901960786,0.6666666666666666,0.47058823529411764],
[0.3333333333333333,0.4392156862745098,0.39215686274509803],
[0.615686274509804,0.6235294117647059,0.4980392156862745],
[0.49411764705882355,0.6196078431372549,0.6],
[0.36470588235294116,0.40784313725490196,0.4470588235294118],
[0.2627450980392157,0.20392156862745098,0.3333333333333333]
], dtype=float32)
VINIK24 =_VINIK24_NORMALIZED * <float>255.0

# https://lospec.com/palette-list/twilioquest-76
# Created by Kerrie Lake
_TWILIOQUEST76_NORMALIZED = array([
[1.0,1.0,1.0],
[0.9176470588235294,0.9176470588235294,0.9098039215686274],
[0.807843137254902,0.792156862745098,0.788235294117647],
[0.6705882352941176,0.6862745098039216,0.7254901960784313],
[0.6313725490196078,0.5333333333333333,0.592156862745098],
[0.4588235294117647,0.3843137254901961,0.4627450980392157],
[0.36470588235294116,0.27450980392156865,0.3764705882352941],
[0.2980392156862745,0.19607843137254902,0.3137254901960784],
[0.2627450980392157,0.14901960784313725,0.2549019607843137],
[0.1568627450980392,0.09803921568627451,0.1843137254901961],
[0.984313725490196,0.4588235294117647,0.4588235294117647],
[0.984313725490196,0.23137254901960785,0.39215686274509803],
[0.7843137254901961,0.19215686274509805,0.3411764705882353],
[0.5568627450980392,0.21568627450980393,0.3607843137254902],
[0.30980392156862746,0.13725490196078433,0.3176470588235294],
[0.20784313725490197,0.08235294117647059,0.26666666666666666],
[0.9686274509803922,0.2901960784313726,0.3254901960784314],
[0.9490196078431372,0.1843137254901961,0.27450980392156865],
[0.7372549019607844,0.08627450980392157,0.25882352941176473],
[0.9882352941176471,0.7725490196078432,0.2235294117647059],
[0.9725490196078431,0.4823529411764706,0.10588235294117647],
[0.9725490196078431,0.25098039215686274,0.10588235294117647],
[0.7411764705882353,0.15294117647058825,0.03529411764705882],
[0.48627450980392156,0.07058823529411765,0.16862745098039217],
[1.0,0.8784313725490196,0.5450980392156862],
[0.9803921568627451,0.7529411764705882,0.35294117647058826],
[0.9215686274509803,0.5607843137254902,0.2823529411764706],
[0.8196078431372549,0.4549019607843137,0.2549019607843137],
[0.7803921568627451,0.3215686274509804,0.2235294117647059],
[0.6941176470588235,0.1607843137254902,0.20784313725490197],
[0.9921568627450981,0.7411764705882353,0.5607843137254902],
[0.9411764705882353,0.5333333333333333,0.4196078431372549],
[0.8274509803921568,0.40784313725490196,0.3254901960784314],
[0.6823529411764706,0.27058823529411763,0.2901960784313726],
[0.5490196078431373,0.19215686274509805,0.19607843137254902],
[0.32941176470588235,0.13725490196078433,0.13725490196078433],
[0.6588235294117647,0.34509803921568627,0.2823529411764706],
[0.5137254901960784,0.25098039215686274,0.2980392156862745],
[0.403921568627451,0.19215686274509805,0.29411764705882354],
[0.24705882352941178,0.13725490196078433,0.13725490196078433],
[0.8313725490196079,0.5843137254901961,0.4666666666666667],
[0.6235294117647059,0.4392156862745098,0.35294117647058826],
[0.5176470588235295,0.3411764705882353,0.3137254901960784],
[0.38823529411764707,0.23137254901960785,0.24705882352941178],
[0.4823529411764706,0.8431372549019608,0.6627450980392157],
[0.3215686274509804,0.6980392156862745,0.5058823529411764],
[0.0784313725490196,0.5215686274509804,0.40784313725490196],
[0.0784313725490196,0.403921568627451,0.33725490196078434],
[0.13333333333333333,0.2784313725490196,0.2980392156862745],
[0.06274509803921569,0.1843137254901961,0.20392156862745098],
[0.9215686274509803,1.0,0.5450980392156862],
[0.7019607843137254,0.8901960784313725,0.38823529411764707],
[0.2980392156862745,0.7411764705882353,0.33725490196078434],
[0.1843137254901961,0.5294117647058824,0.20784313725490197],
[0.043137254901960784,0.34901960784313724,0.19215686274509805],
[0.592156862745098,0.7490196078431373,0.43137254901960786],
[0.5372549019607843,0.6235294117647059,0.4],
[0.3803921568627451,0.5215686274509804,0.35294117647058826],
[0.2980392156862745,0.3764705882352941,0.3176470588235294],
[0.45098039215686275,0.8745098039215686,0.9490196078431372],
[0.16470588235294117,0.7333333333333333,0.8156862745098039],
[0.19215686274509805,0.36470588235294116,0.803921568627451],
[0.2784313725490196,0.16470588235294117,0.611764705882353],
[0.6274509803921569,0.8470588235294118,0.8431372549019608],
[0.49019607843137253,0.7450980392156863,0.9803921568627451],
[0.4,0.5607843137254902,0.6862745098039216],
[0.34509803921568627,0.36470588235294116,0.5058823529411764],
[0.27058823529411763,0.21176470588235294,0.36470588235294116],
[0.9647058823529412,0.7294117647058823,0.996078431372549],
[0.8352941176470589,0.6235294117647059,0.9568627450980393],
[0.6901960784313725,0.4392156862745098,0.9215686274509803],
[0.48627450980392156,0.23529411764705882,0.8823529411764706],
[0.8588235294117647,0.8117647058823529,0.6941176470588235],
[0.6627450980392157,0.6431372549019608,0.5529411764705883],
[0.4823529411764706,0.5137254901960784,0.5098039215686274],
[0.37254901960784315,0.37254901960784315,0.43137254901960786]
], dtype=float32)
TWILIOQUEST76 =_TWILIOQUEST76_NORMALIZED * <float>255.0



# https://lospec.com/palette-list/iridescent-crystal
# Created by LostInIndigo
_IRIDESCENTCRYSTAL_NORMALIZED = array([
[0.984313725490196,0.48627450980392156,0.5254901960784314],
[0.984313725490196,0.5568627450980392,0.5764705882352941],
[0.9882352941176471,0.6313725490196078,0.6274509803921569],
[0.9921568627450981,0.7058823529411765,0.6823529411764706],
[0.996078431372549,0.7803921568627451,0.7333333333333333],
[1.0,0.8549019607843137,0.788235294117647],
[0.8156862745098039,0.396078431372549,0.28627450980392155],
[0.8470588235294118,0.4980392156862745,0.36470588235294116],
[0.8823529411764706,0.6078431372549019,0.44313725490196076],
[0.9176470588235294,0.7058823529411765,0.5254901960784314],
[0.9529411764705882,0.807843137254902,0.6039215686274509],
[0.9882352941176471,0.9137254901960784,0.6862745098039216],
[0.5450980392156862,0.9882352941176471,0.9333333333333333],
[0.4392156862745098,0.8196078431372549,0.788235294117647],
[0.3333333333333333,0.6470588235294118,0.6392156862745098],
[0.2235294117647059,0.4745098039215686,0.49411764705882355],
[0.11764705882352941,0.30196078431372547,0.34509803921568627],
[0.00784313725490196,0.12941176470588237,0.19607843137254902],
[0.1568627450980392,0.027450980392156862,0.13333333333333333],
[0.29411764705882354,0.17254901960784313,0.3058823529411765],
[0.42745098039215684,0.3137254901960784,0.47843137254901963],
[0.5647058823529412,0.4549019607843137,0.6470588235294118],
[0.6980392156862745,0.596078431372549,0.8196078431372549],
[0.8313725490196079,0.7372549019607844,0.9882352941176471],
[0.8823529411764706,0.6666666666666666,0.8352941176470589],
[0.9019607843137255,0.615686274509804,0.8627450980392157],
[0.9215686274509803,0.5647058823529412,0.8901960784313725],
[0.9411764705882353,0.5137254901960784,0.9215686274509803],
[0.7647058823529411,0.4392156862745098,0.8117647058823529],
[0.5882352941176471,0.36470588235294116,0.7019607843137254],
[0.4117647058823529,0.2901960784313726,0.592156862745098],
[0.23529411764705882,0.21568627450980393,0.4823529411764706],
[0.0196078431372549,0.16862745098039217,0.49411764705882355],
[0.08235294117647059,0.23921568627450981,0.5607843137254902],
[0.14901960784313725,0.3137254901960784,0.6313725490196078],
[0.21568627450980393,0.38823529411764707,0.7019607843137254],
[0.2823529411764706,0.4627450980392157,0.7725490196078432],
[0.34901960784313724,0.5372549019607843,0.8431372549019608],
[0.41568627450980394,0.611764705882353,0.9137254901960784],
[0.4823529411764706,0.6862745098039216,0.984313725490196],
[0.6588235294117647,0.7098039215686275,0.984313725490196],
[0.6509803921568628,0.6352941176470588,0.8745098039215686],
[0.6431372549019608,0.5568627450980392,0.7607843137254902],
[0.6352941176470588,0.47843137254901963,0.6509803921568628],
[0.6274509803921569,0.4,0.5372549019607843],
[0.7294117647058823,0.5294117647058824,0.6549019607843137],
[0.8313725490196079,0.6627450980392157,0.7725490196078432],
[0.9372549019607843,0.796078431372549,0.8901960784313725]
], dtype=float32)
IRIDESCENTCRYSTAL =_IRIDESCENTCRYSTAL_NORMALIZED * <float>255.0


# https://lospec.com/palette-list/aap-splendor128
# Created by Adigun A. Polack
_AAPSPLENDOR128_NORMALIZED = array([
[0.0196078431372549,0.01568627450980392,0.011764705882352941],
[0.054901960784313725,0.047058823529411764,0.047058823529411764],
[0.17647058823529413,0.10588235294117647,0.11764705882352941],
[0.3803921568627451,0.15294117647058825,0.12941176470588237],
[0.7254901960784313,0.27058823529411763,0.11372549019607843],
[0.9450980392156862,0.39215686274509803,0.12156862745098039],
[0.9882352941176471,0.6470588235294118,0.4392156862745098],
[1.0,0.8784313725490196,0.7176470588235294],
[1.0,1.0,1.0],
[1.0,0.9411764705882353,0.5372549019607843],
[0.9725490196078431,0.7725490196078432,0.22745098039215686],
[0.9098039215686274,0.5411764705882353,0.21176470588235294],
[0.6901960784313725,0.3568627450980392,0.17254901960784313],
[0.403921568627451,0.2235294117647059,0.19215686274509805],
[0.15294117647058825,0.12156862745098039,0.10588235294117647],
[0.2980392156862745,0.23921568627450981,0.1803921568627451],
[0.5215686274509804,0.37254901960784315,0.2235294117647059],
[0.8274509803921568,0.592156862745098,0.2549019607843137],
[0.9725490196078431,0.9647058823529412,0.26666666666666666],
[0.8352941176470589,0.8627450980392157,0.11372549019607843],
[0.6784313725490196,0.7215686274509804,0.20392156862745098],
[0.4980392156862745,0.5568627450980392,0.26666666666666666],
[0.34509803921568627,0.38823529411764707,0.20784313725490197],
[0.2,0.23529411764705882,0.1411764705882353],
[0.09411764705882353,0.10980392156862745,0.09803921568627451],
[0.1607843137254902,0.24705882352941178,0.12941176470588237],
[0.2784313725490196,0.4470588235294118,0.2196078431372549],
[0.3803921568627451,0.6470588235294118,0.24705882352941178],
[0.5607843137254902,0.8156862745098039,0.19607843137254902],
[0.7686274509803922,0.9450980392156862,0.1607843137254902],
[0.8156862745098039,1.0,0.9176470588235294],
[0.592156862745098,0.9294117647058824,0.792156862745098],
[0.34901960784313724,0.8117647058823529,0.5764705882352941],
[0.25882352941176473,0.6431372549019608,0.34901960784313724],
[0.23921568627450981,0.43529411764705883,0.2627450980392157],
[0.15294117647058825,0.2549019607843137,0.17647058823529413],
[0.0784313725490196,0.07058823529411765,0.11372549019607843],
[0.10588235294117647,0.1411764705882353,0.2784313725490196],
[0.16862745098039217,0.3058823529411765,0.5843137254901961],
[0.15294117647058825,0.5372549019607843,0.803921568627451],
[0.25882352941176473,0.7490196078431373,0.9098039215686274],
[0.45098039215686275,0.9372549019607843,0.9098039215686274],
[0.9450980392156862,0.9490196078431372,1.0],
[0.788235294117647,0.8313725490196079,0.9921568627450981],
[0.5411764705882353,0.6313725490196078,0.9647058823529412],
[0.27058823529411763,0.4470588235294118,0.8901960784313725],
[0.28627450980392155,0.2549019607843137,0.5098039215686274],
[0.47058823529411764,0.39215686274509803,0.7764705882352941],
[0.611764705882353,0.5450980392156862,0.8588235294117647],
[0.807843137254902,0.6666666666666666,0.9294117647058824],
[0.9803921568627451,0.8392156862745098,1.0],
[0.9333333333333333,0.7098039215686275,0.611764705882353],
[0.8313725490196079,0.5019607843137255,0.7333333333333333],
[0.5647058823529412,0.3215686274509804,0.7372549019607844],
[0.09019607843137255,0.08235294117647059,0.08627450980392157],
[0.21568627450980393,0.2,0.20392156862745098],
[0.4117647058823529,0.3568627450980392,0.34901960784313724],
[0.6980392156862745,0.5450980392156862,0.47058823529411764],
[0.8862745098039215,0.6980392156862745,0.49411764705882355],
[0.9647058823529412,0.8470588235294118,0.5882352941176471],
[0.9882352941176471,0.9686274509803922,0.7450980392156863],
[0.9254901960784314,0.9215686274509803,0.9058823529411765],
[0.796078431372549,0.7764705882352941,0.7568627450980392],
[0.6509803921568628,0.6196078431372549,0.6039215686274509],
[0.5019607843137255,0.4823529411764706,0.47843137254901963],
[0.34901960784313724,0.3411764705882353,0.3411764705882353],
[0.19607843137254902,0.19607843137254902,0.19607843137254902],
[0.30980392156862746,0.20392156862745098,0.1843137254901961],
[0.5490196078431373,0.3568627450980392,0.24313725490196078],
[0.7764705882352941,0.5215686274509804,0.33725490196078434],
[0.8392156862745098,0.6588235294117647,0.3176470588235294],
[0.7058823529411765,0.4588235294117647,0.2196078431372549],
[0.4470588235294118,0.29411764705882354,0.17254901960784313],
[0.27058823529411763,0.16470588235294117,0.10588235294117647],
[0.3803921568627451,0.40784313725490196,0.22745098039215686],
[0.5764705882352941,0.5803921568627451,0.27450980392156865],
[0.7764705882352941,0.7215686274509804,0.34509803921568627],
[0.9372549019607843,0.8666666666666667,0.5686274509803921],
[0.7098039215686275,0.9058823529411765,0.796078431372549],
[0.5254901960784314,0.7764705882352941,0.6039215686274509],
[0.36470588235294116,0.6078431372549019,0.4745098039215686],
[0.2823529411764706,0.40784313725490196,0.34901960784313724],
[0.17254901960784313,0.23137254901960785,0.2235294117647059],
[0.09019607843137255,0.09411764705882353,0.09803921568627451],
[0.17254901960784313,0.20392156862745098,0.2196078431372549],
[0.27450980392156865,0.32941176470588235,0.33725490196078434],
[0.39215686274509803,0.5294117647058824,0.5490196078431373],
[0.5411764705882353,0.7686274509803922,0.7647058823529411],
[0.6862745098039216,0.9137254901960784,0.8745098039215686],
[0.8627450980392157,0.9176470588235294,0.9333333333333333],
[0.7215686274509804,0.8,0.8470588235294118],
[0.5333333333333333,0.6392156862745098,0.7372549019607844],
[0.3686274509803922,0.44313725490196076,0.5568627450980392],
[0.2823529411764706,0.3215686274509804,0.3843137254901961],
[0.1568627450980392,0.17254901960784313,0.23529411764705882],
[0.27450980392156865,0.2784313725490196,0.3843137254901961],
[0.4117647058823529,0.4,0.5098039215686274],
[0.6039215686274509,0.592156862745098,0.7254901960784313],
[0.7725490196078432,0.7803921568627451,0.8666666666666667],
[0.9019607843137255,0.9058823529411765,0.9411764705882353],
[0.9333333333333333,0.9019607843137255,0.9176470588235294],
[0.8901960784313725,0.803921568627451,0.8745098039215686],
[0.7490196078431373,0.6470588235294118,0.788235294117647],
[0.5294117647058824,0.45098039215686275,0.5607843137254902],
[0.33725490196078434,0.30980392156862746,0.3568627450980392],
[0.19607843137254902,0.1843137254901961,0.20784313725490197],
[0.21176470588235294,0.1568627450980392,0.16862745098039217],
[0.396078431372549,0.28627450980392155,0.33725490196078434],
[0.5882352941176471,0.40784313725490196,0.5333333333333333],
[0.7529411764705882,0.5647058823529412,0.6627450980392157],
[0.8313725490196079,0.7215686274509804,0.7215686274509804],
[0.9176470588235294,0.8784313725490196,0.8666666666666667],
[0.9450980392156862,0.9215686274509803,0.8588235294117647],
[0.8666666666666667,0.807843137254902,0.7490196078431373],
[0.7411764705882353,0.6431372549019608,0.6],
[0.5333333333333333,0.43137254901960786,0.41568627450980394],
[0.34901960784313724,0.30196078431372547,0.30196078431372547],
[0.2,0.15294117647058825,0.16470588235294117],
[0.6980392156862745,0.5803921568627451,0.4627450980392157],
[0.8823529411764706,0.7490196078431373,0.5372549019607843],
[0.9725490196078431,0.8901960784313725,0.596078431372549],
[1.0,0.9137254901960784,0.8901960784313725],
[0.9921568627450981,0.788235294117647,0.788235294117647],
[0.9647058823529412,0.6352941176470588,0.6588235294117647],
[0.8862745098039215,0.4470588235294118,0.5215686274509804],
[0.6980392156862745,0.3215686274509804,0.4],
[0.39215686274509803,0.21176470588235294,0.29411764705882354],
[0.16470588235294117,0.11764705882352941,0.13725490196078433]
], dtype=float32)
AAPSPLENDOR128 =_AAPSPLENDOR128_NORMALIZED * <float>255.0




# https://lospec.com/palette-list/lospec500
# collaboration
_LOSPEC500_NORMALIZED = array([
[0.06274509803921569,0.07058823529411765,0.10980392156862745],
[0.17254901960784313,0.11764705882352941,0.19215686274509805],
[0.4196078431372549,0.14901960784313725,0.2627450980392157],
[0.6745098039215687,0.1568627450980392,0.2784313725490196],
[0.9254901960784314,0.15294117647058825,0.24705882352941178],
[0.5803921568627451,0.28627450980392155,0.22745098039215686],
[0.8705882352941177,0.36470588235294116,0.22745098039215686],
[0.9137254901960784,0.5215686274509804,0.21568627450980393],
[0.9529411764705882,0.6588235294117647,0.2],
[0.30196078431372547,0.20784313725490197,0.2],
[0.43137254901960786,0.2980392156862745,0.18823529411764706],
[0.6352941176470588,0.42745098039215684,0.24705882352941178],
[0.807843137254902,0.5725490196078431,0.2823529411764706],
[0.8549019607843137,0.6941176470588235,0.38823529411764707],
[0.9098039215686274,0.8235294117647058,0.5098039215686274],
[0.9686274509803922,0.9529411764705882,0.7176470588235294],
[0.11764705882352941,0.25098039215686274,0.26666666666666666],
[0.0,0.396078431372549,0.32941176470588235],
[0.14901960784313725,0.5215686274509804,0.2980392156862745],
[0.35294117647058826,0.7098039215686275,0.3215686274509804],
[0.615686274509804,0.9019607843137255,0.3058823529411765],
[0.0,0.5450980392156862,0.5450980392156862],
[0.3843137254901961,0.6431372549019608,0.4666666666666667],
[0.6509803921568628,0.796078431372549,0.5882352941176471],
[0.8274509803921568,0.9333333333333333,0.8274509803921568],
[0.24313725490196078,0.23137254901960785,0.396078431372549],
[0.2196078431372549,0.34901960784313724,0.7019607843137254],
[0.2,0.5333333333333333,0.8705882352941177],
[0.21176470588235294,0.7725490196078432,0.9568627450980393],
[0.42745098039215684,0.9176470588235294,0.8392156862745098],
[0.3686274509803922,0.3568627450980392,0.5490196078431373],
[0.5490196078431373,0.47058823529411764,0.6470588235294118],
[0.6901960784313725,0.6549019607843137,0.7215686274509804],
[0.8705882352941177,0.807843137254902,0.9294117647058824],
[0.6039215686274509,0.30196078431372547,0.4627450980392157],
[0.7843137254901961,0.47058823529411764,0.6862745098039216],
[0.8,0.6,1.0],
[0.9803921568627451,0.43137254901960786,0.4745098039215686],
[1.0,0.6352941176470588,0.6745098039215687],
[1.0,0.8196078431372549,0.8352941176470589],
[0.9647058823529412,0.9098039215686274,0.8784313725490196],
[1.0,1.0,1.0]
], dtype=float32)
LOSPEC500 =_LOSPEC500_NORMALIZED * <float>255.0





# https://lospec.com/palette-list/famicube
# Created by Arne as part of his Famicube Project.
_FAMICUBE_NORMALIZED = array([
[0.0,0.0,0.0],
[0.8784313725490196,0.23529411764705882,0.1568627450980392],
[1.0,1.0,1.0],
[0.8431372549019608,0.8431372549019608,0.8431372549019608],
[0.6588235294117647,0.6588235294117647,0.6588235294117647],
[0.4823529411764706,0.4823529411764706,0.4823529411764706],
[0.20392156862745098,0.20392156862745098,0.20392156862745098],
[0.08235294117647059,0.08235294117647059,0.08235294117647059],
[0.050980392156862744,0.12549019607843137,0.18823529411764706],
[0.2549019607843137,0.36470588235294116,0.4],
[0.44313725490196076,0.6509803921568628,0.6313725490196078],
[0.7411764705882353,1.0,0.792156862745098],
[0.1450980392156863,0.8862745098039215,0.803921568627451],
[0.0392156862745098,0.596078431372549,0.6745098039215687],
[0.0,0.3215686274509804,0.5019607843137255],
[0.0,0.3764705882352941,0.29411764705882354],
[0.12549019607843137,0.7098039215686275,0.3843137254901961],
[0.34509803921568627,0.8274509803921568,0.19607843137254902],
[0.07450980392156863,0.615686274509804,0.03137254901960784],
[0.0,0.3058823529411765,0.0],
[0.09019607843137255,0.1568627450980392,0.03137254901960784],
[0.21568627450980393,0.42745098039215684,0.011764705882352941],
[0.41568627450980394,0.7058823529411765,0.09019607843137255],
[0.5490196078431373,0.8392156862745098,0.07058823529411765],
[0.7450980392156863,0.9215686274509803,0.44313725490196076],
[0.9333333333333333,1.0,0.6627450980392157],
[0.7137254901960784,0.7568627450980392,0.12941176470588237],
[0.5764705882352941,0.592156862745098,0.09019607843137255],
[0.8,0.5607843137254902,0.08235294117647059],
[1.0,0.7333333333333333,0.19215686274509805],
[1.0,0.9058823529411765,0.21568627450980393],
[0.9647058823529412,0.5607843137254902,0.21568627450980393],
[0.6784313725490196,0.3058823529411765,0.10196078431372549],
[0.13725490196078433,0.09019607843137255,0.07058823529411765],
[0.3607843137254902,0.23529411764705882,0.050980392156862744],
[0.6823529411764706,0.4235294117647059,0.21568627450980393],
[0.7725490196078432,0.592156862745098,0.5098039215686274],
[0.8862745098039215,0.8431372549019608,0.7098039215686275],
[0.30980392156862746,0.08235294117647059,0.027450980392156862],
[0.5098039215686274,0.23529411764705882,0.23921568627450981],
[0.8549019607843137,0.396078431372549,0.3686274509803922],
[0.8823529411764706,0.5098039215686274,0.5372549019607843],
[0.9607843137254902,0.7176470588235294,0.5176470588235295],
[1.0,0.9137254901960784,0.7725490196078432],
[1.0,0.5098039215686274,0.807843137254902],
[0.8117647058823529,0.23529411764705882,0.44313725490196076],
[0.5294117647058824,0.08627450980392157,0.27450980392156865],
[0.6392156862745098,0.1568627450980392,0.7019607843137254],
[0.8,0.4117647058823529,0.8941176470588236],
[0.8352941176470589,0.611764705882353,0.9882352941176471],
[0.996078431372549,0.788235294117647,0.9294117647058824],
[0.8862745098039215,0.788235294117647,1.0],
[0.6509803921568628,0.4588235294117647,0.996078431372549],
[0.41568627450980394,0.19215686274509805,0.792156862745098],
[0.35294117647058826,0.09803921568627451,0.5686274509803921],
[0.12941176470588237,0.08627450980392157,0.25098039215686274],
[0.23921568627450981,0.20392156862745098,0.6470588235294118],
[0.3843137254901961,0.39215686274509803,0.8627450980392157],
[0.6078431372549019,0.6274509803921569,0.9372549019607843],
[0.596078431372549,0.8627450980392157,1.0],
[0.3568627450980392,0.6588235294117647,1.0],
[0.0392156862745098,0.5372549019607843,1.0],
[0.00784313725490196,0.2901960784313726,0.792156862745098],
[0.0,0.09019607843137255,0.49019607843137253]
], dtype=float32)
FAMICUBE =_FAMICUBE_NORMALIZED * <float>255.0


# https://lospec.com/palette-list/smooth-polished-silver
_SILVER_NORMALIZED = array([
[0.0,0.0,0.0],
[0.1568627450980392,0.12941176470588237,0.12156862745098039],
[0.2901960784313726,0.2235294117647059,0.23921568627450981],
[0.403921568627451,0.34509803921568627,0.2980392156862745],
[0.6901960784313725,0.5725490196078431,0.6549019607843137],
[0.9490196078431372,0.8392156862745098,0.8862745098039215],
[1.0,1.0,1.0]
], dtype=float32)
SILVER =_SILVER_NORMALIZED * <float>255.0



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef void palette_change(
        object image_surface,
        object color_palette,
        object temp_buffer
):
    """
    Apply a color palette transformation to an image in-place.

    This function modifies an image by changing its color palette. It uses a given palette 
    to substitute the colors of the input image (represented by a Pygame surface) with new 
    colors from the palette. The process is done in-place, meaning the original image is directly 
    updated without the need for a new surface to be created.

    The available palettes are stored in `LIST_PALETTES` within the PygameShader project.

    Example usage:
    ```python
    from PygameShader.Palette import LIST_PALETTES
    print(LIST_PALETTES.keys())
    ```

    To optimize performance, the function requires a temporary array (`temp_buffer`) that should be pre-declared 
    before each function call. This temporary array improves efficiency by avoiding the need to allocate memory 
    during each frame.

    Example of temporary array declaration:
    ```python
    temp_buffer = numpy.ascontiguousarray(numpy.ndarray(
        (SURFACE.get_width() * SURFACE.get_height(), IRIDESCENTCRYSTAL.shape[0]),
        dtype=float32
    ))
    ```

    :param image_surface: 
        The Pygame surface (`pygame.Surface`) representing the image to be modified.
        It is the input image on which the color palette will be applied.

    :param color_palette: 
        A `numpy.ndarray` containing the colors to use for the palette transformation.
        The array format should be (w, 3), where `w` is the number of colors, and the type should be `float` 
        with values in the range [0.0, 255.0].

        Example:
        ```python
        from PygameShader import IRIDESCENTCRYSTAL
        ```

    :param temp_buffer: 
        A temporary `numpy.ndarray` (contiguous array) used for optimization. 
        It should have the shape `(rgb_array.get_width() * rgb_array.get_height(), 
        len(color_palette.shape[0]))` 
        and type `float32`. This array should be pre-allocated before calling the function to improve performance 
        by avoiding reallocation every frame.

        Example:
        ```python
        temp_buffer = numpy.ascontiguousarray(numpy.ndarray(
            (SURFACE.get_width() * SURFACE.get_height(), IRIDESCENTCRYSTAL.shape[0]),
            dtype=float32
        ))
        ```

    :return: 
        None. This function modifies the `rgb_array` in-place.

    :raises ValueError: 
        If the input surface is not compatible or if an invalid palette is provided.
    :raises TypeError:    
    """

    if not isinstance(image_surface, Surface):
        raise TypeError(
            f"Invalid type for argument 'rgb_array'. Expected a valid pygame.Surface instance, "
            f"but got {type(image_surface).__name__}. Please ensure you pass a valid Pygame Surface object."
        )

    if not isinstance(color_palette, ndarray):
        raise TypeError(
            f"Invalid type for argument 'color_palette'. Expected a numpy.ndarray, "
            f"but got {type(color_palette).__name__}. Please provide a numpy array containing the color palette."
        )

    if not isinstance(temp_buffer, ndarray):
        raise TypeError(
            f"Invalid type for argument 'temp_buffer'. Expected a numpy.ndarray, "
            f"but got {type(temp_buffer).__name__}. Please pass a numpy array for the temporary buffer."
        )

    if not temp_buffer.flags[ 'C_CONTIGUOUS' ]:
        raise TypeError(
            "Argument 'temp_buffer' must be a contiguous array (C_CONTIGUOUS). "
            "This means the memory layout must be contiguous, which is required for optimal performance. "
            "Ensure that the array passed is contiguous using numpy.ascontiguousarray() if needed."
        )

    # Ensure both 'temp_buffer' and 'color_palette' are of dtype float32
    if temp_buffer.dtype != float32 or color_palette.dtype != float32:
        raise ValueError(
            "Both 'temp_buffer' and 'color_palette' must be of type numpy.float32. "
            "Found 'temp_buffer' dtype: '%s', and 'color_palette' dtype: '%s'." %
            (temp_buffer.dtype, color_palette.dtype)
        )

    cdef:
        unsigned char [:, :, :] rgb_array
        float [:, :] palette = color_palette
        float[:, ::1] tmp_buffer = temp_buffer
        Py_ssize_t w, h, p_length

    try:
        rgb_array = image_surface.get_view('3')
    except (pygame.error, ValueError):
        raise ValueError(
            "Failed to get a valid view from 'rgb_array'. Ensure the surface is a valid "
            "Pygame Surface with a compatible format."
        )

    w, h = rgb_array.shape[ 0 ], rgb_array.shape[ 1 ]
    p_length = <object> color_palette.shape[ 0 ]

    # Check if the dimensions of the 'rgb_array' and 'tmp_buffer' match
    if w * h!= temp_buffer.shape[ 0 ]:
        raise ValueError(
            f"Dimension mismatch between 'rgb_array' and 'temp_buffer'. "
            f"The 'rgb_array' array has shape {(<object>rgb_array).shape} while the 'temp_buffer' array has shape "
            f"{(<object>temp_buffer).shape}. "
            f"Please ensure both arrays have the same height, width, and number of dimensions."
        )

    # Now call the palette change function
    palette_change_c(w, h, p_length, rgb_array, palette, tmp_buffer)



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef inline void palette_change_c(
        const Py_ssize_t w,
        const Py_ssize_t h,
        const Py_ssize_t p_length,
        unsigned char [:, :, :] rgb_array,
        const float [:, :] color_palette,
        float [:, ::1] temp_buffer
        )nogil:
    """
    Change the colors of an image by matching each pixel to the closest color in a given palette.

    This function calculates the Euclidean distance between the RGB values of each pixel
    in the input image and the colors in the provided palette. The pixel is then replaced
    with the palette color that has the smallest distance.

    The input color palette is expected to be in RGB format with values in the range of [0.0, 255.0].

    Parameters:
    -----------
    rgb_array : numpy.ndarray
        A 3D array representing the RGB values of the image, with shape (width, height, 3).
        The dtype should be uint8, and the pixel values should be in the range [0, 255].

    color_palette : numpy.ndarray
        A 2D array containing the RGB colors to substitute for the image pixels, with shape (num_colors, 3).
        The dtype should be float, with RGB values in the range [0.0, 255.0].

    temp_buffer : numpy.ndarray
        A 2D contiguous array of shape (width * height, num_colors), used to store the squared distances
        between each image pixel and each color in the palette. The dtype should be float32 for performance reasons.
        This buffer helps avoid reallocation in each frame and speeds up the calculation.

    Notes:
    ------
    - The algorithm computes the Euclidean distance between the current pixel's RGB 
        values and all colors in the palette.
    - The color with the smallest distance is chosen to replace the original pixel.
    - The current implementation assumes that only the first closest color will be selected, 
        even if multiple colors have the same distance.
    - This function is optimized for multi-threading and uses OpenMP-style parallelization to speed up the process.

    Example:
    --------
    # Using the function to change an image's colors based on a predefined palette
    from PygameShader import IRIDESCENTCRYSTAL

    # Create temporary buffer
    tmp_buffer = numpy.ascontiguousarray(
    numpy.zeros((image_width * image_height, len(IRIDESCENTCRYSTAL)), dtype=numpy.float32))

    # Call palette_change_c to apply the color transformation
    palette_change_c(image_surface, IRIDESCENTCRYSTAL, tmp_buffer)
    """

    cdef:
        Py_ssize_t i, j, k
        const float *p2
        float *p3
        float min_v  # Minimum distance found for the current pixel
        unsigned int index  # Index of the closest color in the palette
        unsigned int tmp_index  # Flat index used for accessing the 1D temp_buffer

    # Parallelize the pixel processing across multiple threads
    with nogil:
        for j in prange(h, schedule=SCHEDULE, num_threads=THREADS):  # Process each row (y-axis)

            for i in range(w):  # Process each column (x-axis)

                tmp_index = j * w + i  # Flatten 2D coordinates (i, j) into 1D index

                # Retrieve the RGB values for the current pixel
                p3 = &temp_buffer[ tmp_index, 0 ]

                for k in range(p_length): #, schedule=SCHEDULE, num_threads=THREADS):

                    p2 = &color_palette[ k, 0 ]

                    (p3 + k)[ 0 ] = (<float>rgb_array[ i, j, 0 ] - <float>p2[ 0 ]) ** 2 + \
                                         (<float>rgb_array[ i, j, 1 ] - <float>(p2 + 1)[ 0 ]) ** 2 + \
                                         (<float>rgb_array[ i, j, 2 ] - <float>(p2 + 2)[ 0 ]) ** 2

                # Find the index of the closest color in the palette by checking the minimum distance
                index = 0

                min_v = p3[0]
                for k in range(1, p_length):  # Start from the second color, as the first is already selected

                    if (p3 + k)[ 0 ] < min_v:
                        min_v = (p3 + k)[ 0 ]
                        index = k

                # Replace the current pixel's RGB values with the closest palette color
                rgb_array[ i, j, 0 ] = <unsigned char>color_palette[index, 0]
                rgb_array[ i, j, 1 ] = <unsigned char>color_palette[index, 1]
                rgb_array[ i, j, 2 ] = <unsigned char>color_palette[index, 2]




@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef inline object make_palette(
        const int width,
        const float fh,
        const float fs,
        const float fl
):
    """
    Create a palette of rgb color values mapped from hsl

    This function generates a palette of RGB colors based on the given parameters.
    The colors are derived from HSL (Hue, Saturation, Lightness) space and then converted to RGB.
    Each color's hue is scaled by the `fh` factor, saturation is clipped to a maximum of 
    `fs` (with a maximum value of 255.0), and lightness is clipped to a maximum of `fl`
    (also with a maximum value of 255.0).

    Example:
        # Create a palette of 256 colors where:
        # - Hue is scaled by a factor of 6
        # - Saturation is fixed at 255
        # - Lightness is scaled by a factor of 2
        palette = make_palette(256, 6, 255, 2)

        # Another palette with different settings:
        palette = make_palette(256, 4, 255, 2)

    :param width  : 
        int, The number of colors (palette size) to generate.
    
    :param fh     : 
        float, A factor by which to scale the hue value for each color.
        
    :param fs     : 
        float, The saturation value, which must be in the range (0.0 ... 255.0). 
        This limits the saturation intensity.
        
    :param fl     : 
        float, A factor by which to scale the lightness value for each color.
        
    :return       : 
        numpy.ndarray, A 1D array of RGB color values corresponding to the generated palette.

    Notes:
        The function relies on converting from HSL to RGB, applying the specified factors and clipping values.
        The output is a NumPy array, which can be directly used for visualization or further processing.
    """

    return asarray(make_palette_c(width, fh, fs, fl))



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cdef inline unsigned int [::1] make_palette_c(
        const int width,
        const float fh,
        const float fs,
        const float fl):
    """
    Create a palette of rgb color values mapped from hsl (hue, saturation, lightness)

    This function generates a palette of RGB colors based on the HSL 
    (Hue, Saturation, Lightness) color model.
    Each color in the palette is calculated using the following transformations:
    - The hue (`h`) is scaled by the `fh` factor.
    - The saturation (`s`) is clipped to the `fs` maximum value (range 0 to 255).
    - The lightness (`l`) is clipped to the `fl` maximum value (range 0 to 255).

    The resulting HSL values are converted to RGB, which are then represented as integer values in a 1D array. 
    This palette is designed to be used for generating color gradients, color maps, or visual effects.

    Example:
        # Create a palette of 256 colors:
        # - Hue is scaled by a factor of 6
        # - Saturation is fixed at 255
        # - Lightness is scaled by a factor of 2
        palette = make_palette_c(256, 6, 255, 2)

    :param width  : 
        int, The number of colors (palette size) to generate.
    
    :param fh     : 
        float, Hue scaling factor.
    
    :param fs     : 
        float, Saturation scaling factor (must be in the range 0 to 255).
    
    :param fl     : 
        float, Lightness scaling factor (must be in the range 0 to 255).
    
    :return       : 
        unsigned int [::1], A 1D array of RGB color values represented as 32-bit unsigned integers.

    Notes:
        - The palette is constructed as an array of RGB integer values, 
          where each value is a packed 32-bit integer.
        - The `struct_hsl_to_rgb` function is used to convert HSL values to RGB.
        - The RGB values are then scaled to fit the 0-255 range and packed into a single integer.
        - The `pygame.Surface` object that would be returned in some use cases is
          implied but not directly returned here.
    """

    # Ensure the width is a positive integer
    assert width > 0, "Argument width should be > 0, got %s" % width

    # Declare local variables and set up the palette array to store RGB values
    cdef:
        unsigned int [::1] palette = empty(width, uint32)
        int x
        float h, s, l
        rgb rgb_

    # Perform the palette creation inside a 'with nogil' block to release the GIL
    with nogil:
        # Loop through the palette indices in parallel using prange
        for x in prange(width, schedule=SCHEDULE, num_threads=THREADS):
            # Calculate hue, saturation, and lightness for the current color index
            h, s, l = <float>x * fh, min(fs, <float>255.0), min(<float>x * fl, <float>255.0)

            # Convert the HSL values to RGB using the struct_hsl_to_rgb function
            rgb_ = struct_hsl_to_rgb(h * <float>ONE_360, s * <float>ONE_255, l * <float>ONE_255)

            # Store the RGB value as a 32-bit integer in the palette array
            palette[x] = rgb_to_int_c(
                <unsigned int>(rgb_.r * <float>255.0),
                <unsigned int>(rgb_.g * <float>255.0),
                <unsigned int>(rgb_.b * <float>255.0 * <float>0.5)
            )

    # Return the palette of RGB values
    return palette



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.initializedcheck(False)
@cython.exceptval(check=False)
cpdef object create_surface_from_palette(unsigned int [::1] palette_c):
    """
    Converts a 1D array of RGB palette colors into a Pygame Surface (line).

    This function takes a 1D array of RGB colors, typically generated by the
    `make_palette` function (where each color is stored as a 32-bit unsigned
    integer), and converts it into a Pygame `Surface` object. The colors are
    extracted and placed into a 2D array format suitable for Pygame's surface
    creation.

    Example:
    # Assuming 'palette' is a palette generated with make_palette:
    surface = create_surface_from_palette(palette)

    :param palette_c: 
        numpy.ndarray, A 1D array containing the palette of colors.
        The array should have `unsigned int` type, where each color is represented as a
        32-bit unsigned integer (ARGB format).

    :return: 
        object, A Pygame Surface object created from the palette.

    :raises TypeError: If `palette_c` is not a numpy ndarray or has an incorrect dtype.
    :raises ValueError: If `palette_c` is empty, not contiguous, or has invalid size.

    Notes:
    - The function assumes that the input palette contains RGB values packed into a 32-bit integer.
    - The function processes each color in the palette in parallel for performance optimization.
    - The output is a Pygame Surface object, which can be used for rendering or visualization.
    
    """

    # Validate input
    if not (isinstance(palette_c, ndarray) or is_type_memoryview(palette_c)):
        raise TypeError("Expected input type numpy.ndarray or memoryviewslice, "
                        "but got %s" % type(palette_c).__name__)

    if not is_type_memoryview(palette_c):
        if palette_c.dtype != uint32:
            raise TypeError("Expected input array with dtype 'numpy.uint32', but got %s" % palette_c.dtype)

        if not palette_c.flags['C_CONTIGUOUS']:
            raise ValueError("Input palette must be a contiguous array. The given array is not contiguous.")

        if palette_c.size == 0:
            raise ValueError("Input palette is empty. The palette should contain at least one color.")

    else:
        if len(palette_c) == 0:
            raise ValueError("Input palette is empty. The palette should contain at least one color.")

    # Initialize variables
    cdef:
        int i = 0
        int s = len(<object>palette_c)  # Get the length of the palette array
        unsigned char r, g, b   # Variable to store the RGB color extracted from the palette
        unsigned char [:, :, :] palette_array = empty((s, 1, 3), dtype=uint8)
        # 2D array to store RGB values in uint8 format

    # Process the palette in parallel using 'with nogil' to release the Global Interpreter Lock (GIL)
    with nogil:
        # Loop through the palette array in parallel
        for i in prange(s, schedule=SCHEDULE, num_threads=THREADS):
            # Convert each palette color (32-bit ARGB) into RGB components

            r,g,b = int_to_rgb_c(palette_c[i])

            # Store the individual RGB components in the 2D array
            palette_array[i, 0, 0] = <unsigned char>r  # Red channel
            palette_array[i, 0, 1] = <unsigned char>g  # Green channel
            palette_array[i, 0, 2] = <unsigned char>b  # Blue channel

    # Convert the array to a numpy array and pass it to the make_surface function to create a Pygame surface
    try:
        surface = make_surface(asarray(palette_array))
    except Exception as e:
        raise RuntimeError(f"Error creating Pygame surface: {str(e)}")

    return surface