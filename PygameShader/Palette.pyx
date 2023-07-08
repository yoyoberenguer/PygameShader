# cython: binding=False, boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, profile=False
# cython: optimize.use_switch=True
# cython: warn.maybe_uninitialized=False
# cython: warn.unused=False
# cython: warn.unused_result=False
# cython: warn.unused_arg=False
# cython: language_level=3
# encoding: utf-8

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
           " However, its 320×200 graphics mode is restricted to fixed palettes containing " \
           "only four colors, and the 640×200 graphic mode is only two colors. 16 colors are" \
           " only available in text mode or the tweaked text 160×100 mode. A different set" \
           " of 16 colors is available in composite mode with an NTSC composite monitor. " \
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
        "used following a graphics color attribute. Up to a maximum of 72×69 blocky pixels"
        " can be used on a page.",

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
        "The Apple II features low-res and hi-res modes. The 40x48 pixel lo-res mode allowed"
        " 15 different colors plus a duplicate gray(**).[12] * The majority of Apple graphic"
        " applications used the hi-res mode, which had 280×192 pixels (effectively 140x192"
        " on a color monitor). The hi-res mode allowed six colors: black, white, blue, orange,"
        "green and purple.",
    "APPLE_II_HIGH"      :
        "The Apple II series features a 16-color composite video palette, based on the YIQ"
        " color space used by the NTSC color TV system.[10][11]."
        "The Apple II features low-res and hi-res modes. The 40x48 pixel lo-res mode allowed"
        " 15 different colors plus a duplicate gray(**).[12] * The majority of Apple graphic"
        " applications used the hi-res mode, which had 280×192 pixels (effectively 140x192"
        " on a color monitor). The hi-res mode allowed six colors: black, white, blue, orange,"
        "green and purple.",
    "APPLE_II"           :
        "The Apple II series features a 16-color composite video palette, based on the YIQ"
        " color space used by the NTSC color TV system.[10][11]."
        "The Apple II features low-res and hi-res modes. The 40x48 pixel lo-res mode allowed"
        " 15 different colors plus a duplicate gray(**).[12] * The majority of Apple graphic"
        " applications used the hi-res mode, which had 280×192 pixels (effectively 140x192"
        " on a color monitor). The hi-res mode allowed six colors: black, white, blue, orange,"
        "green and purple.",
    "COMMODORE_64"       :
        "The Commodore 64 has two graphic modes: Multicolor and High Resolution."
        "In the Multicolor 160×200, 16-color mode, every cell of 4×8, 2:1 aspect ratio "
        "pixels can have one of four colors: one shared with the entire screen, the two"
        " background and foreground colors of the corresponding text mode character, and"
        " one more color also stored in the color RAM area, all of them freely selectable"
        " among the entire palette.In the High Resolution 320×200, 16-color mode, every "
        "cell of 8×8 pixels can have one of the two background and foreground colors of the"
        " correspondent text mode character, both freely selectable among the entire palette.",
    "AMSTRAD_CPC"        :
        "The Amstrad CPC 464/664/6128 series of computers generates the available palette with"
        " 3 levels (not bits) for every RGB primary. Thus, there are 27 different RGB "
        "combinations, from which 16 can be simultaneously displayed in low resolution mode,"
        " four in medium resolution mode and two in high resolution mode.[7]",
    "MSX"                :
        "The MSX series has two graphic modes. The MSX BASIC Screen 3 mode is a low-resolution"
        " mode with 15 colors, in which every pixel can be any of the 15 available colors. "
        "Screen mode 2 is a 256×192 high-resolution mode with 15 colors, in which each of "
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
        "Microsoft Windows default 20-color palette.In 256-color mode,"
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

import numpy

_TELETEXT_NORMALIZED = numpy.array([
    [ 0., 0., 0. ],
    [ 0., 0., 255.0/255.0 ],
    [ 255.0/255.0, 0.0, 0.0 ],
    [ 255.0/255.0, 0.0, 255.0/255.0 ],
    [ 0., 255.0/255.0, 0.0 ],
    [ 0.0, 255.0/255.0, 255.0/255.0 ],
    [ 255.0/255.0, 255.0/255.0, 0.0 ],
    [ 255.0/255.0, 255.0/255.0, 255.0/255.0]
], dtype=numpy.float32)
TELETEXT = numpy.multiply(_TELETEXT_NORMALIZED, 255.0).astype(numpy.float32)

BBC_MICRO = TELETEXT

_CGA_MODE4_PAL1_NORMALIZED_LOW = numpy.array([
    [ 0., 0., 0. ],
    [ 0., 0.66666, 0.6666 ],
    [ 0.66666, 0., 0.6666 ],
    [ 0.66666, 0.6666, 0.66666 ]
], dtype=numpy.float32)
CGA_MODE4_PAL1_LOW = numpy.multiply(_CGA_MODE4_PAL1_NORMALIZED_LOW, 255.0).astype(numpy.float32)

_CGA_MODE4_PAL1_NORMALIZED_HIGH = numpy.array([
    [ 0., 0., 0. ],
    [ 85.0/255.0, 255.0/255.0, 255.0/255.0 ],
    [ 255.0/255.0, 85.0/255.0, 255.0/255.0 ],
    [ 255.0/255.0, 255.0/255.0, 255.0/255.0 ]
], dtype=numpy.float32)
CGA_MODE4_PAL1_HIGH = numpy.multiply(_CGA_MODE4_PAL1_NORMALIZED_HIGH, 255.0).astype(numpy.float32)


_CGA_MODE4_PAL2_NORMALIZED_LOW = numpy.array([
    [ 0., 0., 0. ],
    [ 0.0, 170.0/255.0, 0.0 ],
    [ 170.0/255.0, 0.0, 0.0 ],
    [ 170.0/255.0, 85.0/255.0, 0.0 ]
], dtype=numpy.float32)
CGA_MODE4_PAL2_LOW = numpy.multiply(_CGA_MODE4_PAL2_NORMALIZED_LOW, 255.0).astype(numpy.float32)

_CGA_MODE4_PAL2_NORMALIZED_HIGH = numpy.array([
    [ 0., 0., 0. ],
    [ 85.0/255.0, 255.0/255.0, 85.0/255.0],
    [ 255.0/255.0, 85.0/255.0, 85.0/255.0 ],
    [ 255.0/255.0, 255.0/255.0, 85.0/255.0 ]
], dtype=numpy.float32)
CGA_MODE4_PAL2_HIGH = numpy.multiply(_CGA_MODE4_PAL2_NORMALIZED_HIGH, 255.0).astype(numpy.float32)


_CGA_MODE5_NORMALIZED_LOW = numpy.array([
    [ 0., 0., 0. ],
    [ 0.0/255.0, 170.0/255.0, 170./255.0 ],
    [ 170.0/255.0, 0.0, 0.0 ],
    [ 170.0/255.0, 170.0/255.0, 170.0/255.0 ]
], dtype=numpy.float32)
CGA_MODE5_LOW = numpy.multiply(_CGA_MODE5_NORMALIZED_LOW, 255.0).astype(numpy.float32)

_CGA_MODE5_NORMALIZED_HIGH = numpy.array([
    [ 0., 0., 0. ],
    [ 85.0/255.0, 255.0/255.0, 255.0/255.0],
    [ 255.0/255.0, 85.0/255.0, 85.0/255.0 ],
    [ 255.0/255.0, 255.0/255.0, 255.0/255.0 ]
], dtype=numpy.float32)
CGA_MODE5_HIGH = numpy.multiply(_CGA_MODE5_NORMALIZED_HIGH, 255.0).astype(numpy.float32)


_ZX_SPECTRUM_NORMALIZED_LOW = numpy.array([
     [ 0., 0., 0. ] ,
     [ 0., 0.0, 192.0/255.0 ] ,
     [ 192.0/255.0, 0.0, 0.0 ] ,
     [ 192.0/255.0, 0.0, 192.0/255.0 ] ,
     [ 0., 192.0/255.0, 0.0 ],
     [ 0., 192.0/255.0, 192.0/255.0 ],
     [ 192.0/255.0, 192.0/255.0, 0.0 ],
     [ 192.0/255.0, 192.0/255.0, 192.0/255.0 ]
], dtype=numpy.float32)
ZX_SPECTRUM_LOW = numpy.multiply(_ZX_SPECTRUM_NORMALIZED_LOW, 255.0).astype(numpy.float32)


_ZX_SPECTRUM_NORMALIZED_HIGH = numpy.array([
     [ 0., 0., 0. ] ,
     [ 0., 0.0, 255.0/255.0 ] ,
     [ 255.0/255.0, 0.0, 0.0 ] ,
     [ 255.0/255.0, 0.0, 255.0/255.0 ] ,
     [ 0., 255.0/255.0, 0.0 ],
     [ 0., 255.0/255.0, 255.0/255.0 ],
     [ 255.0/255.0, 255.0/255.0, 0.0 ],
     [ 255.0/255.0, 255.0/255.0, 255.0/255.0 ]
], dtype=numpy.float32)
ZX_SPECTRUM_HIGH = numpy.multiply(_ZX_SPECTRUM_NORMALIZED_HIGH, 255.0).astype(numpy.float32)


_APPLE_II_LOW_NORMALIZED = numpy.array([
    [ 0., 0., 0. ],
    [ 108.0/255.0, 41.0/255.0, 64.0/255.0 ],
    [ 64.0/255.0, 53.0/255.0, 120.0/255.0 ],
    [ 217.0/255.0, 60.0/255.0, 240.0/255.0 ],
    [ 19.0/255.0, 87.0/255.0, 64.0/255.0 ],
    [ 128.0/255.0, 128.0/255.0, 128.0/255.0],
    [ 38.0/255.0, 151.0/255.0, 240.0/255.0 ],
    [ 191.0/255.0, 180.0/255.0, 248.0/255.0]
], dtype=numpy.float32)
APPLE_II_LOW = numpy.multiply(_APPLE_II_LOW_NORMALIZED, 255.0).astype(numpy.float32)

_APPLE_II_HIGH_NORMALIZED = numpy.array([
    [ 64.0/255.0, 75.0/255.0, 7.0/255.0 ],
    [ 217.0/255.0, 104.0/255.0, 15.0/255.0 ],
    [ 128.0/255.0, 128.0/255.0, 128.0/255.0 ],
    [ 236.0/255.0, 168.0/255.0, 191.0/255.0 ],
    [ 38.0/255.0, 195.0/255.0, 15.0/255.0],
    [ 191.0/255.0, 202.0/255.0, 135.0/255.0],
    [ 147.0/255.0, 214.0/255.0, 191.0/255.0 ],
    [ 255.0/255.0, 255.0/255.0, 255.0/255.0]
], dtype=numpy.float32)
APPLE_II_HIGH = numpy.multiply(_APPLE_II_HIGH_NORMALIZED, 255.0).astype(numpy.float32)

_APPLE_II_NORMALIZED = numpy.array([
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
], dtype=numpy.float32)
APPLE_II = numpy.multiply(_APPLE_II_NORMALIZED, 255.0).astype(numpy.float32)


_COMMODORE_64_NORMALIZED = numpy.array([
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
], dtype=numpy.float32)
COMMODORE_64 = numpy.multiply(_COMMODORE_64_NORMALIZED, 255.0).astype(numpy.float32)


_AMSTRAD_CPC_NORMALIZED = numpy.array(
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
    [255/255.0, 255/255.0, 255/255.0]], dtype=numpy.float32)

AMSTRAD_CPC = numpy.multiply(_AMSTRAD_CPC_NORMALIZED, 255.0).astype(numpy.float32)


_MSX_NORMALIZED = numpy.array([
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
], dtype=numpy.float32)
MSX = numpy.multiply(_MSX_NORMALIZED, 255.0).astype(numpy.float32)



_TO7_NORMALIZED = numpy.array([
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
], dtype=numpy.float32)
TO7 = numpy.multiply(_TO7_NORMALIZED, 255.0).astype(numpy.float32)

_PICO_8_NORMALIZED = numpy.array([
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
], dtype=numpy.float32)
PICO_8 = numpy.multiply(_PICO_8_NORMALIZED, 255.0).astype(numpy.float32)


_PICO_8_CUSTOM_NORMALIZED = numpy.array([
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
], dtype=numpy.float32)
PICO_8_CUSTOM = numpy.multiply(_PICO_8_CUSTOM_NORMALIZED, 255.0).astype(numpy.float32)



_MICROSOFT_WINDOWS_16_NORMALIZED = numpy.array([
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
], dtype=numpy.float32)
MICROSOFT_WINDOWS_16 = numpy.multiply(_MICROSOFT_WINDOWS_16_NORMALIZED, 255.0).astype(numpy.float32)

_MICROSOFT_WINDOWS_20_NORMALIZED = numpy.array([
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
], dtype=numpy.float32)
MICROSOFT_WINDOWS_20 = numpy.multiply(_MICROSOFT_WINDOWS_20_NORMALIZED, 255.0).astype(numpy.float32)

_MICROSOFT_WINDOWS_PAINT_NORMALIZED = numpy.array([
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
], dtype=numpy.float32)
MICROSOFT_WINDOWS_PAINT = numpy.multiply(
    _MICROSOFT_WINDOWS_PAINT_NORMALIZED, 255.0).astype(numpy.float32)

# https://superuser.com/questions/361297/what-colour-is-t
# he-dark-green-on-old-fashioned-green-screen-computer-displays/1206781#1206781
_MONO_PHOSPHOR_AMBER_NORMALIZED = numpy.array([
    [ 0.15686275, 0.15686275, 0.15686275 ],
    [ 1., 0.69019608, 0. ]
], dtype=numpy.float32)
MONO_PHOSPHOR_AMBER = numpy.multiply(
    _MONO_PHOSPHOR_AMBER_NORMALIZED, 255.0).astype(numpy.float32)

_MONO_PHOSPHOR_LTAMBER_NORMALIZED = numpy.array([
    [ 0.15686275, 0.15686275, 0.15686275 ],
    [ 1., 0.8, 0. ]
], dtype=numpy.float32)
MONO_PHOSPHOR_LTAMBER = numpy.multiply(
    _MONO_PHOSPHOR_LTAMBER_NORMALIZED, 255.0).astype(numpy.float32)

_MONO_PHOSPHOR_GREEN1_NORMALIZED = numpy.array([
    [ 0.15686275, 0.15686275, 0.15686275 ],
    [ 0.2, 1., 0. ]
], dtype=numpy.float32)
MONO_PHOSPHOR_GREEN1 = numpy.multiply(
    _MONO_PHOSPHOR_GREEN1_NORMALIZED, 255.0).astype(numpy.float32)

_MONO_PHOSPHOR_GREEN2_NORMALIZED = numpy.array([
    [ 0.15686275, 0.15686275, 0.15686275 ],
    [ 0, 1., 0.2 ]
], dtype=numpy.float32)
MONO_PHOSPHOR_GREEN2 = numpy.multiply(
    _MONO_PHOSPHOR_GREEN2_NORMALIZED, 255.0).astype(numpy.float32)

_MONO_PHOSPHOR_GREEN3_NORMALIZED = numpy.array([
    [ 0.15686275, 0.15686275, 0.15686275 ],
    [ 0, 1., 0.4 ]
], dtype=numpy.float32)
MONO_PHOSPHOR_GREEN3 = numpy.multiply(
    _MONO_PHOSPHOR_GREEN3_NORMALIZED, 255.0).astype(numpy.float32)

# Created by Adigun A. Polack
# https://lospec.com/palette-list/aap-64
_AAP64_NORMALIZED = numpy.array([
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
], dtype=numpy.float32)
AAP64 = numpy.multiply(_AAP64_NORMALIZED, 255.0).astype(numpy.float32)

# https://lospec.com/palette-list/apollo
# Created by AdamCYounis
_APOLLO_NORMALIZED = numpy.array([
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
], dtype=numpy.float32)
APOLLO = numpy.multiply(_APOLLO_NORMALIZED, 255.0).astype(numpy.float32)

#https://lospec.com/palette-list/funkyfuture-8
# Created by Shamaboy
_FUNKYFUTURE_NORMALIZED = numpy.array([
[0.16862745098039217,0.058823529411764705,0.32941176470588235],
[0.6705882352941176,0.12156862745098039,0.396078431372549],
[1.0,0.30980392156862746,0.4117647058823529],
[1.0,0.9686274509803922,0.9725490196078431],
[1.0,0.5058823529411764,0.25882352941176473],
[1.0,0.8549019607843137,0.27058823529411763],
[0.2,0.40784313725490196,0.8627450980392157],
[0.28627450980392155,0.9058823529411765,0.9254901960784314]
], dtype=numpy.float32)
FUNKYFUTURE = numpy.multiply(_FUNKYFUTURE_NORMALIZED, 255.0).astype(numpy.float32)

# https://lospec.com/palette-list/vinik24
# Created by Vinik
_VINIK24_NORMALIZED = numpy.array([
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
], dtype=numpy.float32)
VINIK24 = numpy.multiply(_VINIK24_NORMALIZED, 255.0).astype(numpy.float32)

# https://lospec.com/palette-list/twilioquest-76
# Created by Kerrie Lake
_TWILIOQUEST76_NORMALIZED = numpy.array([
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
], dtype=numpy.float32)
TWILIOQUEST76 = numpy.multiply(_TWILIOQUEST76_NORMALIZED, 255.0).astype(numpy.float32)



# https://lospec.com/palette-list/iridescent-crystal
# Created by LostInIndigo
_IRIDESCENTCRYSTAL_NORMALIZED = numpy.array([
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
], dtype=numpy.float32)
IRIDESCENTCRYSTAL = numpy.multiply(_IRIDESCENTCRYSTAL_NORMALIZED, 255.0).astype(numpy.float32)


# https://lospec.com/palette-list/aap-splendor128
# Created by Adigun A. Polack
_AAPSPLENDOR128_NORMALIZED = numpy.array([
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
], dtype=numpy.float32)
AAPSPLENDOR128 = numpy.multiply(_AAPSPLENDOR128_NORMALIZED, 255.0).astype(numpy.float32)




# https://lospec.com/palette-list/lospec500
# collaboration
_LOSPEC500_NORMALIZED = numpy.array([
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
], dtype=numpy.float32)
LOSPEC500 = numpy.multiply(_LOSPEC500_NORMALIZED, 255.0).astype(numpy.float32)





# https://lospec.com/palette-list/famicube
# Created by Arne as part of his Famicube Project.
_FAMICUBE_NORMALIZED = numpy.array([
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
], dtype=numpy.float32)
FAMICUBE = numpy.multiply(_FAMICUBE_NORMALIZED, 255.0).astype(numpy.float32)


# https://lospec.com/palette-list/smooth-polished-silver
_SILVER_NORMALIZED = numpy.array([
[0.0,0.0,0.0],
[0.1568627450980392,0.12941176470588237,0.12156862745098039],
[0.2901960784313726,0.2235294117647059,0.23921568627450981],
[0.403921568627451,0.34509803921568627,0.2980392156862745],
[0.6901960784313725,0.5725490196078431,0.6549019607843137],
[0.9490196078431372,0.8392156862745098,0.8862745098039215],
[1.0,1.0,1.0]
], dtype=numpy.float32)
SILVER = numpy.multiply(_SILVER_NORMALIZED, 255.0).astype(numpy.float32)