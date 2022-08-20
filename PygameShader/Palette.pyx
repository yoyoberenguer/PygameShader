# cython: binding=False, boundscheck=False, wraparound=False, nonecheck=False, cdivision=True,
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
    "MONO_PHOSPHOR_GREEN3"   : "MONOCHROME 2 green colors"
}

import numpy

TELETEXT_NORMALIZED = numpy.array([
    [ 0., 0., 0. ],
    [ 0., 0., 255.0/255.0 ],
    [ 255.0/255.0, 0.0, 0.0 ],
    [ 255.0/255.0, 0.0, 255.0/255.0 ],
    [ 0., 255.0/255.0, 0.0 ],
    [ 0.0, 255.0/255.0, 255.0/255.0 ],
    [ 255.0/255.0, 255.0/255.0, 0.0 ],
    [ 255.0/255.0, 255.0/255.0, 255.0/255.0]
], dtype=numpy.float32)
TELETEXT = numpy.multiply(TELETEXT_NORMALIZED, 255.0).astype(numpy.float32)

BBC_MICRO = TELETEXT

CGA_MODE4_PAL1_NORMALIZED_LOW = numpy.array([
    [ 0., 0., 0. ],
    [ 0., 0.66666, 0.6666 ],
    [ 0.66666, 0., 0.6666 ],
    [ 0.66666, 0.6666, 0.66666 ]
], dtype=numpy.float32)
CGA_MODE4_PAL1_LOW = numpy.multiply(CGA_MODE4_PAL1_NORMALIZED_LOW, 255.0).astype(numpy.float32)

CGA_MODE4_PAL1_NORMALIZED_HIGH = numpy.array([
    [ 0., 0., 0. ],
    [ 85.0/255.0, 255.0/255.0, 255.0/255.0 ],
    [ 255.0/255.0, 85.0/255.0, 255.0/255.0 ],
    [ 255.0/255.0, 255.0/255.0, 255.0/255.0 ]
], dtype=numpy.float32)
CGA_MODE4_PAL1_HIGH = numpy.multiply(CGA_MODE4_PAL1_NORMALIZED_HIGH, 255.0).astype(numpy.float32)


CGA_MODE4_PAL2_NORMALIZED_LOW = numpy.array([
    [ 0., 0., 0. ],
    [ 0.0, 170.0/255.0, 0.0 ],
    [ 170.0/255.0, 0.0, 0.0 ],
    [ 170.0/255.0, 85.0/255.0, 0.0 ]
], dtype=numpy.float32)
CGA_MODE4_PAL2_LOW = numpy.multiply(CGA_MODE4_PAL2_NORMALIZED_LOW, 255.0).astype(numpy.float32)

CGA_MODE4_PAL2_NORMALIZED_HIGH = numpy.array([
    [ 0., 0., 0. ],
    [ 85.0/255.0, 255.0/255.0, 85.0/255.0],
    [ 255.0/255.0, 85.0/255.0, 85.0/255.0 ],
    [ 255.0/255.0, 255.0/255.0, 85.0/255.0 ]
], dtype=numpy.float32)
CGA_MODE4_PAL2_HIGH = numpy.multiply(CGA_MODE4_PAL2_NORMALIZED_HIGH, 255.0).astype(numpy.float32)


CGA_MODE5_NORMALIZED_LOW = numpy.array([
    [ 0., 0., 0. ],
    [ 0.0/255.0, 170.0/255.0, 170./255.0 ],
    [ 170.0/255.0, 0.0, 0.0 ],
    [ 170.0/255.0, 170.0/255.0, 170.0/255.0 ]
], dtype=numpy.float32)
CGA_MODE5_LOW = numpy.multiply(CGA_MODE5_NORMALIZED_LOW, 255.0).astype(numpy.float32)

CGA_MODE5_NORMALIZED_HIGH = numpy.array([
    [ 0., 0., 0. ],
    [ 85.0/255.0, 255.0/255.0, 255.0/255.0],
    [ 255.0/255.0, 85.0/255.0, 85.0/255.0 ],
    [ 255.0/255.0, 255.0/255.0, 255.0/255.0 ]
], dtype=numpy.float32)
CGA_MODE5_HIGH = numpy.multiply(CGA_MODE5_NORMALIZED_HIGH, 255.0).astype(numpy.float32)


ZX_SPECTRUM_NORMALIZED_LOW = numpy.array([
     [ 0., 0., 0. ] ,
     [ 0., 0.0, 192.0/255.0 ] ,
     [ 192.0/255.0, 0.0, 0.0 ] ,
     [ 192.0/255.0, 0.0, 192.0/255.0 ] ,
     [ 0., 192.0/255.0, 0.0 ],
     [ 0., 192.0/255.0, 192.0/255.0 ],
     [ 192.0/255.0, 192.0/255.0, 0.0 ],
     [ 192.0/255.0, 192.0/255.0, 192.0/255.0 ]
], dtype=numpy.float32)
ZX_SPECTRUM_LOW = numpy.multiply(ZX_SPECTRUM_NORMALIZED_LOW, 255.0).astype(numpy.float32)


ZX_SPECTRUM_NORMALIZED_HIGH = numpy.array([
     [ 0., 0., 0. ] ,
     [ 0., 0.0, 255.0/255.0 ] ,
     [ 255.0/255.0, 0.0, 0.0 ] ,
     [ 255.0/255.0, 0.0, 255.0/255.0 ] ,
     [ 0., 255.0/255.0, 0.0 ],
     [ 0., 255.0/255.0, 255.0/255.0 ],
     [ 255.0/255.0, 255.0/255.0, 0.0 ],
     [ 255.0/255.0, 255.0/255.0, 255.0/255.0 ]
], dtype=numpy.float32)
ZX_SPECTRUM_HIGH = numpy.multiply(ZX_SPECTRUM_NORMALIZED_HIGH, 255.0).astype(numpy.float32)


APPLE_II_LOW_NORMALIZED = numpy.array([
    [ 0., 0., 0. ],
    [ 108.0/255.0, 41.0/255.0, 64.0/255.0 ],
    [ 64.0/255.0, 53.0/255.0, 120.0/255.0 ],
    [ 217.0/255.0, 60.0/255.0, 240.0/255.0 ],
    [ 19.0/255.0, 87.0/255.0, 64.0/255.0 ],
    [ 128.0/255.0, 128.0/255.0, 128.0/255.0],
    [ 38.0/255.0, 151.0/255.0, 240.0/255.0 ],
    [ 191.0/255.0, 180.0/255.0, 248.0/255.0]
], dtype=numpy.float32)
APPLE_II_LOW = numpy.multiply(APPLE_II_LOW_NORMALIZED, 255.0).astype(numpy.float32)

APPLE_II_HIGH_NORMALIZED = numpy.array([
    [ 64.0/255.0, 75.0/255.0, 7.0/255.0 ],
    [ 217.0/255.0, 104.0/255.0, 15.0/255.0 ],
    [ 128.0/255.0, 128.0/255.0, 128.0/255.0 ],
    [ 236.0/255.0, 168.0/255.0, 191.0/255.0 ],
    [ 38.0/255.0, 195.0/255.0, 15.0/255.0],
    [ 191.0/255.0, 202.0/255.0, 135.0/255.0],
    [ 147.0/255.0, 214.0/255.0, 191.0/255.0 ],
    [ 255.0/255.0, 255.0/255.0, 255.0/255.0]
], dtype=numpy.float32)
APPLE_II_HIGH = numpy.multiply(APPLE_II_HIGH_NORMALIZED, 255.0).astype(numpy.float32)

APPLE_II_NORMALIZED = numpy.array([
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
APPLE_II = numpy.multiply(APPLE_II_NORMALIZED, 255.0).astype(numpy.float32)


COMMODORE_64_NORMALIZED = numpy.array([
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
COMMODORE_64 = numpy.multiply(COMMODORE_64_NORMALIZED, 255.0).astype(numpy.float32)


AMSTRAD_CPC_NORMALIZED = numpy.array(
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

AMSTRAD_CPC = numpy.multiply(AMSTRAD_CPC_NORMALIZED, 255.0).astype(numpy.float32)


MSX_NORMALIZED = numpy.array([
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
MSX = numpy.multiply(MSX_NORMALIZED, 255.0).astype(numpy.float32)



TO7_NORMALIZED = numpy.array([
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
TO7 = numpy.multiply(TO7_NORMALIZED, 255.0).astype(numpy.float32)

PICO_8_NORMALIZED = numpy.array([
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
PICO_8 = numpy.multiply(PICO_8_NORMALIZED, 255.0).astype(numpy.float32)


PICO_8_CUSTOM_NORMALIZED = numpy.array([
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
PICO_8_CUSTOM = numpy.multiply(PICO_8_CUSTOM_NORMALIZED, 255.0).astype(numpy.float32)



MICROSOFT_WINDOWS_16_NORMALIZED = numpy.array([
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
MICROSOFT_WINDOWS_16 = numpy.multiply(MICROSOFT_WINDOWS_16_NORMALIZED, 255.0).astype(numpy.float32)

MICROSOFT_WINDOWS_20_NORMALIZED = numpy.array([
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
MICROSOFT_WINDOWS_20 = numpy.multiply(MICROSOFT_WINDOWS_20_NORMALIZED, 255.0).astype(numpy.float32)

MICROSOFT_WINDOWS_PAINT_NORMALIZED = numpy.array([
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
    MICROSOFT_WINDOWS_PAINT_NORMALIZED, 255.0).astype(numpy.float32)

# https://superuser.com/questions/361297/what-colour-is-t
# he-dark-green-on-old-fashioned-green-screen-computer-displays/1206781#1206781
MONO_PHOSPHOR_AMBER_NORMALIZED = numpy.array([
    [ 0.15686275, 0.15686275, 0.15686275 ],
    [ 1., 0.69019608, 0. ]
], dtype=numpy.float32)
MONO_PHOSPHOR_AMBER = numpy.multiply(
    MONO_PHOSPHOR_AMBER_NORMALIZED, 255.0).astype(numpy.float32)

MONO_PHOSPHOR_LTAMBER_NORMALIZED = numpy.array([
    [ 0.15686275, 0.15686275, 0.15686275 ],
    [ 1., 0.8, 0. ]
], dtype=numpy.float32)
MONO_PHOSPHOR_LTAMBER = numpy.multiply(
    MONO_PHOSPHOR_LTAMBER_NORMALIZED, 255.0).astype(numpy.float32)

MONO_PHOSPHOR_GREEN1_NORMALIZED = numpy.array([
    [ 0.15686275, 0.15686275, 0.15686275 ],
    [ 0.2, 1., 0. ]
], dtype=numpy.float32)
MONO_PHOSPHOR_GREEN1 = numpy.multiply(
    MONO_PHOSPHOR_GREEN1_NORMALIZED, 255.0).astype(numpy.float32)

MONO_PHOSPHOR_GREEN2_NORMALIZED = numpy.array([
    [ 0.15686275, 0.15686275, 0.15686275 ],
    [ 0, 1., 0.2 ]
], dtype=numpy.float32)
MONO_PHOSPHOR_GREEN2 = numpy.multiply(
    MONO_PHOSPHOR_GREEN2_NORMALIZED, 255.0).astype(numpy.float32)

MONO_PHOSPHOR_GREEN3_NORMALIZED = numpy.array([
    [ 0.15686275, 0.15686275, 0.15686275 ],
    [ 0, 1., 0.4 ]
], dtype=numpy.float32)
MONO_PHOSPHOR_GREEN3 = numpy.multiply(
    MONO_PHOSPHOR_GREEN3_NORMALIZED, 255.0).astype(numpy.float32)

