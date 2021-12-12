"""
PygameShader FIRE DEMO
"""

from random import uniform

try:
    from PygameShader.shader import custom_map, rgb_to_int, shader_fire_effect
except ImportError:
    raise ImportError("\n<PygameShader> library is missing on your system."
          "\nTry: \n   C:\\pip install PygameShader on a window command prompt.")

try:
    import numpy
    from numpy import uint8
except ImportError:
    raise ImportError("\n<numpy> library is missing on your system."
          "\nTry: \n   C:\\pip install numpy on a window command prompt.")

# PYGAME IS REQUIRED
try:
    import pygame
    from pygame import Surface, RLEACCEL, QUIT, K_SPACE, BLEND_RGB_ADD

except ImportError:
    raise ImportError("\n<Pygame> library is missing on your system."
          "\nTry: \n   C:\\pip install pygame on a window command prompt.")

# Set the display to 1024 x 768
WIDTH = 1024
HEIGHT = 768
SCREEN = pygame.display.set_mode((WIDTH, HEIGHT), vsync=True)
SCREEN.convert(32, RLEACCEL)
SCREEN.set_alpha(None)

# Load the background image
BACKGROUND = pygame.image.load("../Assets/Background.jpg").convert()
BACKGROUND = pygame.transform.smoothscale(BACKGROUND, (WIDTH, HEIGHT))

image = BACKGROUND.copy()
pygame.display.set_caption("demo fire effect")

FRAME = 0
CLOCK = pygame.time.Clock()
GAME = True


def palette_array() -> tuple:
    """
    Create a C - buffer type data 1D array containing the
    fire color palette (mapped RGB color, integer)

    :return: 1D contiguous array (C buffer type data)
    """
    # Set an array with pre-defined color wavelength
    arr = numpy.array([0, 1,        # violet
                       0, 1,        # blue
                       0, 1,        # green
                       2, 619,      # yellow
                       620, 650,    # orange
                       651, 660],   # red
                      numpy.int)

    heatmap = [custom_map(i - 20, arr, 1.0) for i in range(380, 800)]
    heatmap_array = numpy.zeros((800 - 380, 3), uint8)
    heatmap_rescale = numpy.zeros(255, numpy.uint)

    i = 0
    for t in heatmap:
        heatmap_array[i, 0] = t[0]
        heatmap_array[i, 1] = t[1]
        heatmap_array[i, 2] = t[2]
        i += 1

    for r in range(255):
        s = int(r * (800.0 - 380.0) / 255.0)
        heatmap_rescale[r] = \
            rgb_to_int(heatmap_array[s][0], heatmap_array[s][1], heatmap_array[s][2])

    heatmap_rescale = numpy.ascontiguousarray(heatmap_rescale[::-1])

    return heatmap_rescale


fire_palette = palette_array()
fire_array = numpy.zeros((HEIGHT, WIDTH), dtype=numpy.float32)

while GAME:

    pygame.event.pump()
    for event in pygame.event.get():

        keys = pygame.key.get_pressed()

        if keys[pygame.K_ESCAPE]:
            GAME = False
            break

    # SCREEN.fill((0, 0, 0, 0))
    SCREEN.blit(BACKGROUND, (0, 0))

    # Execute the shader fire effect
    surface_ = shader_fire_effect(
        WIDTH,
        HEIGHT,
        3.97 + uniform(0.002, 0.008),
        fire_palette,
        fire_array,
        reduce_factor_          =3,
        bloom_                  =True,
        fast_bloom_             =False,
        bpf_threshold_          =48,
        brightness_             =True,
        brightness_intensity_   =0.095,
        transpose_              =False,
        border_                 =False,
        low_                    =0,
        high_                   =WIDTH,
        blur_                   =True).convert(32, RLEACCEL)

    SCREEN.blit(surface_, (0, 0), special_flags=BLEND_RGB_ADD)

    pygame.display.flip()
    CLOCK.tick()
    FRAME += 1

    pygame.display.set_caption(
        "Test shader_fire_effect %s fps "
        "(%sx%s)" % (round(CLOCK.get_fps(), 2), WIDTH, HEIGHT))

    image = BACKGROUND.copy()
