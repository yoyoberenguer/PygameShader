"""
PygameShader TRANSITION DEMO
"""

import sys

try:
    from PygameShader.shader import blend
except ImportError:
    raise ImportError("\n<PygameShader> library is missing on your system."
          "\nTry: \n   C:\\pip install PygameShader on a window command prompt.")

try:
    import numpy
    from numpy import uint8
except ImportError:
    raise ImportError("\n<numpy> library is missing on your system."
          "\nTry: \n   C:\\pip install numpy on a window command prompt.")

numpy.set_printoptions(threshold=sys.maxsize)

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
SCREEN = pygame.display.set_mode((WIDTH, HEIGHT), pygame.FULLSCREEN | pygame.SCALED)
SCREEN.convert(32, RLEACCEL)
SCREEN.set_alpha(None)

# Load the background image
BACKGROUND = pygame.image.load("../Assets/background.jpg").convert()
BACKGROUND = pygame.transform.smoothscale(BACKGROUND, (WIDTH, HEIGHT))

DESTINATION = pygame.image.load("../Assets/Aliens.jpg").convert()
DESTINATION = pygame.transform.smoothscale(DESTINATION, (WIDTH, HEIGHT))
DEST_ARRAY  = pygame.surfarray.pixels3d(DESTINATION)

assert BACKGROUND.get_size() == DESTINATION.get_size()

pygame.display.set_caption("demo transition/blend effect")

FRAME = 0
CLOCK = pygame.time.Clock()
GAME = True
VALUE = 0
V = +0.2

while GAME:

    pygame.event.pump()
    for event in pygame.event.get():

        keys = pygame.key.get_pressed()

        if keys[pygame.K_ESCAPE]:
            GAME = False
            break

    transition = blend(
        source_=BACKGROUND, destination_=DEST_ARRAY, percentage_=VALUE)

    SCREEN.blit(transition, (0, 0))

    CLOCK.tick()
    FRAME += 1

    pygame.display.set_caption(
        "Demo blend effect/transition %s percent; %s fps"
        "(%sx%s)" % (round(VALUE, 2), round(CLOCK.get_fps(), 2), WIDTH, HEIGHT))

    pygame.display.flip()

    VALUE += V

    if VALUE >= 100:
        VALUE = 100
        V = -0.2
    if VALUE <= 0:
        VALUE = 0
        V = 0.2


pygame.quit()