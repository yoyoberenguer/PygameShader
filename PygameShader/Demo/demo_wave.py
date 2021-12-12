"""
PygameShader WAVE DEMO
"""
import math

try:
    from PygameShader.shader import shader_wave24bit_inplace
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
pygame.display.set_caption("demo wave effect")

FRAME = 0
CLOCK = pygame.time.Clock()
GAME = True

ANGLE = 0

while GAME:

    pygame.event.pump()
    for event in pygame.event.get():

        keys = pygame.key.get_pressed()

        if keys[pygame.K_ESCAPE]:
            GAME = False
            break

    shader_wave24bit_inplace(image, ANGLE * math.pi / 180.0, 10)
    SCREEN.blit(image, (0, 0))

    pygame.display.flip()
    CLOCK.tick()
    FRAME += 1
    ANGLE += 5
    ANGLE %= 360

    pygame.display.set_caption(
        "Test shader wave effect %s fps "
        "(%sx%s)" % (round(CLOCK.get_fps(), 2), WIDTH, HEIGHT))

    image = BACKGROUND.copy()
