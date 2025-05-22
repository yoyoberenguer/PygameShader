"""
PygameShader TRANSITION DEMO
"""

import sys

try:
    from PygameShader.shader import blend, blend_inplace
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

try:
    BACKGROUND = pygame.image.load("../Assets/space5.jpg").convert()
except FileNotFoundError:
    raise FileNotFoundError(
        '\nImage file space5.jpg is missing from the Assets directory.')

BACKGROUND = pygame.transform.smoothscale(BACKGROUND, (WIDTH, HEIGHT))

try:
    DESTINATION = pygame.image.load("../Assets/Aliens.jpg").convert()
except FileNotFoundError:
    raise FileNotFoundError(
        '\nImage file Aliens.jpg is missing from the Assets directory.')

DESTINATION = pygame.transform.smoothscale(DESTINATION, (WIDTH, HEIGHT))
# DEST_ARRAY  = pygame.surfarray.pixels3d(DESTINATION)

assert BACKGROUND.get_size() == DESTINATION.get_size()

pygame.display.set_caption("demo transition/blend effect")

FRAME = 0
CLOCK = pygame.time.Clock()
GAME = True
VALUE = 0
V = +0.2


pygame.font.init()
font = pygame.font.SysFont("Arial", 15)


def show_fps(screen_, fps_, avg_) -> list:

    """ Show framerate in upper left corner """

    fps = str(f"fps:{fps_:.3f}")
    av = sum(avg_)/len(avg_) if len(avg_) > 0 else 0
    fps_text = font.render(fps, True, pygame.Color("coral"))
    screen_.blit(fps_text, (10, 0))
    if av != 0:
        av = str(f"avg:{av:.3f}")
        avg_text = font.render(av, True, pygame.Color("coral"))
        screen_.blit(avg_text, (120, 0))
    if len(avg_) > 200:
        avg_ = avg_[200:]
    return avg_

avg = []
clock = pygame.time.Clock()

while GAME:

    pygame.event.pump()
    for event in pygame.event.get():

        keys = pygame.key.get_pressed()

        if keys[pygame.K_ESCAPE]:
            GAME = False
            break

    transition = blend(
        source =BACKGROUND, destination =DESTINATION, percentage =VALUE)

    SCREEN.blit(transition, (0, 0))

    clock.tick(500)
    t = clock.get_fps()
    FRAME += 1

    pygame.display.set_caption(
        "Demo blend effect/transition %s percent; %s fps"
        "(%sx%s)" % (round(VALUE, 2), round(CLOCK.get_fps(), 2), WIDTH, HEIGHT))

    avg.append(t)
    avg = show_fps(SCREEN, t, avg)

    pygame.display.flip()

    VALUE += V

    if VALUE >= 100:
        VALUE = 100
        V = -0.2
    if VALUE <= 0:
        VALUE = 0
        V = 0.2


pygame.quit()