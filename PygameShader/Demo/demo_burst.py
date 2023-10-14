"""
PygameShader BURST IMAGE DEMO
"""

import sys
import timeit
try:
    from PygameShader import *
except ImportError:
    raise ImportError("\n<PygameShader> library is missing on your system."
                      "\nTry: \n   C:\\pip install PygameShader on a window command prompt.")

try:
    from PygameShader.misc import *
except ImportError:
    raise ImportError("\n<misc> library is missing on your system.")

from PygameShader.BurstSurface import *

try:
    import numpy
    from numpy import uint8
except ImportError:
    raise ImportError("\n<numpy> library is missing on your system."
                      "\nTry: \n   C:\\pip install numpy on a window command prompt.")

# PYGAME IS REQUIRED
try:
    import pygame
    from pygame import Surface, RLEACCEL, QUIT, K_SPACE, BLEND_RGB_ADD, \
        DOUBLEBUF, FULLSCREEN, BLEND_RGB_SUB, \
        BLEND_RGB_MULT

except ImportError:
    raise ImportError("\n<Pygame> library is missing on your system."
                      "\nTry: \n   C:\\pip install pygame on a window command prompt.")

pygame.init()

pygame.font.init()
font = pygame.font.SysFont("Arial", 15)

# Set the display to 1024 x 768
WIDTH = 800
HEIGHT = 800
pygame.init()
SCREENRECT = pygame.Rect(0, 0, WIDTH, HEIGHT)
from PygameShader.shader import *

SCREEN = pygame.display.set_mode((800, 800), pygame.SCALED, 32)
try:
    IMAGE = pygame.image.load("../Assets/city.jpg")
except FileNotFoundError:
    raise FileNotFoundError(
        '\nImage file city.jpg is missing from the Assets directory.')

IMAGE = smoothscale(IMAGE, (WIDTH, HEIGHT))

clock = pygame.time.Clock()

avg = [ ]
angle = 0
FRAME = 0
GAME = True
event_get = pygame.event.get
get_key = pygame.key.get_pressed
get_pos = pygame.mouse.get_pos

pygame.time.wait(500)

vertex_array = [ ]
burst(
    image_=IMAGE,
    vertex_array_=vertex_array,
    block_size_=8,
    rows_=100,
    columns_=100,
    x_=0,
    y_=0,
    max_angle_=-5
)

tmp_surface = pygame.Surface((WIDTH, HEIGHT))


def show_fps(render_, fps_, avg_) -> list:
    """ Show framerate in upper left corner """

    fps = str(f"CPU fps:{fps_:.3f}")
    av = sum(avg_) / len(avg_) if len(avg_) > 0 else 0

    fps_text = font.render(fps, True, pygame.Color("beige"))
    render_.blit(fps_text, (0, 0), special_flags=0)
    if av != 0:
        av = str(f"avg:{av:.3f}")
        fps_text = font.render(av, True, pygame.Color("beige"))

        render_.blit(fps_text, (130, 0), special_flags=0)
    if len(avg_) > 10:
        avg_ = avg_[ 10: ]

    return avg_


while GAME:

    pygame.event.pump()

    keys = get_key()
    for event in event_get():

        if keys[ pygame.K_ESCAPE ]:
            GAME = False
            break

        if event.type == pygame.MOUSEMOTION:
            mouse_pos = pygame.math.Vector2(get_pos())

    SCREEN.fill((0, 0, 0))

    rebuild_from_frame(
        screen_=SCREEN,
        current_frame_=FRAME,
        start_frame=200,
        vertex_array_=vertex_array,
        blend_=0)

    clock.tick(200)
    t = clock.get_fps()
    avg.append(t)
    avg = show_fps(SCREEN, t, avg)

    pygame.display.flip()
    FRAME += 1


pygame.quit()
