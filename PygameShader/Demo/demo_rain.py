"""
PygameShader DEMO rain effect
"""

try:
    from PygameShader.shader import *
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
    from pygame import Surface, RLEACCEL, QUIT, K_SPACE, BLEND_RGB_ADD, BLEND_RGB_MIN

except ImportError:
    raise ImportError("\n<Pygame> library is missing on your system."
          "\nTry: \n   C:\\pip install pygame on a window command prompt.")

try:
    from PygameShader.misc import *
except ImportError:
    raise ImportError("\n<misc> library is missing on your system.")

import random
from random import randint

pygame.font.init()
font = pygame.font.SysFont("Arial", 15)

sprites = []

from PygameShader.gaussianBlur5x5 import blur


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


class water_droplet(object):
    def __init__(self, size_x: int, size_y: int, dx: float, dy: float, x: int, y: int):
        self.size = pygame.Vector2(size_x, size_y)
        self.model = fisheye_footprint(
            size_x + dx,
            size_y + dy,
            int(size_x + dx) >> 1,
            int(size_y + dy) >> 1)
        self.surf = pygame.Surface((size_x + dx, size_y + dy))
        self.pos = pygame.Vector2(x, y)
        self.dv = pygame.Vector2(0, randint(5, 50))


def water_drop(sprite, SCREEN):

    for s in sprite:

        s.pos += s.dv
        s.pos.x += randint(-5, 5)
        s.surf.blit(SCREEN, (0, 0), (s.pos.x, s.pos.y, s.size.x, s.size.y))

        s.surf.blit(SCREEN, (0, 0), (s.pos.x, s.pos.y, s.size.x, s.size.y))
        fisheye(s.surf, s.model)
        SCREEN.blit(s.surf, (s.pos.x, s.pos.y))
        rect = s.surf.get_rect()
        rect.topleft = (s.pos.x, s.pos.y)

        if not SCREEN.get_rect().contains(rect):
            sprite.remove(s)


def show_fps(screen_, fps_, avg_) -> list:

    """ Show framerate in upper left corner """

    fps = str(f"fps:{fps_:.3f}")
    av = sum(avg_)/len(avg_) if len(avg_) > 0 else 0
    fps_text = font.render(fps, True, pygame.Color("coral"))
    screen_.blit(fps_text, (10, 0))
    if av != 0:
        av = str(f"avg:{av:.3f}")
        avg_text = font.render(av, True, pygame.Color("coral"))
        screen_.blit(avg_text, (100, 0))
    if len(avg_) > 200:
        avg_ = avg_[200:]
    return avg_


width = 800
height = 800

SCREENRECT = pygame.Rect(0, 0, width, height)
SCREEN = pygame.display.set_mode(
    SCREENRECT.size, 32)

try:
    BCK = pygame.image.load("../Assets/city.jpg").convert()
except FileNotFoundError:
    raise FileNotFoundError(
        '\nSound file city.jpg is missing from the Assets directory.')

BCK = pygame.transform.smoothscale(BCK, (width, height))
blur(BCK, 5)

pygame.init()

clock = pygame.time.Clock()
FRAME = 0
STOP_GAME = True

event_pump = pygame.event.pump
event_get = pygame.event.get
get_key = pygame.key.get_pressed


f_model = fisheye_footprint(200, 200, 100, 100)
f_model1 = fisheye_footprint(100, 100, 50, 50)
s_surf = pygame.Surface((200, 200))
s_surf1 = pygame.Surface((100, 100))

SCREEN.blit(BCK, (0, 0))
s_surf.blit(SCREEN, (0, 0))
s_surf1.blit(SCREEN, (0, 0), (300, 300, 100, 100))

x = 0
dx = -20
dx_v = 0.1
dy = 0

avg = []
while STOP_GAME:

    event_pump()

    keys = get_key()
    for event in event_get():

        if keys[pygame.K_ESCAPE]:
            STOP_GAME = False
            break

    SCREEN.blit(BCK, (0, 0))

    water_drop(sprites, SCREEN)

    clock.tick(500)
    t = clock.get_fps()

    FRAME += 1

    if len(sprites) < 50:
        dx = randint(40, 80)
        sprites.append(water_droplet(
            dx, dx + randint(10, 100),
            0, 0, randint(dx, 700), 0))

    avg.append(t)
    avg = show_fps(SCREEN, t, avg)

    pygame.display.flip()
    pygame.display.set_caption(
        "Demo rain effect %s fps "
        "(%sx%s)" % (round(t, 2), width, height))


pygame.quit()