"""
PygameShader SCROLL DEMO 32-bit image
"""

import pygame
from pygame import RLEACCEL
import PygameShader
from PygameShader.misc import scroll32_inplace


WIDTH = 800
HEIGHT = 800
SCREEN = pygame.display.set_mode((WIDTH, HEIGHT), pygame.SCALED)
SCREEN.convert(32, RLEACCEL)
SCREEN.set_alpha(None)

try:
    BACKGROUND = pygame.image.load("../Assets/px.png").convert_alpha()
except FileNotFoundError:
    raise FileNotFoundError(
        '\nImage file space1.jpg is missing from the Assets directory.')


BACKGROUND = pygame.transform.smoothscale(BACKGROUND, (WIDTH, HEIGHT))

try:
    FILM = pygame.image.load("../Assets/film_strip2.png").convert_alpha()
except FileNotFoundError:
    raise FileNotFoundError(
        '\nImage file film_strip2.png is missing from the Assets directory.')

FILM = pygame.transform.smoothscale(FILM, (WIDTH, HEIGHT))

FRAME = 0
CLOCK = pygame.time.Clock()
GAME = True


SCREEN.blit(BACKGROUND, (0, 0))


x = -1
y = 1
v = pygame.math.Vector2(x, y)
dx = 0.01
dy = 0.01
dv = pygame.math.Vector2(dx, dy)

while GAME:

    pygame.event.pump()
    for event in pygame.event.get():

        keys = pygame.key.get_pressed()

        if keys[pygame.K_ESCAPE]:
            GAME = False
            break

    t = CLOCK.get_fps()
    SCREEN.fill((10, 10, 10))

    scroll32_inplace(BACKGROUND, v.x, v.y)

    SCREEN.blit(BACKGROUND, (0, 0))
    SCREEN.blit(FILM, (0, 0))

    v += dv
    if v.x > 10 or v.x < -10:
        dv.x *= -1

    if v.y > 10 or v.y < -10:
        dv.y *= -1

    pygame.display.flip()
    CLOCK.tick()
    FRAME += 1

    pygame.display.set_caption(
        "Test scroll effect %s fps "
        "(%sx%s)" % (round(CLOCK.get_fps(), 2), WIDTH, HEIGHT))

pygame.quit()