"""
PygameShader TUNNEL DEMO
This effect use also ZOOM and SCROLL algorithms

This demo use the CPU power only to generate a tunnel effect
"""


import pygame
from pygame import RLEACCEL
from pygame.surfarray import pixels3d
from PygameShader import tunnel_modeling24, tunnel_render24,\
    zoom, scroll24_inplace, blend_inplace
from PygameShader.BlendFlags import blend_add_surface
import numpy
import math
from math import sin, floor

pygame.font.init()
font = pygame.font.SysFont("Arial", 15)


def show_fps(screen_, fps_, avg_) -> list:
    """ Show framerate in upper left corner """
    fps = str(f"CPU fps:{fps_:.3f}")
    av = sum(avg_)/len(avg_) if len(avg_) > 0 else 0

    fps_text = font.render(fps, True, pygame.Color("beige"))
    screen_.blit(fps_text, (10, 0))
    if av != 0:
        av = str(f"avg:{av:.3f}")
        avg_text = font.render(av, True, pygame.Color("beige"))
        screen_.blit(avg_text, (125, 0))
    if len(avg_) > 200:
        avg_ = avg_[200:]
    return avg_


# !!! WIDTH and HEIGHT MUST BE IDENTICAL !!!
WIDTH = 800
HEIGHT = 800

SCREEN = pygame.display.set_mode((WIDTH, HEIGHT), 32)
SCREEN.convert(32, RLEACCEL)
SCREEN.set_alpha(None)
pygame.init()


FRAME = 0
CLOCK = pygame.time.Clock()
GAME = True


try:
    BCK1 = pygame.image.load("../Assets/Bokeh__Lens_Dirt_54.jpg").convert()
except FileNotFoundError:
    raise FileNotFoundError(
        '\nImage file Bokeh__Lens_Dirt_54.jpg is missing from the Assets directory.')

BCK1 = pygame.transform.smoothscale(BCK1, (WIDTH, HEIGHT))

try:
    BCK2 = pygame.image.load("../Assets/space3.jpg").convert()
except FileNotFoundError:
    raise FileNotFoundError(
        '\nImage file space3.jpg is missing from the Assets directory.')

BCK2 = pygame.transform.smoothscale(BCK2, (WIDTH, HEIGHT))

try:
    BCK3 = pygame.image.load("../Assets/space7.jpg").convert()
except FileNotFoundError:
    raise FileNotFoundError(
        '\nImage file space7.jpg is missing from the Assets directory.')

BCK3 = pygame.transform.smoothscale(BCK3, (WIDTH, HEIGHT))
blend_add_surface(BCK3, BCK3)

try:
    BACKGROUND = pygame.image.load("../Assets/space1_alpha.jpg")
except FileNotFoundError:
    raise FileNotFoundError(
        '\nImage file space1_alpha.jpg is missing from the Assets directory.')

BACKGROUND = pygame.transform.smoothscale(BACKGROUND, (WIDTH, HEIGHT))

FRAME = 0
GAME = True
avg = []
CLOCK = pygame.time.Clock()

distances, angles, shades, scr_data = tunnel_modeling24(WIDTH, HEIGHT, BACKGROUND)
dest_array = numpy.empty((WIDTH * HEIGHT * 4), numpy.uint8)

acceleration = 1.0
dc = 0.02
df = 1
zx = 0


prev_centerx = 400 + floor((400 >> 1) * math.sin(0 * acceleration * 0.25))
prev_centery = 400 + floor((400 >> 1) * math.sin(0 * acceleration * 0.5))


SAMPLERATE = 48000
MODE       = -32
CHANNELS   = 2

pygame.mixer.quit()
pygame.mixer.pre_init(SAMPLERATE, -MODE, CHANNELS, 4095)

if pygame.version.vernum < (2, 0):
    pygame.mixer.init(SAMPLERATE, -MODE, CHANNELS)
else:
    pygame.mixer.init(SAMPLERATE, -MODE, CHANNELS, allowedchanges=0)

print("\nMixer settings :")
print("    ...frequency = %s" % SAMPLERATE)
print("    ...modes      = %s" % MODE)
print("    ...channels  = %s" % CHANNELS)
print("\n")

fire_sound = pygame.mixer.Sound("../Assets/jump_flight.ogg")
fire_sound.play(-1)

while GAME:

    pygame.event.pump()
    for event in pygame.event.get():

        keys = pygame.key.get_pressed()

        if keys[pygame.K_ESCAPE]:
            GAME = False
            break

    centerx = floor((400 >> 1) * math.sin(FRAME * acceleration * 0.25))
    centery = floor((400 >> 1) * math.sin(FRAME * acceleration * 0.5))
    dx = prev_centerx - centerx
    dy = prev_centery - centery

    if dx > 0: dx = -1
    elif dx < 0: dx = 1
    if dy > 0: dy = -1
    elif dy < 0: dy = 1

    scroll24_inplace(BCK2, dx, dy)

    SCREEN.blit(BCK1, (0, 0))

    zx = 0.9999 - (FRAME / float(800.0))
    surf = zoom(BCK2, 400, 400, max(zx, 0))
    SCREEN.blit(surf, (0, 0), special_flags=pygame.BLEND_RGB_ADD)

    blend_inplace(
        source =SCREEN,
        destination =BCK3,
        percentage = (1 - zx) * 60
    )

    surface_ = tunnel_render24(
        FRAME * acceleration,
        WIDTH,
        HEIGHT,
        WIDTH >> 1,
        HEIGHT >> 1,
        distances,
        angles,
        shades,
        scr_data,
        dest_array
    )

    SCREEN.blit(surface_, (0, 0) , special_flags=pygame.BLEND_RGB_ADD)

    CLOCK.tick(500)
    t = CLOCK.get_fps()
    avg.append(t)
    avg = show_fps(SCREEN, t, avg)

    pygame.display.flip()

    FRAME += df
    acceleration += dc

    if acceleration > 15:
        dc *= -1
        df *= -1
    if acceleration < 1:
        dc *= -1

    if FRAME <= 0:
        df *= -1

    pygame.display.set_caption(
        "Demo tunnel effect %s fps "
        "(%sx%s) acceleration %s" % (round(t, 2), WIDTH, HEIGHT, round(acceleration, 2)))

    prev_centerx = centerx
    prev_centery = centery

pygame.quit()


