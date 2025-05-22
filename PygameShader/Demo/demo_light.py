"""
PygameShader demo light

"""


import pygame
import numpy
from pygame.surfarray import pixels3d, array3d

from PygameShader import area24_cc, render_light_effect24, BLEND_RGB_ADD
from PygameShader.shader import blend_add_array

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
        screen_.blit(avg_text, (140, 0))
    if len(avg_) > 200:
        avg_ = avg_[200:]
    return avg_


width = 800
height = 600

SCREENRECT = pygame.Rect(0, 0, width, height)
SCREEN = pygame.display.set_mode(SCREENRECT.size,  32)

pygame.init()

try:
    background = pygame.image.load('..//Assets//Aliens.jpg').convert()
except FileNotFoundError:
    raise FileNotFoundError(
        '\nImage file aliens.jpg is missing from the Assets directory.')

background = pygame.transform.smoothscale(background, (800, 600))

background_rgb = numpy.asarray(pixels3d(background), dtype=numpy.uint8)

w, h = background.get_size()


pygame.display.set_caption("CPU demo light effect")

try:
    light = pygame.image.load('..//Assets//Radial8.png').convert_alpha()
except FileNotFoundError:
    raise FileNotFoundError(
        '\nImage file Radial8.png is missing from the Assets directory.')

light = pygame.transform.smoothscale(light, (400, 400))
lalpha = numpy.asarray(pygame.surfarray.pixels_alpha(light), dtype=numpy.uint8)

lw, lh = light.get_size()
lw2, lh2 = lw >> 1, lh >> 1

c = numpy.array([222.0 / 255.0, 178.0 / 255.0, 128.0 / 255.0], numpy.float32, copy=False)
MOUSE_POS = [0, 0]
FRAME = 0
clock = pygame.time.Clock()
avg = []

# TWEAKS
cget_fps = clock.get_fps
event_pump = pygame.event.pump
event_get = pygame.event.get
get_key = pygame.key.get_pressed
get_pos = pygame.mouse.get_pos
flip = pygame.display.flip

STOP_GAME = True

MOUSE_POS = pygame.math.Vector2()
MOUSE_POS.x = 0
MOUSE_POS.y = 0

while STOP_GAME:

    event_pump()

    keys = get_key()
    for event in event_get():

        if keys[pygame.K_ESCAPE]:
            STOP_GAME = False
            break

        if event.type == pygame.MOUSEMOTION:
            MOUSE_POS = list(event.pos)

            if MOUSE_POS[0] < 0:MOUSE_POS[0] = 0
            if MOUSE_POS[0] > width:MOUSE_POS[0] = width
            if MOUSE_POS[1] < 0:MOUSE_POS[1] = 0
            if MOUSE_POS[1] > height:MOUSE_POS[1] = height

    lit_surface, sw, sh = area24_cc(
        MOUSE_POS[0], MOUSE_POS[1], background_rgb, lalpha, intensity=5, color=c,
        smooth=False, saturation=False, sat_value=0.2, bloom=True, bloom_threshold=0
    )

    if sw < lw and MOUSE_POS[0] <= lw - lw2:
        xx = 0
    else:
        xx = MOUSE_POS[0] - lw2

    if sh < lh and MOUSE_POS[1] <= lh - lh2:
        yy = 0
    else:
        yy = MOUSE_POS[1] - lh2

    SCREEN.fill((0, 0, 0))
    SCREEN.blit(lit_surface, (xx, yy)) # , special_flags=BLEND_RGB_ADD)

    t = clock.get_fps()
    avg.append(t)

    avg = show_fps(SCREEN, t, avg)

    clock.tick()
    FRAME += 1

    flip()

pygame.quit()