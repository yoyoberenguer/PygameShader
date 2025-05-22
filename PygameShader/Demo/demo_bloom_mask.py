"""
PygameShader bloom DEMO

This demo use the CPU power only to generate a bloom effect
"""


import pygame
from pygame import RLEACCEL
from PygameShader import bloom
import numpy

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
        screen_.blit(avg_text, (120, 0))
    if len(avg_) > 200:
        avg_ = avg_[200:]
    return avg_


WIDTH = 800
HEIGHT = 600
SCREEN = pygame.display.set_mode((WIDTH, HEIGHT), 32)
SCREEN.convert(32, RLEACCEL)
SCREEN.set_alpha(None)
pygame.init()

try:
    background = pygame.image.load("../Assets/space3.jpg").convert()
except FileNotFoundError:
    raise FileNotFoundError(
        '\nImage file space3.jpg is missing from the Assets directory.')

background = pygame.transform.smoothscale(background, (WIDTH, HEIGHT))

image = background.copy()

FRAME = 0
CLOCK = pygame.time.Clock()
GAME = True
V = -1
BPF = 255
avg = []

from pygame.surfarray import pixels_alpha
from pygame.transform import smoothscale
# mask = pixels_alpha(smoothscale(pygame.image.load('../Assets/radial4.png'), (WIDTH, HEIGHT)).convert_alpha())/255.0
mask = smoothscale(pygame.image.load('../Assets/radial4.png'),(WIDTH, HEIGHT)).convert_alpha()
mask = pygame.surfarray.pixels_alpha(mask)

while GAME:

    pygame.event.pump()
    for event in pygame.event.get():

        keys = pygame.key.get_pressed()

        if keys[pygame.K_ESCAPE]:
            GAME = False
            break

    SCREEN.blit(image, (0, 0))
    #mask = None
    bloom(SCREEN, threshold_ = BPF, fast_=True, mask_ = mask)

    t = CLOCK.get_fps()
    avg.append(t)
    avg = show_fps(SCREEN, t, avg)

    CLOCK.tick()
    FRAME += 1

    pygame.display.flip()

    if BPF >= 255.0:
        V *= -1

    elif BPF <= 0.0:
        V *= -1

    BPF += V

    if BPF > 255:
        BPF = 255

    if BPF < 0:
        BPF = 0

    pygame.display.set_caption(
        "Demo bloom effect %s fps "
        "(%sx%s)" % (round(t, 2), WIDTH, HEIGHT))

pygame.quit()
