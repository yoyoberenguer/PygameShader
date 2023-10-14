"""
PygameShader DEMO PREDATOR VISION
"""
import time

import pygame
from pygame import RLEACCEL

# from PygameShader import
from PygameShader.shader import bluescale, shader_bloom_fast1, redscale, bpf
from PygameShader.shader_gpu import predator_gpu, block_grid

pygame.font.init()
font = pygame.font.SysFont("Arial", 15)


def show_fps(screen_, fps_, avg_) -> list:
    """ Show framerate in upper left corner """

    fps = str(f"CPU fps:{fps_:.3f}")
    av = sum(avg_) / len(avg_) if len(avg_) > 0 else 0

    fps_text = font.render(fps, True, pygame.Color("beige"))
    screen_.blit(fps_text, (10, 0))
    if av != 0:
        av = str(f"avg:{av:.3f}")
        avg_text = font.render(av, True, pygame.Color("beige"))
        screen_.blit(avg_text, (120, 0))
    if len(avg_) > 10:
        avg_ = avg_[ 10: ]
    return avg_


WIDTH = 800
HEIGHT = 600
SCREEN = pygame.display.set_mode((WIDTH, HEIGHT), pygame.FULLSCREEN | pygame.SCALED)
SCREEN.convert(32, RLEACCEL)
SCREEN.set_alpha(None)
pygame.init()

try:
    IMAGE = pygame.image.load("../Assets/city.jpg").convert()
except FileNotFoundError:
    raise FileNotFoundError(
        '\nImage file city.jpg is missing from the Assets directory.')

IMAGE = pygame.transform.smoothscale(IMAGE, (WIDTH, HEIGHT))

IMAGE_COPY = IMAGE.copy()

FRAME = 0
CLOCK = pygame.time.Clock()
GAME = True
avg = []
grid, block = block_grid(IMAGE.get_width(), IMAGE.get_height())

try:
    pred = pygame.mixer.Sound("../Assets/predator.ogg")
except FileNotFoundError:
    raise FileNotFoundError(
        '\nSound file predator.ogg is missing from the Assets directory.')

pred_play = False

try:
    vision_swap = pygame.mixer.Sound("../Assets/vision_swap.ogg")
except FileNotFoundError:
    raise FileNotFoundError(
        '\nSound file vision_swap.ogg is missing from the Assets directory.')

vision_swap_play = False
vision_swap_play1 = False
vision_swap_play2 = False

bl = 0
vbl = 1
ti = time.time()

while GAME:

    pygame.event.pump()
    for event in pygame.event.get():

        keys = pygame.key.get_pressed()

        if keys[ pygame.K_ESCAPE]:
            GAME = False
            break

    if 4 < time.time() - ti < 9:

        if not pred_play:
            pred.play()
        IMAGE = predator_gpu(
            IMAGE,
            grid, block,
            bloom_smooth = 10,
            bloom_threshold = int(bl),
            inv_colormap=True,
            blend = pygame.BLEND_RGB_MAX,
            bloom_flag = pygame.BLEND_RGB_ADD
        )
        pred_play = True

    elif 9 <= time.time() - ti < 16:
        if vision_swap_play is False:
            pred.stop()
            pred.play()
        shader_bloom_fast1(IMAGE, smooth_=4, threshold_=int(bl), flag_=pygame.BLEND_RGB_ADD,
                           saturation_=True)
        bluescale(IMAGE)
        vision_swap_play = True

    elif 16 <= time.time() - ti < 22:
        if vision_swap_play1 is False:
            pred.stop()
            pred.play()
        shader_bloom_fast1(IMAGE, smooth_=4, threshold_=int(bl), flag_=pygame.BLEND_RGB_ADD,
                           saturation_=True)
        redscale(IMAGE)
        vision_swap_play1 = True

    elif 22 <= time.time() - ti < 50:
        if vision_swap_play2 is False:
            pred.stop()
            pred.play()

        shader_bloom_fast1(IMAGE, smooth_=4, threshold_=int(bl),
                           flag_=pygame.BLEND_RGB_ADD, saturation_=True)

        bpf(IMAGE, threshold=75)

        vision_swap_play2 = True

    SCREEN.blit(IMAGE, (0, 0))

    if bl >= 50:
        vbl = -1
    elif bl <= 0:
        vbl = +1
    bl += vbl

    t = CLOCK.get_fps()
    avg.append(t)
    avg = show_fps(SCREEN, t, avg)

    CLOCK.tick()
    FRAME += 1

    pygame.display.flip()

    IMAGE = IMAGE_COPY.copy()

pygame.quit()
