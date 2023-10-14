import pygame
from pygame import RLEACCEL
import PygameShader
from PygameShader.shader import horizontal_glitch

WIDTH = 1024
HEIGHT = 768
SCREEN = pygame.display.set_mode((WIDTH, HEIGHT), pygame.FULLSCREEN | pygame.SCALED)
SCREEN.convert(32, RLEACCEL)
SCREEN.set_alpha(None)

try:
    BACKGROUND = pygame.image.load("../Assets/img.jpg").convert()
except FileNotFoundError:
    raise FileNotFoundError(
        '\nImage file img.jpg is missing from the Assets directory.')

BACKGROUND = pygame.transform.smoothscale(BACKGROUND, (WIDTH, HEIGHT))

pygame.font.init()
font = pygame.font.SysFont("Arial", 15)

FRAME = 0
CLOCK = pygame.time.Clock()
GAME = True

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

avg=[]

while GAME:

    pygame.event.pump()
    for event in pygame.event.get():

        keys = pygame.key.get_pressed()

        if keys[pygame.K_ESCAPE]:
            GAME = False
            break

    t = CLOCK.get_fps()
    avg.append(t)

    SCREEN.blit(BACKGROUND, (0, 0))

    avg = show_fps(SCREEN, t, avg)

    horizontal_glitch(SCREEN, 0.5, 0.08, FRAME % 20)

    pygame.display.flip()
    CLOCK.tick()
    FRAME += 1

    pygame.display.set_caption(
        "Test glitch effect %s fps "
        "(%sx%s)" % (round(CLOCK.get_fps(), 2), WIDTH, HEIGHT))


pygame.quit()