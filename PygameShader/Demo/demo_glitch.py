import pygame
from pygame import RLEACCEL
import PygameShader
from PygameShader.shader import horizontal_glitch

WIDTH = 1024
HEIGHT = 768
SCREEN = pygame.display.set_mode((WIDTH, HEIGHT), vsync=True)
SCREEN.convert(32, RLEACCEL)
SCREEN.set_alpha(None)

BACKGROUND = pygame.image.load("../Assets/background.jpg").convert()
BACKGROUND = pygame.transform.smoothscale(BACKGROUND, (WIDTH, HEIGHT))

FRAME = 0
CLOCK = pygame.time.Clock()
GAME = True

image = BACKGROUND.copy()

while GAME:

    pygame.event.pump()
    for event in pygame.event.get():

        keys = pygame.key.get_pressed()

        if keys[pygame.K_ESCAPE]:
            GAME = False
            break

    horizontal_glitch(image, 0.5, 0.08, FRAME % 20)

    SCREEN.blit(image, (0, 0))

    pygame.display.flip()
    CLOCK.tick()
    FRAME += 1

    image = BACKGROUND.copy()

pygame.quit()