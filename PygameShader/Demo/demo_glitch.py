import pygame
from pygame import RLEACCEL
import PygameShader
from PygameShader import shader_horizontal_glitch24_inplace

WIDTH = 1024
HEIGHT = 768
SCREEN = pygame.display.set_mode((WIDTH, HEIGHT), vsync=True)
SCREEN.convert(32, RLEACCEL)
SCREEN.set_alpha(None)

BACKGROUND = pygame.image.load("../Assets/Background.jpg").convert()
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

    shader_horizontal_glitch24_inplace(image, 0.5, 0.08, FRAME % 20)

    SCREEN.blit(image, (0, 0))

    pygame.display.flip()
    CLOCK.tick()
    FRAME += 1
    
    image = BACKGROUND.copy()
   
