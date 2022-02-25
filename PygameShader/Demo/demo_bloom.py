import pygame
from pygame import RLEACCEL
import PygameShader
from PygameShader import shader_bloom_effect_array24

WIDTH = 1024
HEIGHT = 768
SCREEN = pygame.display.set_mode((WIDTH, HEIGHT), pygame.FULLSCREEN, vsync=True)
SCREEN.convert(32, RLEACCEL)
SCREEN.set_alpha(None)

background = pygame.image.load("../Assets/Background.jpg").convert()
background = pygame.transform.smoothscale(background, (WIDTH, HEIGHT))    
image = background.copy()
 
FRAME = 0
CLOCK = pygame.time.Clock()
GAME = True
V = 0.5
BPF = 255
        
while GAME:

            pygame.event.pump()
            for event in pygame.event.get():

                keys = pygame.key.get_pressed()

                if keys[pygame.K_ESCAPE]:
                    GAME = False
                    break

            shader_bloom_effect_array24(image, BPF, fast_=True)
           
            SCREEN.blit(image, (0, 0))
            pygame.display.flip()
            CLOCK.tick()
            FRAME += 1
            
            image = background.copy()

            if BPF >= 255.0:
                V *= -1
                BPF = 255.0
            elif BPF <= 0.0:
                V *= -1
                BPF = 0

            BPF += 4 * V
            BPF = min(BPF, 255)
            BPF = max(BPF, 0)
