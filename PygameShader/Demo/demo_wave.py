"""
PygameShader WAVE DEMO
"""
import math

try:
    from PygameShader.shader import wave
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
    from pygame import Surface, RLEACCEL, QUIT, K_SPACE, BLEND_RGB_ADD
    from pygame.transform import scale

except ImportError:
    raise ImportError("\n<Pygame> library is missing on your system."
          "\nTry: \n   C:\\pip install pygame on a window command prompt.")

pygame.font.init()
font = pygame.font.SysFont("Arial", 15)


def show_fps(screen_, fps_, avg_) -> list:
    """ Show framerate in upper left corner """

    fps = str(f"CPU fps:{fps_:.3f}")
    av = sum(avg_)/len(avg_) if len(avg_) > 0 else 0

    fps_text = font.render(fps, True, pygame.Color("green"))
    screen_.blit(fps_text, (10, 0))
    if av != 0:
        av = str(f"avg:{av:.3f}")
        avg_text = font.render(av, True, pygame.Color("green"))
        screen_.blit(avg_text, (120, 0))
    if len(avg_) > 200:
        avg_ = avg_[200:]
    return avg_

# Set the display to 1024 x 768
WIDTH = 800
HEIGHT = 600
SCREEN = pygame.display.set_mode((WIDTH, HEIGHT), pygame.FULLSCREEN | pygame.SCALED, 32)
SCREEN.convert(32, RLEACCEL)
SCREEN.set_alpha(None)
pygame.init()

# Load the background image
try:
    BACKGROUND = pygame.image.load("../Assets/background2.jpg").convert()
except FileNotFoundError:
    raise FileNotFoundError(
        '\nImage file background2.jpg is missing from the Assets directory.')

BACKGROUND = pygame.transform.smoothscale(BACKGROUND, (WIDTH, HEIGHT))

image = BACKGROUND.copy()
pygame.display.set_caption("demo wave effect")

FRAME = 0
CLOCK = pygame.time.Clock()
GAME = True

ANGLE = 0
avg = []


while GAME:

    pygame.event.pump()
    for event in pygame.event.get():

        keys = pygame.key.get_pressed()

        if keys[pygame.K_ESCAPE]:
            GAME = False
            break

    wave(image, ANGLE * math.pi / 180.0, 12)
    image = scale(image, (WIDTH + 90, HEIGHT + 90))
    SCREEN.blit(image, (-50, -50))

    t = CLOCK.get_fps()
    avg.append(t)
    avg = show_fps(SCREEN, t, avg)

    pygame.display.flip()
    CLOCK.tick()
    FRAME += 1
    ANGLE += 5
    ANGLE %= 360

    pygame.display.set_caption(
        "Test shader wave effect %s fps "
        "(%sx%s)" % (round(CLOCK.get_fps(), 2), WIDTH, HEIGHT))

    image = BACKGROUND.copy()

pygame.quit()