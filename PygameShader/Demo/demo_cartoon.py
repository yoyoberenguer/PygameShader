"""
PygameShader DEMO CARTOON EFFECT
"""

try:
    from PygameShader.shader import cartoon
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

except ImportError:
    raise ImportError("\n<Pygame> library is missing on your system."
          "\nTry: \n   C:\\pip install pygame on a window command prompt.")

# Set the display to 640 x 480
WIDTH = 640
HEIGHT = 480
SCREEN = pygame.display.set_mode((WIDTH, HEIGHT), vsync=True)
SCREEN.convert(32, RLEACCEL)
SCREEN.set_alpha(None)

# Load the background image
BACKGROUND = pygame.image.load("../Assets/Background.jpg").convert()
BACKGROUND = pygame.transform.smoothscale(BACKGROUND, (WIDTH, HEIGHT))

image = BACKGROUND.copy()
pygame.display.set_caption("demo cartoon effect")

FRAME = 0
CLOCK = pygame.time.Clock()
GAME = True


while GAME:

    pygame.event.pump()
    for event in pygame.event.get():

        keys = pygame.key.get_pressed()

        if keys[pygame.K_ESCAPE]:
            GAME = False
            break

    surface_ = cartoon(image, sobel_threshold_ = 32, median_kernel_=2, color_=128,
                       flag_=BLEND_RGB_ADD).convert()

    SCREEN.blit(surface_, (0, 0), special_flags=0)

    pygame.display.flip()
    CLOCK.tick()
    FRAME += 1

    pygame.display.set_caption(
        "Test shader_cartoon %s fps "
        "(%sx%s)" % (round(CLOCK.get_fps(), 2), WIDTH, HEIGHT))

    image = BACKGROUND.copy()
