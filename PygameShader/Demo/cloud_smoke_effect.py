"""
PygameShader SMOKE DEMO

This demo use the CPU power only to generate a cloud effect
"""

from random import uniform, randint

try:
    from PygameShader.Fire import cloud_effect
    from PygameShader.misc import create_line_gradient_rgb, rgb_to_int

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
    from pygame import Surface, RLEACCEL, QUIT, K_SPACE, BLEND_RGB_ADD, \
        BLEND_RGB_SUB, BLEND_RGB_MULT, BLEND_RGB_MAX

except ImportError:
    raise ImportError("\n<Pygame> library is missing on your system."
          "\nTry: \n   C:\\pip install pygame on a window command prompt.")

pygame.font.init()
font = pygame.font.SysFont("Arial", 15)


def show_fps(screen_, fps_, avg_) -> list:
    """ Show framerate in upper left corner """

    fps = str(f"CPU fps:{fps_:.3f}")
    av = sum(avg_)/len(avg_) if len(avg_) > 0 else 0

    fps_text = font.render(fps, True, pygame.Color("coral"))
    screen_.blit(fps_text, (10, 0))
    if av != 0:
        av = str(f"avg:{av:.3f}")
        avg_text = font.render(av, True, pygame.Color("coral"))
        screen_.blit(avg_text, (120, 0))
    if len(avg_) > 200:
        avg_ = avg_[200:]
    return avg_


# Set the display to 1024 x 768
WIDTH = 800
HEIGHT = 600
SCREEN = pygame.display.set_mode((WIDTH, HEIGHT), pygame.FULLSCREEN | pygame.SCALED)
SCREEN.convert(32, RLEACCEL)
SCREEN.set_alpha(None)

pygame.init()

# Load the background image

try:
    BACKGROUND = pygame.image.load("../Assets/img.jpg").convert()
except FileNotFoundError:
    raise FileNotFoundError(
        '\nImage file img.png is missing from the Assets directory.')

BACKGROUND = pygame.transform.smoothscale(BACKGROUND, (WIDTH, HEIGHT))

image = BACKGROUND.copy()
pygame.display.set_caption("Cloud & smoke effect")

FRAME = 0
CLOCK = pygame.time.Clock()
GAME = True


CLOUD_ARRAY = numpy.zeros((HEIGHT, WIDTH), dtype=numpy.float32)

heatmap_rescale = numpy.zeros(256 * 2 * 3, numpy.uint32)

arr1 = create_line_gradient_rgb(255, (0, 0, 0), (150, 150, 150))
arr2 = create_line_gradient_rgb(255, (255, 255, 255), (0, 0, 0))
arr3 = numpy.concatenate((arr1, arr2), axis=None)
i = 0
for r in range(0, 1530, 3):
    heatmap_rescale[i] = rgb_to_int(arr3[r], arr3[r + 1], arr3[r + 2])
    i += 1


avg = []
bpf = 0
delta = +0.1
while GAME:

    pygame.event.pump()
    for event in pygame.event.get():

        keys = pygame.key.get_pressed()

        if keys[pygame.K_ESCAPE]:
            GAME = False
            break

    # SCREEN.fill((0, 0, 0, 0))
    SCREEN.blit(BACKGROUND, (0, 0))

    surface_ = cloud_effect(
        WIDTH, HEIGHT, 3.9650 + uniform(-0.012, 0.008),
        heatmap_rescale,
        CLOUD_ARRAY,
        reduce_factor_=3, cloud_intensity_=randint(0, 128),
        smooth_=True, bloom_=True, fast_bloom_=True,
        bpf_threshold_=bpf, low_=0, high_=WIDTH, brightness_=True,
        brightness_intensity_=-0.05,
        transpose_=False, surface_=None, blur_=False
    ).convert(32, RLEACCEL)

    SCREEN.blit(surface_, (0, 0), special_flags=BLEND_RGB_MAX)
    t = CLOCK.get_fps()
    avg.append(t)
    avg = show_fps(SCREEN, t, avg)
    pygame.display.flip()
    CLOCK.tick()
    FRAME += 1

    bpf += delta
    bpf = max(bpf, 45)
    bpf = min(bpf, 0)
    if bpf == 45:
        delta *= -1

    pygame.display.set_caption(
        "Clound & smoke effect %s fps "
        "(%sx%s)" % (round(CLOCK.get_fps(), 2), WIDTH, HEIGHT))

    image = BACKGROUND.copy()