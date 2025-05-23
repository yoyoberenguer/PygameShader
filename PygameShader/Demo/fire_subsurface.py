"""
PygameShader FIRE DEMO
"""

from random import uniform, randint

try:
    from PygameShader.shader import custom_map, rgb_to_int, fire_effect, wave
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
HEIGHT = 800
SCREEN = pygame.display.set_mode((WIDTH, HEIGHT), pygame.SCALED)
SCREEN.convert(32, RLEACCEL)
SCREEN.set_alpha(None)

pygame.init()

# Load the background image
BACKGROUND = pygame.image.load("../Assets/img.png").convert()
BACKGROUND = pygame.transform.smoothscale(BACKGROUND, (WIDTH, HEIGHT))

BACKGROUND_COPY = BACKGROUND.copy()
pygame.display.set_caption("demo fire effect")

FRAME = 0
CLOCK = pygame.time.Clock()
GAME = True

sw = 300
sh = 300

sw1 = 100
sh1 = 100

fire_palette = numpy.zeros(255, dtype=numpy.uint)

fire_array = numpy.zeros((sw, sh), dtype=numpy.float32)
fire_array1 = numpy.zeros((sw1, sh1), dtype=numpy.float32)

avg = []
bpf = 0
delta = +0.1


# Create a tmp surface to speed up the process
# If you do not know the surface size, just add random
# values and look for the error message that will give
# you the exact dimensions
TmpSurface = pygame.Surface((100, 100)).convert()


while GAME:

    pygame.event.pump()
    for event in pygame.event.get():

        keys = pygame.key.get_pressed()

        if keys[pygame.K_ESCAPE]:
            GAME = False
            break

    SCREEN.blit(BACKGROUND, (0, 0))

    # Execute the shader fire effect
    surface_ = fire_effect(
        sw,
        sh,
        4.05 + uniform(-0.150, 0.281),
        fire_palette,
        fire_array,
        fire_intensity_         =randint(0, 5),
        reduce_factor_          =4,
        bloom_                  =True,
        fast_bloom_             =True,
        bpf_threshold_          =bpf,
        brightness_             =True,
        brightness_intensity_   =0.065 + uniform(0.055, 0.09),
        transpose_              =False,
        border_                 =False,
        low_                    =20,
        high_                   =sw-20,
        blur_                   =True,
        smooth_                 =True,
        surface_                =TmpSurface,
        # No need to define a palette pre-processing,
        # the algo will create a new palette with the given
        # hsl_ values
        adjust_palette_=True,
        hsl_=(0.2, 200, 1.8)
    )

    # Execute the shader fire effect
    surface_1 = fire_effect(
        sw1,
        sh1,
        3.95 + uniform(-0.150, 0.281),
        fire_palette,
        fire_array1,
        fire_intensity_=randint(0, 32),
        reduce_factor_=4,
        bloom_=True,
        fast_bloom_=True,
        bpf_threshold_=bpf,
        brightness_=True,
        brightness_intensity_=0.065 + uniform(0.055, 0.09),
        transpose_=False,
        border_=False,
        low_=20,
        high_=sw1-20 ,
        blur_=True,
        smooth_=True,
        # surface_=TmpSurface,
        # No need to define a palette pre-processing,
        # the algo will create a new palette with the given
        # hsl_ values
        adjust_palette_=True,
        hsl_=(0.2, 200, 1.8)
    )

    SCREEN.blit(surface_, (400, 200), special_flags=pygame.BLEND_RGB_MAX)
    SCREEN.blit(surface_1, (200, 450), special_flags=pygame.BLEND_RGB_MAX)

    CLOCK.tick(800)
    t = CLOCK.get_fps()
    avg.append(t)
    avg = show_fps(SCREEN, t, avg)
    pygame.display.flip()

    FRAME += 1

    bpf += delta
    bpf = max(bpf, 45)
    bpf = min(bpf, 0)
    if bpf == 45:
        delta *= -1

    pygame.display.set_caption(
        "Test fire_effect %s fps "
        "(%sx%s)" % (round(CLOCK.get_fps(), 2), WIDTH, HEIGHT))


pygame.quit()