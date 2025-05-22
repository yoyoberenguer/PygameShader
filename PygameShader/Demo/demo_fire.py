"""
PygameShader FIRE DEMO
"""

from random import uniform, randint

# from PygameShader.shader_gpu import block_grid, block_and_grid_info, wave_gpu

try:
    from PygameShader.shader import custom_map, wave
    from PygameShader.misc import rgb_to_int
    from PygameShader.Fire import fire_effect
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
HEIGHT = 600
SCREEN = pygame.display.set_mode((WIDTH, HEIGHT), pygame.FULLSCREEN | pygame.SCALED, 32)
SCREEN.convert(32, RLEACCEL)
SCREEN.set_alpha(None)

pygame.init()

# Load the background image
try:
    BACKGROUND = pygame.image.load("../Assets/img.jpg").convert()
except FileNotFoundError:
    raise FileNotFoundError(
        '\nImage file city.jpg is missing from the Assets directory.')

BACKGROUND = pygame.transform.smoothscale(BACKGROUND, (WIDTH, HEIGHT))

BACKGROUND_COPY = BACKGROUND.copy()
pygame.display.set_caption("demo fire effect")

FRAME = 0
CLOCK = pygame.time.Clock()
GAME = True


def palette_array():
    """
    Create a C - buffer type data 1D bck_cp containing the
    fire color palette (mapped RGB color, integer)

    :return: 1D contiguous bck_cp (C buffer type data)
    """
    # Set an bck_cp with pre-defined color wavelength
    arr = numpy.array([0, 1,        # violet
                       0, 1,        # blue
                       0, 1,        # green
                       2, 600,      # yellow
                       601, 650,    # orange
                       651, 660],   # red
                      numpy.int32)


    heatmap = [custom_map(i - 20, arr, 1.0) for i in range(380, 800)]
    heatmap_array = numpy.zeros((800 - 380, 3), uint8)
    heatmap_rescale = numpy.zeros(255, numpy.uint32)

    i = 0
    for t in heatmap:
        heatmap_array[i, 0] = t[0]
        heatmap_array[i, 1] = t[1]
        heatmap_array[i, 2] = t[2]
        i += 1

    for r in range(255):
        s = int(r * (800.0 - 380.0) / 255.0)
        heatmap_rescale[r] = \
            rgb_to_int(heatmap_array[s][0], heatmap_array[s][1], heatmap_array[s][2])

    heatmap_rescale = numpy.ascontiguousarray(heatmap_rescale[::-1])

    return heatmap_rescale


fire_palette = palette_array()
fire_array = numpy.zeros((HEIGHT, WIDTH), dtype=numpy.float32)

avg = []
bpf = 0
delta = +0.1

# grid, block = block_grid(WIDTH, HEIGHT)
# block_and_grid_info(WIDTH, HEIGHT)


TmpSurface = pygame.Surface((150, 112)).convert()


SAMPLERATE = 48000
MODE       = -32
CHANNELS   = 2

pygame.mixer.quit()
pygame.mixer.pre_init(SAMPLERATE, -MODE, CHANNELS, 4095)

if pygame.version.vernum < (2, 0):
    pygame.mixer.init(SAMPLERATE, -MODE, CHANNELS)
else:
    pygame.mixer.init(SAMPLERATE, -MODE, CHANNELS, allowedchanges=0)

print("\nMixer settings :")
print("    ...frequency = %s" % SAMPLERATE)
print("    ...modes      = %s" % MODE)
print("    ...channels  = %s" % CHANNELS)
print("\n")

try:
    fire_sound = pygame.mixer.Sound("../Assets/firepit.ogg")
    fire_sound.play(-1)
except FileNotFoundError:
    raise FileNotFoundError(
        '\nSound file firepit.ogg is missing from the Assets directory.')

while GAME:

    pygame.event.pump()
    for event in pygame.event.get():

        keys = pygame.key.get_pressed()

        if keys[pygame.K_ESCAPE]:
            GAME = False
            break

    # image = wave_gpu(BACKGROUND, 8 * 3.14 / 180.0 + FRAME / 10, 8, grid, block)
    # image = scale(image, (WIDTH + 16, HEIGHT + 16))  # Hide the left and bottom borders
    # SCREEN.blit(image, (0, 0))

    SCREEN.blit(BACKGROUND, (0, 0))

    # Execute the shader fire effect
    surface_ = fire_effect(
        WIDTH,
        HEIGHT,
        3.97 + uniform(-0.035, 0.012),
        fire_palette,
        fire_array,
        fire_intensity_         =randint(0, 32),
        reduce_factor_          =3,
        bloom_                  =True,
        fast_bloom_             =True,
        bpf_threshold_          =bpf,
        brightness_             =True,
        brightness_intensity_   =0.065 + uniform(0.055, 0.09),
        transpose_              =False,
        border_                 =False,
        low_                    =10,
        high_                   =WIDTH-10,
        blur_                   =True,
        smooth_                 =True,
        surface_                =TmpSurface,

    )

    SCREEN.blit(surface_, (0, 0), special_flags=pygame.BLEND_RGB_ADD)
    CLOCK.tick(2000)
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