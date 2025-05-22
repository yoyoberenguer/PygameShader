"""
PygameShader FISHEYE DEMO
"""
from random import randint, uniform, randrange



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

try:
    import cupy
except ImportError:
    raise ImportError("\n<Pygame> library is missing on your system."
                      "\nTry: \n   C:\\pip install cupy on a window command prompt.")

try:
    import PygameShader
    from PygameShader.shader_gpu import block_grid, ripple_effect_gpu, \
        get_gpu_info, block_and_grid_info, area24_gpu, fisheye_gpu, wave_gpu
    from PygameShader.shader import blend
except ImportError:
    raise ImportError("\n<PygameShader> library is missing on your system."
                      "\nTry: \n   C:\\pip install PygameShader on a window command prompt.")

pygame.font.init()
font = pygame.font.SysFont("Arial", 15)


def show_fps(screen_, fps_, avg_) -> list:
    """ Show framerate in upper left corner """

    fps = str(f"fps:{fps_:.3f}")
    av = sum(avg_)/len(avg_) if len(avg_) > 0 else 0

    fps_text = font.render(fps, True, pygame.Color("coral"))
    screen_.blit(fps_text, (10, 0))
    if av != 0:
        av = str(f"avg:{av:.3f}")
        avg_text = font.render(av, True, pygame.Color("coral"))
        screen_.blit(avg_text, (90, 0))
    if len(avg_) > 200:
        avg_ = avg_[200:]
    return avg_


get_gpu_info()

width = 800
height = 600

SCREENRECT = pygame.Rect(0, 0, width, height)
SCREEN = pygame.display.set_mode(SCREENRECT.size, pygame.FULLSCREEN)

pygame.init()

try:
    background = pygame.image.load('..//Assets//Aliens.jpg').convert()
except FileNotFoundError:
    raise FileNotFoundError(
        '\nImage file Aliens.jpg is missing from the Assets directory.')

background = pygame.transform.smoothscale(background, (800, 600))
background_copy = background.copy()

try:
    city = pygame.image.load('..//Assets//city.jpg').convert()
except FileNotFoundError:
    raise FileNotFoundError(
        '\nImage file city.jpg is missing from the Assets directory.')

city = pygame.transform.smoothscale(city, (800, 600))

MOUSE_POS = [0, 0]
FRAME = 0
clock = pygame.time.Clock()
avg = []

# TWEAKS
cget_fps = clock.get_fps
event_pump = pygame.event.pump
event_get = pygame.event.get
get_key = pygame.key.get_pressed
get_pos = pygame.mouse.get_pos
flip = pygame.display.flip

STOP_GAME = True

grid, block = block_grid(width, height)
block_and_grid_info(width, height)

VALUE = 0.0
V = +0.2
while STOP_GAME:

    event_pump()

    keys = get_key()
    for event in event_get():

        if keys[pygame.K_ESCAPE]:
            STOP_GAME = False
            break

        if event.type == pygame.MOUSEMOTION:
            MOUSE_POS = event.pos

    # transition = blend(
    #     source=background, destination=city, percentage=VALUE)

    image = fisheye_gpu(background, VALUE/49.0, 0.35, grid, block)
    SCREEN.blit(image, (0, 0))

    t = clock.get_fps()
    avg.append(t)
    avg = show_fps(SCREEN, t, avg)
    clock.tick()
    FRAME += 1

    pygame.display.set_caption(
        "Demo Fisheye GPU %s fps"
        "(%sx%s)" % (round(clock.get_fps(), 2), width, height))

    flip()

    VALUE += V

    if VALUE >= 100:
        VALUE = 100
        V = -0.2
    if VALUE <= 0:
        VALUE = 0
        V = 0.2

pygame.quit()