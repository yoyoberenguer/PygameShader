"""
PygameShader GPU LIGHT DEMO
"""


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
        get_gpu_info, block_and_grid_info, area24_gpu
except ImportError:
    raise ImportError("\n<PygameShader> library is missing on your system."
                      "\nTry: \n   C:\\pip install PygameShader on a window command prompt.")


def show_fps(screen_, fps_, avg_) -> None:
    """ Show framerate in upper left corner """
    font = pygame.font.SysFont("Arial", 15)
    fps = str(f"Move your mouse - fps:{fps_:.3f}")
    av = sum(avg_)/len(avg_) if len(avg_) > 0 else 0

    fps_text = font.render(fps, 1, pygame.Color("coral"))
    screen_.blit(fps_text, (10, 0))
    if av != 0:
        av = str(f"avg:{av:.3f}")
        avg_text = font.render(av, 1, pygame.Color("coral"))
        screen_.blit(avg_text, (200, 0))
    if len(avg_) > 200:
        avg_ = avg_[200:]


get_gpu_info()

width = 800
height = 600

SCREENRECT = pygame.Rect(0, 0, width, height)
pygame.display.init()
SCREEN = pygame.display.set_mode(SCREENRECT.size, pygame.FULLSCREEN | pygame.DOUBLEBUF, 32)

pygame.init()

background = pygame.image.load('..//Assets//background2.jpg').convert()
background = pygame.transform.smoothscale(background, (800, 800))
background.set_alpha(10)
background_rgb = cupy.asarray(pygame.surfarray.pixels3d(background), dtype=cupy.uint8)
background_rgb = background_rgb.transpose(1, 0, 2)
w, h = background.get_size()

back = background.copy()
# back.set_alpha(255)

pygame.display.set_caption("GPU demo light effect")

light = pygame.image.load('..//Assets//Radial8.png').convert_alpha()
light = pygame.transform.smoothscale(light, (400, 400))
lalpha = cupy.asarray(pygame.surfarray.pixels_alpha(light), dtype=cupy.uint8)

lw, lh = light.get_size()
lw2, lh2 = lw >> 1, lh >> 1

c = cupy.array([128.0 / 255.0, 128.0 / 255.0, 200.0 / 255.0], numpy.float32, copy=False)
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

while STOP_GAME:

    event_pump()

    keys = get_key()
    for event in event_get():

        if keys[pygame.K_ESCAPE]:
            STOP_GAME = False
            break

        if event.type == pygame.MOUSEMOTION:
            MOUSE_POS = event.pos

    lit_surface, sw, sh = area24_gpu(
        MOUSE_POS[0], MOUSE_POS[1], background_rgb, lalpha, intensity=3.0, color=c)

    if sw < lw and MOUSE_POS[0] <= lw - lw2:
        xx = 0
    else:
        xx = MOUSE_POS[0] - lw2

    if sh < lh and MOUSE_POS[1] <= lh - lh2:
        yy = 0
    else:
        yy = MOUSE_POS[1] - lh2

    SCREEN.fill((0, 0, 0))
    SCREEN.blit(background, (0, 0))

    SCREEN.blit(lit_surface, (xx, yy), special_flags=pygame.BLEND_RGBA_ADD)
    t = clock.get_fps()
    avg.append(t)
    show_fps(SCREEN, t, avg)

    clock.tick()
    FRAME += 1

    # pygame.display.set_caption(
    #     "Demo light GPU %s fps"
    #     "(%sx%s)" % (round(clock.get_fps(), 2), width, height))
    flip()

pygame.quit()