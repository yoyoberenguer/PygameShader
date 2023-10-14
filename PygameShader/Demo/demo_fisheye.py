"""
PygameShader DEMO fisheye
"""

try:
    from PygameShader.shader import *
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
    from pygame import Surface, RLEACCEL, QUIT, K_SPACE, BLEND_RGB_ADD, BLEND_RGB_MIN

except ImportError:
    raise ImportError("\n<Pygame> library is missing on your system."
          "\nTry: \n   C:\\pip install pygame on a window command prompt.")

try:
    from PygameShader.misc import *
except ImportError:
    raise ImportError("\n<misc> library is missing on your system.")

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
        screen_.blit(avg_text, (250, 0))
    if len(avg_) > 200:
        avg_ = avg_[200:]
    return avg_


width = 800
height = 800

SCREENRECT = pygame.Rect(0, 0, width, height)
SCREEN = pygame.display.set_mode(
    SCREENRECT.size, pygame.SCALED, 32)

try:
    BCK = pygame.image.load("../Assets/space1.jpg").convert()
except FileNotFoundError:
    raise FileNotFoundError(
        '\nImage file space1.jpg is missing from the Assets directory.')

BCK = pygame.transform.smoothscale(BCK, (width, height))


pygame.init()

clock = pygame.time.Clock()
FRAME = 0
STOP_GAME = True

event_pump = pygame.event.pump
event_get = pygame.event.get
get_key = pygame.key.get_pressed


f_model = fisheye_footprint(800, 800, 400, 400)
SCREEN.blit(BCK, (0, 0))

while STOP_GAME:

    event_pump()

    keys = get_key()
    for event in event_get():

        if keys[pygame.K_ESCAPE]:
            STOP_GAME = False
            break

    scroll24_inplace(BCK, 2, 2)
    fisheye(SCREEN, f_model)

    clock.tick(500)
    t = clock.get_fps()

    FRAME += 1

    pygame.display.flip()
    pygame.display.set_caption(
        "Demo fisheye effect %s fps "
        "(%sx%s)" % (round(t, 2), width, height))

    SCREEN.blit(BCK, (0, 0))

pygame.quit()