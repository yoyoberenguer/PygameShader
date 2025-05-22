"""
PygameShader DEMO MAGNIFYING GLASS
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

    fps = str(f"fps:{fps_:.3f} move your mouse")
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
    SCREENRECT.size, 32)

try:
    BCK = pygame.image.load("../Assets/img.jpg").convert()
except FileNotFoundError:
    raise FileNotFoundError(
        '\nImage file img.png is missing from the Assets directory.')

BCK = pygame.transform.smoothscale(BCK, (width, height))


pygame.init()

clock = pygame.time.Clock()
FRAME = 0
STOP_GAME = True

event_pump = pygame.event.pump
event_get = pygame.event.get
get_key = pygame.key.get_pressed

tmp = numpy.ndarray((400, 400, 2), dtype=numpy.uint32, order='C')
fisheye_footprint_param(tmp, 200, 200, 1., .6)
surface_mag = pygame.Surface((400, 400))

MOUSE_POS = pygame.math.Vector2()
MOUSE_POS.x = 0
MOUSE_POS.y = 0

get_pos = pygame.mouse.get_pos

xr = 0.3
xz = 1.0
centre_x = 200
centre_y = 200
avg = []

while STOP_GAME:

    event_pump()

    keys = get_key()
    for event in event_get():

        if keys[pygame.K_ESCAPE]:
            STOP_GAME = False
            break

        elif keys[pygame.K_KP_PLUS]:
            xr += 0.01

        elif keys[pygame.K_KP_MINUS]:
            xr -= 0.01

        elif keys[pygame.K_q]:
            xz -= 0.01

        elif keys[pygame.K_a]:
            xz += 0.01

        elif keys[pygame.K_w]:
            centre_x -= 1

        elif keys[pygame.K_e]:
            centre_x += 1

        if event.type == pygame.MOUSEMOTION:
            MOUSE_POS = Vector2(event.pos)
            if MOUSE_POS.x < 0: MOUSE_POS.x = 0
            if MOUSE_POS.x > width: MOUSE_POS.x = width
            if MOUSE_POS.y < 0: MOUSE_POS.y = 0
            if MOUSE_POS.y > height: MOUSE_POS.y = height

    SCREEN.blit(BCK, (0, 0))
    fisheye_footprint_param(tmp, centre_x, centre_y, xz, xr)
    # fisheye_footprint(tmp, 200, 200)
    surface_mag.blit(BCK, (0, 0), (MOUSE_POS.x-200, MOUSE_POS.y-200, 400, 400))
    fisheye(surface_mag, tmp)
    SCREEN.blit(surface_mag, (MOUSE_POS.x-200, MOUSE_POS.y-200))

    clock.tick(500)
    t = clock.get_fps()

    FRAME += 1

    avg.append(t)
    avg = show_fps(SCREEN, t, avg)

    pygame.display.flip()
    pygame.display.set_caption(
        "Demo magnifying effect %s fps; "
        "(%sx%s) focal length %s" % (round(t, 2), width, height, round(xr, 2)))


pygame.quit()