"""
PygameShader CPU ZOOM DEMO
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
    from pygame.surfarray import pixels3d
    from pygame.transform import scale
    from pygame.math import Vector2

except ImportError:
    raise ImportError("\n<Pygame> library is missing on your system."
          "\nTry: \n   C:\\pip install pygame on a window command prompt.")

# try:
#     import cupy
# except ImportError:
#     raise ImportError("\n<Pygame> library is missing on your system."
#                       "\nTry: \n   C:\\pip install cupy on a window command prompt.")

try:
    import PygameShader
    from PygameShader.shader import zoom
except ImportError:
    raise ImportError("\n<PygameShader> library is missing on your system."
                      "\nTry: \n   C:\\pip install PygameShader on a window command prompt.")


def show_fps(screen_, fps_, avg_) -> None:
    """ Show framerate in upper left corner """
    font = pygame.font.SysFont("Arial", 15)
    fps = str(f"fps:{fps_:.3f}")
    av = sum(avg_)/len(avg_) if len(avg_) > 0 else 0

    fps_text = font.render(fps, 1, pygame.Color("coral"))
    screen_.blit(fps_text, (10, 0))
    if av != 0:
        av = str(f"avg:{av:.3f} MOVE YOUR MOUSE")
        avg_text = font.render(av, 1, pygame.Color("coral"))
        screen_.blit(avg_text, (100, 0))
    if len(avg_) > 200:
        avg_ = avg_[200:]


width = 800
height = 600

SCREENRECT = pygame.Rect(0, 0, width, height)
# pygame.display.init()
SCREEN = pygame.display.set_mode(SCREENRECT.size, pygame.FULLSCREEN | pygame.SCALED)
SCREEN.set_alpha(None)
pygame.init()

background = pygame.image.load('..//Assets//city.jpg')
background = pygame.transform.smoothscale(background, (width, height))
background.convert(32, RLEACCEL)
background.set_alpha(None)

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

MOUSE_POS = Vector2()
MOUSE_POS.x = 0
MOUSE_POS.y = 0

while STOP_GAME:

    event_pump()

    keys = get_key()
    for event in event_get():

        if keys[pygame.K_ESCAPE]:
            STOP_GAME = False
            break

        if event.type == pygame.MOUSEMOTION:
            MOUSE_POS = Vector2(event.pos)
            if MOUSE_POS.x < 0:MOUSE_POS.x = 0
            if MOUSE_POS.x > width:MOUSE_POS.x = width
            if MOUSE_POS.y < 0:MOUSE_POS.y = 0
            if MOUSE_POS.y > height:MOUSE_POS.y = height

    surf = zoom(background, MOUSE_POS.x, MOUSE_POS.y, 0.9999 - ((FRAME % 255) / 255.0))

    SCREEN.blit(surf, (0, 0))
    t = clock.get_fps()
    avg.append(t)
    show_fps(SCREEN, t, avg)
    pygame.display.flip()
    clock.tick()
    FRAME += 1

    pygame.display.set_caption(
        "Demo zoom CPU %s fps"
        "(%sx%s)" % (round(clock.get_fps(), 2), width, height))
    avg = avg[10:]

pygame.quit()