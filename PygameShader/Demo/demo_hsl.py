"""
PygameShader CPU HSL


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
    from PygameShader.shader import hsl_effect
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
        screen_.blit(avg_text, (100, 0))
    if len(avg_) > 200:
        avg_ = avg_[200:]
    return avg_


width = 800
height = 600

SCREENRECT = pygame.Rect(0, 0, width, height)
SCREEN = pygame.display.set_mode(SCREENRECT.size)

pygame.init()

background = pygame.image.load('..//Assets//Parrot.jpg')
background = pygame.transform.smoothscale(background, (width, height))
background.convert(32, RLEACCEL)
background.set_alpha(None)

bck_copy = background.copy()

FRAME = 0
clock = pygame.time.Clock()
avg = []
v = 0.001
hsl = 0
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

    hsl_effect(background, hsl)

    SCREEN.blit(background, (0, 0))
    t = clock.get_fps()
    avg.append(t)
    show_fps(SCREEN, t, avg)
    pygame.display.flip()
    clock.tick()
    FRAME += 1

    hsl += v
    if hsl > 1.0:
        hsl = 0.99
        v *= -1
    if hsl < 0.0:
        hsl = 0.01
        v *= -1

    pygame.display.set_caption(
        "Test hsl %s fps, value %s (%sx%s)" %
        (round(clock.get_fps(), 2), round(hsl, 2), width, height))

    background = bck_copy.copy()

pygame.quit()