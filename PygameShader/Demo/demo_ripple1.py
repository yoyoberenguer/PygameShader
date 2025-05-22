"""
PygameShader RIPPLE DEMO
"""

from pygame.surfarray import make_surface, pixels3d, array3d
from pygame.transform import smoothscale, scale2x
from pygame.image import frombuffer

from PygameShader.shader import ripple, ripple_seabed
from PygameShader.misc import _randi, _randf

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

pygame.font.init()
font = pygame.font.SysFont("Arial", 15)


def show_fps(screen_, fps_, avg_) -> list:

    """ Show framerate in upper left corner """

    fps = str(f"Move your mouse - fps:{fps_:.3f}")
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


width = 600
height = 600

SCREENRECT = pygame.Rect(0, 0, width, height)
SCREEN = pygame.display.set_mode(SCREENRECT.size, pygame.SCALED, 32)
SCREEN.set_alpha(None)

pygame.init()

clock = pygame.time.Clock()
FRAME = 0
STOP_GAME = False


try:
    texture = pygame.image.load('../Assets/background2.jpg').convert()
except FileNotFoundError:
    raise FileNotFoundError(
        '\nImage file background2.jpg is missing from the Assets directory.')

texture = pygame.transform.smoothscale(texture, (width, height))
texture.set_colorkey((0, 0, 0, 0), pygame.RLEACCEL)
texture.set_alpha(None)


current = numpy.zeros((width, height), dtype=numpy.float32).copy('C')
previous = numpy.zeros((width, height), dtype=numpy.float32).copy('C')

back_array = pixels3d(SCREEN)

# TWEAKS
cget_fps = clock.get_fps
event_pump = pygame.event.pump
event_get = pygame.event.get
get_key = pygame.key.get_pressed
get_pos = pygame.mouse.get_pos
flip = pygame.display.flip


avg = []
STOP_GAME = True

try:
    WaterDrop1 = pygame.mixer.Sound("..//Assets//ES_WaterDrip1.wav")
except FileNotFoundError:
    raise FileNotFoundError(
        '\nSound file ES_WaterDrip1.wav is missing from the Assets directory.')

try:
    WaterDrop2 = pygame.mixer.Sound("..//Assets//ES_WaterDrip2.wav")
except FileNotFoundError:
    raise FileNotFoundError(
        '\nSound file ES_WaterDrip2.wav is missing from the Assets directory.')

try:
    WaterDrop3 = pygame.mixer.Sound("..//Assets//ES_WaterDrip3.wav")
except FileNotFoundError:
    raise FileNotFoundError(
        '\nSound file ES_WaterDrip2.wav is missing from the Assets directory.')
WaterDrops = [WaterDrop1, WaterDrop2, WaterDrop3]

mouse_pos = pygame.math.Vector2()
mouse_pos.x = 0
mouse_pos.y = 0

p_2_14 = 8000
p_2_10 = 1000
p_2_15 = 5000


surf = pygame.Surface((width, height)).convert()

while STOP_GAME:

    pygame.display.set_caption(
        "DEMO ripple effect CPU : FPS %s " % round(cget_fps(), 2))

    event_pump()

    keys = get_key()
    for event in event_get():

        if keys[pygame.K_ESCAPE]:
            STOP_GAME = False
            break

        if event.type == pygame.MOUSEMOTION:
            mouse_pos = pygame.math.Vector2(get_pos())

            previous[max(int(mouse_pos.x), 0),
                     max(int(mouse_pos.y), 0)] = p_2_14

    if _randi(0, 1000) > 980:
        drip_sound = WaterDrops[_randi(0, 2)]
        drip_sound.set_volume(_randf(0.1, 0.8))
        drip_sound.play()
        previous[_randi(0, width - 2), _randi(0, height - 2)] = _randi(p_2_10, p_2_15)

    previous, current = \
        ripple(height, width, previous, current, back_array, dispersion_=0.0051)

    clock.tick(500)
    t = clock.get_fps()

    # avg.append(t)
    # avg = show_fps(SCREEN, t, avg)
    FRAME += 1

    flip()
    pygame.display.set_caption(
        "Test ripple effect %s fps "
        "(%sx%s)" % (round(t, 2), width, height))

pygame.quit()