"""
PygameShader BURST IMAGE DEMO
"""

import sys

try:
    from PygameShader.shader import *
except ImportError:
    raise ImportError("\n<PygameShader> library is missing on your system."
                      "\nTry: \n   C:\\pip install PygameShader on a window command prompt.")

try:
    from PygameShader.misc import *
except ImportError:
    raise ImportError("\n<misc> library is missing on your system.")


from PygameShader.BurstSurface import *

try:
    import numpy
    from numpy import uint8
except ImportError:
    raise ImportError("\n<numpy> library is missing on your system."
                      "\nTry: \n   C:\\pip install numpy on a window command prompt.")

numpy.set_printoptions(threshold=sys.maxsize)

# PYGAME IS REQUIRED
try:
    import pygame
    from pygame import Surface, RLEACCEL, QUIT, K_SPACE, BLEND_RGB_ADD, \
        DOUBLEBUF, FULLSCREEN, BLEND_RGB_SUB, \
        BLEND_RGB_MULT

except ImportError:
    raise ImportError("\n<Pygame> library is missing on your system."
                      "\nTry: \n   C:\\pip install pygame on a window command prompt.")

pygame.init()

pygame.font.init()
font = pygame.font.SysFont("Arial", 15)

# Set the display to 1024 x 768
WIDTH = 800
HEIGHT = 800
pygame.init()
SCREENRECT = pygame.Rect(0, 0, WIDTH, HEIGHT)
from PygameShader.shader import *


try:
    IMAGE = pygame.image.load("../Assets/city.jpg")
except FileNotFoundError:
    raise FileNotFoundError(
        '\nImage file city.jpg is missing from the Assets directory.')

IMAGE = smoothscale(IMAGE, (WIDTH, HEIGHT))


clock = pygame.time.Clock()

avg = []
angle = 0
FRAME = 0
GAME = True
event_get = pygame.event.get
get_key = pygame.key.get_pressed
get_pos = pygame.mouse.get_pos

from pygame._sdl2.video import Window, Texture, Renderer, Image

screen = Window(title="pygame", size=(800, 800), position=(400, 200), fullscreen=False,
                fullscreen_desktop=False)

pygame.time.wait(500)

rend = Renderer(screen, index=-1, accelerated=0, vsync=False, target_texture=False)
sg = pygame.sprite.Group()

burst_experimental(
    render_=rend,
    image_=IMAGE,
    group_=sg,
    block_size_=8,
    rows_=100,
    columns_=100,
    x_=0,
    y_=0,
    max_angle_=-5
)

#
# burst_into_memory(
#         n_=1,
#         sg_=sg,
#         screenrect=SCREENRECT,
#         warn_ = False,
#         auto_n_ = True)


def show_fps(render_, fps_, avg_) -> list:
    """ Show framerate in upper left corner """

    fps = str(f"CPU fps:{fps_:.3f}")
    av = sum(avg_)/len(avg_) if len(avg_) > 0 else 0

    fps_text = font.render(fps, 1, pygame.Color("beige"))
    fps_surf = Image(Texture.from_surface(render_, fps_text))
    render_.blit(fps_surf, fps_surf.get_rect(), special_flags=0)
    if av != 0:
        av = str(f"avg:{av:.3f}")
        avg_text = font.render(av, 1, pygame.Color("beige"))
        fps_surf = Image(Texture.from_surface(render_, avg_text))
        rect = fps_surf.get_rect()
        rect[0] += 150
        render_.blit(fps_surf, rect, special_flags=0)
    if len(avg_) > 10:
        avg_ = avg_[10:]

    return avg_


while GAME:

    pygame.event.pump()

    keys = get_key()
    for event in event_get():

        if keys[pygame.K_ESCAPE]:
            GAME = False
            break

        if event.type == pygame.MOUSEMOTION:
            mouse_pos = pygame.math.Vector2(get_pos())

    rend.clear()

    # db_experimental(SCREENRECT, rend, sg)

    rff_experimental(
        render_ = rend,
        screenrect_=SCREENRECT,
        current_frame_=FRAME,
        start_frame=200,
        group_=sg,
        blend_ = 0
    )

    # rfm_experimental(
    #     screenrect_=SCREENRECT,
    #     render_=rend,
    #     group_=sg,
    #     blend_=0)

    clock.tick(200)
    t = clock.get_fps()
    avg.append(t)
    avg = show_fps(rend, t, avg)

    rend.present()

    FRAME += 1


pygame.quit()


