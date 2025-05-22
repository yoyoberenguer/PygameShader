"""
PygameShader RIPPLE DEMO
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
        get_gpu_info, block_and_grid_info
except ImportError:
    raise ImportError("\n<PygameShader> library is missing on your system."
                      "\nTry: \n   C:\\pip install PygameShader on a window command prompt.")


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
        screen_.blit(avg_text, (200, 0))
    if len(avg_) > 200:
        avg_ = avg_[200:]
    return avg_


get_gpu_info()

width = 800
height = 600

SCREENRECT = pygame.Rect(0, 0, width, height)
SCREEN = pygame.display.set_mode(
    SCREENRECT.size, pygame.FULLSCREEN | pygame.DOUBLEBUF | pygame.SCALED, 32)

# from PygameShader.VideoRecording import capture_video, write_video

try:
    texture = pygame.image.load('../Assets/background2.jpg').convert()
except FileNotFoundError:
    raise FileNotFoundError(
        '\nImage file background2.jpg is missing from the Assets directory.')

texture = pygame.transform.smoothscale(texture, (width, height))
texture.set_colorkey((0, 0, 0, 0), pygame.RLEACCEL)
texture.set_alpha(None)

try:
    BACKGROUND = pygame.image.load("../Assets/background2.jpg").convert()
except FileNotFoundError:
    raise FileNotFoundError(
        '\nImage file background2.jpg is missing from the Assets directory.')

BACKGROUND = pygame.transform.smoothscale(BACKGROUND, (width, height))
BACKGROUND.set_colorkey((0, 0, 0, 0), pygame.RLEACCEL)
BACKGROUND_COPY = BACKGROUND.copy()

pygame.init()

clock = pygame.time.Clock()
FRAME = 0
STOP_GAME = False

current = cupy.empty((width, height), dtype=cupy.float32)
previous = cupy.empty((width, height), dtype=cupy.float32)
texture_array = cupy.asarray(pygame.surfarray.array3d(texture), dtype=cupy.uint8)
back_array = cupy.asarray(pygame.surfarray.pixels3d(BACKGROUND), dtype=cupy.uint8)

# TWEAKS
cget_fps = clock.get_fps
event_pump = pygame.event.pump
event_get = pygame.event.get
get_key = pygame.key.get_pressed
get_pos = pygame.mouse.get_pos
flip = pygame.display.flip

array = cupy.full((width, height, 3), 0, dtype=cupy.uint8)
grid, block = block_grid(width, height)
block_and_grid_info(width, height)

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
WaterDrops    = [WaterDrop1, WaterDrop2, WaterDrop3]

mouse_pos = pygame.math.Vector2()
mouse_pos.x = 0
mouse_pos.y = 0

while STOP_GAME:

    pygame.display.set_caption("DEMO ripple effect GPU : FPS %s " % round(cget_fps(), 2))

    event_pump()

    keys = get_key()
    for event in event_get():

        if keys[pygame.K_ESCAPE]:
            STOP_GAME = False
            break

        if event.type == pygame.MOUSEMOTION:
            mouse_pos = pygame.math.Vector2(get_pos())

            previous[int(mouse_pos.x % width), int(mouse_pos.y % height)] = 2 ** 14

    rnd = randint(0, 1000)

    if rnd > 990:
        drip_sound = WaterDrops[randrange(0, 2)]
        drip_sound.set_volume(uniform(0.1, 0.8))
        drip_sound.play()
        previous[randint(0, width - 2), randint(0, height - 2)] = randint(2 ** 10, 2**15)

    # SCREEN.fill((0, 0, 0))

    previous, current = ripple_effect_gpu(
        grid, block, width, height,
        current, previous, texture_array, back_array)
    surf = pygame.image.frombuffer(back_array.transpose(1, 0, 2).tobytes(), (width, height), "RGB")

    SCREEN.blit(surf, (0, 0), special_flags=0)

    clock.tick(120)
    t = clock.get_fps()
    avg.append(t)
    avg = show_fps(SCREEN, t, avg)
    FRAME += 1

    # capture_video(SCREEN, w, h, compression_=False)

    flip()

# write_video(SCREEN, SCREEN.get_rect(), fast_=False, fps_=120)
#
# from PygameShader.combine import montage
# montage("sound_recording.wav", "Video.avi")


pygame.quit()