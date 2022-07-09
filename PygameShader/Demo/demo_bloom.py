import pygame
from pygame import RLEACCEL
from PygameShader import shader_bloom_fast, shader_bloom_fast1


def show_fps(screen_, fps_, avg_) -> None:
    """ Show framerate in upper left corner """
    font = pygame.font.SysFont("Arial", 15)
    fps = str(f"CPU fps:{fps_:.3f}")
    av = sum(avg_)/len(avg_) if len(avg_) > 0 else 0

    fps_text = font.render(fps, 1, pygame.Color("beige"))
    screen_.blit(fps_text, (10, 0))
    if av != 0:
        av = str(f"avg:{av:.3f}")
        avg_text = font.render(av, 1, pygame.Color("beige"))
        screen_.blit(avg_text, (120, 0))
    if len(avg_) > 200:
        avg_ = avg_[200:]


WIDTH = 800
HEIGHT = 600
SCREEN = pygame.display.set_mode((WIDTH, HEIGHT), pygame.FULLSCREEN, vsync=True)
SCREEN.convert(32, RLEACCEL)
SCREEN.set_alpha(None)
pygame.init()

# from PygameShader.VideoRecording import capture_video, write_video


background = pygame.image.load("../Assets/city.jpg").convert()
background = pygame.transform.smoothscale(background, (WIDTH, HEIGHT))

image = background.copy()

FRAME = 0
CLOCK = pygame.time.Clock()
GAME = True
V = 0.5
BPF = 255
avg = []
while GAME:

    pygame.event.pump()
    for event in pygame.event.get():

        keys = pygame.key.get_pressed()

        if keys[pygame.K_ESCAPE]:
            GAME = False
            break

    # image = shader_bloom_fast(image, BPF, fast_=False, factor_=3)
    shader_bloom_fast1(image, threshold_=BPF, smooth_=0)

    SCREEN.blit(image, (0, 0))
    t = CLOCK.get_fps()
    avg.append(t)
    show_fps(SCREEN, t, avg)

    CLOCK.tick()
    FRAME += 1

    pygame.display.flip()

    # capture_video(SCREEN, WIDTH, HEIGHT, compression_=False)

    image = background.copy()

    if BPF >= 255.0:
        V *= -1
        BPF = 255.0
    elif BPF <= 0.0:
        V *= -1
        BPF = 0

    BPF += 4 * V
    BPF = min(BPF, 255)
    BPF = max(BPF, 0)
#
# write_video(SCREEN, SCREEN.get_rect(), fast_=False, fps_=120)
#
# from PygameShader.combine import montage
# montage("sound_recording.wav", "Video.avi")
pygame.quit()