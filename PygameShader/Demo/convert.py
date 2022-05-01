"""
PygameShader CONVERT DEMO
"""

import sys

try:
    from PygameShader import *
except ImportError:
    raise ImportError("\n<PygameShader> library is missing on your system."
                      "\nTry: \n   C:\\pip install PygameShader on a window command prompt.")

try:
    from misc import *
except ImportError:
    raise ImportError("\n<misc> library is missing on your system.")

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
    from pygame import Surface, RLEACCEL, QUIT, K_SPACE, BLEND_RGB_ADD

except ImportError:
    raise ImportError("\n<Pygame> library is missing on your system."
                      "\nTry: \n   C:\\pip install pygame on a window command prompt.")

# Set the display to 1024 x 768
WIDTH = 256
HEIGHT = 256
SCREEN = pygame.display.set_mode((WIDTH, HEIGHT), vsync=True)
SCREEN.convert(32, RLEACCEL)
SCREEN.set_alpha(None)
SCREEN.fill((0, 0, 0, 0))

from PygameShader.shader import *
from PygameShader.Palette import *

surf = horizontal_grad3d(256, 256, (12, 25, 210), (200, 198, 0))
SCREEN.blit(surf, (0, 0))
pygame.display.flip()

surf = spectrum(256, 256, gamma=1.8)
SCREEN.blit(surf, (0, 0))
pygame.display.flip()
pygame.image.save(SCREEN, "../Assets/SPECTRUM.png")

surf = spectrum(256, 256, gamma=0.25)
SCREEN.blit(surf, (0, 0))
pygame.display.flip()
pygame.image.save(SCREEN, "../Assets/SPECTRUM0.png")

try:
    import matplotlib.pyplot as plt
except ImportError:
    raise ImportError("\n<matplotlib> library is missing on your system."
                      "\nTry: \n   C:\\pip install matplotlib on a window command prompt.")


def load_image(im="aliens.png"):
    image = pygame.image.load("../Assets/" + im).convert()
    image = pygame.transform.smoothscale(image, (WIDTH, HEIGHT))
    return image


BCK = pygame.image.load("../Assets/background.jpg").convert()
BCK = pygame.transform.smoothscale(BCK, (WIDTH, HEIGHT))
#
# IMAGE = load_image()
# rgb_to_bgr(IMAGE)
# pygame.image.save(IMAGE, "../Assets/Aliens_BGR.jpg")
#
# IMAGE = load_image()
# rgb_to_brg(IMAGE)
# pygame.image.save(IMAGE, "../Assets/Aliens_BRG.jpg")
#
# IMAGE = load_image()
# greyscale(IMAGE)
# pygame.image.save(IMAGE, "../Assets/Aliens_GRAY.jpg")
#
# IMAGE = load_image()
# sepia(IMAGE)
# pygame.image.save(IMAGE, "../Assets/Aliens_SEPIA.jpg")
#
# IMAGE = load_image()
# median(IMAGE)
# pygame.image.save(IMAGE, "../Assets/Aliens_MEDIA_F.jpg")
#
# IMAGE = load_image()
# median(IMAGE, kernel_size_=2,
#        fast_=False,
#        reduce_factor_=1)
# pygame.image.save(IMAGE, "../Assets/Aliens_MEDIA_F2.jpg")
#
# IMAGE = load_image()
# median(IMAGE, kernel_size_=4,
#        fast_=False,
#        reduce_factor_=1)
# pygame.image.save(IMAGE, "../Assets/Aliens_MEDIA_F4.jpg")
#
# IMAGE = load_image()
# median(IMAGE, kernel_size_=8,
#        fast_=False,
#        reduce_factor_=1)
# pygame.image.save(IMAGE, "../Assets/Aliens_MEDIA_F8.jpg")
#
# IMAGE = load_image()
# median_grayscale(IMAGE)
# pygame.image.save(IMAGE, "../Assets/Aliens_MEDIA_G.jpg")
#
# IMAGE = load_image()
# median_grayscale(IMAGE, kernel_size_=8)
# pygame.image.save(IMAGE, "../Assets/Aliens_MEDIA_G8.jpg")
#
# IMAGE = load_image()
# median_avg(IMAGE, kernel_size_=8)
# pygame.image.save(IMAGE, "../Assets/Aliens_MEDIA_AVG8.jpg")
#
# IMAGE = load_image("background.jpg")
# # IMAGE = load_image(im="cat.png")
# color_reduction(IMAGE, color_=4)
# pygame.image.save(IMAGE, "../Assets/bck_COLOR.jpg")
#
# IMAGE = load_image("background.jpg")
# # IMAGE = load_image(im="cat.png")
# color_reduction(IMAGE, color_=16)
# pygame.image.save(IMAGE, "../Assets/bck_COLOR1.jpg")
#
# IMAGE = load_image()
# sobel(IMAGE)
# pygame.image.save(IMAGE, "../Assets/Aliens_SOBEL.jpg")
#
# IMAGE = load_image()
# sobel(IMAGE, threshold_=1)
# pygame.image.save(IMAGE, "../Assets/Aliens_SOBEL1.jpg")
#
# IMAGE = load_image()
# sobel_fast(IMAGE, factor_=1)
# pygame.image.save(IMAGE, "../Assets/Aliens_SOBEL_F1.jpg")
#
# IMAGE = load_image()
# sobel_fast(IMAGE, factor_=2)
# pygame.image.save(IMAGE, "../Assets/Aliens_SOBEL_F2.jpg")
#
# IMAGE = load_image()
# invert(IMAGE)
# pygame.image.save(IMAGE, "../Assets/Aliens_INVERT.jpg")
#
# IMAGE = load_image()
# hsl_effect(IMAGE, 0.5)
# pygame.image.save(IMAGE, "../Assets/Aliens_HSL_1.jpg")
#
# IMAGE = load_image()
# hsl_effect(IMAGE, -0.2)
# pygame.image.save(IMAGE, "../Assets/Aliens_HSL_2.jpg")
#
# IMAGE = load_image()
# blur(IMAGE, t_=4)
# pygame.image.save(IMAGE, "../Assets/Aliens_BLUR.jpg")
#
# IMAGE = load_image()
# blur(IMAGE, t_=8)
# pygame.image.save(IMAGE, "../Assets/Aliens_BLUR_2.jpg")
#
# IMAGE = load_image()
# swirl2(IMAGE, 25)
# pygame.image.save(IMAGE, "../Assets/Aliens_SWIRL25.jpg")
#
# IMAGE = load_image()
# swirl2(IMAGE, 90)
# pygame.image.save(IMAGE, "../Assets/Aliens_SWIRL90.jpg")
#
# IMAGE = load_image()
# plasma_config(IMAGE, 50, 0.5, 0.12, 0.38, 0.1)
# pygame.image.save(IMAGE, "../Assets/Aliens_plasma_config.jpg")
#
# IMAGE = load_image()
# plasma_config(IMAGE, 50, 0.5, 0.12, 0.38, 0.1, 0.5, 0.8)
# pygame.image.save(IMAGE, "../Assets/Aliens_plasma_config1.jpg")
#
# IMAGE = load_image()
# brightness(IMAGE, -0.5)
# pygame.image.save(IMAGE, "../Assets/Aliens_BRIGHT1.jpg")
#
# IMAGE = load_image()
# brightness(IMAGE, 0.4)
# pygame.image.save(IMAGE, "../Assets/Aliens_BRIGHT2.jpg")
#
# IMAGE = load_image()
# brightness_exclude(IMAGE, 0.5, color_=(0, 0, 0))
# pygame.image.save(IMAGE, "../Assets/Aliens_BRIGHT_EXLUDE.jpg")
#
# IMAGE = load_image()
# brightness_bpf(IMAGE, 0.5, bpf_threshold=10)
# pygame.image.save(IMAGE, "../Assets/Aliens_BRIGHT_PBF.jpg")
#
# IMAGE = load_image()
# saturation(IMAGE, -0.1)
# pygame.image.save(IMAGE, "../Assets/Aliens_SAT0.jpg")
#
# IMAGE = load_image()
# saturation(IMAGE, 0.5)
# pygame.image.save(IMAGE, "../Assets/Aliens_SAT.jpg")
#
# IMAGE = load_image()
# bpf(IMAGE, 64)
# pygame.image.save(IMAGE, "../Assets/Aliens_BPF.jpg")
#
# IMAGE = load_image()
# bpf(IMAGE, 200)
# pygame.image.save(IMAGE, "../Assets/Aliens_BPF1.jpg")
#
# IMAGE = load_image()
# bloom(IMAGE, 128)
# pygame.image.save(IMAGE, "../Assets/Aliens_BLOOM.jpg")
#
# IMAGE = load_image()
# bloom(IMAGE, 1)
# pygame.image.save(IMAGE, "../Assets/Aliens_BLOOM1.jpg")
#
# IMAGE = load_image()
# tv_scan(IMAGE)
# pygame.image.save(IMAGE, "../Assets/Aliens_TVSCAN.jpg")
#
# IMAGE = load_image()
# tv_scan(IMAGE, 20)
# pygame.image.save(IMAGE, "../Assets/Aliens_TVSCAN1.jpg")
#
# IMAGE = load_image()
# rgb_split(IMAGE, 16)
# pygame.image.save(IMAGE, "../Assets/Aliens_SPLIT_RGB.jpg")
#
# IMAGE = load_image()
# IMAGE = rgb_split_clean(IMAGE, 2)
# pygame.image.save(IMAGE, "../Assets/Aliens_SPLIT_RGB_CLEAN.jpg")
#
# IMAGE = load_image()
# IMAGE = rgb_split_clean(IMAGE, 8)
# pygame.image.save(IMAGE, "../Assets/Aliens_SPLIT_RGB_CLEAN1.jpg")
#
# IMAGE = load_image()
# heatmap(IMAGE)
# pygame.image.save(IMAGE, "../Assets/Aliens_HEATMAP.jpg")
#
# IMAGE = load_image()
# heatmap(IMAGE, False)
# pygame.image.save(IMAGE, "../Assets/Aliens_HEATMAP_INV.jpg")
#
# IMAGE = load_image()
# predator_vision(IMAGE,
#                 sobel_threshold=1,
#                 bpf_threshold=1,
#                 bloom_threshold=20,
#                 inv_colormap=True,
#                 blend=pygame.BLEND_RGBA_ADD)
# pygame.image.save(IMAGE, "../Assets/Aliens_PREDATOR_ADD.jpg")
#
# IMAGE = load_image()
# predator_vision(IMAGE,
#                 sobel_threshold=1,
#                 bpf_threshold=1,
#                 bloom_threshold=20,
#                 inv_colormap=True,
#                 blend=pygame.BLEND_RGBA_MIN)
# pygame.image.save(IMAGE, "../Assets/Aliens_PREDATOR_MIN.jpg")
#
# IMAGE = load_image()
# predator_vision(IMAGE,
#                 sobel_threshold=1,
#                 bpf_threshold=1,
#                 bloom_threshold=20,
#                 inv_colormap=True,
#                 blend=pygame.BLEND_RGBA_MAX)
# pygame.image.save(IMAGE, "../Assets/Aliens_PREDATOR_MAX.jpg")
#
# IMAGE = load_image()
# predator_vision(IMAGE,
#                 sobel_threshold=1,
#                 bpf_threshold=1,
#                 bloom_threshold=20,
#                 inv_colormap=True,
#                 blend=pygame.BLEND_RGBA_SUB)
# pygame.image.save(IMAGE, "../Assets/Aliens_PREDATOR_SUB.jpg")
#
# IMAGE = load_image()
# predator_vision(IMAGE,
#                 sobel_threshold=1,
#                 bpf_threshold=1,
#                 bloom_threshold=20,
#                 inv_colormap=True,
#                 blend=pygame.BLEND_RGBA_MULT)
# pygame.image.save(IMAGE, "../Assets/Aliens_PREDATOR_MULT.jpg")
#
# IMAGE = load_image()
# predator_vision(IMAGE, inv_colormap=True)
# pygame.image.save(IMAGE, "../Assets/Aliens_PREDATOR_INV.jpg")
#
# IMAGE = load_image()
# mirroring(IMAGE)
# pygame.image.save(IMAGE, "../Assets/Aliens_MIRROR.jpg")
#
# IMAGE = load_image()
# sharpen(IMAGE)
# pygame.image.save(IMAGE, "../Assets/Aliens_SHARP.jpg")
#
# IMAGE = load_image()
# IMAGE = blend(IMAGE, BCK, 50.0)
# pygame.image.save(IMAGE, "../Assets/Aliens_BLEND.jpg")
#
# IMAGE = load_image()
# IMAGE = cartoon(
#     IMAGE,
#     sobel_threshold_=32,
#     median_kernel_=4,
#     color_=32,
#     flag_=BLEND_RGB_ADD
# )
# pygame.image.save(IMAGE, "../Assets/Aliens_CARTOON_ADD.jpg")
#
# IMAGE = load_image()
# IMAGE = cartoon(
#     IMAGE,
#     sobel_threshold_=32,
#     median_kernel_=4,
#     color_=32,
#     flag_=pygame.BLEND_RGB_SUB
# )
# pygame.image.save(IMAGE, "../Assets/Aliens_CARTOON_SUB.jpg")
#
# IMAGE = load_image()
# IMAGE = cartoon(
#     IMAGE,
#     sobel_threshold_=32,
#     median_kernel_=4,
#     color_=32,
#     flag_=pygame.BLEND_RGB_MULT
# )
# pygame.image.save(IMAGE, "../Assets/Aliens_CARTOON_MULT.jpg")
#
# IMAGE = load_image()
# IMAGE = cartoon(
#     IMAGE,
#     sobel_threshold_=32,
#     median_kernel_=4,
#     color_=32,
#     flag_=pygame.BLEND_RGB_MIN
# )
# pygame.image.save(IMAGE, "../Assets/Aliens_CARTOON_MIN.jpg")
#
# IMAGE = load_image()
# IMAGE = cartoon(
#     IMAGE,
#     sobel_threshold_=32,
#     median_kernel_=4,
#     color_=32,
#     flag_=pygame.BLEND_RGB_MAX
# )
# pygame.image.save(IMAGE, "../Assets/Aliens_CARTOON_MAX.jpg")
#
# IMAGE = load_image()
# bluescale(IMAGE)
# pygame.image.save(IMAGE, "../Assets/Aliens_BLUESCALE.jpg")
#
# IMAGE = load_image()
# redscale(IMAGE)
# pygame.image.save(IMAGE, "../Assets/Aliens_REDSCALE.jpg")
#
# IMAGE = load_image()
#
# dirt_lens_image = [
#     pygame.image.load("../Assets/Bokeh__Lens_Dirt_9.jpg").convert(),
#     pygame.image.load("../Assets/Bokeh__Lens_Dirt_38.jpg").convert(),
#     pygame.image.load("../Assets/Bokeh__Lens_Dirt_46.jpg").convert(),
#     pygame.image.load("../Assets/Bokeh__Lens_Dirt_50.jpg").convert(),
#     pygame.image.load("../Assets/Bokeh__Lens_Dirt_54.jpg").convert(),
#     pygame.image.load("../Assets/Bokeh__Lens_Dirt_67.jpg").convert()
# ]
#
# lens = dirt_lens_image[ 1 ]
# lens = pygame.transform.scale(lens, (WIDTH, HEIGHT))
# import timeit
#
# dirt_lens(IMAGE, flag_=BLEND_RGB_ADD, lens_model_=lens, light_=0.2)
# pygame.image.save(IMAGE, "../Assets/Aliens_dirt_lens.jpg")
# t = timeit.timeit("dirt_lens(IMAGE, lens_model_=lens, flag_=BLEND_RGB_ADD,  light_=-0.2)",
#                   "from __main__ import dirt_lens, IMAGE, BLEND_RGB_ADD, lens", number=100)
# print("DIRT LENS :", t / 100)
#
# IMAGE = load_image()
# fisheye_model = fisheye_footprint(WIDTH + 1, HEIGHT + 1)
# fisheye(IMAGE, fisheye_model)
# pygame.image.save(IMAGE, "../Assets/Aliens_FISHEYE.jpg")
#
# IMAGE = load_image()
# blood_surface = pygame.image.load("../Assets/redvignette.png").convert_alpha()
# blood_surface = pygame.transform.smoothscale(blood_surface, (WIDTH, HEIGHT))
# BLOOD_MASK = numpy.asarray(pygame.surfarray.pixels_alpha(blood_surface) / 255.0, numpy.float32)
# PERCENTAGE = 0.5
# blood(IMAGE, BLOOD_MASK, PERCENTAGE)
# pygame.image.save(IMAGE, "../Assets/Aliens_BLOOD.jpg")
#
# IMAGE = load_image()
# PERCENTAGE = 0.9
# blood(IMAGE, BLOOD_MASK, PERCENTAGE)
# pygame.image.save(IMAGE, "../Assets/Aliens_BLOOD1.jpg")

IMAGE = load_image(im="cat.png")
# greyscale(IMAGE)
IMAGE = dithering(IMAGE, factor_=4)
pygame.image.save(IMAGE, "../Assets/cat_DITHERING.jpg")
IMAGE = load_image(im="cat.png")
# greyscale(IMAGE)
IMAGE = dithering(IMAGE, factor_=2)
pygame.image.save(IMAGE, "../Assets/cat_DITHERING2.jpg")

import timeit
import time


#
# IMAGE = load_image(im="cat.png")
# new_IMAGE = pixelation(IMAGE)
# pygame.image.save(new_IMAGE,  "../Assets/Aliens_PIXELATE.jpg")

IMAGE = load_image(im="parrot.png")
palette_change(IMAGE, PICO_8_CUSTOM)
pygame.image.save(IMAGE, "../Assets/Aliens_PICO_8_CUSTOM.png")



#
# IMAGE = load_image(im="parrot.jpg").convert()
# palette_change(pixels3d(IMAGE), COMMODORE_64)
# pygame.image.save(IMAGE, "../Assets/Aliens_APPLE.jpg")
#
#
# IMAGE = load_image(im="cat.png")
# greyscale(IMAGE)
# IMAGE = emboss(IMAGE, BLEND_RGB_MAX)
# pygame.image.save(IMAGE, "../Assets/Aliens_EMBOSS.jpg")
#

IMAGE = load_image(im="cat.png")
# greyscale(IMAGE)
dithering_int(IMAGE, factor_=8)
pygame.image.save(IMAGE, "../Assets/cat_DITHERING_INT.jpg")
t = timeit.timeit("dithering_int(IMAGE, factor_=4)",
                  "from __main__ import dithering_int, IMAGE", number=100)
print("DITHERING INT : ", t / 100)
#
# IMAGE = load_image(im="cat.png").convert()
# convert_27colors(IMAGE)
# pygame.image.save(IMAGE, "../Assets/Aliens_CPC64.png")
#
# IMAGE = load_image(im="cat.png").convert()
# cpc464(IMAGE)
# pygame.image.save(IMAGE, "../Assets/Aliens_CPC641.png")
# t = timeit.timeit("cpc464(IMAGE)",
#                   "from __main__ import cpc464, IMAGE, pixels3d", number=100)
# print("CPC464 : ", t / 100)
#
# IMAGE = load_image(im="cat.png").convert()
# IMAGE = bilateral(IMAGE, 2.5, 80)
# pygame.image.save(IMAGE, "../Assets/Aliens_BILATERAL.png")
# t = timeit.timeit("bilateral(IMAGE, 2, 80)",
#                   "from __main__ import bilateral, IMAGE", number=100)
# print("BILATERAL FILTER : ", t / 100)
