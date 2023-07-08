"""
PygameShader CONVERT DEMO
"""
import math
import sys

import cupy
import random
try:
    from PygameShader.shader import *
except ImportError:
    raise ImportError("\n<PygameShader> library is missing on your system."
                      "\nTry: \n   C:\\pip install PygameShader on a window command prompt.")

try:
    from PygameShader.misc import *
except ImportError:
    raise ImportError("\n<misc> library is missing on your system.")

# from PygameShader.shader_gpu import invert_gpu, sepia_gpu, \
#     bpf_gpu, gaussian_5x5_gpu, sobel_gpu, canny_gpu, gaussian_3x3_gpu, median_gpu, median1_gpu, \
#     color_reduction_gpu, hsv_gpu, grayscale_gpu, grayscale_lum_gpu, bloom_gpu, \
#     prewitt_gpu, cartoon_gpu, invert_inplace_cupy, sepia_inplace_cupy, blending_gpu, sharpen_gpu, ripple_effect_gpu, \
#     sharpen1_gpu, block_grid, bpf1_gpu, mirroring_gpu, saturation_gpu, bilateral_gpu, emboss5x5_gpu, \
#     mult_downscale_gpu, get_max_grid_per_block, get_gpu_free_mem, get_gpu_maxmem, get_gpu_pci_bus_id, \
#     get_compute_capability, block_and_grid_info, get_gpu_info, hsl_gpu, brightness_gpu, \
#     fisheye_gpu, swirl_gpu, wave_gpu, rgb_split_gpu, zoom_gpu, chromatic_gpu, dithering_gpu, wavelength_map_gpu, \
#     heatmap_gpu, heatmap_gpu_inplace, predator_gpu, downscale_surface_gpu

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
    from pygame import Surface, RLEACCEL, QUIT, K_SPACE, BLEND_RGB_ADD, DOUBLEBUF, FULLSCREEN, BLEND_RGB_SUB, \
        BLEND_RGB_MULT

except ImportError:
    raise ImportError("\n<Pygame> library is missing on your system."
                      "\nTry: \n   C:\\pip install pygame on a window command prompt.")

# Set the display to 1024 x 768
WIDTH = 800
HEIGHT = 1280
SCREEN = pygame.display.set_mode((WIDTH, HEIGHT), DOUBLEBUF, vsync=False)
pygame.init()
#SCREEN.convert(32, RLEACCEL)
#SCREEN.set_alpha(None)
SCREEN.fill((0, 0, 0, 0))

from PygameShader.shader import *
from PygameShader.Palette import *
import cupy as cp

# surf = horizontal_grad3d(256, 256, (12, 25, 210), (200, 198, 0))
# SCREEN.blit(surf, (0, 0))
# pygame.display.flip()
#
# surf = spectrum(256, 256, gamma=1.8)
# SCREEN.blit(surf, (0, 0))
# pygame.display.flip()
# pygame.image.save(SCREEN, "../Assets/SPECTRUM.png")
#
# surf = spectrum(256, 256, gamma=0.25)
# SCREEN.blit(surf, (0, 0))
# pygame.display.flip()
# pygame.image.save(SCREEN, "../Assets/SPECTRUM0.png")

try:
    import matplotlib.pyplot as plt
except ImportError:
    raise ImportError("\n<matplotlib> library is missing on your system."
                      "\nTry: \n   C:\\pip install matplotlib on a window command prompt.")

import colorsys
from colorsys import *


def load_image(im="aliens.jpg"):
    image = pygame.image.load("../Assets/" + im).convert()
    image = pygame.transform.smoothscale(image, (WIDTH, HEIGHT))
    return image


import timeit
BCK = pygame.image.load("../Assets/Aliens.jpg").convert()
BCK = pygame.transform.smoothscale(BCK, (WIDTH, HEIGHT))

blood_surface = pygame.image.load("../Assets/redvignette.png").convert_alpha()
blood_surface = pygame.transform.smoothscale(blood_surface, (WIDTH, HEIGHT))
BLOOD_MASK = numpy.asarray(
    pygame.surfarray.pixels_alpha(blood_surface) / float(255.0), numpy.float32)


brightness_exclude(BCK, 0.1, (0, 0, 0))
SCREEN.blit(BCK, (0, 0))
pygame.display.flip()

t = timeit.timeit("brightness_exclude(BCK, 0.8, (0, 0, 0))",
                  "from __main__ import brightness_exclude, BCK", number=1000)
print("cython : ", t/1000)


IMAGE = load_image(im="Capture.png")
# palette_change(IMAGE, SILVER)
IMAGE = pygame.transform.smoothscale(IMAGE, (256, 256))
# pygame.image.save(IMAGE, "../Assets/Capture.jpg")



# IMAGE = pygame.image.load("C:\\Users\\yoyob\\Desktop\\jt00c0tnu60b1.webp")
#IMAGE = load_image("city.jpg")
#IMAGE = pygame.transform.smoothscale(IMAGE, (WIDTH, HEIGHT))
# Luma_GreyScale(IMAGE)
saturation(IMAGE, 2)
# pygame.image.save(IMAGE, "../Assets/CaptureY0.png")
grid, block = block_grid(IMAGE.get_width(), IMAGE.get_height())
# IMAGE = predator_gpu(
#     IMAGE,
#     grid, block,
# )
#
#
SCREEN.blit(IMAGE, (0, 0))
pygame.display.flip()
#
#
# while 1:
#     SCREEN.blit(IMAGE, (0, 0))
#     pygame.display.flip()
#
#     pass
#
t = timeit.timeit("dithering_inplace(IMAGE)", "from __main__ import dithering_inplace, IMAGE", number=1000)
print("cython : ", t/1000)
#
# t = timeit.timeit("dithering(IMAGE)", "from __main__ import dithering, IMAGE", number=100)
# print("cython : ", t/100)

# while 1:
#     SCREEN.blit(IMAGE, (0, 0))
#     pygame.display.flip()
#     pass



#
# BCK = pygame.transform.smoothscale(BCK, (2048, 2048))
# T = 200
# import timeit
# t = timeit.timeit(
#     "blur(BCK)",
#     "from __main__ import blur, BCK",
#     number=T)
# print("cython : ", t/T)


T = 1000
# ------ CYTHON ------------------------
# IMAGE = load_image("background.jpg").convert()
# sepia(IMAGE)
# pygame.image.save(IMAGE, "../Assets/TESTN.png")

# import timeit
# t = timeit.timeit(
#     "sepia(IMAGE)",
#     "from __main__ import sepia, IMAGE",
#     number=T)
# print("cython : ", t/T)
# ----------------------------------------
# # #
# # # # -------- NUMPY ------------------------
# IMAGE = load_image("background.jpg").convert()
# invert_surface_numpy(pixels3d(IMAGE))
# pygame.image.save(IMAGE, "../Assets/TESTCY.png")
#
# arr_ = pixels3d(IMAGE)
# t = timeit.timeit(
#     "invert_surface_numpy(arr_)",
#     "from __main__ import invert_surface_numpy, arr_",
#     number=T)
# print("numpy : ", t/T)
#
#
# # ----------------------------------------
#
# # --------- CUPY INPLACE -------------------------
# #
# T = 1000
# IMAGE = load_image("background.jpg").convert()
# invert_surface_cupy_inplace(pixels3d(IMAGE))
# IMAGE = pygame.transform.smoothscale(IMAGE, (1280, 1024))
# pygame.image.save(IMAGE, "../Assets/TESTCUPY_INPLACE.png")
#
# arr_= pixels3d(IMAGE)
# t = timeit.timeit(
#     "invert_surface_cupy_inplace(arr_)",
#     "from __main__ import invert_surface_cupy_inplace, arr_",
#     number=T)
# print("cupy inplace : ", t/T)
# # # ----------------------------------------


# # --------- CUPY  -------------------------
# T = 1000
#
# IMAGE = load_image("background.jpg").convert()
# IMAGE = pygame.transform.smoothscale(IMAGE, (1280, 1024))
# arr = cp.asarray(pixels3d(IMAGE))
# full_255 = cp.full((1280, 1024, 3), 255, dtype=uint8)
# # IMAGE = invert_gpu(arr, full_255)
# # pygame.image.save(IMAGE, "../Assets/TESTCUPY.png")
#
# t = timeit.timeit(
#     "invert_gpu(arr, full_255)",
#     "from __main__ import arr, invert_gpu, full_255",
#     number=T)
# print("cupy : ", t/T)
# # ----------------------------------------





def show_fps(screen_, fps_, avg_) -> None:
    """ Show framerate in upper left corner """
    font = pygame.font.SysFont("Arial", 15)
    fps = str(f"fps:{fps_:.3f}")
    av = sum(avg_)/len(avg_) if len(avg_) > 0 else 0

    fps_text = font.render(fps, 1, pygame.Color("coral"))
    screen_.blit(fps_text, (10, 0))
    if av != 0:
        av = str(f"avg:{av:.3f}")
        avg_text = font.render(av, 1, pygame.Color("coral"))
        screen_.blit(avg_text, (80, 0))
    if len(avg_) > 200:
        avg_ = avg_[200:]


IMAGE = load_image("city.jpg").convert_alpha()
IMAGE = smoothscale(IMAGE, (WIDTH, HEIGHT))
#
# t = timeit.timeit(
#     "sepia_gpu(IMAGE)",
#     "from __main__ import sepia_gpu, IMAGE",
#     number=T)
# print("SEPIA : ", t/T)
from scipy import misc
clock = pygame.time.Clock()

# greyscale(IMAGE)
IMAGE_COPY = IMAGE.copy()


# screen_arr = cp.asarray((pixels3d(SCREEN)))

# arr = pixels3d(IMAGE)


# bpf(IMAGE, 250)


#
# t = timeit.timeit("shader_blur5x5_numba(pixels3d(IMAGE))",
#     "from __main__ import shader_blur5x5_numba, pixels3d, IMAGE", number=100)
# print(t/ 100)
#
#
# t = timeit.timeit("blur(IMAGE)",
#     "from __main__ import blur, IMAGE", number=100)
# print(t/ 100)
#
# from gaussianBlur5x5 import blur5x5_array24_inplace_c
#
# t = timeit.timeit("blur5x5_array24_inplace_c(pixels3d(IMAGE))",
#     "from __main__ import blur5x5_array24_inplace_c, pixels3d, IMAGE", number=100)
# print(t/ 100)

# arr = cp.asarray(pixels3d(IMAGE))



# IMAGE = bilinear(IMAGE, 800, 800)

# print (timeit.timeit("bilinear(IMAGE, 800, 800)", "from __main__ import bilinear, IMAGE", number=1000)/1000)
# print (timeit.timeit("smoothscale(IMAGE, (800, 800))", "from __main__ import smoothscale, IMAGE", number=1000)/1000)

# IMAGE = make_surface(test(pixels3d(IMAGE)))
#
# arr_x2, arr_x4, arr_x8, arr_x16 = downscale_x2_gpu(pixels3d(IMAGE))
# print(arr_x2.shape, arr_x4.shape, arr_x8.shape, arr_x16.shape)
# IMAGE = make_surface(arr_x2)

# arr_x2 = upscale_x2_gpu(pixels3d(IMAGE))
# print(arr_x2.shape)
# IMAGE = make_surface(arr_x2)


#IMAGE, s2, s4, s8, s16 = bloom_gpu(IMAGE, False, threshold_=10)

thresh = 255
f = 1
arr = cupy.asarray(pixels3d(IMAGE))
avg = []
grid, block = block_grid(IMAGE.get_width(), IMAGE.get_height())
v = 0.01
s = 0.01
hsv = 0
sat = 0

# source = pygame.image.load("../Assets/Parrot_alpha.png").convert_alpha()
# source = smoothscale(source, (WIDTH, HEIGHT))
# destination = pygame.image.load("../Assets/background_alpha.png").convert_alpha()
# destination = smoothscale(destination, (WIDTH, HEIGHT))
# destination_copy = destination.copy()



get_gpu_info()
block_and_grid_info(WIDTH, HEIGHT)



# IMAGE = scale(IMAGE, (800, 800))
# t1 = timeit.timeit("test(IMAGE)", "from __main__ import IMAGE, test", number=500)
# print(t1/500)

# test_write_float_surface(IMAGE)

pred = pygame.mixer.Sound("../Assets/predator.ogg")
pred_play = False
vision_swap = pygame.mixer.Sound("../Assets/vision_swap.ogg")
vision_swap_play = False
vision_swap_play1 = False
vision_swap_play2 = False

FISHEYE_MODEL = fisheye_footprint(WIDTH, HEIGHT, 400, 300)

angle = 0
FRAME = 0
GAME = True
event_get = pygame.event.get
get_key = pygame.key.get_pressed
get_pos = pygame.mouse.get_pos
bl = 0
vbl = 1

import time
ti = time.time()


grad = create_horizontal_gradient_1d(math.sqrt(800 * 800 + 300 * 300), start_color=(255, 255, 220), end_color=(0, 0, 0))

#
# IMAGE = pygame.image.load("C:\\Users\\yoyob\\Desktop\\jt00c0tnu60b1.webp").convert()
# IMAGE = pygame.transform.smoothscale(IMAGE, (WIDTH, HEIGHT))


while GAME:
    # IMAGE = make_surface(hsv_gpu(cp.asarray(pixels3d(IMAGE).astype(dtype=cp.float32)), 0.1))
    # bloom(IMAGE, thresh, False)
    pygame.event.pump()

    keys = get_key()
    for event in event_get():

        if keys[pygame.K_ESCAPE]:
            GAME = False
            break

        if keys[pygame.K_F9]:
            im = smoothscale(SCREEN, (256, 256))
            pygame.image.save(im, "zoom1.png")

        if event.type == pygame.MOUSEMOTION:
            mouse_pos = pygame.math.Vector2(get_pos())

    # SCREEN.fill((1, 1, 1))
    #SCREEN.blit(IMAGE, (0, 0), special_flags=BLEND_RGB_ADD)

    #
    # IMAGE = invert_gpu(IMAGE)
    # invert_inplace_cupy(pixels3d(IMAGE))
    #
    # IMAGE = sepia_gpu(IMAGE)
    # sepia_inplace_cupy(pixels3d(IMAGE))

    # IMAGE = bpf_gpu(IMAGE, thresh)
    # IMAGE = bpf1_gpu(IMAGE, grid, block, thresh)
    #
    #
    # IMAGE = grayscale_gpu(IMAGE)
    # IMAGE = grayscale_lum_gpu(IMAGE)
    #
    # IMAGE = median_gpu(IMAGE)
    # IMAGE = median1_gpu(IMAGE)

    # IMAGE = gaussian_3x3_gpu(IMAGE)
    # IMAGE = gaussian_5x5_gpu(IMAGE)

    # IMAGE = sobel_gpu(IMAGE)
    # IMAGE = prewitt_gpu(IMAGE)
    # IMAGE = canny_gpu(IMAGE)

    # IMAGE = color_reduction_gpu(IMAGE)

    # IMAGE = hsv_gpu(IMAGE, hsv, grid, block)

    # surf = hsl_gpu(IMAGE, hsv, grid, block)

    # hsl_effect(IMAGE, hsv)

    # IMAGE = brightness_gpu(IMAGE, sat, grid, block)


    # IMAGE = cartoon_gpu(
    #     IMAGE ,
    #     sobel_threshold_= 255,
    #     median_kernel_  = 4,
    #     color_          = 8,
    #     contour_        = True,
    #     flag_           = pygame.BLEND_RGB_ADD)

    # IMAGE = blending_gpu(source, destination, 0)
    # IMAGE = blend(source, destination, 50.0)
    # IMAGE = alpha_blending(source, destination)
    # alpha_blending_inplace(source, destination)

    # IMAGE = sharpen_gpu(IMAGE)

    # IMAGE = bloom_gpu(IMAGE, thresh, False, BLEND_RGB_ADD, factor_ = 3)

    # bpf(IMAGE, thresh)
    # x2, x4, x8, x16 = mult_downscale_gpu(cp.asarray(pixels3d(IMAGE)))

    # bloom(IMAGE, thresh, fast_=False)

    # IMAGE = shader_bloom_fast(IMAGE, threshold_=thresh)

    # shader_bloom_fast1(IMAGE, threshold_=thresh, smooth_=3, flag_=BLEND_RGB_ADD, saturation_=False)

    # sharpen1_gpu(IMAGE, grid, block)

    # IMAGE = mirroring_gpu(IMAGE, grid, block, 1)

    # IMAGE = saturation_gpu(IMAGE, grid, block, sat)

    # IMAGE = bilateral_gpu(IMAGE, 4)

    # IMAGE = bilateral(IMAGE, 16.0, 80.0, kernel_size=3)

    # IMAGE = bilateral_fast_gpu(IMAGE, 4)

    # IMAGE = emboss5x5_gpu(IMAGE)

    # IMAGE = dithering_gpu(cp.asarray(pixels3d(IMAGE)), grid, block, 5.0)
    # dithering_int(IMAGE, 10)

    # IMAGE = test(IMAGE, grid, block)

    # surf = fisheye_gpu(IMAGE, thresh/125.0, 0.35, grid, block)
    # fisheye(IMAGE, FISHEYE_MODEL)

    # surf = swirl_gpu(IMAGE, angle/2 * 3.14/180.0, grid, block, 400, 300)
    # swirl2(IMAGE, 170.0)

    # wave(IMAGE, 8 * math.pi/180.0 + FRAME/10, 6)
    # surf = wave_gpu(IMAGE, 8 * math.pi/180.0 + FRAME/10, 8, grid, block)
    # surf = scale(surf, (WIDTH + 16, HEIGHT + 16))

    # surf = rgb_split_gpu(IMAGE, 10, 10, grid, block)

    # surf = zoom_gpu(IMAGE, mouse_pos.x, mouse_pos.y, grid, block, abs(hsv))


    # IMAGE = predator_vision(IMAGE,
    #                 sobel_threshold=1,
    #                 bpf_threshold=0,
    #                 bloom_threshold=0,
    #                 inv_colormap=False,
    #                 blend=pygame.BLEND_RGBA_ADD)
    # surf = chromatic_gpu(IMAGE, mouse_pos.x, mouse_pos.y, grid, block, 0.999, fx=abs(hsv / 12))

    # IMAGE = chromatic(IMAGE, 400, 300, 1.0, fx=0.0)

    # IMAGE = zoom(IMAGE, 400, 300, zx=abs(sat))

    # IMAGE = wavelength_map_gpu(IMAGE, grid, block, 0)

    # IMAGE = heatmap_gpu(IMAGE, grid, block, 1)
    # heatmap_gpu_inplace(IMAGE, grid, block, 0)

    # shader_rgb_to_yiq_inplace(IMAGE)

    # surf = downscale_surface_gpu(IMAGE, (75, 100), (1, 1), 0.9999, 100, 75)

    # surf = create_radial_gradient(800, 600, grad, factor_=1.0/1.8, threads_=10)
    # grad = create_horizontal_gradient_1d(
    #     1000,
    #     start_color=((255 - (thresh + 12)) % 255, (0 + thresh + 2) % 255 , (255 + (thresh + 8) ) % 255 ),
    #     end_color=(0, 0, 0))

    # SCREEN.blit(BCK, (0, 0))
    #SCREEN.fill((0, 0, 0, 0))

    # surf = create_quarter_radial_gradient_alpha\
    #     (800, 600,
    #
    #      start_color_   = (255-thresh, 0+thresh, 0, 255),
    #      end_color_     = (0, 0, 8, 0),
    #      factor_        = 1.0/1.4,
    #      threads_       = 8
    # )



    # blood(SCREEN, BLOOD_MASK, hsv)


    IMAGE_COPY = IMAGE.copy()
    # shader_bloom_fast1(SCREEN, smooth_=2, threshold_=0, flag_=BLEND_RGB_ADD, saturation_=True)

    #IMAGE = zoom(IMAGE, 360, 550, zx=0.291)
    #IMAGE = bilateral_gpu(IMAGE, 8)
    #bloom(IMAGE, 128)
    SCREEN.blit(IMAGE, (0, 0))


    if 4 < time.time() - ti <= 9:

        if not pred_play:
            pred.play()
        IMAGE = predator_gpu(
            IMAGE,
            grid, block,
            bloom_smooth = 10,
            bloom_threshold = int(bl),
            inv_colormap=True,
            blend = pygame.BLEND_RGB_MAX,
            bloom_flag = pygame.BLEND_RGB_ADD
        )
        pred_play = True



    elif 10 < time.time() -ti < 16:
        if vision_swap_play is False:
            pred.stop()
            pred.play()
        shader_bloom_fast1(IMAGE, smooth_=int(bl), threshold_=0, flag_=pygame.BLEND_RGB_ADD,
                           saturation_=True)
        IMAGE = wavelength_map_gpu(IMAGE, grid, block, 2)
        vision_swap_play = True


    elif 17 < time.time() -ti < 22:
        if vision_swap_play1 is False:
            pred.stop()
            pred.play()
        blur(IMAGE, t_=4)
        shader_bloom_fast1(IMAGE, smooth_=int(bl), threshold_=0, flag_=pygame.BLEND_RGB_ADD,
                           saturation_=True)
        IMAGE = wavelength_map_gpu(IMAGE, grid, block, 0)
        vision_swap_play1 = True


    elif 24 < time.time() -ti < 50:
        if vision_swap_play2 is False:
            pred.stop()
            pred.play()
        blur(IMAGE, t_=4)
        shader_bloom_fast1(IMAGE, smooth_=int(bl), threshold_=0, flag_=pygame.BLEND_RGB_ADD,
                           saturation_=True)
        # IMAGE = wavelength_map_gpu(IMAGE, grid, block, 1)

        IMAGE = bpf_gpu(IMAGE, threshold_=150)

        #IMAGE = sobel_gpu(IMAGE)
        #IMAGE.blit(IMAGE_, (0,0), special_flags=pygame.BLEND_RGB_SUB)
        vision_swap_play2 = True

    SCREEN.blit(IMAGE, (0, 0))

    if bl>=50:
        vbl = -1
    elif bl <=0:
        vbl = +1
    bl += vbl



    thresh -= f
    if thresh <= 0:
        thresh = 0
        f *= -1
    if thresh >= 255:
        f *= -1
        tresh = 255
    thresh %= 255

    hsv += v
    if hsv > 1.0:
        hsv = 0.99
        v *= -1
    if hsv < 0.0:
        hsv = 0.01
        v *= -1

    sat += s
    if sat >= 1.0:
        sat = 0.99
        s *= -1
    if sat <= -1.0:
        sat = -0.99
        s *= -1

    angle += 5
    # angle %= 360

    # im = make_surface(im.transpose(1, 0, 2))
    # arr_x2, arr_x4, arr_x8, arr_x16 = downscale_x2_gpu(pixels3d(IMAGE))
    # arr_x2 = downscale_x2_gpu(pixels3d(IMAGE))
    # im = make_surface(im)
    #SCREEN.fill((0, 0, 0))


    # x16 = smoothscale(make_surface(cp.asnumpy(x2)), (WIDTH, HEIGHT))
    # x2 = make_surface(cp.asnumpy(x2))

    # SCREEN.blit(surf, (0, 0)) # , special_flags=BLEND_RGB_ADD)




    clock.tick(8000)
    t = clock.get_fps()
    avg.append(t)
    show_fps(SCREEN, t, avg)

    IMAGE = IMAGE_COPY.copy()

    FRAME += 1
    pygame.display.flip()


pygame.quit()




raise SystemExit




v = 0.01
hsv = 0
avg = []
grid, block = block_grid(IMAGE.get_width(), IMAGE.get_height())
while 1:
    arr = hsv_gpu(cp.asarray(pixels3d(IMAGE), dtype=cp.float32), hsv, grid, block)
    # hsl_effect(IMAGE, hsv)
    IMAGE = make_surface(arr)

    hsv += v
    if hsv > 1.0:
        hsv = 1.0
        v *= -1
    if hsv < 0.0:
        hsv = 0.0
        v *= -1

    # arr = grayscale_lum_gpu(cp.asarray(pixels3d(IMAGE), dtype=cp.uint8))



    # sobel(IMAGE, 0)
    # cartoon(IMAGE)
    # median(IMAGE, 8)
    # SCREEN.fill((0, 0, 0))
    pygame.event.pump()
    # r = sepia_cupy1(arr)
    # print(r, r.shape)
    # surf = make_surface(r)
    # pygame.surfarray.blit_array(SCREEN, arr)
    SCREEN.blit(IMAGE, (0, 0))
    IMAGE = IMAGE_COPY.copy()
    clock.tick(8000)
    t = clock.get_fps()
    avg.append(t)
    show_fps(SCREEN, t, avg)
    # print(t)
    pygame.display.flip()
    pass


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

IMAGE = load_image(im="parrot.jpg")
palette_change(IMAGE, PICO_8)
pygame.image.save(IMAGE, "../Assets/parrot_PICO_8_CUSTOM.png")
t = timeit.timeit("palette_change(IMAGE, PICO_8_CUSTOM)",
              "from __main__ import palette_change, IMAGE, PICO_8_CUSTOM",
              number=10)
print(t/10)

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
#
# IMAGE = load_image(im="cat.png")
# # greyscale(IMAGE)
# dithering_int(IMAGE, factor_=8)
# pygame.image.save(IMAGE, "../Assets/cat_DITHERING_INT.jpg")
# t = timeit.timeit("dithering_int(IMAGE, factor_=4)",
#                   "from __main__ import dithering_int, IMAGE", number=100)
# print("DITHERING INT : ", t / 100)
# #
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
