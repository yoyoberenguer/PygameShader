""""
TEST LIBRARY shader
"""
import ctypes
import math
import sys
import unittest
import os
import time
from random import uniform, randint

import cython
import numpy
from numpy import uint8, asarray
from pygame.surfarray import make_surface
from pygame.transform import scale, smoothscale

from PygameShader.misc import create_horizontal_gradient_1d

import pygame
from pygame import BLEND_RGB_ADD, RLEACCEL

from PygameShader.shader import shader_rgb_to_bgr_inplace, shader_rgb_to_brg_inplace, \
    shader_greyscale_luminosity24_inplace, shader_sepia24_inplace, \
    shader_median_filter24_inplace, shader_median_grayscale_filter24_inplace, \
    shader_median_filter24_avg_inplace, shader_color_reduction24_inplace, shader_sobel24_inplace, \
    shader_sobel24_fast_inplace, shader_invert_surface_24bit_inplace, \
    shader_hsl_surface24bit_inplace, shader_hsl_surface24bit_fast_inplace, rgb_to_hsl_model, \
    hsl_to_rgb_model, shader_blur5x5_array24_inplace, shader_wave24bit_inplace, \
    shader_swirl24bit_inplace, shader_swirl24bit_inplace1, shader_plasma24bit_inplace, \
    shader_brightness24_inplace, shader_saturation_array24_inplace, shader_bpf24_inplace, \
    shader_bloom_effect_array24, shader_fisheye24_inplace, shader_fisheye24_footprint_inplace, \
    shader_tv_scanline_inplace, heatmap_surface24_conv_inplace, predator_vision_mode, \
    shader_blood_inplace, shader_sharpen_filter_inplace, shader_fire_effect, custom_map, rgb_to_int, \
    dampening_effect, lateral_dampening_effect, mirroring_inplace, shader_cloud_effect, \
    tunnel_render32, tunnel_modeling32, shader_ripple, shader_rgb_split_inplace, \
    shader_heatwave24_vertical_inplace, shader_horizontal_glitch24_inplace, shader_plasma

import PygameShader

PROJECT_PATH = list(PygameShader.__path__)
os.chdir(PROJECT_PATH[0] + "\\tests")

WIDTH = 1024
HEIGHT = 768
SCREEN = pygame.display.set_mode((WIDTH, HEIGHT), vsync=True)
SCREEN.convert(32, RLEACCEL)
SCREEN.set_alpha(None)


shader_list =\
    {
        "shader_rgb_to_bgr_inplace"                 : "RGB to BGR",
        "shader_rgb_to_brg_inplace"                 : "RGB to BRG",
        "shader_greyscale_luminosity24_inplace"     : "Luminosity",
        "shader_sepia24_inplace"                    : "Sepia",
        "shader_median_filter24_inplace"            : "Media",
        "shader_median_grayscale_filter24_inplace"  : "Median grayscale",
        "shader_median_filter24_avg_inplace"        : "Median avg",
        "shader_color_reduction24_inplace"          : "Color reduction",
        "shader_sobel24_inplace"                    : "Sobel",
        "shader_sobel24_fast_inplace"               : "Sobel fast",
        "shader_invert_surface_24bit_inplace"       : "Invert",
        "shader_hsl_surface24bit_inplace"           : "HSL",
        "shader_hsl_surface24bit_fast_inplace"      : "HSL fast",
        "shader_blur5x5_array24_inplace"            : "Blur",
        "shader_wave24bit_inplace"                  : "Wave",
        "shader_swirl24bit_inplace"                 : "Swirl",
        "shader_swirl24bit_inplace1"                : "Swirl1",
        "shader_plasma24bit_inplace"                : "Plasma",
        "shader_brightness24_inplace"               : "Brightness",
        "shader_saturation_array24_inplace"         : "Saturation",
        "shader_bpf24_inplace"                      : "Bright pass filter",
        "shader_bloom_effect_array24"               : "Bloom",
        "shader_fisheye24_inplace"                  : "Fisheye",
        "shader_tv_scanline_inplace"                : "TV",
        "heatmap_surface24_conv_inplace"            : "Heatmap",
        "predator_vision_mode"                      : "Predator Vision",
        "shader_blood_inplace"                      : "Blood",
        "shader_sharpen_filter_inplace"             : "Sharpen",
        "shader_fire_effect"                        : "Fire Effect 1",
        "dampening_effect"                          : "Dampening",
        "lateral_dampening_effect"                  : "Lateral dampening",
        "mirroring_inplace"                         : "Mirror image",
        "shader_cloud_effect"                       : "Shader cloud",
        "tunnel_render32"                           : "Tunnel",
        "shader_ripple"                             : "Water Ripple",
        "shader_rgb_split_inplace"                  : "RGB split",
        "shader_heatwave24_vertical_inplace"        : "Heatwave",
        "shader_horizontal_glitch24_inplace"        : "Horizontal glitch",
        "shader_plasma"                             : "Plasma 2"
}


def display_shader(shader_name: str, timer = 5, flag=0, *args, **kwargs):

    background = pygame.image.load("../Assets/background.jpg").convert()
    background = pygame.transform.smoothscale(background, (WIDTH, HEIGHT))
    background.convert(32, RLEACCEL)
    image = background.copy()

    all_shaders = shader_list.keys()
    if shader_name not in all_shaders:
        raise ValueError("\nShader name is invalid got %s" % shader_name)

    try:
        pygame.display.set_caption("Testing %s" % shader_list[shader_name])

    except IndexError:
        raise IndexError('\nShader name is incorrect got %s' % shader_name)

    shader_ = shader_list[shader_name]

    frame = 0
    clock = pygame.time.Clock()
    game = True

    t = time.time()

    options = " "
    for opt in args:
        options += ", " + str(opt)

    hsl_rotation = False

    if shader_name in ("shader_hsl_surface24bit_inplace",
                       "shader_hsl_surface24bit_fast_inplace"):
        hsl_rotation = True
        v = 0.001
        hsl_value = 0.0

    angle = 0

    while game:

        if hsl_rotation:
            if args is None:
                exec(str(shader_name) + "(image, hsl_value)")
        else:
            exec(str(shader_name)+"(image)"
                 if kwargs is None else str(shader_name)+"(image" + options + ")")

        pygame.event.pump()
        for event in pygame.event.get():

            keys = pygame.key.get_pressed()

            if keys[pygame.K_ESCAPE]:
                game = False

        SCREEN.blit(image, (0, 0), special_flags=flag)
        pygame.display.flip()
        clock.tick()
        frame += 1

        pygame.display.set_caption(
            "Testing %s @ %s fps (%sx%s) " %
            (shader_, round(clock.get_fps(), 2), WIDTH, HEIGHT))
        image = background.copy()

        if hsl_rotation:

            if hsl_value > 1.0:
                v *= -1
            elif hsl_value < -1.0:
                v *= -1

            hsl_value += v

            if hsl_value > 1.0:
                hsl_value = 1.0

            elif hsl_value < -1.0:
                hsl_value = -1.0

        if time.time() - t > timer:
            break

        angle += 0.01
        angle %= 1


class ShaderRgbToBgrInplace(unittest.TestCase):
    """
    Test shader_rgb_to_bgr_inplace
    """

    # pylint: disable=too-many-statements
    @staticmethod
    def runTest() -> None:
        """

        :return:  void
        """
        display_shader("shader_rgb_to_bgr_inplace", timer = 5, flag=0)


class TestShaderRgbToBrgInplace(unittest.TestCase):
    """
    Test shader_rgb_to_brg_inplace
    """

    # pylint: disable=too-many-statements
    @staticmethod
    def runTest() -> None:
        """

        :return:  void
        """
        display_shader("shader_rgb_to_brg_inplace", timer = 5, flag=0)


class TestShaderGreyscaleLuminosity24Inplace(unittest.TestCase):
    """
    Test shader_greyscale_luminosity24_inplace
    """

    # pylint: disable=too-many-statements
    @staticmethod
    def runTest() -> None:
        """

        :return:  void
        """
        display_shader("shader_greyscale_luminosity24_inplace", timer = 5, flag=0)


class TestShaderSepia24Inplace(unittest.TestCase):
    """
    Test shader_sepia24_inplace
    """

    # pylint: disable=too-many-statements
    @staticmethod
    def runTest() -> None:
        """

        :return:  void
        """
        display_shader("shader_sepia24_inplace", timer = 5, flag=0)


class TestShaderMedianFilter24Inplace(unittest.TestCase):
    """
    Test shader_median_filter24_inplace
    """

    # pylint: disable=too-many-statements
    @staticmethod
    def runTest() -> None:
        """

        :return:  void
        """
        display_shader("shader_median_filter24_inplace", timer = 5, flag=0)


class TestShaderMedianGrayscaleFilter24Inplace(unittest.TestCase):
    """
    Test shader_median_grayscale_filter24_inplace
    """

    # pylint: disable=too-many-statements
    @staticmethod
    def runTest() -> None:
        """

        :return:  void
        """
        display_shader("shader_median_grayscale_filter24_inplace", timer = 5, flag=0)


class TestShaderMedianFilter24AvgInplace(unittest.TestCase):
    """
    Test shader_median_filter24_avg_inplace
    """

    # pylint: disable=too-many-statements
    @staticmethod
    def runTest() -> None:
        """

        :return:  void
        """
        display_shader("shader_median_filter24_avg_inplace", timer = 5, flag=0)


class TestShaderColorReduction24Inplace(unittest.TestCase):
    """
    Test shader_color_reduction24_inplace
    """

    # pylint: disable=too-many-statements
    @staticmethod
    def runTest() -> None:
        """

        :return:  void
        """
        background = pygame.image.load("../Assets/background.jpg").convert()
        background = pygame.transform.smoothscale(background, (WIDTH, HEIGHT))
        background.convert(32, RLEACCEL)
        image = background.copy()

        pygame.display.set_caption("Test shader_color_reduction24_inplace")

        FRAME = 0
        CLOCK = pygame.time.Clock()
        GAME = True

        COLORS = [64, 32, 24, 16, 8, 4, 2, 1]
        INDEX = 0
        t = time.time()
        while GAME:
            shader_color_reduction24_inplace(image, color_=COLORS[INDEX])
            pygame.event.pump()
            for event in pygame.event.get():

                keys = pygame.key.get_pressed()

                if keys[pygame.K_ESCAPE]:
                    GAME = False

            # SCREEN.fill((0, 0, 0))
            SCREEN.blit(image, (0, 0))
            pygame.display.flip()
            CLOCK.tick()
            FRAME += 1
            pygame.display.set_caption(
                "Test shader_color_reduction24_inplace %s fps "
                "(%sx%s) %s colors" % (round(CLOCK.get_fps(), 2), WIDTH, HEIGHT,
                                       COLORS[INDEX] ** 2 * 2))
            image = background.copy()
            if FRAME % 1000 == 0:
                INDEX += 1
                if INDEX == len(COLORS):
                    INDEX = 0
            if time.time() - t > 5:
                break


class TestShaderSobel24Inplace(unittest.TestCase):
    """
    Test shader_sobel24_inplace
    """

    # pylint: disable=too-many-statements
    @staticmethod
    def runTest() -> None:
        """

        :return:  void
        """
        display_shader("shader_sobel24_inplace")


class TestShaderSobel24FastInplace(unittest.TestCase):
    """
    Test shader_sobel24_fast_inplace
    """

    # pylint: disable=too-many-statements
    @staticmethod
    def runTest() -> None:
        """

        :return:  void
        """
        display_shader("shader_sobel24_fast_inplace", timer = 5, flag=0)


class TestShaderInvertSurface24bitInplace(unittest.TestCase):
    """
    Test shader_invert_surface_24bit_inplace
    """

    # pylint: disable=too-many-statements
    @staticmethod
    def runTest() -> None:
        """

        :return:  void
        """
        display_shader("shader_invert_surface_24bit_inplace", timer = 5, flag=0)


class TestShaderHslSurface24bitInplace(unittest.TestCase):
    """
    Test shader_hsl_surface24bit_inplace
    """

    # pylint: disable=too-many-statements
    @staticmethod
    def runTest() -> None:
        """

        :return:  void
        """
        display_shader("shader_hsl_surface24bit_inplace", timer = 5, flag=0)


class TestShaderHslSurface24bitFastInplace(unittest.TestCase):
    """
    Test shader_hsl_surface24bit_fast_inplace
    """

    # pylint: disable=too-many-statements
    @staticmethod
    def runTest() -> None:
        """

        :return:  void
        """
        background = pygame.image.load("../Assets/background.jpg").convert()
        background = pygame.transform.smoothscale(background, (WIDTH, HEIGHT))
        background.convert(32, RLEACCEL)
        image = background.copy()

        pygame.display.set_caption("Test shader_hsl_surface24bit_fast_inplace")

        frame = 0
        clock = pygame.time.Clock()
        game = True

        v = 0.001
        hsl_value = 0.0

        rgb2hsl_model = hsl_to_rgb_model()
        hsl2rgb_model = rgb_to_hsl_model()

        t = time.time()
        while game:
            shader_hsl_surface24bit_fast_inplace(
                image,
                hsl_value,
                hsl_model_=hsl2rgb_model,
                rgb_model_=rgb2hsl_model)

            pygame.event.pump()
            for event in pygame.event.get():

                keys = pygame.key.get_pressed()

                if keys[pygame.K_ESCAPE]:
                    game = False

            SCREEN.blit(image, (0, 0))
            pygame.display.flip()
            clock.tick()
            frame += 1
            pygame.display.set_caption(
                "Test shader_hsl_surface24bit_fast_inplace %s fps "
                "(%sx%s)" % (round(clock.get_fps(), 2), WIDTH, HEIGHT))
            image = background.copy()
            if hsl_value >= 1.0:
                v *= -1
            elif hsl_value <= -1.0:
                v *= -1
            hsl_value += v
            if time.time() - t > 5:
                break


class TestShaderBlur5x5Array24Inplace(unittest.TestCase):
    """
    Test shader_blur5x5_array24_inplace
    """

    # pylint: disable=too-many-statements
    @staticmethod
    def runTest() -> None:
        """

        :return:  void
        """

        display_shader("shader_blur5x5_array24_inplace")


class TestShaderWave24bitInplace(unittest.TestCase):
    """
    Test shader_wave24bit_inplace
    """

    # pylint: disable=too-many-statements
    @staticmethod
    def runTest() -> None:
        """

        :return:  void
        """
        background = pygame.image.load("../Assets/background.jpg").convert()
        background = pygame.transform.smoothscale(background, (WIDTH, HEIGHT))
        background.convert(32, RLEACCEL)
        image = background.copy()
        pygame.display.set_caption("Test shader_wave24bit_inplace")

        frame = 0
        clock = pygame.time.Clock()
        game = True
        angle = 0
        t = time.time()
        while game:
            shader_wave24bit_inplace(image, angle * math.pi / 180, 10)

            pygame.event.pump()
            for event in pygame.event.get():

                keys = pygame.key.get_pressed()

                if keys[pygame.K_ESCAPE]:
                    game = False

            SCREEN.blit(image, (0, 0))
            pygame.display.flip()
            clock.tick()
            frame += 1
            pygame.display.set_caption(
                "Test shader_wave24bit_inplace %s fps "
                "(%sx%s)" % (round(clock.get_fps(), 2), WIDTH, HEIGHT))
            image = background.copy()
            angle += 5
            angle %= 360
            if time.time() - t > 5:
                break


class TestShaderSwirl24bitInplace(unittest.TestCase):
    """
    Test shader_swirl24bit_inplace
    """

    # pylint: disable=too-many-statements
    @staticmethod
    def runTest() -> None:
        """

        :return:  void
        """
        background = pygame.image.load("../Assets/background.jpg").convert()
        background = pygame.transform.smoothscale(background, (WIDTH, HEIGHT))
        background.convert(32, RLEACCEL)
        image = background.copy()

        pygame.display.set_caption("Test shader_swirl24bit_inplace")

        frame = 0
        clock = pygame.time.Clock()
        game = True
        angle = 0
        t = time.time()
        while game:
            shader_swirl24bit_inplace(image, angle)

            pygame.event.pump()
            for event in pygame.event.get():

                keys = pygame.key.get_pressed()

                if keys[pygame.K_ESCAPE]:
                    game = False

            # SCREEN.fill((0, 0, 0))
            SCREEN.blit(image, (0, 0))
            pygame.display.flip()
            clock.tick()
            frame += 1
            pygame.display.set_caption(
                "Test shader_swirl24bit_inplace %s fps "
                "(%sx%s)" % (round(clock.get_fps(), 2), WIDTH, HEIGHT))
            image = background.copy()
            angle += 0.01
            angle %= 1
            if time.time() - t > 5:

                break


class TestShaderSwirl24bitInplace1(unittest.TestCase):
    """
    Test shader_swirl24bit_inplace1
    """

    # pylint: disable=too-many-statements
    @staticmethod
    def runTest() -> None:
        """

        :return:  void
        """
        background = pygame.image.load("../Assets/background.jpg").convert()
        background = pygame.transform.smoothscale(background, (WIDTH, HEIGHT))
        background.convert(32, RLEACCEL)
        image = background.copy()
        pygame.display.set_caption("Test shader_swirl24bit_inplace1")

        FRAME = 0
        CLOCK = pygame.time.Clock()
        GAME = True
        ANGLE = 0
        t = time.time()
        while GAME:
            shader_swirl24bit_inplace1(image, ANGLE)

            pygame.event.pump()
            for event in pygame.event.get():

                keys = pygame.key.get_pressed()

                if keys[pygame.K_ESCAPE]:
                    GAME = False

            # SCREEN.fill((0, 0, 0))
            SCREEN.blit(image, (0, 0))
            pygame.display.flip()
            CLOCK.tick()
            FRAME += 1
            pygame.display.set_caption(
                "Test shader_swirl24bit_inplace1 %s fps "
                "(%sx%s)" % (round(CLOCK.get_fps(), 2), WIDTH, HEIGHT))
            image = background.copy()
            ANGLE += 0.01
            ANGLE %= 1
            if time.time() - t > 5:
                break


class TestShaderPlasmaInplace(unittest.TestCase):
    """
    Test shader_plasma
    """

    # pylint: disable=too-many-statements
    @staticmethod
    def runTest() -> None:
        """

        :return:  void
        """
        background = pygame.image.load("../Assets/background.jpg").convert()
        background = pygame.transform.smoothscale(background, (WIDTH, HEIGHT))
        background.convert(32, RLEACCEL)
        image = background.copy()
        pygame.display.set_caption("Test shader_plasma")

        FRAME = 0.0
        CLOCK = pygame.time.Clock()
        GAME = True

        arr = numpy.array([0, 1,  # violet
                           0, 1,  # blue
                           0, 1,  # green
                           2, 619,  # yellow
                           620, 650,  # orange
                           651, 660],  # red
                          numpy.int)

        HEATMAP = [custom_map(i - 20, arr, 1.0) for i in range(380, 800)]
        heatmap_array = numpy.zeros((800 - 380, 3), uint8)
        heatmap_rescale = numpy.zeros(255, numpy.uint)

        i = 0
        for t in HEATMAP:
            heatmap_array[i, 0] = t[0]
            heatmap_array[i, 1] = t[1]
            heatmap_array[i, 2] = t[2]
            i += 1
        for r in range(0, 255):
            s = int(r * (800.0 - 380.0) / 255)
            heatmap_rescale[r] = \
                rgb_to_int(heatmap_array[s][0], heatmap_array[s][1], heatmap_array[s][2])

        heatmap_rescale = numpy.ascontiguousarray(heatmap_rescale[::-1])
        t = time.time()
        while GAME:

            pygame.event.pump()
            for event in pygame.event.get():

                keys = pygame.key.get_pressed()

                if keys[pygame.K_ESCAPE]:
                    GAME = False

            im = image.copy()
            shader_plasma(im, FRAME, heatmap_rescale)
            im.blit(image, (0, 0), special_flags=BLEND_RGB_ADD)
            SCREEN.blit(im, (0, 0))
            pygame.display.flip()
            CLOCK.tick()
            FRAME += 0.2
            pygame.display.set_caption(
                "Test shader_plasma %s fps "
                "(%sx%s)" % (round(CLOCK.get_fps(), 2), WIDTH, HEIGHT))
            if time.time() - t > 5:
                break


class TestShaderPlasma24bitInplace(unittest.TestCase):
    """
    Test shader_plasma24bit_inplace
    """

    # pylint: disable=too-many-statements
    @staticmethod
    def runTest() -> None:
        """

        :return:  void
        """
        background = pygame.image.load("../Assets/background.jpg").convert()
        background = pygame.transform.smoothscale(background, (WIDTH, HEIGHT))
        background.convert(32, RLEACCEL)
        image = background.copy()
        pygame.display.set_caption("Test shader_plasma24bit_inplace")

        FRAME = 0.0
        CLOCK = pygame.time.Clock()
        GAME = True
        t = time.time()
        while GAME:
            shader_plasma24bit_inplace(image, FRAME)

            pygame.event.pump()
            for event in pygame.event.get():

                keys = pygame.key.get_pressed()

                if keys[pygame.K_ESCAPE]:
                    GAME = False

            # SCREEN.fill((0, 0, 0))
            SCREEN.blit(image, (0, 0))
            pygame.display.flip()
            CLOCK.tick()
            FRAME += 0.2
            pygame.display.set_caption(
                "Test shader_plasma24bit_inplace %s fps "
                "(%sx%s)" % (round(CLOCK.get_fps(), 2), WIDTH, HEIGHT))
            image = background.copy()
            if time.time() - t > 5:
                break


class TestShaderBrightness24Inplace(unittest.TestCase):
    """
    Test shader_brightness24_inplace
    """

    # pylint: disable=too-many-statements
    @staticmethod
    def runTest() -> None:
        """

        :return:  void
        """
        background = pygame.image.load("../Assets/background.jpg").convert()
        background = pygame.transform.smoothscale(background, (WIDTH, HEIGHT))
        background.convert(32, RLEACCEL)
        image = background.copy()
        pygame.display.set_caption("Test shader_brightness24_inplace")

        FRAME = 0
        CLOCK = pygame.time.Clock()
        GAME = True
        V = 0.001
        BRIGHT = 0.0
        t = time.time()
        while GAME:

            pygame.event.pump()
            for event in pygame.event.get():

                keys = pygame.key.get_pressed()

                if keys[pygame.K_ESCAPE]:
                    GAME = False
                    break

            shader_brightness24_inplace(image, BRIGHT)
            # SCREEN.fill((0, 0, 0))
            SCREEN.blit(image, (0, 0))
            pygame.display.flip()
            CLOCK.tick()
            FRAME += 1
            pygame.display.set_caption(
                "Test shader_brightness24_inplace %s fps "
                "(%sx%s)" % (round(CLOCK.get_fps(), 2), WIDTH, HEIGHT))
            image = background.copy()
            if BRIGHT >= 1.0:
                V *= -1
                BRIGHT = 1.0
            elif BRIGHT <= -1.0:
                V *= -1
                BRIGHT = -1

            BRIGHT += V
            if time.time() - t > 5:
                break

class TestShaderSaturationArray24Inplace(unittest.TestCase):
    """
    Test shader_saturation_array24_inplace
    """

    # pylint: disable=too-many-statements
    @staticmethod
    def runTest() -> None:
        """

        :return:  void
        """
        background = pygame.image.load("../Assets/background.jpg").convert()
        background = pygame.transform.smoothscale(background, (WIDTH, HEIGHT))
        background.convert(32, RLEACCEL)
        image = background.copy()

        pygame.display.set_caption("Test shader_saturation_array24_inplace")

        FRAME = 0
        CLOCK = pygame.time.Clock()
        GAME = True
        V = 0.01
        SATURATION = 0.0
        t = time.time()
        while GAME:

            pygame.event.pump()
            for event in pygame.event.get():

                keys = pygame.key.get_pressed()

                if keys[pygame.K_ESCAPE]:
                    GAME = False
                    break

            shader_saturation_array24_inplace(image, SATURATION)
            # SCREEN.fill((0, 0, 0))
            SCREEN.blit(image, (0, 0))
            pygame.display.flip()
            CLOCK.tick()
            FRAME += 1
            pygame.display.set_caption(
                "Test shader_saturation_array24_inplace %s fps "
                "(%sx%s)" % (round(CLOCK.get_fps(), 2), WIDTH, HEIGHT))
            image = background.copy()
            if SATURATION >= 1.0:
                V *= -1
                SATURATION = 1.0
            elif SATURATION <= -1.0:
                V *= -1
                SATURATION = -1

            SATURATION += V
            if time.time() - t > 5:
                break

class TestShaderBpf24Inplace(unittest.TestCase):
    """
    Test shader_bpf24_inplace
    """

    # pylint: disable=too-many-statements
    @staticmethod
    def runTest() -> None:
        """

        :return:  void
        """
        background = pygame.image.load("../Assets/background.jpg").convert()
        background = pygame.transform.smoothscale(background, (WIDTH, HEIGHT))
        background.convert(32, RLEACCEL)
        image = background.copy()

        pygame.display.set_caption("Test shader_bpf24_inplace")

        FRAME = 0
        CLOCK = pygame.time.Clock()
        GAME = True
        V = 0.1
        BPF = 0.0
        t=time.time()
        while GAME:

            pygame.event.pump()
            for event in pygame.event.get():

                keys = pygame.key.get_pressed()

                if keys[pygame.K_ESCAPE]:
                    GAME = False
                    break

            shader_bpf24_inplace(image, BPF)
            # SCREEN.fill((0, 0, 0))
            SCREEN.blit(image, (0, 0))
            pygame.display.flip()
            CLOCK.tick()
            FRAME += 1
            pygame.display.set_caption(
                "Test shader_bpf24_inplace %s fps "
                "(%sx%s)" % (round(CLOCK.get_fps(), 2), WIDTH, HEIGHT))
            image = background.copy()

            if BPF >= 255.0:
                V *= -1
                BPF = 255.0
            elif BPF <= 0.0:
                V *= -1
                BPF = 0

            BPF += V
            if time.time() - t > 5:
                break

class TestShaderBloomEffectArray24(unittest.TestCase):
    """
    Test shader_bloom_effect_array24
    """

    # pylint: disable=too-many-statements
    @staticmethod
    def runTest() -> None:
        """

        :return:  void
        """
        background = pygame.image.load("../Assets/background.jpg").convert()
        background = pygame.transform.smoothscale(background, (WIDTH, HEIGHT))
        background.convert(32, RLEACCEL)
        image = background.copy()
        pygame.display.set_caption("Test shader_bloom_effect_array24")

        FRAME = 0
        CLOCK = pygame.time.Clock()
        GAME = True
        V = 0.5
        BPF = 255
        t=time.time()
        while GAME:

            pygame.event.pump()
            for event in pygame.event.get():

                keys = pygame.key.get_pressed()

                if keys[pygame.K_ESCAPE]:
                    GAME = False
                    break

            shader_bloom_effect_array24(image, BPF, fast_=True)
            # SCREEN.fill((0, 0, 0))
            SCREEN.blit(image, (0, 0))
            pygame.display.flip()
            CLOCK.tick()
            FRAME += 1
            pygame.display.set_caption(
                "Test shader_bloom_effect_array24 %s fps "
                "(%sx%s)" % (round(CLOCK.get_fps(), 2), WIDTH, HEIGHT))
            image = background.copy()

            if BPF >= 255.0:
                V *= -1
                BPF = 255.0
            elif BPF <= 0.0:
                V *= -1
                BPF = 0

            BPF += V
            if time.time() - t > 5:
                break


class TestShaderFisheye24Inplace(unittest.TestCase):
    """
    Test shader_fisheye24_inplace
    """

    # pylint: disable=too-many-statements
    @staticmethod
    def runTest() -> None:
        """

        :return:  void
        """
        background = pygame.image.load("../Assets/background.jpg").convert()
        background = pygame.transform.smoothscale(background, (WIDTH, HEIGHT))
        background.convert(32, RLEACCEL)
        image = background.copy()

        pygame.display.set_caption("Test shader_fisheye24_inplace")

        FRAME = 0
        CLOCK = pygame.time.Clock()
        GAME = True

        fisheye_model = shader_fisheye24_footprint_inplace(WIDTH + 1, HEIGHT + 1)
        t = time.time()
        while GAME:

            pygame.event.pump()
            for event in pygame.event.get():

                keys = pygame.key.get_pressed()

                if keys[pygame.K_ESCAPE]:
                    GAME = False
                    break

            shader_fisheye24_inplace(image, fisheye_model)
            # SCREEN.fill((0, 0, 0))
            SCREEN.blit(image, (0, 0))
            pygame.display.flip()
            CLOCK.tick()
            FRAME += 1
            pygame.display.set_caption(
                "Test shader_fisheye24_inplace %s fps "
                "(%sx%s)" % (round(CLOCK.get_fps(), 2), WIDTH, HEIGHT))
            image = background.copy()
            if time.time() - t > 5:
                break


class TestShaderTvScanlineInplace(unittest.TestCase):
    """
    Test shader_tv_scanline_inplace
    """

    # pylint: disable=too-many-statements
    @staticmethod
    def runTest() -> None:
        """

        :return:  void
        """
        background = pygame.image.load("../Assets/background.jpg").convert()
        background = pygame.transform.smoothscale(background, (WIDTH, HEIGHT))
        background.convert(32, RLEACCEL)
        image = background.copy()
        pygame.display.set_caption("Test shader_tv_scanline_inplace")

        FRAME = 0
        CLOCK = pygame.time.Clock()
        GAME = True
        t = time.time()
        while GAME:

            pygame.event.pump()
            for event in pygame.event.get():

                keys = pygame.key.get_pressed()

                if keys[pygame.K_ESCAPE]:
                    GAME = False
                    break

            shader_tv_scanline_inplace(image, FRAME + 1)

            SCREEN.blit(image, (0, 0))
            pygame.display.flip()
            CLOCK.tick()
            FRAME += 1
            FRAME %= 10

            pygame.display.set_caption(
                "Test shader_tv_scanline_inplace %s fps "
                "(%sx%s)" % (round(CLOCK.get_fps(), 2), WIDTH, HEIGHT))
            image = background.copy()
            if time.time() - t > 5:
                break


class TESTHeatmapSurface24ConvInplace(unittest.TestCase):
    """
    Test heatmap_surface24_conv_inplace
    """

    # pylint: disable=too-many-statements
    @staticmethod
    def runTest() -> None:
        """

        :return:  void
        """
        background = pygame.image.load("../Assets/background.jpg").convert()
        background = pygame.transform.smoothscale(background, (WIDTH, HEIGHT))
        background.convert(32, RLEACCEL)
        image = background.copy()
        pygame.display.set_caption("Test heatmap_surface24_conv_inplace")

        FRAME = 0
        CLOCK = pygame.time.Clock()
        GAME = True
        t = time.time()
        while GAME:

            pygame.event.pump()
            for event in pygame.event.get():

                keys = pygame.key.get_pressed()

                if keys[pygame.K_ESCAPE]:
                    GAME = False
                    break

            heatmap_surface24_conv_inplace(image, True)
            # SCREEN.fill((0, 0, 0))
            SCREEN.blit(image, (0, 0))
            pygame.display.flip()
            CLOCK.tick()
            FRAME += 1
            pygame.display.set_caption(
                "Test heatmap_surface24_conv_inplace %s fps "
                "(%sx%s)" % (round(CLOCK.get_fps(), 2), WIDTH, HEIGHT))
            image = background.copy()
            if time.time() - t > 5:
                break


class TestPredatorVisionMode(unittest.TestCase):
    """
    Test predator_vision_mode
    """

    # pylint: disable=too-many-statements
    @staticmethod
    def runTest() -> None:
        """

        :return:  void
        """
        BACKGROUND = pygame.image.load("../Assets/background.jpg").convert()
        BACKGROUND = pygame.transform.smoothscale(BACKGROUND, (WIDTH, HEIGHT))
        image = BACKGROUND.copy()
        pygame.display.set_caption("Test predator_vision_mode")

        FRAME = 0
        CLOCK = pygame.time.Clock()
        GAME = True
        t = time.time()
        while GAME:

            pygame.event.pump()
            for event in pygame.event.get():

                keys = pygame.key.get_pressed()

                if keys[pygame.K_ESCAPE]:
                    GAME = False
                    break

            surface_ = predator_vision_mode(image, sobel_threshold=80, bpf_threshold=0,
                                            bloom_threshold=0, inv_colormap=True, fast=False)
            # SCREEN.fill((0, 0, 0))
            SCREEN.blit(surface_, (0, 0))
            pygame.display.flip()
            CLOCK.tick()
            FRAME += 1
            pygame.display.set_caption(
                "Test predator_vision_mode %s fps "
                "(%sx%s)" % (round(CLOCK.get_fps(), 2), WIDTH, HEIGHT))
            image = BACKGROUND.copy()
            if time.time() - t > 5:
                break


class TestShaderBloodInplace(unittest.TestCase):
    """
    Test shader_blood_inplace
    """

    # pylint: disable=too-many-statements
    @staticmethod
    def runTest() -> None:
        """

        :return:  void
        """
        background = pygame.image.load("../Assets/background.jpg").convert()
        background = pygame.transform.smoothscale(background, (WIDTH, HEIGHT))
        background.convert(32, RLEACCEL)
        image = background.copy()
        pygame.display.set_caption("Test shader_blood_inplace")

        FRAME = 0
        CLOCK = pygame.time.Clock()
        GAME = True
        blood_surface = pygame.image.load("../Assets/redvignette.png").convert_alpha()
        blood_surface = pygame.transform.smoothscale(blood_surface, (WIDTH, HEIGHT))
        BLOOD_MASK = numpy.asarray(
            pygame.surfarray.pixels_alpha(blood_surface) / 255.0, numpy.float32)
        PERCENTAGE = 0.0
        V = 0.005
        t = time.time()
        while GAME:

            pygame.event.pump()
            for event in pygame.event.get():

                keys = pygame.key.get_pressed()

                if keys[pygame.K_ESCAPE]:
                    GAME = False
                    break

            shader_blood_inplace(image, BLOOD_MASK, PERCENTAGE)
            # SCREEN.fill((0, 0, 0))
            SCREEN.blit(image, (0, 0))
            pygame.display.flip()
            CLOCK.tick()
            FRAME += 1
            pygame.display.set_caption(
                "Test shader_blood_inplace %s fps "
                "(%sx%s)" % (round(CLOCK.get_fps(), 2), WIDTH, HEIGHT))
            image = background.copy()
            PERCENTAGE += V
            if PERCENTAGE >= 1.0:
                V *= -1
                PERCENTAGE = 1.0
            elif PERCENTAGE <= 0.0:
                V *= -1
                PERCENTAGE = 0.0
            if time.time() - t > 5:
                break

class TestShaderSharpenFilterInplace(unittest.TestCase):
    """
    Test shader_sharpen_filter_inplace
    """

    # pylint: disable=too-many-statements
    @staticmethod
    def runTest() -> None:
        """

        :return:  void
        """
        background = pygame.image.load("../Assets/background.jpg").convert()
        background = pygame.transform.smoothscale(background, (WIDTH, HEIGHT))
        background.convert(32, RLEACCEL)
        image = background.copy()
        pygame.display.set_caption("Test shader_sharpen_filter_inplace")

        FRAME = 0
        CLOCK = pygame.time.Clock()
        GAME = True
        t = time.time()
        while GAME:

            pygame.event.pump()
            for event in pygame.event.get():

                keys = pygame.key.get_pressed()

                if keys[pygame.K_ESCAPE]:
                    GAME = False
                    break

            shader_sharpen_filter_inplace(image)
            SCREEN.blit(image, (0, 0))
            pygame.display.flip()
            CLOCK.tick()
            FRAME += 1
            pygame.display.set_caption(
                "Test shader_sharpen_filter_inplace %s fps "
                "(%sx%s)" % (round(CLOCK.get_fps(), 2), WIDTH, HEIGHT))
            image = background.copy()
            if time.time() - t > 5:
                break


class TestshaderFireEffect(unittest.TestCase):
    """
    Test shader_fire_effect with adjust_palette_=True
    """

    # pylint: disable=too-many-statements
    @staticmethod
    def runTest() -> None:
        """

        :return:  void
        """

        BACKGROUND = pygame.image.load("../Assets/background.jpg")
        BACKGROUND = pygame.transform.smoothscale(BACKGROUND, (WIDTH, HEIGHT))
        image = BACKGROUND.copy()
        pygame.display.set_caption("Test shader_fire_effect")

        FRAME = 0
        CLOCK = pygame.time.Clock()
        GAME = True

        arr = numpy.array([0, 1,  # violet
                           0, 1,  # blue
                           0, 1,  # green
                           2, 619,  # yellow
                           620, 650,  # orange
                           651, 660],  # red
                          numpy.int)

        HEATMAP = [custom_map(i - 20, arr, 1.0) for i in range(380, 800)]
        heatmap_array = numpy.zeros((800 - 380, 3), uint8)
        heatmap_rescale = numpy.zeros(255, numpy.uint)

        i = 0
        for t in HEATMAP:
            heatmap_array[i, 0] = t[0]
            heatmap_array[i, 1] = t[1]
            heatmap_array[i, 2] = t[2]
            i += 1
        for r in range(0, 255):
            s = int(r * (800.0 - 380.0) / 255)
            heatmap_rescale[r] = rgb_to_int(heatmap_array[s][0], heatmap_array[s][1],
                                            heatmap_array[s][2])

        heatmap_rescale = numpy.ascontiguousarray(heatmap_rescale[::-1])
        FIRE_ARRAY = numpy.zeros((HEIGHT, WIDTH), dtype=numpy.float32)
        t = time.time()
        while GAME:

            pygame.event.pump()
            for event in pygame.event.get():

                keys = pygame.key.get_pressed()

                if keys[pygame.K_ESCAPE]:
                    GAME = False
                    break
            SCREEN.blit(BACKGROUND, (0, 0))

            surface_ = shader_fire_effect(
                WIDTH, HEIGHT, 3.97 + uniform(0.002, 0.008),
                heatmap_rescale,
                FIRE_ARRAY, reduce_factor_=3, bloom_=True, fast_bloom_=False, bpf_threshold_=48,
                brightness_=True, brightness_intensity_=0.095, transpose_=False, border_=False,
                low_=0, high_=WIDTH, blur_=True).convert(
                32, RLEACCEL)

            SCREEN.blit(surface_, (0, 0), special_flags=BLEND_RGB_ADD)

            pygame.display.flip()
            CLOCK.tick()
            FRAME += 1
            pygame.display.set_caption(
                "Test shader_fire_effect %s fps "
                "(%sx%s)" % (round(CLOCK.get_fps(), 2), WIDTH, HEIGHT))
            image = BACKGROUND.copy()
            if time.time() - t > 5:
                break


class TestshaderFireEffect1(unittest.TestCase):
    """
    Test shader_fire_effect
    """

    # pylint: disable=too-many-statements
    @staticmethod
    def runTest() -> None:
        """

        :return:  void
        """
        BACKGROUND = pygame.image.load("../Assets/background.jpg")
        BACKGROUND = pygame.transform.smoothscale(BACKGROUND, (WIDTH, HEIGHT))
        image = BACKGROUND.copy()
        pygame.display.set_caption("Test shader_fire_effect")

        FRAME = 0
        CLOCK = pygame.time.Clock()
        GAME = True

        arr = numpy.array([0, 1,  # violet
                           0, 1,  # blue
                           0, 1,  # green
                           2, 619,  # yellow
                           620, 650,  # orange
                           651, 660],  # red
                          numpy.int)

        HEATMAP = [custom_map(i - 20, arr, 1.0) for i in range(380, 800)]
        heatmap_array = numpy.zeros((800 - 380, 3), uint8)
        heatmap_rescale = numpy.zeros(255, numpy.uint)

        i = 0
        for t in HEATMAP:
            heatmap_array[i, 0] = t[0]
            heatmap_array[i, 1] = t[1]
            heatmap_array[i, 2] = t[2]
            i += 1
        for r in range(0, 255):
            s = int(r * (800.0 - 380.0) / 255)
            heatmap_rescale[r] = rgb_to_int(heatmap_array[s][0], heatmap_array[s][1],
                                            heatmap_array[s][2])

        heatmap_rescale = numpy.ascontiguousarray(heatmap_rescale[::-1])
        FIRE_ARRAY = numpy.zeros((HEIGHT, WIDTH), dtype=numpy.float32)
        t = time.time()
        while GAME:

            pygame.event.pump()
            for event in pygame.event.get():

                keys = pygame.key.get_pressed()

                if keys[pygame.K_ESCAPE]:
                    GAME = False
                    break
            SCREEN.blit(BACKGROUND, (0, 0))
            surface_ = shader_fire_effect(
                WIDTH, HEIGHT, 3.97 + uniform(0.002, 0.008),
                heatmap_rescale,
                FIRE_ARRAY,
                reduce_factor_=3, fire_intensity_=16,
                smooth_=True, bloom_=True, fast_bloom_=False,
                bpf_threshold_=128, low_=1, high_=WIDTH, brightness_=True,
                brightness_intensity_=0.1, adjust_palette_=True,
                hsl_=(10, 80, 1.8), transpose_=False, border_=False,
                surface_=None).convert(32, RLEACCEL)

            SCREEN.blit(surface_, (0, 0), special_flags=BLEND_RGB_ADD)

            pygame.display.flip()
            CLOCK.tick()
            FRAME += 1
            pygame.display.set_caption(
                "Test shader_fire_effect %s fps "
                "(%sx%s)" % (round(CLOCK.get_fps(), 2), WIDTH, HEIGHT))
            image = BACKGROUND.copy()
            if time.time() - t > 5:
                break


class TestLateralDampeningEffect(unittest.TestCase):
    """
    Test lateral_dampening_effect
    """

    # pylint: disable=too-many-statements
    @staticmethod
    def runTest() -> None:
        """

        :return:  void
        """
        BACKGROUND = pygame.image.load("../Assets/background.jpg")
        BACKGROUND = pygame.transform.smoothscale(BACKGROUND, (WIDTH, HEIGHT))
        image = BACKGROUND.copy()
        pygame.display.set_caption("Test lateral_dampening_effect")

        FRAME = 0
        CLOCK = pygame.time.Clock()
        GAME = True
        t = time.time()
        while GAME:

            pygame.event.pump()
            for event in pygame.event.get():

                keys = pygame.key.get_pressed()

                if keys[pygame.K_ESCAPE]:
                    GAME = False
                    break
            # ZOOM IN
            # surf, xx, yy = dampening_effect(BACKGROUND, FRAME, WIDTH, HEIGHT, amplitude_=60,
            #                                 duration_=100, freq_=0.8)
            # SCREEN.blit(surf, (xx, yy))

            tm = lateral_dampening_effect(FRAME, amplitude_=10.0, duration_=10, freq_=50.0)
            SCREEN.blit(BACKGROUND, (tm, 0), special_flags=0)

            pygame.display.flip()
            CLOCK.tick()
            FRAME += 1
            pygame.display.set_caption(
                "Test lateral_dampening_effect %s fps "
                "(%sx%s)" % (round(CLOCK.get_fps(), 2), WIDTH, HEIGHT))
            image = BACKGROUND.copy()
            if time.time() - t > 5:
                break


class TestDampeningEffect(unittest.TestCase):
    """
    Test dampening_effect
    """

    # pylint: disable=too-many-statements
    @staticmethod
    def runTest() -> None:
        """

        :return:  void
        """
        BACKGROUND = pygame.image.load("../Assets/background.jpg")
        BACKGROUND = pygame.transform.smoothscale(BACKGROUND, (WIDTH, HEIGHT))
        image = BACKGROUND.copy()
        pygame.display.set_caption("Test dampening_effect")

        FRAME = 0
        CLOCK = pygame.time.Clock()
        GAME = True
        t = time.time()
        while GAME:

            pygame.event.pump()
            for event in pygame.event.get():

                keys = pygame.key.get_pressed()

                if keys[pygame.K_ESCAPE]:
                    GAME = False
                    break
            surf, xx, yy = dampening_effect(
                BACKGROUND,
                FRAME,
                WIDTH,
                HEIGHT,
                amplitude_=15.0,
                duration_=10,
                freq_=25.0)
            surf.convert(32, RLEACCEL)
            SCREEN.blit(surf, (xx, yy), special_flags=0)

            pygame.display.flip()
            CLOCK.tick()
            FRAME += 1
            pygame.display.set_caption(
                "Test dampening_effect %s fps "
                "(%sx%s)" % (round(CLOCK.get_fps(), 2), WIDTH, HEIGHT))
            image = BACKGROUND.copy()
            if time.time() - t > 5:
                break


class TestMirroringInplace(unittest.TestCase):
    """
    Test mirroring_inplace
    """

    # pylint: disable=too-many-statements
    @staticmethod
    def runTest() -> None:
        """

        :return:  void
        """
        BACKGROUND = pygame.image.load("../Assets/background.jpg")
        BACKGROUND = pygame.transform.smoothscale(BACKGROUND, (WIDTH, HEIGHT))
        image = BACKGROUND.copy()
        pygame.display.set_caption("Test mirroring_inplace")

        FRAME = 0
        CLOCK = pygame.time.Clock()
        GAME = True
        t = time.time()
        while GAME:

            pygame.event.pump()
            for event in pygame.event.get():

                keys = pygame.key.get_pressed()

                if keys[pygame.K_ESCAPE]:
                    GAME = False
                    break

            mirroring_inplace(image)

            SCREEN.blit(image, (0, 0))
            image = BACKGROUND.copy()
            pygame.display.flip()
            CLOCK.tick()
            FRAME += 1
            pygame.display.set_caption(
                "Test mirroring_inplace %s fps "
                "(%sx%s)" % (round(CLOCK.get_fps(), 2), WIDTH, HEIGHT))
            if time.time() - t > 5:
                break


class TestShaderCloudEffect(unittest.TestCase):
    """
    Test shader_cloud_effect
    """

    # pylint: disable=too-many-statements
    @staticmethod
    def runTest() -> None:
        """

        :return:  void
        """
        WIDTH = 1024
        HEIGHT = 768
        BACKGROUND = pygame.image.load("../Assets/space1.jpg")
        BACKGROUND = pygame.transform.smoothscale(BACKGROUND, (WIDTH, HEIGHT))
        image = BACKGROUND.copy()
        pygame.display.set_caption("Test cloud/Smoke shader")

        FRAME = 0
        CLOCK = pygame.time.Clock()
        GAME = True

        arr = numpy.array([0, 1,  # violet
                           0, 1,  # blue
                           0, 1,  # green
                           2, 619,  # yellow
                           620, 650,  # orange
                           651, 660],  # red
                          numpy.int)

        CLOUD_ARRAY = numpy.zeros((HEIGHT, WIDTH), dtype=numpy.float32)

        heatmap_rescale = numpy.zeros(256 * 2 * 3, numpy.uint)

        arr1 = create_horizontal_gradient_1d(255, (0, 0, 0), (255, 255, 255))
        arr2 = create_horizontal_gradient_1d(255, (255, 255, 255), (0, 0, 0))
        arr3 = numpy.concatenate((arr1, arr2), axis=None)
        i = 0
        for r in range(0, 1530, 3):
            heatmap_rescale[i] = rgb_to_int(arr3[r], arr3[r + 1], arr3[r + 2])
            i += 1
        t = time.time()
        while GAME:

            pygame.event.pump()
            for event in pygame.event.get():

                keys = pygame.key.get_pressed()

                if keys[pygame.K_ESCAPE]:
                    GAME = False
                    break
            SCREEN.blit(BACKGROUND, (0, 0))

            surface_ = shader_cloud_effect(
                WIDTH, HEIGHT, 3.970 + uniform(0.002, 0.008),
                heatmap_rescale,
                CLOUD_ARRAY,
                reduce_factor_=3, cloud_intensity_=randint(60, 80),
                smooth_=True, bloom_=True, fast_bloom_=True,
                bpf_threshold_=80, low_=0, high_=WIDTH, brightness_=True,
                brightness_intensity_=0.15,
                transpose_=False, surface_=None, blur_=True)

            SCREEN.blit(surface_, (0, 0), special_flags=BLEND_RGB_ADD)

            pygame.display.flip()
            CLOCK.tick()
            FRAME += 1
            pygame.display.set_caption(
                "Test shader_cloud_effect %s fps "
                "(%sx%s)" % (round(CLOCK.get_fps(), 2), WIDTH, HEIGHT))
            image = BACKGROUND.copy()
            if time.time() - t > 5:
                break


class TestTunnelRender32(unittest.TestCase):
    """
    Test tunnel_render32
    """

    # pylint: disable=too-many-statements
    @staticmethod
    def runTest() -> None:
        """

        :return:  void
        """
        WIDTH = 800
        HEIGHT = 800

        BACKGROUND = pygame.image.load("../Assets/space2.jpg").convert()
        BACKGROUND = pygame.transform.smoothscale(BACKGROUND, (WIDTH, HEIGHT))

        FRAME = 0
        CLOCK = pygame.time.Clock()
        GAME = True

        distances, angles, shades, scr_data = tunnel_modeling32(WIDTH, HEIGHT)
        dest_array = numpy.empty((WIDTH * HEIGHT * 4), numpy.uint8)
        t= time.time()
        while GAME:

            pygame.event.pump()
            for event in pygame.event.get():

                keys = pygame.key.get_pressed()

                if keys[pygame.K_ESCAPE]:
                    GAME = False
                    break

            SCREEN.blit(BACKGROUND, (0, 0))

            surface_ = tunnel_render32(
                FRAME * 25,
                WIDTH,
                HEIGHT,
                WIDTH >> 1,
                HEIGHT >> 1,
                distances,
                angles,
                shades,
                scr_data,
                dest_array
            )

            SCREEN.blit(surface_, (0, 0))  # , special_flags=pygame.BLEND_RGB_MAX)

            pygame.display.flip()
            CLOCK.tick()
            FRAME += 1
            pygame.display.set_caption(
                "Test tunnel %s fps "
                "(%sx%s)" % (round(CLOCK.get_fps(), 2), WIDTH, HEIGHT))
            if time.time() - t > 5:
                break


class TestShaderRipple(unittest.TestCase):
    """
    Test shader_ripple
    """

    # pylint: disable=too-many-statements
    @staticmethod
    def runTest() -> None:
        """

        :return:  void
        """
        # WIDTH = 800
        # HEIGHT = 800

        BACKGROUND = pygame.image.load("../Assets/space2.jpg").convert()
        BACKGROUND = pygame.transform.smoothscale(BACKGROUND, (WIDTH, HEIGHT))

        FRAME = 0
        CLOCK = pygame.time.Clock()
        GAME = True

        W2 = WIDTH >> 1
        H2 = HEIGHT >> 1

        current = numpy.zeros((W2, H2), dtype=numpy.float32)
        previous = numpy.zeros((W2, H2), dtype=numpy.float32)
        array_ = numpy.full((W2, H2, 3), 0, numpy.uint8)

        image = BACKGROUND.copy()
        t = time.time()
        while GAME:

            pygame.event.pump()
            for event in pygame.event.get():

                keys = pygame.key.get_pressed()

                if keys[pygame.K_ESCAPE]:
                    GAME = False
                    break

            SCREEN.blit(BACKGROUND, (0, 0))

            rnd = randint(0, 1000)
            if rnd > 900:
                m = randint(1000, 10000)
                previous[randint(0, W2 - 1), randint(0, H2 - 1)] = m

            previous, current, array_ = shader_ripple(W2, H2, previous, current, array_)

            surf = make_surface(asarray(array_, dtype=uint8))
            surf = smoothscale(surf, (WIDTH, HEIGHT))
            image.blit(surf, (0, 0), special_flags=pygame.BLEND_RGBA_ADD)

            SCREEN.blit(image, (0, 0))

            pygame.display.flip()
            CLOCK.tick()
            FRAME += 1
            pygame.display.set_caption(
                "Test ripple rain drop effect %s fps "
                "(%sx%s)" % (round(CLOCK.get_fps(), 2), WIDTH, HEIGHT))
            image = BACKGROUND.copy()
            if time.time() - t > 5:
                break


class TestRgbSplit(unittest.TestCase):
    """
    Test shader_rgb_split_inplace
    """

    # pylint: disable=too-many-statements
    @staticmethod
    def runTest() -> None:
        """

        :return:  void
        """

        BACKGROUND = pygame.image.load("../Assets/background.jpg").convert()
        BACKGROUND = pygame.transform.smoothscale(BACKGROUND, (WIDTH, HEIGHT))

        FRAME = 0
        CLOCK = pygame.time.Clock()
        GAME = True

        image = BACKGROUND.copy()
        t = time.time()
        while GAME:

            pygame.event.pump()
            for event in pygame.event.get():

                keys = pygame.key.get_pressed()

                if keys[pygame.K_ESCAPE]:
                    GAME = False
                    break

            shader_rgb_split_inplace(image, 15)

            SCREEN.blit(image, (0, 0))

            pygame.display.flip()
            CLOCK.tick()
            FRAME += 1
            pygame.display.set_caption(
                "Test rgb split effect %s fps "
                "(%sx%s)" % (round(CLOCK.get_fps(), 2), WIDTH, HEIGHT))
            image = BACKGROUND.copy()
            if time.time() - t > 5:
                break


class TestShaderHorizontalGlitch24Inplace(unittest.TestCase):
    """
    Test shader_horizontal_glitch24_inplace
    """

    # pylint: disable=too-many-statements
    @staticmethod
    def runTest() -> None:
        """

        :return:  void
        """
        # WIDTH = 800
        # HEIGHT = 600

        BACKGROUND = pygame.image.load("../Assets/background.jpg").convert()
        BACKGROUND = pygame.transform.smoothscale(BACKGROUND, (WIDTH, HEIGHT))

        FRAME = 0
        CLOCK = pygame.time.Clock()
        GAME = True

        image = BACKGROUND.copy()
        t = time.time()
        while GAME:

            pygame.event.pump()
            for event in pygame.event.get():

                keys = pygame.key.get_pressed()

                if keys[pygame.K_ESCAPE]:
                    GAME = False
                    break

            shader_horizontal_glitch24_inplace(image, 0.5, 0.08, FRAME % 20)

            SCREEN.blit(image, (0, 0))

            pygame.display.flip()
            CLOCK.tick()
            FRAME += 1
            pygame.display.set_caption(
                "Test horizontal glitch effect %s fps "
                "(%sx%s)" % (round(CLOCK.get_fps(), 2), WIDTH, HEIGHT))
            image = BACKGROUND.copy()
            if time.time() - t > 5:
                break


class TestShaderHeatwave24VerticalInplace(unittest.TestCase):
    """
    Test shader_heatwave24_vertical_inplace
    """

    # pylint: disable=too-many-statements
    @staticmethod
    def runTest() -> None:
        """

        :return:  void
        """
        # WIDTH = 800
        # HEIGHT = 600

        BACKGROUND = pygame.image.load("../Assets/background.jpg").convert()
        BACKGROUND = pygame.transform.smoothscale(BACKGROUND, (WIDTH, HEIGHT))

        FRAME = 0
        CLOCK = pygame.time.Clock()
        GAME = True

        image = BACKGROUND.copy()
        t = time.time()
        while GAME:

            pygame.event.pump()
            for event in pygame.event.get():

                keys = pygame.key.get_pressed()

                if keys[pygame.K_ESCAPE]:
                    GAME = False
                    break

            b = 0.1
            shader_heatwave24_vertical_inplace(
                image, numpy.full((WIDTH, HEIGHT), 255, dtype=numpy.uint8),
                b * uniform(150, 800), 0,
                sigma_=uniform(2, 4), mu_=b * 2
            )

            SCREEN.blit(image, (0, 0))

            pygame.display.flip()
            CLOCK.tick()
            FRAME += 1
            pygame.display.set_caption(
                "Test heatwave effect %s fps "
                "(%sx%s)" % (round(CLOCK.get_fps(), 2), WIDTH, HEIGHT))
            image = BACKGROUND.copy()
            if time.time() - t > 5:
                break


def run_testsuite():
    """
    test suite

    :return: void
    """

    suite = unittest.TestSuite()

    suite.addTests([
        ShaderRgbToBgrInplace(),
        TestShaderRgbToBrgInplace(),
        TestShaderGreyscaleLuminosity24Inplace(),
        TestShaderSepia24Inplace(),
        TestShaderMedianFilter24Inplace(),
        TestShaderMedianGrayscaleFilter24Inplace(),
        TestShaderMedianFilter24AvgInplace(),
        TestShaderColorReduction24Inplace(),
        TestShaderSobel24Inplace(),
        TestShaderSobel24FastInplace(),
        TestShaderInvertSurface24bitInplace(),
        TestShaderHslSurface24bitInplace(),
        TestShaderHslSurface24bitFastInplace(),
        TestShaderBlur5x5Array24Inplace(),
        TestShaderWave24bitInplace(),
        TestShaderSwirl24bitInplace(),
        TestShaderSwirl24bitInplace1(),
        TestShaderPlasmaInplace(),
        TestShaderPlasma24bitInplace(),
        TestShaderBrightness24Inplace(),
        TestShaderSaturationArray24Inplace(),
        TestShaderBpf24Inplace(),
        TestShaderBloomEffectArray24(),
        TestShaderFisheye24Inplace(),
        TestShaderTvScanlineInplace(),
        TESTHeatmapSurface24ConvInplace(),
        TestPredatorVisionMode(),
        TestShaderBloodInplace(),
        TestShaderSharpenFilterInplace(),
        TestshaderFireEffect(),
        TestshaderFireEffect1(),
        TestLateralDampeningEffect(),
        TestDampeningEffect(),
        TestMirroringInplace(),
        TestShaderCloudEffect(),
        TestTunnelRender32(),
        TestShaderRipple(),
        TestRgbSplit(),
        TestShaderHorizontalGlitch24Inplace(),
        TestShaderHeatwave24VerticalInplace()
    ])

    unittest.TextTestRunner().run(suite)
    sys.exit(0)


if __name__ == '__main__':
    run_testsuite()
    sys.exit(0)
