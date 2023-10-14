""""
TEST LIBRARY shader
"""
import ctypes
import math
import sys
import unittest
import os
import time
from random import uniform, randint, random


_CUPY = True
try:
    import cupy
except ImportError:
    _CUPY = False


import cython
import numpy
from numpy import uint8, asarray
from pygame.freetype import STYLE_NORMAL
from pygame.surfarray import make_surface, pixels3d
from pygame.transform import scale, smoothscale

from PygameShader.misc import create_horizontal_gradient_1d

import pygame
from pygame import BLEND_RGB_ADD, RLEACCEL, BLEND_RGB_MAX, BLEND_RGB_MIN, BLEND_RGB_SUB, \
    BLEND_RGB_MULT, freetype

import PygameShader
from PygameShader import rgb_to_bgr, rgb_to_brg, \
    greyscale, sepia, \
    color_reduction, sobel, \
    sobel_fast, invert, \
    hsl_effect, hsl_fast, rgb_to_hsl_model, \
    hsl_to_rgb_model, blur, wave, \
    swirl, swirl_inplace, plasma_config, \
    brightness, saturation, bpf, \
    bloom, fisheye, fisheye_footprint, \
    tv_scan, heatmap, predator_vision, \
    blood, sharpen, fire_effect, custom_map, rgb_to_int, \
    dampening, lateral_dampening, mirroring, cloud_effect, \
    tunnel_render32, tunnel_modeling32, ripple, rgb_split, \
    heatwave_vertical, horizontal_glitch, plasma, \
    area24_c, stack_buffer_c, \
    cartoon, blend, dirt_lens, area24_cc, tunnel_modeling24, tunnel_render24

if _CUPY:
    try:
        from PygameShader.shader_gpu import area24_gpu, ripple_effect_gpu, block_grid
    except:
        _CUPY = False

PROJECT_PATH = list(PygameShader.__path__)
os.chdir(PROJECT_PATH[0] + "/tests")

WIDTH = 800
HEIGHT = 600
# SCREEN = pygame.display.set_mode((WIDTH, HEIGHT), pygame.FULLSCREEN, pygame.OPENGL, vsync=True)
SCREEN = pygame.display.set_mode((WIDTH, HEIGHT), vsync=True)
SCREEN.convert(32, RLEACCEL)
SCREEN.set_alpha(None)



shader_list =\
    {
        "rgb_to_bgr"                 : "RGB to BGR",
        "rgb_to_brg"                 : "RGB to BRG",
        "greyscale"     : "grayscale Luminosity",
        "sepia"                    : "Sepia",
        # "median"            : "Median",
        # "median_grayscale"  : "Median grayscale",
        "color_reduction"          : "Color reduction",
        "sobel"                    : "Sobel",
        "sobel_fast"               : "Sobel fast",
        "invert"       : "Invert",
        "hsl_effect"           : "HSL",
        "hsl_fast"      : "HSL fast",
        "blur"            : "Blur",
        "wave"                  : "Wave",
        "swirl"                 : "Swirl",
        "swirl2"                : "Swirl1",
        "plasma_config"                : "Plasma",
        "brightness"               : "Brightness",
        "saturation"         : "Saturation",
        "bpf"                      : "Bright pass filter",
        "bloom"               : "Bloom",
        "fisheye"                  : "Fisheye",
        "tv_scan"                : "TV",
        "heatmap"            : "Heatmap",
        "predator_vision"                      : "Predator Vision",
        "blood"                      : "Blood",
        "sharpen"             : "Sharpen",
        "fire_effect"                        : "Fire Effect 1",
        "dampening"                          : "Dampening",
        "lateral_dampening"                  : "Lateral dampening",
        "mirroring"                         : "Mirror image",
        "cloud_effect"                       : "Shader cloud",
        "tunnel_render32"                           : "Tunnel",
        "ripple"                             : "Water Ripple",
        "rgb_split"                  : "RGB split",
        "heatwave_vertical"        : "Heatwave",
        "horizontal_glitch"        : "Horizontal glitch",
        "plasma"                             : "Plasma 2"
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

    if shader_name in ("hsl_effect",
                       "hsl_fast"):
        hsl_rotation = True
        v = 0.001
        hsl_value = 0.0

    angle = 0

    while game:

        if hsl_rotation:

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
    Test rgb_to_bgr
    """

    # pylint: disable=too-many-statements
    @staticmethod
    def runTest() -> None:
        """

        :return:  void
        """
        display_shader("rgb_to_bgr", timer = 5, flag=0)


class TestShaderRgbToBrgInplace(unittest.TestCase):
    """
    Test rgb_to_brg
    """

    # pylint: disable=too-many-statements
    @staticmethod
    def runTest() -> None:
        """

        :return:  void
        """
        display_shader("rgb_to_brg", timer = 5, flag=0)


class TestShaderGreyscaleLuminosity24Inplace(unittest.TestCase):
    """
    Test greyscale
    """

    # pylint: disable=too-many-statements
    @staticmethod
    def runTest() -> None:
        """

        :return:  void
        """
        display_shader("greyscale", timer = 5, flag=0)


class TestShaderSepia24Inplace(unittest.TestCase):
    """
    Test sepia
    """

    # pylint: disable=too-many-statements
    @staticmethod
    def runTest() -> None:
        """

        :return:  void
        """
        display_shader("sepia", timer = 5, flag=0)

#
# class TestShaderMedianFilter24Inplace(unittest.TestCase):
#     """
#     Test median
#     """
#
#     # pylint: disable=too-many-statements
#     @staticmethod
#     def runTest() -> None:
#         """
#
#         :return:  void
#         """
#         display_shader("median", timer = 5, flag=0)

#
# class TestShaderMedianGrayscaleFilter24Inplace(unittest.TestCase):
#     """
#     Test median_grayscale
#     """
#
#     # pylint: disable=too-many-statements
#     @staticmethod
#     def runTest() -> None:
#         """
#
#         :return:  void
#         """
#         display_shader("median_grayscale", timer = 5, flag=0)

#
# class TestShaderMedianFilter24AvgInplace(unittest.TestCase):
#     """
#     Test median_avg
#     """
#
#     # pylint: disable=too-many-statements
#     @staticmethod
#     def runTest() -> None:
#         """
#
#         :return:  void
#         """
#         display_shader("median_avg", timer = 5, flag=0)


class TestShaderColorReduction24Inplace(unittest.TestCase):
    """
    Test color_reduction
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

        pygame.display.set_caption("Test color_reduction")

        FRAME = 0
        CLOCK = pygame.time.Clock()
        GAME = True

        COLORS = [64, 32, 24, 16, 8, 4, 2]
        INDEX = 0
        t = time.time()
        while GAME:

            pygame.event.pump()
            for event in pygame.event.get():

                keys = pygame.key.get_pressed()

                if keys[pygame.K_ESCAPE]:
                    GAME = False

            color_reduction(image, color_=COLORS[ INDEX ])
            # SCREEN.fill((0, 0, 0))
            SCREEN.blit(image, (0, 0))
            pygame.display.flip()
            CLOCK.tick()
            FRAME += 1
            pygame.display.set_caption(
                "Test color_reduction %s fps "
                "(%sx%s) %s colors" % (round(CLOCK.get_fps(), 2), WIDTH, HEIGHT,
                                       COLORS[INDEX] ** 2 * 2))

            if FRAME % 250 == 0:
                INDEX += 1
                if INDEX >= len(COLORS):
                    INDEX = 0

            if time.time() - t > 10:
                break


class TestShaderSobel24Inplace(unittest.TestCase):
    """
    Test sobel
    """

    # pylint: disable=too-many-statements
    @staticmethod
    def runTest() -> None:
        """

        :return:  void
        """
        display_shader("sobel")


class TestShaderSobel24FastInplace(unittest.TestCase):
    """
    Test sobel_fast
    """

    # pylint: disable=too-many-statements
    @staticmethod
    def runTest() -> None:
        """

        :return:  void
        """
        display_shader("sobel_fast", timer = 5, flag=0)


class TestShaderInvertSurface24bitInplace(unittest.TestCase):
    """
    Test invert
    """

    # pylint: disable=too-many-statements
    @staticmethod
    def runTest() -> None:
        """

        :return:  void
        """
        display_shader("invert", timer = 5, flag=0)


class TestShaderHslSurface24bitInplace(unittest.TestCase):
    """
    Test hsl
    """

    # pylint: disable=too-many-statements
    @staticmethod
    def runTest() -> None:
        """

        :return:  void
        """
        display_shader("hsl_effect", timer = 10, flag=0, hsl_rotation=True)


class TestShaderHslSurface24bitFastInplace(unittest.TestCase):
    """
    Test hsl_fast
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

        pygame.display.set_caption("Test hsl_fast")

        frame = 0
        clock = pygame.time.Clock()
        game = True

        v = 0.001
        hsl_value = 0.0


        hsl2rgb_model = hsl_to_rgb_model()
        rgb2hsl_model = rgb_to_hsl_model()

        t = time.time()

        while game:
            hsl_fast(
                image,
                hsl_value,
                rgb_to_hsl_=rgb2hsl_model,
                hsl_to_rgb_=hsl2rgb_model)

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
                "Test hsl_fast %s fps "
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
    Test blur
    """

    # pylint: disable=too-many-statements
    @staticmethod
    def runTest() -> None:
        """

        :return:  void
        """

        display_shader("blur")


class TestShaderWave24bitInplace(unittest.TestCase):
    """
    Test wave
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
        pygame.display.set_caption("Test wave")

        frame = 0
        clock = pygame.time.Clock()
        game = True
        angle = 0
        t = time.time()
        while game:
            wave(image, angle * math.pi / 180, 10)

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
                "Test wave %s fps "
                "(%sx%s)" % (round(clock.get_fps(), 2), WIDTH, HEIGHT))
            image = background.copy()
            angle += 5
            angle %= 360
            if time.time() - t > 5:
                break


class TestShaderSwirl24bitInplace(unittest.TestCase):
    """
    Test swirl
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

        pygame.display.set_caption("Test swirl")

        frame = 0
        clock = pygame.time.Clock()
        game = True
        angle = 0
        t = time.time()
        while game:
            swirl(image, angle)

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
                "Test swirl %s fps "
                "(%sx%s)" % (round(clock.get_fps(), 2), WIDTH, HEIGHT))
            image = background.copy()
            angle += 1
            # angle %= 1
            if time.time() - t > 5:

                break


class TestShaderSwirl24bitInplace1(unittest.TestCase):
    """
    Test swirl2
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
        pygame.display.set_caption("Test swirl2")

        FRAME = 0
        CLOCK = pygame.time.Clock()
        GAME = True
        ANGLE = 0
        t = time.time()
        while GAME:
            swirl_inplace(image, ANGLE)

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
                "Test swirl2 %s fps "
                "(%sx%s)" % (round(CLOCK.get_fps(), 2), WIDTH, HEIGHT))
            image = background.copy()
            ANGLE += 1

            if time.time() - t > 5:
                break


class TestShaderPlasmaInplace(unittest.TestCase):
    """
    Test plasma
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
        pygame.display.set_caption("Test plasma")

        FRAME = 0.0
        CLOCK = pygame.time.Clock()
        GAME = True

        arr = numpy.array([0, 1,  # violet
                           0, 1,  # blue
                           0, 1,  # green
                           2, 619,  # yellow
                           620, 650,  # orange
                           651, 660],  # red
                          numpy.int32)

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
            plasma(im, FRAME, heatmap_rescale)
            im.blit(image, (0, 0), special_flags=BLEND_RGB_ADD)
            SCREEN.blit(im, (0, 0))
            pygame.display.flip()
            CLOCK.tick()
            FRAME += 0.2
            pygame.display.set_caption(
                "Test plasma %s fps "
                "(%sx%s)" % (round(CLOCK.get_fps(), 2), WIDTH, HEIGHT))
            if time.time() - t > 5:
                break


class TestShaderPlasma24bitInplace(unittest.TestCase):
    """
    Test plasma_config
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
        pygame.display.set_caption("Test plasma_config")

        FRAME = 0.0
        CLOCK = pygame.time.Clock()
        GAME = True
        t = time.time()
        while GAME:
            plasma_config(image, FRAME, 0.5, 0.12, 0.38, 0.1)

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
                "Test plasma_config %s fps "
                "(%sx%s)" % (round(CLOCK.get_fps(), 2), WIDTH, HEIGHT))
            image = background.copy()
            if time.time() - t > 5:
                break


class TestShaderBrightness24Inplace(unittest.TestCase):
    """
    Test brightness
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
        pygame.display.set_caption("Test brightness")

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

            brightness(image, BRIGHT)
            # SCREEN.fill((0, 0, 0))
            SCREEN.blit(image, (0, 0))
            pygame.display.flip()
            CLOCK.tick()
            FRAME += 1
            pygame.display.set_caption(
                "Test brightness %s fps "
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
    Test saturation
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

        pygame.display.set_caption("Test saturation")

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

            saturation(image, SATURATION)
            # SCREEN.fill((0, 0, 0))
            SCREEN.blit(image, (0, 0))
            pygame.display.flip()
            CLOCK.tick()
            FRAME += 1
            pygame.display.set_caption(
                "Test saturation %s fps "
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
    Test bpf
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

        pygame.display.set_caption("Test bpf")

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

            bpf(image, BPF)
            # SCREEN.fill((0, 0, 0))
            SCREEN.blit(image, (0, 0))
            pygame.display.flip()
            CLOCK.tick()
            FRAME += 1
            pygame.display.set_caption(
                "Test bpf %s fps "
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
    Test bloom
    """

    # pylint: disable=too-many-statements
    @staticmethod
    def runTest() -> None:
        """

        :return:  void
        """
        background = pygame.image.load("../Assets/city.jpg").convert()
        background = pygame.transform.smoothscale(background, (WIDTH, HEIGHT))
        # background.convert(32, RLEACCEL)
        image = background.copy()
        pygame.display.set_caption("Test bloom")

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

            bloom(image, BPF, fast_=True)
            # SCREEN.fill((0, 0, 0))
            SCREEN.blit(image, (0, 0))
            pygame.display.flip()
            CLOCK.tick()
            FRAME += 1
            pygame.display.set_caption(
                "Test bloom %s fps "
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
    Test fisheye
    """

    # pylint: disable=too-many-statements
    @staticmethod
    def runTest() -> None:
        """

        :return:  void
        """
        background = pygame.image.load("../Assets/background.jpg").convert()
        background = pygame.transform.smoothscale(background, (WIDTH, HEIGHT))
        background.convert()
        image = background.copy()

        pygame.display.set_caption("Test fisheye")

        FRAME = 0
        CLOCK = pygame.time.Clock()
        GAME = True

        fisheye_model = fisheye_footprint(WIDTH + 1, HEIGHT + 1, WIDTH >> 1, HEIGHT >> 1)
        t = time.time()
        while GAME:

            pygame.event.pump()
            for event in pygame.event.get():

                keys = pygame.key.get_pressed()

                if keys[pygame.K_ESCAPE]:
                    GAME = False
                    break

            fisheye(image, fisheye_model)

            p = image.get_at((0, 0))
            image.set_colorkey(p)

            SCREEN.blit(image, (0, 0))
            pygame.display.flip()
            CLOCK.tick()
            FRAME += 1
            pygame.display.set_caption(
                "Test fisheye %s fps "
                "(%sx%s)" % (round(CLOCK.get_fps(), 2), WIDTH, HEIGHT))

            image = background.copy()
            if time.time() - t > 5:
                break


class TestShaderTvScanlineInplace(unittest.TestCase):
    """
    Test tv_scan
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
        pygame.display.set_caption("Test tv_scan")

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

            tv_scan(image, FRAME + 1)

            SCREEN.blit(image, (0, 0))
            pygame.display.flip()
            CLOCK.tick()
            FRAME += 1
            FRAME %= 10

            pygame.display.set_caption(
                "Test tv_scan %s fps "
                "(%sx%s)" % (round(CLOCK.get_fps(), 2), WIDTH, HEIGHT))
            image = background.copy()
            if time.time() - t > 5:
                break


class TESTHeatmapSurface24ConvInplace(unittest.TestCase):
    """
    Test heatmap
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
        pygame.display.set_caption("Test heatmap")

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

            heatmap(image, True)
            # SCREEN.fill((0, 0, 0))
            SCREEN.blit(image, (0, 0))
            pygame.display.flip()
            CLOCK.tick()
            FRAME += 1
            pygame.display.set_caption(
                "Test heatmap %s fps "
                "(%sx%s)" % (round(CLOCK.get_fps(), 2), WIDTH, HEIGHT))
            image = background.copy()
            if time.time() - t > 5:
                break


class TestPredatorVisionMode(unittest.TestCase):
    """
    Test predator_vision
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

        BACKGROUND1 = pygame.image.load("../Assets/city.jpg").convert()
        BACKGROUND1 = pygame.transform.smoothscale(BACKGROUND1, (WIDTH, HEIGHT))
        image1 = BACKGROUND1.copy()

        pygame.display.set_caption("Test predator_vision")

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

            if FRAME < 30:
                surface_ = predator_vision(
                    image.copy(), sobel_threshold=80, bpf_threshold=0,
                    bloom_threshold=0, inv_colormap=True, fast=True)
            else:
                surface_ = predator_vision(
                    image1.copy(), sobel_threshold=10, bpf_threshold=0,
                    bloom_threshold=50, inv_colormap=True, fast=True)

            # SCREEN.fill((0, 0, 0))
            SCREEN.blit(surface_, (0, 0))
            pygame.display.flip()
            CLOCK.tick()
            FRAME += 1
            pygame.display.set_caption(
                "Test predator_vision %s fps "
                "(%sx%s)" % (round(CLOCK.get_fps(), 2), WIDTH, HEIGHT))

            if time.time() - t > 5:
                break


class TestShaderBloodInplace(unittest.TestCase):
    """
    Test blood
    """

    # pylint: disable=too-many-statements
    @staticmethod
    def runTest() -> None:
        """

        :return:  void
        """
        background = pygame.image.load("../Assets/Aliens.jpg").convert()
        background = pygame.transform.smoothscale(background, (WIDTH, HEIGHT))
        background.convert(32, RLEACCEL)
        image = background.copy()
        pygame.display.set_caption("Test blood")

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

            blood(image, BLOOD_MASK, PERCENTAGE)
            # SCREEN.fill((0, 0, 0))
            SCREEN.blit(image, (0, 0))
            pygame.display.flip()
            CLOCK.tick()
            FRAME += 1
            pygame.display.set_caption(
                "Test blood %s fps "
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
    Test sharpen
    """

    # pylint: disable=too-many-statements
    @staticmethod
    def runTest() -> None:
        """

        :return:  void
        """
        background = pygame.image.load("../Assets/city.jpg").convert()
        background = pygame.transform.smoothscale(background, (WIDTH, HEIGHT))
        background.convert(32, RLEACCEL)
        image = background.copy()
        pygame.display.set_caption("Test sharpen")

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

            sharpen(image)
            SCREEN.blit(image, (0, 0))
            pygame.display.flip()
            CLOCK.tick()
            FRAME += 1
            pygame.display.set_caption(
                "Test sharpen %s fps "
                "(%sx%s)" % (round(CLOCK.get_fps(), 2), WIDTH, HEIGHT))
            image = background.copy()
            if time.time() - t > 5:
                break


class TestshaderFireEffect(unittest.TestCase):
    """
    Test fire_effect with adjust_palette_=True
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
        pygame.display.set_caption("Test fire_effect")

        FRAME = 0
        CLOCK = pygame.time.Clock()
        GAME = True

        arr = numpy.array([0, 1,  # violet
                           0, 1,  # blue
                           0, 1,  # green
                           0, 619,  # yellow
                           620, 650,  # orange
                           651, 680],  # red
                          numpy.int32)

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

            surface_ = fire_effect(
                WIDTH, HEIGHT, 3.97 + uniform(0.002, 0.008),
                heatmap_rescale,
                FIRE_ARRAY,
                reduce_factor_=3, fire_intensity_=32,
                smooth_=True, bloom_=True, fast_bloom_=False,
                bpf_threshold_=128, low_=1, high_=WIDTH, brightness_=True,
                brightness_intensity_=0.1, adjust_palette_=True,
                hsl_=(10, 80, 1.8), transpose_=False, border_=False,
                surface_=None)

            SCREEN.blit(surface_, (0, 0), special_flags=pygame.BLEND_RGBA_MAX)

            pygame.display.flip()
            CLOCK.tick()
            FRAME += 1
            pygame.display.set_caption(
                "Test fire_effect %s fps "
                "(%sx%s)" % (round(CLOCK.get_fps(), 2), WIDTH, HEIGHT))
            image = BACKGROUND.copy()
            if time.time() - t > 5:
                break


class TestshaderFireEffect1(unittest.TestCase):
    """
    Test fire_effect
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
        pygame.display.set_caption("Test fire_effect")

        FRAME = 0
        CLOCK = pygame.time.Clock()
        GAME = True

        arr = numpy.array([0, 1,  # violet
                           0, 1,  # blue
                           0, 1,  # green
                           2, 619,  # yellow
                           620, 650,  # orange
                           651, 660],  # red
                          numpy.int32)

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
            surface_ = fire_effect(
                WIDTH, HEIGHT, 3.97 + uniform(0.002, 0.008),
                heatmap_rescale,
                FIRE_ARRAY,
                reduce_factor_=3, fire_intensity_=16,
                smooth_=True, bloom_=True, fast_bloom_=False,
                bpf_threshold_=1, low_=1, high_=WIDTH, brightness_=True,
                brightness_intensity_=0.15, adjust_palette_=False,
                hsl_=(10, 80, 1.8), transpose_=False, border_=False,
                surface_=None)

            SCREEN.blit(surface_, (0, 0), special_flags=BLEND_RGB_ADD)

            pygame.display.flip()
            CLOCK.tick()
            FRAME += 1
            pygame.display.set_caption(
                "Test fire_effect %s fps "
                "(%sx%s)" % (round(CLOCK.get_fps(), 2), WIDTH, HEIGHT))
            image = BACKGROUND.copy()
            if time.time() - t > 5:
                break


class TestLateralDampeningEffect(unittest.TestCase):
    """
    Test lateral_dampening
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
        pygame.display.set_caption("Test lateral_dampening")

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
            # surf, xx, yy = dampening(BACKGROUND, FRAME, WIDTH, HEIGHT, amplitude_=60,
            #                                 duration_=100, freq_=0.8)
            # SCREEN.blit(surf, (xx, yy))

            tm = lateral_dampening(FRAME, amplitude_=10.0, duration_=10, freq_=50.0)
            SCREEN.blit(BACKGROUND, (tm, 0), special_flags=0)

            pygame.display.flip()
            CLOCK.tick()
            FRAME += 1
            pygame.display.set_caption(
                "Test lateral_dampening %s fps "
                "(%sx%s)" % (round(CLOCK.get_fps(), 2), WIDTH, HEIGHT))
            image = BACKGROUND.copy()
            if time.time() - t > 5:
                break


class TestDampeningEffect(unittest.TestCase):
    """
    Test dampening
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
        pygame.display.set_caption("Test dampening")

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
            surf, xx, yy = dampening(
                BACKGROUND,
                FRAME,
                WIDTH,
                HEIGHT,
                amplitude_=15.0,
                duration_=10,
                freq_=25.0)
            # surf.convert(32, RLEACCEL)
            SCREEN.blit(surf, (xx, yy), special_flags=0)

            pygame.display.flip()
            CLOCK.tick()
            FRAME += 1
            pygame.display.set_caption(
                "Test dampening %s fps "
                "(%sx%s)" % (round(CLOCK.get_fps(), 2), WIDTH, HEIGHT))
            image = BACKGROUND.copy()
            if time.time() - t > 5:
                break


class TestMirroringInplace(unittest.TestCase):
    """
    Test mirroring
    """

    # pylint: disable=too-many-statements
    @staticmethod
    def runTest() -> None:
        """

        :return:  void
        """
        BACKGROUND = pygame.image.load("../Assets/city.jpg")
        BACKGROUND = pygame.transform.smoothscale(BACKGROUND, (WIDTH, HEIGHT))
        image = BACKGROUND.copy()
        pygame.display.set_caption("Test mirroring")

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

            mirroring(image)

            SCREEN.blit(image, (0, 0))
            image = BACKGROUND.copy()
            pygame.display.flip()
            CLOCK.tick()
            FRAME += 1
            pygame.display.set_caption(
                "Test mirroring %s fps "
                "(%sx%s)" % (round(CLOCK.get_fps(), 2), WIDTH, HEIGHT))
            if time.time() - t > 5:
                break


class TestShaderCloudEffect(unittest.TestCase):
    """
    Test cloud_effect
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
                          numpy.int32)

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

            surface_ = cloud_effect(
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
                "Test cloud_effect %s fps "
                "(%sx%s)" % (round(CLOCK.get_fps(), 2), WIDTH, HEIGHT))
            image = BACKGROUND.copy()
            if time.time() - t > 5:
                break


class TestTunnelRender24(unittest.TestCase):
    """
    Test tunnel_render24
    """

    # pylint: disable=too-many-statements
    @staticmethod
    def runTest() -> None:
        """

        :return:  void
        """
        WIDTH = 800
        HEIGHT = 800

        BCK1 =  pygame.image.load("../Assets/space2.jpg").convert()
        BCK1 = pygame.transform.smoothscale(BCK1, (WIDTH, HEIGHT))
        BACKGROUND = pygame.image.load("../Assets/space1.jpg")
        BACKGROUND = pygame.transform.smoothscale(BACKGROUND, (WIDTH, HEIGHT))

        FRAME = 0
        CLOCK = pygame.time.Clock()
        GAME = True

        distances, angles, shades, scr_data = tunnel_modeling24(WIDTH, HEIGHT, BACKGROUND)
        dest_array = numpy.empty((WIDTH * HEIGHT * 4), numpy.uint8)
        t= time.time()
        while GAME:

            pygame.event.pump()
            for event in pygame.event.get():

                keys = pygame.key.get_pressed()

                if keys[pygame.K_ESCAPE]:
                    GAME = False
                    break

            SCREEN.blit(BCK1, (0, 0))
            # SCREEN.fill((0, 0, 0, 0))
            surface_ = tunnel_render24(
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
            # pygame.time.delay(15)
            SCREEN.blit(surface_, (0, 0), special_flags=pygame.BLEND_RGB_ADD)

            pygame.display.flip()
            CLOCK.tick()
            FRAME += 1
            pygame.display.set_caption(
                "Test tunnel %s fps "
                "(%sx%s)" % (round(CLOCK.get_fps(), 2), WIDTH, HEIGHT))
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

        BCK1 =  pygame.image.load("../Assets/space2.jpg").convert()
        BCK1 = pygame.transform.smoothscale(BCK1, (WIDTH, HEIGHT))
        BACKGROUND = pygame.image.load("../Assets/space2_seamless_alpha.jpg")
        BACKGROUND = pygame.transform.smoothscale(BACKGROUND, (WIDTH, HEIGHT))

        FRAME = 0
        CLOCK = pygame.time.Clock()
        GAME = True


        distances, angles, shades, scr_data = tunnel_modeling32(WIDTH, HEIGHT, BACKGROUND)
        dest_array = numpy.empty((WIDTH * HEIGHT * 4), numpy.uint8)
        t= time.time()
        while GAME:

            pygame.event.pump()
            for event in pygame.event.get():

                keys = pygame.key.get_pressed()

                if keys[pygame.K_ESCAPE]:
                    GAME = False
                    break

            SCREEN.blit(BCK1, (0, 0))
            # SCREEN.fill((0, 0, 0, 0))
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
            # pygame.time.delay(15)
            SCREEN.blit(surface_, (0, 0), special_flags=pygame.BLEND_RGB_ADD)

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
    Test ripple
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

            previous, current = ripple(W2, H2, previous, current, array_)

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


def ripple_2(previous, current):

    val = (cupy.roll(previous, 1, axis=0).astype(dtype=cupy.float32) +
           cupy.roll(previous, -1, axis=0).astype(dtype=cupy.float32) +
           cupy.roll(previous, 1, axis=1).astype(dtype=cupy.float32) +
           cupy.roll(previous, -1, axis=1).astype(dtype=cupy.float32)) / float(2.0) - current.astype(dtype=cupy.float32)

    current = (val * 0.98).astype(dtype=cupy.float32)

    return previous, current

pygame.font.init()
FONT = pygame.font.SysFont("Arial", 15)


def show_fps(screen_, fps_, avg_) -> None:
    """ Show framerate in upper left corner """

    fps = str(f"fps:{fps_:.3f}")
    av = sum(avg_)/len(avg_) if len(avg_) > 0 else 0

    fps_text = FONT.render(fps, 1, pygame.Color("coral"))
    screen_.blit(fps_text, (10, 0))

    if av != 0:
        av = str(f"avg:{av:.3f}")
        avg_text = FONT.render(av, 1, pygame.Color("coral"))
        screen_.blit(avg_text, (80, 0))
    if len(avg_) > 200:
        avg_ = avg_[200:]
    return avg_


class TestShaderRippleGpu(unittest.TestCase):
    """
    Test ripple
    """

    # pylint: disable=too-many-statements
    @staticmethod
    def runTest() -> None:
        """

        :return:  void
        """
        if not _CUPY:
            return

        width = 900
        height = 600

        texture = pygame.image.load('../Assets/background2.jpg').convert()
        texture = pygame.transform.smoothscale(texture, (width, height))
        texture.set_colorkey((0, 0, 0, 0), pygame.RLEACCEL)
        texture.set_alpha(None)

        BACKGROUND = pygame.image.load("../Assets/background2.jpg").convert()
        BACKGROUND = pygame.transform.smoothscale(BACKGROUND, (width, height))
        BACKGROUND.set_colorkey((0, 0, 0, 0), pygame.RLEACCEL)
        BACKGROUND_COPY = BACKGROUND.copy()

        SCREENRECT = pygame.Rect(0, 0, width, height)
        pygame.display.init()
        # SCREEN = pygame.display.set_mode(SCREENRECT.size, pygame.DOUBLEBUF, 32)


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
        avg = []
        while not STOP_GAME:

            pygame.display.set_caption("DEMO ripple effect GPU : FPS %s " % round(cget_fps(), 2))

            event_pump()

            keys = get_key()
            # SCREEN.fill((0, 0, 0, 0))
            for event in event_get():

                if keys[pygame.K_ESCAPE]:
                    STOP_GAME = True
                    break

                if event.type == pygame.MOUSEMOTION:
                    mouse_pos = pygame.math.Vector2(get_pos())

                    previous[int(mouse_pos.x % width), int(mouse_pos.y % height)] = 2 ** 12

            rnd = randint(0, 1000)

            if rnd > 820:

                previous[randint(0, width - 2), randint(0, height - 2)] = 2 ** 10

            # METHOD 1
            # previous, current = ripple_2(current, previous)
            # array[:, :, 0:3] = current.reshape((width, height, 1))
            #
            # surf = pygame.image.frombuffer(array.transpose(1, 0, 2).tobytes(), (width, height), "RGB")
            # BACKGROUND.blit(surf, (0, 0), special_flags=pygame.BLEND_RGBA_ADD)
            # SCREEN.blit(BACKGROUND, (0, 0))
            # BACKGROUND = BACKGROUND_COPY.copy()

            SCREEN.fill((0, 0, 0))


            # METHOD 2
            previous, current = ripple_effect_gpu(
                grid, block, width, height,
                current, previous, texture_array, back_array)
            surf = pygame.image.frombuffer(
                back_array.transpose(1, 0, 2).tobytes(), (width, height), "RGB")
            SCREEN.blit(surf, (0, 0))

            clock.tick(8000)
            t = clock.get_fps()
            avg.append(t)
            show_fps(SCREEN, t, avg)
            FRAME += 1
            flip()
        pygame.mixer.stop()




class TestRgbSplit(unittest.TestCase):
    """
    Test rgb_split
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

            rgb_split(image, 15)

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
    Test horizontal_glitch
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

            horizontal_glitch(image, 1, 0.08, FRAME % 100)

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
    Test heatwave_vertical
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
            heatwave_vertical(
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


class TestShaderCartoon(unittest.TestCase):
    """
    Test cartoon
    """

    # pylint: disable=too-many-statements
    @staticmethod
    def runTest() -> None:
        """

        :return:  void
        """

        # Load the background image
        BACKGROUND = pygame.image.load("../Assets/background.jpg").convert()
        BACKGROUND = pygame.transform.smoothscale(BACKGROUND, (WIDTH, HEIGHT))

        image = BACKGROUND.copy()
        pygame.display.set_caption("demo cartoon effect")

        FRAME = 0
        CLOCK = pygame.time.Clock()
        GAME = True
        t = time.time()
        while GAME:

            SCREEN.fill((0, 0, 0))
            pygame.event.pump()
            for event in pygame.event.get():

                keys = pygame.key.get_pressed()

                if keys[pygame.K_ESCAPE]:
                    GAME = False
                    break

            surf = cartoon(image, median_kernel_=2, color_=8, sobel_threshold_=64)

            SCREEN.blit(surf, (0, 0), special_flags=0)

            pygame.display.flip()
            CLOCK.tick()
            FRAME += 1

            pygame.display.set_caption(
                "Test shader cartoon %s fps "
                "(%sx%s)" % (round(CLOCK.get_fps(), 2), WIDTH, HEIGHT))

            image = BACKGROUND.copy()
            if time.time() - t > 100006:
                break


class TestShaderBlend24(unittest.TestCase):
    """
    Test blend
    """

    # pylint: disable=too-many-statements
    @staticmethod
    def runTest() -> None:
        """

        :return:  void
        """

        # Load the background image
        BACKGROUND = pygame.image.load("../Assets/background.jpg").convert()
        BACKGROUND = pygame.transform.smoothscale(BACKGROUND, (WIDTH, HEIGHT))

        DESTINATION = pygame.image.load("../Assets/Aliens.jpg").convert()
        DESTINATION = pygame.transform.smoothscale(DESTINATION, (WIDTH, HEIGHT))
        DESTINATION_ARRAY = pixels3d(DESTINATION)
        assert BACKGROUND.get_size() == DESTINATION.get_size()

        pygame.display.set_caption("demo blend inplace")

        FRAME = 0
        CLOCK = pygame.time.Clock()
        GAME = True
        VALUE = 0
        V = +0.2

        while GAME:

            pygame.event.pump()
            for event in pygame.event.get():

                keys = pygame.key.get_pressed()

                if keys[pygame.K_ESCAPE]:
                    GAME = False
                    break

            transition = blend(
                source_=BACKGROUND, destination_=DESTINATION_ARRAY, percentage_=VALUE)

            SCREEN.blit(transition, (0, 0))


            pygame.display.flip()
            CLOCK.tick()
            FRAME += 1

            pygame.display.set_caption(
                "Demo blend effect/transition %s percent; %s fps"
                "(%sx%s)" % (round(VALUE, 2), round(CLOCK.get_fps(), 2), WIDTH, HEIGHT))
            VALUE += V

            if VALUE >= 100:
                VALUE = 100
                V = -0.2
            if VALUE <= 0:
                VALUE = 0
                V = 0.2


class TestDirtLens(unittest.TestCase):
    """
    Test Dirt Lens
    """

    # pylint: disable=too-many-statements
    @staticmethod
    def runTest() -> None:
        """

        :return:  void
        """
        IMAGE = pygame.image.load("../Assets/Aliens.jpg").convert()
        IMAGE = pygame.transform.smoothscale(IMAGE, (WIDTH, HEIGHT))
        IMAGE_COPY = IMAGE.copy()

        dirt_lens_image = [
            pygame.image.load("../Assets/Bokeh__Lens_Dirt_9.jpg").convert(),
            pygame.image.load("../Assets/Bokeh__Lens_Dirt_38.jpg").convert(),
            pygame.image.load("../Assets/Bokeh__Lens_Dirt_46.jpg").convert(),
            pygame.image.load("../Assets/Bokeh__Lens_Dirt_50.jpg").convert(),
            pygame.image.load("../Assets/Bokeh__Lens_Dirt_54.jpg").convert(),
            pygame.image.load("../Assets/Bokeh__Lens_Dirt_67.jpg").convert()
        ]

        lens = dirt_lens_image[0]
        lens = pygame.transform.scale(lens, (WIDTH, HEIGHT)).convert()
        lens_copy = lens.copy()

        pygame.display.set_caption("demo Dirt Lens effect")

        FRAME = 0
        CLOCK = pygame.time.Clock()
        GAME = True
        VALUE = 0.2
        V = -0.005
        freetype.init(cache_size=64, resolution=72)
        FONT = freetype.SysFont('ARIALBLACK', size=18)
        while GAME:

            pygame.event.pump()
            for event in pygame.event.get():

                keys = pygame.key.get_pressed()

                if keys[pygame.K_ESCAPE]:
                    GAME = False
                    break

            dirt_lens(IMAGE, flag_=BLEND_RGB_ADD, lens_model_=lens, light_=VALUE)

            SCREEN.blit(IMAGE, (0, 0))
            image, rect = FONT.render(
                "FPS %s" % round(CLOCK.get_fps(), 2),
                fgcolor=(255, 255, 255, 255),
                style=STYLE_NORMAL,
                size=15)
            SCREEN.blit(image, (100, 100))
            pygame.display.flip()
            CLOCK.tick()
            FRAME += 1

            pygame.display.set_caption(
                "Demo Dirt Lens effect %s percent; %s fps"
                "(%sx%s)" % (round(VALUE, 2), round(CLOCK.get_fps(), 2), WIDTH, HEIGHT))
            VALUE += V

            if VALUE >= 0.2:
                VALUE = 0.2
                V = -0.005
            if VALUE <= -1.0:
                VALUE = -1.0
                V = 0.005

            IMAGE = IMAGE_COPY.copy()
            lens = lens_copy.copy()



class TestLight_CPU(unittest.TestCase):
    """
    Test Light
    """

    # pylint: disable=too-many-statements
    @staticmethod
    def runTest() -> None:
        """

        :return:  void
        """



        SCREENRECT = pygame.Rect(0, 0, 800, 600)
        pygame.display.init()
        SCREEN = pygame.display.set_mode(SCREENRECT.size, 32)
        pygame.init()
        background = pygame.image.load('../Assets/Aliens.jpg').convert()
        background.set_alpha(None)

        background = pygame.transform.smoothscale(background, (800, 800))
        background_rgb = pygame.surfarray.pixels3d(background)

        pygame.display.set_caption("demo light effect")

        light = pygame.image.load('../Assets/Radial8.png').convert_alpha()
        light = pygame.transform.smoothscale(light, (400, 400))
        lw, lh = light.get_size()
        lw2, lh2 = lw >> 1, lh >> 1

        lalpha = pygame.surfarray.pixels_alpha(light)

        c = numpy.array([128.0 / 255.0, 128.0 / 255.0, 200.0 / 255.0], numpy.float32, copy=False)
        MOUSE_POS = [0, 0]
        FRAME = 0
        CLOCK = pygame.time.Clock()
        GAME = True
        avg = []
        while GAME:

            pygame.event.pump()
            for event in pygame.event.get():
                if event.type == pygame.MOUSEMOTION:
                    MOUSE_POS = event.pos
                keys = pygame.key.get_pressed()

                if keys[pygame.K_ESCAPE]:
                    GAME = False
                    break

            # lit_surface, sw, sh = area24_cc(
            #     MOUSE_POS[0], MOUSE_POS[1], background_rgb, lalpha, intensity=3.0, color=c,
            #     smooth=False, saturation=False, sat_value=0.4, bloom=False)
            lit_surface, sw, sh = area24_cc(
                MOUSE_POS[ 0 ], MOUSE_POS[ 1 ], background_rgb, lalpha, intensity=5, color=c,
                smooth=False, saturation=False, sat_value=0.2, bloom=False, bloom_threshold=64
            )

            if sw < lw and MOUSE_POS[0] <= lw - lw2:
                xx = 0
            else:
                xx = MOUSE_POS[0] - lw2

            if sh < lh and MOUSE_POS[1] <= lh - lh2:
                yy = 0
            else:
                yy = MOUSE_POS[1] - lh2

            SCREEN.fill((0, 0, 0))
            SCREEN.blit(lit_surface, (xx, yy)) # , special_flags=pygame.BLEND_RGBA_ADD)

            CLOCK.tick(2000)
            t = CLOCK.get_fps()
            avg.append(t)
            show_fps(SCREEN, t, avg)
            pygame.display.flip()

            FRAME += 1

            pygame.display.set_caption(
                "Test light CPU %s fps"
                "(%sx%s)" % (round(CLOCK.get_fps(), 2), WIDTH, HEIGHT))




class TestLight_GPU(unittest.TestCase):
    """
    Test Light
    """

    # pylint: disable=too-many-statements
    @staticmethod
    def runTest() -> None:
        if not _CUPY:
            return

        """

        :return:  void
        """
        SCREENRECT = pygame.Rect(0, 0, 800, 600)
        pygame.display.init()
        SCREEN = pygame.display.set_mode(SCREENRECT.size,  pygame.DOUBLEBUF, 32)
        pygame.init()

        background = pygame.image.load('../Assets/Aliens.jpg').convert()
        background.set_alpha(None)

        background = pygame.transform.smoothscale(background, (800, 800))
        background_rgb = cupy.asarray(pygame.surfarray.pixels3d(background), dtype=cupy.uint8)
        background_rgb = background_rgb.transpose(1, 0, 2)
        w, h = background.get_size()

        back = background.copy()
        back.set_alpha(255)

        pygame.display.set_caption("demo GPU light effect")

        light = pygame.image.load('../Assets/Radial8.png').convert_alpha()
        light = pygame.transform.smoothscale(light, (600, 600))
        lalpha = cupy.asarray(pygame.surfarray.pixels_alpha(light), dtype=cupy.uint8)

        lw, lh = light.get_size()
        lw2, lh2 = lw >> 1, lh >> 1

        c = cupy.array([128.0 / 255.0, 128.0 / 255.0, 200.0 / 255.0], numpy.float32, copy=False)
        MOUSE_POS = [0, 0]
        FRAME = 0
        CLOCK = pygame.time.Clock()
        GAME = True
        avg = []
        while GAME:

            pygame.event.pump()
            for event in pygame.event.get():
                if event.type == pygame.MOUSEMOTION:
                    MOUSE_POS = event.pos
                keys = pygame.key.get_pressed()

                if keys[pygame.K_ESCAPE]:
                    GAME = False
                    break

            lit_surface, sw, sh = area24_gpu(
                MOUSE_POS[0], MOUSE_POS[1], background_rgb, lalpha, intensity=8.0, color=c)


            if sw < lw and MOUSE_POS[0] <= lw - lw2:
                xx = 0
            else:
                xx = MOUSE_POS[0] - lw2

            if sh < lh and MOUSE_POS[1] <= lh - lh2:
                yy = 0
            else:
                yy = MOUSE_POS[1] - lh2

            SCREEN.fill((0, 0, 0))
            SCREEN.blit(lit_surface, (xx, yy), special_flags=pygame.BLEND_RGBA_ADD)
            t = CLOCK.get_fps()
            avg.append(t)
            show_fps(SCREEN, t, avg)
            pygame.display.flip()
            CLOCK.tick()
            FRAME += 1

            pygame.display.set_caption(
                "Demo light GPU %s fps"
                "(%sx%s)" % (round(CLOCK.get_fps(), 2), WIDTH, HEIGHT))

from os import path
FULL_VIDEO = []
PATH_TEXTURE = '../Assets/'
VIDEO_FLAGS  = pygame.HWSURFACE | pygame.RLEACCEL
SCREEN_BIT_DEPTH = 32


def run_testsuite():
    """
    test suite

    :return: void
    """
    # TestShaderMedianFilter24Inplace(),
    # TestShaderMedianGrayscaleFilter24Inplace(),
    # TestShaderMedianFilter24AvgInplace(),

    suite = unittest.TestSuite()

    suite.addTests([
        ShaderRgbToBgrInplace(),
        TestShaderRgbToBrgInplace(),
        TestShaderGreyscaleLuminosity24Inplace(),
        TestShaderSepia24Inplace(),

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
        TestTunnelRender24(),
        TestTunnelRender32(),
        TestShaderRipple(),
        TestRgbSplit(),
        TestShaderHorizontalGlitch24Inplace(),
        TestShaderHeatwave24VerticalInplace(),
        TestShaderCartoon(),
        TestShaderBlend24(),
        TestDirtLens(),

    ])
    global _CUPY
    if _CUPY:
        suite.addTests([TestLight_GPU(), TestShaderRippleGpu()])

    unittest.TextTestRunner().run(suite)
    sys.exit(0)


if __name__ == '__main__':
    run_testsuite()
    sys.exit(0)

