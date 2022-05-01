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
from pygame.freetype import STYLE_NORMAL
from pygame.surfarray import make_surface
from pygame.transform import scale, smoothscale

from PygameShader.misc import create_horizontal_gradient_1d

import pygame
from pygame import BLEND_RGB_ADD, RLEACCEL, BLEND_RGB_MAX, BLEND_RGB_MIN, BLEND_RGB_SUB, \
    BLEND_RGB_MULT, freetype

from PygameShader.shader import rgb_to_bgr, rgb_to_brg, \
    greyscale, sepia, \
    median, median_grayscale, \
    median_avg, color_reduction, sobel, \
    sobel_fast, invert, \
    hsl_effect, hsl_fast, rgb_to_hsl_model, \
    hsl_to_rgb_model, blur, wave, \
    swirl, swirl2, plasma_config, \
    brightness, saturation, bpf, \
    bloom, fisheye, fisheye_footprint, \
    tv_scan, heatmap, predator_vision, \
    blood, sharpen, fire_effect, custom_map, rgb_to_int, \
    dampening, lateral_dampening, mirroring, cloud_effect, \
    tunnel_render32, tunnel_modeling32, ripple, rgb_split, \
    heatwave_vertical, horizontal_glitch, plasma, \
    rain_fisheye, rain_footprint

import PygameShader
from PygameShader import cartoon, blend, dirt_lens, dirt_lens_blur

PROJECT_PATH = list(PygameShader.__path__)
os.chdir(PROJECT_PATH[0] + "\\tests")

WIDTH = 1024
HEIGHT = 768
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
        "median"            : "Median",
        "median_grayscale"  : "Median grayscale",
        "median_avg"        : "Median avg",
        "color_reduction"          : "Color reduction",
        "sobel"                    : "Sobel",
        "sobel_fast"               : "Sobel fast",
        "invert"       : "Invert",
        "hsl"           : "HSL",
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

    if shader_name in ("hsl",
                       "hsl_fast"):
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


class TestShaderMedianFilter24Inplace(unittest.TestCase):
    """
    Test median
    """

    # pylint: disable=too-many-statements
    @staticmethod
    def runTest() -> None:
        """

        :return:  void
        """
        display_shader("median", timer = 5, flag=0)


class TestShaderMedianGrayscaleFilter24Inplace(unittest.TestCase):
    """
    Test median_grayscale
    """

    # pylint: disable=too-many-statements
    @staticmethod
    def runTest() -> None:
        """

        :return:  void
        """
        display_shader("median_grayscale", timer = 5, flag=0)


class TestShaderMedianFilter24AvgInplace(unittest.TestCase):
    """
    Test median_avg
    """

    # pylint: disable=too-many-statements
    @staticmethod
    def runTest() -> None:
        """

        :return:  void
        """
        display_shader("median_avg", timer = 5, flag=0)


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
            color_reduction(image, color_=COLORS[INDEX ])
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
                "Test color_reduction %s fps "
                "(%sx%s) %s colors" % (round(CLOCK.get_fps(), 2), WIDTH, HEIGHT,
                                       COLORS[INDEX] ** 2 * 2))
            image = background.copy()
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
        display_shader("hsl", timer = 5, flag=0, hsl_rotation=True)


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

        rgb2hsl_model = hsl_to_rgb_model()
        hsl2rgb_model = rgb_to_hsl_model()

        t = time.time()
        while game:
            hsl_fast(
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
            swirl2(image, ANGLE)

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
        background = pygame.image.load("../Assets/background.jpg").convert()
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

        fisheye_model = fisheye_footprint(WIDTH + 1, HEIGHT + 1)
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

            surface_ = predator_vision(image, sobel_threshold=80, bpf_threshold=0,
                                       bloom_threshold=0, inv_colormap=True, fast=False)
            # SCREEN.fill((0, 0, 0))
            SCREEN.blit(surface_, (0, 0))
            pygame.display.flip()
            CLOCK.tick()
            FRAME += 1
            pygame.display.set_caption(
                "Test predator_vision %s fps "
                "(%sx%s)" % (round(CLOCK.get_fps(), 2), WIDTH, HEIGHT))
            image = BACKGROUND.copy()
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
        background = pygame.image.load("../Assets/background.jpg").convert()
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
        background = pygame.image.load("../Assets/background.jpg").convert()
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

            surface_ = fire_effect(
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
            surface_ = fire_effect(
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
            surf.convert(32, RLEACCEL)
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
        BACKGROUND = pygame.image.load("../Assets/background.jpg")
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

            previous, current, array_ = ripple(W2, H2, previous, current, array_)

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

            horizontal_glitch(image, 0.5, 0.08, FRAME % 20)

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
        BACKGROUND = pygame.image.load("../Assets/Background.jpg").convert()
        BACKGROUND = pygame.transform.smoothscale(BACKGROUND, (WIDTH, HEIGHT))

        image = BACKGROUND.copy()
        pygame.display.set_caption("demo cartoon effect")

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

            surface_ = cartoon(image, sobel_threshold_=32, median_kernel_=2, color_=128,
                               flag_=BLEND_RGB_ADD).convert()

            SCREEN.blit(surface_, (0, 0), special_flags=0)

            pygame.display.flip()
            CLOCK.tick()
            FRAME += 1

            pygame.display.set_caption(
                "Test shader cartoon %s fps "
                "(%sx%s)" % (round(CLOCK.get_fps(), 2), WIDTH, HEIGHT))

            image = BACKGROUND.copy()
            if time.time() - t > 5:
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
        BACKGROUND = pygame.image.load("../Assets/Background.jpg").convert()
        BACKGROUND = pygame.transform.smoothscale(BACKGROUND, (WIDTH, HEIGHT))

        DESTINATION = pygame.image.load("../Assets/Aliens.jpg").convert()
        DESTINATION = pygame.transform.smoothscale(DESTINATION, (WIDTH, HEIGHT))

        assert BACKGROUND.get_size() == DESTINATION.get_size()

        pygame.display.set_caption("demo wave effect")

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
                source_=BACKGROUND, destination_=DESTINATION, percentage_=VALUE)

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

            dirt_lens_blur(IMAGE, flag_=BLEND_RGB_ADD, lens_model_=lens, light_=VALUE)

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

import pyglet
from os import path
FULL_VIDEO = []
PATH_TEXTURE = '../Assets/'
VIDEO_FLAGS  = pygame.HWSURFACE | pygame.RLEACCEL
SCREEN_BIT_DEPTH = 32


def play_intro():

    path_join = path.join
    media_player = pyglet.media.Player()

    ffmpeg = pyglet.media.have_ffmpeg()
    if not ffmpeg:
        print('\nFFmpeg library is missing on your system.')
        print('Video intro is skipped.')
        return
    try:

        video_name = "demo.avi"
        source = pyglet.media.load(path_join(PATH_TEXTURE, video_name))
        video_format = source.video_format
        source_info = source.info
        duration = source.duration
        width = video_format.width
        height = video_format.height
        sample_aspect = video_format.sample_aspect
        frame_rate = int(video_format.frame_rate)
        print("Length            : %ss " % round(duration, 2))
        print("Resolution        : %sx%s " % (width, height))
        print("Aspect resolution : %s " % sample_aspect)
        print("Frame rate        : %s fps" % frame_rate)
    except FileNotFoundError:
        print('\nVideo %s not found in %s ' % (video_name, PATH_TEXTURE))
        # IGNORE THE ERROR
        return

    pygame.init()
    vformat = source.video_format
    media_player.queue(source)
    media_player.play()
    screen = pygame.display.set_mode(
        (vformat.width, vformat.height), VIDEO_FLAGS, SCREEN_BIT_DEPTH)

    frame = 0
    running = True

    display_flip = pygame.display.flip
    media_get_texture = media_player.get_texture
    event_pump = pygame.event.pump

    screen.fill((0, 0, 0, 255))

    print('\nBuffering video \nPlease wait...(ESC to abort)')
    while running:

        event_pump()
        keys = pygame.key.get_pressed()
        if keys[pygame.K_ESCAPE]:
            running = False
            pygame.event.clear()
            break
        media_player.play()

        texture = media_get_texture()

        if texture is not None:

            raw = texture.get_image_data().get_data('RGBA', texture.width << 2)
            buff = numpy.frombuffer(raw, dtype=uint8).reshape(texture.height, texture.width, 4)
            # buff[:, :, :3] keep RGB no alpha channel
            # After removing the alpha channel (loosing alpha channel information).
            # The final result is a 24 bit pygame Surface.
            # Applying convert_alpha() will have no effect has the alpha channel has been removed
            # (the image will be converted to a 32 bit format with a layer set at
            # maximum opacity, 255 values).
            # If you wish to apply transparency to the surface use the method
            # set_colorkey to apply transparency
            # based on color key.
            arr = buff[:, :, :3].transpose(1, 0, 2)
            # blit_array(screen, arr)

            surf = pygame.surfarray.make_surface(arr).convert()
            # surf.set_colorkey((0, 0, 0), RLEACCEL)
            # surf = pygame.transform.scale(surf, (GL.SCREENRECT_W, GL.SCREENRECT_H))
            FULL_VIDEO.append(surf)

            pyglet.clock.tick(frame_rate)

            frame += 1
            if frame >= 200:
                running = False
        else:
            running = False
        try:
            media_player.seek_next_frame()
        except:
            pass

    print('[Done]')

    # ----- RESIZE VIDEO HERE ---------
    new_size = (1024, 768)
    i = 0
    for frame in FULL_VIDEO:
        FULL_VIDEO[i] = pygame.transform.smoothscale(frame, new_size)
        i += 1

    # ------ RESET THE DISPLAY WITH THE NEW SIZE
    screen = pygame.display.set_mode(
        new_size, VIDEO_FLAGS, SCREEN_BIT_DEPTH)


    frame = 0

    clock = pygame.time.Clock()
    running = True

    if running:
        # SOUND EFFECT
        # PLAY HERE

        # time per frame (T period)
        timing = duration / len(FULL_VIDEO)
        fps = 1.0 / timing

    # PLAY THE VIDEO FROM THE BUFFER
    while running:
        try:
            event_pump()
            keys = pygame.key.get_pressed()
            if keys[pygame.K_ESCAPE]:
                running = False
                pygame.event.clear()
                # CONTROL SOUND HERE
                # HERE
                break
            surface = FULL_VIDEO[frame]
            screen.blit(surface, (0, 0))

            display_flip()
            clock.tick(60)

            frame += 1

        except IndexError as e:
            running = False



class TestShaderMultiFisheyes(unittest.TestCase):
    """
        Test RainDrops
        """

    # pylint: disable=too-many-statements
    @staticmethod
    def runTest() -> None:
        """

        :return:  void
        """

        play_intro()

        pygame.display.set_caption("Test RainDrops")

        FRAME = 0
        CLOCK = pygame.time.Clock()
        GAME = True

        TEXTURE_SIZE = 512
        s, arr = rain_footprint(TEXTURE_SIZE, TEXTURE_SIZE)

        RAIN_LIST = []

        for i in range(2):
            RAIN_LIST.append((randint(0, 1024), randint(0, 200)))


        t = time.time()

        while GAME:

            pygame.event.pump()
            for event in pygame.event.get():

                keys = pygame.key.get_pressed()

                if keys[pygame.K_ESCAPE]:
                    GAME = False
                    break

            surface = FULL_VIDEO[FRAME % len(FULL_VIDEO)]
            SCREEN.blit(surface, (0, 0))

            ss = pygame.transform.scale(
                SCREEN, (TEXTURE_SIZE, TEXTURE_SIZE))
            rain_fisheye(ss, arr)
            p = ss.get_at((0, 0))
            ss.set_colorkey(p)

            surf = pygame.Surface((TEXTURE_SIZE, TEXTURE_SIZE)).convert()
            surf.fill((0, 0, 0))
            surf.blit(ss, (0, 0))

            for i in range(2):
                SCREEN.blit(surf,
                            (RAIN_LIST[i][0], RAIN_LIST[i][1]), special_flags=pygame.BLEND_RGB_MAX)


            pygame.display.flip()
            CLOCK.tick(60)
            FRAME += 1

            pygame.display.set_caption(
                "Test fisheye %s fps "
                "(%sx%s)" % (round(CLOCK.get_fps(), 2), WIDTH, HEIGHT))

            # if time.time() - t > 200:
            #     break


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
        TestShaderHeatwave24VerticalInplace(),
        TestShaderCartoon(),
        TestShaderBlend24(),
        # TestShaderMultiFisheyes()
    ])

    unittest.TextTestRunner().run(suite)
    sys.exit(0)


if __name__ == '__main__':
    # play_intro()
    run_testsuite()
    sys.exit(0)

