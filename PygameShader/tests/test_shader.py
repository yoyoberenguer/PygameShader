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
from os import path

import PygameShader
from PygameShader.PygameTools import *
from PygameShader import *
from PygameShader.misc import *
from PygameShader.gaussianBlur5x5 import *
from PygameShader.RGBConvert import *


import math
from math import sqrt

import cython
import numpy
from numpy import uint8, asarray
import pygame
from pygame.freetype import STYLE_NORMAL
from pygame.surfarray import make_surface, pixels3d, array3d
from pygame.transform import scale, smoothscale
from pygame.image import tostring
from pygame import BLEND_RGB_ADD, RLEACCEL, BLEND_RGB_MAX, BLEND_RGB_MIN, BLEND_RGB_SUB, \
    BLEND_RGB_MULT, freetype
import os

_CUPY = True
try:
    import cupy
except ImportError:
    _CUPY = False

if _CUPY:
    try:
        from PygameShader.shader_gpu import *
    except:
        _CUPY = False

os.environ[ 'PYGAME_HIDE_SUPPORT_PROMPT' ] = "hide"

PROJECT_PATH = list(PygameShader.__path__)
os.chdir(PROJECT_PATH[ 0 ] + "/tests")

print(PROJECT_PATH, PROJECT_PATH[ 0 ])

WIDTH = 800
HEIGHT = 600
SCREEN = pygame.display.set_mode((WIDTH, HEIGHT), vsync = True)
SCREEN.convert(32, RLEACCEL)
SCREEN.set_alpha(None)

shader_list = \
    {
        "bgr": "RGB to BGR",
        "brg": "RGB to BRG",
        "grey": "grayscale Luminosity",
        "sepia": "Sepia",
        # "median"            : "Median",
        "median_grayscale": "Median grayscale",
        "color_reduction": "Color reduction",
        "sobel": "Sobel",
        "sobel_fast": "Sobel fast_",
        "invert": "Invert",
        "blur": "Blur",
        "wave": "Wave",
        "swirl": "Swirl",
        "swirl2": "Swirl1",
        "plasma_config": "Plasma",
        "brightness": "Brightness",
        "saturation": "Saturation",
        "bpf": "Bright pass filter",
        "bloom": "Bloom",
        "fisheye": "Fisheye",
        "tv_scan": "TV",
        "heatmap": "Heatmap",
        "predator_vision": "Predator Vision",
        "blood": "Blood",
        "sharpen": "Sharpen",
        "fire_effect": "Fire Effect 1",
        "dampening": "Dampening",
        "lateral_dampening": "Lateral dampening",
        "mirroring": "Mirror image",
        "cloud_effect": "Shader cloud",
        "tunnel_render32": "Tunnel",
        "ripple": "Water Ripple",
        "heatconvection": "Heatwave",
        "horizontal_glitch": "Horizontal glitch",
        "plasma": "Plasma 2"
    }


def display_shader(shader_name: str, timer = 5, flag = 0, *args, **kwargs):
    background = pygame.image.load("../Assets/background.jpg").convert()
    background = pygame.transform.smoothscale(background, (WIDTH, HEIGHT))
    background.convert(32, RLEACCEL)
    image = background.copy()

    all_shaders = shader_list.keys()
    if shader_name not in all_shaders:
        raise ValueError("\nShader name is invalid got %s" % shader_name)

    try:
        pygame.display.set_caption("Testing %s" % shader_list[ shader_name ])

    except IndexError:
        raise IndexError('\nShader name is incorrect got %s' % shader_name)

    shader_ = shader_list[ shader_name ]

    frame = 0
    clock = pygame.time.Clock()
    game = True

    t = time.time()

    options = " "
    for opt in args:
        options += ", " + str(opt)

    hsl_rotation = False
    hsl_value = 0.0
    if shader_name in ("hsl_effect",
                       "hsl_fast"):
        hsl_rotation = True

        hsl_value = 0.0

    angle = 0

    while game:

        if hsl_rotation:

            exec(str(shader_name) + "(image, hsl_value)")
        else:
            exec(str(shader_name) + "(image)"
                 if kwargs is None else str(shader_name) + "(image" + options + ")")

        pygame.event.pump()
        for _ in pygame.event.get():

            keys = pygame.key.get_pressed()

            if keys[ pygame.K_ESCAPE ]:
                game = False

        SCREEN.blit(image, (0, 0), special_flags = flag)
        pygame.display.flip()
        clock.tick()
        frame += 1
        v = 0
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
    Test bgr_copy
    """

    # pylint: disable=too-many-statements
    @staticmethod
    def runTest() -> None:
        """

        :return:  void
        """

        WIDTH = 800
        HEIGHT = 800

        BCK1 = pygame.image.load("../Assets/background.jpg")  # .convert_alpha()
        BCK1 = pygame.transform.smoothscale(BCK1, (WIDTH, HEIGHT))

        FRAME = 0
        CLOCK = pygame.time.Clock()
        GAME = True

        t = time.time()
        while GAME:

            pygame.event.pump()
            for _ in pygame.event.get():

                keys = pygame.key.get_pressed()

                if keys[ pygame.K_ESCAPE ]:
                    GAME = False
                    break

            im = bgr_copy(BCK1)  # .convert_alpha()
            SCREEN.fill((0, 10, 5))
            SCREEN.blit(im, (0, 0))

            pygame.display.flip()

            CLOCK.tick()
            FRAME += 1
            pygame.display.set_caption(
                "Test bgr_copy %s fps "
                "(%sx%s)" % (round(CLOCK.get_fps(), 2), WIDTH, HEIGHT))
            if time.time() - t > 5:
                break

        # display_shader("bgr", timer = 5, flag=0)


class TestShaderRgbToBrgInplace(unittest.TestCase):
    """
    Test brg
    """

    # pylint: disable=too-many-statements
    @staticmethod
    def runTest() -> None:
        """

        :return:  void
        """
        WIDTH = 800
        HEIGHT = 800

        BCK1 = pygame.image.load("../Assets/background.jpg")  # .convert_alpha()
        BCK1 = pygame.transform.smoothscale(BCK1, (WIDTH, HEIGHT))
        bck_cp = BCK1.copy()
        FRAME = 0
        CLOCK = pygame.time.Clock()
        GAME = True

        t = time.time()
        while GAME:

            pygame.event.pump()
            for _ in pygame.event.get():

                keys = pygame.key.get_pressed()

                if keys[ pygame.K_ESCAPE ]:
                    GAME = False
                    break

            # brg(BCK1)
            brg(BCK1)  # .convert_alpha()
            SCREEN.fill((0, 10, 5))
            SCREEN.blit(BCK1, (0, 0))
            pygame.display.flip()

            BCK1 = bck_cp.copy()

            CLOCK.tick()
            FRAME += 1
            pygame.display.set_caption(
                "Test brg %s fps "
                "(%sx%s)" % (round(CLOCK.get_fps(), 2), WIDTH, HEIGHT))
            if time.time() - t > 5:
                break

        # display_shader("brg", timer = 5, flag=0)


class TestShaderGreyscaleLuminosity24Inplace(unittest.TestCase):
    """
    Test grey_copy
    """

    # pylint: disable=too-many-statements
    @staticmethod
    def runTest() -> None:
        """

        :return:  void
        """
        WIDTH = 800
        HEIGHT = 800

        BCK1 = pygame.image.load("../Assets/background.jpg")  # .convert_alpha()
        BCK1 = pygame.transform.smoothscale(BCK1, (WIDTH, HEIGHT))

        FRAME = 0
        CLOCK = pygame.time.Clock()
        GAME = True

        t = time.time()
        while GAME:

            pygame.event.pump()
            for _ in pygame.event.get():

                keys = pygame.key.get_pressed()

                if keys[ pygame.K_ESCAPE ]:
                    GAME = False
                    break

            im = grey_copy(BCK1)  # .convert_alpha()
            SCREEN.fill((0, 10, 5))
            SCREEN.blit(im, (0, 0))
            pygame.display.flip()

            CLOCK.tick()
            FRAME += 1
            pygame.display.set_caption(
                "Test grey_copy %s fps "
                "(%sx%s)" % (round(CLOCK.get_fps(), 2), WIDTH, HEIGHT))
            if time.time() - t > 5:
                break
        # display_shader("grey", timer = 5, flag=0)


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
        WIDTH = 800
        HEIGHT = 800

        BCK1 = pygame.image.load("../Assets/background.jpg")  # .convert_alpha()
        BCK1 = pygame.transform.smoothscale(BCK1, (WIDTH, HEIGHT))

        FRAME = 0
        CLOCK = pygame.time.Clock()
        GAME = True

        t = time.time()
        while GAME:

            pygame.event.pump()
            for _ in pygame.event.get():

                keys = pygame.key.get_pressed()

                if keys[ pygame.K_ESCAPE ]:
                    GAME = False
                    break

            im = sepia_copy(BCK1)  # .convert_alpha()
            SCREEN.fill((0, 10, 5))
            SCREEN.blit(im, (0, 0))
            pygame.display.flip()

            CLOCK.tick()
            FRAME += 1
            pygame.display.set_caption(
                "Test SEPIA %s fps "
                "(%sx%s)" % (round(CLOCK.get_fps(), 2), WIDTH, HEIGHT))
            if time.time() - t > 500000000000000000000:
                break
        # display_shader("sepia", timer = 5, flag=0)


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
        display_shader("median_grayscale", timer = 5, flag = 0)


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

        COLORS = [ 64, 32, 24, 16, 8, 4, 2 ]
        INDEX = 0
        t = time.time()
        while GAME:

            pygame.event.pump()
            for _ in pygame.event.get():

                keys = pygame.key.get_pressed()

                if keys[ pygame.K_ESCAPE ]:
                    GAME = False

            posterize_surface(image, color_ = COLORS[ INDEX ])
            # SCREEN.fill((0, 0, 0))
            SCREEN.blit(image, (0, 0))
            pygame.display.flip()
            CLOCK.tick()
            FRAME += 1
            pygame.display.set_caption(
                "Test color_reduction %s fps "
                "(%sx%s) %s colors" % (round(CLOCK.get_fps(), 2), WIDTH, HEIGHT,
                                       COLORS[ INDEX ] ** 2 * 2))

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
        display_shader("sobel_fast", timer = 5, flag = 0)


class TestShaderInvertSurface24bitInplace(unittest.TestCase):
    """
    Test invert_copy
    """

    # pylint: disable=too-many-statements
    @staticmethod
    def runTest() -> None:
        """

        :return:  void
        """
        WIDTH = 800
        HEIGHT = 800

        BCK1 = pygame.image.load("../Assets/background.jpg")
        BCK1 = pygame.transform.smoothscale(BCK1, (WIDTH, HEIGHT))

        FRAME = 0
        CLOCK = pygame.time.Clock()
        GAME = True

        t = time.time()
        while GAME:

            pygame.event.pump()
            for _ in pygame.event.get():

                keys = pygame.key.get_pressed()

                if keys[ pygame.K_ESCAPE ]:
                    GAME = False
                    break

            im = invert_copy(BCK1)
            # SCREEN.fill((0, 10, 5))
            SCREEN.blit(im, (0, 0))

            pygame.display.flip()

            CLOCK.tick()
            FRAME += 1

            pygame.display.set_caption(
                "Test invert_copy %s fps "
                "(%sx%s)" % (round(CLOCK.get_fps(), 2), WIDTH, HEIGHT))
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
        background = pygame.image.load("../Assets/px.png").convert_alpha()
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

            SCREEN.fill((50, 0, 0))
            wave(image, angle * math.pi / 180, 10)

            pygame.event.pump()
            for _ in pygame.event.get():

                keys = pygame.key.get_pressed()

                if keys[ pygame.K_ESCAPE ]:
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
        background = pygame.image.load("../Assets/background.jpg").convert(24)
        background = pygame.transform.smoothscale(background, (WIDTH, HEIGHT))
        background_cp = background.copy()

        pygame.display.set_caption("Test swirl")

        frame = 0
        clock = pygame.time.Clock()
        game = True
        angle = 0
        t = time.time()
        while game:

            swirl(background_cp, pixels3d(background), angle)

            pygame.event.pump()
            for _ in pygame.event.get():

                keys = pygame.key.get_pressed()

                if keys[ pygame.K_ESCAPE ]:
                    game = False

            # SCREEN.fill((10, 10, 50))
            SCREEN.blit(background_cp, (0, 0))
            pygame.display.flip()
            clock.tick()
            frame += 1
            pygame.display.set_caption(
                "Test swirl %s fps "
                "(%sx%s)" % (round(clock.get_fps(), 2), WIDTH, HEIGHT))

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
            swirlf(image, ANGLE)

            pygame.event.pump()
            for _ in pygame.event.get():

                keys = pygame.key.get_pressed()

                if keys[ pygame.K_ESCAPE ]:
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

        arr = numpy.array([ 0, 1,  # violet
                            0, 1,  # blue
                            0, 1,  # green
                            2, 619,  # yellow
                            620, 650,  # orange
                            651, 660 ],  # red
                          numpy.int32)

        HEATMAP = [ custom_map(i - 20, arr, 1.0) for i in range(380, 800) ]
        heatmap_array = numpy.zeros((800 - 380, 3), uint8)
        heatmap_rescale = numpy.zeros(255, numpy.uint32)

        i = 0
        for t in HEATMAP:
            heatmap_array[ i, 0 ] = t[ 0 ]
            heatmap_array[ i, 1 ] = t[ 1 ]
            heatmap_array[ i, 2 ] = t[ 2 ]
            i += 1
        for r in range(0, 255):
            s = int(r * (800.0 - 380.0) / 255)
            heatmap_rescale[ r ] = \
                rgb_to_int(heatmap_array[ s ][ 0 ], heatmap_array[ s ][ 1 ], heatmap_array[ s ][ 2 ])

        heatmap_rescale = numpy.ascontiguousarray(heatmap_rescale[ ::-1 ])
        t = time.time()
        while GAME:

            pygame.event.pump()
            for _ in pygame.event.get():

                keys = pygame.key.get_pressed()

                if keys[ pygame.K_ESCAPE ]:
                    GAME = False

            im = image.copy()
            plasma(im, FRAME, heatmap_rescale)
            im.blit(image, (0, 0), special_flags = BLEND_RGB_ADD)
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
            plasma_config(image, FRAME, sat_ = 0.12, value_ = 0.38)

            pygame.event.pump()
            for _ in pygame.event.get():

                keys = pygame.key.get_pressed()

                if keys[ pygame.K_ESCAPE ]:
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
        background = pygame.image.load("../Assets/aliens.jpg").convert()
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
            for _ in pygame.event.get():

                keys = pygame.key.get_pressed()

                if keys[ pygame.K_ESCAPE ]:
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
            for _ in pygame.event.get():

                keys = pygame.key.get_pressed()

                if keys[ pygame.K_ESCAPE ]:
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
        t = time.time()
        while GAME:

            pygame.event.pump()
            for _ in pygame.event.get():

                keys = pygame.key.get_pressed()

                if keys[ pygame.K_ESCAPE ]:
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
        t = time.time()
        while GAME:

            pygame.event.pump()
            for _ in pygame.event.get():

                keys = pygame.key.get_pressed()

                if keys[ pygame.K_ESCAPE ]:
                    GAME = False
                    break

            bloom(image, BPF, fast_ = True)
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
        background = pygame.image.load("../Assets/px.png").convert(24)
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
            for _ in pygame.event.get():

                keys = pygame.key.get_pressed()

                if keys[ pygame.K_ESCAPE ]:
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
        background = pygame.image.load("../Assets/px.png").convert_alpha()
        background = pygame.transform.smoothscale(background, (WIDTH, HEIGHT))
        # background.convert(32, RLEACCEL)
        image = background.copy()
        pygame.display.set_caption("Test tv_scan")

        FRAME = 10
        CLOCK = pygame.time.Clock()
        GAME = True
        t = time.time()
        while GAME:

            pygame.event.pump()
            for _ in pygame.event.get():

                keys = pygame.key.get_pressed()

                if keys[ pygame.K_ESCAPE ]:
                    GAME = False
                    break

            SCREEN.fill((50, 0, 0))
            tv_scan(image, FRAME)

            SCREEN.blit(image, (0, 0))
            pygame.display.flip()
            CLOCK.tick()
            # FRAME += 1
            # FRAME %= 10

            pygame.display.set_caption(
                "Test tv_scan %s fps "
                "(%sx%s)" % (round(CLOCK.get_fps(), 2), WIDTH, HEIGHT))
            image = background.copy()
            if time.time() - t > 5:
                break


class Testwavelength2rgb(unittest.TestCase):
    """
    Test wavelength2rgb
    """

    # pylint: disable=too-many-statements
    @staticmethod
    def runTest() -> None:
        """

        :return:  void
        """
        pygame.display.set_caption("Test wavelength2rgb")
        rgb = wavelength2rgb(610)
        print(rgb)
        arr = numpy.array([ 0, 1,  # violet
                            0, 1,  # blue
                            0, 1,  # green
                            0, 1,  # yellow
                            2, 650,  # orange
                            651, 750 ],  # red
                          numpy.int32)

        spectrum_array = numpy.empty((370, 2, 3), numpy.uint8)

        for i in range(380, 750):
            r, g, b = custom_map(i, arr, 1.0)
            k = i - 380
            spectrum_array[ k, 0, 0 ] = r
            spectrum_array[ k, 0, 1 ] = g
            spectrum_array[ k, 0, 2 ] = b
            spectrum_array[ k, 1, 0 ] = r
            spectrum_array[ k, 1, 1 ] = g
            spectrum_array[ k, 1, 2 ] = b

        # while 1:
        #     SCREEN.fill((50, 10, 10))
        #     SCREEN.blit(image, (0, 0))
        #     pygame.display.flip()
        print(custom_map(2, arr, 1.0))


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
        background = pygame.image.load("../Assets/px.png").convert_alpha()
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
            for _ in pygame.event.get():

                keys = pygame.key.get_pressed()

                if keys[ pygame.K_ESCAPE ]:
                    GAME = False
                    break

            SCREEN.fill((50, 10, 10))
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
            for _ in pygame.event.get():

                keys = pygame.key.get_pressed()

                if keys[ pygame.K_ESCAPE ]:
                    GAME = False
                    break

            if FRAME < 90:
                surface_ = predator_vision(
                    image.copy(), sobel_threshold = 80, bpf_threshold = 0,
                    bloom_threshold = 0, inv_colormap = True, fast = True)
            else:
                surface_ = predator_vision(
                    image1.copy(), sobel_threshold = 80, bpf_threshold = 0,
                    bloom_threshold = 10, inv_colormap = True, fast = True)

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
            for _ in pygame.event.get():

                keys = pygame.key.get_pressed()

                if keys[ pygame.K_ESCAPE ]:
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
        background = pygame.image.load("../Assets/city.jpg").convert(24)
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
            for _ in pygame.event.get():

                keys = pygame.key.get_pressed()

                if keys[ pygame.K_ESCAPE ]:
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

        pygame.display.set_caption("Test fire_effect")

        FRAME = 0
        CLOCK = pygame.time.Clock()
        GAME = True

        arr = numpy.array([ 0, 1,  # violet
                            0, 1,  # blue
                            0, 1,  # green
                            0, 619,  # yellow
                            620, 650,  # orange
                            651, 680 ],  # red
                          numpy.int32)

        HEATMAP = [ custom_map(i - 20, arr, 1.0) for i in range(380, 800) ]
        heatmap_array = numpy.zeros((800 - 380, 3), uint8)
        heatmap_rescale = numpy.zeros(255, numpy.uint32)

        i = 0
        for t in HEATMAP:
            heatmap_array[ i, 0 ] = t[ 0 ]
            heatmap_array[ i, 1 ] = t[ 1 ]
            heatmap_array[ i, 2 ] = t[ 2 ]
            i += 1
        for r in range(0, 255):
            s = int(r * (800.0 - 380.0) / 255)
            heatmap_rescale[ r ] = rgb_to_int(heatmap_array[ s ][ 0 ], heatmap_array[ s ][ 1 ],
                                              heatmap_array[ s ][ 2 ])

        heatmap_rescale = numpy.ascontiguousarray(heatmap_rescale[ ::-1 ])
        FIRE_ARRAY = numpy.zeros((HEIGHT, WIDTH), dtype = numpy.float32)
        t = time.time()
        while GAME:

            pygame.event.pump()
            for _ in pygame.event.get():

                keys = pygame.key.get_pressed()

                if keys[ pygame.K_ESCAPE ]:
                    GAME = False
                    break
            SCREEN.blit(BACKGROUND, (0, 0))

            surface_ = fire_effect(
                WIDTH, HEIGHT, 3.97 + uniform(0.002, 0.008),
                heatmap_rescale,
                FIRE_ARRAY,
                reduce_factor_ = 3, fire_intensity_ = 32,
                smooth_ = True, bloom_ = True, fast_bloom_ = False,
                bpf_threshold_ = 128, low_ = 1, high_ = WIDTH, brightness_ = True,
                brightness_intensity_ = 0.1, adjust_palette_ = True,
                hsl_ = (10, 80, 1.8), transpose_ = False, border_ = False,
                surface_ = None)

            SCREEN.blit(surface_, (0, 0), special_flags = pygame.BLEND_RGBA_MAX)

            pygame.display.flip()
            CLOCK.tick()
            FRAME += 1
            pygame.display.set_caption(
                "Test fire_effect %s fps "
                "(%sx%s)" % (round(CLOCK.get_fps(), 2), WIDTH, HEIGHT))

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

        pygame.display.set_caption("Test fire_effect")

        FRAME = 0
        CLOCK = pygame.time.Clock()
        GAME = True

        arr = numpy.array([ 0, 1,  # violet
                            0, 1,  # blue
                            0, 1,  # green
                            2, 619,  # yellow
                            620, 650,  # orange
                            651, 660 ],  # red
                          numpy.int32)

        HEATMAP = [ custom_map(i - 20, arr, 1.0) for i in range(380, 800) ]
        heatmap_array = numpy.zeros((800 - 380, 3), uint8)
        heatmap_rescale = numpy.zeros(255, numpy.uint32)

        i = 0
        for t in HEATMAP:
            heatmap_array[ i, 0 ] = t[ 0 ]
            heatmap_array[ i, 1 ] = t[ 1 ]
            heatmap_array[ i, 2 ] = t[ 2 ]
            i += 1
        for r in range(0, 255):
            s = int(r * (800.0 - 380.0) / 255)
            heatmap_rescale[ r ] = rgb_to_int(heatmap_array[ s ][ 0 ], heatmap_array[ s ][ 1 ],
                                              heatmap_array[ s ][ 2 ])

        heatmap_rescale = numpy.ascontiguousarray(heatmap_rescale[ ::-1 ])
        FIRE_ARRAY = numpy.zeros((HEIGHT, WIDTH), dtype = numpy.float32)
        t = time.time()
        while GAME:

            pygame.event.pump()
            for _ in pygame.event.get():

                keys = pygame.key.get_pressed()

                if keys[ pygame.K_ESCAPE ]:
                    GAME = False
                    break
            SCREEN.blit(BACKGROUND, (0, 0))
            surface_ = fire_effect(
                WIDTH, HEIGHT, 3.97 + uniform(0.002, 0.008),
                heatmap_rescale,
                FIRE_ARRAY,
                reduce_factor_ = 3, fire_intensity_ = 16,
                smooth_ = True, bloom_ = True, fast_bloom_ = False,
                bpf_threshold_ = 1, low_ = 1, high_ = WIDTH, brightness_ = True,
                brightness_intensity_ = 0.15, adjust_palette_ = False,
                hsl_ = (10, 80, 1.8), transpose_ = False, border_ = False,
                surface_ = None)

            SCREEN.blit(surface_, (0, 0), special_flags = BLEND_RGB_ADD)

            pygame.display.flip()
            CLOCK.tick()
            FRAME += 1
            pygame.display.set_caption(
                "Test fire_effect %s fps "
                "(%sx%s)" % (round(CLOCK.get_fps(), 2), WIDTH, HEIGHT))

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

        pygame.display.set_caption("Test lateral_dampening")

        FRAME = 0
        CLOCK = pygame.time.Clock()
        GAME = True
        t = time.time()
        while GAME:

            pygame.event.pump()
            for _ in pygame.event.get():

                keys = pygame.key.get_pressed()

                if keys[ pygame.K_ESCAPE ]:
                    GAME = False
                    break
            # ZOOM IN
            # surf, xx, yy = dampening(BACKGROUND, FRAME, WIDTH, HEIGHT, amplitude=60,
            #                                 duration=100, freq=0.8)
            # SCREEN.blit(surf, (xx, yy))

            tm = lateral_dampening(FRAME, amplitude = 10.0, duration = 10, freq = 50.0)
            SCREEN.blit(BACKGROUND, (tm, 0), special_flags = 0)

            pygame.display.flip()
            CLOCK.tick()
            FRAME += 1
            pygame.display.set_caption(
                "Test lateral_dampening %s fps "
                "(%sx%s)" % (round(CLOCK.get_fps(), 2), WIDTH, HEIGHT))

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

        pygame.display.set_caption("Test dampening")

        FRAME = 0
        CLOCK = pygame.time.Clock()
        GAME = True
        t = time.time()
        while GAME:

            pygame.event.pump()
            for _ in pygame.event.get():

                keys = pygame.key.get_pressed()

                if keys[ pygame.K_ESCAPE ]:
                    GAME = False
                    break
            surf, xx, yy = dampening(
                BACKGROUND,
                FRAME,
                WIDTH,
                HEIGHT,
                amplitude = 15.0,
                duration = 10,
                freq = 25.0)
            # surf.convert(32, RLEACCEL)
            SCREEN.blit(surf, (xx, yy), special_flags = 0)

            pygame.display.flip()
            CLOCK.tick()
            FRAME += 1
            pygame.display.set_caption(
                "Test dampening %s fps "
                "(%sx%s)" % (round(CLOCK.get_fps(), 2), WIDTH, HEIGHT))

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
        BACKGROUND = pygame.image.load("../Assets/Aliens.jpg").convert(24)
        BACKGROUND = pygame.transform.smoothscale(BACKGROUND, (WIDTH, HEIGHT))
        image = BACKGROUND.copy()
        pygame.display.set_caption("Test mirroring")

        FRAME = 0
        CLOCK = pygame.time.Clock()
        GAME = True
        t = time.time()
        while GAME:

            pygame.event.pump()
            for _ in pygame.event.get():

                keys = pygame.key.get_pressed()

                if keys[ pygame.K_ESCAPE ]:
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

        pygame.display.set_caption("Test cloud/Smoke shader")

        FRAME = 0
        CLOCK = pygame.time.Clock()
        GAME = True

        CLOUD_ARRAY = numpy.zeros((HEIGHT, WIDTH), dtype = numpy.float32)

        heatmap_rescale = numpy.zeros(256 * 2 * 3, numpy.uint32)

        arr1 = create_line_gradient_rgb(255, (0, 0, 0), (255, 255, 255))
        arr2 = create_line_gradient_rgb(255, (255, 255, 255), (0, 0, 0))
        arr3 = numpy.concatenate((arr1, arr2), axis = None)
        i = 0
        for r in range(0, 1530, 3):
            heatmap_rescale[ i ] = rgb_to_int(arr3[ r ], arr3[ r + 1 ], arr3[ r + 2 ])
            i += 1
        t = time.time()
        while GAME:

            pygame.event.pump()
            for _ in pygame.event.get():

                keys = pygame.key.get_pressed()

                if keys[ pygame.K_ESCAPE ]:
                    GAME = False
                    break
            SCREEN.blit(BACKGROUND, (0, 0))

            surface_ = cloud_effect(
                WIDTH, HEIGHT, 3.970 + uniform(0.002, 0.008),
                heatmap_rescale,
                CLOUD_ARRAY,
                reduce_factor_ = 3, cloud_intensity_ = randint(60, 80),
                smooth_ = True, bloom_ = True, fast_bloom_ = True,
                bpf_threshold_ = 80, low_ = 0, high_ = WIDTH, brightness_ = True,
                brightness_intensity_ = 0.15,
                transpose_ = False, surface_ = None, blur_ = True)

            SCREEN.blit(surface_, (0, 0), special_flags = BLEND_RGB_ADD)

            pygame.display.flip()
            CLOCK.tick()
            FRAME += 1
            pygame.display.set_caption(
                "Test cloud_effect %s fps "
                "(%sx%s)" % (round(CLOCK.get_fps(), 2), WIDTH, HEIGHT))

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

        BCK1 = pygame.image.load("../Assets/space2.jpg").convert(24)
        BCK1 = pygame.transform.smoothscale(BCK1, (WIDTH, HEIGHT))
        BACKGROUND = pygame.image.load("../Assets/space1.jpg")
        BACKGROUND = pygame.transform.smoothscale(BACKGROUND, (WIDTH, HEIGHT))

        FRAME = 0
        CLOCK = pygame.time.Clock()
        GAME = True

        distances, angles, shades, scr_data = tunnel_modeling24(WIDTH, HEIGHT, BACKGROUND)
        dest_array = numpy.empty((WIDTH * HEIGHT * 4), numpy.uint8)

        t = time.time()
        while GAME:

            pygame.event.pump()
            for _ in pygame.event.get():

                keys = pygame.key.get_pressed()

                if keys[ pygame.K_ESCAPE ]:
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
            SCREEN.blit(surface_, (0, 0), special_flags = pygame.BLEND_RGB_ADD)

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

        BCK1 = pygame.image.load("../Assets/space2.jpg").convert()
        BCK1 = pygame.transform.smoothscale(BCK1, (WIDTH, HEIGHT))
        BACKGROUND = pygame.image.load("../Assets/space2_seamless_alpha.jpg")
        BACKGROUND = pygame.transform.smoothscale(BACKGROUND, (WIDTH, HEIGHT))

        FRAME = 0
        CLOCK = pygame.time.Clock()
        GAME = True

        distances, angles, shades, scr_data = tunnel_modeling32(WIDTH, HEIGHT, BACKGROUND)
        dest_array = numpy.empty((WIDTH * HEIGHT * 4), numpy.uint8)
        t = time.time()
        while GAME:

            pygame.event.pump()
            for _ in pygame.event.get():

                keys = pygame.key.get_pressed()

                if keys[ pygame.K_ESCAPE ]:
                    GAME = False
                    break

            SCREEN.blit(BCK1, (0, 0))
            # SCREEN.fill((0, 0, 0, 0))
            surface_ = tunnel_render32(
                FRAME * 5,
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
            SCREEN.blit(surface_, (0, 0), special_flags = pygame.BLEND_RGB_ADD)

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

        current = numpy.zeros((W2, H2), dtype = numpy.float32)
        previous = numpy.zeros((W2, H2), dtype = numpy.float32)
        array_ = numpy.full((W2, H2, 3), 0, numpy.uint8)

        image = BACKGROUND.copy()
        t = time.time()
        while GAME:

            pygame.event.pump()
            for _ in pygame.event.get():

                keys = pygame.key.get_pressed()

                if keys[ pygame.K_ESCAPE ]:
                    GAME = False
                    break

            SCREEN.blit(BACKGROUND, (0, 0))

            rnd = randint(0, 1000)
            if rnd > 900:
                m = randint(1000, 10000)
                previous[ randint(0, W2 - 1), randint(0, H2 - 1) ] = m

            previous, current = ripple(W2, H2, previous, current, array_)

            surf = make_surface(asarray(array_, dtype = uint8))
            surf = smoothscale(surf, (WIDTH, HEIGHT))
            image.blit(surf, (0, 0), special_flags = pygame.BLEND_RGBA_ADD)

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
    val = (cupy.roll(previous, 1, axis = 0).astype(dtype = cupy.float32) +
           cupy.roll(previous, -1, axis = 0).astype(dtype = cupy.float32) +
           cupy.roll(previous, 1, axis = 1).astype(dtype = cupy.float32) +
           cupy.roll(previous, -1, axis = 1).astype(dtype = cupy.float32)) / float(2.0) - current.astype(
        dtype = cupy.float32)

    current = (val * 0.98).astype(dtype = cupy.float32)

    return previous, current


pygame.font.init()
FONT = pygame.font.SysFont("Arial", 15)


def show_fps(screen_, fps_, avg_) -> list:
    """ Show framerate in upper left corner """

    fps = str(f"fps: {fps_: .3f}")
    av = sum(avg_) / len(avg_) if len(avg_) > 0 else 0

    fps_text = FONT.render(fps, 1, pygame.Color("coral"))
    screen_.blit(fps_text, (10, 0))

    if av != 0:
        av = str(f"avg: {av: .3f}")
        avg_text = FONT.render(av, 1, pygame.Color("coral"))
        screen_.blit(avg_text, (80, 0))
    if len(avg_) > 200:
        avg_ = avg_[ 200: ]
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

        pygame.display.init()
        # SCREEN = pygame.display.set_mode(SCREENRECT.size, pygame.DOUBLEBUF, 32)

        pygame.init()

        clock = pygame.time.Clock()
        FRAME = 0
        STOP_GAME = False

        current = cupy.empty((width, height), dtype = cupy.float32)
        previous = cupy.empty((width, height), dtype = cupy.float32)
        texture_array = cupy.asarray(pygame.surfarray.array3d(texture), dtype = cupy.uint8)
        back_array = cupy.asarray(pygame.surfarray.pixels3d(BACKGROUND), dtype = cupy.uint8)

        # TWEAKS
        cget_fps = clock.get_fps
        event_pump = pygame.event.pump
        event_get = pygame.event.get
        get_key = pygame.key.get_pressed
        get_pos = pygame.mouse.get_pos
        flip = pygame.display.flip

        grid, block = block_grid(width, height)
        avg = [ ]
        while not STOP_GAME:

            pygame.display.set_caption("DEMO ripple effect GPU : FPS %s " % round(cget_fps(), 2))

            event_pump()

            keys = get_key()
            # SCREEN.fill((0, 0, 0, 0))
            for event in event_get():

                if keys[ pygame.K_ESCAPE ]:
                    STOP_GAME = True
                    break

                if event.type == pygame.MOUSEMOTION:
                    mouse_pos = pygame.math.Vector2(get_pos())

                    previous[ int(mouse_pos.x % width), int(mouse_pos.y % height) ] = 2 ** 12

            rnd = randint(0, 1000)

            if rnd > 820:
                previous[ randint(0, width - 2), randint(0, height - 2) ] = 2 ** 10

            # METHOD 1
            # previous_, current_ = ripple_2(current_, previous_)
            # bck_cp[:, :, 0:3] = current_.reshape((w, h, 1))
            #
            # surf = pygame.image.frombuffer(bck_cp.transpose(1, 0, 2).tobytes(), (w, h), "RGB")
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
            avg = show_fps(SCREEN, t, avg)
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
            for _ in pygame.event.get():

                keys = pygame.key.get_pressed()

                if keys[ pygame.K_ESCAPE ]:
                    GAME = False
                    break

            split_channels_inplace(image, 15)

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
            for _ in pygame.event.get():

                keys = pygame.key.get_pressed()

                if keys[ pygame.K_ESCAPE ]:
                    GAME = False
                    break

            horizontal_glitch(image, 1.0, 0.08, FRAME % 100)

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
    Test heatconvection
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
            for _ in pygame.event.get():

                keys = pygame.key.get_pressed()

                if keys[ pygame.K_ESCAPE ]:
                    GAME = False
                    break

            b = 0.1
            heatconvection(
                surface_ = image,
                amplitude = b * uniform(150, 800),
                center = 0,
                sigma = uniform(2, 4),
                mu = b * 2
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
            for _ in pygame.event.get():

                keys = pygame.key.get_pressed()

                if keys[ pygame.K_ESCAPE ]:
                    GAME = False
                    break

            surf = cartoon(image, median_kernel = 2, color = 8, sobel_threshold = 64)

            SCREEN.blit(surf, (0, 0), special_flags = 0)

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
        BACKGROUND = pygame.image.load("../Assets/background.jpg").convert(24)
        BACKGROUND = pygame.transform.smoothscale(BACKGROUND, (WIDTH, HEIGHT))

        DESTINATION = pygame.image.load("../Assets/Aliens.jpg").convert(24)
        DESTINATION = pygame.transform.smoothscale(DESTINATION, (WIDTH, HEIGHT))
        assert BACKGROUND.get_size() == DESTINATION.get_size()

        pygame.display.set_caption("demo blend inplace")

        FRAME = 0
        CLOCK = pygame.time.Clock()
        GAME = True
        VALUE = 0
        V = +0.2

        while GAME:

            pygame.event.pump()
            for _ in pygame.event.get():

                keys = pygame.key.get_pressed()

                if keys[ pygame.K_ESCAPE ]:
                    GAME = False
                    break

            transition = blend(
                source = BACKGROUND, destination = DESTINATION, percentage = VALUE)

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

        lens = dirt_lens_image[ 0 ]
        lens = pygame.transform.scale(lens, (WIDTH, HEIGHT)).convert()
        lens_copy = lens.copy()

        pygame.display.set_caption("demo Dirt Lens effect")

        FRAME = 0
        CLOCK = pygame.time.Clock()
        GAME = True
        VALUE = 0.2
        V = -0.005
        freetype.init(cache_size = 64, resolution = 72)
        FONT = freetype.SysFont('ARIALBLACK', size = 18)
        while GAME:

            pygame.event.pump()
            for _ in pygame.event.get():

                keys = pygame.key.get_pressed()

                if keys[ pygame.K_ESCAPE ]:
                    GAME = False
                    break

            dirt_lens(IMAGE, flag_ = BLEND_RGB_ADD, lens_model_ = lens, light_ = VALUE)

            SCREEN.blit(IMAGE, (0, 0))
            image, rect = FONT.render(
                "FPS %s" % round(CLOCK.get_fps(), 2),
                fgcolor = (255, 255, 255, 255),
                style = STYLE_NORMAL,
                size = 15)
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
        SCREEN = pygame.display.set_mode(SCREENRECT.size, pygame.DOUBLEBUF, 32)
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

        c = numpy.asarray([ 128.0 / 255.0, 128.0 / 255.0, 200.0 / 255.0 ], numpy.float32)
        MOUSE_POS = [ 0, 0 ]
        FRAME = 0
        CLOCK = pygame.time.Clock()
        GAME = True
        avg = [ ]
        while GAME:

            pygame.event.pump()
            for event in pygame.event.get():
                if event.type == pygame.MOUSEMOTION:
                    MOUSE_POS = event.pos
                keys = pygame.key.get_pressed()

                if keys[ pygame.K_ESCAPE ]:
                    GAME = False
                    break

            # lit_surface, sw, sh = area24_cc(
            #     MOUSE_POS[0], MOUSE_POS[1], background_rgb, lalpha, intensity=3.0, color=c,
            #     smooth=False, saturation=False, sat_value=0.4, bloom=False)
            lit_surface, sw, sh = area24_cc(
                MOUSE_POS[ 0 ], MOUSE_POS[ 1 ], background_rgb, lalpha, intensity = 5, color = c,
                smooth = False, saturation = False, sat_value = 0.2, bloom = False, bloom_threshold = 64
            )

            if sw < lw and MOUSE_POS[ 0 ] <= lw - lw2:
                xx = 0
            else:
                xx = MOUSE_POS[ 0 ] - lw2

            if sh < lh and MOUSE_POS[ 1 ] <= lh - lh2:
                yy = 0
            else:
                yy = MOUSE_POS[ 1 ] - lh2

            SCREEN.fill((0, 0, 0))
            SCREEN.blit(lit_surface, (xx, yy))  # , special_flags=pygame.BLEND_RGBA_ADD)

            CLOCK.tick(2000)
            t = CLOCK.get_fps()
            avg.append(t)
            avg = show_fps(SCREEN, t, avg)
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
        SCREEN = pygame.display.set_mode(SCREENRECT.size, pygame.DOUBLEBUF, 32)
        pygame.init()

        background = pygame.image.load('../Assets/Aliens.jpg').convert()
        background.set_alpha(None)

        background = pygame.transform.smoothscale(background, (800, 800))
        background_rgb = cupy.asarray(pygame.surfarray.pixels3d(background), dtype = cupy.uint8)
        background_rgb = background_rgb.transpose(1, 0, 2)

        back = background.copy()
        back.set_alpha(255)

        pygame.display.set_caption("demo GPU light effect")

        light = pygame.image.load('../Assets/Radial8.png').convert_alpha()
        light = pygame.transform.smoothscale(light, (600, 600))
        lalpha = cupy.asarray(pygame.surfarray.pixels_alpha(light), dtype = cupy.uint8)

        lw, lh = light.get_size()
        lw2, lh2 = lw >> 1, lh >> 1

        c = cupy.array([ 128.0 / 255.0, 128.0 / 255.0, 200.0 / 255.0 ], numpy.float32)
        MOUSE_POS = [ 0, 0 ]
        FRAME = 0
        CLOCK = pygame.time.Clock()
        GAME = True
        avg = [ ]
        while GAME:

            pygame.event.pump()
            for event in pygame.event.get():
                if event.type == pygame.MOUSEMOTION:
                    MOUSE_POS = event.pos
                keys = pygame.key.get_pressed()

                if keys[ pygame.K_ESCAPE ]:
                    GAME = False
                    break

            lit_surface, sw, sh = area24_gpu(
                MOUSE_POS[ 0 ], MOUSE_POS[ 1 ], background_rgb, lalpha, intensity = 8.0, color = c)

            if sw < lw and MOUSE_POS[ 0 ] <= lw - lw2:
                xx = 0
            else:
                xx = MOUSE_POS[ 0 ] - lw2

            if sh < lh and MOUSE_POS[ 1 ] <= lh - lh2:
                yy = 0
            else:
                yy = MOUSE_POS[ 1 ] - lh2

            SCREEN.fill((0, 0, 0))
            SCREEN.blit(lit_surface, (xx, yy), special_flags = pygame.BLEND_RGBA_ADD)
            t = CLOCK.get_fps()
            avg.append(t)
            avg = show_fps(SCREEN, t, avg)
            pygame.display.flip()
            CLOCK.tick()
            FRAME += 1

            pygame.display.set_caption(
                "Demo light GPU %s fps"
                "(%sx%s)" % (round(CLOCK.get_fps(), 2), WIDTH, HEIGHT))


FULL_VIDEO = [ ]
PATH_TEXTURE = '../Assets/'
VIDEO_FLAGS = pygame.HWSURFACE | pygame.RLEACCEL
SCREEN_BIT_DEPTH = 32


class Testhsv_effect(unittest.TestCase):
    """
    Test hsv3d method
    """

    # pylint: disable=too-many-statements
    @staticmethod
    def runTest() -> None:

        SCREENRECT = pygame.Rect(0, 0, 800, 600)
        pygame.display.init()
        SCREEN = pygame.display.set_mode(SCREENRECT.size, pygame.DOUBLEBUF, 32)
        pygame.init()

        image = pygame.image.load('../Assets/px.png').convert_alpha()
        image = pygame.transform.smoothscale(image, (WIDTH, HEIGHT))
        image_copy = image.copy()

        angle = 0

        FRAME = 0
        CLOCK = pygame.time.Clock()
        GAME = True
        avg = [ ]

        pygame.display.set_caption("hsv_effect testing")

        while GAME:

            pygame.event.pump()
            for _ in pygame.event.get():

                keys = pygame.key.get_pressed()

                if keys[ pygame.K_ESCAPE ]:
                    GAME = False
                    break

            hsv_effect(image, angle / 360.0)

            SCREEN.fill((0, 0, 0))
            SCREEN.blit(image, (0, 0))

            t = CLOCK.get_fps()
            avg.append(t)
            avg = show_fps(SCREEN, t, avg)
            pygame.display.flip()
            CLOCK.tick()
            FRAME += 1
            angle += 0.1

            angle = angle % 360

            image = image_copy.copy()

            pygame.display.set_caption(
                "hsv_effect %s fps"
                "(%sx%s), angle: %s degrees" % (round(CLOCK.get_fps(), 2), WIDTH, HEIGHT, round(angle)))

        return


class Testhsv3d(unittest.TestCase):
    """
    Test hsv3d method
    """

    # pylint: disable=too-many-statements
    @staticmethod
    def runTest() -> None:

        SCREENRECT = pygame.Rect(0, 0, 800, 600)
        pygame.display.init()
        SCREEN = pygame.display.set_mode(SCREENRECT.size, pygame.DOUBLEBUF, 32)
        pygame.init()

        image = pygame.image.load('../Assets/px.png').convert_alpha()
        image = pygame.transform.smoothscale(image, (WIDTH, HEIGHT))
        image_copy = image.copy()
        array3d = pygame.surfarray.pixels3d(image)

        angle = 0
        FRAME = 0
        CLOCK = pygame.time.Clock()
        GAME = True
        avg = [ ]

        pygame.display.set_caption("hsv3d testing")

        while GAME:

            pygame.event.pump()
            for _ in pygame.event.get():

                keys = pygame.key.get_pressed()

                if keys[ pygame.K_ESCAPE ]:
                    GAME = False
                    break

            hsv3d(array3d, angle / 360.0)

            SCREEN.fill((0, 0, 0))
            pygame.surfarray.blit_array(SCREEN, array3d)

            t = CLOCK.get_fps()
            avg.append(t)
            avg = show_fps(SCREEN, t, avg)
            pygame.display.flip()
            CLOCK.tick()
            FRAME += 1
            angle += 0.1

            angle = angle % 360

            image = image_copy.copy()
            array3d = pygame.surfarray.pixels3d(image)

            pygame.display.set_caption(
                "hsv3d %s fps"
                "(%sx%s), angle %s degrees" % (round(CLOCK.get_fps(), 2), WIDTH, HEIGHT, round(angle)))

        return


class Testhsv1d(unittest.TestCase):
    """
    Test hsv1d method
    """

    # pylint: disable=too-many-statements
    @staticmethod
    def runTest() -> None:

        SCREENRECT = pygame.Rect(0, 0, 800, 600)
        pygame.display.init()
        SCREEN = pygame.display.set_mode(SCREENRECT.size, pygame.DOUBLEBUF, 32)
        pygame.init()

        image = pygame.image.load('../Assets/px.png').convert(24)
        image = pygame.transform.smoothscale(image, (WIDTH, HEIGHT))
        image_copy = image.copy()

        angle = 0
        FRAME = 0
        CLOCK = pygame.time.Clock()
        GAME = True
        avg = [ ]

        pygame.display.set_caption("hsv1d testing")

        while GAME:

            pygame.event.pump()
            for _ in pygame.event.get():

                keys = pygame.key.get_pressed()

                if keys[ pygame.K_ESCAPE ]:
                    GAME = False
                    break

            hsv1d(image.get_buffer(), angle / 360.0, format_32 = False)

            SCREEN.fill((0, 0, 0))
            SCREEN.blit(image, (0, 0))

            t = CLOCK.get_fps()
            avg.append(t)
            avg = show_fps(SCREEN, t, avg)
            pygame.display.flip()
            CLOCK.tick()
            FRAME += 1
            angle += 0.1

            angle = angle % 360

            image = image_copy.copy()

            pygame.display.set_caption(
                "hsv1d %s fps"
                "(%sx%s), angle %s degrees" % (round(CLOCK.get_fps(), 2), WIDTH, HEIGHT, round(angle)))

        return


class Testhsv1d_cp(unittest.TestCase):
    """
    Test hsv1d_cp method
    """

    # pylint: disable=too-many-statements
    @staticmethod
    def runTest() -> None:

        SCREENRECT = pygame.Rect(0, 0, 800, 600)
        pygame.display.init()
        SCREEN = pygame.display.set_mode(SCREENRECT.size, pygame.DOUBLEBUF, 32)
        pygame.init()

        image = pygame.image.load('../Assets/px.png').convert_alpha()
        image = pygame.transform.smoothscale(image, (WIDTH, HEIGHT))
        image_copy = image.copy()

        angle = 0
        FRAME = 0
        CLOCK = pygame.time.Clock()
        GAME = True
        avg = [ ]

        pygame.display.set_caption("hsv1d_cp testing")

        while GAME:

            pygame.event.pump()
            for _ in pygame.event.get():

                keys = pygame.key.get_pressed()

                if keys[ pygame.K_ESCAPE ]:
                    GAME = False
                    break

            arr = hsv1d_cp(image.get_buffer(), angle / 360.0, format_32 = True)
            image = pygame.image.frombuffer(arr, (WIDTH, HEIGHT), "BGRA")

            SCREEN.fill((0, 0, 0))
            SCREEN.blit(image, (0, 0))

            t = CLOCK.get_fps()
            avg.append(t)
            avg = show_fps(SCREEN, t, avg)
            pygame.display.flip()
            CLOCK.tick()
            FRAME += 1
            angle += 0.1

            angle = angle % 360

            image = image_copy.copy()

            pygame.display.set_caption(
                "hsv1d_cp %s fps"
                "(%sx%s), angle %s degrees" % (round(CLOCK.get_fps(), 2), WIDTH, HEIGHT, round(angle)))

        return


class Testhsl3d(unittest.TestCase):
    """
    Test hsl3d method
    """

    # pylint: disable=too-many-statements
    @staticmethod
    def runTest() -> None:

        SCREENRECT = pygame.Rect(0, 0, 800, 600)
        pygame.display.init()
        SCREEN = pygame.display.set_mode(SCREENRECT.size, pygame.DOUBLEBUF, 32)
        pygame.init()

        image = pygame.image.load('../Assets/px.png').convert(24)
        image = pygame.transform.smoothscale(image, (WIDTH, HEIGHT))
        image_copy = image.copy()
        # array3d = pygame.surfarray.pixels3d(image)
        array3d = pixels3d(image)

        angle = 0
        FRAME = 0
        CLOCK = pygame.time.Clock()
        GAME = True
        avg = [ ]

        pygame.display.set_caption("hsl3d testing")

        while GAME:

            pygame.event.pump()
            for _ in pygame.event.get():

                keys = pygame.key.get_pressed()

                if keys[ pygame.K_ESCAPE ]:
                    GAME = False
                    break

            hsl3d(array3d, angle / 360.0)
            del array3d
            SCREEN.fill((0, 0, 0))
            SCREEN.blit(image, (0, 0))

            t = CLOCK.get_fps()
            avg.append(t)
            avg = show_fps(SCREEN, t, avg)
            pygame.display.flip()
            CLOCK.tick()
            FRAME += 1
            angle += 0.1

            angle = angle % 360

            image = image_copy.copy()
            array3d = pygame.surfarray.pixels3d(image)

            pygame.display.set_caption(
                "hsl3d %s fps"
                "(%sx%s), angle %s degrees" % (round(CLOCK.get_fps(), 2), WIDTH, HEIGHT, round(angle)))

        return


class Testhsl1d(unittest.TestCase):
    """
    Test hsl1d method
    """

    # pylint: disable=too-many-statements
    @staticmethod
    def runTest() -> None:

        SCREENRECT = pygame.Rect(0, 0, 800, 600)
        pygame.display.init()
        SCREEN = pygame.display.set_mode(SCREENRECT.size, pygame.DOUBLEBUF, 32)
        pygame.init()

        image = pygame.image.load('../Assets/px.png').convert_alpha()
        image = pygame.transform.smoothscale(image, (WIDTH, HEIGHT))
        image_copy = image.copy()
        array3d = image.get_buffer()

        angle = 0
        FRAME = 0
        CLOCK = pygame.time.Clock()
        GAME = True
        avg = [ ]

        pygame.display.set_caption("hsl1d testing")

        while GAME:

            pygame.event.pump()
            for _ in pygame.event.get():

                keys = pygame.key.get_pressed()

                if keys[ pygame.K_ESCAPE ]:
                    GAME = False
                    break

            hsl1d(array3d, angle / 360.0, format_32 = True)
            del array3d

            SCREEN.fill((0, 0, 0))
            SCREEN.blit(image, (0, 0))

            t = CLOCK.get_fps()
            avg.append(t)
            avg = show_fps(SCREEN, t, avg)
            pygame.display.flip()
            CLOCK.tick()
            FRAME += 1
            angle += 0.1

            angle = angle % 360

            image = image_copy.copy()
            array3d = image.get_buffer()

            pygame.display.set_caption(
                "hsl1d %s fps"
                "(%sx%s), angle %s degrees" % (round(CLOCK.get_fps(), 2), WIDTH, HEIGHT, round(angle)))

        return


class Testhsl1d_cp(unittest.TestCase):
    """
    Test hsl1d_cp method
    """

    # pylint: disable=too-many-statements
    @staticmethod
    def runTest() -> None:

        SCREENRECT = pygame.Rect(0, 0, 800, 600)
        pygame.display.init()
        SCREEN = pygame.display.set_mode(SCREENRECT.size, pygame.DOUBLEBUF, 32)
        pygame.init()

        image = pygame.image.load('../Assets/px.png').convert(24)
        image = pygame.transform.smoothscale(image, (WIDTH, HEIGHT))
        image_copy = image.copy()

        angle = 0
        FRAME = 0
        CLOCK = pygame.time.Clock()
        GAME = True
        avg = [ ]

        pygame.display.set_caption("hsl1d_cp testing")

        while GAME:

            pygame.event.pump()
            for _ in pygame.event.get():

                keys = pygame.key.get_pressed()

                if keys[ pygame.K_ESCAPE ]:
                    GAME = False
                    break

            arr = hsl1d_cp(image.get_buffer(), angle / 360.0, format_32 = False)
            image = pygame.image.frombuffer(arr, (WIDTH, HEIGHT), "BGR")

            SCREEN.fill((0, 0, 0))
            SCREEN.blit(image, (0, 0))

            t = CLOCK.get_fps()
            avg.append(t)
            avg = show_fps(SCREEN, t, avg)
            pygame.display.flip()
            CLOCK.tick()
            FRAME += 1
            angle += 0.1

            angle = angle % 360

            image = image_copy.copy()

            pygame.display.set_caption(
                "hsl1d_cp %s fps"
                "(%sx%s), angle %s degrees" % (round(CLOCK.get_fps(), 2), WIDTH, HEIGHT, round(angle)))

        return


class Testinvert3d(unittest.TestCase):
    """
    Test invert3d method
    """

    # pylint: disable=too-many-statements
    @staticmethod
    def runTest() -> None:

        SCREENRECT = pygame.Rect(0, 0, 800, 600)
        pygame.display.init()
        SCREEN = pygame.display.set_mode(SCREENRECT.size, pygame.DOUBLEBUF, 32)
        pygame.init()

        image = pygame.image.load('../Assets/px.png').convert_alpha()
        image = pygame.transform.smoothscale(image, (WIDTH, HEIGHT))
        image_copy = image.copy()
        array3d = pixels3d(image)

        FRAME = 0
        CLOCK = pygame.time.Clock()
        GAME = True
        avg = [ ]

        pygame.display.set_caption("invert3d testing")

        while GAME:

            pygame.event.pump()
            for _ in pygame.event.get():

                keys = pygame.key.get_pressed()

                if keys[ pygame.K_ESCAPE ]:
                    GAME = False
                    break

            invert3d(array3d)
            del array3d
            SCREEN.fill((0, 0, 0))
            SCREEN.blit(image, (0, 0))

            t = CLOCK.get_fps()
            avg.append(t)
            avg = show_fps(SCREEN, t, avg)
            pygame.display.flip()
            CLOCK.tick()
            FRAME += 1

            image = image_copy.copy()
            array3d = pixels3d(image)

            pygame.display.set_caption(
                "invert3d %s fps"
                "(%sx%s)" % (round(CLOCK.get_fps(), 2), WIDTH, HEIGHT))

        return


class Testinvert1d(unittest.TestCase):
    """
    Test invert1d method
    """

    # pylint: disable=too-many-statements
    @staticmethod
    def runTest() -> None:

        SCREENRECT = pygame.Rect(0, 0, 800, 600)
        pygame.display.init()
        SCREEN = pygame.display.set_mode(SCREENRECT.size, pygame.DOUBLEBUF, 32)
        pygame.init()

        image = pygame.image.load('../Assets/px.png').convert(24)
        image = pygame.transform.smoothscale(image, (WIDTH, HEIGHT))
        image_copy = image.copy()

        FRAME = 0
        CLOCK = pygame.time.Clock()
        GAME = True
        avg = [ ]

        pygame.display.set_caption("invert1d testing")

        while GAME:

            pygame.event.pump()
            for _ in pygame.event.get():

                keys = pygame.key.get_pressed()

                if keys[ pygame.K_ESCAPE ]:
                    GAME = False
                    break

            invert1d(image.get_buffer(), False)

            SCREEN.fill((0, 0, 0))
            SCREEN.blit(image, (0, 0))

            t = CLOCK.get_fps()
            avg.append(t)
            avg = show_fps(SCREEN, t, avg)
            pygame.display.flip()
            CLOCK.tick()
            FRAME += 1

            image = image_copy.copy()

            pygame.display.set_caption(
                "invert1d %s fps"
                "(%sx%s)" % (round(CLOCK.get_fps(), 2), WIDTH, HEIGHT))

        return


class Testinvert1d_cp(unittest.TestCase):
    """
    Test invert1d_cp method
    """

    # pylint: disable=too-many-statements
    @staticmethod
    def runTest() -> None:

        SCREENRECT = pygame.Rect(0, 0, 800, 600)
        pygame.display.init()
        SCREEN = pygame.display.set_mode(SCREENRECT.size, pygame.DOUBLEBUF, 32)
        pygame.init()

        image = pygame.image.load('../Assets/px.png').convert_alpha()
        image = pygame.transform.smoothscale(image, (WIDTH, HEIGHT))
        image_copy = image.copy()

        FRAME = 0
        CLOCK = pygame.time.Clock()
        GAME = True
        avg = [ ]

        pygame.display.set_caption("invert1d_cp testing")

        while GAME:

            pygame.event.pump()
            for _ in pygame.event.get():

                keys = pygame.key.get_pressed()

                if keys[ pygame.K_ESCAPE ]:
                    GAME = False
                    break

            arr3d = invert1d_cp(image.get_buffer(), format_32 = True)
            image = pygame.image.frombuffer(arr3d, (WIDTH, HEIGHT), "BGRA")

            SCREEN.fill((0, 0, 0))
            SCREEN.blit(image, (0, 0))

            t = CLOCK.get_fps()
            avg.append(t)
            avg = show_fps(SCREEN, t, avg)
            pygame.display.flip()
            CLOCK.tick()
            FRAME += 1

            image = image_copy.copy()

            pygame.display.set_caption(
                "invert1d_cp %s fps"
                "(%sx%s)" % (round(CLOCK.get_fps(), 2), WIDTH, HEIGHT))

        return


class Testbrightness_exclude(unittest.TestCase):
    """
    Test brightness_exclude method
    """

    # pylint: disable=too-many-statements
    @staticmethod
    def runTest() -> None:

        SCREENRECT = pygame.Rect(0, 0, 800, 600)
        pygame.display.init()
        SCREEN = pygame.display.set_mode(SCREENRECT.size, pygame.DOUBLEBUF, 32)
        pygame.init()

        image = pygame.image.load('../Assets/px.png').convert_alpha()
        image = pygame.transform.smoothscale(image, (WIDTH, HEIGHT))
        image_copy = image.copy()

        FRAME = 0
        CLOCK = pygame.time.Clock()
        GAME = True
        avg = [ ]

        pygame.display.set_caption("brightness_exclude testing")

        while GAME:

            pygame.event.pump()
            for _ in pygame.event.get():

                keys = pygame.key.get_pressed()

                if keys[ pygame.K_ESCAPE ]:
                    GAME = False
                    break

            brightness_exclude(image, 0.5, color_ = (0, 0, 0))

            SCREEN.fill((8, 0, 0))
            SCREEN.blit(image, (0, 0))

            t = CLOCK.get_fps()
            avg.append(t)
            avg = show_fps(SCREEN, t, avg)
            pygame.display.flip()
            CLOCK.tick()
            FRAME += 1

            image = image_copy.copy()

            pygame.display.set_caption(
                "brightness_exclude %s fps"
                "(%sx%s)" % (round(CLOCK.get_fps(), 2), WIDTH, HEIGHT))

        return


class Testbrightness_bpf(unittest.TestCase):
    """
    Test brightness_bpf method
    """

    # pylint: disable=too-many-statements
    @staticmethod
    def runTest() -> None:

        SCREENRECT = pygame.Rect(0, 0, 800, 600)
        pygame.display.init()
        SCREEN = pygame.display.set_mode(SCREENRECT.size, pygame.DOUBLEBUF, 32)
        pygame.init()

        image = pygame.image.load('../Assets/px.png').convert()
        image = pygame.transform.smoothscale(image, (WIDTH, HEIGHT))
        image_copy = image.copy()

        FRAME = 0
        CLOCK = pygame.time.Clock()
        GAME = True
        avg = [ ]

        pygame.display.set_caption("brightness_bpf testing")

        while GAME:

            pygame.event.pump()
            for _ in pygame.event.get():

                keys = pygame.key.get_pressed()

                if keys[ pygame.K_ESCAPE ]:
                    GAME = False
                    break

            brightness_bpf(image, 0.5, bpf_threshold = 200)

            SCREEN.fill((8, 0, 0))
            SCREEN.blit(image, (0, 0))

            t = CLOCK.get_fps()
            avg.append(t)
            avg = show_fps(SCREEN, t, avg)
            pygame.display.flip()
            CLOCK.tick()
            FRAME += 1

            image = image_copy.copy()

            pygame.display.set_caption(
                "brightness_bpf %s fps"
                "(%sx%s)" % (round(CLOCK.get_fps(), 2), WIDTH, HEIGHT))

        return


class Testbpf(unittest.TestCase):
    """
    Test bpf method
    """

    # pylint: disable=too-many-statements
    @staticmethod
    def runTest() -> None:

        SCREENRECT = pygame.Rect(0, 0, 800, 600)
        pygame.display.init()
        SCREEN = pygame.display.set_mode(SCREENRECT.size, pygame.DOUBLEBUF, 32)
        pygame.init()

        image = pygame.image.load('../Assets/px.png').convert(24)
        image = pygame.transform.smoothscale(image, (WIDTH, HEIGHT))
        image_copy = image.copy()

        FRAME = 0
        CLOCK = pygame.time.Clock()
        GAME = True
        avg = [ ]

        pygame.display.set_caption("bpf testing")

        while GAME:

            pygame.event.pump()
            for _ in pygame.event.get():

                keys = pygame.key.get_pressed()

                if keys[ pygame.K_ESCAPE ]:
                    GAME = False
                    break

            bpf(image, threshold = 60)

            SCREEN.fill((8, 0, 0))
            SCREEN.blit(image, (0, 0))

            t = CLOCK.get_fps()
            avg.append(t)
            avg = show_fps(SCREEN, t, avg)
            pygame.display.flip()
            CLOCK.tick()
            FRAME += 1

            image = image_copy.copy()

            pygame.display.set_caption(
                "bpf %s fps"
                "(%sx%s)" % (round(CLOCK.get_fps(), 2), WIDTH, HEIGHT))

        return


class Testdithering(unittest.TestCase):
    """
    Test dithering method
    """

    # pylint: disable=too-many-statements
    @staticmethod
    def runTest() -> None:

        SCREENRECT = pygame.Rect(0, 0, 800, 600)
        pygame.display.init()
        SCREEN = pygame.display.set_mode(SCREENRECT.size, pygame.DOUBLEBUF, 32)
        pygame.init()

        image = pygame.image.load('../Assets/px.png').convert_alpha()
        image = pygame.transform.smoothscale(image, (800, 600))
        image_copy = image.copy()

        FRAME = 0
        CLOCK = pygame.time.Clock()
        GAME = True
        avg = [ ]

        pygame.display.set_caption("dithering testing")

        while GAME:

            pygame.event.pump()
            for _ in pygame.event.get():

                keys = pygame.key.get_pressed()

                if keys[ pygame.K_ESCAPE ]:
                    GAME = False
                    break

            image = dithering(image)

            SCREEN.fill((50, 0, 0))
            SCREEN.blit(image, (0, 0))

            t = CLOCK.get_fps()
            avg.append(t)
            avg = show_fps(SCREEN, t, avg)
            pygame.display.flip()
            CLOCK.tick()
            FRAME += 1

            image = image_copy.copy()

            pygame.display.set_caption(
                "dithering %s fps"
                "(%sx%s)" % (round(CLOCK.get_fps(), 2), WIDTH, HEIGHT))

        return


class Testdithering_inplace(unittest.TestCase):
    """
    Test dithering_inplace method
    """

    # pylint: disable=too-many-statements
    @staticmethod
    def runTest() -> None:

        SCREENRECT = pygame.Rect(0, 0, 800, 600)
        pygame.display.init()
        SCREEN = pygame.display.set_mode(SCREENRECT.size, pygame.DOUBLEBUF, 32)
        pygame.init()

        image = pygame.image.load('../Assets/px.png').convert(24)
        image = pygame.transform.smoothscale(image, (800, 600))
        image_copy = image.copy()

        FRAME = 0
        CLOCK = pygame.time.Clock()
        GAME = True
        avg = [ ]

        pygame.display.set_caption("dithering_inplace testing")

        while GAME:

            pygame.event.pump()
            for _ in pygame.event.get():

                keys = pygame.key.get_pressed()

                if keys[ pygame.K_ESCAPE ]:
                    GAME = False
                    break

            dithering_inplace(image)

            SCREEN.fill((50, 0, 0))
            SCREEN.blit(image, (0, 0))

            t = CLOCK.get_fps()
            avg.append(t)
            avg = show_fps(SCREEN, t, avg)
            pygame.display.flip()
            CLOCK.tick()
            FRAME += 1

            image = image_copy.copy()

            pygame.display.set_caption(
                "dithering_inplace %s fps"
                "(%sx%s)" % (round(CLOCK.get_fps(), 2), WIDTH, HEIGHT))

        return


class Testdithering_atkinson(unittest.TestCase):
    """
    Test dithering_atkinson method
    """

    # pylint: disable=too-many-statements
    @staticmethod
    def runTest() -> None:

        SCREENRECT = pygame.Rect(0, 0, 800, 600)
        pygame.display.init()
        SCREEN = pygame.display.set_mode(SCREENRECT.size, pygame.DOUBLEBUF, 32)
        pygame.init()

        image = pygame.image.load('../Assets/px.png').convert(24)
        image = pygame.transform.smoothscale(image, (800, 600))
        image_copy = image.copy()

        FRAME = 0
        CLOCK = pygame.time.Clock()
        GAME = True
        avg = [ ]

        pygame.display.set_caption("dithering_atkinson testing")

        while GAME:

            pygame.event.pump()
            for _ in pygame.event.get():

                keys = pygame.key.get_pressed()

                if keys[ pygame.K_ESCAPE ]:
                    GAME = False
                    break

            image = dithering_atkinson(image)

            SCREEN.fill((50, 0, 0))
            SCREEN.blit(image, (0, 0))

            t = CLOCK.get_fps()
            avg.append(t)
            avg = show_fps(SCREEN, t, avg)
            pygame.display.flip()
            CLOCK.tick()
            FRAME += 1

            image = image_copy.copy()

            pygame.display.set_caption(
                "dithering_atkinson %s fps"
                "(%sx%s)" % (round(CLOCK.get_fps(), 2), WIDTH, HEIGHT))

        return


class Testconvert_27(unittest.TestCase):
    """
    Test convert_27 method
    """

    # pylint: disable=too-many-statements
    @staticmethod
    def runTest() -> None:

        SCREENRECT = pygame.Rect(0, 0, 800, 600)
        pygame.display.init()
        SCREEN = pygame.display.set_mode(SCREENRECT.size, pygame.DOUBLEBUF, 32)
        pygame.init()

        image = pygame.image.load('../Assets/px.png').convert_alpha()
        image = pygame.transform.smoothscale(image, (800, 600))
        image_copy = image.copy()

        FRAME = 0
        CLOCK = pygame.time.Clock()
        GAME = True
        avg = [ ]

        pygame.display.set_caption("convert_27 testing")

        while GAME:

            pygame.event.pump()
            for _ in pygame.event.get():

                keys = pygame.key.get_pressed()

                if keys[ pygame.K_ESCAPE ]:
                    GAME = False
                    break

            convert_27(image)

            SCREEN.fill((50, 0, 0))
            SCREEN.blit(image, (0, 0))

            t = CLOCK.get_fps()
            avg.append(t)
            avg = show_fps(SCREEN, t, avg)
            pygame.display.flip()
            CLOCK.tick()
            FRAME += 1

            image = image_copy.copy()

            pygame.display.set_caption(
                "convert_27 %s fps"
                "(%sx%s)" % (round(CLOCK.get_fps(), 2), WIDTH, HEIGHT))

        return


class Testbilateral(unittest.TestCase):
    """
    Test bilateral method
    """

    # pylint: disable=too-many-statements
    @staticmethod
    def runTest() -> None:

        SCREENRECT = pygame.Rect(0, 0, 800, 600)
        pygame.display.init()
        SCREEN = pygame.display.set_mode(SCREENRECT.size, pygame.DOUBLEBUF, 32)
        pygame.init()

        image = pygame.image.load('../Assets/px.png').convert_alpha()
        image = pygame.transform.smoothscale(image, (800, 600))
        image_copy = image.copy()

        FRAME = 0
        CLOCK = pygame.time.Clock()
        GAME = True
        avg = [ ]

        pygame.display.set_caption("bilateral testing")

        while GAME:

            pygame.event.pump()
            for _ in pygame.event.get():

                keys = pygame.key.get_pressed()

                if keys[ pygame.K_ESCAPE ]:
                    GAME = False
                    break

            image = bilateral(image, 16, 18, 3)

            SCREEN.fill((50, 0, 0))
            SCREEN.blit(image, (0, 0))

            t = CLOCK.get_fps()
            avg.append(t)
            avg = show_fps(SCREEN, t, avg)
            pygame.display.flip()
            CLOCK.tick()
            FRAME += 1

            image = image_copy.copy()

            pygame.display.set_caption(
                "bilateral %s fps"
                "(%sx%s)" % (round(CLOCK.get_fps(), 2), WIDTH, HEIGHT))

        return


class Testemboss(unittest.TestCase):
    """
    Test emboss method
    """

    # pylint: disable=too-many-statements
    @staticmethod
    def runTest() -> None:

        SCREENRECT = pygame.Rect(0, 0, 800, 600)
        pygame.display.init()
        SCREEN = pygame.display.set_mode(SCREENRECT.size, pygame.DOUBLEBUF, 32)
        pygame.init()

        image = pygame.image.load('../Assets/px.png').convert_alpha()
        image = pygame.transform.smoothscale(image, (800, 600))
        image_copy = image.copy()

        FRAME = 0
        CLOCK = pygame.time.Clock()
        GAME = True
        avg = [ ]

        pygame.display.set_caption("emboss testing")

        while GAME:

            pygame.event.pump()
            for _ in pygame.event.get():

                keys = pygame.key.get_pressed()

                if keys[ pygame.K_ESCAPE ]:
                    GAME = False
                    break

            image = emboss(image, 0)

            SCREEN.fill((50, 0, 0))
            SCREEN.blit(image, (0, 0))

            t = CLOCK.get_fps()
            avg.append(t)
            avg = show_fps(SCREEN, t, avg)
            pygame.display.flip()
            CLOCK.tick()
            FRAME += 1

            image = image_copy.copy()

            pygame.display.set_caption(
                "emboss %s fps"
                "(%sx%s)" % (round(CLOCK.get_fps(), 2), WIDTH, HEIGHT))

        return


class Testemboss_gray(unittest.TestCase):
    """
    Test emboss_gray method
    """

    # pylint: disable=too-many-statements
    @staticmethod
    def runTest() -> None:

        SCREENRECT = pygame.Rect(0, 0, 800, 600)
        pygame.display.init()
        SCREEN = pygame.display.set_mode(SCREENRECT.size, pygame.DOUBLEBUF, 32)
        pygame.init()

        image = pygame.image.load('../Assets/px.png').convert_alpha()
        image = pygame.transform.smoothscale(image, (800, 600))
        image_copy = image.copy()

        FRAME = 0
        CLOCK = pygame.time.Clock()
        GAME = True
        avg = [ ]

        pygame.display.set_caption("emboss_gray testing")

        while GAME:

            pygame.event.pump()
            for _ in pygame.event.get():

                keys = pygame.key.get_pressed()

                if keys[ pygame.K_ESCAPE ]:
                    GAME = False
                    break

            image = emboss_gray(image)

            SCREEN.fill((50, 0, 0))
            SCREEN.blit(image, (0, 0))

            t = CLOCK.get_fps()
            avg.append(t)
            avg = show_fps(SCREEN, t, avg)
            pygame.display.flip()
            CLOCK.tick()
            FRAME += 1

            image = image_copy.copy()

            pygame.display.set_caption(
                "emboss_gray %s fps"
                "(%sx%s)" % (round(CLOCK.get_fps(), 2), WIDTH, HEIGHT))

        return


class Testemboss_inplace(unittest.TestCase):
    """
    Test emboss3d_inplace_c method
    """

    # pylint: disable=too-many-statements
    @staticmethod
    def runTest() -> None:

        SCREENRECT = pygame.Rect(0, 0, 800, 600)
        pygame.display.init()
        SCREEN = pygame.display.set_mode(SCREENRECT.size, pygame.DOUBLEBUF, 32)
        pygame.init()

        image = pygame.image.load('../Assets/px.png').convert(24)
        image = pygame.transform.smoothscale(image, (800, 600))
        image_copy = image.copy()
        pixel_copy = numpy.ascontiguousarray(array3d(image_copy).transpose(1, 0, 2))
        # pixel_copy = None

        FRAME = 0
        CLOCK = pygame.time.Clock()
        GAME = True
        avg = [ ]

        pygame.display.set_caption("emboss3d_inplace_c testing")

        while GAME:

            pygame.event.pump()
            for _ in pygame.event.get():

                keys = pygame.key.get_pressed()

                if keys[ pygame.K_ESCAPE ]:
                    GAME = False
                    break

            emboss_inplace(image, copy = pixel_copy)

            SCREEN.fill((50, 0, 0))
            SCREEN.blit(image, (0, 0))

            t = CLOCK.get_fps()
            avg.append(t)
            avg = show_fps(SCREEN, t, avg)
            pygame.display.flip()
            CLOCK.tick()
            FRAME += 1

            image = image_copy.copy()

            pygame.display.set_caption(
                "emboss3d_inplace_c %s fps"
                "(%sx%s)" % (round(CLOCK.get_fps(), 2), WIDTH, HEIGHT))

        return


class Testemboss1d(unittest.TestCase):
    """
    Test emboss1d method
    """

    # pylint: disable=too-many-statements
    @staticmethod
    def runTest() -> None:

        SCREENRECT = pygame.Rect(0, 0, 800, 600)
        pygame.display.init()
        SCREEN = pygame.display.set_mode(SCREENRECT.size, pygame.DOUBLEBUF, 32)
        pygame.init()

        image = pygame.image.load('../Assets/px.png').convert_alpha()
        image = pygame.transform.smoothscale(image, (800, 600))
        image_copy = image.copy()

        FRAME = 0
        CLOCK = pygame.time.Clock()
        GAME = True
        avg = [ ]

        pygame.display.set_caption("emboss1d testing")

        while GAME:

            pygame.event.pump()
            for _ in pygame.event.get():

                keys = pygame.key.get_pressed()

                if keys[ pygame.K_ESCAPE ]:
                    GAME = False
                    break

            emboss1d(800, 600, image.get_buffer(), None, True)

            SCREEN.fill((50, 0, 0))
            SCREEN.blit(image, (0, 0))

            t = CLOCK.get_fps()
            avg.append(t)
            avg = show_fps(SCREEN, t, avg)
            pygame.display.flip()
            CLOCK.tick()
            FRAME += 1

            image = image_copy.copy()

            pygame.display.set_caption(
                "emboss1d %s fps"
                "(%sx%s)" % (round(CLOCK.get_fps(), 2), WIDTH, HEIGHT))

        return


class Testbilinear(unittest.TestCase):
    """
    Test bilinear method
    """

    # pylint: disable=too-many-statements
    @staticmethod
    def runTest() -> None:

        SCREENRECT = pygame.Rect(0, 0, 800, 600)
        pygame.display.init()
        SCREEN = pygame.display.set_mode(SCREENRECT.size, pygame.DOUBLEBUF, 32)
        pygame.init()

        image = pygame.image.load('../Assets/px.png').convert_alpha()
        image = pygame.transform.smoothscale(image, (800, 600))
        image_copy = image.copy()

        FRAME = 0
        CLOCK = pygame.time.Clock()
        GAME = True
        avg = [ ]

        pygame.display.set_caption("bilinear testing")

        while GAME:

            pygame.event.pump()
            for _ in pygame.event.get():

                keys = pygame.key.get_pressed()

                if keys[ pygame.K_ESCAPE ]:
                    GAME = False
                    break

            image = bilinear(image, (600, 600))  # , 2, 2)

            SCREEN.fill((50, 0, 0))
            SCREEN.blit(image, (0, 0))

            t = CLOCK.get_fps()
            avg.append(t)
            avg = show_fps(SCREEN, t, avg)
            pygame.display.flip()
            CLOCK.tick()
            FRAME += 1

            image = image_copy.copy()

            pygame.display.set_caption(
                "bilinear %s fps"
                "(%sx%s)" % (round(CLOCK.get_fps(), 2), WIDTH, HEIGHT))
        return


class Testsepia_1d(unittest.TestCase):
    """
    Test sepia_1d method
    """

    # pylint: disable=too-many-statements
    @staticmethod
    def runTest() -> None:

        w = 800
        h = 600
        SCREENRECT = pygame.Rect(0, 0, 800, 600)
        pygame.display.init()
        SCREEN = pygame.display.set_mode(SCREENRECT.size, pygame.DOUBLEBUF, 32)
        pygame.init()

        image = pygame.image.load('../Assets/px.png').convert_alpha()
        image = pygame.transform.smoothscale(image, (800, 600))
        image_copy = image.copy()

        FRAME = 0
        CLOCK = pygame.time.Clock()
        GAME = True
        avg = [ ]

        pygame.display.set_caption("sepia_1d testing")

        sepia_1d(image.get_buffer(), False)

        while GAME:

            pygame.event.pump()
            for _ in pygame.event.get():

                keys = pygame.key.get_pressed()

                if keys[ pygame.K_ESCAPE ]:
                    GAME = False
                    break

            sepia_1d(numpy.ndarray(shape = (w * h * 4), buffer = image.get_view('1'), dtype = uint8), True)

            SCREEN.fill((50, 0, 0))
            SCREEN.blit(image, (0, 0))

            t = CLOCK.get_fps()
            avg.append(t)
            avg = show_fps(SCREEN, t, avg)
            pygame.display.flip()
            CLOCK.tick()
            FRAME += 1

            image = image_copy.copy()

            pygame.display.set_caption(
                "sepia_1d %s fps"
                "(%sx%s)" % (round(CLOCK.get_fps(), 2), WIDTH, HEIGHT))
        return


class Testsepia_3d(unittest.TestCase):
    """
    Test sepia_3d method
    """

    # pylint: disable=too-many-statements
    @staticmethod
    def runTest() -> None:

        SCREENRECT = pygame.Rect(0, 0, 800, 600)
        pygame.display.init()
        SCREEN = pygame.display.set_mode(SCREENRECT.size, pygame.DOUBLEBUF, 32)
        pygame.init()

        image = pygame.image.load('../Assets/px.png').convert(24)
        image = pygame.transform.smoothscale(image, (800, 600))
        image_copy = image.copy()

        FRAME = 0
        CLOCK = pygame.time.Clock()
        GAME = True
        avg = [ ]

        pygame.display.set_caption("sepia_3d testing")

        while GAME:

            pygame.event.pump()
            for _ in pygame.event.get():

                keys = pygame.key.get_pressed()

                if keys[ pygame.K_ESCAPE ]:
                    GAME = False
                    break
            sepia_3d(pixels3d(image))
            SCREEN.fill((50, 0, 0))
            SCREEN.blit(image, (0, 0))

            t = CLOCK.get_fps()
            avg.append(t)
            avg = show_fps(SCREEN, t, avg)
            pygame.display.flip()
            CLOCK.tick()
            FRAME += 1

            image = image_copy.copy()

            pygame.display.set_caption(
                "sepia_3d %s fps"
                "(%sx%s)" % (round(CLOCK.get_fps(), 2), WIDTH, HEIGHT))
        return


class Testmedian(unittest.TestCase):
    """
    Test median method
    """

    # pylint: disable=too-many-statements
    @staticmethod
    def runTest() -> None:

        SCREENRECT = pygame.Rect(0, 0, 800, 600)
        pygame.display.init()
        SCREEN = pygame.display.set_mode(SCREENRECT.size, pygame.DOUBLEBUF, 32)
        pygame.init()

        image = pygame.image.load('../Assets/city.jpg').convert(24)
        image = pygame.transform.smoothscale(image, (800, 600))
        image_copy = image.copy()

        FRAME = 0
        CLOCK = pygame.time.Clock()
        GAME = True
        avg = [ ]

        pygame.display.set_caption("median testing")

        while GAME:

            pygame.event.pump()
            for _ in pygame.event.get():

                keys = pygame.key.get_pressed()

                if keys[ pygame.K_ESCAPE ]:
                    GAME = False
                    break

            median(image, fast_ = False)
            SCREEN.fill((50, 0, 0))
            SCREEN.blit(image, (0, 0))

            t = CLOCK.get_fps()
            avg.append(t)
            avg = show_fps(SCREEN, t, avg)
            pygame.display.flip()
            CLOCK.tick()
            FRAME += 1

            image = image_copy.copy()

            pygame.display.set_caption(
                "median %s fps"
                "(%sx%s)" % (round(CLOCK.get_fps(), 2), WIDTH, HEIGHT))
        return

# todo does not work
class Testpainting(unittest.TestCase):
    """
    Test painting method
    """

    # pylint: disable=too-many-statements
    @staticmethod
    def runTest() -> None:

        SCREENRECT = pygame.Rect(0, 0, 800, 600)
        pygame.display.init()
        SCREEN = pygame.display.set_mode(SCREENRECT.size, pygame.DOUBLEBUF, 32)
        pygame.init()

        image = pygame.image.load('../Assets/px.png').convert(24)
        image = pygame.transform.smoothscale(image, (800, 600))
        image_copy = image.copy()

        FRAME = 0
        CLOCK = pygame.time.Clock()
        GAME = True
        avg = [ ]

        pygame.display.set_caption("painting testing")

        while GAME:

            pygame.event.pump()
            for _ in pygame.event.get():

                keys = pygame.key.get_pressed()

                if keys[ pygame.K_ESCAPE ]:
                    GAME = False
                    break

            # painting(image)
            SCREEN.fill((50, 0, 0))
            SCREEN.blit(image, (0, 0))

            t = CLOCK.get_fps()
            avg.append(t)
            avg = show_fps(SCREEN, t, avg)
            pygame.display.flip()
            CLOCK.tick()
            FRAME += 1

            image = image_copy.copy()

            pygame.display.set_caption(
                "painting %s fps"
                "(%sx%s)" % (round(CLOCK.get_fps(), 2), WIDTH, HEIGHT))
        return


class Testpixels(unittest.TestCase):
    """
    Test pixels method
    """

    # pylint: disable=too-many-statements
    @staticmethod
    def runTest() -> None:

        SCREENRECT = pygame.Rect(0, 0, 800, 600)
        pygame.display.init()
        SCREEN = pygame.display.set_mode(SCREENRECT.size, pygame.DOUBLEBUF, 32)
        pygame.init()

        image = pygame.image.load('../Assets/px.png').convert(24)
        image = pygame.transform.smoothscale(image, (800, 600))
        image_copy = image.copy()

        FRAME = 0
        CLOCK = pygame.time.Clock()
        GAME = True
        avg = [ ]

        pygame.display.set_caption("pixels testing")

        while GAME:

            pygame.event.pump()
            for _ in pygame.event.get():

                keys = pygame.key.get_pressed()

                if keys[ pygame.K_ESCAPE ]:
                    GAME = False
                    break

            pixels(image)
            SCREEN.fill((50, 0, 0))
            SCREEN.blit(image, (0, 0))

            t = CLOCK.get_fps()
            avg.append(t)
            avg = show_fps(SCREEN, t, avg)
            pygame.display.flip()
            CLOCK.tick()
            FRAME += 1

            image = image_copy.copy()

            pygame.display.set_caption(
                "pixels %s fps"
                "(%sx%s)" % (round(CLOCK.get_fps(), 2), WIDTH, HEIGHT))
        return


class Testsobel_1d(unittest.TestCase):
    """
    Test sobel_1d method
    """

    # pylint: disable=too-many-statements
    @staticmethod
    def runTest() -> None:

        w = 800
        h = 600
        SCREENRECT = pygame.Rect(0, 0, 800, 600)
        pygame.display.init()
        SCREEN = pygame.display.set_mode(SCREENRECT.size, pygame.DOUBLEBUF, 32)
        pygame.init()

        image = pygame.image.load('../Assets/px.png').convert_alpha()
        image = pygame.transform.smoothscale(image, (800, 600))
        # grey(image)
        image_copy = image.copy()

        FRAME = 0
        CLOCK = pygame.time.Clock()
        GAME = True
        avg = [ ]

        pygame.display.set_caption("sobel_1d testing")

        while GAME:

            pygame.event.pump()
            for _ in pygame.event.get():

                keys = pygame.key.get_pressed()

                if keys[ pygame.K_ESCAPE ]:
                    GAME = False
                    break

            sobel_1d(w, h, image.get_buffer(), image_copy.get_buffer(), threshold = 25.0, format_32 = True,
                     greyscale = True)

            SCREEN.fill((50, 0, 0))
            SCREEN.blit(image, (0, 0))

            t = CLOCK.get_fps()
            avg.append(t)
            avg = show_fps(SCREEN, t, avg)
            pygame.display.flip()
            CLOCK.tick()
            FRAME += 1

            image = image_copy.copy()

            pygame.display.set_caption(
                "sobel_1d %s fps"
                "(%sx%s)" % (round(CLOCK.get_fps(), 2), WIDTH, HEIGHT))
        return


class Testsharpen_1d(unittest.TestCase):
    """
    Test sharpen_1d method
    """

    # pylint: disable=too-many-statements
    @staticmethod
    def runTest() -> None:

        w = 800
        h = 600
        SCREENRECT = pygame.Rect(0, 0, 800, 600)
        pygame.display.init()
        SCREEN = pygame.display.set_mode(SCREENRECT.size, pygame.DOUBLEBUF, 32)
        pygame.init()

        image = pygame.image.load('../Assets/px.png').convert_alpha()
        image = pygame.transform.smoothscale(image, (800, 600))
        # grey(image)
        image_copy = image.copy()

        FRAME = 0
        CLOCK = pygame.time.Clock()
        GAME = True
        avg = [ ]

        pygame.display.set_caption("sharpen_1d testing")

        while GAME:

            pygame.event.pump()
            for _ in pygame.event.get():

                keys = pygame.key.get_pressed()

                if keys[ pygame.K_ESCAPE ]:
                    GAME = False
                    break

            sharpen_1d(w, h, image.get_buffer(), format_32 = True)

            SCREEN.fill((50, 0, 0))
            SCREEN.blit(image, (0, 0))

            t = CLOCK.get_fps()
            avg.append(t)
            avg = show_fps(SCREEN, t, avg)
            pygame.display.flip()
            CLOCK.tick()
            FRAME += 1

            image = image_copy.copy()

            pygame.display.set_caption(
                "sharpen_1d %s fps"
                "(%sx%s)" % (round(CLOCK.get_fps(), 2), WIDTH, HEIGHT))
        return


class Testsharpen_1d_cp(unittest.TestCase):
    """
    Test sharpen_1d_cp method
    """

    # pylint: disable=too-many-statements
    @staticmethod
    def runTest() -> None:

        w = 800
        h = 600
        SCREENRECT = pygame.Rect(0, 0, 800, 600)
        pygame.display.init()
        SCREEN = pygame.display.set_mode(SCREENRECT.size, pygame.DOUBLEBUF, 32)
        pygame.init()

        image = pygame.image.load('../Assets/px.png').convert_alpha()
        image = pygame.transform.smoothscale(image, (800, 600))
        # grey(image)
        image_copy = image.copy()

        FRAME = 0
        CLOCK = pygame.time.Clock()
        GAME = True
        avg = [ ]

        pygame.display.set_caption("sharpen_1d_cp testing")

        while GAME:

            pygame.event.pump()
            for _ in pygame.event.get():

                keys = pygame.key.get_pressed()

                if keys[ pygame.K_ESCAPE ]:
                    GAME = False
                    break

            arr = sharpen_1d_cp(w, h, image.get_buffer(), format_32 = True)
            image = pygame.image.frombuffer(arr, (800, 600), "RGBA").convert_alpha()

            SCREEN.fill((50, 0, 0))
            SCREEN.blit(image, (0, 0))

            t = CLOCK.get_fps()
            avg.append(t)
            avg = show_fps(SCREEN, t, avg)
            pygame.display.flip()
            CLOCK.tick()
            FRAME += 1

            image = image_copy.copy()

            pygame.display.set_caption(
                "sharpen_1d_cp %s fps"
                "(%sx%s)" % (round(CLOCK.get_fps(), 2), WIDTH, HEIGHT))
        return


class Testmirroring_array(unittest.TestCase):
    """
    Test mirroring_array method
    """

    # pylint: disable=too-many-statements
    @staticmethod
    def runTest() -> None:

        SCREENRECT = pygame.Rect(0, 0, 800, 600)
        pygame.display.init()
        SCREEN = pygame.display.set_mode(SCREENRECT.size, pygame.DOUBLEBUF, 32)
        pygame.init()

        image = pygame.image.load('../Assets/px.png').convert_alpha()
        image = pygame.transform.smoothscale(image, (800, 600))
        # grey(image)
        image_copy = image.copy()

        FRAME = 0
        CLOCK = pygame.time.Clock()
        GAME = True
        avg = [ ]

        pygame.display.set_caption("mirroring_array testing")

        while GAME:

            pygame.event.pump()
            for _ in pygame.event.get():

                keys = pygame.key.get_pressed()

                if keys[ pygame.K_ESCAPE ]:
                    GAME = False
                    break

            arr = mirroring_array(pixels3d(image))
            # image = pygame.image.frombuffer(arr, (800, 600), "RGB")
            image = make_surface(arr)

            SCREEN.fill((50, 0, 0))
            SCREEN.blit(image, (0, 0))

            t = CLOCK.get_fps()
            avg.append(t)
            avg = show_fps(SCREEN, t, avg)
            pygame.display.flip()
            CLOCK.tick()
            FRAME += 1

            image = image_copy.copy()

            pygame.display.set_caption(
                "mirroring_array %s fps"
                "(%sx%s)" % (round(CLOCK.get_fps(), 2), WIDTH, HEIGHT))
        return


class Testwave32(unittest.TestCase):
    """
    Test wave32
    """

    # pylint: disable=too-many-statements
    @staticmethod
    def runTest() -> None:
        """

        :return:  void
        """
        background = pygame.image.load("../Assets/px.png").convert_alpha()
        background = pygame.transform.smoothscale(background, (WIDTH, HEIGHT))
        # background.convert(32, RLEACCEL)
        image = background.copy()
        pygame.display.set_caption("Test wave32")

        frame = 0
        clock = pygame.time.Clock()
        game = True
        angle = 0
        t = time.time()
        while game:

            SCREEN.fill((50, 0, 0))
            wave32(image, angle * math.pi / 180, 10)

            pygame.event.pump()
            for _ in pygame.event.get():

                keys = pygame.key.get_pressed()

                if keys[ pygame.K_ESCAPE ]:
                    game = False

            SCREEN.blit(image, (0, 0))
            pygame.display.flip()

            clock.tick()
            frame += 1
            pygame.display.set_caption(
                "Test wave32 %s fps "
                "(%sx%s)" % (round(clock.get_fps(), 2), WIDTH, HEIGHT))

            image = background.copy()
            angle += 5
            angle %= 360
            if time.time() - t > 5:
                break


class Testsaturation1d(unittest.TestCase):
    """
    Test saturation1d
    """

    # pylint: disable=too-many-statements
    @staticmethod
    def runTest() -> None:
        """

        :return:  void
        """
        image = pygame.image.load("../Assets/px.png").convert_alpha()
        # image = pygame.image.load("../Assets/px.png").convert(24)
        image = pygame.transform.smoothscale(image, (WIDTH, HEIGHT))
        image_copy = image.copy()
        pygame.display.set_caption("Test saturation1d")

        frame = 0
        clock = pygame.time.Clock()
        game = True
        t = time.time()
        while game:

            SCREEN.fill((10, 10, 10))
            saturation1d(image.get_buffer(), 0.4, True)
            # saturation1d(image.get_buffer(), 0.4, False)

            pygame.event.pump()
            for _ in pygame.event.get():

                keys = pygame.key.get_pressed()

                if keys[ pygame.K_ESCAPE ]:
                    game = False

            SCREEN.blit(image, (0, 0))
            pygame.display.flip()

            clock.tick()
            frame += 1
            pygame.display.set_caption(
                "Test saturation1d %s fps "
                "(%sx%s)" % (round(clock.get_fps(), 2), WIDTH, HEIGHT))

            image = image_copy.copy()

            if time.time() - t > 5:
                break


class Testsaturation1d_cp(unittest.TestCase):
    """
    Test saturation1d_cp
    """

    # pylint: disable=too-many-statements
    @staticmethod
    def runTest() -> None:
        """

        :return:  void
        """
        # image = pygame.image.load("../Assets/px.png").convert(24)
        image = pygame.image.load("../Assets/px.png").convert_alpha()
        image = pygame.transform.smoothscale(image, (WIDTH, HEIGHT))
        image_copy = image.copy()
        pygame.display.set_caption("Test saturation1d_cp")

        frame = 0
        clock = pygame.time.Clock()
        game = True
        t = time.time()
        while game:

            SCREEN.fill((0, 0, 0))
            # new_buffer = saturation1d_cp(image.get_buffer(), 0.4, False)
            # image_cp = pygame.image.frombuffer(new_buffer, (WIDTH, HEIGHT), 'RGB')
            new_buffer = saturation1d_cp(image.get_buffer(), 0.4, True)
            image_cp = pygame.image.frombuffer(new_buffer, (WIDTH, HEIGHT), 'RGBA')

            pygame.event.pump()
            for _ in pygame.event.get():

                keys = pygame.key.get_pressed()

                if keys[ pygame.K_ESCAPE ]:
                    game = False

            SCREEN.blit(image_cp, (0, 0))
            pygame.display.flip()

            clock.tick()
            frame += 1
            pygame.display.set_caption(
                "Test saturation1d_cp %s fps "
                "(%sx%s)" % (round(clock.get_fps(), 2), WIDTH, HEIGHT))

            image = image_copy.copy()

            if time.time() - t > 5:
                break


class Testsharpen32(unittest.TestCase):
    """
    Test sharpen32
    """

    # pylint: disable=too-many-statements
    @staticmethod
    def runTest() -> None:
        """

        :return:  void
        """
        # image = pygame.image.load("../Assets/px.png").convert(24)
        # image = pygame.image.load("../Assets/px.png").convert()
        image = pygame.image.load("../Assets/px.png").convert_alpha()
        image = pygame.transform.smoothscale(image, (WIDTH, HEIGHT))
        image_copy = image.copy()
        pygame.display.set_caption("Test sharpen32")

        frame = 0
        clock = pygame.time.Clock()
        game = True
        t = time.time()
        while game:

            SCREEN.fill((10, 0, 0))
            sharpen32(image)

            pygame.event.pump()
            for _ in pygame.event.get():

                keys = pygame.key.get_pressed()

                if keys[ pygame.K_ESCAPE ]:
                    game = False

            SCREEN.blit(image, (0, 0))
            pygame.display.flip()

            clock.tick()
            frame += 1
            pygame.display.set_caption(
                "Test sharpen32 %s fps "
                "(%sx%s)" % (round(clock.get_fps(), 2), WIDTH, HEIGHT))

            image = image_copy.copy()

            if time.time() - t > 5:
                break


class Testdithering_atkinson1d(unittest.TestCase):
    """
    Test dithering_atkinson method
    """

    # pylint: disable=too-many-statements
    @staticmethod
    def runTest() -> None:

        SCREENRECT = pygame.Rect(0, 0, 800, 600)
        pygame.display.init()
        SCREEN = pygame.display.set_mode(SCREENRECT.size, pygame.DOUBLEBUF, 32)
        pygame.init()

        # image = pygame.image.load('../Assets/px.png').convert_alpha()
        image = pygame.image.load('../Assets/px.png').convert(24)
        image = pygame.transform.smoothscale(image, (800, 600))
        image_copy = image.copy()

        FRAME = 0
        CLOCK = pygame.time.Clock()
        GAME = True
        avg = [ ]

        pygame.display.set_caption("dithering_atkinson1d testing")

        while GAME:

            pygame.event.pump()
            for _ in pygame.event.get():

                keys = pygame.key.get_pressed()

                if keys[ pygame.K_ESCAPE ]:
                    GAME = False
                    break

            dithering_atkinson1d(800, 600, image.get_buffer(), False)

            SCREEN.fill((50, 0, 0))
            SCREEN.blit(image, (0, 0))

            t = CLOCK.get_fps()
            avg.append(t)
            avg = show_fps(SCREEN, t, avg)
            pygame.display.flip()
            CLOCK.tick()
            FRAME += 1

            image = image_copy.copy()

            pygame.display.set_caption(
                "dithering_atkinson1d %s fps"
                "(%sx%s)" % (round(CLOCK.get_fps(), 2), WIDTH, HEIGHT))

        return


class Testpixelation(unittest.TestCase):
    """
    Test pixelation method
    """

    # pylint: disable=too-many-statements
    @staticmethod
    def runTest() -> None:

        SCREENRECT = pygame.Rect(0, 0, 800, 600)
        pygame.display.init()
        SCREEN = pygame.display.set_mode(SCREENRECT.size, pygame.DOUBLEBUF, 32)
        pygame.init()

        image = pygame.image.load('../Assets/px.png').convert_alpha()
        # image = pygame.image.load('../Assets/px.png').convert(24)
        image = pygame.transform.smoothscale(image, (800, 600))
        image_copy = image.copy()

        FRAME = 0
        CLOCK = pygame.time.Clock()
        GAME = True
        avg = [ ]

        pygame.display.set_caption("pixelation testing")

        while GAME:

            pygame.event.pump()
            for _ in pygame.event.get():

                keys = pygame.key.get_pressed()

                if keys[ pygame.K_ESCAPE ]:
                    GAME = False
                    break

            image = pixelation(image)

            SCREEN.fill((50, 0, 0))
            SCREEN.blit(image, (0, 0))

            t = CLOCK.get_fps()
            avg.append(t)
            avg = show_fps(SCREEN, t, avg)
            pygame.display.flip()
            CLOCK.tick()
            FRAME += 1

            image = image_copy.copy()

            pygame.display.set_caption(
                "pixelation %s fps"
                "(%sx%s)" % (round(CLOCK.get_fps(), 2), WIDTH, HEIGHT))

        return


class Testblend1d(unittest.TestCase):
    """
    Test blend1d method
    """

    # pylint: disable=too-many-statements
    @staticmethod
    def runTest() -> None:
        """

        :return:  void
        """

        # Load the background image
        BACKGROUND = pygame.image.load("../Assets/background.jpg").convert(24)
        BACKGROUND = pygame.transform.smoothscale(BACKGROUND, (WIDTH, HEIGHT))

        DESTINATION = pygame.image.load("../Assets/Aliens.jpg").convert(24)
        DESTINATION = pygame.transform.smoothscale(DESTINATION, (WIDTH, HEIGHT))
        assert BACKGROUND.get_size() == DESTINATION.get_size()

        pygame.display.set_caption("demo blend inplace")

        FRAME = 0
        CLOCK = pygame.time.Clock()
        GAME = True
        VALUE = 0
        V = +0.2

        while GAME:

            pygame.event.pump()
            for _ in pygame.event.get():

                keys = pygame.key.get_pressed()

                if keys[ pygame.K_ESCAPE ]:
                    GAME = False
                    break

            transition = blend1d(800, 600,
                                 source = BACKGROUND.get_buffer(), destination = DESTINATION.get_buffer(),
                                 percentage = VALUE,
                                 modes = 'RGB(X)', format_32 = False)

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


class Testblend1d(unittest.TestCase):
    """
    Test blend_inplace method
    """

    # pylint: disable=too-many-statements
    @staticmethod
    def runTest() -> None:
        """

        :return:  void
        """

        # Load the background image
        BACKGROUND = pygame.image.load("../Assets/background.jpg").convert(24)
        BACKGROUND = pygame.transform.smoothscale(BACKGROUND, (WIDTH, HEIGHT))

        DESTINATION = pygame.image.load("../Assets/Aliens.jpg").convert(24)
        DESTINATION = pygame.transform.smoothscale(DESTINATION, (WIDTH, HEIGHT))
        assert BACKGROUND.get_size() == DESTINATION.get_size()
        DESTINATION_CP = DESTINATION.copy()

        pygame.display.set_caption("demo blend_inplace")

        FRAME = 0
        CLOCK = pygame.time.Clock()
        GAME = True
        VALUE = 0
        V = +0.2

        while GAME:

            pygame.event.pump()
            for _ in pygame.event.get():

                keys = pygame.key.get_pressed()

                if keys[ pygame.K_ESCAPE ]:
                    GAME = False
                    break

            blend_inplace(DESTINATION, BACKGROUND, percentage = VALUE)

            SCREEN.blit(DESTINATION, (0, 0))

            pygame.display.flip()
            CLOCK.tick()
            FRAME += 1

            pygame.display.set_caption(
                "Demo blend_inplace %s percent; %s fps"
                "(%sx%s)" % (round(VALUE, 2), round(CLOCK.get_fps(), 2), WIDTH, HEIGHT))
            VALUE += V

            if VALUE >= 100:
                VALUE = 100
                V = -0.2
            if VALUE <= 0:
                VALUE = 0
                V = 0.2

            DESTINATION = DESTINATION_CP.copy()


class Testarea24_c(unittest.TestCase):
    """
    Test area24_c
    """

    # pylint: disable=too-many-statements
    @staticmethod
    def runTest() -> None:
        """

        :return:  void
        """

        SCREENRECT = pygame.Rect(0, 0, 800, 600)
        pygame.display.init()
        SCREEN = pygame.display.set_mode(SCREENRECT.size, pygame.DOUBLEBUF, 32)
        pygame.init()
        background = pygame.image.load('../Assets/Aliens.jpg').convert()
        background.set_alpha(None)

        background = pygame.transform.smoothscale(background, (800, 800))
        background_rgb = pygame.surfarray.pixels3d(background)

        pygame.display.set_caption("demo area24_c")

        light = pygame.image.load('../Assets/Radial8.png').convert_alpha()
        light = pygame.transform.smoothscale(light, (400, 400))
        lw, lh = light.get_size()
        lw2, lh2 = lw >> 1, lh >> 1

        lalpha = pygame.surfarray.pixels_alpha(light)

        c = numpy.array([ 128.0 / 255.0, 128.0 / 255.0, 200.0 / 255.0 ], numpy.float32)
        MOUSE_POS = [ 0, 0 ]
        FRAME = 0
        CLOCK = pygame.time.Clock()
        GAME = True
        avg = [ ]
        while GAME:

            pygame.event.pump()
            for event in pygame.event.get():
                if event.type == pygame.MOUSEMOTION:
                    MOUSE_POS = event.pos
                keys = pygame.key.get_pressed()

                if keys[ pygame.K_ESCAPE ]:
                    GAME = False
                    break

            lit_surface, sw, sh = render_light_effect24(
                MOUSE_POS[ 0 ],
                MOUSE_POS[ 1 ],
                background_rgb,
                lalpha,
                intensity = 5,
                color = c,
                smooth = False,
                saturation = False,
                sat_value = 0.2,
                bloom = False,
                threshold = 64,
                heat = False,
                frequency = 1
            )

            if sw < lw and MOUSE_POS[ 0 ] <= lw - lw2:
                xx = 0
            else:
                xx = MOUSE_POS[ 0 ] - lw2

            if sh < lh and MOUSE_POS[ 1 ] <= lh - lh2:
                yy = 0
            else:
                yy = MOUSE_POS[ 1 ] - lh2

            SCREEN.fill((0, 0, 0))
            SCREEN.blit(lit_surface, (xx, yy))  # , special_flags=pygame.BLEND_RGBA_ADD)

            CLOCK.tick(2000)
            t = CLOCK.get_fps()
            avg.append(t)
            avg = show_fps(SCREEN, t, avg)
            pygame.display.flip()

            FRAME += 1

            pygame.display.set_caption(
                "Test area24_c %s fps"
                "(%sx%s)" % (round(CLOCK.get_fps(), 2), WIDTH, HEIGHT))


class Testshader_bloom_fast1(unittest.TestCase):
    """
    Test shader_bloom_fast1
    """

    # pylint: disable=too-many-statements
    @staticmethod
    def runTest() -> None:
        """

        :return:  void
        """

        SCREENRECT = pygame.Rect(0, 0, 800, 600)
        pygame.display.init()
        SCREEN = pygame.display.set_mode(SCREENRECT.size, pygame.DOUBLEBUF, 32)
        pygame.init()

        try:
            image = pygame.image.load('../Assets/Aliens.jpg').convert()
        except:
            exit()

        image = pygame.transform.smoothscale(image, (800, 600))
        image_cp = image.copy()

        try:
            mask = pygame.image.load('../Assets/alpha.png').convert_alpha()
        except:
            sys.exit()

        mask = pygame.transform.smoothscale(mask, (800, 600))
        mask_array = pygame.surfarray.pixels_alpha(mask)
        mask_array = binary_mask(mask_array)

        pygame.display.set_caption("demo shader_bloom_fast1")

        FRAME = 0
        CLOCK = pygame.time.Clock()
        GAME = True
        avg = [ ]
        while GAME:

            pygame.event.pump()
            for _ in pygame.event.get():

                keys = pygame.key.get_pressed()

                if keys[ pygame.K_ESCAPE ]:
                    GAME = False
                    break

            shader_bloom_fast1(image, mask_ = mask_array)

            SCREEN.fill((0, 0, 0))
            SCREEN.blit(image, (0, 0))

            CLOCK.tick(2000)
            t = CLOCK.get_fps()
            avg.append(t)
            avg = show_fps(SCREEN, t, avg)
            pygame.display.flip()

            FRAME += 1

            pygame.display.set_caption(
                "Test shader_bloom_fast1 %s fps"
                "(%sx%s)" % (round(CLOCK.get_fps(), 2), WIDTH, HEIGHT))
            image = image_cp.copy()


class Testswirl32(unittest.TestCase):
    """
    Test swirl32
    """

    # pylint: disable=too-many-statements
    @staticmethod
    def runTest() -> None:
        """

        :return:  void
        """

        SCREENRECT = pygame.Rect(0, 0, 800, 600)
        pygame.display.init()
        SCREEN = pygame.display.set_mode(SCREENRECT.size, pygame.DOUBLEBUF, 32)
        pygame.init()
        image = pygame.image.load('../Assets/px.png').convert(24)
        image = pygame.transform.smoothscale(image, (800, 600))
        image_cp = image.copy()

        pygame.display.set_caption("demo swirl32")

        FRAME = 0
        CLOCK = pygame.time.Clock()
        GAME = True
        avg = [ ]
        while GAME:

            pygame.event.pump()
            for _ in pygame.event.get():

                keys = pygame.key.get_pressed()

                if keys[ pygame.K_ESCAPE ]:
                    GAME = False
                    break

            swirl32(image, FRAME)

            SCREEN.fill((10, 10, 50))
            SCREEN.blit(image, (0, 0))

            CLOCK.tick(2000)
            t = CLOCK.get_fps()
            avg.append(t)
            avg = show_fps(SCREEN, t, avg)
            pygame.display.flip()

            FRAME += 1

            pygame.display.set_caption(
                "Test swirl32 %s fps"
                "(%sx%s)" % (round(CLOCK.get_fps(), 2), WIDTH, HEIGHT))

            image = image_cp.copy()


class Testbrightness1d_copy(unittest.TestCase):
    """
    Test brightness1d_copy
    """

    # pylint: disable=too-many-statements
    @staticmethod
    def runTest() -> None:
        """

        :return:  void
        """

        SCREENRECT = pygame.Rect(0, 0, 800, 600)
        pygame.display.init()
        SCREEN = pygame.display.set_mode(SCREENRECT.size, pygame.DOUBLEBUF, 32)
        pygame.init()

        background = pygame.image.load('../Assets/px.png').convert_alpha()
        background = pygame.transform.smoothscale(background, (800, 600))
        background_cp = background.copy()
        pygame.display.set_caption("demo brightness1d_copy")

        FRAME = 0
        CLOCK = pygame.time.Clock()
        GAME = True
        avg = [ ]
        while GAME:

            pygame.event.pump()
            for _ in pygame.event.get():

                keys = pygame.key.get_pressed()

                if keys[ pygame.K_ESCAPE ]:
                    GAME = False
                    break

            array_bck = brightness1d_copy(background.get_buffer(), 0.1, True)
            background = pygame.image.frombuffer(array_bck, (800, 600), 'BGRA')

            SCREEN.fill((10, 10, 50))
            SCREEN.blit(background, (0, 0))

            CLOCK.tick(2000)
            t = CLOCK.get_fps()
            avg.append(t)
            avg = show_fps(SCREEN, t, avg)
            pygame.display.flip()

            FRAME += 1

            pygame.display.set_caption(
                "Test brightness1d_copy %s fps"
                "(%sx%s)" % (round(CLOCK.get_fps(), 2), WIDTH, HEIGHT))

            background = background_cp.copy()


class Testwave_static(unittest.TestCase):
    """
    Test wave_static
    """

    # pylint: disable=too-many-statements
    @staticmethod
    def runTest() -> None:
        """

        :return:  void
        """

        SCREENRECT = pygame.Rect(0, 0, 800, 600)
        pygame.display.init()
        SCREEN = pygame.display.set_mode(SCREENRECT.size, pygame.DOUBLEBUF, 32)
        pygame.init()

        background = pygame.image.load('../Assets/px.png').convert(24)
        background = pygame.transform.smoothscale(background, (800, 600))
        background_cp = background.copy()
        pygame.display.set_caption("demo wave_static")

        FRAME = 0
        CLOCK = pygame.time.Clock()
        GAME = True
        avg = [ ]
        while GAME:

            pygame.event.pump()
            for _ in pygame.event.get():

                keys = pygame.key.get_pressed()

                if keys[ pygame.K_ESCAPE ]:
                    GAME = False
                    break

            wave_static(background, pixels3d(background_cp), FRAME * math.pi / 180, 5)

            # SCREEN.fill((10, 10, 50))
            SCREEN.blit(background, (0, 0))

            CLOCK.tick(2000)
            t = CLOCK.get_fps()
            avg.append(t)
            avg = show_fps(SCREEN, t, avg)
            pygame.display.flip()

            FRAME += 1

            pygame.display.set_caption(
                "Test wave_static %s fps"
                "(%sx%s)" % (round(CLOCK.get_fps(), 2), WIDTH, HEIGHT))

            background = background_cp.copy()


#
# class Testsaturation_mask(unittest.TestCase):
#     """
#     Test saturation_mask
#     """
#
#     # pylint: disable=too-many-statements
#     @staticmethod
#     def runTest() -> None:
#         """
#
#         :return:  void
#         """
#
#         SCREENRECT = pygame.Rect(0, 0, 800, 600)
#         pygame.display.init()
#         SCREEN = pygame.display.set_mode(SCREENRECT.size, pygame.DOUBLEBUF, 32)
#         pygame.init()
#
#
#         background = pygame.image.load('../Assets/px.png').convert(24)
#         background = pygame.transform.smoothscale(background, (800, 600))
#         background_cp = background.copy()
#         pygame.display.set_caption("demo saturation_mask")
#
#         mask = pygame.image.load('../Assets/alpha.png').convert_alpha()
#         mask = pygame.transform.smoothscale(mask, (800, 600))
#         mask_array = pygame.surfarray.pixels_alpha(mask)
#         mask_array = binary_mask(mask_array)
#
#         FRAME = 0
#         CLOCK = pygame.time.Clock()
#         GAME = True
#         avg = []
#         while GAME:
#
#             pygame.event.pump()
#             for _ in pygame.event.get():
#                 if event.type == pygame.MOUSEMOTION:
#                     MOUSE_POS = event.pos
#                 keys = pygame.key.get_pressed()
#
#                 if keys[pygame.K_ESCAPE]:
#                     GAME = False
#                     break
#
#             background = saturation_mask(pixels3d(background), 0.5, mask_array)
#
#             #SCREEN.fill((10, 10, 50))
#             SCREEN.blit(background, (0, 0))
#
#             CLOCK.tick(2000)
#             t = CLOCK.get_fps()
#             avg.append(t)
#             avg=show_fps(SCREEN, t, avg)
#             pygame.display.flip()
#
#             FRAME += 1
#
#             pygame.display.set_caption(
#                 "Test saturation_mask %s fps"
#                 "(%sx%s)" % (round(CLOCK.get_fps(), 2), WIDTH, HEIGHT))
#
#             background = background_cp.copy()


#
# class Testsaturation_mask_inplace(unittest.TestCase):
#     """
#     Test saturation_mask_inplace
#     """
#
#     # pylint: disable=too-many-statements
#     @staticmethod
#     def runTest() -> None:
#         """
#
#         :return:  void
#         """
#
#         SCREENRECT = pygame.Rect(0, 0, 800, 600)
#         pygame.display.init()
#         SCREEN = pygame.display.set_mode(SCREENRECT.size, pygame.DOUBLEBUF, 32)
#         pygame.init()
#
#
#         background = pygame.image.load('../Assets/px.png').convert(24)
#         background = pygame.transform.smoothscale(background, (800, 600))
#         background_cp = background.copy()
#         pygame.display.set_caption("demo saturation_mask_inplace")
#
#         mask = pygame.image.load('../Assets/alpha.png').convert_alpha()
#         mask = pygame.transform.smoothscale(mask, (800, 600))
#         mask_array = pygame.surfarray.pixels_alpha(mask)
#         mask_array = binary_mask(mask_array).T
#
#         FRAME = 0
#         CLOCK = pygame.time.Clock()
#         GAME = True
#         avg = []
#         while GAME:
#
#             pygame.event.pump()
#             for _ in pygame.event.get():
#                 if event.type == pygame.MOUSEMOTION:
#                     MOUSE_POS = event.pos
#                 keys = pygame.key.get_pressed()
#
#                 if keys[pygame.K_ESCAPE]:
#                     GAME = False
#                     break
#
#             saturation_mask_inplace(pixels3d(background), 0.5, mask_array, HEIGHT, WIDTH)
#
#             #SCREEN.fill((10, 10, 50))
#             SCREEN.blit(background, (0, 0))
#
#             CLOCK.tick(2000)
#             t = CLOCK.get_fps()
#             avg.append(t)
#             avg=show_fps(SCREEN, t, avg)
#             pygame.display.flip()
#
#             FRAME += 1
#
#             pygame.display.set_caption(
#                 "Test saturation_mask_inplace %s fps"
#                 "(%sx%s)" % (round(CLOCK.get_fps(), 2), WIDTH, HEIGHT))
#
#             background = background_cp.copy()


class Testalpha_blending(unittest.TestCase):
    """
    Test alpha_blending
    """

    # pylint: disable=too-many-statements
    @staticmethod
    def runTest() -> None:
        """

        :return:  void
        """

        SCREENRECT = pygame.Rect(0, 0, 800, 600)
        pygame.display.init()
        SCREEN = pygame.display.set_mode(SCREENRECT.size, pygame.DOUBLEBUF, 32)
        pygame.init()

        source = pygame.image.load('../Assets/px.png').convert_alpha()
        source = pygame.transform.smoothscale(source, (800, 600))

        pygame.display.set_caption("demo alpha_blending")

        try:
            destination = pygame.image.load('../Assets/teslaColor03_m.png').convert_alpha()
        except:
            sys.exit()

        destination = pygame.transform.smoothscale(destination, (800, 600))

        FRAME = 0
        CLOCK = pygame.time.Clock()
        GAME = True
        avg = [ ]
        while GAME:

            pygame.event.pump()
            for _ in pygame.event.get():

                keys = pygame.key.get_pressed()

                if keys[ pygame.K_ESCAPE ]:
                    GAME = False
                    break

            new_image = alpha_blending(source, destination)

            SCREEN.fill((10, 10, 50))
            SCREEN.blit(new_image, (0, 0))

            CLOCK.tick(2000)
            t = CLOCK.get_fps()
            avg.append(t)
            avg = show_fps(SCREEN, t, avg)
            pygame.display.flip()

            FRAME += 1

            pygame.display.set_caption(
                "Test alpha_blending %s fps"
                "(%sx%s)" % (round(CLOCK.get_fps(), 2), WIDTH, HEIGHT))

            # background = background_cp.copy()


class Testalpha_blending_inplace(unittest.TestCase):
    """
    Test alpha_blending_inplace
    """

    # pylint: disable=too-many-statements
    @staticmethod
    def runTest() -> None:
        """

        :return:  void
        """

        SCREENRECT = pygame.Rect(0, 0, 800, 600)
        pygame.display.init()
        SCREEN = pygame.display.set_mode(SCREENRECT.size, pygame.DOUBLEBUF, 32)
        pygame.init()

        source = pygame.image.load('../Assets/px.png').convert_alpha()
        source = pygame.transform.smoothscale(source, (800, 600))

        pygame.display.set_caption("demo alpha_blending_inplace")

        try:
            destination = pygame.image.load('../Assets/teslaColor03_m.png').convert_alpha()
        except:
            sys.exit()

        destination = pygame.transform.smoothscale(destination, (800, 600))
        destination_cp = destination.copy()

        FRAME = 0
        CLOCK = pygame.time.Clock()
        GAME = True
        avg = [ ]
        while GAME:

            pygame.event.pump()
            for _ in pygame.event.get():

                keys = pygame.key.get_pressed()

                if keys[ pygame.K_ESCAPE ]:
                    GAME = False
                    break

            alpha_blending_inplace(source, destination)

            SCREEN.fill((10, 10, 50))
            SCREEN.blit(destination, (0, 0))

            CLOCK.tick(2000)
            t = CLOCK.get_fps()
            avg.append(t)
            avg = show_fps(SCREEN, t, avg)
            pygame.display.flip()

            FRAME += 1

            pygame.display.set_caption(
                "Test alpha_blending_inplace %s fps"
                "(%sx%s)" % (round(CLOCK.get_fps(), 2), WIDTH, HEIGHT))

            destination = destination_cp.copy()


class Testzoom_inplace(unittest.TestCase):
    """
    Test zoom_inplace
    """

    # pylint: disable=too-many-statements
    @staticmethod
    def runTest() -> None:
        """

        :return:  void
        """

        SCREENRECT = pygame.Rect(0, 0, 800, 600)
        pygame.display.init()
        SCREEN = pygame.display.set_mode(SCREENRECT.size, pygame.DOUBLEBUF, 32)
        pygame.init()

        source = pygame.image.load('../Assets/px.png').convert_alpha()
        source = pygame.transform.smoothscale(source, (800, 600))

        pygame.display.set_caption("demo zoom_inplace")

        # zoom_inplace(source, 40, 30, 0.8)

        FRAME = 0
        CLOCK = pygame.time.Clock()
        GAME = True
        avg = [ ]
        while GAME:

            pygame.event.pump()
            for _ in pygame.event.get():

                keys = pygame.key.get_pressed()

                if keys[ pygame.K_ESCAPE ]:
                    GAME = False
                    break

            zoom_inplace(source, 400, 300, 0.9)

            SCREEN.fill((10, 10, 50))
            SCREEN.blit(source, (0, 0))

            CLOCK.tick(2000)
            t = CLOCK.get_fps()
            avg.append(t)
            avg = show_fps(SCREEN, t, avg)
            pygame.display.flip()

            FRAME += 1

            pygame.display.set_caption(
                "Test zoom_inplace %s fps"
                "(%sx%s)" % (round(CLOCK.get_fps(), 2), WIDTH, HEIGHT))

            # source = source_cp.copy()


class Testzoom(unittest.TestCase):
    """
    Test zoom
    """

    # pylint: disable=too-many-statements
    @staticmethod
    def runTest() -> None:
        """

        :return:  void
        """

        SCREENRECT = pygame.Rect(0, 0, 800, 600)
        pygame.display.init()
        SCREEN = pygame.display.set_mode(SCREENRECT.size, pygame.DOUBLEBUF, 32)
        pygame.init()

        source = pygame.image.load('../Assets/px.png').convert_alpha()
        source = pygame.transform.smoothscale(source, (800, 600))

        pygame.display.set_caption("demo zoom")

        FRAME = 0
        CLOCK = pygame.time.Clock()
        GAME = True
        avg = [ ]
        while GAME:

            pygame.event.pump()
            for _ in pygame.event.get():

                keys = pygame.key.get_pressed()

                if keys[ pygame.K_ESCAPE ]:
                    GAME = False
                    break

            z = 0.9999 - ((FRAME % 255) / 255.0)
            surf = zoom(source, 400, 300, z)

            SCREEN.fill((10, 10, 50))
            SCREEN.blit(surf, (0, 0))

            CLOCK.tick(2000)
            t = CLOCK.get_fps()
            avg.append(t)
            avg = show_fps(SCREEN, t, avg)
            pygame.display.flip()

            FRAME += 1

            pygame.display.set_caption(
                "Test zoom %s fps"
                "(%sx%s)" % (round(CLOCK.get_fps(), 2), WIDTH, HEIGHT))

            # source = source_cp.copy()


#
# class Testfiltering_inplace_c(unittest.TestCase):
#     """
#     Test filtering_inplace_c
#     """
#
#     # pylint: disable=too-many-statements
#     @staticmethod
#     def runTest() -> None:
#         """
#
#         :return:  void
#         """
#
#         SCREENRECT = pygame.Rect(0, 0, 800, 600)
#         pygame.display.init()
#         SCREEN = pygame.display.set_mode(SCREENRECT.size, pygame.DOUBLEBUF, 32)
#         pygame.init()
#
#
#         source = pygame.image.load('../Assets/px.png').convert(24)
#         source = pygame.transform.smoothscale(source, (800, 600))
#         source_cp = source.copy()
#
#         pygame.display.set_caption("demo filtering_inplace_c")
#
#         mask = pygame.image.load('../Assets/alpha.png').convert_alpha()
#         mask = pygame.transform.smoothscale(mask, (800, 600))
#         mask_array = pygame.surfarray.pixels_alpha(mask)
#         # mask_array = binary_mask(mask_array)
#
#         FRAME = 0
#         CLOCK = pygame.time.Clock()
#         GAME = True
#         avg = []
#         while GAME:
#
#             pygame.event.pump()
#             for _ in pygame.event.get():
#                 if event.type == pygame.MOUSEMOTION:
#                     MOUSE_POS = event.pos
#                 keys = pygame.key.get_pressed()
#
#                 if keys[pygame.K_ESCAPE]:
#                     GAME = False
#                     break
#
#             filtering_inplace_c(source, mask_array)
#
#             SCREEN.fill((10, 10, 50))
#             SCREEN.blit(source, (0, 0))
#
#             CLOCK.tick(2000)
#             t = CLOCK.get_fps()
#             avg.append(t)
#             avg=show_fps(SCREEN, t, avg)
#             pygame.display.flip()
#
#             FRAME += 1
#
#             pygame.display.set_caption(
#                 "Test filtering_inplace_c %s fps"
#                 "(%sx%s)" % (round(CLOCK.get_fps(), 2), WIDTH, HEIGHT))
#
#             source = source_cp.copy()
#
#
#
# class Testfiltering1d_inplace_c(unittest.TestCase):
#     """
#     Test filtering1d_inplace_c
#     """
#
#     # pylint: disable=too-many-statements
#     @staticmethod
#     def runTest() -> None:
#         """
#
#         :return:  void
#         """
#
#         SCREENRECT = pygame.Rect(0, 0, 800, 600)
#         pygame.display.init()
#         SCREEN = pygame.display.set_mode(SCREENRECT.size, pygame.DOUBLEBUF, 32)
#         pygame.init()
#
#
#         source = pygame.image.load('../Assets/px.png').convert()
#         source = pygame.transform.smoothscale(source, (800, 600))
#         source_cp = source.copy()
#
#         pygame.display.set_caption("demo filtering1d_inplace_c")
#
#         mask = pygame.image.load('../Assets/alpha.png').convert_alpha()
#         mask = pygame.transform.smoothscale(mask, (800, 600))
#
#
#         FRAME = 0
#         CLOCK = pygame.time.Clock()
#         GAME = True
#         avg = []
#         while GAME:
#
#             pygame.event.pump()
#             for _ in pygame.event.get():
#                 if event.type == pygame.MOUSEMOTION:
#                     MOUSE_POS = event.pos
#                 keys = pygame.key.get_pressed()
#
#                 if keys[pygame.K_ESCAPE]:
#                     GAME = False
#                     break
#
#             filtering1d_inplace_c(source, mask)
#
#             SCREEN.fill((10, 10, 50))
#             SCREEN.blit(source, (0, 0))
#
#             CLOCK.tick(2000)
#             t = CLOCK.get_fps()
#             avg.append(t)
#             avg=show_fps(SCREEN, t, avg)
#             pygame.display.flip()
#
#             FRAME += 1
#
#             pygame.display.set_caption(
#                 "Test filtering1d_inplace_c %s fps"
#                 "(%sx%s)" % (round(CLOCK.get_fps(), 2), WIDTH, HEIGHT))
#
#             source = source_cp.copy()


class Testbloom_effect_array24_c2(unittest.TestCase):
    """
    Test bloom_effect_array24_c2
    """

    # pylint: disable=too-many-statements
    @staticmethod
    def runTest() -> None:
        """

        :return:  void
        """

        SCREENRECT = pygame.Rect(0, 0, 800, 600)
        pygame.display.init()
        SCREEN = pygame.display.set_mode(SCREENRECT.size, pygame.DOUBLEBUF, 32)
        pygame.init()

        source = pygame.image.load('../Assets/px.png').convert(24)
        source = pygame.transform.smoothscale(source, (800, 600))
        source_cp = source.copy()

        pygame.display.set_caption("demo bloom_effect_array24_c2")

        try:
            mask = pygame.image.load('../Assets/alpha.png').convert_alpha()
        except:
            sys.exit()

        mask = pygame.transform.smoothscale(mask, (800, 600))
        # mask = numpy.asarray(pygame.surfarray.pixels_alpha(mask) / 255.0, dtype=numpy.float32)

        FRAME = 0
        CLOCK = pygame.time.Clock()
        GAME = True
        avg = [ ]
        while GAME:

            pygame.event.pump()
            for _ in pygame.event.get():

                keys = pygame.key.get_pressed()

                if keys[ pygame.K_ESCAPE ]:
                    GAME = False
                    break

            source = bloom_effect_array24_c2(source, 25, 1, mask, True)

            SCREEN.fill((10, 10, 50))
            SCREEN.blit(source, (0, 0))

            CLOCK.tick(2000)
            t = CLOCK.get_fps()
            avg.append(t)
            avg = show_fps(SCREEN, t, avg)
            pygame.display.flip()

            FRAME += 1

            pygame.display.set_caption(
                "Test bloom_effect_array24_c2 %s fps"
                "(%sx%s)" % (round(CLOCK.get_fps(), 2), WIDTH, HEIGHT))

            source = source_cp.copy()


# class Testbpf24_c2(unittest.TestCase):
#     """
#     Test bpf24_c2
#     """
#
#     # pylint: disable=too-many-statements
#     @staticmethod
#     def runTest() -> None:
#         """
#
#         :return:  void
#         """
#
#         SCREENRECT = pygame.Rect(0, 0, 800, 600)
#         pygame.display.init()
#         SCREEN = pygame.display.set_mode(SCREENRECT.size, pygame.DOUBLEBUF, 32)
#         pygame.init()
#
#
#         source = pygame.image.load('../Assets/px.png').convert()
#         source = pygame.transform.smoothscale(source, (800, 600))
#         source_cp = source.copy()
#
#         pygame.display.set_caption("demo bpf24_c2")
#
#         FRAME = 0
#         CLOCK = pygame.time.Clock()
#         GAME = True
#         avg = []
#         while GAME:
#
#             pygame.event.pump()
#             for _ in pygame.event.get():
#                 if event.type == pygame.MOUSEMOTION:
#                     MOUSE_POS = event.pos
#                 keys = pygame.key.get_pressed()
#
#                 if keys[pygame.K_ESCAPE]:
#                     GAME = False
#                     break
#
#             source, arr = bpf24_c2(source, 64, transpose=True)
#
#             SCREEN.fill((10, 10, 50))
#             SCREEN.blit(source, (0, 0))
#
#             CLOCK.tick(2000)
#             t = CLOCK.get_fps()
#             avg.append(t)
#             avg=show_fps(SCREEN, t, avg)
#             pygame.display.flip()
#
#             FRAME += 1
#
#             pygame.display.set_caption(
#                 "Test bpf24_c2 %s fps"
#                 "(%sx%s)" % (round(CLOCK.get_fps(), 2), WIDTH, HEIGHT))
#
#             source = source_cp.copy()


#
# class Testfiltering24_c(unittest.TestCase):
#     """
#     Test filtering24_c
#     """
#
#     # pylint: disable=too-many-statements
#     @staticmethod
#     def runTest() -> None:
#         """
#
#         :return:  void
#         """
#
#         SCREENRECT = pygame.Rect(0, 0, 800, 600)
#         pygame.display.init()
#         SCREEN = pygame.display.set_mode(SCREENRECT.size, pygame.DOUBLEBUF, 32)
#         pygame.init()
#
#         source = pygame.image.load('../Assets/px.png').convert()
#         source = pygame.transform.smoothscale(source, (800, 600))
#         source_cp = source.copy()
#
#         mask = pygame.image.load('../Assets/alpha.png').convert_alpha()
#         mask = pygame.transform.smoothscale(mask, (800, 600))
#         # mask = numpy.asarray(pygame.surfarray.pixels_alpha(mask) / 255.0, dtype = numpy.float32)
#
#         pygame.display.set_caption("demo filtering24_c")
#
#         FRAME = 0
#         CLOCK = pygame.time.Clock()
#         GAME = True
#         avg = []
#         while GAME:
#
#             pygame.event.pump()
#             for _ in pygame.event.get():
#                 if event.type == pygame.MOUSEMOTION:
#                     MOUSE_POS = event.pos
#                 keys = pygame.key.get_pressed()
#
#                 if keys[pygame.K_ESCAPE]:
#                     GAME = False
#                     break
#
#             filtering24_c(source, mask)
#
#             SCREEN.fill((10, 10, 50))
#             SCREEN.blit(source, (0, 0))
#
#             CLOCK.tick(2000)
#             t = CLOCK.get_fps()
#             avg.append(t)
#             avg=show_fps(SCREEN, t, avg)
#             pygame.display.flip()
#
#             FRAME += 1
#
#             pygame.display.set_caption(
#                 "Test filtering24_c %s fps"
#                 "(%sx%s)" % (round(CLOCK.get_fps(), 2), WIDTH, HEIGHT))
#
#             source = source_cp.copy()


#
# class Testblur3d_cp_c(unittest.TestCase):
#     """
#     Test blur3d_cp_c
#     """
#
#     # pylint: disable=too-many-statements
#     @staticmethod
#     def runTest() -> None:
#         """
#
#         :return:  void
#         """
#
#         SCREENRECT = pygame.Rect(0, 0, 800, 600)
#         pygame.display.init()
#         SCREEN = pygame.display.set_mode(SCREENRECT.size, pygame.DOUBLEBUF, 32)
#         pygame.init()
#
#         source = pygame.image.load('../Assets/px.png').convert(24)
#         source = pygame.transform.smoothscale(source, (800, 600))
#
#         source_cp = source.copy()
#
#         pygame.display.set_caption("demo blur3d_cp_c")
#
#         FRAME = 0
#         CLOCK = pygame.time.Clock()
#         GAME = True
#         avg = []
#         while GAME:
#
#             pygame.event.pump()
#             for _ in pygame.event.get():
#                 if event.type == pygame.MOUSEMOTION:
#                     MOUSE_POS = event.pos
#                 keys = pygame.key.get_pressed()
#
#                 if keys[pygame.K_ESCAPE]:
#                     GAME = False
#                     break
#
#             src_array = blur3d_cp_c(pixels3d(source))
#             source = pygame.image.frombuffer(src_array, (800, 600), 'RGB')
#
#             SCREEN.fill((10, 10, 50))
#             SCREEN.blit(source, (0, 0))
#
#             CLOCK.tick(2000)
#             t = CLOCK.get_fps()
#             avg.append(t)
#             avg=show_fps(SCREEN, t, avg)
#             pygame.display.flip()
#
#             FRAME += 1
#
#             pygame.display.set_caption(
#                 "Test blur3d_cp_c %s fps"
#                 "(%sx%s)" % (round(CLOCK.get_fps(), 2), WIDTH, HEIGHT))
#
#             source = source_cp.copy()


class Testblur1d_cp(unittest.TestCase):
    """
    Test blur1d_cp
    """

    # pylint: disable=too-many-statements
    @staticmethod
    def runTest() -> None:
        """

        :return:  void
        """

        SCREENRECT = pygame.Rect(0, 0, 800, 600)
        pygame.display.init()
        SCREEN = pygame.display.set_mode(SCREENRECT.size, pygame.DOUBLEBUF, 32)
        pygame.init()

        source = pygame.image.load('../Assets/px.png').convert_alpha()
        source = pygame.transform.smoothscale(source, (800, 600))

        source_cp = source.copy()

        pygame.display.set_caption("demo blur1d_cp")

        FRAME = 0
        CLOCK = pygame.time.Clock()
        GAME = True
        avg = [ ]
        while GAME:

            pygame.event.pump()
            for _ in pygame.event.get():

                keys = pygame.key.get_pressed()

                if keys[ pygame.K_ESCAPE ]:
                    GAME = False
                    break

            arr = blur1d_cp(
                source.get_buffer(),
                800, 600,
                npass = 1,
                format_32 = True
            )
            source = pygame.image.frombuffer(arr, (800, 600), 'BGRA')

            SCREEN.fill((10, 10, 50))
            SCREEN.blit(source, (0, 0))

            CLOCK.tick(2000)
            t = CLOCK.get_fps()
            avg.append(t)
            avg = show_fps(SCREEN, t, avg)
            pygame.display.flip()

            FRAME += 1

            pygame.display.set_caption(
                "Test blur1d_cp %s fps"
                "(%sx%s)" % (round(CLOCK.get_fps(), 2), WIDTH, HEIGHT))

            source = source_cp.copy()


class Testchromatic(unittest.TestCase):
    """
    Test chromatic
    """

    # pylint: disable=too-many-statements
    @staticmethod
    def runTest() -> None:
        """

        :return:  void
        """

        SCREENRECT = pygame.Rect(0, 0, 800, 600)
        pygame.display.init()
        SCREEN = pygame.display.set_mode(SCREENRECT.size, pygame.DOUBLEBUF, 32)
        pygame.init()

        source = pygame.image.load('../Assets/px.png').convert(24)
        source = pygame.transform.smoothscale(source, (800, 600))

        source_cp = source.copy()

        pygame.display.set_caption("demo chromatic")

        FRAME = 0
        CLOCK = pygame.time.Clock()
        GAME = True
        avg = [ ]
        while GAME:

            pygame.event.pump()
            for _ in pygame.event.get():

                keys = pygame.key.get_pressed()

                if keys[ pygame.K_ESCAPE ]:
                    GAME = False
                    break

            source = chromatic(source, 100, 300, 0.9999999, fx = 0.008)

            SCREEN.fill((10, 10, 50))
            SCREEN.blit(source, (0, 0))

            CLOCK.tick(2000)
            t = CLOCK.get_fps()
            avg.append(t)
            avg = show_fps(SCREEN, t, avg)
            pygame.display.flip()

            FRAME += 1

            pygame.display.set_caption(
                "Test chromatic %s fps"
                "(%sx%s)" % (round(CLOCK.get_fps(), 2), WIDTH, HEIGHT))

            source = source_cp.copy()


class Testdithering1D_atkinson_c(unittest.TestCase):
    """
    Test dithering1D_atkinson_c
    """

    # pylint: disable=too-many-statements
    @staticmethod
    def runTest() -> None:
        """

        :return:  void
        """

        SCREENRECT = pygame.Rect(0, 0, 800, 600)
        pygame.display.init()
        SCREEN = pygame.display.set_mode(SCREENRECT.size, pygame.DOUBLEBUF, 32)
        pygame.init()

        source = pygame.image.load('../Assets/px.png').convert(24)
        source = pygame.transform.smoothscale(source, (800, 600))

        pygame.display.set_caption("demo dithering1D_atkinson_c")

        FRAME = 0
        CLOCK = pygame.time.Clock()
        GAME = True
        avg = [ ]
        while GAME:

            pygame.event.pump()
            for _ in pygame.event.get():

                keys = pygame.key.get_pressed()

                if keys[ pygame.K_ESCAPE ]:
                    GAME = False
                    break

            dithering_atkinson1d(800, 600, source.get_buffer(), False)

            SCREEN.fill((10, 10, 50))
            SCREEN.blit(source, (0, 0))

            CLOCK.tick(2000)
            t = CLOCK.get_fps()
            avg.append(t)
            avg = show_fps(SCREEN, t, avg)
            pygame.display.flip()

            FRAME += 1

            pygame.display.set_caption(
                "Test dithering1D_atkinson_c %s fps"
                "(%sx%s)" % (round(CLOCK.get_fps(), 2), WIDTH, HEIGHT))

            # source = source_cp.copy()


class Testyiq_2_rgb(unittest.TestCase):
    """
    Test yiq_2_rgb
    """

    # pylint: disable=too-many-statements
    @staticmethod
    def runTest() -> None:
        """

        :return:  void
        """

        SCREENRECT = pygame.Rect(0, 0, 800, 600)
        pygame.display.init()
        SCREEN = pygame.display.set_mode(SCREENRECT.size, pygame.DOUBLEBUF, 32)
        pygame.init()

        pygame.display.set_caption("demo yiq_2_rgb")

        CLOCK = pygame.time.Clock()
        GAME = True
        avg = [ ]
        while GAME:

            pygame.event.pump()
            for _ in pygame.event.get():

                keys = pygame.key.get_pressed()

                if keys[ pygame.K_ESCAPE ]:
                    GAME = False
                    break

            y, i, q = rgb_2_yiq(255, 128, 64)
            r, g, b = yiq_2_rgb(y, i, q)
            print(r, g, b)

            # h, s, l = _rgb_to_hsl(255, 128, 64)
            # r, g, b = _hsl_to_rgb(h/360.0, s/100.0, l/100.0)

            # h, s, v = _rgb_to_hsv(255, 128, 64)
            # r, g, b = _hsv_to_rgb(h/360.0, s/100.0, v/100.0)


            SCREEN.fill((10, 10, 50))

            CLOCK.tick(2000)
            t = CLOCK.get_fps()
            avg.append(t)
            avg = show_fps(SCREEN, t, avg)
            pygame.display.flip()

            pygame.display.set_caption(
                "Test yiq_2_rgb %s fps"
                "(%sx%s)" % (round(CLOCK.get_fps(), 2), WIDTH, HEIGHT))

class TestRGB_TO_YIQ_I0_inplace(unittest.TestCase):
    """
    Test RGB_TO_YIQ_I0_inplace
    """

    # pylint: disable=too-many-statements
    @staticmethod
    def runTest() -> None:
        """

        :return:  void
        """

        SCREENRECT = pygame.Rect(0, 0, 800, 600)
        pygame.display.init()
        SCREEN = pygame.display.set_mode(SCREENRECT.size, pygame.DOUBLEBUF, 32)
        pygame.init()

        source = pygame.image.load('../Assets/px.png').convert(24)
        source = pygame.transform.smoothscale(source, (800, 600))
        source_cp = source.copy()

        pygame.display.set_caption("demo RGB_TO_YIQ_I0_inplace")

        FRAME = 0
        CLOCK = pygame.time.Clock()
        GAME = True
        avg = [ ]
        while GAME:

            pygame.event.pump()
            for _ in pygame.event.get():

                keys = pygame.key.get_pressed()

                if keys[ pygame.K_ESCAPE ]:
                    GAME = False
                    break

            # RGB_TO_YIQ_I0_inplace(source)
            # RGB_TO_YIQ_Q0_inplace(source)
            # RGB_TO_YIQ_Y0_inplace(source)
            RGB_TO_YIQ_inplace(source, 1, 0, 1)

            SCREEN.fill((10, 10, 50))
            SCREEN.blit(source, (0, 0))

            CLOCK.tick(2000)
            t = CLOCK.get_fps()
            avg.append(t)
            avg = show_fps(SCREEN, t, avg)
            pygame.display.flip()

            FRAME += 1

            pygame.display.set_caption(
                "Test RGB_TO_YIQ_I0_inplace %s fps"
                "(%sx%s)" % (round(CLOCK.get_fps(), 2), WIDTH, HEIGHT))

            source = source_cp.copy()



class Testbufferize(unittest.TestCase):
    """
    Test bufferize
    """

    # pylint: disable=too-many-statements
    @staticmethod
    def runTest() -> None:
        """

        :return:  void
        """

        SCREENRECT = pygame.Rect(0, 0, 800, 800)
        pygame.display.init()
        SCREEN = pygame.display.set_mode(SCREENRECT.size, pygame.DOUBLEBUF, 32)
        pygame.init()

        w = 400
        h = 600
        source = pygame.image.load('../Assets/px.png').convert(24) #.convert_alpha()
        source = pygame.transform.smoothscale(source, (w, h))
        source_cp = source.copy()

        pygame.display.set_caption("demo bufferize")

        CLOCK = pygame.time.Clock()
        GAME = True
        avg = [ ]
        numpy.set_printoptions(threshold = sys.maxsize)
        while GAME:

            pygame.event.pump()
            for _ in pygame.event.get():

                keys = pygame.key.get_pressed()

                if keys[ pygame.K_ESCAPE ]:
                    GAME = False
                    break
            # rgb_array = RemoveAlpha(numpy.frombuffer(source.get_buffer(), dtype=numpy.uint8).reshape(800, 600, 4))

            # buffer = bufferize(rgb_array)
            # rgb_array = unbufferize(buffer, 800, 600, 3)
            # surface = make_surface(numpy.array(rgb_array))

            # arr = resize_array(pixels3d(source), 400, 300)
            # surface = make_surface(numpy.array(arr))

            # rgba_array = make_rgba_array(pixels3d(source), pygame.surfarray.pixels_alpha(source), True)
            # surface = pygame.image.frombuffer(rgba_array, (800, 600), 'RGBA').convert_alpha()

            # surface = create_rgba_surface(pixels3d(source), pygame.surfarray.pixels_alpha(source)).convert_alpha()

            # surface = channels_to_surface(pygame.surfarray.array_red(source),
            #                               pygame.surfarray.array_green(source),
            #                               pygame.surfarray.array_blue(source),
            #                               pygame.surfarray.array_alpha(source),
            #                               tmp_array_ = None
            #                               ).convert_alpha()

            # array2d = pygame.surfarray.map_array(source, pixels3d(source))
            # preallocated_array = numpy.empty((800, 600, 3), dtype = numpy.uint8)
            # rgb_array = unmapping_array(array2d, preallocated_array)
            # surface = pygame.Surface((800, 600))
            # pygame.pixelcopy.array_to_surface(surface, rgb_array)

            # array2d = mapping_array(pixels3d(source))
            # array3d = unmapping_array(array2d)
            # surface = pygame.Surface((800, 600))
            # pygame.pixelcopy.array_to_surface(surface, array3d)

            # array2d = binary_mask(pygame.surfarray.pixels_alpha(source))

            # ----------------------------------------------------------------
            # arr = numpy.empty(w*h*3, dtype=numpy.uint8)
            # arr = bgr_buffer_transpose(w, h, source.get_buffer(), arr)
            #
            # buff2 = source.get_buffer()
            # buff2 = numpy.frombuffer(buff2, dtype=numpy.uint8).reshape(w, h, 3)
            # buff2 = buff2.transpose(1, 0, 2)
            # buff2 = buff2.flatten()
            #
            # surface = pygame.image.frombuffer(arr, (h, w), 'BGR')
            # surface1 = pygame.image.frombuffer(buff2, (h, w), 'BGR')
            #
            # # print(buff2.shape, arr.shape, numpy.array_equal(buff2, arr))
            #
            # SCREEN.fill((10, 10, 50))
            # SCREEN.blit(surface, (0, 0))
            # SCREEN.blit(surface1, (400, 0))
            # -----------------------------------------------------------------

            # arr = numpy.ascontiguousarray(pixels3d(source))
            # arr = pixels3d(source).transpose(1, 0, 2)

            #print(pixels3d(source).shape)
            # arr = pixels3d(source).flatten()

            #arr = bgr_buffer_transpose(h, w, arr)
            #arr = numpy.frombuffer(arr, dtype=numpy.uint8) # .reshape(w, h, 3)
            #surface = pygame.image.frombuffer(arr, (w, h), 'RGB')

            precomputed_gradient = create_line_gradient_rgb(
                math.sqrt(800 ** 2 + 800 ** 2),
                start_rgb = (255, 0, 0),
                end_rgb = (0, 0, 0))
            # print(precomputed_gradient.shape)
            surface = create_radial_gradient(800, 800,
                offset_x              = 0.5,
                offset_y              = 0.5,
                color_start           = (255, 0, 0),
                color_end             = (0, 0, 0),
                precomputed_gradient  = precomputed_gradient  # Renamed from gradient_array_
                                             )

            precomputed_gradient = create_line_gradient_rgba(
                math.sqrt(800 ** 2 + 800 ** 2),
                start_rgba = (255, 0, 0, 255),
                end_rgba = (0, 0, 0, 0))

            surface = create_radial_gradient_alpha(800, 800 ,
                                                   precomputed_gradient = precomputed_gradient).convert_alpha()

            SCREEN.fill((10, 10, 50))
            SCREEN.blit(surface, (0, 0))


            CLOCK.tick(2000)
            t = CLOCK.get_fps()
            avg.append(t)
            avg = show_fps(SCREEN, t, avg)
            pygame.display.flip()

            pygame.display.set_caption(
                "Test bufferize %s fps"
                "(%sx%s)" % (round(CLOCK.get_fps(), 2), WIDTH, HEIGHT))


class Testblur1d(unittest.TestCase):
    """
    Test blur1d
    """

    # pylint: disable=too-many-statements
    @staticmethod
    def runTest() -> None:
        """

        :return:  void
        """

        SCREENRECT = pygame.Rect(0, 0, 800, 600)
        pygame.display.init()
        SCREEN = pygame.display.set_mode(SCREENRECT.size, pygame.DOUBLEBUF, 32)
        pygame.init()

        source = pygame.image.load('../Assets/px.png').convert_alpha()
        source = pygame.transform.smoothscale(source, (800, 600))

        source_cp = source.copy()

        pygame.display.set_caption("demo blur1d")

        FRAME = 0
        CLOCK = pygame.time.Clock()
        GAME = True
        avg = [ ]
        while GAME:

            pygame.event.pump()
            for _ in pygame.event.get():

                keys = pygame.key.get_pressed()

                if keys[ pygame.K_ESCAPE ]:
                    GAME = False
                    break

            # blur1d(
            #     source.get_buffer(),
            #     800, 600,
            #     npass = 1,
            #     format_32 = True
            # )

            # blur(source)

            source = pygame.image.frombuffer(blur3d_cp(pixels3d(source)), (800, 600), 'RGB')

            SCREEN.fill((10, 10, 50))
            SCREEN.blit(source, (0, 0))

            CLOCK.tick(2000)
            t = CLOCK.get_fps()
            avg.append(t)
            avg = show_fps(SCREEN, t, avg)
            pygame.display.flip()

            FRAME += 1

            pygame.display.set_caption(
                "Test blur1d %s fps"
                "(%sx%s)" % (round(CLOCK.get_fps(), 2), WIDTH, HEIGHT))

            source = source_cp.copy()




class Testbrightness_gpu(unittest.TestCase):
    """
    Test brightness_gpu
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
        SCREEN = pygame.display.set_mode(SCREENRECT.size, pygame.DOUBLEBUF, 32)
        pygame.init()
        grid, block = block_grid(800, 600)


        background = pygame.image.load('../Assets/Aliens.jpg').convert()
        background.set_alpha(None)
        background_cp = background.copy()

        pygame.display.set_caption("demo brightness_gpu")

        MOUSE_POS = [ 0, 0 ]
        FRAME = 0
        CLOCK = pygame.time.Clock()
        GAME = True
        avg = [ ]
        while GAME:

            pygame.event.pump()

            for event in pygame.event.get():
                if event.type == pygame.MOUSEMOTION:
                    MOUSE_POS = event.pos
                keys = pygame.key.get_pressed()

                if keys[ pygame.K_ESCAPE ]:
                    GAME = False
                    break

            background = brightness_gpu(background, 0.1, grid, block)
            SCREEN.fill((0, 0, 0))
            SCREEN.blit(background, (0, 0))
            t = CLOCK.get_fps()
            avg.append(t)
            avg = show_fps(SCREEN, t, avg)
            pygame.display.flip()
            CLOCK.tick()
            FRAME += 1
            background = background_cp.copy()
            pygame.display.set_caption(
                "Demo brightness_gpu %s fps"
                "(%sx%s)" % (round(CLOCK.get_fps(), 2), WIDTH, HEIGHT))


class Testemboss5x5_gpu(unittest.TestCase):
    """
    Test emboss5x5_gpu
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
        SCREEN = pygame.display.set_mode(SCREENRECT.size, pygame.DOUBLEBUF, 32)
        pygame.init()
        grid, block = block_grid(800, 600)


        background = pygame.image.load('../Assets/Aliens.jpg').convert()
        background.set_alpha(None)
        background_cp = background.copy()

        pygame.display.set_caption("demo emboss5x5_gpu")

        MOUSE_POS = [ 0, 0 ]
        FRAME = 0
        CLOCK = pygame.time.Clock()
        GAME = True
        avg = [ ]
        while GAME:

            pygame.event.pump()

            for event in pygame.event.get():
                if event.type == pygame.MOUSEMOTION:
                    MOUSE_POS = event.pos
                keys = pygame.key.get_pressed()

                if keys[ pygame.K_ESCAPE ]:
                    GAME = False
                    break

            background = emboss5x5_gpu(background)
            SCREEN.fill((0, 0, 0))
            SCREEN.blit(background, (0, 0))
            t = CLOCK.get_fps()
            avg.append(t)
            avg = show_fps(SCREEN, t, avg)
            pygame.display.flip()
            CLOCK.tick()
            FRAME += 1
            background = background_cp.copy()
            pygame.display.set_caption(
                "Demo emboss5x5_gpu %s fps"
                "(%sx%s)" % (round(CLOCK.get_fps(), 2), WIDTH, HEIGHT))



class Testbilateral_gpu(unittest.TestCase):
    """
    Test bilateral_gpu
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
        SCREEN = pygame.display.set_mode(SCREENRECT.size, pygame.DOUBLEBUF, 32)
        pygame.init()
        grid, block = block_grid(800, 600)

        background = pygame.image.load('../Assets/Aliens.jpg').convert()
        background.set_alpha(None)
        background_cp = background.copy()

        pygame.display.set_caption("demo bilateral_gpu")

        MOUSE_POS = [ 0, 0 ]
        FRAME = 0
        CLOCK = pygame.time.Clock()
        GAME = True
        avg = [ ]
        while GAME:

            pygame.event.pump()

            for event in pygame.event.get():
                if event.type == pygame.MOUSEMOTION:
                    MOUSE_POS = event.pos
                keys = pygame.key.get_pressed()

                if keys[ pygame.K_ESCAPE ]:
                    GAME = False
                    break

            background = bilateral_gpu(background, 5)

            SCREEN.fill((0, 0, 0))
            SCREEN.blit(background, (0, 0))
            t = CLOCK.get_fps()
            avg.append(t)
            avg = show_fps(SCREEN, t, avg)
            pygame.display.flip()
            CLOCK.tick()
            FRAME += 1
            background = background_cp.copy()
            pygame.display.set_caption(
                "Demo bilateral_gpu %s fps"
                "(%sx%s)" % (round(CLOCK.get_fps(), 2), WIDTH, HEIGHT))


class Testsaturation_gpu(unittest.TestCase):
    """
    Test saturation_gpu
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
        SCREEN = pygame.display.set_mode(SCREENRECT.size, pygame.DOUBLEBUF, 32)
        pygame.init()
        grid, block = block_grid(800, 600)

        background = pygame.image.load('../Assets/Aliens.jpg').convert()
        background = pygame.transform.smoothscale(background, (800, 600))
        background.set_alpha(None)
        background_cp = background.copy()

        pygame.display.set_caption("demo saturation_gpu")

        MOUSE_POS = [ 0, 0 ]
        FRAME = 0
        CLOCK = pygame.time.Clock()
        GAME = True
        avg = [ ]
        while GAME:

            pygame.event.pump()

            for event in pygame.event.get():
                if event.type == pygame.MOUSEMOTION:
                    MOUSE_POS = event.pos
                keys = pygame.key.get_pressed()

                if keys[ pygame.K_ESCAPE ]:
                    GAME = False
                    break

            background = saturation_gpu(background, grid, block, 0.5)

            SCREEN.fill((0, 0, 0))
            SCREEN.blit(background, (0, 0))
            t = CLOCK.get_fps()
            avg.append(t)
            avg = show_fps(SCREEN, t, avg)
            pygame.display.flip()
            CLOCK.tick()
            FRAME += 1
            background = background_cp.copy()
            pygame.display.set_caption(
                "Demo saturation_gpu %s fps"
                "(%sx%s)" % (round(CLOCK.get_fps(), 2), WIDTH, HEIGHT))


class Testmirroring_gpu(unittest.TestCase):
    """
    Test mirroring_gpu
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
        SCREEN = pygame.display.set_mode(SCREENRECT.size, pygame.DOUBLEBUF, 32)
        pygame.init()
        grid, block = block_grid(800, 600)

        background = pygame.image.load('../Assets/Aliens.jpg').convert(24)
        background = pygame.transform.smoothscale(background, (800, 600))
        background_cp = background.copy()

        pygame.display.set_caption("demo mirroring_gpu")

        MOUSE_POS = [ 0, 0 ]
        FRAME = 0
        CLOCK = pygame.time.Clock()
        GAME = True
        avg = [ ]
        while GAME:

            pygame.event.pump()

            for event in pygame.event.get():
                if event.type == pygame.MOUSEMOTION:
                    MOUSE_POS = event.pos
                keys = pygame.key.get_pressed()

                if keys[ pygame.K_ESCAPE ]:
                    GAME = False
                    break

            background = mirroring_gpu(background, grid, block, False)

            SCREEN.fill((0, 0, 0))
            SCREEN.blit(background, (0, 0))
            t = CLOCK.get_fps()
            avg.append(t)
            avg = show_fps(SCREEN, t, avg)
            pygame.display.flip()
            CLOCK.tick(2000)
            FRAME += 1
            background = background_cp.copy()
            pygame.display.set_caption(
                "Demo mirroring_gpu %s fps"
                "(%sx%s)" % (round(CLOCK.get_fps(), 2), WIDTH, HEIGHT))


class Testbubble_sort(unittest.TestCase):
    """
    Test mirroring_gpu
    """

    # pylint: disable=too-many-statements
    @staticmethod
    def runTest() -> None:


        """

        :return:  void
        """
        from PygameShader.misc import bubble_sort
        import timeit





def run_testsuite():
    """
    test suite

    :return: void
    """
    # TestShaderMedianFilter24Inplace(),

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
        Testhsv_effect(),
        Testhsv3d(),
        Testhsv1d(),
        Testhsv1d_cp(),

        Testhsl3d(),
        Testhsl1d(),

        TestShaderMedianGrayscaleFilter24Inplace(),

        Testinvert3d(),
        Testinvert1d(),
        Testinvert1d_cp(),
        Testbrightness_bpf(),
        Testbpf(),

        Testdithering(),
        Testdithering_inplace(),
        Testdithering_atkinson(),
        Testconvert_27(),
        Testbilateral(),
        Testemboss(),
        Testemboss_gray(),
        Testemboss_inplace(),
        Testemboss1d(),
        Testbilinear(),
        Testsepia_1d(),
        Testsepia_3d(),
        Testmedian(),
        Testpainting(),
        Testpixels(),
        Testsobel_1d(),
        Testsharpen_1d(),
        Testsharpen_1d_cp(),
        Testmirroring_array(),
        Testwave32(),

        Testwavelength2rgb(),
        Testsaturation1d(),
        Testsaturation1d_cp(),
        Testsharpen32(),
        Testdithering_atkinson1d(),
        Testpixelation(),
        Testblend1d(),

        Testarea24_c(),
        Testshader_bloom_fast1(),
        Testswirl32(),
        Testbrightness1d_copy(),
        Testwave_static(),
        # Testsaturation_mask()
        # Testsaturation_mask_inplace()
        Testalpha_blending(),
        Testalpha_blending_inplace(),

        # cdef
        # Testfiltering_inplace_c(),
        # Testfiltering1d_inplace_c(),

        Testzoom_inplace(),
        Testzoom(),
        Testbloom_effect_array24_c2(),
        # Testbpf24_c2(),
        # Testfiltering24_c()
        # Testblur3d_cp_c(),
        Testblur1d_cp(),
        Testchromatic(),
        Testdithering1D_atkinson_c(),
        TestRGB_TO_YIQ_I0_inplace(),
        Testbufferize(),
        Testblur1d()

    ])
    global _CUPY
    if _CUPY:
        suite.addTests([ TestLight_GPU(), TestShaderRippleGpu(), Testbrightness_gpu(),
                         Testemboss5x5_gpu(),Testbilateral_gpu(), Testsaturation_gpu(),
                         Testmirroring_gpu()])

    unittest.TextTestRunner().run(suite)
    sys.exit(0)


if __name__ == '__main__':
    run_testsuite()
    sys.exit(0)
