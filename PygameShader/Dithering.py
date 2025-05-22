import numpy
import pygame
import numba
import timeit
import functools

@numba.jit("f4[:, :, :](f4[:,:,:])", nopython=True, nogil=True)
def atkinson(array_):
    frac = 8
    lx, ly, lc = array_.shape

    for j in range(ly):
        for i in range(lx):
            for c in range(lc):
                rounded = round(array_[i, j, c])
                err = array_[i, j, c] - rounded
                err_frac = err / frac
                array_[i, j, c] = rounded
                if i < lx - 1: array_[i + 1, j, c] += err_frac
                if i < lx - 2: array_[i + 2, j, c] += err_frac
                if j < ly - 1:
                    array_[i, j + 1, c] += err_frac
                    if i > 0: array_[i - 1, j + 1, c] += err_frac
                    if i < lx - 1: array_[i + 1, j + 1, c] += err_frac
                if j < ly - 2: array_[i, j + 2, c] += err_frac
    return array_


@numba.jit("uint8[:, :, :](uint8[:,:,:])", nopython=True, nogil=True)
def gray(array_):
    lx, ly, lc = array_.shape

    for j in range(ly):
        for i in range(lx):
            for c in range(lc):
                array_[i, j, c] = array_[i, j, c]/2.0
    return array_


if __name__ == '__main__':
    image = pygame.image.load("Assets/background2.jpg")
    image = pygame.transform.smoothscale(image, (800, 800))
    rgb_array_normalized = pygame.surfarray.array3d(image) / float(255.0)
    rgb_array_normalized = numpy.array(rgb_array_normalized).astype(numpy.float32)
    array_ = atkinson(rgb_array_normalized)
    surf = pygame.surfarray.make_surface(array_ * 255.0)

    screen = pygame.display.set_mode((800, 800))

    t = timeit.timeit(
        "atkinson(rgb_array_normalized)",
        "from __main__ import atkinson, rgb_array_normalized", number=100
    )
    print(t / 100)
    pygame.time.wait(1000)

    # t = timeit.timeit(
    #     "gray(rgb_array_normalized)",
    #     "from __main__ import gray, rgb_array_normalized", number=100
    # )
    # print(t / 100)

    clock = pygame.time.Clock()

    while 1:
        screen.fill((0, 0, 0))
        pygame.event.pump()
        array_ = atkinson((pygame.surfarray.pixels3d(image) / float(255.0)).astype(numpy.float32))
        surf = pygame.surfarray.make_surface(array_ * 255.0)

        #bgr_array = pygame.surfarray.array3d(image)
        #bgr_array = gray(bgr_array)
        #surf = pygame.surfarray.make_surface(bgr_array)

        screen.blit(surf, (0, 0))
        pygame.display.flip()
        clock.tick()
        fps = clock.get_fps()

        pygame.display.set_caption(
            "testing %s fps "
            "" % (round(fps, 2)))