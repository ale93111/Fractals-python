from numba import jit
import numpy as np
import matplotlib.pyplot as plt
import cv2

@jit(nopython=True)
def mandel(x, y, max_iters):
    """
    Given the real and imaginary parts of a complex number,
    determine if it is a candidate for membership in the Mandelbrot
    set given a fixed number of iterations.
    """
    i = 0
    c = complex(x,y)
    z = 0.0j
    for i in range(max_iters):
        z = z * z + c
        if (z.real * z.real + z.imag * z.imag) >= 4:
            return i

    return 0

@jit(nopython=True)
def create_fractal(min_x, max_x, min_y, max_y, image, iters):
    height = image.shape[0]
    width = image.shape[1]

    pixel_size_x = (max_x - min_x) / width
    pixel_size_y = (max_y - min_y) / height
    for x in range(width):
        real = min_x + x * pixel_size_x
        for y in range(height):
            imag = min_y + y * pixel_size_y
            it = mandel(real, imag, iters)
            image[y, x] = it


image = np.zeros((10000, 10000), dtype=np.uint8)

create_fractal(-2.0, 0.6, -1.3, 1.3, image, 500)

image_plot = cv2.resize(((255*(image/image.max())).astype(np.uint8)), (720, 720), interpolation = cv2.INTER_AREA)
image_plot = np.power(image_plot, 0.20)

cm = plt.get_cmap('magma')

colored_image_plot = cm(image_plot/image_plot.max())

cv2.imwrite('mandelbrot.png',(255*colored_image_plot[:,:,[2,1,0]]).astype(np.uint8))
