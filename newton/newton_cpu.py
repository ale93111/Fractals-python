from numba import jit
import numpy as np
import math
import cv2

@jit(nopython=True)
def f(z):
    return z**3 -1

@jit(nopython=True)
def df(z):
    return 3 * z**2

@jit(nopython=True)
def newton_method(z0, max_iter, epsilon):
    z = z0
    
    for i in range(max_iter):
        dz = f(z) / df(z)
        
        if np.abs(dz) < epsilon:
            return z, i
        
        z -= dz
        
    return z, max_iter
    
@jit(nopython=True)
def create_fractal(min_x, max_x, min_y, max_y, image, max_iter, epsilon, roots, colors):
    height = image.shape[0]
    width = image.shape[1]

    pixel_size_x = (max_x - min_x) / width
    pixel_size_y = (max_y - min_y) / height
    for x in range(width):
        real = min_x + x * pixel_size_x
        for y in range(height):
            imag = min_y + y * pixel_size_y
            
            z = complex(real,imag)
            
            z_out, it = newton_method(z, max_iter, epsilon)

            dr = np.abs(z_out - roots)

            col = colors[np.argmin(dr)]

            image[y,x] = np.clip(0.3*col + 8*col*(it/max_iter),0,1)
            
            
width, height = 10001, 10001
x_min, x_max = -2, 2
y_min, y_max = -2, 2
max_iter = 100
epsilon = 1e-6

roots = np.array([1   + 0.00001j, -0.5 + 0.86603j, -0.5 - 0.86603j])
colors = np.array([[0.75,0.25,0.25], [0.25,0.75,0.25], [0.25,0.25,0.75]])


image = np.zeros((height, width, 3), dtype=np.float32)


create_fractal(x_min, x_max, y_min, y_max, image, max_iter, epsilon, roots, colors)


image_plot = cv2.resize(((255*(image[:,:,[2,1,0]])).astype(np.uint8)), (720, 720), interpolation = cv2.INTER_AREA)
cv2.imwrite('newton.png',image_plot)

