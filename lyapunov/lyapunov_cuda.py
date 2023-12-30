from numba import cuda, jit
import numpy as np
import math
import matplotlib.pyplot as plt
import cv2

def apply_color1(data):
    height,width = data.shape
    image = np.zeros((height,width,3))

    for j, pos_y in enumerate(range(height)):
        for i, pos_x in enumerate(range(width)):
            lambda_i = data[j,i]

            if lambda_i < 0.0:
                lambda_i = np.abs(lambda_i)
                lambda_i = np.clip(lambda_i, 0.0, 1.0)
                lambda_i = np.power(lambda_i, 0.25)
                col = np.array([1.0, lambda_i, 0.0])
            else:
                lambda_i = np.abs(lambda_i)
                lambda_i = np.clip(lambda_i, 0.0, 1.0)
                lambda_i = np.power(lambda_i, 0.25)
                col = np.array([0.5, 0.5, lambda_i])

            image[j,i] = col
            
    return image

def apply_color2(data):
    height,width = data.shape
    image = np.zeros((height,width,3))

    for j, pos_y in enumerate(range(height)):
        for i, pos_x in enumerate(range(width)):
            lambda_i = data[j,i]

            if lambda_i < 0.0:
                lambda_i = np.abs(lambda_i)
                lambda_i = np.clip(lambda_i, 0.0, 1.0)
                lambda_i = np.power(lambda_i, 0.5)
                col = np.array([(1-lambda_i)**1.5, (1-lambda_i)**6, 0.0])
            else:
                lambda_i = np.abs(lambda_i)
                lambda_i = np.clip(lambda_i, 0.0, 1.0)
                lambda_i = np.power(lambda_i, 0.25)
                col = np.array([(1-lambda_i)**2, 0.0, 0.0])

            image[j,i] = col
            
    return image

def apply_color3(data):
    height,width = data.shape
    image = np.zeros((height,width,3))

    for j, pos_y in enumerate(range(height)):
        for i, pos_x in enumerate(range(width)):
            lambda_i = data[j,i]

            if lambda_i < 0.0:
                lambda_i = np.abs(lambda_i)
                lambda_i = np.clip(lambda_i, 0.0, 1.0)
                lambda_i = np.power(lambda_i, 0.15)
                col = np.array([1.0, lambda_i, 0.0])
            else:
                lambda_i = np.abs(lambda_i)
                lambda_i = np.clip(lambda_i, 0.0, 1.0)
#                 lambda_i = np.power(lambda_i, 0.5)
                col = np.array([(1-lambda_i)**3, 0.0, 0.4])

            image[j,i] = col
            
    return image

def apply_color4(data):
    height,width = data.shape
    image = np.zeros((height,width,3))

    for j, pos_y in enumerate(range(height)):
        for i, pos_x in enumerate(range(width)):
            lambda_i = data[j,i]

            if lambda_i < 0.0:
                lambda_i = np.abs(lambda_i)
                lambda_i = np.clip(lambda_i, 0.0, 1.0)
                lambda_i = np.power(lambda_i, 0.9)
                col = np.array([(0.9-lambda_i)**6, (1.0-lambda_i)**5, (1-lambda_i)**2])
            else:
                lambda_i = np.abs(lambda_i)
                lambda_i = np.clip(lambda_i, 0.0, 1.0)
                # lambda_i = np.power(lambda_i, 0.5)
                col = np.array([(0.9-lambda_i)**6, (1.0-lambda_i)**6, (1-lambda_i)**2])

            image[j,i] = col
            
    return image

@cuda.jit('float64(float64, float64, uint8[::1], float32, float32, int32)',device=True)
def lambda_exponent(pos_x, pos_y, S, X0, alpha, max_iterations):
    X = X0
    
    lambda_ = 0.0
    for i in range(max_iterations):
        for i, Sn in enumerate(S):
            rn = pos_x if Sn else pos_y
            
            if X > 0.5:
                X = rn * X * (1.0 - X)
            else:
                X = rn * X * (1.0 - X) + 0.25 * (alpha-1) * (rn - 2)
        
            lambda_ += math.log(abs(rn * (1.0 - 2.0 * X)))
        
    lambda_ = lambda_ / (max_iterations * len(S))
    
    return lambda_

@cuda.jit('void(float32, float32, float32, float32, float32[:,:], uint8[::1], float32, float32, int32, int32)')
def lyapunov_kernel(min_x, max_x, min_y, max_y, image, S, X0, alpha, max_iterations, AA):
    height = image.shape[0]
    width  = image.shape[1]
    
    pixel_size_x = (max_x - min_x) / width
    pixel_size_y = (max_y - min_y) / height

    startX, startY = cuda.grid(2)
    gridX = cuda.gridDim.x * cuda.blockDim.x;
    gridY = cuda.gridDim.y * cuda.blockDim.y;

    for x in range(startX, width, gridX):
        pos_x = min_x + x * pixel_size_x
        for y in range(startY, height, gridY):
            pos_y = min_y + y * pixel_size_y 

            lambda_tot = 0.0
            for xi in range(AA):
                for yi in range(AA):
                    pos_x_i = pos_x + xi * pixel_size_x / float(AA)
                    pos_y_i = pos_y + yi * pixel_size_y / float(AA)
                    
                    lambda_tot += lambda_exponent(pos_x_i, pos_y_i, S, X0, alpha, max_iterations)

            lambda_tot /= float(AA * AA)
            
            if lambda_tot > 1:
                lambda_tot = 1
            if lambda_tot <-1:
                lambda_tot = -1

            image[y, x] = lambda_tot
            

# Select configuration

AA = 4

conf = 1
width=360
height=360*2
S = np.array([1,1,1,1,1,1,0,0,0,0,0,0], dtype=np.uint8)
X0 = 0.5
alpha = 1.0
max_iterations = 300
min_x, max_x, min_y, max_y = 3.4, 4.0, 2.45, 3.65

#conf = 2
#width=720
#height=720
#S = np.array([1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,1,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0], dtype=np.uint8)
#X0 = 0.8315
#alpha = 1.0
#max_iterations = 300
#min_x, max_x, min_y, max_y = 1.155 + 0.0065, 1.197, 3.769, 3.8175

# conf = 3
# width=720
# height=720
# S = np.array([1,1,0,1,0,1,1,0,1,0], dtype=np.uint8)
# X0 = 0.499
# alpha = 0.908
# max_iterations = 600
# min_x, max_x, min_y, max_y = 3.8091, 3.825, 3.8091, 3.825

#conf = 3
#width=720
#height=720
#S = np.array([1,1,0,1,0,1,1,0,1,0], dtype=np.uint8)
#X0 = 0.499
#alpha = 0.908
#max_iterations = 600
#min_x, max_x, min_y, max_y = 3.798, 3.825, 3.798, 3.825

#conf = 4
#width=720
#height=720
#S = np.array([1,0,1,0,1,0,1,0], dtype=np.uint8)
#X0 = 0.7
#alpha = 0.9935
#max_iterations = 600
#min_x, max_x, min_y, max_y = 3.82, 3.87, 3.82, 3.87

data = np.ones((height, width), dtype=np.float32)

d_data = cuda.to_device(data)
d_S = cuda.to_device(S)

nthreads = 16
nblocksy = (height // nthreads) + 1
nblocksx = (width // nthreads) + 1

lyapunov_kernel[(nblocksx, nblocksy), (nthreads, nthreads)](min_x, max_x, min_y, max_y, d_data, d_S, X0, alpha, max_iterations, AA)

d_data.copy_to_host(data)

print('done, saving...')


if conf == 1:
    image = apply_color1(data)
elif conf == 2:
    image = apply_color2(data)
elif conf == 3:
    image = apply_color3(data)
elif conf == 4:
    image = apply_color4(data)
    
cv2.imwrite('lyapunov_'+str(conf)+'.png',((255*np.rot90(image[:,:,[2,1,0]])).astype(np.uint8)))