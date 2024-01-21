from numba import jit
import numpy as np
import matplotlib.pyplot as plt
import cv2

@jit(nopython=True)
def saturate(x):
    return np.maximum(0.0, np.minimum(1.0, x))

@jit(nopython=True)
def smoothstep(a, b, x):
    t = saturate((x - a) / (b - a))
    return t * t * (3.0 - (2.0 * t))

@jit(nopython=True)
def normalize_hitcount(hitcount):
    img_out = hitcount.astype(np.float32)
    dublmax = np.array([np.max(img_out[:, :, k])+1 for k in range(img_out.shape[2])])

    for k in range(hitcount.shape[2]):
        img_out[:, :, k] = smoothstep(0, np.sqrt(dublmax[k]), np.sqrt(img_out[:, :, k]))
    
    return img_out

@jit(nopython=True)
def increment(img, point, width, height):
    u = int(((point.real-(-2))/(2-(-2)))*width )
    v = int(((point.imag-(-2))/(2-(-2)))*height)

    if 0 <= u < width and 0 <= v < height:
        img[u,  v] += 1
        img[u, -v] += 1  # y-axis symmetry

@jit(nopython=True)
def increment_rgb(img, point, count, width, height):
    u = int(((point.real-(-2))/(2-(-2)))*width )
    v = int(((point.imag-(-2))/(2-(-2)))*height)

    max_iterB = 50
    max_iterG = 200
    max_iterR = 800
    if 0 <= u < width and 0 <= v < height:
        if count < max_iterB:
            img[u,  v, 2] += 1
            img[u, -v, 2] += 1  # y-axis symmetry
        elif count < max_iterG:
            img[u,  v, 1] += 1
            img[u, -v, 1] += 1  # y-axis symmetry
        elif count < max_iterR:
            img[u,  v, 0] += 1
            img[u, -v, 0] += 1  # y-axis symmetry
        
@jit(nopython=True)
def buddhabrot(img, store_z, store_c, width, height, max_iter, min_iter, num_samples, radius):

    for _ in range(num_samples):
        is_periodic = False
        
        x = np.random.uniform(-2, 2)  # random number between -2 and 2
        y = np.random.uniform(-2, 2)
        
        c = complex(x, y)
        
        # are we inside the main cardiod?
        if (abs(c) * (8 * abs(c) - 3) < 3 / 32. - c.real):
            continue
        # are we inside the 2nd order period bulb?
        if (x * x + 2 * x + y * y < -15 / 16.):
            continue
        # smaller bulb left of the period-2 bulb
        if ((x + 1.309) * (x + 1.309) + y * y < 0.00345):
            continue
        # smaller bulb top of the main cardioid
        if ((x + 0.125) * (x + 0.125) + (y + 0.744) * (y + 0.744) < 0.0088):
            continue
        # smaller bulb bottom of the main cardioid
        if ((x + 0.125) * (x + 0.125) + (y - 0.744) * (y - 0.744) < 0.0088):
            continue

        z = 0.0j
        oldz = 0.0j
        i = 0
        
        while (abs(z) < radius and i < max_iter):
            store_z[i] = z
            store_c[i] = c

            if (i != 0) and ((i & (i - 1)) == 0):
                oldz = z  # brent's method

            z = z * z + c
            i += 1
            if (i != 0 and z == oldz):
                is_periodic = True
                break

        if is_periodic:
            continue

        if i < max_iter and i > min_iter:
            for j in range(i - 1):
                # store_points[j] = store_z[j]
                point = store_z[j]
                
                # increment(img, point, width, height)
                increment_rgb(img, point, j, width, height)


    return img

if __name__ == "__main__":
    width, height = 4800, 4800
    max_iter = 800
    num_samples = 100000000

    img = np.zeros((width, height, 3), dtype=np.uint16)
    # store_points = np.zeros(max_iter, dtype=complex)
    store_z = np.zeros(max_iter, dtype=complex)
    store_c = np.zeros(max_iter, dtype=complex)
    
    buddha_img = buddhabrot(img, store_z, store_c, width, height, max_iter, 5, num_samples, 4.0)
    buddha_img = normalize_hitcount(buddha_img)
    buddha_img = (255*buddha_img).astype(np.uint8)
    buddha_img = cv2.fastNlMeansDenoisingColored(buddha_img,None,5,5,7,21)
    buddha_img = buddha_img[:-300, 150:-150]
    buddha_img = cv2.resize(buddha_img, (720, 720), interpolation = cv2.INTER_AREA)
    cv2.imwrite('buddhabrot.png',buddha_img[:,:,[2,1,0]])
#     plt.imshow(buddha_img)
#     plt.show()