from numba import jit
import numpy as np
import matplotlib.pyplot as plt
import cv2
from operator import itemgetter
from tqdm import tqdm
import imageio

PHI = (1 + 5 ** 0.5) / 2

width  = 512 
height = 512
iterations = 100

min_x, max_x = -1.9, 3.1
min_y, max_y = -2.6, 2.4

S1 = np.power(0.5, 0.5)
S2 = np.power(1./PHI, 1./PHI)
N = 16
it = 250

#@jit(nopython=True)
def t1(x,n,N,points_list,t):
    point = np.zeros(2)
    #point[0] = (S1-(S1-S2)*t)*np.cos(np.pi/4.-(np.pi/4.-32.893818*np.pi/180.+2*np.pi)*t)*x[0]-(S1-(S1-S2)*t)*np.sin(np.pi/4.-(np.pi/4.-32.893818*np.pi/180.)*t)*x[1]
    #point[1] = (S1-(S1-S2)*t)*np.sin(np.pi/4.-(np.pi/4.-32.893818*np.pi/180.+2*np.pi)*t)*x[0]+(S1-(S1-S2)*t)*np.cos(np.pi/4.-(np.pi/4.-32.893818*np.pi/180.)*t)*x[1]
    point[0] = (S1 - (S1 - S2) * t) * np.cos(np.pi/4. - (np.pi/4. - 32.893818 * np.pi/180. + 2 * np.pi) * t) * x[0] - (S1 - (S1 - S2) * t) * np.sin(np.pi/4. - (np.pi/4. - 32.893818 * np.pi/180. + 2 * np.pi) * t) * x[1]
    point[1] = (S1 - (S1 - S2) * t) * np.sin(np.pi/4. - (np.pi/4. - 32.893818 * np.pi/180. + 2 * np.pi) * t) * x[0] + (S1 - (S1 - S2) * t) * np.cos(np.pi/4. - (np.pi/4. - 32.893818 * np.pi/180. + 2 * np.pi) * t) * x[1]
    
    col = 1

    points_list.append([point,col,n])
    
    generate_points(point,n,N,points_list,t)

#@jit(nopython=True)
def t2(x,n,N,points_list,t):
    point = np.zeros(2)
    #point[0]= (S1-(S1-S2*S2)*t)*np.cos(3*np.pi/4.-(3*np.pi/4.-133.0140178*np.pi/180.+2*np.pi)*t)*x[0]-(S1-(S1-S2*S2)*t)*np.sin(-np.pi/4.-(-np.pi/4.-133.0140178*np.pi/180.+2*np.pi)*t)*x[1]+1 
    #point[1]= (S1-(S1-S2*S2)*t)*np.sin(3*np.pi/4.-(3*np.pi/4.-133.0140178*np.pi/180.+2*np.pi)*t)*x[0]+(S1-(S1-S2*S2)*t)*np.cos(-np.pi/4.-(-np.pi/4.-133.0140178*np.pi/180.+2*np.pi)*t)*x[1]
    point[0] = (S1 - (S1 - S2*S2) * t) * np.cos(3 * np.pi/4. - (3 * np.pi/4. - 133.0140178 * np.pi/180. + 2 * np.pi) * t) * x[0] - (S1 - (S1 - S2*S2) * t) * np.sin(3 * np.pi/4. - (3 * np.pi/4. - 133.0140178 * np.pi/180. + 2 * np.pi) * t) * x[1] + 1
    point[1] = (S1 - (S1 - S2*S2) * t) * np.sin(3 * np.pi/4. - (3 * np.pi/4. - 133.0140178 * np.pi/180. + 2 * np.pi) * t) * x[0] + (S1 - (S1 - S2*S2) * t) * np.cos(3 * np.pi/4. - (3 * np.pi/4. - 133.0140178 * np.pi/180. + 2 * np.pi) * t) * x[1]
    
    col = 0

    points_list.append([point,col,n])
    
    generate_points(point,n,N,points_list,t)
    
#@jit(nopython=True)
def generate_points(x, n, N, points_list, t):
    if n < N:
        t1(x,n+1,N,points_list,t)
    if n < N:
        t2(x,n+1,N,points_list,t)



def create_image(all_points_list, width, height):
    image = np.zeros((height, width, 3), dtype=np.uint8)
    # image[:,:] = np.array([50,50,40])

    pixel_size_x = (max_x - min_x) / width
    pixel_size_y = (max_y - min_y) / height

    all_points_list = sorted(all_points_list, key=itemgetter(1))

    for point, col, n in all_points_list:
        x = int((point[0] - min_x) / pixel_size_x)
        y = int((point[1] - min_y) / pixel_size_y)

        color = np.array([1.0,0.0,0.0]) if col else np.array([0.0,0.0,1.0])
        #col = 0.5*col + 0.5*col*(np.array([0.25, 0.25,0]))*n/N

        image = cv2.circle(image, (x,y), radius=1, color=(255*color).tolist() , thickness=-1)
    
    return image
    

# Single image
#nullx = np.zeros(2)
#all_points_list = []

#generate_points(nullx,0,N,all_points_list,0)

#width, height = 720, 720

#image = create_image(all_points_list,width, height)
#image_plot = cv2.resize((((image[:,:,[2,1,0]])).astype(np.uint8)), (720, 720), interpolation = cv2.INTER_CUBIC)
#cv2.imwrite('dragon.png',image_plot)

# Animation
images_animation = []
width, height = 720, 720

for t in tqdm(np.linspace(0,1,it)):
    nullx = np.zeros(2)
    all_points_list = []

    generate_points(nullx,0,N,all_points_list,t)
    
    image = create_image(all_points_list,width, height)
    
    images_animation.append(image)
    
    if t == 0 or t == 1:
        for i in range(10):
            images_animation.append(image)
            
            
images_animation = images_animation + images_animation[::-1]

imageio.mimwrite('dragon_animation.gif', images_animation, duration=80)
