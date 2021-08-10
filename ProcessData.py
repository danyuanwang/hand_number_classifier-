import numpy as np
import skimage.io as io
from skimage.transform import rotate, AffineTransform, warp
from skimage.util import random_noise
#from skimage import filters
import os
import cv2
import random
print(os.getcwd())
number = 0
for image in os.listdir(os.path.join(os.getcwd(), "archive/test")):
    
    
    directory = os.path.join("archive/test", image)
    img = io.imread(os.path.join(os.getcwd(), directory))
    randomAngle = (random.random() *360) + 1
    RotatedImg = rotate(img, angle = randomAngle, mode = 'wrap')
    randomXShift = (random.random() * 60) - 30
    randomYShift = (random.random() * 60) - 30
    #print(randomXShift, randomYShift)
    transform = AffineTransform(translation = (randomXShift, randomYShift))
    transformedImg = warp(RotatedImg,transform,mode='wrap')
    sigma = 0.2
    noisyImg = random_noise(transformedImg, var = sigma ** 2)
    #blurredImg  = filters.gaussian(image,sigma=1,multichannel=True)
    io.imsave("archive/AugmentedTest/a" + image, noisyImg)
    number += 1
    print(image + "  " +  str(number))
    
    
