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
for image in os.listdir(os.path.join(os.getcwd(), "archive/train")):
    
    
    directory = os.path.join("archive/train", image)
    img = io.imread(os.path.join(os.getcwd(), directory))
    randomAngle = (random.random() *360) + 1
    RotatedImg = rotate(img, angle = randomAngle, mode = 'wrap')
    randomXShift = (random.random() * 100) - 50
    randomYShift = (random.random() * 100) - 50
    #print(randomXShift, randomYShift)
    transform = AffineTransform(translation = (randomXShift, randomYShift))
    transformedImg = warp(RotatedImg,transform,mode='wrap')
    sigma = 0.2
    noisyImg = random_noise(transformedImg, var = sigma ** 2)
    #blurredImg  = filters.gaussian(image,sigma=1,multichannel=True)
    io.imsave("archive/AugmentedTrain/a" + img, RotatedImg)
    number += 1
    print(image + "  " +  str(number))
    
    
