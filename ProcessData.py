import numpy as np
import skimage.io as io
from skimage.transform import rotate, AffineTransform, warp
from skimage.util import random_noise
#from skimage import filters
import os
import cv2


for image in os.listdir(os.getcwd() + "\\archive\\train"):
    number = 0
    img = io.imread(image)
    randomAngle = np.random(0, 360)
    RotatedImg = rotate(img, angle = randomAngle, mode = 'wrap')
    randomXShift = np.random(-61, 61)
    randomYShift = np.random(-61, 61)
    transform = AffineTransform(translation = (randomXShift, randomYShift))
    transformedImg = warp(image,transform,mode='wrap')
    sigma = 0.2
    noisyImg = random_noise(transformedImg, var = sigma ** 2)
    #blurredImg  = filters.gaussian(image,sigma=1,multichannel=True)
    cv2.imwrite("archive/AugmentedTrain/augmentedTrain" + number + ".png", noisyImg)
    number += 1
    print(image)