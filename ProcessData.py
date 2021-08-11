import numpy as np
import skimage.io as io
from skimage.transform import rotate, AffineTransform, warp, resize
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
    #print(img.shape)
    #img = resize(img, (224, 224))
    #print(img.shape)
    
    randomAngle = (random.random() *360) + 1
    RotatedImg = rotate(img, angle = randomAngle, mode = 'wrap')
    randomXShift = (random.random() * 60) - 30
    randomYShift = (random.random() * 60) - 30
    #print(randomXShift, randomYShift)
    transform = AffineTransform(translation = (randomXShift, randomYShift))
    transformedImg = warp(RotatedImg,transform,mode='wrap')
    sigma = 0.1
    #noisyImg = random_noise(transformedImg, var = sigma ** 2)
    #blurredImg  = filters.gaussian(image,sigma=1,multichannel=True)
    FinalImg = cv2.normalize(transformedImg, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    io.imsave("archive/AugmentedTrain/a" + image, FinalImg)
    number += 1
    print(image + "  " +  str(number))
    #break
    
    
