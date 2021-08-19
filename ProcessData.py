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
    img = cv2.imread(os.path.join(os.getcwd(), directory))
    #grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #(thresh, BWImg) = cv2.threshold(grayImg, 127, 255, cv2.THRESH_BINARY)

    randomAngle = (random.random() *60) - 30
    RotatedImg = rotate(img, angle = randomAngle, mode = 'wrap')
    randomXShift = (random.random() * 60) - 30
    randomYShift = (random.random() * 60) - 30
    #print(randomXShift, randomYShift)
    transform = AffineTransform(translation = (randomXShift, randomYShift))
    transformedImg = warp(RotatedImg,transform,mode='wrap')
    

    #sigma = 0.1
    #noisyImg = random_noise(transformedImg, var = sigma ** 2)
    #blurredImg  = filters.gaussian(image,sigma=1,multichannel=True)
    FinalImg = cv2.normalize(transformedImg, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    #print(FinalImg.shape)
    cv2.imwrite("archive/AugmentedTrain/a" + image, FinalImg)
    number += 1
    print(image + "  " +  str(number))

    
    
