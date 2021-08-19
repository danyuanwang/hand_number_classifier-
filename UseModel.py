import re
import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import sys
from PIL import Image
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPooling2D
import cv2
import time
from tensorflow.keras.applications.vgg16 import VGG16

CURRENT_FOLDER = os.getcwd()
#tf.saved_model.LoadOptions(experimental_io_device = "/job:localhost")

print(CURRENT_FOLDER + "image.PNG")

def create_model():
    feature_extractor = VGG16(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet',
    pooling='avg'
    )

    feature_extractor.trainable = False
    model = tf.keras.Sequential()
    model.add(feature_extractor)
    model.add(Dense(128, activation = "relu"))
    model.add(Dense(6, activation = "softmax"))
    return model


def loadModel(file):
    model = create_model()
    model = tf.keras.models.load_model(file)
    return model


def compressFile(file):
    img = cv2.imread(file)
    img = cv2.resize(img, (224, 224))
    img = img[np.newaxis, ...]
    return img

def processImage(img):
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (thresh, BWImg) = cv2.threshold(grayImg, 127, 255, cv2.THRESH_BINARY)
    #FinalImg = cv2.normalize(BWImg, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    print(BWImg.shape)
    return BWImg
model = loadModel("modelDanyuan")


    
    

cam = cv2.VideoCapture(0)
img = None
while (True):
    # reading from frame
    ret, img = cam.read()

    grayImg = processImage(img)
    cv2.waitKey(1)
    cv2.imshow('img', grayImg)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.imwrite('capture.png', grayImg)
        break

cv2.destroyAllWindows()
img = compressFile('capture.png')
FinalImage = cv2.imread('capture.png')
prediction = model.predict(img)
print(prediction)
print("shape of image:", FinalImage.shape)