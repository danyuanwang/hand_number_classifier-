import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import sys
from PIL import Image
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPooling2D
import cv2

CURRENT_FOLDER = os.getcwd()
#tf.saved_model.LoadOptions(experimental_io_device = "/job:localhost")

print(CURRENT_FOLDER + "image.PNG")

def create_model():
    model = tf.keras.Sequential()
    model.add(feature_extractor)
    model.add(Dense(128, activation = "relu"))
    model.add(Dense(6, activation = "softmax"))
    return model


def loadModel(file):
    model = create_model()
    model = tf.keras.models.load_model("mlmodel")
    ModuleNotFoundError()


def compressFile(file):
    img = Image.open(CURRENT_FOLDER + file)
    img = img.resize((256, 256))
    return img

model = loadModel("mlmodel")

camera = cv2.VideoCapture(0)

while True:
    ret, img = camera.read()
    rgbImage = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imshow("webcam", img)
    cv2.waitKey(1)
    

img = compressFile(rgbImage)
prediction = model.predict(img)
print(prediction)