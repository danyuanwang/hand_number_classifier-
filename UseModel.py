import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import sys
from PIL import Image
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPooling2D

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
img = compressFile()
prediction = model.predict(img)
print(prediction)