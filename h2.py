import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPooling2D
import os
import cv2

from tensorflow.keras.applications.vgg16 import VGG16

CURRENT_FOLDER = os.getcwd()

def createData(directory):
    dataX = []
    dataY = []
    for img in os.listdir(directory):
        imgArr = cv2.imread(os.path.join(directory, img))
        imgArr = cv2.resize(imgArr, (224, 224))
        numFing = int(img.split('_')[1][0])
        dataX.append(imgArr)
        dataY.append(numFing)

    return np.array(dataX), np.array(dataY)




trainDir = CURRENT_FOLDER + "/archive/train"
trX, trY = createData(trainDir)
'''
plt.imshow(trX[3534], cmap = 'gray')
plt.show()
print(trY[3534])

'''
print(trX.shape)
print(trY.shape)



#------------------------------------------------

testDir =CURRENT_FOLDER + "/archive/test"
teX, teY = createData(testDir)

print(teX.shape)
print(teY.shape)



#-----------------------------------------------


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

model = create_model()
print(model.summary())
print()

model.compile(optimizer='adam', loss = tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
'''checkpoint_filepath =CURRENT_FOLDER +  "/checkpoint.h5"
model_checkpoints = tf.keras.callbacks.ModelCheckpoint(
    filepath = checkpoint_filepath,
    save_weights_only = False,
    monitor = 'val_accuracy',
    mode = 'max',
    save_best_only = True) '''
model.fit(trX, trY, epochs=1, batch_size = 32)

loss, acc = model.evaluate(teX, teY)
print('\ntest_accuracy: ' + str(acc))

model.save("modelDanyuan")
