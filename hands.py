import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPooling2D
import os
import cv2




def createData(directory):
    dataX = []
    dataY = []
    for img in os.listdir(directory):
        imgArr = cv2.imread(os.path.join(directory, img), cv2.IMREAD_GRAYSCALE)
        imgArr = cv2.resize(imgArr, (72, 72))
        numFing = int(img.split('_')[1][0])
        dataX.append(imgArr)
        dataY.append(numFing)

    return np.array(dataX), np.array(dataY)




trainDir ="C:/Users/stunt/OneDrive/Documents/Hands/archive/train"
trX, trY = createData(trainDir)
'''
plt.imshow(trX[3534], cmap = 'gray')
plt.show()
print(trY[3534])

'''
print(trX.shape)
print(trY.shape)

trX = tf.keras.utils.normalize(trX, axis=1)
trX = trX[...,np.newaxis]

#------------------------------------------------

testDir ="C:/Users/stunt/OneDrive/Documents/Hands/archive/test"
teX, teY = createData(testDir)

print(teX.shape)
print(teY.shape)

teX = tf.keras.utils.normalize(teX, axis=1)
teX = teX[...,np.newaxis]

#-----------------------------------------------

model = tf.keras.Sequential()
model.add(Conv2D(64, (10, 10), input_shape = (72, 72, 1), activation = "relu"))
model.add(MaxPooling2D(pool_size=(5,5)))
model.add(Conv2D(24, (3, 3), activation = "relu"))
model.add(MaxPooling2D(pool_size=(3,3)))
model.add(Flatten())
model.add(Dense(128, activation = "relu"))
model.add(Dense(64, activation = "relu"))
model.add(Dense(32, activation = "relu"))
model.add(Dense(6))

print(model.summary())
print()

model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
model.fit(trX, trY, epochs=3, batch_size = 10)

loss, acc = model.evaluate(teX, teY)
print('\ntest_accuracy: ' + str(acc))

model.save("ml.model")
