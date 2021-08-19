import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPooling2D
import os
import cv2

from tensorflow.keras.applications.vgg19 import VGG19

CURRENT_FOLDER = os.getcwd()
def processImage(img):
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (thresh, BWImg) = cv2.threshold(grayImg, 170, 255, cv2.THRESH_BINARY)
    FinalImg = cv2.normalize(BWImg, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    #print(BWImg.shape)
    return FinalImg

def createData(directory):
    dataX = []
    dataY = []
    for img in os.listdir(directory):
        imgArr = cv2.imread(os.path.join(directory, img))
        imgArr = processImage(imgArr)
        imgArr = cv2.resize(imgArr, (224, 224))
        numFing = int(img.split('_')[1][0])
        dataX.append(imgArr)
        dataY.append(numFing)

    return np.array(dataX), np.array(dataY)




trainDir = CURRENT_FOLDER + "/archive/AugmentedTrain"
trX, trY = createData(trainDir)
'''
plt.imshow(trX[3534], cmap = 'gray')
plt.show()
print(trY[3534])

'''
print(trX.shape)
print(trY.shape)



#------------------------------------------------

testDir =CURRENT_FOLDER + "/archive/AugmentedTest"
teX, teY = createData(testDir)

print(teX.shape)
print(teY.shape)



#-----------------------------------------------
feature_extractor = VGG19(
    input_shape=(224, 224, 1),
    include_top=False,
    weights=  None,#'imagenet',
    pooling='avg'
)

feature_extractor.trainable = False


model = tf.keras.Sequential()
model.add(feature_extractor)
model.add(Dense(128, activation = "relu"))

model.add(Dense(6, activation = "softmax"))

print(model.summary())
print()

model.compile(optimizer='adam', loss = tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
<<<<<<< HEAD
model.fit(trX, trY, epochs=2, batch_size = 32)
=======
model.fit(trX, trY, epochs=5, batch_size = 32)
>>>>>>> 74db9b324453ba122fab53acab4f315fb7f574f2

loss, acc = model.evaluate(teX, teY)
print('\ntest_accuracy: ' + str(acc))

model.save("BW3Epoc.model")
