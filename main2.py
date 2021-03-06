import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2
import keyboard

import time
THRESHOLD = 170
model = tf.keras.models.load_model("GreyScale.model")

def guess(img):
    img = cv2.resize(img, (224, 224))

    img = img[np.newaxis,...]
    img = np.expand_dims(img, 3)
    hi = model.predict(img)[0]
    #if (np.argmax(hi) == 3 and hi[2] > 5 * (10**-4)):
    #    return 2
    #print(hi[2], hi[3])
    #print(hi)
    #print(np.argmax(hi))
    return np.argmax(hi)

def convertBW(img):
    #img = cv2.resize(img, (224, 224))
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (thresh, BWImg) = cv2.threshold(grayImg, THRESHOLD, 255, cv2.THRESH_BINARY)
    #FinalImg = cv2.normalize(BWImg, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    return BWImg

cam = cv2.VideoCapture(0)



while True:
    ret, frame = cam.read()
    rgbFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    BWimg = convertBW(rgbFrame)
    #print(BWimg.shape)
    #print(rgbFrame.shape)
    pred = guess(BWimg)
    cv2.putText(frame, str(pred), (575,475), cv2.FONT_HERSHEY_DUPLEX, 1, (123, 142, 97), 2)
    #print(str(pred))
    cv2.imshow('test', frame)
    cv2.imshow('BW', BWimg)
    cv2.waitKey(1)
    try:
        if keyboard.is_pressed('w'):
            THRESHOLD += 1
            print("Threshold: ",THRESHOLD)
        elif keyboard.is_pressed('s'):
            THRESHOLD -= 1
            print("Threshold: ",THRESHOLD)
    except:
        continue
        
    
       
    


cam.release()