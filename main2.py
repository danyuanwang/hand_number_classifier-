import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2

import time

model = tf.keras.models.load_model("new.model")

def guess(img):
    img = cv2.resize(img, (224, 224))

    img = img[np.newaxis,...]
    hi = model.predict(img)[0]
    #print(hi)
    print(np.argmax(hi))
    return np.argmax(hi)


cam = cv2.VideoCapture(0)


while True:
    ret, frame = cam.read()
    rgbFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    cv2.putText(frame, str(guess(rgbFrame)), (575,475), cv2.FONT_HERSHEY_DUPLEX, 1, (123, 142, 97), 2)
        
    cv2.imshow('test', frame)
    cv2.waitKey(1)
    
       
    


cam.release()






