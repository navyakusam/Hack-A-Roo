# Loading the imported librarues for training the data
import numpy
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Input, BatchNormalization
from keras.constraints import maxnorm
from keras.models import Model
from keras.optimizers import *
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
import keras

# Loading the imported modules to convert image to array
import numpy as np
import os
from matplotlib import pyplot as plt
import cv2
import random
import pickle
import glob
import threading
import time


key = cv2.waitKey(1)
webcam = cv2.VideoCapture(0)


def loadModel():
    model = keras.models.load_model('/Users/mdvpr/Documents/UMKC/Python:Deep Learning/Project/model.h5')
    return model


model = loadModel()


def predict(predict_image):
    IMG_SIZE = 120
    # test_image = "/content/drive/MyDrive/project/Hand Gesture Recognition Database Leapmode/scissor/610.jpg"
    ia = cv2.imread(predict_image, cv2.IMREAD_GRAYSCALE)
    ia = cv2.resize(ia, (IMG_SIZE, IMG_SIZE))
    predict = np.array(ia).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    predict = predict.astype('float32')
    predict /= 255

    return np.argmax(model.predict(predict.reshape(-1, 120, 120, 1)))


key = cv2. waitKey(1)
webcam = cv2.VideoCapture(0)

prev_time = time.time()

while True:
    try:
        check, frame = webcam.read()
        print(check) #prints true as long as the webcam is running
        print(frame) #prints matrix values of each framecd 
        cv2.imshow("Capturing", frame)
        key = cv2.waitKey(1)
        
        curr_time = time.time()
        time_elapsed = curr_time - prev_time
        
        if (key == ord('s')) or (time_elapsed > float(10)):
            prev_time = curr_time
            cv2.imwrite(filename='saved_img.jpg', img=frame)
            webcam.release()
            img_new = cv2.imread('saved_img.jpg', cv2.IMREAD_GRAYSCALE)
            img_new = cv2.imshow("Captured Image", img_new)
            cv2.waitKey(1650)
            cv2.destroyAllWindows()
            print("Processing image...")
            img_ = cv2.imread('saved_img.jpg', cv2.IMREAD_ANYCOLOR)
            print("Converting RGB image to grayscale...")
            gray = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)
            print("Converted RGB image to grayscale...")
            print("Resizing image to 28x28 scale...")
            img_ = cv2.resize(gray,(28,28))
            print("Resized...")
            img_resized = cv2.imwrite(filename='saved_img-final.jpg', img=img_)
            print("Image saved!")
            predicted = predict('saved_img.jpg')
            print("predicted : " + str(predicted))
            break
        elif key == ord('q'):
            print("Turning off camera.")
            webcam.release()
            print("Camera off.")
            print("Program ended.")
            cv2.destroyAllWindows()
            break
        
    except(KeyboardInterrupt):
        print("Turning off camera.")
        webcam.release()
        print("Camera off.")
        print("Program ended.")
        cv2.destroyAllWindows()
        break


def capture():
    #threading.Timer(5.0, capture).start()
    print("Capturing....")
    try:
        check, frame = webcam.read()
        print(check)  # prints true as long as the webcam is running
        print(frame)  # prints matrix values of each framecd
        cv2.imshow("Capturing", frame)
        key = cv2.waitKey(1)
        #save_filename = 'saved_img.jpg'
        #cv2.imwrite(filename=save_filename, img=frame)
        predicted = predict(save_filename)
        print("Predicted : " + predicted)
    except:
        print("Turning off camera.")
        webcam.release()
        print("Camera off.")
        print("Program ended.")
        cv2.destroyAllWindows()
        return

# while True:
#     try:
#         check, frame = webcam.read()
#         print(check) #prints true as long as the webcam is running
#         print(frame) #prints matrix values of each framecd
#         cv2.imshow("Capturing", frame)
#         key = cv2.waitKey(1)
#
#
#         if key == ord('s'):
#             cv2.imwrite(filename='saved_img.jpg', img=frame)
#             webcam.release()
#             img_new = cv2.imread('saved_img.jpg', cv2.IMREAD_GRAYSCALE)
#             img_new = cv2.imshow("Captured Image", img_new)
#             cv2.waitKey(1650)
#             cv2.destroyAllWindows()
#             print("Processing image...")
#             img_ = cv2.imread('saved_img.jpg', cv2.IMREAD_ANYCOLOR)
#             print("Converting RGB image to grayscale...")
#             gray = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)
#             print("Converted RGB image to grayscale...")
#             print("Resizing image to 28x28 scale...")
#             img_ = cv2.resize(gray,(28,28))
#             print("Resized...")
#             img_resized = cv2.imwrite(filename='saved_img-final.jpg', img=img_)
#             print("Image saved!")
#
#             break
#         elif key == ord('q'):
#             print("Turning off camera.")
#             webcam.release()
#             print("Camera off.")
#             print("Program ended.")
#             cv2.destroyAllWindows()
#             break
#
#     except(KeyboardInterrupt):
#         print("Turning off camera.")
#         webcam.release()
#         print("Camera off.")
#         print("Program ended.")
#         cv2.destroyAllWindows()
#         break