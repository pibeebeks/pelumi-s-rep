#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 01:23:04 2019

@author: lumi
"""

#%%
#import some package to use
import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.utils import  to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 
import random
import os, cv2, itertools
import matplotlib.pyplot as plt




#set the path for both image folder directory
plate_numbers_dir = '/Users/lumi/Downloads/data_set/Train/plate_numbers/'#'./plate_numbers'
non_plate_numbers_dir = '/Users/lumi/Downloads/data_set/Train/non_plate_numbers/'#'./non_plate_numbers'

ROWS = 64
COLS = 64
CHANNELS = 3

#collect images in a list
plate_numbers_img = [plate_numbers_dir+i for i in os.listdir(plate_numbers_dir)]
#to remove broken file in both list
#plate_numbers_img.remove('/Users/lumi/Downloads/data_set/Train/plate_numbers/.DS_Store')
non_plate_numbers_img = [non_plate_numbers_dir+i for i in os.listdir(non_plate_numbers_dir)]
#non_plate_numbers_img.remove('/Users/lumi/Downloads/data_set/Train/non_plate_numbers/.DS_Store')


#create a function  to read and resize images

def read_image(file_path):
    img = cv2.imread(file_path, cv2.IMREAD_COLOR)
    resized_img = cv2.resize(img, (ROWS, COLS), interpolation = cv2.INTER_AREA)
    return resized_img

# function to prepare images for the use of our model
    
def prep_data(images):
    m = len(images)
    n_x = ROWS*COLS*CHANNELS
    X = np.ndarray((n_x, m), dtype = np.uint8)
    y = np.zeros((1, m))
    print("X.shape is {}".format(X.shape))
    for i,image_file in enumerate(images):
        image = read_image(image_file)
        print(i, 'done')
        X[:,i] = np.squeeze(image.reshape((n_x, 1)))
        
        if '-' in image_file.lower():
            y[0,i] = 1
            
        elif '1' or '2' in image_file.lower():
            y[0,i] = 0
            
        if i%100 == 0:
            print("Proceed {} of {}".format(i, m))
    return X, y



plate_img, non_plate_img = prep_data(plate_numbers_img + non_plate_numbers_img) 

#setting classes in the dic as 1 for plate numbers and 0 for non plate numbers
 
classes = {0: 'not a plate_number_image', 1: 'a plate_number_image'}

# A function for displaying images
def show_images(X, y, idx):
    image = X[idx]
    image = image.reshape((ROWS, COLS, CHANNELS))
    plt.figure(figsize=(4,2))
    plt.imshow(image)
    plt.title(classes[y[idx,0]])
    plt.show()
    
    
show_images(plate_img.T, non_plate_img.T, 0)

#This the Linear Regression Model
#import  logistic regression cv model from sklearn
# initialise the LRCV
# fit our data into the model 
# Print the model accuracy on our training set
from sklearn.linear_model import LogisticRegressionCV
clf = LogisticRegressionCV()
plate_img_lr , non_plate_img_lr = plate_img.T, non_plate_img.T.ravel()

clf.fit(plate_img_lr, non_plate_img_lr)
print("Model accuracy: {:.2f}%".format(clf.score(plate_img_lr, non_plate_img_lr)*100))


#function to show image prediction
def show_image_prediction(X, idx, model):
    image = X[idx].reshape(1,-1)
    image_class = classes[model.predict(image).item()]
    image = image.reshape((ROWS, COLS, CHANNELS))
    plt.figure(figsize = (4,2))
    plt.imshow(image)
    plt.title("Test {} : I think this is {}".format(idx, image_class))
    
    plt.show()

#since we don't have a test set, we predict our model on our training set.
    
plate_img_lr , non_plate_img_lr = plate_img.T, non_plate_img.T

for i in np.random.randint(0, len(plate_img_lr), 5):
    show_image_prediction(plate_img_lr, i, clf)

#This is the KNN Classification model
#this model performs poorly in comparison to LRCV model

#from sklearn.neighbors import KNeighborsClassifier
#
#knn = KNeighborsClassifier()
#knn.fit(plate_img_lr, non_plate_img_lr)
#print('Model accuracy: {:.2f}%'.format(knn.score(plate_img_lr, non_plate_img_lr)*100))
#
#plate_img_lr, non_plate_img_lr = plate_img.T, non_plate_img.T
#
#for i in np.random.randint(0, len(plate_img_lr),5):
#    show_image_prediction(plate_img_lr, i, knn)

# This is Radius nearest  neighbor
# first print the accuracy 

#from sklearn.neighbors import RadiusNeighborsClassifier
#rnc = RadiusNeighborsClassifier()
#rnc.fit(plate_img_lr, non_plate_img_lr)
#print('Model accuracy: {:.2f}%'.format(rnc.score(plate_img_lr, non_plate_img_lr)*100))
#
##try the model
#plate_img_lr, non_plate_img_lr = plate_img.T, non_plate_img.T
#
#for i in np.random.randint(0, len(plate_img_lr),5):
#    show_image_prediction(plate_img_lr, i, rnc)i
#
#
