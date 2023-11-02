# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 15:06:09 2023

@author: RubenSilva
"""
from tensorflow import keras
import numpy as np 

#import skimage.io as io
#import skimage.transform as trans
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
import glob
import cv2

#from slice_classification_framework import Width,Length
W=256
L=W

def cnn(pretrained_weights = None,input_size = (W,L,1)):
    #Now we can go ahead and create our Convolution model
   
    model = Sequential()  
    # 3 Camadas convolucionais 32 filters cada
    model.add(Conv2D(32, (3, 3), input_shape=(W,L,1), padding='same',
                     activation='relu')) 
    
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(AveragePooling2D(pool_size=(2, 2),strides=(1,1)))
    
    model.add(Flatten())
    
    model.add(Dense(256, activation='relu'))
   
    model.add(Dense(64, activation='relu'))
    
    #our 2 classes
    model.add(Dense(1, activation='sigmoid'))
    
    return model

