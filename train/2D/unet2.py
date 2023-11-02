# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 12:46:59 2023

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
#from new_model import Width,Length
#Width,Length=256,256

def conv_block(input, num_filters, Bath_norm=False):
    x = Conv2D(num_filters, 3, padding="same",kernel_initializer = 'he_normal')(input)
    if Bath_norm==True:
        x = BatchNormalization()(x)   #Not in the original network. 
    x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, padding="same",kernel_initializer = 'he_normal')(x)
    if Bath_norm==True:
        x = BatchNormalization()(x)  #Not in the original network
    x = Activation("relu")(x)

    return x

#Encoder block: Conv block followed by maxpooling


def encoder_block(input, num_filters,Drop_out=False):
    
    x = conv_block(input, num_filters)
    if Drop_out==True:
        x= Dropout(0.5)(x)
        
    p = MaxPooling2D((2, 2))(x)
    return x, p   

#Decoder block
#skip features gets input from encoder for concatenation

def decoder_block(input, s, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same",kernel_initializer = 'he_normal')(input)
    #s = attention_gate(x, s, num_filters)
    x = Concatenate()([x, s])
    x = conv_block(x, num_filters)
    return x

def attention_gate(g, s, num_filters):
    Wg = Conv2D(num_filters, 1, padding="same",kernel_initializer = 'he_normal')(g)
    Wg = BatchNormalization()(Wg)
 
    Ws = Conv2D(num_filters, 1, padding="same",kernel_initializer = 'he_normal')(s)
    Ws = BatchNormalization()(Ws)
 
    out = Activation("relu")(Wg + Ws)
    out = Conv2D(num_filters, 1, padding="same",kernel_initializer = 'he_normal')(out)
    out = Activation("sigmoid")(out)
 
    return out * s

#Build Unet using the blocks
def build_unet(input_shape, n_classes):
    inputs = Input(input_shape)

    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512,Drop_out=True)
    
    
    b1 = conv_block(p4, 1024) #Bridge
    b1 = Dropout(0.5)(b1)

    d1 = decoder_block(b1, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)

    if n_classes == 1:  #Binary
      activation = 'sigmoid'
    else:
      activation = 'softmax'
      
    d5 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(d4)
    outputs = Conv2D(n_classes, 1, padding="same", activation=activation)(d5)  #Change the activation based on n_classes
    print(activation)

    model = Model(inputs, outputs, name="U-Net")
    return model

# model=build_unet((Width,Length,1),1)
# model.summary()
