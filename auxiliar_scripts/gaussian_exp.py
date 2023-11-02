# -*- coding: utf-8 -*-
"""
Created on Wed May 24 20:44:48 2023

@author: RubenSilva
"""


import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import random

def noisy_gaussian(image,mask):
      row,col= image.shape
      shape=row,col
      
      points_mask = np.argwhere(mask == 1)
      if len(points_mask>0):
          index=random.randint(0,len(points_mask)-1)
          
          random_n=1  #random.randint(1,2)
          
          if random_n==1:
              
              index2=random.randint(0,len(points_mask)-1)
              gauss2=makeGaussian(row,center=points_mask[index2])
              gauss2=gauss2.reshape(row,col)*mask
              
          else:
              gauss2=0
              
          gauss=makeGaussian(row,center=points_mask[index])
          gauss=gauss.reshape(row,col)*mask
          
          
          #gauss=gauss.astype(np.uint16)
          noisy = image + gauss + gauss2
          noisy[noisy>1] = 1
          #noisy=noisy.astype(np.uint16)
          
      else:
          noisy=image
      return noisy

def makeGaussian(size, sigma_x=10,sigma_y=10,theta=0,center=None):
    """ Make a square gaussian kernel.

    size is the length of a side of the square
    fwhm is full-width-half-maximum, which
    can be thought of as an effective radius.
    """

    x = np.arange(0, size, 1, float)
    y = x[:,np.newaxis]

    if center is None:
        xo = yo = size // 2
    else:
        xo = center[1]
        yo = center[0]
    
    sigma_x=random.randint(1,8)
    sigma_y=random.randint(1,8)
    theta=random.randint(0,180)
    amplitude=1#  4*(10e3)

    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)

    g = amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) 
                            + c*((y-yo)**2)))
    
    return g.ravel().reshape(size, size)



PATH ="X:/Ruben/TESE/New_training_Unet/data_only_pericardium/Cardiac_fat_new/4"
path_to_copy="X:/Ruben/TESE/Data/Dataset_public/RioFatSegm/Organization/teste_calcification"
list_patients=sorted(os.listdir(PATH+'/Dicom'))
list_patients=[list_patients[0]]
# path_dicom='X:/Ruben/TESE/New_training_Unet/all_data/OSIC_new/4/Dicom/'
# path_label='X:/Ruben/TESE/New_training_Unet/all_data/OSIC_new/4/Mask/'
import glob

for patient in list_patients:
    dicom_files= os.path.join(PATH,'Dicom',patient)
    files=sorted(glob.glob(dicom_files+'/*'))
    for file in files:
        
        name_file=os.path.split(file)[-1] #only the name of the image, not the entire path
        file_mask=os.path.join(PATH,'Mask',patient,name_file)
        
        img=cv2.imread(file,flags=cv2.IMREAD_ANYDEPTH)
        img_mask=cv2.imread(file_mask,0)
        
        path_test_calcification=os.path.join(path_to_copy, patient)
        isExist = os.path.exists(path_test_calcification)
        
        if not isExist:                         
          # Create a new directory because it does not exist 
          os.makedirs(path_test_calcification)
        
        print(file)
        img_with_noise=noisy_gaussian(img/65535., img_mask/255.)*65535
        cv2.imwrite(os.path.join(path_test_calcification,name_file),img_with_noise.astype(np.uint16))
        
        #plt.imshow(img_with_noise,cmap='gray')
  