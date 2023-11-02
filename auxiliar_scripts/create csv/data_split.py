# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 13:21:10 2022

@author: RubenSilva
"""

"""Separate Pericardium masks with those that dont have pericardium in other fold""" 
import os
import glob
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import save_img
from tensorflow.keras.preprocessing.image import img_to_array
import shutil

path="X:/Ruben/TESE/Data/Dataset_public/RioFatSegm/Organization/split_by_patient_two"  

path_to_copy="X:/Ruben/TESE/New_training_Unet/all_data"

dataset_name="Cardiac_fat"

list_patients=sorted(os.listdir(os.path.join(path,"Dicom")))

"""Number patients in train"""
n_train=15
n_val=5

"""TRAINING DATA"""
for patient in list_patients[0:n_train]:
    files_y=sorted(glob.glob(os.path.join(path,"Convex_Mask",patient+'/*')))
    
    for file_y in files_y:
        
       
            path_train_masks=os.path.join(path_to_copy,dataset_name,"train_masks/train", patient)
            isExist = os.path.exists(path_train_masks)
            if not isExist:                         
              # Create a new directory because it does not exist 
              os.makedirs(path_train_masks)
            shutil.copy(file_y, path_train_masks)  
            
            #Save Dicom
            name_filey=os.path.split(file_y)[-1] #only the name of the image, not the entire path
            
            path_train_dicom=os.path.join(path_to_copy,dataset_name,"train_images/train", patient)
            isExist = os.path.exists(path_train_dicom)
            if not isExist:                         
             # Create a new directory because it does not exist 
             os.makedirs(path_train_dicom)
            file_x=os.path.join(path,"Dicom",patient,name_filey)
            shutil.copy(file_x, path_train_dicom)

""" VALIDATION DATA"""            
for patient in list_patients[n_train:n_train+n_val]:
    files_y=sorted(glob.glob(os.path.join(path,"Convex_Mask",patient+'/*')))
    
    for file_y in files_y:
        
      
            path_train_masks=os.path.join(path_to_copy,dataset_name,"val_masks/val", patient)
            isExist = os.path.exists(path_train_masks)
            if not isExist:                         
              # Create a new directory because it does not exist 
              os.makedirs(path_train_masks)
            shutil.copy(file_y, path_train_masks)  
            
            #Save Dicom
            name_filey=os.path.split(file_y)[-1] #only the name of the image, not the entire path
            dataset_name
            path_train_dicom=os.path.join(path_to_copy,dataset_name,"val_images/val", patient)
            isExist = os.path.exists(path_train_dicom)
            if not isExist:                         
             # Create a new directory because it does not exist 
             os.makedirs(path_train_dicom)
            file_x=os.path.join(path,"Dicom",patient,name_filey)
            shutil.copy(file_x, path_train_dicom)            