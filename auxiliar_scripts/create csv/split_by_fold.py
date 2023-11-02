# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 16:16:29 2023

@author: RubenSilva
"""


import os
import glob
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import save_img
from tensorflow.keras.preprocessing.image import img_to_array
import shutil

path="X:/Ruben/TESE/Data/Dataset_public/RioFatSegm/Organization/split_by_patient_two/only_pericardium"  

path_to_copy="X:/Ruben/TESE/New_training_Unet/data_only_pericardium"

dataset_name="Cardiac_fat_new"

list_patients=sorted(os.listdir(os.path.join(path,"Dicom-1000_1000")))


number_folds=5
patients_per_fold=int(len(list_patients)/5)


initial_patient=0
for fold in range(number_folds):
    
    for patient in list_patients[initial_patient:initial_patient+patients_per_fold]:
        
        files_y=sorted(glob.glob(os.path.join(path,"Convex_Mask",patient+'/*')))
        
        for file_y in files_y:
            
           
                path_train_masks=os.path.join(path_to_copy,dataset_name,str(fold),"Mask", patient)
                isExist = os.path.exists(path_train_masks)
                if not isExist:                         
                  # Create a new directory because it does not exist 
                  os.makedirs(path_train_masks)
                shutil.copy(file_y, path_train_masks)  
                
                #Save Dicom
                name_filey=os.path.split(file_y)[-1] #only the name of the image, not the entire path
                
                path_train_dicom=os.path.join(path_to_copy,dataset_name,str(fold),"Dicom", patient)
                isExist = os.path.exists(path_train_dicom)
                if not isExist:                         
                 # Create a new directory because it does not exist 
                 os.makedirs(path_train_dicom)
                file_x=os.path.join(path,"Dicom-1000_1000",patient,name_filey)
                shutil.copy(file_x, path_train_dicom)
    
    initial_patient=initial_patient+patients_per_fold