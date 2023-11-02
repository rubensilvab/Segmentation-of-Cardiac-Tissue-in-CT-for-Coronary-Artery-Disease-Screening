# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 11:47:02 2023

@author: RubenSilva
"""
import os
import glob
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import save_img
from tensorflow.keras.preprocessing.image import img_to_array
import shutil
import csv
import cv2

def sort_specific(files):
  sorted_files=[]
  for file in files:
         order=file[-7:-3]
         if order[1]=='_':
             sorted_files.append(file)
  for file in files:
         order=file[-7:-3]
         if order[0]=="_":
             sorted_files.append(file)  
  for file in files:
         order=file[-8:-3]
         if order[0]=="_":
             sorted_files.append(file)  
  return sorted_files  


path="X:/Ruben/TESE/New_training_Unet/all_data/Cardiac_fat_new"  
#path="X:/Ruben/TESE/Data/hospital_gaia/all_data_carolina"

folds=sorted(os.listdir(path))
folds = [fold for fold in folds if '.csv' not in fold] 

# "Caso do hospital folds=0"
# folds=['Dicom-1000_1000']

os.chdir(path)
#write csv
name_csv=os.path.split(path)[-1]
with open(name_csv+'_sorted_'+str(len(folds))+'.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Patient", "Fold", "Path_image","Path_Mask","Label"])

    for fold in folds:
        
        path_dicom=os.path.join(path,str(fold),"Dicom") 
        path_mask=os.path.join(path,str(fold),"Mask")
        
        # path_dicom=os.path.join(path,str(fold)) # para hospital
        # path_mask=os.path.join(path,"Mask")
        
        list_patients=sorted(os.listdir(path_dicom))
        
        for patient in list_patients:
            
            files=sort_specific(sorted(glob.glob(os.path.join(path_dicom,str(patient),'*'))))
        
            for file in files:
                #save dicom path to csv   
                print(file)
                #Save mask path
                name_file=os.path.split(file)[-1] #only the name of the image, not the entire path
                
                path_mask_img=os.path.join(path_mask,patient,name_file)
                mask=cv2.imread(path_mask_img,0)
                
                if mask.sum()>1:
                    label=1
                else:
                    label=0
                
                print(path_mask_img)
               
                writer.writerow([str(patient), fold, file,path_mask_img,label])
                    