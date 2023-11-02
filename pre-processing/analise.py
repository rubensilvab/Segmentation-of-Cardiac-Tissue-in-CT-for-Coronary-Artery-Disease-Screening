# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 22:41:52 2022

@author: RubenSilva
"""
import os
import glob
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import save_img
from tensorflow.keras.preprocessing.image import img_to_array
import cv2
import matplotlib.pyplot as plt
import pydicom

PATH_X="X:/Ruben/TESE/Data/hospital_gaia/all_data_carolina/Dicom/"
PATH_Y="X:/Ruben/TESE/Data/hospital_gaia/all_data_carolina/Mask/"
list_patients=sorted(os.listdir(PATH_X))

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

             
  return sorted_files           

numero_pixeis_peri=0
numero_pixeis_sem_ser_pericardio=0
for patient in list_patients:
    files_x=sorted(glob.glob(PATH_X+patient+'/*'))
    files_y=sorted(glob.glob(PATH_Y+patient+'/*'))
    files_x=sort_specific(files_x)
    files_y=sort_specific(files_y)
    
    #pick the middle slice to compare
    slice_m=int(len(files_x)/2)
    print("slice m:",slice_m,"patient: ", patient)
    fig=plt.figure(figsize=(10,10))
    img_x=cv2.imread(files_x[slice_m],0)
    img_y=cv2.imread(files_y[slice_m],0)
    plt.subplot(1,2,1)
    plt.imshow(img_x,cmap='gray')
    plt.title(str(patient)+" s: "+str(slice_m))
    plt.subplot(1,2,2)
    plt.imshow(img_y,cmap='gray')
    plt.title(str(patient)+" s: "+str(slice_m))
    
    
# numero_pixeis_peri=0
# numero_pixeis_no_peri=0    
# Width=256
# Length=256
# PATH_X="X:/Ruben/TESE/New_training_Unet/output/train_masks/train" 
   

# files_y=sorted(glob.glob(PATH_X+'/*'))


# #pick the middle slice to compare
# for file in files_y:
#     img_y=cv2.imread(file,0)
#     img_y=cv2.resize(img_y, (Width, Length))
#     img_yn=img_y/255
#     img_yn=(img_yn>0.5).astype(np.uint8)
#     peri=np.sum(img_yn)
#     numero_pixeis_peri=numero_pixeis_peri+peri
#     no_peri=img_yn.shape[0]*img_yn.shape[1]-peri
#     numero_pixeis_no_peri=numero_pixeis_no_peri+no_peri
#     print("pixeis_peri:",peri,"pixeis_no_peri:",no_peri,"shape:",img_yn.shape)
    
# """Separate Pericardium masks with those that dont have pericardium in other fold""" 

# import shutil

# path_masks="X:/Ruben/TESE/Data/Dataset_public/Orcya/img_png/Mask_convex"  
# path_dicom="X:/Ruben/TESE/Data/Dataset_public/Orcya/img_png/Dicom"

# path_to_copy="X:/Ruben/TESE/Data/Dataset_public/Orcya/img_png/only_pericardium"

# list_patients=sorted(os.listdir(path_masks))

# for patient in list_patients:
#     files_y=sorted(glob.glob(os.path.join(path_masks,patient+'/*')))
    
#     for file_y in files_y:
        
#         img_y=cv2.imread(file_y,0)
#         img_yn=img_y/255
#         img_yn=(img_yn>0.5).astype(np.uint8)
#         peri=np.sum(img_yn)
#         if peri>0:
#             path=os.path.join(path_to_copy,"Convex_Mask", patient)
#             isExist = os.path.exists(path)
#             if not isExist:                         
#               # Create a new directory because it does not exist 
#               os.makedirs(path)
#             shutil.copy(file_y, path)  
            
#             #Save Dicom
#             name_filey=os.path.split(file_y)[-1] #only the name of the image, not the entire path
            
#             path=os.path.join(path_to_copy,"Dicom", patient)
#             isExist = os.path.exists(path)
#             if not isExist:                         
#              # Create a new directory because it does not exist 
#              os.makedirs(path)
#             file_x=os.path.join(path_dicom,patient,name_filey)
#             shutil.copy(file_x, path)