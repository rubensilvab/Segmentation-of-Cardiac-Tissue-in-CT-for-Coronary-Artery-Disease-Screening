# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 16:12:15 2023

@author: RubenSilva
"""

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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pandas as pd

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


#path="X:/Ruben/TESE/New_training_Unet/all_data/OSIC_new"  
path1='X:/Ruben/TESE/New_training_Unet/Models_only_pub_dset/Mix_datasets/All_data/predict/2D_Unet/Dice_loss/Hospital_tif/L0_W2000_tif/new_with_resize'
path2='X:/Ruben/TESE/New_training_Unet/Models_only_pub_dset/Mix_datasets/All_data/predict/2D_Unet/Dice_loss/Hospital_tif/L0_W2000_tif_calc/new_with_resize_improved'



list_patients=[407299

]

name1=path1.split('/')[-2]
name2=path2.split('/')[-2]


comp=True # se queremos comparar ou nao,se nao: copiamos o name1

if comp:
    
    name_path=name1+'_VS_'+name2
    path_to_copy="C:/Users/RubenSilva/Desktop/Results/Hospital/2d/Exemplos_dice"

else:
    name_path="melhores"
    path_to_copy="C:/Users/RubenSilva/Desktop/Results/Hospital/2d/Exemplos/"+name1

path_1_csv= pd.read_excel(os.path.join(path1,'NRRD','combined_data.xlsx'))
path_2_csv= pd.read_excel(os.path.join(path2,'NRRD','combined_data.xlsx'))
   
for patient in list_patients:
    
    files_1=sort_specific(sorted(glob.glob(os.path.join(path1,str(patient)+'/*'))))
    files_2=sort_specific(sorted(glob.glob(os.path.join(path2,str(patient)+'/*'))))
    
    inf1=path_1_csv.loc[path_1_csv['patient']==int(patient)]
    inf2=path_2_csv.loc[path_2_csv['patient']==int(patient)]
    
    for sli in range(len(files_1)):
        
          
            path_to_copy_3=os.path.join(path_to_copy,name_path, str(patient))
            isExist = os.path.exists(path_to_copy_3)
            if not isExist:                         
              # Create a new directory because it does not exist 
              os.makedirs(path_to_copy_3)
              
            os.chdir(path_to_copy_3)
            # Create the subplot
            # Create the plot with a single row and two columns
            
            if comp:
                fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(12,6))
            else:
                fig=plt.figure(figsize=(12,6))
            
            image1 = plt.imread(files_1[sli])
            image2 = plt.imread(files_2[sli])
            
            if comp:
                
                # Set the first subplot with the first image
                ax[0].imshow(image1)
                ax[0].axis('off')
                ax[0].text(0.2, 1.1, str(name1), fontsize=11, ha='center', va='bottom', transform=ax[0].transAxes)
                ax[0].text(0.6, 1.1, str(inf1), fontsize=7, ha='left', va='bottom', transform=ax[0].transAxes)
    
    
                # Set the first subplot with the first image
                ax[1].imshow(image2)
                ax[1].axis('off')
                ax[0].text(0.2, -0.1, str(name2), fontsize=11, ha='center', va='bottom', transform=ax[0].transAxes)
                ax[0].text(0.6, -0.1, str(inf2), fontsize=7, ha='left', va='bottom', transform=ax[0].transAxes)

                
            else:
                
                plt.imshow(image1)
                plt.axis('off')
                #plt.text(0.2, 1.1, str(name1), fontsize=11, ha='center', va='bottom')
                plt.text(8, 1.1, str(inf1), fontsize=10, ha='left', va='bottom')
           
            
               
            # Save the subplot as a JPEG image
            plt.savefig(str(patient)+'_'+str(sli)+'.jpg', dpi=300, bbox_inches='tight')
            print(files_1[sli])
            print('')