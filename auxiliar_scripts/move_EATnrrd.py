# -*- coding: utf-8 -*-
"""
Created on Thu May  4 17:52:27 2023

@author: RubenSilva
"""
import os
import glob
import shutil


path_nrrd="X:/Ruben/TESE/New_training_Unet/Models_only_pub_dset/Mix_datasets/All_data/predict/2.5D_Unet/Dice_loss/Hospital_tif/L0_W2000_calc_augm_tif/EAT_segm_nHU/NRRD"
# path_to_move=path_nrrd.split('/')

# path_to_move=os.path.join(('/').join(path_to_move[:12]),'EAT_segm')

path_to_move="C:/Users/RubenSilva/Desktop/segmentation_inter_intra/selection/inverted/EAT_segm_nHU/NRRD"

patients=[patient for patient in os.listdir(path_to_move) if os.path.isdir(os.path.join(path_to_move, patient)) ]

for patient in patients:
    path_patient=os.path.join(path_to_move,patient)
    
    
    isExist = os.path.exists(path_patient)
    if not isExist:                         
        # Create a new directory because it does not exist 
        os.makedirs(path_patient)
        
    path_nrrd_patient=os.path.join(path_nrrd,patient)
    
    #buscar nrrds
    files_nrrd=glob.glob(os.path.join(path_nrrd_patient,'*'))
    
    for file in files_nrrd:
        if 'EAT'in file:
         
            name_file=os.path.split(file)[-1] #only the name of the file
            print(name_file)
            shutil.copy(file, path_patient +'/'+name_file)
            
        elif file[-6]=='6':    
            name_file=os.path.split(file)[-1] #only the name of the file
            print(name_file)
            shutil.copy(file, path_patient +'/'+name_file)
