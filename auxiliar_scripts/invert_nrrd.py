# -*- coding: utf-8 -*-
"""
Created on Fri May 26 17:32:09 2023

@author: RubenSilva
"""

# Load the 3D NRRD file
#patient=746429
#filename = 'X:/Ruben/TESE/New_training_Unet/Models_only_pub_dset/Mix_datasets/All_data/predict/2D_Unet/Dice_loss/Hospital_tif/L0_W2000_tif_augm/Peri_segm/NRRD/'+str(patient)+'/'+str(patient)+'_256_UNet.nrrd'


def reshape_nrrd_to_arr(nrrd):
  
  nrrd=np.array(nrrd)
  nrrd=np.transpose(nrrd,(2,1,0))
  
   
  return nrrd  


def reshape_nrrd(nrrd):
  nrrd=nrrd[::-1] 
  nrrd=np.transpose(nrrd,(2,1,0))  
 
  return nrrd  

import os
import cc3d
import numpy as np
import nrrd
from matplotlib import pyplot as plt
import glob
 #     # Read the data back from file
 
path_nrrd="C:/Users/RubenSilva/Desktop/segmentation_inter_intra/selection"
# Iterate over the data and write it out row by row.
patients=os.listdir(path_nrrd)
patients=patients[:-1]

for patient, i in zip(patients,range(len(patients))):
    #Predict
    #buscar nrrd da previsão e manual
    files_nrrd=glob.glob(os.path.join(path_nrrd,patient,'*'))
    filename=files_nrrd[1]
    
    readdata, header = nrrd.read(filename)
    pred_test=reshape_nrrd_to_arr(readdata)
    print(filename)
    path_to=os.path.join(path_nrrd,'inverted',patient)
    
    isExist = os.path.exists(path_to)
    if not isExist:                         
        # Create a new directory because it does not exist 
        os.makedirs(path_to)
    os.chdir(path_to)  
   
    nrrd.write(str(patient)+'_inv_fabio'+'.nrrd', reshape_nrrd(pred_test),header=header)