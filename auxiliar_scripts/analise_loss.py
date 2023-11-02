# -*- coding: utf-8 -*-
"""
Created on Sat May  6 19:41:52 2023

@author: RubenSilva
"""


# Load the 3D NRRD file
filename = 'X:/Ruben/TESE/New_training_Unet/Models_only_pub_dset/Mix_datasets/All_data/predict/2.5D_Unet/Dice_loss/Hospital_tif/L0_W2000_calc_augm_tif/NRRD/777072/777072_256_2.5UNet.nrrd'
# Load the 3D NRRD file
filename_manual = 'X:/Ruben/TESE/New_training_Unet/Models_only_pub_dset/Mix_datasets/All_data/predict/2.5D_Unet/Dice_loss/Hospital_tif/L0_W2000_calc_augm_tif/NRRD/777072/777072_256_manual.nrrd'


def reshape_nrrd(nrrd):
  
  nrrd=np.array(nrrd)
  nrrd=np.transpose(nrrd,(2,1,0))
  #nrrd=nrrd[::-1]  
  return nrrd  

import cc3d
import numpy as np
import nrrd
from matplotlib import pyplot as plt
 #     # Read the data back from file
 #Predict
predict, header = nrrd.read(filename)
#Manual
manual, header = nrrd.read(filename_manual)

predict,manual=reshape_nrrd(predict).resize(-1,256,256),reshape_nrrd(manual).resize(-1,256,256)
import segmentation_models as sm

Dice_loss=sm.losses.DiceLoss()
print(Dice_loss(manual[0],predict[0]).numpy)