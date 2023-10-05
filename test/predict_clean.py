# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 16:18:44 2022

@author: RubenSilva
"""

import os
import glob
import numpy as np
from numpy import load
import cv2
from keras.models import Model
from keras.models import load_model
import segmentation_models as sm
from tensorflow import keras
import shutil
from save_results import *
from predict_eat import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from openpyxl import Workbook
import nrrd

Width,Length=256,256

"""Load images teste"""

"Load CSV"

cfat_all_df = pd.read_csv('X:/Ruben/TESE/New_training_Unet/all_data/Cardiac_fat_new/Cardiac_fat_new_folds_5.csv')
osic_all_df = pd.read_csv('X:/Ruben/TESE/New_training_Unet/all_data/OSIC_new/OSIC_new_folds_5.csv')

test_cfat_df=cfat_all_df.loc[cfat_all_df['Fold'].isin([4])]
test_osic_df=osic_all_df.loc[osic_all_df['Fold'].isin([4])]

test_hospit_df= pd.read_csv('X:/Ruben/TESE/Data/hospital_gaia/all_data_carolina/all_data_carolina_hospital_1.csv')

all_df=[test_cfat_df,test_osic_df,test_hospit_df]

"""Load the model"""
path_model="X:/Ruben/TESE/New_training_Unet/Models_only_pub_dset/Mix_datasets/All_data/models/2.5D_Unet/Dice_loss/L0_W2000_calc_augm_tif/Sat Apr  8 22_16_42 2023"
os.chdir(path_model)

model = load_model('Loss_Dice_loss__epochs_4000_batch_size_12_wl256Lr_decreasing_0.0001fold_train_0_1_2_time_Sat_Apr_8_22_16_42_2023.h5',compile=False)

for i,df in enumerate(all_df):
  if i==2:  
    if i==0:
        name_df='Cardiac_fat_tif'
    elif i==1:
        name_df='OSIC_tif_new'
    else:
        name_df='Hospital_tif'
        
    path_to=os.path.split(path_model)[0]
    path_to=path_to.split('/')
    range_hu=path_to[-1]
    path_to[7],path_to[-1]='predict', name_df
    path_to.append(range_hu)
    path_to.append('Peri_segm')
    path_to='/'.join(path_to)
    #path_to=os.path.join(path_to,'New')
    print('')
    print(path_to)
    
    if '350' in range_hu:
        change_hu=True
    else:
        change_hu=False
    
    print('Range HU:',range_hu,change_hu)
    
    # """Ver resultados excel por paciente"""
     
    # name="results_dice_"+str(name_df)+str(range_hu)
    # excel_results(df,model,name,path_to,Width,Length,change_hu=change_hu)
    
    # """See results"""
    # save_img_results(df,path_to,model,Width,Length,change_hu=change_hu) 


    """Converter resultados em nrrd"""
    
    path=os.path.join(path_to,'NRRD')
    #convert_nrrd(df,model,path,Width,Length,change_hu=change_hu)
    
    """Predict EAT"""
    predict_EAT(path)