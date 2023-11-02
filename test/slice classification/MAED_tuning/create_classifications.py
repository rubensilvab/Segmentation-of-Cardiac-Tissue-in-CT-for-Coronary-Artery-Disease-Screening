# -*- coding: utf-8 -*-
"""
Created on Sat Jul  8 17:00:22 2023

@author: RubenSilva
"""
import nrrd
#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pydicom 
import numpy as np
import numpy as np
from skimage import io, exposure, img_as_uint, img_as_float
import os
import glob
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import save_img
from tensorflow.keras.preprocessing.image import img_to_array
import cv2
import pandas as pd
import openpyxl
def reshape_nrrd(nrrd):
  
  nrrd=np.array(nrrd)
  nrrd=np.transpose(nrrd,(2,1,0))
  nrrd=nrrd[::-1]  
  return nrrd  

def get_excel_row(path,patient,name,slic,label,pred):
    
    isExist = os.path.exists(path)
    if not isExist:                         
        # Create a new directory because it does not exist 
        os.makedirs(path)
        
    os.chdir(path) 
    
        
    filename = str(name)+'.xlsx'
    
    # Check if the file already exists
    if not os.path.isfile(filename):
        # Create a new workbook object
        book = openpyxl.Workbook()
    
        # Select the worksheet to add data to
        sheet = book.active
    
        # Add a header row to the worksheet
        sheet.append(['Patient', 'Slice', 'Label', 'Prediction'])
    else:
        # Open an existing workbook
        book = openpyxl.load_workbook(filename)
    
        # Select the worksheet to add data to
        sheet = book.active
    
    # Append a new row of data to the worksheet
    sheet.append([patient, slic, label, pred])
    #print([patient, slic, label, pred])
    # Save the workbook to a file
    book.save(filename)

"Load CSV"

cfat_all_df = pd.read_csv('X:/Ruben/TESE/New_training_Unet/all_data/Cardiac_fat_new/Cardiac_fat_new_sorted_5.csv')
osic_all_df = pd.read_csv('X:/Ruben/TESE/New_training_Unet/all_data/OSIC_new/OSIC_new_sorted_5.csv')
osic_3d=pd.read_csv('X:/Ruben/TESE/New_training_Unet/all_data/OSIC_new/OSIC_3D_test_set.csv')
"Teste set"
test_cfat_df=cfat_all_df.loc[cfat_all_df['Fold'].isin([4])]
test_osic_df=osic_all_df.loc[osic_all_df['Fold'].isin([4])]
test_3d_df=osic_3d.loc[osic_3d['Fold'].isin([4])]
path='X:/Ruben/TESE/New_training_Unet/Models_only_pub_dset/Mix_datasets/All_data/predict/3D_Unet/BCE/OSIC_tif/L0_W2000_augm_calc_tif/NRRD'

csv_ori=test_3d_df
patients=np.unique(csv_ori['Patient'])

for patient, i in zip(patients,range(len(patients))):
    #print(path, patient, str(patient)+'_64_3DNet')  
    PATH_auto =os.path.join(path, str(patient), str(patient)+'_64_3DNet'+'.nrrd')
    PATH_manual=os.path.join(path, str(patient), str(patient)+'_64_manual'+'.nrrd')
    
    #Predict
    readdata, header = nrrd.read(PATH_auto)
    #Manual
    readdata2, header = nrrd.read(PATH_manual)
    
    print('Faltam',len(patients)-i,patient)
    pred=reshape_nrrd(readdata)
    manual=reshape_nrrd(readdata2)
    
    csv_ori_p=csv_ori.loc[(csv_ori['Patient']==patient)]
    
    for sli in range(pred.shape[0]):
        if np.sum(manual[sli]>0):
          peri_manual=1
        else:
            peri_manual=0
            
        if np.sum(pred[sli]>0):
          peri_pred=1
        else:
            peri_pred=0    
        
        name='OSIC_sli_classification_3d'
        slic=csv_ori_p['Path_image'].iloc[sli]
        path_to='X:/Ruben/TESE/New_training_Unet/Models_only_pub_dset/Mix_datasets/All_data/classification_models'
        get_excel_row(path_to,patient,name,slic,peri_manual,peri_pred)
   
    osic_down=pd.read_excel(os.path.join(path_to,name+'.xlsx'))
    osic_down_p=osic_down.loc[(osic_down['Patient']==patient)]
   
    osic_down_p=(osic_down_p.loc[(osic_down_p['Prediction']==1)]).dropna() 
    
    csv_file=test_osic_df.loc[test_osic_df['Patient'].isin([(patient)])]
    first_slc,last_slc=osic_down_p['Slice'].iloc[0],osic_down_p['Slice'].iloc[-1]

    first_index,last_index=csv_file[csv_file['Path_image'] == first_slc].index.tolist(),csv_file[csv_file['Path_image'] == last_slc].index.tolist()
    
    
    for sli in range(len(csv_file)):
            if (csv_file.index[sli]<first_index[0]) or (csv_file.index[sli]>last_index[0]):
              peri_pred=0   
            else:
                peri_pred=1
                
            peri_manual=csv_file.iloc[sli]['Label'] 
            
            name='OSIC_sli_classification_3d_orii'
            slic=csv_file['Path_image'].iloc[sli]
            path_to='X:/Ruben/TESE/New_training_Unet/Models_only_pub_dset/Mix_datasets/All_data/classification_models'
            get_excel_row(path_to,patient,name,slic,peri_manual,peri_pred)
       
    
        