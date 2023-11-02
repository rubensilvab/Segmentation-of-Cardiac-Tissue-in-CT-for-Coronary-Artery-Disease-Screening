# -*- coding: utf-8 -*-
"""
Created on Sat Jul  8 16:31:17 2023

@author: RubenSilva
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 12:12:03 2023

@author: RubenSilva
"""
import xlsxwriter
import matplotlib
import matplotlib.pyplot as plt

import os
import glob
import numpy as np
import shutil
import openpyxl
import cv2
import pydicom
import pandas as pd



def max_error(csv):
    #print(csv)
    y_true=csv['Label']
    y_pred=csv['Prediction']
    #First and last indices
    indices_true=[]
    indices_pred=[]
    
    for i in range(len(csv)):
        #print(i)
        if y_pred.iloc[i]==1:
            #print(y_true[i])
            indices_pred.append(i)
            
        if y_true.iloc[i]==1:
            #print(y_true[i])
            indices_true.append(i)
           
    #print(indices_pred)        
    #prediction
    try:
        first_idx_p,last_idx_p=indices_pred[0],indices_pred[-1]
    except:
        first_idx_p,last_idx_p=0,0
    #true
    first_idx,last_idx=indices_true[0],indices_true[-1]
   
    absolute_dif_sup=abs(first_idx_p-first_idx)
    absolute_dif_inf=abs(last_idx_p-last_idx)
    
    max_value=np.mean([absolute_dif_sup,absolute_dif_inf])
    
    return max_value,absolute_dif_sup,absolute_dif_inf


#csv_ori=pd.read_excel(os.path.join('X:/Ruben/TESE/New_training_Unet/Models_only_pub_dset/Mix_datasets/All_data/classification_models','Hosp_sli_classification_3d'+'.xlsx'))
csv_ori=pd.read_excel(os.path.join('X:/Ruben/TESE/New_training_Unet/Models_only_pub_dset/Mix_datasets/All_data/classification_models/predict/2D_CNN/BCE/Hospital_tif/L0_W2000_tif_calc_augm/th_0.517566','results_dice_Hospital_tifL0_W2000_tif_calc_augm'+'.xlsx'))

#csv_ori_osic=pd.read_excel(os.path.join('X:/Ruben/TESE/New_training_Unet/Models_only_pub_dset/Mix_datasets/All_data/classification_models/predict/2D_CNN/BCE/OSIC_tif/L0_W2000_tif_calc_augm/Val_tuning','results_dice_OSIC_tifL0_W2000_tif_calc_augm'+'.xlsx'))


y_pred=csv_ori['Prediction']
patients=np.unique(csv_ori['Patient'])
csv_all_results=pd.DataFrame({'Thresold':[],'MeanMeanValue':[],'Std':[],'Meansup':[],'Std':[],'Meaninf':[],'Std':[]})


csv_pat_results=pd.DataFrame({'Patient':[],'MeanValue':[],'superior':[],'inferior':[]})

csv_thick=pd.read_csv('X:/Ruben/TESE/New_training_Unet/all_data/OSIC_new/OSIC_test_set_thickness.csv')

for patient, i in zip(patients,range(len(patients))):
    

    csv_file=csv_ori.loc[csv_ori['Patient'].isin([(patient)])]
    #print('th:',th,'patient:',patient)
    max_value,absolute_dif_sup,absolute_dif_inf=max_error(csv_file)
    thick_p=csv_thick.loc[csv_thick['Patient'].isin([(patient)])]
    thick_p=3
    print(max_value)
    csv_max_values=pd.DataFrame({'Patient':[patient],'MeanValue':[max_value*thick_p],'superior':[absolute_dif_sup*thick_p],'inferior':[absolute_dif_inf*thick_p]})
    csv_pat_results=pd.concat([csv_pat_results,csv_max_values])

mean_values_p=np.mean(csv_pat_results['MeanValue'])
std=np.std(csv_pat_results['MeanValue'])

mean_values_sup=np.mean(csv_pat_results['superior'])
std_sup=np.std(csv_pat_results['superior'])

mean_values_inf=np.mean(csv_pat_results['inferior'])
std_inf=np.std(csv_pat_results['inferior'])


csv_mean_values=pd.DataFrame({'Thresold':[0.517566],'MeanMeanValue':[mean_values_p],'Std':[std],'Meansup':[mean_values_sup],'Std_sup':[std_sup],'Meaninf':[mean_values_inf],'Std_inf':[std_inf]})
csv_all_results=pd.concat([csv_all_results,csv_mean_values])
"Desenhar grafico:"


