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

def max_error(csv,th):
    #print(csv)
    y_true=csv['Label']
    y_pred=csv['Prediction']
    y_pred=(y_pred>th).astype(np.uint8)
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
    
    return max_value


csv_ori_cfat=pd.read_excel(os.path.join('X:/Ruben/TESE/New_training_Unet/Models_only_pub_dset/Mix_datasets/All_data/classification_models/predict/2D_CNN/BCE/Cardiac_fat_tif/L0_W2000_tif_calc_augm/Val_tuning','results_dice_Cardiac_fat_tifL0_W2000_tif_calc_augm'+'.xlsx'))
csv_ori_osic=pd.read_excel(os.path.join('X:/Ruben/TESE/New_training_Unet/Models_only_pub_dset/Mix_datasets/All_data/classification_models/predict/2D_CNN/BCE/OSIC_tif/L0_W2000_tif_calc_augm/Val_tuning','results_dice_OSIC_tifL0_W2000_tif_calc_augm'+'.xlsx'))

"Concat the two"

csv_ori=pd.concat([csv_ori_cfat,csv_ori_osic])

y_pred=csv_ori['Prediction']
ths_to_test=(np.unique(y_pred))
ths_to_test=sorted(ths_to_test)

patients=np.unique(csv_ori['Patient'])

csv_all_results=pd.DataFrame({'Thresold':[],'MeanMaxValue':[]})


for th in ths_to_test:
    
    for patient, i in zip(patients,range(len(patients))):
        csv_pat_results=pd.DataFrame({'Patient':[],'MaxValue':[]})

        csv_file=csv_ori.loc[csv_ori['Patient'].isin([(patient)])]
        #print('th:',th,'patient:',patient)
        max_value=max_error(csv_file,th)
        csv_max_values=pd.DataFrame({'Patient':[patient],'MaxValue':[max_value]})
        csv_pat_results=pd.concat([csv_pat_results,csv_max_values])
    
    mean_values_p=np.mean(csv_pat_results['MaxValue'])
    csv_mean_values=pd.DataFrame({'Thresold':[th],'MeanMaxValue':[mean_values_p]})
    csv_all_results=pd.concat([csv_all_results,csv_mean_values])
"Desenhar grafico:"

fig=plt.figure(figsize=(15,15))
plt.plot(csv_all_results['Thresold'],csv_all_results['MeanMaxValue'])
# Customize the plot
plt.xlabel('ThresoldS')
plt.ylabel('MeanMaxValue')
plt.title('Tuning TH')

plt.show()    
plt.savefig('tuning',dpi=300)

