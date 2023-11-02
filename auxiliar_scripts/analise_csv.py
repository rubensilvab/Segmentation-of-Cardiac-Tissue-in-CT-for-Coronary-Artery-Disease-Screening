# -*- coding: utf-8 -*-
"""
Created on Sat Mar  4 12:36:51 2023

@author: RubenSilva
"""
import os
#os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.6/bin")

import numpy as np 
from matplotlib import pyplot as plt
import numpy as np
import glob
import cv2
import time
import pandas as pd
"Import CSV with dicom and masks informations"

#path1='X:/Ruben/TESE/New_training_Unet/Models_only_pub_dset/Mix_datasets/All_data/classification_models/predict/2D_CNN/BCE/Cardiac_fat_tif/L0_W2000_tif_calc_augm/th_0.517566/Peri_segm/NRRD'
path1='X:/Ruben/TESE/New_training_Unet/Models_only_pub_dset/Mix_datasets/All_data/predict/2.5D_Unet/Dice_loss/Hospital_tif/L0_W2000_calc_augm_tif/Peri_segm/NRRD'
path2='X:/Ruben/TESE/New_training_Unet/Models_only_pub_dset/Mix_datasets/All_data/predict/2.5D_Unet/Dice_loss/Hospital_tif/L0_W2000_calc_augm_tif/Peri_segm/NRRD'

#W2000= pd.read_csv(os.path.join(path1,'NRRD_distmet_manual_slc+2.5d.csv'))
#W2000= pd.read_csv(os.path.join(path1,'NRRD_distmet_manual_3D2DNet.csv'))
W2000= pd.read_excel(os.path.join(path1,'combined_data_pp+conv.xlsx'))
W350 = pd.read_excel(os.path.join(path2,'combined_data.xlsx'))
#W350 = pd.read_csv(os.path.join(path2,'NRRD_distmet_manual_2.5UNet.csv'))
#NRRD_distmet_manual_UNet.csv
"Análise Dice"

W2000_dcs,W350_dsc=W2000['dice'],W350['dice']

#name1=path1.split('/')[-4]+'_slc_2.5d'
name1=path1.split('/')[-3]+'_2.5d_conv'
name2=path2.split('/')[-3]+'_2.5d'

path_save_csv='/'.join(path1.split('/')[:-3])
print(name1,name2,path_save_csv)
if not os.path.exists(path_save_csv):                         
     # Create a new directory because it does not exist 
     os.makedirs(path_save_csv)
os.chdir(path_save_csv) 

f= open('report'+name1+"_vs_"+name2+".txt","w+")

limit1=0.1
patients_to_see_dcs=[]
f.write("Dice analysis with limit: "+str(limit1)+'\r\n')
for i,dice in enumerate(W2000_dcs):
    patient=W2000['patient'][i]
    p_index = W350.index[W350['patient'] == patient][0]
    W350_dsc=W350['dice'][p_index]
    dif=abs(dice-W350_dsc)
    print(patient,W350_dsc)
    if dif>limit1:
       patients_to_see_dcs.append(W2000['patient'][i]) 
       f.write("Patient: "+str(W2000['patient'][i])+', ')
       f.write(name1+': ' +str(dice)+', '+name2+' :'+ str(W350_dsc) +'\n')    
              
   
"Análise jaccard"    

W2000_jcc,W350_jcc=W2000['jaccard'],W350['jaccard']
f.write('\r\n')
patients_to_see_jcc=[]
f.write("Jaccard analysis with limit: "+str(limit1)+'\r\n')
for i,jcc in enumerate(W2000_jcc):
    patient=W2000['patient'][i]
    p_index = W350.index[W350['patient'] == patient][0]
    
    W350_jcc=W350['jaccard'][p_index]
    dif=abs(jcc-W350_jcc)
    
    if dif>limit1:
       patients_to_see_jcc.append(W2000['patient'][i])
       f.write("Patient: "+str(W2000['patient'][i])+', ') 
       f.write(name1+': ' +str(jcc)+', '+name2+' :'+ str(W350_jcc) +'\n')        
           # f.write("Date end: "+(local_time_end)+'\n')

"Análise hd"    
limithd=50

W2000_hd,W350_hd=W2000['hd'],W350['hd']
f.write('\r\n')
patients_to_see_hd=[]
f.write("HD analysis with limit: "+str(limithd)+'\r\n')
for i,hd in enumerate(W2000_hd):
    patient=W2000['patient'][i]
    p_index = W350.index[W350['patient'] == patient][0]
    
    W350_hd=W350['hd'][p_index]
      
    dif=abs(hd-W350_hd)
    
    if dif>limithd:
       patients_to_see_hd.append(W2000['patient'][i])
       f.write("Patient: "+str(W2000['patient'][i])+', ')
       f.write(name1+': ' +str(hd)+', '+name2+' :'+ str(W350_hd) +'\n')  
"Análise mad"    

limitmad=0.4

W2000_mad,W350_mad=W2000['mad'],W350['mad']
f.write('\r\n')
patients_to_see_mad=[]
f.write("MAD analysis with limit: "+str(limitmad)+'\r\n')
for i,mad in enumerate(W2000_mad):
    patient=W2000['patient'][i]
    p_index = W350.index[W350['patient'] == patient][0]
    
    W350_mad=W350['mad'][p_index]
    
     
    dif=abs(mad-W350_mad)
     
    
    if dif>limitmad:
        patients_to_see_mad.append(W2000['patient'][i])   
        f.write("Patient: "+str(W2000['patient'][i])+', ')
        f.write(name1+': ' +str(mad)+', '+name2+' :'+ str(W350_mad) +'\n')  

f.close()    
   
    
   
