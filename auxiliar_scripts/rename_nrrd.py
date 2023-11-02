# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 23:05:55 2023

@author: RubenSilva
"""


import os
import glob
os.getcwd()
collection = "X:/Ruben/TESE/New_training_Unet/Models_only_pub_dset/Mix_datasets/All_data/predict/Dice_loss/Hospital_tif/L50_W350_tif/new/NRRD"
#fold_patients="X:/Ruben/TESE/Data/Dataset_public/RioFatSegm/Organization/split_by_patient_two/Dicom-1000_1000"
patients=sorted(os.listdir(collection))
#name_patients=sorted(os.listdir(fold_patients))

for patient in patients:
    
    files=sorted(glob.glob(os.path.join(collection,str(patient),'*')))
    # old_name=os.path.join(collection,str(patient))
    # new_name=os.path.join(collection,name_patients[int(patient)])
    # print(new_name)  
    # os.rename(old_name, new_name)

    
    for file in files:
        
        old_name=file.split('_')
        if old_name[-1]=='manuall.nrrd':
            old_name=old_name[:-1]
            old_name='_'.join(old_name)
            new_name=old_name+'_manual.nrrd'
            print('antiga manual:',file,old_name,new_name)
            os.rename(file, new_name)
        # if old_name[-1]=='Unet.nrrd':
        #     old_name=old_name[:-1]
        #     old_name='_'.join(old_name)
        #     new_name=old_name+'_manuall.nrrd'   
        #     print('antiga Unet:',file,old_name,new_name)
        #     os.rename(file, new_name)
       
            
       
       # os.rename(file, new_name)
    