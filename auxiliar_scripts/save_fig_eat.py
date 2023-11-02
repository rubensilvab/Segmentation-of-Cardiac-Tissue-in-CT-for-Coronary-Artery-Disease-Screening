# -*- coding: utf-8 -*-
"""
Created on Thu May  4 19:37:55 2023

@author: RubenSilva
"""
import nrrd
import matplotlib.pyplot as plt
import numpy as np
import os,glob

def reshape_nrrd_to_arr(nrrd,n):
  
  nrrd=np.array(nrrd)
  nrrd=np.transpose(nrrd,(2,1,0))
  if n==1:  
      nrrd=nrrd[::-1] 
  #nrrd=nrrd[::-1]  
  return nrrd  

def single_dice_coef(y_true, y_pred_bin):
    # shape of y_true and y_pred_bin: (height, width)
    intersection = np.sum(y_true * y_pred_bin)
    if (np.sum(y_true)==0) and (np.sum(y_pred_bin)==0):
        return 1
    return round((2*intersection) / (np.sum(y_true) + np.sum(y_pred_bin)),3)

def save_fig(X,Y,pred_test,path,patient):
    
    isExist = os.path.exists(path)
    if not isExist:                         
        # Create a new directory because it does not exist 
        os.makedirs(path)
        
    os.chdir(path) 
    
    s=0
    for i in range (len(X)):
     s=s+1 
     fig=plt.figure(figsize=(16,6))
     fig.suptitle('Dice:'+str(round(single_dice_coef(Y[i], np.squeeze(pred_test[i])),3)))
     plt.subplot(1,3,1)
     plt.imshow(np.squeeze(X[i]),cmap='gray')
     plt.title('Original Teste_'+str(s))
     plt.subplot(1,3,2)
     plt.imshow(np.squeeze(Y[i]),cmap='gray')
     plt.title('label Test_'+str(s))
     plt.subplot(1,3,3)
     plt.imshow(np.squeeze(pred_test[i]),cmap='gray')
     plt.title('Predict_'+str(s))
     fig.savefig('Predicts_test_'+str(patient)+"_"+str(s)+'.jpg')
     plt.close('all')


path_nrrd="X:/Ruben/TESE/New_training_Unet/Models_only_pub_dset/Mix_datasets/All_data/predict/2.5D_Unet/Dice_loss/Hospital_tif/L0_W2000_calc_augm_tif/Peri_segm/NRRD"

path_to_move=path_nrrd.split('/')
path_to_move=('/').join(path_to_move[:13])
     
patients=[patient for patient in os.listdir(path_nrrd) if os.path.isdir(os.path.join(path_nrrd, patient)) ]

c=0
for patient in patients:

    
    if 'ospit' in path_nrrd:
        
        print('hospital')
        n=0
    elif 'ardiac' in path_nrrd:
        
        print('Cardiac')
        n=1
    else:
        
        print('OSIC')
        
        n=1    

    print('Faltam',len(patients)-c,' pacientes. Atual:',patient)
    path_nrrd_patient=os.path.join(path_nrrd,patient)
    
    #buscar nrrds
    files_nrrd=glob.glob(os.path.join(path_nrrd_patient,'*'))

    for file in files_nrrd:
        
        if file[-6]=='v':
            Convex_mask, header = nrrd.read(file)
            Convex_mask=reshape_nrrd_to_arr(Convex_mask,n)
        elif file[-6]=='d':   
            fill2d_mask, header = nrrd.read(file)
            fill2d_mask=reshape_nrrd_to_arr(fill2d_mask,n)
        elif file[-6]=='6':   
            dicom, header = nrrd.read(file)
            dicom=reshape_nrrd_to_arr(dicom,n)
        elif file[-6]=='l':   
            manual, header = nrrd.read(file)
            manual=reshape_nrrd_to_arr(manual,n)
    
    path_convex=os.path.join(path_to_move,'Convex Hull',patient)
    path_fill2d=os.path.join(path_to_move,'Fill 2d',patient)  
    
    save_fig(dicom,manual,Convex_mask,path_convex,patient)    
    save_fig(dicom,manual,fill2d_mask,path_fill2d,patient)      
    c+=1 