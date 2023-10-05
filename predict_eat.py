# -*- coding: utf-8 -*-
"""
Created on Wed May  3 11:51:50 2023

@author: RubenSilva
"""
import matplotlib.pyplot as plt
import os
import glob
import numpy as np
import shutil
import openpyxl
import cv2
import pydicom

import nrrd

def reshape_nrrd_to_arr(nrrd,n):
  
  nrrd=np.array(nrrd)
  nrrd=np.transpose(nrrd,(2,1,0))
  if n==1:  
      nrrd=nrrd[::-1] 
  #nrrd=nrrd[::-1]  
  return nrrd  

def reshape_arr_to_nrrd(nrrd,n):
  
  if n==1:  
      nrrd=nrrd[::-1] 
      
  nrrd=np.transpose(nrrd,(2,1,0))  
  return nrrd  


def eat_HU(array,minHU,maxHU,L=0,W=2000,rescale=True):
       
       img_min = L - W//2 #minimum HU level, escala em HU da imagem original
       img_max =L + W//2 #maximum HU level
       
       reconvert_img=(array/65535)*(img_max - img_min) +  img_min # Reconvertido para HU
       
       
       new_img_min = minHU #minimum HU level, que pretendemos truncar
       new_img_max =maxHU #maximum HU level, que pretendemos truncar
       
       reconvert_img[reconvert_img<new_img_min] = new_img_min #set img_min for all HU levels less than minimum HU level
       reconvert_img[reconvert_img>new_img_max] = new_img_min #set img_max for all HU levels higher than maximum HU level
       
       if rescale: 
           reconvert_img = (reconvert_img - new_img_min) / (new_img_max - new_img_min)*65535 # Para 16 bit
           
           
       return reconvert_img

def eat_mask(pericardio):
    EAT=eat_HU(pericardio,-150,-50,L=0,W=2000,rescale=True)
    return EAT


def predict_EAT(path_nrrd):
    patients_prob=[]
    #path_nrrd_rd2="C:/Users/RubenSilva/Desktop/segmentation_inter_intra/selection/inverted"
    patients=[ patient for patient in os.listdir(path_nrrd) if os.path.isdir(os.path.join(path_nrrd, patient)) ]
    
    path_to_move=path_nrrd.split('/')
    path_to_move=os.path.join(('/').join(path_to_move[:12]),'EAT_segm_nHU')
    
    
    
    if 'ospit' in path_nrrd:
        
        print('hospital')
        n=0
    elif 'ardiac' in path_nrrd:
        
        print('Cardiac')
        n=1
    else:
        
        print('OSIC')
        
        n=1
    c=0
    for patient in patients:
        
        print('Faltam',len(patients)-c,' pacientes. Atual:',patient)
        
        #buscar nrrd da previsão e manual
        files_nrrd=glob.glob(os.path.join(path_nrrd,patient,'*'))
        #files_nrrd_reader2=glob.glob(os.path.join(path_nrrd_rd2,patient,'*'))
        
        if len(files_nrrd)==0:
            print('patient doesnt exist ', patient )
            patients_prob.append(patient)
            
        
        for file in files_nrrd:
            if file[-6]=='v':
                Convex_mask, header = nrrd.read(file)
                Convex_mask=reshape_nrrd_to_arr(Convex_mask,n)
            
            elif file[-6]=='6':   
                dicom, header = nrrd.read(file)
                dicom=reshape_nrrd_to_arr(dicom,n)
            elif file[-6]=='l':   
                manual, header = nrrd.read(file)
                manual=reshape_nrrd_to_arr(manual,n)
            
        # for filee in files_nrrd_reader2:
        #     if 'fab' in filee:
                
        #      manual_fab, header_fab= nrrd.read(filee)
        #      manual_fab=reshape_nrrd_to_arr(manual_fab,n) 
        #      print('FABB:',filee)
             
                 
             
        #     else:
        #         manual_car, header_car= nrrd.read(filee)
        #         manual_car=reshape_nrrd_to_arr(manual_car,n) 
        #         print('CAROL:',filee)
             
            
        #Label Pericardio:
        # try:
        #   peri_label_fab=dicom*manual_fab
        #   eat_label_fab=np.zeros_like(peri_label_fab)
        # except:
        #     print('Error in fab, patient', patient)
            
        # try:
        #     peri_label_car=dicom*manual_car
        #     eat_label_car=np.zeros_like(peri_label_car)
        # except:
        #     print('Error in carol, patient', patient)
            
        
        #Predicts Pericardio
        peri_predict_conv=dicom*Convex_mask
        #peri_predict_fill2d=dicom*fill2d_mask
        peri_manual=dicom*manual
        
        #Extract EAT from predictions:
        eat_convex=np.zeros_like(Convex_mask)
        #eat_fill2d=np.zeros_like(Convex_mask)
        eat_manual=np.zeros_like(manual)
        
        for i in range(dicom.shape[0]):
            
            eat_convex[i,:,:]=eat_mask(peri_predict_conv[i,:,:])
            #eat_fill2d[i,:,:]=eat_mask(peri_predict_fill2d[i,:,:])
            eat_manual[i,:,:]=eat_mask(peri_manual[i,:,:])

            #eat_label_fab[i,:,:]=eat_mask(peri_label_fab[i,:,:])
            #eat_label_car[i,:,:]=eat_mask(peri_label_car[i,:,:])
            
            
        #eat_convex_1=eat_convex.copy()    
            
        eat_convex[eat_convex>0]=1    
        #eat_fill2d[eat_fill2d>0]=1    
        #eat_label_fab[eat_label_fab>0]=1 
        #eat_label_car[eat_label_car>0]=1 
        eat_manual[eat_manual>0]=1    

        
        eat_convex=eat_convex.astype(np.uint8)
        #eat_label_fab=eat_label_fab.astype(np.uint8)
        #eat_label_car=eat_label_car.astype(np.uint8)
        eat_manual=eat_manual.astype(np.uint8)
        #eat_convex_1=eat_convex_1.astype(np.uint8)
        
        #path_to_move=os.path.join(path_nrrd_rd2,'EAT_segm_nHU')
        
        
        path_to=os.path.join(path_to_move,'NRRD',str(patient))
        isExist = os.path.exists(path_to)
        if not isExist:                         
            # Create a new directory because it does not exist 
            os.makedirs(path_to)
            
        os.chdir(path_to)  
        
        #nrrd.write(str(patient)+'_'+str(256)+'_EAT+rangehu+conv'+'.nrrd', reshape_arr_to_nrrd(eat_convex_1, n),header=header)
        
        nrrd.write(str(patient)+'_'+str(256)+'_EAT+pp+conv'+'.nrrd', reshape_arr_to_nrrd(eat_convex, n),header=header)

        #nrrd.write(str(patient)+'_'+str(256)+'_EAT+fill2d+fill3d'+'.nrrd', reshape_arr_to_nrrd(eat_fill2d,n),header=header)
       
        #nrrd.write(str(patient)+'_'+str(256)+'_EATfabio'+'.nrrd', reshape_arr_to_nrrd(eat_label_fab,n),header=header)
        
        #nrrd.write(str(patient)+'_'+str(256)+'_EATcarol'+'.nrrd', reshape_arr_to_nrrd(eat_label_car,n),header=header)
        
        nrrd.write(str(patient)+'_'+str(256)+'_EATmanual'+'.nrrd', reshape_arr_to_nrrd(eat_manual,n),header=header)
        
        
        c+=1   

        print(patients_prob)
        
path_nrrd="X:/Ruben/TESE/New_training_Unet/Models_only_pub_dset/Mix_datasets/All_data/predict/2.5D_Unet/Dice_loss/Hospital_tif/L0_W2000_calc_augm_tif/Peri_segm/NRRD"
predict_EAT(path_nrrd)        
            