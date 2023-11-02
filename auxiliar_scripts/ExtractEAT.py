# -*- coding: utf-8 -*-
"""
Created on Tue May  2 11:12:25 2023

@author: RubenSilva
"""
import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt

path_image="X:/Ruben/TESE/Data/Dataset_public/Orcya/img_png/Dicom-1000_1000"
path_mask="X:/Ruben/TESE/Data/Dataset_public/Orcya/img_png/Convex_Mask"
path_save="X:/Ruben/TESE/Data/Dataset_public/Orcya/img_png/EAT_mask"
patients=os.listdir(path_image)

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
    EAT=eat_HU(pericardio,-200,-30,L=0,W=2000,rescale=True)
    return EAT

i=0
for patient in patients:
    files=sorted(glob.glob(os.path.join(path_image,patient,'*')))
    
    for file in files:
        
        image=cv2.imread(file, flags=cv2.IMREAD_ANYDEPTH)
        name_image= os.path.split(file)[-1]
        file_peri=os.path.join(path_mask,patient,name_image)
        
        
        peri_mask=cv2.imread(file_peri, flags=cv2.IMREAD_ANYDEPTH)
        peri_mask=peri_mask/255
        # print(np.unique(peri_mask))
        
        pericardio=image*peri_mask
        
        eat=eat_mask(pericardio)
        # print(np.unique(eat))
        eat[eat>0]=1*255
        
        eat_path=os.path.join(path_save,patient)
        isExist = os.path.exists(eat_path)
        
        if not isExist:                         
          # Create a new directory because it does not exist 
          os.makedirs(eat_path)
        os.chdir(eat_path)
        
        cv2.imwrite(name_image[:-3]+'png', eat)
        #print(file,file_peri)
    i+=1   
    print('Faltam',len(patients)-i,' pacientes. Atual:',patient)
        # fig=plt.figure(figsize=(10,10))
        # plt.subplot(1,3,1)
        # plt.imshow(image, cmap='gray')
        # plt.title('image')
        # plt.subplot(1,3,2)
        # plt.imshow(pericardio, cmap='gray')
        # plt.title('pericardio')
        # plt.subplot(1,3,3)
        # plt.imshow(eat, cmap='gray')
        # plt.title('EAT')
        
        