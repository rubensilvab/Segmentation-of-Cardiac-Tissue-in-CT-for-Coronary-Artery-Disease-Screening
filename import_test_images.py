# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 16:08:33 2022

@author: RubenSilva
"""

""" As imagens tens que estar em pastas relativas a cada paciente"""

import os
import glob
import numpy as np
import cv2

"""Importar imagens DICOM de teste  """


def reconvert_HU(array,window_center=50,window_width=350,L=0,W=2000,rescale=True):
    
    img_min = L - W//2 #minimum HU level, escala em HU da imagem original
    img_max =L + W//2 #maximum HU level
    
    reconvert_img=(array/65535)*(img_max - img_min) +  img_min # Reconvertido para HU
    
    
    new_img_min = window_center - window_width//2 #minimum HU level, que pretendemos truncar
    new_img_max =window_center + window_width//2 #maximum HU level, que pretendemos truncar
    
    reconvert_img[reconvert_img<new_img_min] = new_img_min #set img_min for all HU levels less than minimum HU level
    reconvert_img[reconvert_img>new_img_max] = new_img_max #set img_max for all HU levels higher than maximum HU level
    
    if rescale: 
        reconvert_img = (reconvert_img - new_img_min) / (new_img_max - new_img_min)*65535 # Para 16 bit
        
     
        
    return reconvert_img

def sort_specific(files):
  sorted_files=[]
  for file in files:
         order=file[-7:-3]
         if order[1]=='_':
             sorted_files.append(file)
  for file in files:
         order=file[-7:-3]
         if order[0]=="_":
             sorted_files.append(file)  
  for file in files:
         order=file[-8:-3]
         if order[0]=="_":
             sorted_files.append(file)  
  return sorted_files  

             
  return sorted_files      


def turn2array(x):
      for i in range(len(x)):
          x[i]=np.array(x[i])
      return x 

def import_test_images(path_dicom,path_label,Width,Length):
    my_dict = dict()
    list_patients=sorted(os.listdir(path_dicom))
    contador=0
    X=[]
    X_512=[]
    Y=[]
    patients=[]
    for patient in list_patients:
        patients.append(patient)
        dicom_tr=sorted(glob.glob(path_dicom+patient+'/*.tif'))
        label_tr=sorted(glob.glob(path_label+patient+'/*.tif'))
        dicom_tr=sort_specific(dicom_tr)
        label_tr=sort_specific(label_tr)
        Xp=[]
        Xp_512=[]
        Yp=[]
        
        
        
        for file_x,file_y in zip(dicom_tr,label_tr):
          img_x=cv2.imread(file_x,flags=cv2.IMREAD_ANYDEPTH)
          img_x=reconvert_HU(img_x,50,350)
          
          
          #img_x=cv2.imread(file_x,0)
          img_y=cv2.imread(file_y,0)
        
          img_x_resize=cv2.resize(img_x, (Width, Length))
        
          "Para HD e MAD, nao fazer resize"
          #img_y=cv2.resize(img_y, (Width, Length))
        
          Xp.append(list(img_x_resize))
          Xp_512.append(list(img_x))
        
          Yp.append(list(img_y))
        
        
       # my_dict[patient]=img_x.shape
        Xp=np.array(Xp)/65535
        Yp=np.array(Yp)/255
        Yp=(Yp>0.5).astype(np.uint8)
        
        X.append(Xp)
        X_512.append(Xp_512)
        Y.append(Yp)    
        contador+=1
        print("Falta fazer load das imagens de teste para ",len(list_patients)-contador," pacientes")
    
    X=turn2array(X)
    X_512=turn2array(X_512)
    Y=turn2array(Y)    
    
    return X,X_512,Y,patients