# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 17:09:19 2023

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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from openpyxl import Workbook
import nrrd
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img

cfat_all_df = pd.read_csv('X:/Ruben/TESE/Data/hospital_gaia/all_data_carolina/all_data_carolina_hospital_1.csv')
test_cfat_df=cfat_all_df.loc[cfat_all_df['Fold'].isin([0])]
from TestGenerator import *

def switch(x):
  
  X=np.transpose(x,(2,1,0))  
 
  return X
def save_img_results(test_dataframe,path_to,change_hu=False):
    patients=np.unique(test_dataframe['Patient'])
    n_patients=len(np.unique(test_dataframe['Patient']))
    
    X_p=[]
    i=0
    for patient in patients[0:2]: 
        
       """Path to go predicts"""
       
       path=os.path.join(path_to,str(patient))
       isExist = os.path.exists(path)
       #print(path_to_cpy,isExist)
       
       if not isExist:                         
           # Create a new directory because it does not exist 
           os.makedirs(path)
         
       
       input_col = 'Path_image'
       mask_col = 'Path_Mask'
       cols=[input_col,mask_col]
       
       csv_file=test_dataframe.loc[test_dataframe['Patient'].isin([(patient)])]
       batch_size=len(csv_file)
       data_generator=generators(CustomDataGenerator(csv_file,cols, batch_size=batch_size,input_shape=(Width,Length),change_hu=change_hu))
       s=0
       print("Faltam as imagens previstas para ",len(patients[0:2])-i," pacientes, atual:",str(patient))
       
       i=i+1
       while True:
           try:
               X, Y, X_original = next(data_generator)         
               X_p=X_original
               Y_p=Y
           
               print(X_p.shape)
               
               isExist = os.path.exists(os.path.join(path,'Coronal'))
               #print(path_to_cpy,isExist)
               
               if not isExist:                         
                   # Create a new directory because it does not exist 
                   os.makedirs(os.path.join(path,'Coronal'))
                   
               os.chdir(os.path.join(path,'Coronal')) 
               
               for i in range (X_p.shape[1]):
                 s=s+1 
                 fig=plt.figure(figsize=(16,6))
                 fig.suptitle('Teste:')
                 plt.subplot(1,2,1)
                 
                 img_coronal=X_p[:,i,:]
                 #print(img_sagital.shape,img_2.shape)
                 l,w=img_coronal.shape
                 new_w=int(w*0.4)
                 new_l=int(l*3)
                 
                 # Resize the image using cv2.resize() with the new width and original height
                 resized_img = cv2.resize(img_coronal, (new_l, new_w))
                 #img=array_to_img(img.reshape(w,l,1))
                 # make a contiguous copy of the array
                 #resized_img=np.transpose(resized_img,(1,0))

                 plt.imshow(resized_img,cmap='gray')
                 plt.title('Original Teste_'+str(s))
                 plt.subplot(1,2,2)
                 
                 mask_coronal=Y_p[:,i,:]
                 # make a contiguous copy of the array
                 
                 #mask=array_to_img(mask.reshape(w,l,1))
                 resized_mask = cv2.resize(mask_coronal, (new_l, new_w))
                 #resized_mask=np.transpose(resized_mask,(1,0))
                 plt.imshow(resized_mask,cmap='gray')
                 plt.title('label Test_'+str(s))
                 fig.savefig('Predicts_test_'+str(patient)+"_"+str(s)+'.jpg')
                 plt.close('all')
               
               isExist = os.path.exists(os.path.join(path,'Sagital'))
               #print(path_to_cpy,isExist)
               
               if not isExist:                         
                   # Create a new directory because it does not exist 
                   os.makedirs(os.path.join(path,'Sagital'))
                   
               os.chdir(os.path.join(path,'Sagital'))  
               for i in range (X_p.shape[1]):
                   s=s+1 
                   fig=plt.figure(figsize=(16,6))
                   fig.suptitle('Teste:')
                   plt.subplot(1,2,1)
                   img_sagital=X_p[:,:,i]
                   
                   #print(img_sagital.shape,img_2.shape)
                   l,w=img_sagital.shape
                   new_w=int(w*0.4)
                   new_l=int(l*3)
                   
                   # Resize the image using cv2.resize() with the new width and original height
                   resized_img = cv2.resize(img_sagital, (new_l, new_w))
                   
                   #resized_img=np.transpose(resized_img,(1,0))

                   plt.imshow(resized_img,cmap='gray')
                   plt.title('Original Teste_'+str(s))
                   plt.subplot(1,2,2)
                   mask_sagital=Y_p[:,:,i]
                   mask_coronal=Y_p[:,i,:]
                   # make a contiguous copy of the array
                   
                   #mask=array_to_img(mask.reshape(w,l,1))
                   resized_mask = cv2.resize(mask_sagital, (new_l, new_w))
                   #resized_mask=np.transpose(resized_mask,(1,0))
                   plt.imshow(resized_mask,cmap='gray')
                   plt.title('label Test_'+str(s))
                   fig.savefig('Predicts_test_'+str(patient)+"_"+str(s)+'.jpg')
                   plt.close('all')
                 
               
        # do something with the batches
           except StopIteration:
        # stop the loop when the generator raises StopIteration
                break    
         
Width,Length=256,256            
path_to='X:/Ruben/TESE/New_training_Unet/2.5D_model_scripts/teste_coronal'
change_hu=False
X=save_img_results(cfat_all_df,path_to,change_hu=change_hu)
# def switch(x):
  
#   X=np.transpose(x,(2,1,0))  
 
#   return X
# """See results"""


# for i in range(len(X)):
#     Xp=switch(X[i])
   
#     for i in range (Xp.shape[2]):
#      fig=plt.figure(figsize=(16,6))
#      plt.imshow(Xp[:,:,i])
     
# s=0
# for i in range (len(X[1])):
#   s=s+1 
#   fig=plt.figure(figsize=(16,6))
#   fig.suptitle('Coronal:')
#   plt.subplot(1,3,1)
#   plt.imshow(np.squeeze(X[:,:,i]),cmap='gray')
#   plt.title('Sagital'+str(s))
#   plt.subplot(1,3,2)
#   plt.imshow(np.squeeze(X[i]),cmap='gray')
#   plt.title('Axial'+str(s))
#   plt.subplot(1,3,3)
#   plt.imshow(np.squeeze(pred_test[i]),cmap='gray')
#   plt.title('Predict_'+str(s))
#   fig.savefig('Predicts_test_'+str(patients[patient])+"_"+str(s)+'.jpg')
#   plt.close('all')