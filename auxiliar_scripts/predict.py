# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 16:39:35 2022

@author: RubenSilva
"""

import os
import glob
import numpy as np

import cv2

Width,Length=256,256

"""Importar imagens DICOM de teste  """

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

path_dicom="X:/Ruben/TESE/New_training_Unet/output/test/images/"
path_label="X:/Ruben/TESE/New_training_Unet/output/test/mask/"
list_patients=sorted(os.listdir(path_dicom))

X=[]
Y=[]
patients=[]
for patient in list_patients:
    patients.append(patient)
    dicom_tr=sorted(glob.glob(path_dicom+patient+'/*.png'))
    label_tr=sorted(glob.glob(path_label+patient+'/*.png'))
    dicom_tr=sort_specific(dicom_tr)
    label_tr=sort_specific(label_tr)
    Xp=[]
    Yp=[]
    
    for file_x,file_y in zip(dicom_tr,label_tr):
      img_x=cv2.imread(file_x,0)
      img_y=cv2.imread(file_y,0)
      img_x=cv2.resize(img_x, (Width, Length))
      img_y=cv2.resize(img_y, (Width, Length))
      Xp.append(list(img_x))
      Yp.append(list(img_y))
      print(file_x,file_y)
      
    Xp=np.array(Xp)/255
    Yp=np.array(Yp)/255
    Yp=(Yp>0.5).astype(np.uint8)
    X.append(Xp)
    Y.append(Yp)    

def turn2array(x):
     for i in range(len(x)):
         x[i]=np.array(x[i])
     return x   
    
X=turn2array(X)
Y=turn2array(Y)    

# """Import np array- hospital data""" 

# from numpy import load
# array_path="F:/Ruben/TESE/Data/hospital_gaia/imgs_png/array"
# os.chdir(array_path) 

# X_dicom=load('Dicom_gaia_512_nova.npy',allow_pickle=True)
# Y_masks=load('masks_gaia_512_nova.npy',allow_pickle=True) 
  

from numpy import load
path_model="X:/Ruben/TESE/New_training_Unet/Models/Results/dice+focal"
os.chdir(path_model)

from keras.models import Model
from keras.models import load_model
import segmentation_models as sm
from tensorflow import keras
model = load_model('Loss_Dice+_focal_loss_pesos_0.5_0.5_epochs_200_batch_size_10_wl256Lr_0.0001.h5',compile=False)



"""Função para fazer predict de um conjunto de imagens"""

def predict(model,X):
  prediction=[]
  for i in range(len(X)):
    pred=model.predict(X[i].reshape(1,Width,Length,1))
    pred=(pred>0.5).astype(np.uint8)
    prediction.append(pred)
  return np.array(prediction)

def single_dice_coef(y_true, y_pred_bin):
    # shape of y_true and y_pred_bin: (height, width)
    intersection = np.sum(y_true * y_pred_bin)
    if (np.sum(y_true)==0) and (np.sum(y_pred_bin)==0):
        return 1
    return round((2*intersection) / (np.sum(y_true) + np.sum(y_pred_bin)),3)

def mean_dice(y_true,y_pred): #calcular  dice das fatias todas como volume 3d
  dices=[]
  for i in range(len(y_true)):
    dice=single_dice_coef(y_true[i],y_pred[i])
    dices.append(dice)
  return round(np.mean(dices),3),round(np.std(dices),3)

# def delete_slice_without_peri(X,Y):
#     X_new=[]
#     Y_new=[]
#     for p in range(len(Y)):
#         slices=[]
#         for sli in range(Y[p].shape[0]):
            
#             if np.sum(Y[p][sli])==0:
#                 slices.append(sli)
                
#         Y_p=np.delete(Y[p],slices,axis=0)
#         X_p=np.delete(X[p],slices,axis=0)
#         X_new.append(X_p)
#         Y_new.append(Y_p)
        
#     return X_new,Y_new

#X,Y=delete_slice_without_peri(X,Y)    

"""Fazer predict com base no model"""

#pred_test=predict(model,X[patient])


import matplotlib.pyplot as plt



for patient in range(len(X)): 
    
   """Path to go predicts"""
   path_to="X:/Ruben/TESE/New_training_Unet/Models/Results/dice+focal/predicts"
   
   path=os.path.join(path_to,str(patients[patient]))
   isExist = os.path.exists(path)
   #print(path_to_cpy,isExist)
   
   if not isExist:                         
       # Create a new directory because it does not exist 
       os.makedirs(path)
   os.chdir(path)   
   pred_test=predict(model,X[patient])
   for i in range (X[patient].shape[0]):
     
    fig=plt.figure(figsize=(16,6))
    fig.suptitle('Dice:'+str(round(single_dice_coef(Y[patient][i], np.squeeze(pred_test[i])),3)))
    plt.subplot(1,3,1)
    plt.imshow(np.squeeze(X[patient][i]),cmap='gray')
    plt.title('Original Teste_'+str(i))
    plt.subplot(1,3,2)
    plt.imshow(np.squeeze(Y[patient][i]),cmap='gray')
    plt.title('label Test_'+str(i))
    plt.subplot(1,3,3)
    plt.imshow(np.squeeze(pred_test[i]),cmap='gray')
    plt.title('Predict_'+str(i))
    fig.savefig('Predicts_test_'+str(patients[patient])+"_"+str(i)+'.jpg')

mean_dices,std_dices=mean_dice(Y[patient],np.squeeze(pred_test))
print('Mean dices nos dados de teste: ', mean_dices,'+-',std_dices)   

# import xlsxwriter module
import xlsxwriter
 
array_path="X:/Ruben/TESE/New_training_Unet/Models/Results/dice+focal"
os.chdir(array_path)

workbook = xlsxwriter.Workbook('Loss_Dice+_focal_loss_pesos_0.5_0.5_epochs_200_batch_size_10_wl256Lr_0.0001.xlsx')
 
# By default worksheet names in the spreadsheet will be
# Sheet1, Sheet2 etc., but we can also specify a name.
worksheet = workbook.add_worksheet("Results 256")
 
# Some data we want to write to the worksheet.
# patients = ['248949',
#  '262601',
#  '354985',
#  '441689',
#  '499545',
#  '584613',
#  '699615',
#  '711314',
#  '719172',
#  '72093',
#  '730705',
#  '738655',
#  '741814',
#  '744233',
#  '746168',
#  '74808',
#  '749729',
#  '755870',
#  '759755',
#  '771992']
 
# i have already

# Start from the first cell. Rows and
# columns are zero indexed.
row = 0
col = 0
worksheet.write(row, col,"patient")
worksheet.write(row, col + 1,"mean")
worksheet.write(row, col + 2,"std")
 
# Iterate over the data and write it out row by row.
for patient, i in zip(patients,range(len(patients))):
    row += 1
    print(patient,i)
    pred_test=predict(model,X[i])
    #mean_dices,std_dices=mean_dice(Y[i],np.squeeze(pred_test))
    dice=single_dice_coef(Y[i],np.squeeze(pred_test))
    worksheet.write(row, col, patient)
    worksheet.write(row, col + 1,dice)
    #worksheet.write(row, col + 2,std_dices)
    
 
workbook.close()

"""Convert to NRRD to view in 3Dslicer"""

# def reshape_nrrd(nrrd):
#   nrrd=np.transpose(nrrd,(2,1,0))  
#   return nrrd  

# import nrrd

# for i in range(len(X)):
#     pred_test=predict(model,X[i])
#     pred_test=np.squeeze(pred_test)
    
#     path="F:/Ruben/TESE/Training_Unet/nrrd_results_hospital/Dicom"
#     os.chdir(path) 
    
#     nrrd.write(str(patients[i])+'_256.nrrd', reshape_nrrd(X[i]))
    
#     path="F:/Ruben/TESE/Training_Unet/nrrd_results_hospital/Predicts"
#     os.chdir(path) 
    
#     nrrd.write(str(patients[i])+'_predict_256.nrrd', reshape_nrrd(pred_test))

#     path="F:/Ruben/TESE/Training_Unet/nrrd_results_hospital/Masks"
#     os.chdir(path) 
    
#     nrrd.write(str(patients[i])+'_masks_256.nrrd', reshape_nrrd(Y[i]))
