# -*- coding: utf-8 -*-
"""
Created on Sat May  6 12:45:09 2023

@author: RubenSilva
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 20:36:32 2023

@author: RubenSilva
"""

import os
import numpy as np
import pandas as pd
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import glob
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2


class CustomDataGenerator3D(Sequence):
    
    def __init__(self, csv_file,col,batch_size=1, cube_size=64,input_shape=(64, 64), shuffle=False, seed=18,img_data_gen_args=None ):
        
        self.csv_file = csv_file
        self.cube_size = cube_size
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.shuffle = shuffle
        self.col=col
        self.seed=seed
       
        
        self.patients=np.unique(csv_file['Patient'])
        
    def __len__(self):
       
        return int(np.ceil(len(self.patients) / self.batch_size))

    def __getitem__(self, index):
        
            batch_patients = self.patients[index * self.batch_size:(index + 1) * self.batch_size]
            
            batch_x = []
            batch_y = []
            
            for patient in batch_patients:
                
                cube_x=[]
                cube_y=[]
                
                all_data=self.csv_file.loc[self.csv_file['Patient'].isin([(patient)])]
                #only_peri=all_data.loc[(all_data['Label']==1)]
                #print(batch_data)
                first_indice_peri=all_data.index[0]
                last_indice_peri=all_data.index[-1]
                
                if len(all_data)<=self.cube_size: # se temos menos dados que os 64 slices
                    csv_to_use=all_data
                    
                else:
                    
                    numb_slices=len(all_data)
                    number_middle=numb_slices//2
                    index_middle=all_data.index[number_middle]
                    #print(index_middle)
                    index_sup,index_inf=index_middle-32,index_middle+31
                    csv_to_use=all_data.loc[index_sup:index_inf]
                    #print('entrou:', len(csv_to_use),'first index:',first_indice_peri)
                
                #print('Todos slices:', len(all_data))
                csv_real=csv_to_use.copy()    
                #print(csv_real)
                for _, row in csv_to_use.iterrows():
                    
                    input_path = row[self.col[0]]
                    mask_path=row[self.col[1]]
                    
                    #Load image
                    input_img = cv2.resize(cv2.imread(input_path, flags=cv2.IMREAD_ANYDEPTH), self.input_shape)
                    #Load mask
                    mask=cv2.resize(cv2.imread(mask_path, flags=cv2.IMREAD_ANYDEPTH), self.input_shape)
                    
                   
                    w,l=input_img.shape
                    
                    
                    cube_x.append(input_img.reshape(w,l,1))
                    cube_y.append(mask.reshape(w,l,1))
                    
                  
                    
                if (len(cube_x)<self.cube_size):  #Se não existem slices(colocar pretos)
                    #reiniciar csv_real
                    csv_real='empty'
                    
                    cube_x_aux=[]#variavel auxiliar para guardar os slices de cima
                    cube_y_aux=[]
                    inf_to_add=pd.DataFrame({'Patient':[patient],'Fold':[np.nan],'Path_image':[np.nan],'Path_Mask':[np.nan],'Label':[0]})
                    csv_aux=pd.DataFrame({'Patient':[],'Fold':[],'Path_image':[],'Path_Mask':[],'Label':[]})
                    
                    #print('colocar pretos')
                    sli_add=self.cube_size-len(cube_x)
                    number_to_add=sli_add//2
                    extra=sli_add%2
                    
                    slices_add_up=[np.zeros((w,l,1)) for i in range(number_to_add)]
                    
                    #adicionar inf csv
                    for i in range(number_to_add): 
                       #csv_aux=csv_aux.append(inf_to_add,ignore_index=True) 
                       csv_aux=pd.concat([csv_aux, inf_to_add], ignore_index=True)
                       
                    cube_x_aux.extend(slices_add_up)# guarda os slices de cima
                    cube_y_aux.extend(slices_add_up)
                    
                    #csv_real=csv_aux.append(csv_to_use)
                    csv_real=pd.concat([csv_aux,csv_to_use])
                    
                    
                    cube_x.extend(slices_add_up)# concatena os slices de baixo ao cubo com pericardio
                    cube_y.extend(slices_add_up)
                    
                    cube_x_aux.extend(cube_x) # adiciona o cubo todo
                    cube_y_aux.extend(cube_y)
                    csv_real=pd.concat([csv_real, csv_aux])
                    #csv_real=csv_real.append(csv_aux)
                    
                    if extra==1:
                        slice_add=[np.zeros((w,l,1))]
                        
                        cube_x_aux.extend(slice_add) # adiciona o cubo todo
                        cube_y_aux.extend(slice_add)
                        aux=pd.DataFrame({'Patient':[],'Fold':[],'Path_image':[],'Path_Mask':[],'Label':[]})
                        aux=pd.concat([aux, inf_to_add], ignore_index=True)
                        csv_real=pd.concat([csv_real, aux])

                        #aux=aux.append(inf_to_add,ignore_index=True) 
                        #csv_real=csv_real.append(aux)
                        
                        
                    #voltar a definir cube_x
                    cube_x=cube_x_aux
                    cube_y=cube_y_aux
                
                
                if True:
                    
                    cube_x_aug=[]
                    cube_y_aug=[]
                    
                    for sli in range(self.cube_size):
                        #Normalize
                        cube_x_n=(np.array(cube_x[sli])/65535).astype(np.float32)
                        mask_s=(np.array(cube_y[sli])/255)    
                 
                        #Binarize masks
                        thresh = 0.5 # Threshold at 0.5
                        cube_y_n = np.where(mask_s > thresh, 1, 0).astype(np.uint8)
                        
                        cube_x_aug.append(cube_x_n.reshape(w,l,1))
                        cube_y_aug.append(cube_y_n.reshape(w,l,1))
                        
                    cube_x=cube_x_aug
                    cube_y=cube_y_aug
                    del cube_x_aug,cube_y_aug
                    
                batch_x.append(np.array(cube_x))
                batch_y.append(np.array(cube_y))

            return np.array(batch_x),np.array(batch_y),csv_real
        
    def slices_before(self,image_path,number_to_add,indice,w,l):
        
        slices_before=[]
        for i in range(number_to_add):
            try:
                img_path = image_path.loc[indice-number_to_add+i]
                #print('images_before:',img_path)
                img = cv2.imread(img_path, flags=cv2.IMREAD_ANYDEPTH)
                resized_img = cv2.resize(img, self.input_shape).reshape(w,l,1)
                slices_before.append(resized_img)
                
            except :
                #print('criou zeros')
                slice_add=np.zeros((w,l,1))
                slices_before.append(slice_add)
        return slices_before     
                
    def slices_after(self,image_path,number_to_add,indice,w,l):
        
        slices_after=[]
        for i in range(number_to_add):
            try:
                img_path = image_path.loc[indice+i+1]
                #print('images_after:',img_path)
                img = cv2.imread(img_path, flags=cv2.IMREAD_ANYDEPTH)
                resized_img = cv2.resize(img, self.input_shape).reshape(w,l,1)
                slices_after.append(resized_img)
                
            except :
                #print('criou zeros')
                slice_add=np.zeros((w,l,1))
                slices_after.append(slice_add)
                
        return slices_after        
                
    def sort_specific(self,files):
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
  
    def on_epoch_end(self):
        if self.shuffle:
            self.csv_file = self.csv_file.sample(frac=1,random_state=self.seed).reset_index(drop=True)
    
    def __iter__(self):
       
        while True:
            for index in range(len(self)):
                yield self[index]

def generators(datagen_image):
    for batch_image in datagen_image:
        yield batch_image

def slices_add(csv,indexes):
      image_path=csv['Path_image']
      mask_path=csv['Path_Mask']
      X=[]
      Y=[]
      Y_pred=[]
      for i in indexes:
              img_path = image_path.loc[i]
              print('images_before:',img_path)
              img = cv2.imread(img_path, flags=cv2.IMREAD_ANYDEPTH)
              print(img.shape)
              w,l=img.shape
           
              X.append(img)
              Y_pred.append(np.zeros((w,l)))
             
              #Load mask
              msk_path=mask_path.loc[i]
              print('maks_before:',msk_path)
              mask=cv2.imread(msk_path, flags=cv2.IMREAD_ANYDEPTH)
              mask=mask/255.
             
              #Binarize masks
              thresh = 0.5 # Threshold at 0.5
              mask = np.where(mask > thresh, 1, 0).astype(np.uint8)
              Y.append(mask)
              
      return X,Y_pred,Y 

# def organize_data(csv_original_file,csv_25d):#,X25d,Y25d,Y):
#     # Get the first common index
#     common_index = set(csv_original_file.index).intersection(csv_25d.index)
#     first_common_index = min(common_index)
#     last_common_index=max(common_index)
    
#     first_index_original=csv_original_file.index[0]
#     n_to_add_before=first_common_index-first_index_original 
#     last_index_original=csv_original_file.index[-1]  
#     n_to_add_after=last_index_original-last_common_index 

#     indexes_before=[first_index_original+i for i in range(n_to_add_before)]
#     indexes_after=[last_common_index+i+1 for i in range(n_to_add_after)]
    
#     #indexes_before = csv_original_file.loc[first_index_original:first_common_index-1].index
#     #indexes_after= csv_original_file.loc[last_common_index+1:last_index_original].index
#     print('outro metodo:',indexes_before,'antigo metodo:',indexes_after)
    
    
#     print(first_common_index,last_common_index)
#     #X_before,Y_pred_before,Y_before=slices_add(csv_original_file,indexes_before)
#     #X_after,Y_pred_after,Y_after=slices_add(csv_original_file,indexes_after)
    
#     #X,Y_pred,Y=np.concatenate([X_before,X25d,X_after], axis=0),np.concatenate([Y_pred_before,Y25d,Y_pred_after], axis=0),np.concatenate([Y_before,Y,Y_after], axis=0)
    

# input_col = 'Path_image'
# mask_col = 'Path_Mask'
# cols=[input_col,mask_col]

# "Import CSV with dicom and masks informations"

# cfat_all_df = pd.read_csv('X:/Ruben/TESE/New_training_Unet/all_data/Cardiac_fat_new/Cardiac_fat_new_folds_5.csv')
# osic_3d_df = pd.read_csv('X:/Ruben/TESE/New_training_Unet/all_data/OSIC_new/OSIC_3D.csv')
# osic_3d_df=osic_3d_df.loc[osic_3d_df['Patient'].isin(['id00009637202177434476278'])]


# train_cfat_3d=cfat_all_df.loc[cfat_all_df['Fold'].isin([0])]
# train_cfat_3d=train_cfat_3d.loc[train_cfat_3d['Patient'].isin(['ACel'])]

# val_osic_3d=osic_3d_df.loc[osic_3d_df['Fold'].isin([3])]
# test_hospit_df= pd.read_csv('X:/Ruben/TESE/Data/hospital_gaia/all_data_carolina/all_data_carolina_hospital_1.csv')
# test_hospit_df=test_hospit_df.loc[test_hospit_df['Patient'].isin([583912])]

# batch_size=1


    
    
# Width=64
# train_combine_generator=generators(CustomDataGenerator3D(test_hospit_df, cols, batch_size, input_shape=(Width, Width)))
# val_combine_generator=generators(CustomDataGenerator3D(val_osic_3d, cols, batch_size, input_shape=(Width, Width)))

# x,y,slices= next(train_combine_generator)
# csv_3d=slices.dropna()
# organize_data(test_hospit_df,csv_3d)

#datagen_image = CustomDataGenerator(cfat_peri_df, cols, batch_size, input_shape=(256, 256),img_data_gen_args=(img_data_gen_args))


