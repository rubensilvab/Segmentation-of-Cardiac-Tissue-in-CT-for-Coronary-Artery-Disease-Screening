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


class CustomDataGenerator(Sequence):
    
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
            batch_x_original=[]
            batch_y = []
            
            for patient in batch_patients:
                
                cube_x=[]
                cube_x_original=[]
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
                
                for _, row in csv_to_use.iterrows():
                    
                    input_path = row[self.col[0]]
                    mask_path=row[self.col[1]]
                    
                    #Load image
                    input_img = cv2.resize(cv2.imread(input_path, flags=cv2.IMREAD_ANYDEPTH), self.input_shape)
                    #original image
                    original_img = cv2.imread(input_path, flags=cv2.IMREAD_ANYDEPTH)

                    #Load mask
                    mask=cv2.imread(mask_path, flags=cv2.IMREAD_ANYDEPTH)
                    
                   
                    w,l=input_img.shape
                    wo,lo=mask.shape
                    
                    cube_x.append(input_img.reshape(w,l,1))
                    cube_x_original.append(original_img.reshape(wo,lo,1))
                    cube_y.append(mask.reshape(wo,lo,1))
                
                    #print(input_path,input_img.shape )
               
                    
                if (len(cube_x)<self.cube_size):  #Se não existem slices(colocar pretos)
                    
                    cube_x_aux=[]#variavel auxiliar para guardar os slices de cima
                    cube_x_orig_aux=[]
                    cube_y_aux=[]
                    
                    inf_to_add=pd.DataFrame({'Patient':[patient],'Fold':[np.nan],'Path_image':[np.nan],'Path_Mask':[np.nan],'Label':[0]})
                    csv_aux=pd.DataFrame({'Patient':[],'Fold':[],'Path_image':[],'Path_Mask':[],'Label':[]})
                    
               
                    #print('colocar pretos')
                    sli_add=self.cube_size-len(cube_x)
                    number_to_add=sli_add//2
                    extra=sli_add%2
                    
                    #adicionar inf csv
                    for i in range(number_to_add): 
                       #csv_aux=csv_aux.append(inf_to_add,ignore_index=True) 
                       csv_aux=pd.concat([csv_aux, inf_to_add], ignore_index=True)
                    
                    
                    slices_add_up=[np.zeros((w,l,1)) for i in range(number_to_add)]
                    slices_add_up_ori=[np.zeros((wo,lo,1)) for i in range(number_to_add)]
                    
                    cube_x_aux.extend(slices_add_up)# guarda os slices de cima
                    cube_y_aux.extend(slices_add_up_ori)
                    cube_x_orig_aux.extend(slices_add_up_ori)
                    
                    csv_real=pd.concat([csv_aux,csv_to_use])
                    
                    cube_x.extend(slices_add_up)# concatena os slices de baixo ao cubo com pericardio
                    cube_y.extend(slices_add_up_ori)
                    cube_x_original.extend(slices_add_up_ori)
                    
                    cube_x_aux.extend(cube_x) # adiciona o cubo todo
                    cube_y_aux.extend(cube_y)
                    cube_x_orig_aux.extend(cube_x_original)
                    csv_real=pd.concat([csv_real, csv_aux])
                    
                    if extra==1:
                        slice_add=[np.zeros((w,l,1))]
                        slice_add_ori=[np.zeros((wo,lo,1))]
                        
                        cube_x_aux.extend(slice_add) # adiciona o cubo todo
                        cube_y_aux.extend(slice_add_ori)
                        cube_x_orig_aux.extend(slice_add_ori)
                        
                        aux=pd.DataFrame({'Patient':[],'Fold':[],'Path_image':[],'Path_Mask':[],'Label':[]})
                        aux=pd.concat([aux, inf_to_add], ignore_index=True)
                        csv_real=pd.concat([csv_real, aux])
                        
                    #voltar a definir cube_x
                    cube_x=cube_x_aux
                    cube_y=cube_y_aux
                    cube_x_original=cube_x_orig_aux
                
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
                        cube_y_aug.append(cube_y_n.reshape(wo,lo,1))
                        
                    cube_x=cube_x_aug
                    cube_y=cube_y_aug
                    del cube_x_aug,cube_y_aug
                    
                batch_x.append(np.array(cube_x))
                batch_x_original.append(np.array(cube_x_original))
                batch_y.append(np.array(cube_y))

            return np.array(batch_x),np.array(batch_x_original),np.array(batch_y),csv_real
        
                
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


# input_col = 'Path_image'
# mask_col = 'Path_Mask'
# cols=[input_col,mask_col]

# "Import CSV with dicom and masks informations"

# cfat_all_df = pd.read_csv('X:/Ruben/TESE/New_training_Unet/all_data/Cardiac_fat_new/Cardiac_fat_new_folds_5.csv')
# osic_3d_df = pd.read_csv('X:/Ruben/TESE/New_training_Unet/all_data/OSIC_new/OSIC_new_sorted_5.csv')

# train_osic_3d=osic_3d_df.loc[osic_3d_df['Fold'].isin([4])]
# val_osic_3d=osic_3d_df.loc[osic_3d_df['Fold'].isin([3])]

# batch_size=1

# def generators(datagen_image):
#     for batch_image in datagen_image:
#         yield batch_image
        

# Width=64
# train_combine_generator=generators(CustomDataGenerator(cfat_all_df, cols, batch_size, input_shape=(Width, Width)))
# val_combine_generator=generators(CustomDataGenerator(val_osic_3d, cols, batch_size, input_shape=(Width, Width)))

# x,y= next(val_combine_generator)


# datagen_image = CustomDataGenerator(cfat_peri_df, cols, batch_size, input_shape=(256, 256),img_data_gen_args=(img_data_gen_args))


# "Teste treinar modelo 3d"

# """Definir modelo"""
# import time
# # get the current time in seconds since the epoch
# seconds = time.time()
# # convert the time in seconds since the epoch to a readable format
# local_times = time.ctime(seconds)

# local_time = '_'.join(local_times.split())
# local_time = '_'.join(local_time.split(':'))


# from teste_architecture import *


# """Model creation and Summary"""
# model=build_unet((Width,Width,Width,1),1)
# #model=threeD_Unet()
# print(model.summary())

# """Onde o modelo vai ser guardado"""

    
# model_path="X:/Ruben/TESE/New_training_Unet/Models_only_pub_dset/Mix_datasets/"+str('teste')+"/models/3D_Unet/Dice_loss/teste"
# model_path=os.path.join(model_path,str(local_time))

# if not os.path.exists(model_path):                         
#     # Create a new directory because it does not exist 
#     os.makedirs(model_path)
# os.chdir(model_path) 


# NUM_EPOCHS=4000
# lr=0.0001

# from keras.callbacks import ReduceLROnPlateau

# reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10,
#                           verbose=1, min_lr=1e-5)


# def schedule(epoch, lr):
 
#       return lr
 
    
# from keras.callbacks import CSVLogger
# import segmentation_models as sm
# from keras import callbacks
# from keras.callbacks  import Callback 

# "Loss function and optimizer"

# Dice_loss=sm.losses.DiceLoss()
# opt = keras.optimizers.Adam(learning_rate= lr)


# loss_str="Dice_loss_"


# "Model compilation"

# model.compile(optimizer = opt, loss =Dice_loss,  metrics = ['accuracy'])

# """Name of the model and Callbacks"""

# class CustomCSVLogger(CSVLogger):
#     def on_epoch_end(self, epoch, logs=None):
#         logs = logs or {}
#         logs['learning_rate'] = self.model.optimizer.lr.numpy()
#         super().on_epoch_end(epoch, logs)

# model_name="Loss_"+loss_str+"_epochs_"+str(NUM_EPOCHS)+"_batch_size_"+str(batch_size)+"_wl"+str(Width)+"Lr_decreasing_"+str(lr)+"_time_"+local_time
# csv_logger = CustomCSVLogger(model_name+".csv", append=True)

# my_callbacks = [
#     callbacks.EarlyStopping(patience=20,monitor='val_loss'),#callbacks.LearningRateScheduler(schedule, verbose=1),
#     callbacks.ModelCheckpoint(filepath=model_name+'.h5',save_best_only=True, verbose=1),csv_logger,reduce_lr]

# "Definition steps per epoch"



# num_train_imgs=len(np.unique(train_peri_cfat['Patient']))  # CARDIAC FAT 

# #num_train_imgs_1=len(train_osic) # OSIC

# total_num_train_imgs=num_train_imgs #+num_train_imgs_1
# steps_per_epoch =total_num_train_imgs//batch_size#total_num_train_imgs //(batch_size*16)

# "Definition train and val generator"
# train_generator=train_combine_generator
# val_generator=val_combine_generator

# "Model training"    
# history = model.fit(train_generator,validation_data=val_generator,steps_per_epoch=steps_per_epoch, 
#                     validation_steps=steps_per_epoch,epochs=NUM_EPOCHS,callbacks=[my_callbacks])
    
# patients=np.unique(osic_peri_df['Patient'])
# i=0
# dic={}
# for patient in patients:
#     i=i+1
#     p=osic_peri_df.loc[(osic_peri_df['Patient']==patient)]
#     print(len(p),'patient:',patient, i)
#     dic[patient]=len(p)

# np.max(dic)