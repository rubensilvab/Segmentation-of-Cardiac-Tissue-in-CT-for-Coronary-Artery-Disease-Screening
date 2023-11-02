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
       
        self.img_data_gen_args = img_data_gen_args.copy()
        self.mask_data_gen_args = img_data_gen_args.copy()
        self.mask_data_gen_args['rescale'] = 1/255.
        
        #print(self.mask_data_gen_args)
        if "preprocessing_function" in self.mask_data_gen_args:
            self.mask_data_gen_args.pop("preprocessing_function")
            
        self.on_epoch_end()
        self.data_augmentation_img = ImageDataGenerator(**self.img_data_gen_args)
        self.data_augmentation_mask = ImageDataGenerator(**self.mask_data_gen_args)
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
                only_peri=all_data.loc[(all_data['Label']==1)]
                #print(batch_data)
                first_indice_peri=only_peri.index[0]
                last_indice_peri=only_peri.index[-1]
                
                if len(all_data)<=self.cube_size: # se temos menos dados que os 64 slices
                    csv_to_use=all_data
                else:
                    csv_to_use=only_peri
                
                #print('Todos slices:', len(all_data),'Peri:',len(only_peri))
                    
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
                
                    #print(input_path,input_img.shape )
                    
                #Se o cubo não tem 64 slices e temos mais slices no csv all_data 
                if (len(cube_x)<self.cube_size)&(len(all_data)>self.cube_size):
                    #print("cubo tem menos de 64 slices")
                    cube_x_aux=[]#variavel auxiliar para guardar os slices de cima
                    cube_y_aux=[]
                    
                    sli_add=self.cube_size-len(cube_x)
                    number_to_add=sli_add//2
                    extra=sli_add%2
                    
                    image_path=all_data[self.col[0]] # Path image do csv contendo todos os slices
                    path_mask=all_data[self.col[1]]
                    
                    #print(image_path)
                    #print('slices pericardio:',len(cube_x),'all slices',len(all_data),'slices cada lado:',number_to_add)
                    slices_before=self.slices_before(image_path, number_to_add, first_indice_peri, w, l)
                    masks_before=self.slices_before(path_mask, number_to_add, first_indice_peri, w, l)
                    
                    #print('primeiro indice:',first_indice_peri,'primeiro slice de cima:',(image_path.loc[first_indice_peri-number_to_add]))
                    # slices_after=[cv2.resize(cv2.imread(image_path.loc[last_indice_peri+i+1], flags=cv2.IMREAD_ANYDEPTH), self.input_shape).reshape(w,l,1) for i in range(number_to_add)]
                    # masks_after=[cv2.resize(cv2.imread(path_mask.loc[last_indice_peri+i+1], flags=cv2.IMREAD_ANYDEPTH), self.input_shape).reshape(w,l,1) for i in range(number_to_add)]
                    
                    slices_after=self.slices_after(image_path, number_to_add, last_indice_peri, w, l)
                    masks_after=self.slices_after(path_mask, number_to_add, last_indice_peri, w, l)

                    
                    #print('primeiro slice de baixo:',(image_path.loc[last_indice_peri+1]))

                    cube_x_aux.extend(slices_before)# guarda os slices de cima
                    cube_y_aux.extend(masks_before)
                    
                    cube_x.extend(slices_after)# concatena os slices de baixo ao cubo com pericardio
                    cube_y.extend(masks_after)
                    
                    cube_x_aux.extend(cube_x) # adiciona o cubo todo
                    cube_y_aux.extend(cube_y)
                    
                    if extra==1:
                        try:
                            last_slice=[cv2.resize(cv2.imread(image_path.loc[last_indice_peri+number_to_add+1], flags=cv2.IMREAD_ANYDEPTH), self.input_shape).reshape(w,l,1)]
                            last_mask=[cv2.resize(cv2.imread(path_mask.loc[last_indice_peri+number_to_add+1], flags=cv2.IMREAD_ANYDEPTH), self.input_shape).reshape(w,l,1)]
                        
                        except:
                            last_slice=[np.zeros((w,l,1))]
                            last_mask=[np.zeros((w,l,1))]
                        
                        
                        cube_x_aux.extend(last_slice) # adiciona o cubo todo
                        cube_y_aux.extend(last_mask)
                        #print('Entrou extra:')
                    #voltar a definir cube_x
                    cube_x=cube_x_aux
                    cube_y=cube_y_aux
                    
                else:  #Se não existem slices(colocar pretos)
                    
                    cube_x_aux=[]#variavel auxiliar para guardar os slices de cima
                    cube_y_aux=[]
                    #print('colocar pretos')
                    sli_add=self.cube_size-len(cube_x)
                    number_to_add=sli_add//2
                    extra=sli_add%2
                    
                    slices_add_up=[np.zeros((w,l,1)) for i in range(number_to_add)]
                    
                    cube_x_aux.extend(slices_add_up)# guarda os slices de cima
                    cube_y_aux.extend(slices_add_up)
                    
                    cube_x.extend(slices_add_up)# concatena os slices de baixo ao cubo com pericardio
                    cube_y.extend(slices_add_up)
                    
                    cube_x_aux.extend(cube_x) # adiciona o cubo todo
                    cube_y_aux.extend(cube_y)
                    
                    if extra==1:
                        slice_add=[np.zeros((w,l,1))]
                     
                        cube_x_aux.extend(slice_add) # adiciona o cubo todo
                        cube_y_aux.extend(slice_add)
                        
                    #voltar a definir cube_x
                    cube_x=cube_x_aux
                    cube_y=cube_y_aux
                
                
                if self.img_data_gen_args!=None:
                    seed=np.random.randint(0,2**16)
                    cube_x_aug=[]
                    cube_y_aug=[]
                    for sli in range(self.cube_size):
                        
                        input_img_gen=self.data_augmentation_img.flow(np.array(cube_x[sli]).reshape(1,w,l,1),seed=seed)
                        cube_x_n = next(input_img_gen)
                        
                        mask_gen=self.data_augmentation_mask.flow(np.array(cube_y[sli]).reshape(1,w,l,1),seed=seed)
                        cube_y_n=next(mask_gen)
                        
                        #Binarize masks
                        thresh = 0.5 # Threshold at 0.5
                        cube_y_n = np.where(cube_y_n > thresh, 1, 0).astype(np.float32)
                        
                        cube_x_aug.append(cube_x_n.reshape(w,l,1))
                        cube_y_aug.append(cube_y_n.reshape(w,l,1))
                        
                    cube_x=cube_x_aug
                    cube_y=cube_y_aug
                    del cube_x_aug,cube_y_aug
                    
                batch_x.append(np.array(cube_x))
                batch_y.append(np.array(cube_y))

            return np.array(batch_x),np.array(batch_y)
        
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
# seed=2
    
# img_data_gen_args = dict(#preprocessing_function=preprocessing_image,
#                           # rotation_range=5,
#                           # horizontal_flip=True,
#                           # vertical_flip=True,
#                           # width_shift_range=0.1,
#                           # height_shift_range=0.1,
#                           # zoom_range=0.2,
#                           # fill_mode='constant',  # Use black color to fill empty areas
#                           # cval=0,  
#                           rescale=1/65535.
                    
#                       )

# input_col = 'Path_image'
# mask_col = 'Path_Mask'
# cols=[input_col,mask_col]

# "Import CSV with dicom and masks informations"

# cfat_peri_df = pd.read_csv('X:/Ruben/TESE/New_training_Unet/data_only_pericardium/Cardiac_fat_new/Cardiac_fat_new_folds_5.csv')
# cfat_all_df = pd.read_csv('X:/Ruben/TESE/New_training_Unet/all_data/Cardiac_fat_new/Cardiac_fat_new_folds_5.csv')
# osic_peri_df = pd.read_csv('X:/Ruben/TESE/New_training_Unet/data_only_pericardium/OSIC_new/OSIC_new_folds_5.csv')
# osic_all_df = pd.read_csv('X:/Ruben/TESE/New_training_Unet/all_data/OSIC_new/OSIC_new_folds_5.csv')

# osic_3d_df = pd.read_csv('X:/Ruben/TESE/New_training_Unet/all_data/OSIC_new/OSIC_3D.csv')


# train_osic_3d=osic_3d_df.loc[osic_3d_df['Fold'].isin([0,1,2])]
# val_osic_3d=osic_3d_df.loc[osic_3d_df['Fold'].isin([3])]

 
# batch_size=1

# def generators(datagen_image):
#     for batch_image in datagen_image:
#         yield batch_image
        

# def threeD_gen(csv_file, cols, batch_size, input_shape=(256, 256),img_data_gen_args=(img_data_gen_args)):
#   patients=np.unique(csv_file['Patient'])
#   while True:
#       for patient in patients:
#          #print(patient)
#          csv_patient=csv_file.loc[csv_file['Patient'].isin([(patient)])] 
#          datagen_image = CustomDataGenerator(csv_patient, cols, batch_size, input_shape=input_shape,img_data_gen_args=(img_data_gen_args))
#          gen=generators(datagen_image)
        
#          for images in gen:
#                 yield images[0], images[1]

# Width=64
# train_combine_generator=generators(CustomDataGenerator(cfat_all_df, cols, batch_size, input_shape=(Width, Width),img_data_gen_args=(img_data_gen_args)))
# val_combine_generator=generators(CustomDataGenerator(val_osic_3d, cols, batch_size, input_shape=(Width, Width),img_data_gen_args=(img_data_gen_args)))

# generator= next(train_combine_generator)


# datagen_image = CustomDataGenerator(cfat_peri_df, cols, batch_size, input_shape=(256, 256),img_data_gen_args=(img_data_gen_args))

# def generator(datagen_image,datagen_mask):
#     for batch_image,batch_mask in zip(datagen_image,datagen_mask):
#         yield batch_image,batch_mask

# gene=generators(datagen_image)
# generator= next(gene)
# i=0
# for p in range(6):
   
#     generator= next(train_combine_generator)
   
#     from matplotlib import pyplot as plt
    
    
#     # Display the original image and the masked image side by side
#     vol=6
#     batch_size=2
#     import cv2
#     for i in range(batch_size): 
       
           
#         fig=plt.figure(figsize=(10,10))
#         fig_mask=plt.figure(figsize=(10,10)) 
           
#         ax1=fig.add_subplot(batch_size,vol,1)
#         ax1.imshow(generator[0][i,0,:,:,0], cmap='gray')
        
#         ax2= fig.add_subplot(batch_size,vol,2)
#         ax2.imshow(generator[0][i,1,:,:,0], cmap='gray')
        
#         ax3=fig.add_subplot(batch_size,vol,3)
#         ax3.imshow(generator[0][i,2,:,:,0], cmap='gray')
        
#         ax4=fig.add_subplot(batch_size,vol,4)
#         ax4.imshow(generator[0][i,3,:,:,0], cmap='gray')
        
#         ax5= fig.add_subplot(batch_size,vol,5)
#         ax5.imshow(generator[0][i,4,:,:,0], cmap='gray')
        
#         ax6=fig.add_subplot(batch_size,vol,6)
#         ax6.imshow(generator[0][i,5,:,:,0], cmap='gray')
        
        
#         ax7=fig.add_subplot(batch_size,vol,7)
#         ax7.imshow(generator[0][i,6,:,:,0], cmap='gray')
        
#         ax8= fig.add_subplot(batch_size,vol,8)
#         ax8.imshow(generator[0][i,7,:,:,0], cmap='gray')
        
#         ax9=fig.add_subplot(batch_size,vol,9)
#         ax9.imshow(generator[0][i,8,:,:,0], cmap='gray')
        
#         ax10=fig.add_subplot(batch_size,vol,10)
#         ax10.imshow(generator[0][i,9,:,:,0], cmap='gray')
        
#         ax11= fig.add_subplot(batch_size,vol,11)
#         ax11.imshow(generator[0][i,10,:,:,0], cmap='gray')
        
#         ax12=fig.add_subplot(batch_size,vol,12)
#         ax12.imshow(generator[0][i,11,:,:,0], cmap='gray')
        
        
        
#         img=np.squeeze(generator[0][i,2,:,:,0])
#         mask=(np.squeeze(generator[1][i,2,:,:,0])*255).astype('uint8')
        
#         masked_img = cv2.bitwise_and(img,img, mask=mask)
#             # Blend the masked image and the original image
#         alpha = 0.5
#         beta = 1.0 - alpha
#         overlay = cv2.addWeighted(masked_img, alpha, img, beta, 0)
#         ax7=fig_mask.add_subplot(batch_size,vol,3)
#         ax7.imshow(overlay,cmap="gray")
#         print(p,i)
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