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
    
    def __init__(self, csv_file,col, batch_size=12, input_shape=(256, 256), shuffle=True, seed=None,img_data_gen_args=None ):
        
        self.csv_file = csv_file
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
        
        self.count=0
        
    def __len__(self):
       
        return int(np.ceil(len(self.csv_file) / self.batch_size))

    def __getitem__(self, index):
       
        batch_data = self.csv_file.iloc[index * self.batch_size:(index + 1) * self.batch_size]
        batch_x = []
        batch_y = []
        self.count=self.count+self.batch_size
        #print('count:',self.count)
        for _, row in batch_data.iterrows():
            # Load the input image and mask from the CSV file
            input_path = row[self.col[0]]
            mask_path=row[self.col[1]]
            
            #Load image
            input_img = cv2.resize(cv2.imread(input_path, flags=cv2.IMREAD_ANYDEPTH), self.input_shape)
            #Load mask
            mask=cv2.resize(cv2.imread(mask_path, flags=cv2.IMREAD_ANYDEPTH), self.input_shape)
            
           
            
            #Check the number of slices for apply condition, if the slice chosen is the last one
            number_slices=os.path.split(input_path)[0]
            all_slices_path=self.sort_specific(sorted(glob.glob(number_slices+'/*.tif')))
            total_slices=len(all_slices_path)
            index_of_input = all_slices_path.index(input_path)
            
            paths_before,paths_after=[],[]
            #print(index_of_input)
            
            for i in range(2):
                ind=i+1
                ind_after=index_of_input+ind
                ind_before=index_of_input-ind
                
                if ind_before>=0:
                    try:
                        slice_after=all_slices_path[ind_after]
                        paths_after.append(slice_after)
                    except:
                        if (index_of_input==total_slices-1):
                            slice_after=all_slices_path[index_of_input]
                            paths_after.append(slice_after)

                        else:    
                            paths_after.append(paths_after[-1])
                        
                    try:   
                        slice_before=all_slices_path[ind_before]
                        paths_before.append(slice_before)
                        
                    except:
                        paths_before.append(paths_before[-1])
              
                else:  
                    
                    try:
                        slice_after=all_slices_path[ind_after]
                        paths_after.append(slice_after)
                    except:
                        paths_after.append(paths_after[-1])
                        
                    try:   
                        slice_before=all_slices_path[0]
                        paths_before.append(slice_before)
                        
                    except:
                        paths_before.append(paths_before[-1])
                    
                    
            aux=paths_before[0]
            paths_before[0]=paths_before[1]
            paths_before[1]=aux
           
            paths_before.append(input_path)
            
            paths_5=paths_before+paths_after
            
            #print('Path_atual:', input_path,'Paths:', paths_5)
            
            # print(path_top,',',input_path,',',path_bottom)
            #print(mask_path)
             #print(" ")
            input_img=[]
            for path in paths_5:
            # Load the top and bottom slices of the current image
                sli = cv2.resize(cv2.imread(path, flags=cv2.IMREAD_ANYDEPTH), self.input_shape)
                sli=sli.reshape(sli.shape[0],sli.shape[1])
                
                input_img.append(sli)
           
            input_img=np.array(input_img)
            input_img=np.transpose(input_img,(1,2,0))  
            #print(input_img.shape)
            w,l,ch=input_img.shape
            
            
            if self.img_data_gen_args!=None:
                seed=np.random.randint(0,2**16)
                
                input_img_gen=self.data_augmentation_img.flow(input_img.reshape(1,w,l,ch),seed=seed)
                input_img = next(input_img_gen)
                
                mask_gen=self.data_augmentation_mask.flow(mask.reshape(1,w,l,1),seed=seed)
                mask=next(mask_gen)
                
                #Binarize masks
                thresh = 0.5 # Threshold at 0.5
                mask = np.where(mask > thresh, 1, 0).astype(np.float32)
                
                
            batch_x.append(input_img.reshape(w,l,ch))
            batch_y.append(mask.reshape(w,l,1))

        return np.array(batch_x),np.array(batch_y)

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
        self.count=0
        if self.shuffle:
            self.csv_file = self.csv_file.sample(frac=1,random_state=self.seed).reset_index(drop=True)
    
    def __iter__(self):
       
        while True:
            for index in range(len(self)):
                yield self[index]
seed=2
    
img_data_gen_args = dict(#preprocessing_function=preprocessing_image,
                          rotation_range=5,
                          horizontal_flip=True,
                          vertical_flip=True,
                          width_shift_range=0.1,
                          height_shift_range=0.1,
                          zoom_range=0.2,
                          fill_mode='constant',  # Use black color to fill empty areas
                          cval=0,  
                          rescale=1/65535.
                    
                      )

# input_col = 'Path_image'
# mask_col = 'Path_Mask'
# cols=[input_col,mask_col]
# cfat_peri_df = pd.read_csv('X:/Ruben/TESE/New_training_Unet/data_only_pericardium/Cardiac_fat_new/Cardiac_fat_new_folds_5.csv')
# osic_peri_df = pd.read_csv('X:/Ruben/TESE/New_training_Unet/data_only_pericardium/OSIC_new/OSIC_new_folds_5.csv')
# batch_size=1

# datagen_image = CustomDataGenerator(osic_peri_df, cols, batch_size, input_shape=(256, 256),seed=seed,img_data_gen_args=img_data_gen_args)


# def generators(datagen_image):
#     for batch_image in datagen_image:
#         yield batch_image
        
# gene=generators(datagen_image)
# i=0
# for i in range(24):
   
#     generator= next(gene)
   
#     from matplotlib import pyplot as plt
    
#     fig=plt.figure(figsize=(10,10))
#     fig_mask=plt.figure(figsize=(10,10))
    
#     # Display the original image and the masked image side by side
    
#     import cv2
#     for i in range(batch_size): 
       
#         ax1=fig.add_subplot(batch_size,3,i*3+1)
#         ax1.imshow(generator[0][i,:,:,0], cmap='gray')
        
#         ax2= fig.add_subplot(batch_size,3,i*3+2)
#         ax2.imshow(generator[0][i,:,:,1], cmap='gray')
        
#         ax3=fig.add_subplot(batch_size,3,i*3+3)
#         ax3.imshow(generator[0][i,:,:,2], cmap='gray')
        
#         img=np.squeeze(generator[0][i,:,:,1])
#         mask=(np.squeeze(generator[1][i,:,:,0])*255).astype('uint8')
        
#         masked_img = cv2.bitwise_and(img,img, mask=mask)
#             # Blend the masked image and the original image
#         alpha = 0.5
#         beta = 1.0 - alpha
#         overlay = cv2.addWeighted(masked_img, alpha, img, beta, 0)
#         ax4=fig_mask.add_subplot(batch_size,3,i*3+1)
#         ax4.imshow(overlay,cmap="gray")
        
        
       
        