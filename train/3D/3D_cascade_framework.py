# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 18:53:25 2023

@author: RubenSilva
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 17:01:50 2023

@author: RubenSilva
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 12:29:26 2022

@author: RubenSilva
"""

import os
#os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.6/bin")

import segmentation_models as sm
from tensorflow import keras
import numpy as np 
from matplotlib import pyplot as plt

import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
import glob
import cv2
#from keras import backend as keras
# import the time module
import time
from keras.callbacks import CSVLogger
#Keras
from keras import backend as K
from tensorflow.python.ops import math_ops
from keras.models import load_model
from keras import callbacks
from keras.callbacks  import Callback 
import pandas as pd



"""Primeiro objetivo: formar um batch que contenha 5 imagens do cardiac fat e 5 images do OSIC"""
"""Nota: como o cardiac fat o tem 20 patients pensei fazer data augmentation"""

from tensorflow.keras.preprocessing.image import ImageDataGenerator

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

def preprocessing_image(img):
    pre_process_img=reconvert_HU(img)
    
    
    return pre_process_img
    
    
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

        
def generators(datagen_image):
    for batch_image in datagen_image:
        yield batch_image

from customdatagenerator import *        

def my_generator(dataframe,img_data_gen_args,batch_size,seed,Width,Length,calc='y'):
    input_col = 'Path_image'
    mask_col = 'Path_Mask'
    cols=[input_col,mask_col]
    
    image_generator=CustomDataGenerator(dataframe, cols, batch_size, input_shape=(Width, Width),seed=seed,img_data_gen_args=(img_data_gen_args))
    
    #image_generator=CustomDataGenerator(dataframe, cols, batch_size, input_shape=(Width,Length),seed=seed,img_data_gen_args=img_data_gen_args)
    image_generator=generators(image_generator)
    
    if (calc=='n'):
      generator=image_generator
    else:
      generator=image_and_mask_generator(image_generator)
    
    return generator 

"import function calficification"

from gaussian_calc import *

def image_and_mask_generator(image_generator):
    while True:
        # Get a batch of augmented images and masks
        gen=next(image_generator)
        batch_images = gen[0]
        batch_masks = gen[1]
        #print(batch_images.shape,batch_masks.shape)
        
        # Apply specific augmentation to each image based on its corresponding mask
        
        augmented_images = []
        augmented_masks = []
                
        random_n=random.randint(1,3) #"Colocar probabilidade em 33%"
        
        batch_c=batch_images
        if ( random_n==1 ):
          # Apply the specific augmentation to the image   
          batch_c=noisy_gaussian(batch_images,batch_masks)
          
        
        # Convert the list of augmented images and masks to arrays and yield them
        augmented_images = np.array(batch_c)
        augmented_masks = np.array(batch_masks)
        #print(augmented_images.shape)
        yield (augmented_images, augmented_masks)
            


def combine_gen(*gens):
    while True:
        for g in gens:
            yield next(g)
                                                                 
"Só para teste"

Width=64
Length=Width
seed=10
batch_size= 1

"Definition of train, val and test set"

number_folds=5
n_train=int(number_folds*0.60)
n_val=int(number_folds*0.20)
n_test=int(number_folds*0.20)

folds=[]
[folds.append(p) for p in range(number_folds)]

from itertools import cycle


"Import CSV with dicom and masks informations"

#cfat_peri_df = pd.read_csv('X:/Ruben/TESE/New_training_Unet/data_only_pericardium/Cardiac_fat_new/Cardiac_fat_new_folds_5.csv')
cfat_all_df = pd.read_csv('X:/Ruben/TESE/New_training_Unet/all_data/Cardiac_fat_new/Cardiac_fat_new_sorted_5.csv')
#osic_peri_df = pd.read_csv('X:/Ruben/TESE/New_training_Unet/data_only_pericardium/OSIC_new/OSIC_new_folds_5.csv')
osic_all_df = pd.read_csv('X:/Ruben/TESE/New_training_Unet/all_data/OSIC_new/OSIC_3D.csv')


# get the current time in seconds since the epoch
seconds = time.time()
# convert the time in seconds since the epoch to a readable format
local_time_cross = time.ctime(seconds)
local_time_cross = '_'.join(local_time_cross.split(':'))

import mlnotify 

"Pretendemos Cross-Validation ?"

cross_validation="n"
only_peri=False

for index, *ans in zip(range(number_folds), *[cycle(folds)] * (number_folds+1)):
   
    "Definition of the folds to train, valid and test the data" 
   
    folds_train=ans[0:n_train]
    folds_val=ans[n_train:n_train+n_val]
    folds_test=ans[n_train+n_val:n_train+n_val+n_test]
    print('iter:',index)
    print('treino:',folds_train)
    print('val:',folds_val)
    print('test:',folds_test)
    
    
   
    if only_peri:
    
        "Slipt the three sets:"
        
        "Data with only pericardium: "
        
        "--> CARDIAC FAT"
        train_peri_cfat=cfat_peri_df.loc[cfat_peri_df['Fold'].isin(folds_train)]
        val_peri_cfat=cfat_peri_df.loc[cfat_peri_df['Fold'].isin(folds_val)]
        test_peri_cfat=cfat_peri_df.loc[cfat_peri_df['Fold'].isin(folds_test)]
        
        print("Data with only pericardium:CARDIAC FAT ")
        train_generator_peri_cfat=my_generator(train_peri_cfat,img_data_gen_args,batch_size,seed,Width,Length)
        val_generator_peri_cfat=my_generator(val_peri_cfat,img_data_gen_args,batch_size,seed,Width,Length)
        test_generator_peri_cfat=my_generator(test_peri_cfat,img_data_gen_args,len(test_peri_cfat),seed,Width,Length)
        
        "--> OSIC DATASET"
        train_peri_osic=osic_peri_df.loc[osic_peri_df['Fold'].isin(folds_train)]
        val_peri_osic=osic_peri_df.loc[osic_peri_df['Fold'].isin(folds_val)]
        test_peri_osic=osic_peri_df.loc[osic_peri_df['Fold'].isin(folds_test)]
        
        print("Data with only pericardium: OSIC ")
        train_generator_peri_osic=my_generator(train_peri_osic,img_data_gen_args,batch_size,seed,Width,Length)
        val_generator_peri_osic=my_generator(val_peri_osic,img_data_gen_args,batch_size,seed,Width,Length)
        test_generator_peri_osic=my_generator(test_peri_osic,img_data_gen_args,len(test_peri_osic),seed,Width,Length)
        
        train_combine_generator = combine_gen(train_generator_peri_cfat, train_generator_peri_osic)
        val_combine_generator = combine_gen(val_generator_peri_cfat, val_generator_peri_osic)
        
        
   
    else:
        
        "Data with all data: "
    
        "--> CARDIAC FAT"
        train_cfat=cfat_all_df.loc[cfat_all_df['Fold'].isin(folds_train)]
        val_cfat=cfat_all_df.loc[cfat_all_df['Fold'].isin(folds_val)]
        test_cfat=cfat_all_df.loc[cfat_all_df['Fold'].isin(folds_test)]
        
        print("Data with all data: CARDIAC FAT ")
        train_generator_cfat=my_generator(train_cfat,img_data_gen_args,batch_size,seed,Width,Length)
        val_generator_cfat=my_generator(val_cfat,img_data_gen_args,batch_size,seed,Width,Length)
        test_generator_cfat=my_generator(test_cfat,img_data_gen_args,len(test_cfat),seed,Width,Length)
        
        "--> OSIC DATASET"
        train_osic=osic_all_df.loc[osic_all_df['Fold'].isin(folds_train)]
        val_osic=osic_all_df.loc[osic_all_df['Fold'].isin(folds_val)]
        test_osic=osic_all_df.loc[osic_all_df['Fold'].isin(folds_test)]
        
        print("Data with with all data: OSIC ")
        train_generator_osic=my_generator(train_osic,img_data_gen_args,batch_size,seed,Width,Length)
        val_generator_osic=my_generator(val_osic,img_data_gen_args,batch_size,seed,Width,Length)
        test_generator_osic=my_generator(test_osic,img_data_gen_args,len(test_osic),seed,Width,Length)
        
        train_combine_generator = combine_gen(train_generator_cfat, train_generator_osic)
        val_combine_generator = combine_gen(val_generator_cfat, val_generator_osic)
        
    batch_size=1
    
    from matplotlib import pyplot as plt
    
    # for i in range(1):
    #     generator= next(train_combine_generator)
       
    #     # Display the original image and the masked image side by side
        
    #     #import cv2
    #     #for i in range(batch_size): 
           
    #         # fig=plt.figure(figsize=(10,10))
    #         # plt.subplot(1,2,1)
    #         # plt.imshow(generator[0][i,:,:,1], cmap='gray')
    #         # plt.subplot(1,2,2)
    #         # plt.imshow(generator[1][i,:,:,0])
    #     for i in range(batch_size)    :
    #      for s in range(64):
           
    #         fig=plt.figure(figsize=(10,10))
    #         plt.subplot(1,2,1)
    #         plt.imshow(generator[0][i,s,:,:,0], cmap='gray')
    #         plt.subplot(1,2,2)
    #         plt.imshow(generator[1][i,s,:,:,0], cmap='gray')
            
    #erro
    """Definir modelo"""
    
    # get the current time in seconds since the epoch
    seconds = time.time()
    # convert the time in seconds since the epoch to a readable format
    local_times = time.ctime(seconds)
    
    local_time = '_'.join(local_times.split())
    local_time = '_'.join(local_time.split(':'))
    folds_trains='_'.join(str(folds_train)[1:-1].split(','))
    folds_trains=''.join(folds_trains.split())
    
    from teste_architecture import *


    """Model creation and Summary"""
    model=build_unet((Width,Width,Width,1),1)
    
    # if not only_peri:
    #     """Load the model in case of cascade """
    #     path_model="X:/Ruben/TESE/New_training_Unet/Models_only_pub_dset/Mix_datasets/Only_pericardium/models/2.5D_Unet/Dice_loss/L50_W350_calc_augm_tif/Sun Apr  9 16_11_18 2023" 
    #     model = load_model(os.path.join(path_model,'Loss_Dice_loss__epochs_4000_batch_size_12_wl256Lr_decreasing_0.0001fold_train_0_1_2_time_Sun_Apr_9_16_11_18_2023.h5'),compile=False)     
    # else:
    #     print('Doesnt need to load any model')            
    
    """Onde o modelo vai ser guardado"""
    
    if only_peri:
        name_path='Only_pericardium'
    else:
        name_path='All_data'
        
    model_path="X:/Ruben/TESE/New_training_Unet/Models_only_pub_dset/Mix_datasets/"+str(name_path)+"/models/3D_Unet/BCE/L0_W2000_augm_calc_tif"
    model_path=os.path.join(model_path,str(local_time_cross))
    
    if not os.path.exists(model_path):                         
        # Create a new directory because it does not exist 
        os.makedirs(model_path)
    os.chdir(model_path) 
    
    
    NUM_EPOCHS=4000
    lr=0.0001
    
    from keras.callbacks import ReduceLROnPlateau

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10,
                              verbose=1, min_lr=1e-5)
    
#     # def schedule(epoch, lr):
#     #   if epoch < 14:
#     #       return lr
#     #   else:
#     #       return lr * tf.math.exp(-0.1)
     
    def schedule(epoch, lr):
      #if epoch < 20:
          return lr
      #elif (epoch > 20 and epoch <= 55 ):
        #   return 0.00001
      #else:
        #   return 0.000001
        

    
    "Loss function and optimizer"
    
    Dice_loss=sm.losses.DiceLoss()
    opt = keras.optimizers.Adam(learning_rate= lr)
    #focal_loss=sm.losses.CategoricalFocalLoss()
    #total_loss=Dice_loss + (1*focal_loss)
       
    
    
    #loss_str="Dice+_focal_loss_pesos_"+str(classe_0)+"_"+str(classe_1)
    #loss_str="Dice_loss_"
    loss_str="Binary_cross_entropy_loss"
    
    "Model compilation"
    
    model.compile(optimizer = opt, loss ='binary_crossentropy',  metrics = ['accuracy'])
    
    
    class CustomCSVLogger(CSVLogger):
        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            logs['learning_rate'] = self.model.optimizer.lr.numpy()
            super().on_epoch_end(epoch, logs)

    
    """Name of the model and Callbacks"""
    
    model_name="Loss_"+loss_str+"_time_"+local_time
    csv_logger = CustomCSVLogger(model_name+".csv", append=True)
    
    my_callbacks = [
        callbacks.EarlyStopping(patience=20,monitor='val_loss'),#callbacks.LearningRateScheduler(schedule, verbose=1),
        callbacks.ModelCheckpoint(filepath=model_name+'.h5',save_best_only=True, verbose=1),csv_logger,reduce_lr]
    
    "Definition steps per epoch"
    
    if only_peri:
        
        train_cfat=train_peri_cfat # Quando for apenas pericárdio
        val_cfat=val_peri_cfat
        test_cfat=test_peri_cfat
        
        train_osic=train_peri_osic
        val_osic=val_peri_osic
        test_osic=test_peri_osic
    
    num_train_imgs=len(np.unique(train_cfat['Patient']))  # CARDIAC FAT 
    num_train_imgs_1=len(np.unique(train_osic['Patient'])) # OSIC
    
    total_num_train_imgs=np.max([num_train_imgs,num_train_imgs_1])*2
    steps_per_epoch = total_num_train_imgs //batch_size
    
    "Definition train and val generator"
    train_generator=train_combine_generator
    val_generator=val_combine_generator
    
    "Model training"    
    history = model.fit(train_generator,validation_data=val_generator,steps_per_epoch=steps_per_epoch, 
                        validation_steps=steps_per_epoch,epochs=NUM_EPOCHS,callbacks=[my_callbacks])

    "Add fold information to CSV"
    
    Fold_information = pd.read_csv(model_name+".csv")  
    Fold_information["Train"] = str(folds_train) 
    Fold_information["Validation"] = str(folds_val) 
    Fold_information["Test"] = str(folds_test) 
    Fold_information.to_csv(model_name+".csv") 
    
    "Write .txt with training information"
    
    f= open(model_name+".txt","w+")
    
    # get the current time in seconds since the epoch
    seconds = time.time()
    # convert the time in seconds since the epoch to a readable format
    local_time_end = time.ctime(seconds)
    
    f.write("Date start: "+(local_times)+'\n')
    f.write("Date end: "+(local_time_end)+'\n')
    f.write("Model: 3D U-net"+'\n')
    f.write("Dataset: "+ (model_path)+'\r\n')
    f.write("Cross-Validation ? "+str(cross_validation)+'\r\n')
    f.write("Folds used to train: "+str(folds_train)+'\n')
    f.write("Images used to train: Cardiac Fat: "+ str(len(train_cfat))+" + OSIC: "+str(len(train_osic))+'\r\n')
    f.write("Folds used to validation: "+str(folds_val)+'\n')
    f.write("Images used to validation: Cardiac Fat: "+ str(len(val_cfat))+" + OSIC: "+str(len(val_osic))+'\r\n')
    f.write("Folds used to test: "+str(folds_test)+'\n')
    f.write("Images used to test: Cardiac Fat: "+ str(len(test_cfat))+" + OSIC: "+str(len(test_osic))+'\r\n')
    f.write("Size: %d\n" % (Width))
    f.write("Loss: "+str(loss_str)+'\n')
    f.write("Learning rate: "+str(lr)+'\n')
    f.write("Batch size: %d\n" % (batch_size))
    f.write("Data augmentation: Artificial Calcification: "+str(img_data_gen_args))
    f.close() 
    
    if (cross_validation !="y"):
        break
    



# # """Resultados"""

# # #import pandas as pd
# # #from IPython.display import display
# # import matplotlib.pyplot as plt

# # # plot the model training history
# # N = len(history.history["accuracy"])

# # plt.style.use("ggplot")
# # plt.figure()
# # plt.plot(np.arange(1,N+1), history.history["loss"], label="train_loss")
# # plt.plot(np.arange(1, 1+N), history.history["val_loss"], label="val_loss")
# # plt.title(" Loss on Training and Validation Set")
# # plt.xlabel("Epoch #")
# # plt.ylabel("Loss")
# # plt.legend(loc="upper right")

# # path=os.path.join(model_path,"Curves_"+str(Width))
# # isExist = os.path.exists(path)

# # if not isExist:                         
# #     # Create a new directory because it does not exist 
# #     os.makedirs(path)
# # os.chdir(path)   

# # plt.savefig("Loss_"+model_name+".png")

# # plt.figure()
# # plt.plot(np.arange(1, 1+N), history.history["accuracy"], label="train_acc")
# # plt.plot(np.arange(1, 1+N), history.history["val_accuracy"], label="val_acc")
# # plt.title("Acurracy on Training and Validation Set ")
# # plt.xlabel("Epoch #")
# # plt.ylabel("Acurracy")
# # plt.legend(loc="lower right")

# # plt.savefig("Acc_"+model_name+".png")



# # from numpy import load
# # from keras.models import Model
# # from keras.models import load_model

# # ##Width,Length=256,256


# # """Função para fazer predict de um conjunto de imagens"""

# # x_train = next(val_generator_all)
# # x=x_train[0]
# # y=x_train[1]

# # def predict(model,X):
# #   prediction=[]
# #   for i in range(X.shape[0]):
# #     pred=model.predict(X[i].reshape(1,Width,Length,1))
# #     pred=(pred>0.5).astype(np.uint8)
# #     prediction.append(pred)
# #   return np.array(prediction)

# # """Fazer predict com base no model"""

# # pred_train=predict(model,x)

# # for i in range (x.shape[0]):
# #     fig=plt.figure(figsize=(16,6))
# #     plt.subplot(1,3,1)
# #     plt.imshow(np.squeeze(x[i]),cmap='gray')
# #     plt.title('Original Train_'+str(i))
# #     plt.subplot(1,3,2)
# #     plt.imshow(np.squeeze(y[i]),cmap='gray')
# #     plt.title('label Train_'+str(i))
# #     plt.subplot(1,3,3)
# #     plt.imshow(np.squeeze(pred_train[i]),cmap='gray')
# #     plt.title('Predict_'+str(i))
   
