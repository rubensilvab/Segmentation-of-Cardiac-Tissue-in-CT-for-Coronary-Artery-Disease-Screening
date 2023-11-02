# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 16:19:19 2023

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

mask_data_gen_args = dict(rescale = 1/255.,
                          rotation_range=5,
                          horizontal_flip=True,
                          vertical_flip=True,
                          width_shift_range=0.1,
                          height_shift_range=0.1,
                          zoom_range=0.2,
                          fill_mode='constant',  # Use black color to fill empty areas
                          cval=0,  #Original pixel values are 0 and 255. So rescaling to 0 to 1
                    
                      ) #Binarize the output again. 

      

def image_data_generator_dataframe(dataframe,col_name,img_data_gen_args,batch_size,seed,Width,Length):
    

    image_data_generator = ImageDataGenerator(**img_data_gen_args)
    image_generator = image_data_generator.flow_from_dataframe(
    dataframe,
    directory=None,
    x_col=col_name,
    y_col='Label',
    seed=seed, 
    batch_size=batch_size,
    target_size=(Width,Length),
    color_mode = 'grayscale',
    class_mode='binary'
)
    
    return image_generator   

def my_generator(dataframe,img_data_gen_args,mask_data_gen_args,batch_size,seed,Width,Length,val='n'):
    
    image_generator=image_data_generator_dataframe(dataframe,'Path_image',img_data_gen_args,batch_size,seed,Width,Length)
    mask_generator=image_data_generator_dataframe(dataframe,'Path_Mask',mask_data_gen_args,batch_size,seed,Width,Length)
    
    if (val=='y'):
      generator=image_generator
    else:
      generator=image_and_mask_generator(image_generator, mask_generator)
    
    return generator 

"import function calficification"

from gaussian_calc import *

def image_and_mask_generator(image_generator, mask_generator):
    while True:
        # Get a batch of augmented images and masks
        batch_images = next(image_generator)
        batch_masks = next(mask_generator)
       
        # Apply specific augmentation to each image based on its corresponding mask
        
        augmented_images = []
        augmented_masks = []
        
        
        for i in range(batch_images[0].shape[0]):
            image = batch_images[0][i]
            mask = batch_masks[0][i]
            
            pixels_mask=mask.sum()
            t_pixeis=image.shape[0]*image.shape[0]
            ratio_mask_image=pixels_mask/t_pixeis
            
            random_n=random.randint(1,3) #"Colocar probabilidade em 33%"
            
            if (mask.sum()>0 and random_n==1 and ratio_mask_image>0.09):
              # Apply the specific augmentation to the image   
              image=noisy_gaussian(image,mask)
                
            # Add the augmented image and mask to the list
            augmented_images.append(image)
            
        # Convert the list of augmented images and masks to arrays and yield them
        augmented_images = np.array(augmented_images)
        
        
        yield (augmented_images,batch_images[1])
            


def combine_gen(*gens):
    while True:
        for g in gens:
            yield next(g)
                                                                 
"Só para teste"

Width=256
Length=Width
seed=18
batch_size= 12

"Definition of train, val and test set"

number_folds=5
n_train=int(number_folds*0.60)
n_val=int(number_folds*0.20)
n_test=int(number_folds*0.20)

folds=[]
[folds.append(p) for p in range(number_folds)]

from itertools import cycle


"Import CSV with dicom and masks informations"

cfat_all_df = pd.read_csv('X:/Ruben/TESE/New_training_Unet/all_data/Cardiac_fat_new/Cardiac_fat_new_folds_5.csv')
osic_all_df = pd.read_csv('X:/Ruben/TESE/New_training_Unet/all_data/OSIC_new/OSIC_new_sorted_5.csv')

cfat_all_df ['Label'] = cfat_all_df['Label'].astype(str)
osic_all_df ['Label'] = osic_all_df['Label'].astype(str)

# get the current time in seconds since the epoch
seconds = time.time()
# convert the time in seconds since the epoch to a readable format
local_time_cross = time.ctime(seconds)
local_time_cross = '_'.join(local_time_cross.split(':'))

import mlnotify 

"Pretendemos Cross-Validation ?"

cross_validation="n"

for index, *ans in zip(range(number_folds), *[cycle(folds)] * (number_folds+1)):
   
    "Definition of the folds to train, valid and test the data" 
   
    folds_train=ans[0:n_train]
    folds_val=ans[n_train:n_train+n_val]
    folds_test=ans[n_train+n_val:n_train+n_val+n_test]
    print('iter:',index)
    print('treino:',folds_train)
    print('val:',folds_val)
    print('test:',folds_test)
    
    "Slipt the three sets:"
    
    
    "Data with all data: "
    
    "--> CARDIAC FAT"
   
    train_cfat=cfat_all_df.loc[cfat_all_df['Fold'].isin(folds_train)]
    val_cfat=cfat_all_df.loc[cfat_all_df['Fold'].isin(folds_val)]
    test_cfat=cfat_all_df.loc[cfat_all_df['Fold'].isin(folds_test)]
    
    print("Data with all data: CARDIAC FAT ")
    train_generator_cfat=my_generator(train_cfat,img_data_gen_args,mask_data_gen_args,batch_size,seed,Width,Length)
    val_generator_cfat=my_generator(val_cfat,img_data_gen_args,mask_data_gen_args,batch_size,seed,Width,Length)
    test_generator_cfat=my_generator(test_cfat,img_data_gen_args,mask_data_gen_args,len(test_cfat),seed,Width,Length)
    
    "--> OSIC DATASET"
    train_osic=osic_all_df.loc[osic_all_df['Fold'].isin(folds_train)]
    val_osic=osic_all_df.loc[osic_all_df['Fold'].isin(folds_val)]
    test_osic=osic_all_df.loc[osic_all_df['Fold'].isin(folds_test)]
    
    print("Data with with all data: OSIC ")
    train_generator_osic=my_generator(train_osic,img_data_gen_args,mask_data_gen_args,batch_size,seed,Width,Length)
    val_generator_osic=my_generator(val_osic,img_data_gen_args,mask_data_gen_args,batch_size,seed,Width,Length)
    test_generator_osic=my_generator(test_osic,img_data_gen_args,mask_data_gen_args,len(test_osic),seed,Width,Length)
    
    train_combine_generator = combine_gen(train_generator_cfat, train_generator_osic)
    val_combine_generator = combine_gen(val_generator_cfat, val_generator_osic)
    
    generator= next(train_combine_generator)
     
    for i in range(batch_size): 
        
            fig=plt.figure(figsize=(10,10))
            plt.title('label: '+str(generator[1][i]))
            plt.imshow(generator[0][i], cmap='gray')
            
    "Lidar com classes desbalanceadas"     
    num_class_0= len(train_cfat.loc[train_cfat['Label'].isin(['0'])])+ len(train_osic.loc[train_osic['Label'].isin(['0'])])     
    num_class_1= len(train_cfat.loc[train_cfat['Label'].isin(['1'])])+ len(train_osic.loc[train_osic['Label'].isin(['1'])])     
    total=num_class_0+num_class_1
    ratio_0=total/num_class_0
    ratio_1=total/num_class_1
    class_weights = {}
    #class_weights[0],class_weights[1]=1-ratio,ratio
    class_weights[0],class_weights[1]=ratio_0,ratio_1

    
    """Definir modelo"""
    
    # get the current time in seconds since the epoch
    seconds = time.time()
    # convert the time in seconds since the epoch to a readable format
    local_times = time.ctime(seconds)
    
    local_time = '_'.join(local_times.split())
    local_time = '_'.join(local_time.split(':'))
    folds_trains='_'.join(str(folds_train)[1:-1].split(','))
    folds_trains=''.join(folds_trains.split())
    
    from cnn_slice_classification import cnn
    
    """Model creation and Summary"""
   
    
    model=cnn()
    print(model.summary())
           
    """Onde o modelo vai ser guardado"""
    
    model_path="X:/Ruben/TESE/New_training_Unet/Models_only_pub_dset/Mix_datasets/All_data/classification_models/BCE/L0_W2000_tif_calc_augm"
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
    
    """Name of the model and Callbacks"""
    
    class CustomCSVLogger(CSVLogger):
        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            logs['learning_rate'] = self.model.optimizer.lr.numpy()
            super().on_epoch_end(epoch, logs)
    
    name_str=model_path.split('\\')[0].split('/')[-1]+'_2D'
    model_name=name_str+"_time_"+local_time
    csv_logger = CustomCSVLogger(model_name+".csv", append=True)

    my_callbacks = [
        callbacks.EarlyStopping(patience=20,monitor='val_loss'),#callbacks.LearningRateScheduler(schedule, verbose=1),
        callbacks.ModelCheckpoint(filepath=model_name+'.h5',save_best_only=True, verbose=1),csv_logger,reduce_lr]
    
        
    "Definition steps per epoch"
    
    num_train_imgs=len(train_cfat)  # CARDIAC FAT 
    num_train_imgs_1=len(train_osic) # OSIC
    
    total_num_train_imgs=num_train_imgs+num_train_imgs_1
    steps_per_epoch = total_num_train_imgs //batch_size
    
    "Definition train and val generator"
    train_generator=train_combine_generator
    val_generator=val_combine_generator
    
    "Model training"    
    history = model.fit(train_generator,validation_data=val_generator,steps_per_epoch=steps_per_epoch, 
                        validation_steps=steps_per_epoch,class_weight=class_weights,epochs=NUM_EPOCHS,callbacks=[my_callbacks])

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
    f.write("Model: CNN"+'\n')
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
    f.write("Data augmentation: Yes (Artificial Calcification) " +str(img_data_gen_args))
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
   
