# -*- coding: utf-8 -*-
"""
Created on Mon May  1 03:44:13 2023

@author: RubenSilva
"""

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

os.chdir("X:/Ruben/TESE/New_training_Unet") 

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
#                             rotation_range=5,
#                             horizontal_flip=True,
#                             vertical_flip=True,
#                             width_shift_range=0.1,
#                             height_shift_range=0.1,
#                             zoom_range=0.2,
#                             fill_mode='constant',  # Use black color to fill empty areas
#                             cval=0,  
                         rescale=1/65535.
                    
                      )

def image_data_generator_dataframe(dataframe,col_name,img_data_gen_args,batch_size,seed,Width,Length):
    

    image_data_generator = ImageDataGenerator(**img_data_gen_args)
    image_generator = image_data_generator.flow_from_dataframe(
    dataframe,
    directory=None,
    x_col=col_name,
    seed=seed, 
    batch_size=batch_size,
    target_size=(Width,Length),
    color_mode = 'grayscale',
    class_mode=None
)
    
    return image_generator   


def my_generator(dataframe,img_data_gen_args,batch_size,seed,Width,Length,calc='n'):
    input_col = 'Path_image'
    mask_col = 'Path_Mask'
    cols=[input_col,mask_col]
    
    img_data_gen_args = img_data_gen_args.copy()
    mask_data_gen_args = img_data_gen_args.copy()
    mask_data_gen_args['rescale'] = 1/255.
    
    #print(self.mask_data_gen_args)
    if "preprocessing_function" in mask_data_gen_args:
       mask_data_gen_args.pop("preprocessing_function")
    
    image_generator=image_data_generator_dataframe(dataframe,cols[0],img_data_gen_args,batch_size,seed,Width,Length)
    mask_generator=image_data_generator_dataframe(dataframe,cols[1],mask_data_gen_args,batch_size,seed,Width,Length)
    
    if (calc=='n'):
      generator=zip(image_generator,mask_generator)
    else:
      generator=image_and_mask_generator(image_generator, mask_generator)
    
    
    return generator 

"import function calcification"

from gaussian_calc import *

def image_and_mask_generator(image_generator, mask_generator):
    while True:
        # Get a batch of augmented images and masks
        batch_images = next(image_generator)
        batch_masks = next(mask_generator)
        #print(batch_images.shape,batch_masks.shape)
        
        # Apply specific augmentation to each image based on its corresponding mask
        
        augmented_images = []
        augmented_masks = []
        
        for i in range(batch_images.shape[0]):
            image = batch_images[i]
            mask = batch_masks[i]
            #print(np.unique(mask),np.unique(image))
            
            pixels_mask=mask.sum()
            t_pixeis=image.shape[0]*image.shape[0]
            ratio_mask_image=pixels_mask/t_pixeis
            
            random_n=random.randint(1,3) #"Colocar probabilidade em 33%"
            
            if (mask.sum()>0 and random_n==1 and ratio_mask_image>0.09):
              # Apply the specific augmentation to the image   
              image=noisy_gaussian(image,mask)
            
            # Add the augmented image and mask to the list
            augmented_images.append(image)
            augmented_masks.append(mask)
        # Convert the list of augmented images and masks to arrays and yield them
        augmented_images = np.array(augmented_images)
        augmented_masks = np.array(augmented_masks)
        
        yield (augmented_images, augmented_masks)
            


def combine_gen(*gens):
    while True:
        for g in gens:
            yield next(g)
                                                                 
"Só para teste"

Width=512
Length=Width
seed=17
batch_size= 4

"Definition of train, val and test set"

number_folds=5
n_train=int(number_folds*0.60)
n_val=int(number_folds*0.20)
n_test=int(number_folds*0.20)

folds=[]
[folds.append(p) for p in range(number_folds)]

from itertools import cycle


"Import CSV with dicom and masks informations"

cfat_peri_df = pd.read_csv('X:/Ruben/TESE/New_training_Unet/data_only_pericardium/Cardiac_fat_new/Cardiac_fat_new_folds_5.csv')
cfat_all_df = pd.read_csv('X:/Ruben/TESE/New_training_Unet/all_data/Cardiac_fat_new/Cardiac_fat_new_folds_5.csv')
osic_peri_df = pd.read_csv('X:/Ruben/TESE/New_training_Unet/data_only_pericardium/OSIC_new/OSIC_new_folds_5.csv')
osic_all_df = pd.read_csv('X:/Ruben/TESE/New_training_Unet/all_data/OSIC_new/OSIC_new_folds_5.csv')


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
        
    #batch_size=12
    
    # for i in range(10):
    #     generator= next(train_combine_generator)
        
    #     from matplotlib import pyplot as plt
        
    #     fig=plt.figure(figsize=(10,10))
    #     fig_mask=plt.figure(figsize=(10,10))
        
    #     # Display the original image and the masked image side by side
        
    #     import cv2
    #     for i in range(batch_size): 
           
    #         fig=plt.figure(figsize=(10,10))
    #         plt.subplot(1,2,1)
    #         plt.imshow(generator[0][i,:,:,0], cmap='gray')
    #         plt.subplot(1,2,2)
    #         plt.imshow(generator[1][i,:,:,0])
            
            
            
    #     erro
    """Definir modelo"""
    
    # get the current time in seconds since the epoch
    seconds = time.time()
    # convert the time in seconds since the epoch to a readable format
    local_times = time.ctime(seconds)
    
    local_time = '_'.join(local_times.split())
    local_time = '_'.join(local_time.split(':'))
    folds_trains='_'.join(str(folds_train)[1:-1].split(','))
    folds_trains=''.join(folds_trains.split())
    
    #from twofiveDUnet import *
    from unet2 import *
    """Model creation and Summary"""
 
    model=build_unet((Width,Length,1),1)
    print(model.summary())
    
    #model=twofiveD_Unet()
    #print(model.summary())
    
    if not only_peri:
        """Load the model in case of cascade """
        path_model="X:/Ruben/TESE/New_training_Unet/Models_only_pub_dset/Mix_datasets/Only_pericardium/models/2D_Unet/Dice_loss/L0_W2000_tif/512/Wed May 17 01_42_01 2023" 
        model = load_model(os.path.join(path_model,'512_2D_time_Wed_May_17_01_42_02_2023.h5'),compile=False)     
    else:
        print('Doesnt need to load any model')            
    
    """Onde o modelo vai ser guardado"""
    
    if only_peri:
        name_path='Only_pericardium'
    else:
        name_path='All_data'
        
    model_path="X:/Ruben/TESE/New_training_Unet/Models_only_pub_dset/Mix_datasets/"+str(name_path)+"/models/2D_Unet/Dice_loss/L0_W2000_tif/512"
    model_path=os.path.join(model_path,str(local_time_cross))
    
    if not os.path.exists(model_path):                         
        # Create a new directory because it does not exist 
        os.makedirs(model_path)
    os.chdir(model_path) 
    
    
    NUM_EPOCHS=4000
    lr=0.000001
    
    from keras.callbacks import ReduceLROnPlateau

    # reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10,
    #                           verbose=1, min_lr=1e-5)
    
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
       
    "Caso do deep supervised learning"
    
    # def deep_supervision_loss(y_true, y_pred):
    # # Upsample out2 and out3 to match the size of the input and compute the losses
    #     #Dice_loss=sm.losses.DiceLoss()
    #     loss_weights=[0.4,0.3,0.3]
    #     total_loss=0 
    #     # for i,pred in enumerate(y_pred):
    #     #         y_resized = tf.image.resize(y_true,[*pred.shape[1:3]])
    #     #         loss=Dice_loss(y_resized, pred)
    #     #         total_loss=total_loss+loss_weights[i]*loss
    #     #         print(tf.shape(y_true),tf.shape(y_pred[i]))
    #     def compute_loss(pred):
    #         #print(pred.shape[0:2])
    #         y_resized = tf.image.resize(y_true, size=pred.shape[0:2])
    #         print('True resized: ',y_resized.shape)
    #         loss = Dice_loss(y_resized, pred)
    #         return loss
        
    #     print('Y true:',y_true.shape,'Pred: ',y_pred.shape)
       
    #     losses = tf.map_fn(compute_loss, y_pred, dtype=tf.float32)
    #     print('Loss:', losses)
    #     #print(y_true.shape,y_pred[1].shape)
    #     print('')
    #     total_loss = tf.reduce_sum(loss_weights * losses)
    #     print('total Loss:', total_loss[0])
    #    # print(losses,total_loss)
    #     return total_loss
    
    
    
    #loss_str="Dice+_focal_loss_pesos_"+str(classe_0)+"_"+str(classe_1)
    loss_str="Dice_loss_"
    #loss_str="Binary_cross_entropy_loss"
    #loss_str="Deeo_supervision_dice_loss"
    
    
    
    
    "Model compilation"
    
    model.compile(optimizer = opt, loss =Dice_loss,  metrics = ['accuracy'])
    
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
        callbacks.EarlyStopping(patience=20,monitor='val_loss'),callbacks.LearningRateScheduler(schedule, verbose=1),
        callbacks.ModelCheckpoint(filepath=model_name+'.h5',save_best_only=True, verbose=1),csv_logger]#,reduce_lr]
    
    "Definition steps per epoch"
    
    if only_peri:
        
        train_cfat=train_peri_cfat # Quando for apenas pericárdio
        val_cfat=val_peri_cfat
        test_cfat=test_peri_cfat
        
        train_osic=train_peri_osic
        val_osic=val_peri_osic
        test_osic=test_peri_osic
    
    num_train_imgs=len(train_cfat)  # CARDIAC FAT 
    num_train_imgs_1=len(train_osic) # OSIC
    
    #total_num_train_imgs=np.max([num_train_imgs,num_train_imgs_1])*2
    total_num_train_imgs=num_train_imgs+num_train_imgs_1

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
    f.write("Model: 2D U-net"+'\n')
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
    

import subprocess
import os
if only_peri:
# run the second script
       
   runfile(os.path.join('X:/Ruben/TESE/New_training_Unet','new_framework_2D_unet.py'))
# start the second script in a new process


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
   
