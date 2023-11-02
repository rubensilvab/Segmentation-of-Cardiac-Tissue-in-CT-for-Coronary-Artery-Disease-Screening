# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 15:10:59 2023

@author: RubenSilva
"""

import nrrd
#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pydicom 
import numpy as np
import numpy as np
from skimage import io, exposure, img_as_uint, img_as_float
import os
import glob
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import save_img
from tensorflow.keras.preprocessing.image import img_to_array
import cv2

def reshape_nrrd(nrrd):
  
  nrrd=np.array(nrrd)
  nrrd=np.transpose(nrrd,(2,1,0))
  nrrd=nrrd[::-1]  
  return nrrd  

"Análise Hospital"

# PATH ='X:/Ruben/TESE/Data/hospital_gaia/EPICHEART_Carolina/selection/' 
# path ='X:/Ruben/TESE/New_training_Unet/Models_only_pub_dset/Mix_datasets/All_data/predict/Dice_loss/Hospital_tif/L50_W350_tif/new/NRRD' 

# list_patients=sorted(os.listdir(PATH))

# nu=0
# nu1=0
# patients=[]
# patients_prob=[]

# for patient in list_patients[0:2]:
#     patients.append(patient)
#     n=0
    
#     PATH_nrd_original =os.path.join(PATH, patient, "segm_manual_Carolina.nrrd")
#     PATH_nrd =os.path.join(path, patient, str(patient)+'_predict_'+str(256)+'.nrrd') 
#     try:
#         # Read the data back from file
#         header_original=nrrd.read_header(PATH_nrd_original)
#         #header_mine = nrrd.read_header(PATH_nrd) 
#         #print(header_mine,header_original)
#         #new=reshape_nrrd(readdata)
#         #print(header)
#         # path ='X:/Ruben/TESE/Data/hospital_gaia/imgs_tif/data1/teste' 
#         # path_to =os.path.join(path,patient) 
#         # isExist = os.path.exists(path_to)
#         # #print(path_to_cpy,isExist)
#         # if not isExist:                         
#         #     # Create a new directory because it does not exist 
#         #     os.makedirs(path_to)
#         # os.chdir(path_to) 
        
#         # nrrd.write(str(patient)+'.nrrd', readdata,header=header)
        
        
        
#     except:
#         print("error , dont exist segm_nrd,patient :",patient)
#         patients_prob.append(patient)
#         pass
# print("FEITO")

# PATH ='X:/Ruben/TESE/Data/Dataset_public/RioFatSegm/RioFatSegm/Dicom _ Treino/' 

# path_dicom='X:/Ruben/TESE/New_training_Unet/all_data/Cardiac_fat_new/4/Dicom/'
# path_label='X:/Ruben/TESE/New_training_Unet/all_data/Cardiac_fat_new/4/Mask/'
# list_patients=sorted(os.listdir(PATH))

# nu=0
# nu1=0
# patients=[]
# patients_prob=[]
# from collections import OrderedDict
# for patient in list_patients[0:2]:
#     patients.append(patient)
#     n=0
#     dicom_files= os.path.join(PATH,patient)
#     files=sorted(glob.glob(dicom_files+'/*'))
#     # Read the data back from file
#     data = pydicom.read_file(files[0])
    
    
#     pix_spacing= data.get("PixelSpacing")
#     thick= data.get('SpacingBetweenSlices')
#     origin=data.get('ImagePositionPatient')
#     origins=np.array([origin[0],origin[1],origin[2]])
    
#     pix_spacing.append(thick)
#     space_directions=np.diag(pix_spacing)
    
#     header=OrderedDict()
#     header['space directions']=space_directions
#     print(patient,files[0],header)
        
        # header_mine = nrrd.read_header(PATH_nrd) 
        # print(header_mine,header_original)
        # new=reshape_nrrd(readdata)
        # print(header)
        # path ='X:/Ruben/TESE/Data/hospital_gaia/imgs_tif/data1/teste' 
        # path_to =os.path.join(path,patient) 
        # isExist = os.path.exists(path_to)
        # #print(path_to_cpy,isExist)
        # if not isExist:                         
        #     # Create a new directory because it does not exist 
        #     os.makedirs(path_to)
        # os.chdir(path_to) 
        
        # nrrd.write(str(patient)+'.nrrd', readdata,header=header)
        
        

# print("FEITO")



"Análise OSIC"

PATH ='X:/Ruben/TESE/Data/Dataset_public/Orcya/nrrd_heart/' 
path_dicom='X:/Ruben/TESE/New_training_Unet/all_data/OSIC_new/4/Dicom/'
path_dcm="X:/Ruben/TESE/Data/Dataset_public/Orcya/orcic/"
# path_label='X:/Ruben/TESE/New_training_Unet/all_data/OSIC_new/4/Mask/'
list_patients=sorted(os.listdir(path_dicom))

nu=0
nu1=0
patients=[]
patients_prob=[]

for patient in list_patients[0:6]:
    patients.append(patient)
    n=0
    
    dicom_files= os.path.join(path_dcm,patient)
    files=sorted(glob.glob(dicom_files+'/*'))
    # Read the data back from file
    data = pydicom.read_file(files[0])
    

    thick=data.get('SliceThickness')
    PATH_nrd_original =os.path.join(PATH,(patient).upper()+"_heart.nrrd")
    #PATH_nrd =os.path.join(path, patient, str(patient)+'_predict_'+str(256)+'.nrrd') 
    try:
        # Read the data back from file
        header_original=nrrd.read_header(PATH_nrd_original)
        header_original['space directions'][2][2]=float(thick)
        header_original['space directions']=abs(header_original['space directions'])
        #header_mine = nrrd.read_header(PATH_nrd) 
       # print(patient.upper(),header_original)
        print(patient.upper(),header_original['space directions'],thick)
        #new=reshape_nrrd(readdata)
        #print(header)
        # path ='X:/Ruben/TESE/Data/hospital_gaia/imgs_tif/data1/teste' 
        # path_to =os.path.join(path,patient) 
        # isExist = os.path.exists(path_to)
        # #print(path_to_cpy,isExist)
        # if not isExist:                         
        #     # Create a new directory because it does not exist 
        #     os.makedirs(path_to)
        # os.chdir(path_to) 
        
        # nrrd.write(str(patient)+'.nrrd', readdata,header=header)
        
        
        
    except:
        print("error , dont exist segm_nrd,patient :",patient)
        patients_prob.append(patient)
        pass
print("FEITO")

# "Teste de resize de um array"

# def sort_specific(files):
#   sorted_files=[]
#   for file in files:
#          order=file[-7:-3]
#          if order[1]=='_':
#              sorted_files.append(file)
#   for file in files:
#          order=file[-7:-3]
#          if order[0]=="_":
#              sorted_files.append(file)  
#   for file in files:
#          order=file[-8:-3]
#          if order[0]=="_":
#              sorted_files.append(file)  
#   return sorted_files  

             
#   return sorted_files     

# def turn2array(x):
#       for i in range(len(x)):
          
#           x[i]=np.array(x[i])
           
#       return x 

# def arrayresize_512(x):
#     X=[]
#     for p in range(len(x)):
#         Xp=[]
#         for s in range(len(x[p])):
#             x_new=cv2.resize(x[p][s], (512, 512)) 
#             Xp.append(list(x_new))
#         Xp=np.array(Xp)
#         X.append(Xp)
#     X=turn2array(X)    
#     return X

# def import_test_images(path_dicom,path_label,Width,Length):

#     list_patients=sorted(os.listdir(path_dicom))
#     contador=0
#     X=[]
#     Y=[]
#     patients=[]
#     for patient in list_patients:
#         patients.append(patient)
#         dicom_tr=sorted(glob.glob(path_dicom+patient+'/*.tif'))
#         label_tr=sorted(glob.glob(path_label+patient+'/*.tif'))
#         dicom_tr=sort_specific(dicom_tr)
#         label_tr=sort_specific(label_tr)
#         Xp=[]
#         Yp=[]
        
#         for file_x,file_y in zip(dicom_tr,label_tr):
#           img_x=cv2.imread(file_x,flags=cv2.IMREAD_ANYDEPTH)
#           print(file_x, patient,img_x)
#           #img_x=reconvert_HU(img_x,50,350)
#           #img_x=cv2.imread(file_x,0)
#           img_y=cv2.imread(file_y,0)
#           img_x=cv2.resize(img_x, (Width, Length))
#           img_y=cv2.resize(img_y, (Width, Length))
#           Xp.append(list(img_x))
#           Yp.append(list(img_y))
          
          
#         Xp=np.array(Xp)/65535
#         Yp=np.array(Yp)/255
#         Yp=(Yp>0.5).astype(np.uint8)
#         X.append(Xp)
#         Y.append(Yp)    
#         contador+=1
#         print("Falta fazer load das imagens de teste para ",len(list_patients)-contador," pacientes")
#     X=turn2array(X)
#     Y=turn2array(Y)    
           
#     return X,Y,patients

# path_dicom='X:/Ruben/TESE/New_training_Unet/all_data/Cardiac_fat_new/4/Dicom/'
# path_label='X:/Ruben/TESE/New_training_Unet/all_data/Cardiac_fat_new/4/Mask/'


# X,Y,patients=import_test_images(path_dicom,path_label,256,256)
# x512=arrayresize_512(X)