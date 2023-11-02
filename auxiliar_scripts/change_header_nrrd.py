# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 12:16:45 2023

@author: RubenSilva
"""

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
import pandas as pd

def reshape_nrrd(nrrd):
  
  nrrd=np.array(nrrd)
  nrrd=np.transpose(nrrd,(2,1,0))
  nrrd=nrrd[::-1]  
  return nrrd  

"Análise OSIC"

PATH ='X:/Ruben/TESE/New_training_Unet/Models_only_pub_dset/Mix_datasets/All_data/classification_models/predict/2D_CNN/BCE/OSIC_tif/L0_W2000_tif_calc_augm/th_0.517566/Peri_segm/NRRD' 
csv_thick=pd.read_csv('X:/Ruben/TESE/New_training_Unet/all_data/OSIC_new/OSIC_test_set_thickness.csv')

# path_label='X:/Ruben/TESE/New_training_Unet/all_data/OSIC_new/4/Mask/'
list_patients=sorted(os.listdir(PATH))

nu=0
nu1=0
patients=[]
patients_prob=[]

for patient in list_patients:
    patients.append(patient)
    n=0
    
    path_p=os.path.join(PATH,patient)
    files=sorted(glob.glob(path_p+'/*'))
    thick_p=csv_thick.loc[csv_thick['Patient'].isin([(patient)])]
    #print(thick_p)
    thick_p=thick_p['Thickness']
    #print(thick_p)
    
    for file in files:
            
            try:
                # Read the data back from file
                readdata, header_original = nrrd.read(file)
               # header_original=nrrd.read_header(file)
                #print(header_original['space directions'])
                header_original['space directions'][2][2]=float(thick_p)
                #header_original['space directions']=abs(header_original['space directions'])
                
                #print(patient.upper(),header_original['space directions'],thick_p)
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
                
                nrrd.write(file, readdata,header=header_original)
                print(patient)
                
                
            except:
                print("error , dont exist segm_nrd,patient :",patient)
                patients_prob.append(patient)
                pass
print("FEITO")

