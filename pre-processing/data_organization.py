# -*- coding: utf-8 -*-
"""
Created on Sun Feb 12 21:50:00 2023

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



"""Functions to process Dicom files"""

def window_image(img, window_center,window_width, intercept, slope,raw,rescale):
    img = (img*slope +intercept) #for translation adjustments given in the dicom file. 
    if raw==False:
        img_min = window_center - window_width//2 #minimum HU level
        img_max = window_center + window_width//2 #maximum HU level
        img[img<img_min] = img_min #set img_min for all HU levels less than minimum HU level
        img[img>img_max] = img_max #set img_max for all HU levels higher than maximum HU level
        if rescale: 
            img = (img - img_min) / (img_max - img_min)*65535 # Para 16 bit
            #img = (img - img_min) / (img_max - img_min)*255.0 # Para 8 bit
            img=img.astype(np.uint16)
    return img

def reconvert_HU(array,window_center,window_width,L=0,W=2000,rescale=True):
    
    img_min = L - W//2 #minimum HU level
    img_max =L + W//2 #maximum HU level
    
    reconvert_img=(array/65535)*(img_max - img_min) +  img_min # Reconvertido para HU
    
    
    new_img_min = window_center - window_width//2 #minimum HU level, que pretendemos truncar
    new_img_max =window_center + window_width//2 #maximum HU level, que pretendemos truncar
    
    reconvert_img[reconvert_img<new_img_min] = new_img_min #set img_min for all HU levels less than minimum HU level
    reconvert_img[reconvert_img>new_img_max] = new_img_max #set img_max for all HU levels higher than maximum HU level
    
    if rescale: 
        reconvert_img = (reconvert_img - new_img_min) / (new_img_max - new_img_min)*65535 # Para 16 bit
        #reconvert_img = (reconvert_img - new_img_min) / (new_img_max - new_img_min)*255 # Para 8 bit
        reconvert_img=reconvert_img.astype(np.uint16)
     
        
    return reconvert_img
    
def get_first_of_dicom_field_as_int(x):
    #get x[0] as in int is x is a 'pydicom.multival.MultiValue', otherwise get int(x)
    if type(x) == pydicom.multival.MultiValue: return int(x[0])
    else: return int(x)
    

  
def get_windowing(data):
    dicom_fields = [data[('0028','1050')].value, #window center
                    data[('0028','1051')].value, #window width
                    data[('0028','1052')].value, #intercept
                    data[('0028','1053')].value] #slope
    return [get_first_of_dicom_field_as_int(x) for x in dicom_fields]

def window_level(dicom_path,L=0,W=2000,raw=False,rescale=True):
  #Default: [-1000,1000] 
  
  data = pydicom.read_file(dicom_path)
  window_center , window_width, intercept, slope = get_windowing(data)
  img = pydicom.read_file(dicom_path).pixel_array
  img2 = window_image(img, L, W, intercept, slope,raw,rescale)   
  return img2    

"""Funcions to adjust nrrd"""

def reshape_nrrd(nrrd):
  
  nrrd=np.array(nrrd)
  nrrd=np.transpose(nrrd,(2,1,0))
  nrrd=nrrd[::-1]  
  return nrrd  


PATH ='X:/Ruben/TESE/Data/hospital_gaia/imgs_tif/data1/teste/' 

"""
- Pasta principal(teste)
         - Paciente 1
                 - slice1.dcm
                 -slice2.dcm
                 - ...
         - Paciente 2
                 - slice1.dcm
                 -slice2.dcm
                 - ...

"""


list_patients=sorted(os.listdir(PATH))

nu=0
nu1=0
patients=[]
patients_prob=[]

for patient in list_patients:
    patients.append(patient)
    n=0
    files=sorted(glob.glob(PATH+patient+'/DICOM/*'))
    
    for file in files:
            try: 
                n=n+1
                           
                "Apenas guardar imagem com -1000 a 1000 HU e 16 bit"
                
                #Pasta onde queres guardar as imagens .tif
                dicom_img_path="X:/Ruben/TESE/Data/hospital_gaia/imgs_tif/data1/dicom"
                 
                path=os.path.join(dicom_img_path,str(patient))
                isExist = os.path.exists(path)
                if not isExist:                         
                    # Create a new directory because it does not exist 
                    os.makedirs(path)
                os.chdir(path) 
                 
                img=window_level(file)
                #img_dicom=window_level(file,raw=True)
                cv2.imwrite(str(patient)+'_'+str(n)+'.tif', img)
                img_read=cv2.imread(str(patient)+'_'+str(n)+'.tif',flags=cv2.IMREAD_ANYDEPTH)
                 
                nu=nu+1
                print(file,"dcm",n,"total:",nu)
                 
                
                "Aplicando window level- só para fins de teste"
                #Pasta onde vao ser guardadas estas imagens de teste
                
                dicom_img_path="X:/Ruben/TESE/Data/hospital_gaia/imgs_tif/data1/WL"
             
                path=os.path.join(dicom_img_path,str(patient))
                isExist = os.path.exists(path)
                if not isExist:                         
                # Create a new directory because it does not exist 
                  os.makedirs(path)
                os.chdir(path) 
             
                reconvert_img=reconvert_HU(img_read,50,350,rescale=True)
                cv2.imwrite(str(patient)+'_'+str(n)+'.tif', reconvert_img)
                img_read_reconvert=cv2.imread(str(patient)+'_'+str(n)+'.tif',flags=cv2.IMREAD_ANYDEPTH)
                
                #plt.imshow(reconvert_img,cmap="gray")
                #plt.show()
                nu=nu+1
                print(file,"dcm",n,"total:",nu)
             
    
                
            except:
                print("error ")
                patients_prob.append(patient)
                pass
    
    # PATH_nrd =os.path.join(PATH, patient, "segm_manual_Carolina.nrrd") 
    # try:
    #     # Read the data back from file
    #     readdata, header = nrrd.read(PATH_nrd)
    #     new=reshape_nrrd(readdata)
    #     for i in range(new.shape[0]):
    #         # save the  dicom image 
    #         dicom_img_path="X:/Ruben/TESE/Data/hospital_gaia/all_data_carolina/Mask_tif"
    #         path=os.path.join(dicom_img_path,str(patient))
    #         isExist = os.path.exists(path)
    #         #print(path_to_cpy,isExist)
    #         if not isExist:                         
    #             # Create a new directory because it does not exist 
    #             os.makedirs(path)
    #         os.chdir(path) 
    #         cv2.imwrite(str(patient)+'_'+str(i+1)+'.tif', new[i,:,:].reshape(512,512,1))
    #         #save_img(str(patient)+'_'+str(i+1)+'.tif', new[i,:,:].reshape(512,512,1))
          
    #         nu1=nu1+1    
    #         print(str(patient)+'_'+str(i+1)+'.png',"mask",i+1,"total:",nu1)
    # except:
    #     print("error , dont exist segm_nrd,patient :",patient)
    #     patients_prob.append(patient)
    #     pass
print("FEITO")



