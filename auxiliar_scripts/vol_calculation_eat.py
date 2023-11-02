# -*- coding: utf-8 -*-
"""
Created on Fri May  5 00:00:03 2023

@author: RubenSilva
"""
import nrrd
import matplotlib.pyplot as plt
import numpy as np
import os,glob
import xlsxwriter,openpyxl

path_nrrd="X:/Ruben/TESE/New_training_Unet/Models_only_pub_dset/Mix_datasets/All_data/predict/2.5D_Unet/Dice_loss/Hospital_tif/L0_W2000_calc_augm_tif/EAT_segm_nHU/NRRD"

# path_to_move=path_nrrd.split('/')
# path_to_move=('/').join(path_to_move[:13])
     
patients=[patient for patient in os.listdir(path_nrrd) if os.path.isdir(os.path.join(path_nrrd, patient)) ]

def reshape_nrrd_to_arr(nrrd,n):
  
  nrrd=np.array(nrrd)
  nrrd=np.transpose(nrrd,(2,1,0))
  if n==1:  
      nrrd=nrrd[::-1] 
  #nrrd=nrrd[::-1]  
  return nrrd  


def excel_results(path,patient,name,row,vol_manual,vol_convex):#,vol_convex,vol_fill2d):
    
    isExist = os.path.exists(path)
    if not isExist:                         
        # Create a new directory because it does not exist 
        os.makedirs(path)
        
    os.chdir(path) 
    
        
    filename = str(name)+'.xlsx'
    
    # Check if the file already exists
    if not os.path.isfile(filename):
        # Create a new workbook object
        book = openpyxl.Workbook()
    
        # Select the worksheet to add data to
        sheet = book.active
    
        # Add a header row to the worksheet
        sheet.append(['Patient', 'Volume EAT manual (cm^3)','Volume EAT convex (cm^3)'])#, 'Volume EAT convex (cm^3)', 'Volume EAT fill 2d (cm^3)'])
    else:
        # Open an existing workbook
        book = openpyxl.load_workbook(filename)
    
        # Select the worksheet to add data to
        sheet = book.active
    
    # Append a new row of data to the worksheet
    sheet.append([patient, vol_manual,vol_convex])#, vol_convex, vol_fill2d])
    #print(vol_fill2d)
    # Save the workbook to a file
    book.save(filename)


c=0
row=0
n=0
for patient in patients:

    
    # if 'ospit' in path_nrrd:
        
    #     print('hospital')
    #     n=0
    # elif 'ardiac' in path_nrrd:
        
    #     print('Cardiac')
    #     n=1
    # else:
        
    #     print('OSIC')
        
    #     n=1    

    print('Faltam',len(patients)-c,' pacientes. Atual:',patient)
    path_nrrd_patient=os.path.join(path_nrrd,patient)
    
    #buscar nrrds
    files_nrrd=glob.glob(os.path.join(path_nrrd_patient,'*'))

    for file in files_nrrd:
        
        if file[-6]=='v':
             Convex_mask, header = nrrd.read(file)
             Convex_mask=reshape_nrrd_to_arr(Convex_mask,n)
        # elif file[-6]=='d':   
        #     fill2d_mask, header = nrrd.read(file)
        #     fill2d_mask=reshape_nrrd_to_arr(fill2d_mask,n)
        # elif file[-6]=='6':   
        #     dicom, header = nrrd.read(file)
        #     dicom=reshape_nrrd_to_arr(dicom,n)
        if 'manual' in file:   
            manual, header = nrrd.read(file)
            manual=reshape_nrrd_to_arr(manual,n)
            
    voxel_directions=header['space directions']
    volume_voxel=voxel_directions[0,0]*voxel_directions[1,1]*voxel_directions[2,2]
    path_to_move="X:/Ruben/TESE/New_training_Unet/Models_only_pub_dset/Mix_datasets/All_data/predict/2.5D_Unet/Dice_loss/Hospital_tif/L0_W2000_calc_augm_tif/EAT_segm_nHU"

    path_convex=os.path.join(path_to_move,'Volume Results')
    
    vol_manual=np.sum(manual)*volume_voxel*(1e-3) # em cm3
    vol_convex=np.sum(Convex_mask)*volume_voxel*(1e-3) # em cm3   
    #vol_fill2d=np.sum(fill2d_mask)*volume_voxel*(1e-3) # em cm3  
    
    row+=1
    name="volumes_eat_r1auto"
    excel_results(path_convex,patient,name,row,vol_manual,vol_convex)#,vol_fill2d)
    
    
    c+=1 


