# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 00:13:42 2023

@author: RubenSilva
"""

import os
import glob
import pydicom
import pandas as pd
import numpy as np
path_osic="X:/Ruben/TESE/Data/Dataset_public/Orcya/orcic/"

osic_all_df = pd.read_csv('X:/Ruben/TESE/New_training_Unet/all_data/OSIC_new/OSIC_new_sorted_5.csv')
osic_peri_df=osic_all_df.loc[(osic_all_df['Label']==1)]
osic_peri_df=osic_peri_df.loc[osic_peri_df['Fold'].isin([4])]
path_to_copy=""

osic_3d=pd.DataFrame()

dic={}
number_slices_peri={}

new_thick=pd.DataFrame({'Patient':[],'Thickness':[]})

# """Para treino """
# i=0
# patients=np.unique(osic_all_df['Patient'])
# for patient in patients:
    
#     files=glob.glob(os.path.join(path_osic,patient,'*'))
    
#     p=osic_peri_df.loc[(osic_peri_df['Patient']==patient)]
#     p_all=osic_all_df.loc[(osic_all_df['Patient']==patient)]
    
#     data = pydicom.read_file(files[0])
#     thick=data.get('SliceThickness')
    
#     """Pacientes em que temos o pericárdio <64 e não é preciso fazer downsampling"""
#     if len(p)<64:
#         i=i+1
#         print(patient,'Only:peri', len(p),'All:',len(files),'Thickness:',thick,'number:',i)
#         osic_3d=pd.concat([osic_3d, p_all])
#         #number_slices_peri[patient]=len(p)
     
#     else:
#         """Pacientes em que é preciso fazer downsampling"""   
#         ratio=2/thick #tentar espessura perto de 2mm
        
#         ratio=round(ratio)
        
#         if len(p)/ratio>64:
#             ratio=ratio+1
#             dic[patient]=ratio
#             print('')
#             print('patient:', patient,' ratio:', ratio,'number slices: ',len(files))
#             print('real thickness:',thick , 'new thickness:',ratio*thick)
#             print('old slices (peri):', len(p), 'new slices(peri):', len(p)/ratio)
#             print('')
#             number_slices_peri[patient]=len(p)/ratio
            
#         else:
#             print('')
#             print('patient:', patient,' ratio:', ratio,'number slices: ',len(files))
#             print('real thickness:',thick , 'new thickness:',ratio*thick)
#             print('old slices (peri):', len(p), 'new slices(peri):', len(p)/ratio)
#             dic[patient]=ratio
#             print('')
#             number_slices_peri[patient]=len(p)/ratio

#"""Para teste """
i=0
pret_thick=3 # espessura que pretendemos 

patients=np.unique(osic_peri_df['Patient'])

for patient in patients:
    
    files=glob.glob(os.path.join(path_osic,patient,'*'))
    
    
    p=osic_peri_df.loc[(osic_peri_df['Patient']==patient)]
   
    p_all=osic_all_df.loc[(osic_all_df['Patient']==patient)]
    
    data = pydicom.read_file(files[0])
    thick=data.get('SliceThickness')
    
    if (thick==1 and len(p)<64):
        ratio=1
        dic[patient]=ratio
        print('')
        print('patient:', patient,' ratio:', ratio,'number slices: ',len(files))
        print('real thickness:',thick , 'new thickness:',10*thick)
        print('old slices (peri):', len(p), 'new slices(peri):', len(p)/ratio)
        print('')
        row_dict=pd.DataFrame({'Patient':[patient],'Thickness':[ratio*thick*10]})
        new_thick=pd.concat([new_thick,row_dict],ignore_index=True)
    
    else:
        """Apenas corrigir espessuras de 1mm, o resto fica igual"""
        row_dict=pd.DataFrame({'Patient':[patient],'Thickness':[thick]})
        new_thick=pd.concat([new_thick,row_dict],ignore_index=True)
        
    "Para 3D"    
#     else:    
#         ratio=pret_thick/thick #tentar espessura perto de 3mm
        
#         ratio=round(ratio)
#         if (ratio==0): ratio=1
#         dic[patient]=ratio
#         print('')
#         print('patient:', patient,' ratio:', ratio,'number slices: ',len(files))
#         print('real thickness:',thick , 'new thickness:',ratio*thick)
#         print('old slices (peri):', len(p), 'new slices(peri):', len(p)/ratio)
#         print('')
#         number_slices_peri[patient]=len(p)/ratio
#         row_dict=pd.DataFrame({'Patient':[patient],'Thickness':[ratio*thick]})
#         new_thick=pd.concat([new_thick,row_dict],ignore_index=True)
        
# """Fazer downsampling (Só pelo CSV)"""

# patients_d=dic.keys()
# slices_peri_after_down={}

# for patient_d in patients_d:
    
#     ratio_d=dic[patient_d]
#     p_all=osic_all_df.loc[(osic_all_df['Patient']==patient_d)]
    
#     first_indice=p_all.index[0]
#     indices=[first_indice+i*ratio_d for i in range(len(p_all))]
    
#     for indice in indices:
#         try:
#           df_downs= p_all.loc[indice].to_frame().T
#           osic_3d=pd.concat([osic_3d,df_downs])  
#           print(p_all.loc[indice])
#         except:
#           print("Indice:", indice, "não existe no paciente ",patient_d)
          
#     "Verificar slices do pericardio depois do downsampling"
#     p_3d=osic_3d.loc[(osic_3d['Patient']==patient_d) & (osic_3d['Label']==1)] 
#     slices_peri_after_down[patient_d]=len(p_3d)    
    
        
#"""Fazer downsampling (Só pelo CSV)"""

# patients_d=dic.keys()
# slices_peri_after_down={}

# for patient_d in patients_d:
    
#     ratio_d=dic[patient_d]
#     p_all=osic_all_df.loc[(osic_all_df['Patient']==patient_d)]
    
#     first_indice=p_all.index[0]
#     indices=[first_indice+i*ratio_d for i in range(len(p_all))]
    
#     for indice in indices:
#         try:
#           df_downs= p_all.loc[indice].to_frame().T
#           osic_3d=pd.concat([osic_3d,df_downs])  
#           print(p_all.loc[indice])
#         except:
#           print("Indice:", indice, "não existe no paciente ",patient_d)
          
#     "Verificar slices do pericardio depois do downsampling"
#     p_3d=osic_3d.loc[(osic_3d['Patient']==patient_d) & (osic_3d['Label']==1)] 
#     slices_peri_after_down[patient_d]=len(p_3d)
    
    
# """Ver se está tudo bem """    

# import time

# patients = np.unique(osic_all_df['Patient'])
# for patient in patients_d:
#     pat = osic_3d.loc[(osic_3d['Patient']==patient)]
    
#     files=pat['Path_image']
#     for file in files:
#         print (file)

#     time.sleep(10)  # wait for 5 seconds
    
#save the DataFrame as a CSV file
#osic_3d.to_csv('OSIC_3D_test_set.csv', index=False)    
#new_thick.to_csv('OSIC_test_set_thickness.csv', index=False)