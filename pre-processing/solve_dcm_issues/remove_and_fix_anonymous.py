# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 10:18:05 2022

@author: RubenSilva
"""
import os
import pydicom
import glob
import shutil

PATH ='X:/Ruben/TESE/Data/hospital_gaia/EPICHEART_Carolina/selection' 

"""Encontras pacientes com nome Anonymous"""

list_patients=sorted(os.listdir(PATH+'/'))
patients_anony=[]
for patient in list_patients:

    files=sorted(glob.glob(PATH+'/'+patient+'/DICOM/*'))
    #print(files)
    if "Anony" in os.path.basename(files[0]):
        #print(files[0])
        patients_anony.append(patient)

"""Pasta onde estão os anonimos corretamente"""

Path_corrected="X:/Ruben/TESE/Data/sort_dicom/carolina"
list_patients_sort=sorted(os.listdir(Path_corrected+'/'))
i=0
for patient in patients_anony:
    path_to=os.path.join(PATH,str(patient)) # originais
    files=sorted(glob.glob(path_to+'/DICOM/*'))
    number_patient=os.path.basename(files[0]).split('_')[1]
    #print("pasta original: ",number_patient)
    for patient_sort in list_patients_sort: # na pasta com os corrigidos
        if "anony" in patient_sort:
         number_patient_sort=patient_sort.split('_')[1]
         #print("pasta corrigida: ",number_patient_sort)
         if number_patient_sort==number_patient:
            
             path_sort=os.path.join(Path_corrected,str(patient_sort))
             files_sort=sorted(glob.glob(path_sort+'/*'))
             path_to_copy=os.path.join(path_to,'DICOM')
             print("vai copiar:",os.path.basename(files[0]),patient_sort)
             #Copiar os novos
             for file_sort in files_sort:
                 shutil.copy(file_sort, path_to_copy)
             #remover antigos
             for file in files:    
                 os.remove(file) 
             i +=1
             print("patients copiados: ",i)