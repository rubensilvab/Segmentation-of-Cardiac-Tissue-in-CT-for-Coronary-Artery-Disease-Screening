# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 12:02:19 2022

@author: RubenSilva
"""

# Alex Weston
# Digital Innovation Lab, Mayo Clinic
import shutil
import os
import pydicom # pydicom is using the gdcm package for decompression

def clean_text(string):
    # clean and standardize text descriptions, which makes searching files easier
    forbidden_symbols = ["*", ".", ",", "\"", "\\", "/", "|", "[", "]", ":", ";", " "]
    for symbol in forbidden_symbols:
        string = string.replace(symbol, "_") # replace everything with an underscore
    return string.lower()  
   
# user specified parameters
src = "X:/Ruben/TESE/Data/hospital_gaia/EPICHEART_Carolina"
dst = "X:/Ruben/TESE/Data/sort_dicom/carolina/teste"

print('reading file list...')
unsortedList = []
for root, dirs, files in os.walk(src):
    for file in files: 
        print(file)
        if ".dcm" in file:# exclude non-dicoms, good for messy folders
            unsortedList.append(os.path.join(root, file))

print('%s files found.' % len(unsortedList))
       
for dicom_loc in unsortedList:
    # read the file
    ds = pydicom.read_file(dicom_loc, force=True)
   
    # get patient, study, and series information
    patientID = clean_text(ds.get("PatientID", "NA"))
    studyDate = clean_text(ds.get("StudyDate", "NA"))
    studyDescription = clean_text(ds.get("StudyDescription", "NA"))
    seriesDescription = clean_text(ds.get("SeriesDescription", "NA"))
   
    # generate new, standardized file name
    modality = ds.get("Modality","NA")
    studyInstanceUID = ds.get("StudyInstanceUID","NA")
    seriesInstanceUID = ds.get("SeriesInstanceUID","NA")
    
    instanceNumber = str(ds.get("InstanceNumber","0"))
    fileName = modality + "." + seriesInstanceUID + "." +"{:04d}".format(int(instanceNumber))+".dcm"
    print(fileName)   
    # uncompress files (using the gdcm package)
    try:
        ds.decompress()
    except:
        print('an instance in file %s - %s - %s - %s" could not be decompressed. exiting.' % (patientID, studyDate, studyDescription, seriesDescription ))
   
    # save files to a 4-tier nested folder structure
    if not os.path.exists(os.path.join(dst, patientID)):
        os.makedirs(os.path.join(dst, patientID))
   
        print('Saving out file: %s - %s - %s - %s.' % (patientID, studyDate, studyDescription, seriesDescription ))
       
    ds.save_as(os.path.join(dst, patientID, fileName))

print('done.')

"""OUTRA PARTE PARA REORDENAR ORDEM(INVERTER CASO NECESSÁRIO)"""

import glob

""" Esta função irá copiar os pacientes que tiverem em ordem invertida e corrigir, para uma pasta que nós queiramos"""

def reorder_by_slide_position(path,patient,path_to_copyy):
    patients_prob=[]
    files=sorted(glob.glob(path+'/'+patient+'/DICOM/*'))
    total_slices=len(files)
    # ver se o .0001.dcm e o ultimo.dcm,tem positions dos slides corretas
    ds1 = pydicom.read_file(files[0], force=True)
    ds_l = pydicom.read_file(files[-1], force=True)
    
    try:
        slice_location_01=float((ds1.get("ImagePositionPatient"))[2])
        slice_location_last=float((ds_l.get("ImagePositionPatient"))[2])
    
        
        if  slice_location_01 < slice_location_last:# está invertido
              
            print("Está invertido:",patient)
            i=0
            for file in files:
            
                new_order=total_slices-i
                name_file=(file.split("\\"))[1] #extrair apenas nome do ficheiro
                
                new_filename=name_file[:-9]+"."+"{:04d}".format(int(new_order))+".dcm"
               
                path_to_copy=os.path.join(path_to_copyy,str(patient))
                isExist = os.path.exists(path_to_copy)
                
                if not isExist:                         
                    # Create a new directory because it does not exist 
                    os.makedirs(path_to_copy)
                os.chdir(path_to_copy) 
                
                shutil.copy(file, os.path.join(path_to_copy,new_filename))
                print(file,"new_file: ",os.path.join(path_to_copy,new_filename))
                
                i=i+1  
        else:
            print("Não está invertido:", patient)
    except:
        print("problem to read", patient)
        patients_prob.append(patient)
        
    return patients_prob

import shutil            
path = "X:/Ruben/TESE/Data/hospital_gaia/EPICHEART_Carolina/selection"
path_to_copyy="X:/Ruben/TESE/Data/sort_dicom/correct_inverted/carol"
list_patients=sorted(os.listdir(path+'/'))
#list_patients=["id00032637202181710233084"]
patients_probb=[]
for patient in list_patients:
    patients_prob=reorder_by_slide_position(path,patient,path_to_copyy) 
    patients_probb.append(patients_prob)
print("pacientes com alguma falta de informação dos slices: ",patients_probb)
  