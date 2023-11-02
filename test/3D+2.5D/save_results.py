# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 22:35:34 2023

@author: RubenSilva
"""

# import xlsxwriter module
import xlsxwriter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import glob
import numpy as np
import shutil
import openpyxl
import cv2
import pydicom

from TestGenerator_3D import *
from TestGenerator_25d import *
 
"""Função para fazer predict de um conjunto de imagens"""
def single_dice_coef(y_true, y_pred_bin):
    # shape of y_true and y_pred_bin: (height, width)
    intersection = np.sum(y_true * y_pred_bin)
    if (np.sum(y_true)==0) and (np.sum(y_pred_bin)==0):
        return 1
    return round((2*intersection) / (np.sum(y_true) + np.sum(y_pred_bin)),3)

def predict_3D(model,X,Width,Length):
  prediction=[]
      
  pred=model.predict(X.reshape(1,Width,Width,Width,1),verbose=0)
  "Reshape para calcular direito MAD e HD"
  pred=np.squeeze(pred)
  # if resize:
  #       woriginal,loriginal=X_original[i].shape[0],X_original[i].shape[1]
  #       pred=cv2.resize(pred, (woriginal,loriginal))
        
  pred=(pred>0.5).astype(np.uint8)
  prediction.append(pred)
  return np.array(prediction) 

def predict_25d(model,X,X_original,Width,Length):
  prediction=[]
  for i in range(len(X)):
    pred=model.predict(X[i].reshape(1,Width,Length,3),verbose=0)
    "Reshape para calcular direito MAD e HD"
    pred=np.squeeze(pred)
    woriginal,loriginal=X_original[i].shape[0],X_original[i].shape[1]
    pred=cv2.resize(pred, (woriginal,loriginal))
    pred=(pred>0.5).astype(np.uint8)
    prediction.append(pred)
     
  return np.array(prediction)

def generators(datagen_image):
    for batch_image in datagen_image:
        yield batch_image

def save_img_results(path_to,patient,X,Y,pred_test):
    
        
       """Path to go predicts"""
       
       path=os.path.join(path_to,str(patient))
       isExist = os.path.exists(path)
       #print(path_to_cpy,isExist)
       
       if not isExist:                         
           # Create a new directory because it does not exist 
           os.makedirs(path)
       os.chdir(path)   
       s=0
       print("A guardar imagem para o paciente:",str(patient))
      
       for i in range (Y.shape[0]):
           
         s=s+1 
         fig=plt.figure(figsize=(16,6))
         fig.suptitle('Dice:'+str(round(single_dice_coef(np.squeeze(Y[i]), np.squeeze(pred_test[i])),3)))
         plt.subplot(1,3,1)
         plt.imshow(np.squeeze(X[i]),cmap='gray')
         plt.title('Original Teste_'+str(s))
         plt.subplot(1,3,2)
         plt.imshow(np.squeeze(Y[i]),cmap='gray')
         plt.title('label Test_'+str(s))
         plt.subplot(1,3,3)
         plt.imshow(np.squeeze(pred_test[i]),cmap='gray')
         plt.title('Predict_'+str(s))
         fig.savefig('Predicts_test_'+str(patient)+"_"+str(s)+'.jpg')
         plt.close('all')
               
def get_excel_row(path,patient,name,Y,Y_pred):
    
    isExist = os.path.exists(path)
    if not isExist:                         
        # Create a new directory because it does not exist 
        os.makedirs(path)
        
    os.chdir(path) 
    
    print('A calcular a dice do paciente: ',patient)
        
    filename = str(name)+'.xlsx'
    
    # Check if the file already exists
    if not os.path.isfile(filename):
        
        """Load do excel modelo"""
        file_excel="X:/Ruben/TESE/New_training_Unet/Models_only_pub_dset/Mix_datasets/excel_modelo.xlsx"
        
        isExist = os.path.exists(path)
        
        if not isExist:                         
            # Create a new directory because it does not exist 
            os.makedirs(path)
           
        
        shutil.copy(file_excel, path)
        old_name=os.path.join(path,"excel_modelo.xlsx")
        new_name=os.path.join(path,filename)
        os.rename(old_name,new_name)
        
        """Overwrite"""
        os.chdir(path)
        book = openpyxl.load_workbook(filename)
    
        # Select the worksheet to add data to
        sheet = book.active
    
        # Add a header row to the worksheet
        sheet.append(['Patient', 'Dice 3D'])
    else:
        # Open an existing workbook
        book = openpyxl.load_workbook(filename)
    
        # Select the worksheet to add data to
        sheet = book.active
    
    # Append a new row of data to the worksheet
    dice=single_dice_coef(Y,np.squeeze(Y_pred))
    sheet.append([patient, dice])
    #print([patient, slic, label, pred])
    # Save the workbook to a file
    book.save(filename)
         


"""Convert to NRRD to view in 3Dslicer"""

import cc3d

def reshape_nrrd(nrrd,n):
  if n==1:  
      nrrd=nrrd[::-1] 
  nrrd=np.transpose(nrrd,(2,1,0))  
 
  return nrrd  

"Funçoes pos processamento"

def fill_3d(labels_out):
    mask_convex=np.zeros(labels_out.shape)
    for i in range(labels_out.shape[0]):
        if( i==0 or i==labels_out.shape[0]-1):
            mask_convex[i,:,:]=labels_out[i,:,:]
        else:
            mask_convex[i,:,:]=np.logical_or(labels_out[i,:,:],np.logical_and(labels_out[i-1,:,:], labels_out[i+1,:,:]))
    return mask_convex.astype(np.uint8)

def connected_components(pred_test):
     # Get a labeling of the k largest objects in the image.
     # The output will be relabeled from 1 to N.
     labels_out, N = cc3d.largest_k(
       pred_test, k=1, 
       connectivity=6, delta=0,
       return_N=True,
     )
    
     labels_out=labels_out.astype(np.uint8)
     
     return labels_out
 
import cv2
import skimage.morphology, skimage.data

def fill_holes(labels_out):
    mask_fill=labels_out.copy()
    filled_img = np.zeros_like(mask_fill)
    for i in range(labels_out.shape[0]):
      # Find contours
        contours, _ = cv2.findContours(mask_fill[i], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw contours on black image
        contour_img = np.zeros_like(mask_fill[i])
        cv2.drawContours(contour_img, contours, -1, 1, 1)
        
        # Fill enclosed regions
        
        for contour in contours:
            cv2.fillPoly(filled_img[i], [contour], 1)
          
    return filled_img.astype(np.uint8)


from scipy.spatial import ConvexHull

from PIL import Image, ImageDraw

def convex_hull_image(data):
    w,l=data.shape[0],data.shape[1]
    region = np.argwhere(data)
    try:   
        hull = ConvexHull(region)
        verts = [(region[v,0], region[v,1]) for v in hull.vertices]
        img = Image.new('L', data.shape, 0)
        ImageDraw.Draw(img).polygon(verts, outline=1, fill=1)
        mask = np.array(img)
    except:    
        mask=np.zeros((w,l))
    return mask.T

def convex_mask(labels_out):
    mask_convex=np.zeros(labels_out.shape)
    for i in range(labels_out.shape[0]):
            mask_convex[i,:,:]=convex_hull_image(labels_out[i,:,:])
    return mask_convex.astype(np.uint8)


def pos_process(pred):
    connected=connected_components(pred)   
    
    convex_predict=fill_3d(connected)
    fill_2d_predict=fill_3d(connected)
    
    convex_predict=convex_mask(convex_predict)
    fill_2d_predict=fill_holes(fill_2d_predict)
    
   
    return convex_predict,fill_2d_predict
     
import nrrd
from collections import OrderedDict

"Processo de integrar 3D com 2D"

def get_excel_prediction(slices,prediction):
    """prediction=(64,64,64)"""
    csv_w_pred=slices.copy()
    pred=[]
    #print(prediction.shape)
    for i in range(prediction.shape[0]):
        if np.sum(prediction[i])>0:
            pred.append(1)
        else:
            pred.append(0)
    
    csv_w_pred['Pred']=pred
            
    return csv_w_pred      

def slices_add(csv,indexes):
     image_path=csv['Path_image']
     mask_path=csv['Path_Mask']
     X=[]
     Y=[]
     Y_pred=[]
     #print(indexes)
     for i in indexes:
             img_path = image_path.loc[i]
             #print('images_before:',img_path)
             img = cv2.imread(img_path, flags=cv2.IMREAD_ANYDEPTH)
             #print(img.shape)
             w,l=img.shape
             
             X.append(np.array(img))
             Y_pred.append(np.zeros((w,l)))
             
             #Load mask
             msk_path=mask_path.loc[i]
             #print('maks_before:',msk_path)
             mask=cv2.imread(msk_path, flags=cv2.IMREAD_ANYDEPTH)
             mask=mask/255.
            
             #Binarize masks
             thresh = 0.5 # Threshold at 0.5
             mask = np.where(mask > thresh, 1, 0).astype(np.uint8)
             Y.append(np.array(mask))
     #print(X)         
     #return np.concatenate(X, axis=0),np.concatenate(Y_pred, axis=0),np.concatenate(Y, axis=0) 
     return X,Y_pred,Y
 
def organize_data(csv_original_file,csv_25d,X25d,Y25d,Y):
   # Get the first common index
    common_index = set(csv_original_file.index).intersection(csv_25d.index)
    #print(common_index)
    first_common_index = min(common_index)
    last_common_index=max(common_index)
    #print(common_index)
    first_index_original=csv_original_file.index[0]
    n_to_add_before=first_common_index-first_index_original 
    last_index_original=csv_original_file.index[-1]  
    n_to_add_after=last_index_original-last_common_index 
    
    indexes_before=[first_index_original+i for i in range(n_to_add_before)]
    indexes_after=[last_common_index+i+1 for i in range(n_to_add_after)]
    
    X_before,Y_pred_before,Y_before=slices_add(csv_original_file,indexes_before)
    X_after,Y_pred_after,Y_after=slices_add(csv_original_file,indexes_after)
    
    
    
    if len(X_before)>0 and len(X_after)>0:
     X,Y_pred,Y=np.concatenate([X_before,X25d,X_after], axis=0),np.concatenate([Y_pred_before,Y25d,Y_pred_after], axis=0),np.concatenate([Y_before,Y,Y_after], axis=0)
    
    elif len(X_before)==0 and len(X_after)>0:
     X,Y_pred,Y=np.concatenate([X25d,X_after], axis=0),np.concatenate([Y25d,Y_pred_after], axis=0),np.concatenate([Y,Y_after], axis=0)
    
    elif len(X_before)>0 and len(X_after)==0:
     X,Y_pred,Y=np.concatenate([X_before,X25d], axis=0),np.concatenate([Y_pred_before,Y25d], axis=0),np.concatenate([Y_before,Y], axis=0)
    
    else:
     X,Y_pred,Y=X25d,Y25d,Y
    
        
    return X,Y,Y_pred

def convert_nrrd(test_dataframe,model3d,model25d,path,Width,Length,name_excel,change_hu=False):
    #Copy header
    if 'ospit' in path:
        print('entrou ceerto')
        PATH_header ='X:/Ruben/TESE/Data/hospital_gaia/EPICHEART_Carolina/selection'
        n=0
    elif 'ardiac' in path:
        
        PATH_header='X:/Ruben/TESE/Data/Dataset_public/RioFatSegm/RioFatSegm/Dicom _ Treino'
        n=1
    else:
        print('entrou errado')
        PATH_header='X:/Ruben/TESE/Data/Dataset_public/Orcya/nrrd_heart' 
        csv_thick=pd.read_csv('X:/Ruben/TESE/New_training_Unet/all_data/OSIC_new/OSIC_3D_test_set_thickness.csv')
        n=1
    
    patients=np.unique(test_dataframe['Patient'])
    
    for i in range(len(patients)):
        
        "CSV contendo os dados dos pacientes"
        
        csv_file=test_dataframe.loc[test_dataframe['Patient'].isin([(patients[i])])]
        
        "Buscar header original"
        if 'ospit' in path:
            PATH_header_nrrd =os.path.join(PATH_header, str(patients[i]), "segm_manual_Carolina.nrrd")
            header = nrrd.read_header(PATH_header_nrrd) 
            
        elif 'ardiac' in path:
            
            PATH_header_nrrd=os.path.join(PATH_header, str(patients[i]))
            files=sorted(glob.glob(PATH_header_nrrd+'/*'))
            # Read the data back from file
            slic=len(csv_file)-1
            data = pydicom.read_file(files[slic])
            pix_spacing= data.get("PixelSpacing")
            pix_spacing[0],pix_spacing[1]=float(pix_spacing[0]),float(pix_spacing[1])
            thick=data.get('SliceThickness')
            origin=data.get('ImagePositionPatient')
            origins=np.array([origin[0],origin[1],origin[2]])
            # print(thick)
            # if thick=='None':
            #     thick=data.get('SliceThickness')
            pix_spacing.append(thick)
            space_directions=np.diag(pix_spacing)
            
            PATH_header_copy ='X:/Ruben/TESE/Data/hospital_gaia/EPICHEART_Carolina/selection' 
            PATH_header_nrrd =os.path.join(PATH_header_copy, str(107780), "segm_manual_Carolina.nrrd")
            header=nrrd.read_header(PATH_header_nrrd)
            header['space directions']=space_directions
            header['space origin']=origins
            
            #print(header)
         
        else:  
            path_dcm="X:/Ruben/TESE/Data/Dataset_public/Orcya/orcic/"
            
            PATH_header_nrrd =os.path.join(PATH_header, str(patients[i]).upper()+"_heart.nrrd")
            header = nrrd.read_header(PATH_header_nrrd) 
            
            #Buscar informaçao dicom
            dicom_files= os.path.join(path_dcm,str(patients[i]))
            files=sorted(glob.glob(dicom_files+'/*'))
            # Read the data back from file
            data = pydicom.read_file(files[0])
            
            thick_p=csv_thick.loc[csv_thick['Patient'].isin([(patients[i])])]
            thick_p=thick_p['Thickness']
            #thick=data.get('SliceThickness')
            #print(thick_p)
            header['space directions'][2][2]=float(thick_p)
            header['space directions']=abs(header['space directions'])
        
        
        
        "Predicts and generator"
        
        input_col = 'Path_image'
        mask_col = 'Path_Mask'
        cols=[input_col,mask_col]
        
        batch_size=1
        data_generator=generators(CustomDataGenerator3D(csv_file,cols, batch_size=batch_size,input_shape=(64,64)))
        
        print("Falta a extraçao dos resultados para ",len(patients)-i," pacientes, atual: ",str(patients[i]))
       
        X_p=[]
        X_p_original=[]
        Y_p=[]
        
        
        X, Y, slices= next(data_generator)         
        X_p=np.squeeze(np.array(X)*65535)
        Y_p=np.squeeze(np.array(Y))
        
        "3D Prediction"
        #print(X_p_original.shape,Y_p.shape,np.unique(Y_p))        
        pred_test=predict_3D(model3d,X,64,64)
        pred_test=np.squeeze(pred_test)
        
        "Fazer CSV dos slices previstos com o nome dos slices e previsão se tem o pericárdio"
        csv_w_pred=get_excel_prediction(slices,pred_test)
        csv_to_25d=(csv_w_pred.loc[(csv_w_pred['Pred']==1)]).dropna()  
        
        if 'OSIC' in path:
            csv_files=pd.read_csv('X:/Ruben/TESE/New_training_Unet/all_data/OSIC_new/OSIC_new_sorted_5.csv')
            csv_file=csv_files.loc[csv_files['Patient'].isin([(patients[i])])]
            first_slc,last_slc=csv_to_25d['Path_Mask'].iloc[0],csv_to_25d['Path_Mask'].iloc[-1]
            
            first_index,last_index=csv_file[csv_file['Path_Mask'] == first_slc].index.tolist(),csv_file[csv_file['Path_Mask'] == last_slc].index.tolist()
            
            #csv_all_slices=
            csv_to_25d=csv_file.loc[first_index[0]:last_index[0]]
            
            
        #print(csv_to_25d)
        "Repetir processo para 2.5d"
        batch_size25d=12
        data_generator25d=generators(CustomDataGenerator25d(csv_to_25d,cols,batch_size=batch_size25d,input_shape=(Width,Length)))
        
        X_p25d=[]
        X_p_original25d=[]
        Y_p25d=[]
        
        while True:
            try:
                X25d, Y25d, X_original25d = next(data_generator25d)         
                X_p25d.append(np.array(X25d))
                X_p_original25d.append(np.array(X_original25d))
                Y_p25d.append(np.array(Y25d))
            
            except StopIteration:
             # stop the loop when the generator raises StopIteration
                     break  
        
        try: #se csv_to_25d tem mais de 2            
            X_p25d,X_p_original25d,Y_p25d=np.concatenate(X_p25d, axis=0),np.concatenate(X_p_original25d,axis=0),np.concatenate(Y_p25d,axis=0)
            
            #print(X_p_original.shape,Y_p.shape,np.unique(Y_p))        
            pred_test25d=predict_25d(model25d,X_p25d,X_p_original25d,Width,Length)
            pred_test25d=np.squeeze(pred_test25d)
            
            "voltar a colocar todos os slices por ordem do paciente (mesmo aqueles sem pericárdio)"
            
            X25D,Y25D,Y_pred_25d=organize_data(csv_file, csv_to_25d, X_p_original25d, pred_test25d, Y_p25d)
        
        except: #se csv_to_25d tem menos de 2 
            print(str(patients[i]),'não segmentou nada')
            indexes=csv_file.index
            X25D,Y_pred_25d,Y25D=slices_add(csv_file,indexes)
            X25D,Y_pred_25d,Y25D=np.concatenate([X25D], axis=0),np.concatenate([Y_pred_25d], axis=0),np.concatenate([Y25D], axis=0) 
            
        #print(X25D.shape,Y25D.shape,Y_pred_25d.shape)
        save_img_results(path,str(patients[i]),X25D,Y25D,Y_pred_25d)
        
        get_excel_row(path,str(patients[i]),name_excel,Y25D,Y_pred_25d)
        #print(pred_test.shape)
        
        #Connected components and fill slices:
        convex_predict,fill_2d_predict=pos_process(Y_pred_25d)  
        convex_predict,fill_2d_predict=convex_predict.astype(Y_pred_25d.dtype),fill_2d_predict.astype(Y_pred_25d.dtype)
        
        
        path_nrrd=os.path.join(path,'NRRD')
        path_to=os.path.join(path_nrrd,str(patients[i]))
        isExist = os.path.exists(path_to)
        if not isExist:                         
            # Create a new directory because it does not exist 
            os.makedirs(path_to)
            
        os.chdir(path_to)  
        
        
        nrrd.write(str(patients[i])+'_'+str(Width)+'.nrrd', reshape_nrrd(X25D,n),header=header)
        
        nrrd.write(str(patients[i])+'_'+str(Width)+'_3D2DNet'+'.nrrd', reshape_nrrd(Y_pred_25d,n),header=header)
    
        nrrd.write(str(patients[i])+'_'+str(Width)+'_manual'+'.nrrd', reshape_nrrd(Y25D,n),header=header)
        
        nrrd.write(str(patients[i])+'_'+str(Width)+'_3D2DNet+pp+conv'+'.nrrd', reshape_nrrd(convex_predict,n),header=header)

        #nrrd.write(str(patients[i])+'_'+str(Width)+'_3D2DNet+pp+fill2d'+'.nrrd', reshape_nrrd(fill_2d_predict,n),header=header)
       
       
 