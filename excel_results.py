# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 15:59:51 2022

@author: RubenSilva
"""

# import xlsxwriter module
import xlsxwriter

import os
import glob
import numpy as np

import cv2
 
"""Função para fazer predict de um conjunto de imagens"""

def predict(model,X,Width,Length):
  prediction=[]
  for i in range(len(X)):
    pred=model.predict(X[i].reshape(1,Width,Length,1),verbose=0)
    pred=(pred>0.5).astype(np.uint8)
    prediction.append(pred)
  return np.array(prediction)

def single_dice_coef(y_true, y_pred_bin):
    # shape of y_true and y_pred_bin: (height, width)
    intersection = np.sum(y_true * y_pred_bin)
    if (np.sum(y_true)==0) and (np.sum(y_pred_bin)==0):
        return 1
    return round((2*intersection) / (np.sum(y_true) + np.sum(y_pred_bin)),3)



def excel_results(model,X,Y,path,name,patients):

    os.chdir(path)
    
    workbook = xlsxwriter.Workbook(str(name)+'.xlsx')
     
    # By default worksheet names in the spreadsheet will be
    # Sheet1, Sheet2 etc., but we can also specify a name.
    worksheet = workbook.add_worksheet("Results 256")
     
    # Some data we want to write to the worksheet.

    
    # Start from the first cell. Rows and
    # columns are zero indexed.
    row = 0
    col = 0
    worksheet.write(row, col,"patient")
    worksheet.write(row, col + 1,"mean")
    worksheet.write(row, col + 2,"std")
     
    # Iterate over the data and write it out row by row.
    for patient, i in zip(patients,range(len(patients))):
        row += 1
        
        
        pred_test=predict(model,X[i])
        #mean_dices,std_dices=mean_dice(Y[i],np.squeeze(pred_test))
        dice=single_dice_coef(Y[i],np.squeeze(pred_test))
        worksheet.write(row, col, patient)
        worksheet.write(row, col + 1,dice)
        #worksheet.write(row, col + 2,std_dices)
        print("Faltam os resulatdos para ",len(patients)-i-1," patients")
          
    workbook.close()