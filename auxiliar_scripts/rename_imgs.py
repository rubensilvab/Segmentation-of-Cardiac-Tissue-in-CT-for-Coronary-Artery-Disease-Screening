# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 00:59:56 2022

@author: RubenSilva
"""

import os
os.getcwd()
collection = "F:/Ruben/TESE/New_training_Unet/output/test/images"
folders=os.listdir(collection)
for i, filename in enumerate(folders):
    print(i)
    print(filename)
    num=int(filename[3:6])+1
    print(num)
    os.rename("F:/Ruben/TESE/New_training_Unet/output/test/images/" + filename, "F:/Ruben/TESE/New_training_Unet/output/test/images/" + "img" +str(num+1)+ ".png")
    os.rename("F:/Ruben/TESE/New_training_Unet/output/test/mask/" + filename, "F:/Ruben/TESE/New_training_Unet/output/test/mask/" + "msk" +str(num+1)+ ".png")
  
    #train
collection = "F:/Ruben/TESE/New_training_Unet/output/train/images"
folders=os.listdir(collection)
for i, filename in enumerate(folders):
    print(i)
    print(filename)
    num=int(filename[0:3])+1
    print(num)
    os.rename("F:/Ruben/TESE/New_training_Unet/output/train/images/" + filename, "F:/Ruben/TESE/New_training_Unet/output/train/images/" + "img" +str(num)+ ".png")
    os.rename("F:/Ruben/TESE/New_training_Unet/output/train/mask/" + filename, "F:/Ruben/TESE/New_training_Unet/output/train/mask/" + "msk" +str(num)+ ".png")   
    
 #validation
collection = "F:/Ruben/TESE/New_training_Unet/output/val/images"
folders=os.listdir(collection)
for i, filename in enumerate(folders):
    print(i)
    print(filename)
    num=int(filename[0:3])+1
    print(num)
    os.rename("F:/Ruben/TESE/New_training_Unet/output/val/images/" + filename, "F:/Ruben/TESE/New_training_Unet/output/val/images/" + "img" +str(num)+ ".png")
    os.rename("F:/Ruben/TESE/New_training_Unet/output/val/mask/" + filename, "F:/Ruben/TESE/New_training_Unet/output/val/mask/" + "msk" +str(num)+ ".png")       