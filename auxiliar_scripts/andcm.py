# -*- coding: utf-8 -*-
"""
Created on Thu May 18 16:30:30 2023

@author: RubenSilva
"""


import nrrd
import matplotlib.pyplot as plt
import pydicom 
import numpy as np
from scipy.spatial import ConvexHull
from PIL import Image, ImageDraw

import os
import glob
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import save_img
from tensorflow.keras.preprocessing.image import img_to_array
import cv2
import numpy as np

path_dcm="X:/Ruben/TESE/Data/Dataset_public/RioFatSegm/RioFatSegm/Dicom _ Treino/"

path_p="X:/Ruben/TESE/Data/Dataset_public/Orcya/img_png/Dicom-1000_1000"

patients=sorted(os.listdir(path_dcm))
dic={}
val=[]
for patient in patients:
    #patient=str(patient).upper()
    files=sorted(glob.glob(path_dcm+patient+'/*'))
    data = pydicom.read_file(files[0])
    pix_spacing= data.get("PixelSpacing")
    #manu=data.get('Columns')
    print(pix_spacing)
    dic[patient]=pix_spacing
    val.append(pix_spacing[0])
    
 
print(np.mean(val))   