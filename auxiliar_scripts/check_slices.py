# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 22:43:44 2023

@author: RubenSilva
"""

import os
import glob
import pydicom
import pandas as pd
import numpy as np
path_osic="X:/Ruben/TESE/Data/Dataset_public/Orcya/orcic/"

test_hospit_df= pd.read_csv('X:/Ruben/TESE/Data/hospital_gaia/all_data_carolina/all_data_carolina_hospital_1.csv')
test_peri_hospit=test_hospit_df.loc[(test_hospit_df['Label']==1)]

hospit_3d=pd.DataFrame()

dic={}
number_slices_peri={}

i=0
patients=np.unique(test_peri_hospit['Patient'])
for patient in patients:
    p=test_hospit_df.loc[(test_hospit_df['Patient']==patient)]
    dic[patient]=len(p)
    if len(p)>=64:
        print(patient)