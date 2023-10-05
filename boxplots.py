# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 18:50:22 2023

@author: RubenSilva
"""

import os
#os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.6/bin")

import numpy as np 
from matplotlib import pyplot as plt
import numpy as np
import glob
import cv2
import time
import pandas as pd

"Import CSV with dicom and masks informations"

path_dice_cfat='C:/Users/RubenSilva/Desktop/Results/Cardiac Fat/2d/DICE'
path_bce_cfat='C:/Users/RubenSilva/Desktop/Results/Cardiac Fat/2d/BCE'
path_25d_cfat='C:/Users/RubenSilva/Desktop/Results/Cardiac Fat/2.5d'
path_3d_cfat='C:/Users/RubenSilva/Desktop/Results/Cardiac Fat/3d'
path_325d_cfat='C:/Users/RubenSilva/Desktop/Results/Cardiac Fat/3d_2.5d'
path_slc25d_cfat='C:/Users/RubenSilva/Desktop/Results/Cardiac Fat/slc_2.5d'

path_dice_osic='C:/Users/RubenSilva/Desktop/Results/OSIC/2d/DICE'
path_bce_osic='C:/Users/RubenSilva/Desktop/Results/OSIC/2d/BCE'
path_25d_osic='C:/Users/RubenSilva/Desktop/Results/OSIC/2.5d'
path_3d_osic='C:/Users/RubenSilva/Desktop/Results/OSIC/3d'
path_325d_osic='C:/Users/RubenSilva/Desktop/Results/OSIC/3d_2.5d'
path_slc25d_osic='C:/Users/RubenSilva/Desktop/Results/OSIC/slc_2.5d'


path_dice_hosp='C:/Users/RubenSilva/Desktop/Results/Hospital/2d/DICE'
path_bce_hosp='C:/Users/RubenSilva/Desktop/Results/Hospital/2d/BCE'
path_25d_hosp='C:/Users/RubenSilva/Desktop/Results/Hospital/2.5d'
path_3d_hosp='C:/Users/RubenSilva/Desktop/Results/Hospital/3d'
path_325d_hosp='C:/Users/RubenSilva/Desktop/Results/Hospital/3d_2.5d'
path_slc25d_hosp='C:/Users/RubenSilva/Desktop/Results/Hospital/slc_2.5d'

"CFat"

def_2000_dice_cfat= ((pd.read_excel(os.path.join(path_dice_cfat,'Analise_L0_W2000.xlsx'))).iloc[:,:5]).dropna() # Remove rows with any NaN values
def_2000_512_cfat=((pd.read_excel(os.path.join(path_dice_cfat,'512','Analise_L0_W2000.xlsx'))).iloc[:,:5]).dropna() 

def_2000_bce_cfat= ((pd.read_excel(os.path.join(path_bce_cfat,'Analise_L0_W2000_BCE.xlsx'))).iloc[:,:5]).dropna() 
def_350_bce_cfat=((pd.read_excel(os.path.join(path_bce_cfat,'Analise_L50_W350_BCE.xlsx'))).iloc[:,:5]).dropna() 

def_350_cfat= ((pd.read_excel(os.path.join(path_dice_cfat,'Analise_L50_W350.xlsx'))).iloc[:,:5]).dropna()
da_2000_cfat= ((pd.read_excel(os.path.join(path_dice_cfat,'Analise_L0_W2000_augm.xlsx'))).iloc[:,:5]).dropna()
da_350_cfat= ((pd.read_excel(os.path.join(path_dice_cfat,'Analise_L50_W350_augm.xlsx'))).iloc[:,:5]).dropna()
da_ca_2000_cfat= ((pd.read_excel(os.path.join(path_dice_cfat,'Analise_L0_W2000_calc_augm.xlsx'))).iloc[:,:5]).dropna()
da_ca_350_cfat= ((pd.read_excel(os.path.join(path_dice_cfat,'Analise_L50_W350_Calc_augm.xlsx'))).iloc[:,:5]).dropna()

#2.5d
def_2000_25d_cfat=((pd.read_excel(os.path.join(path_25d_cfat,'Analise_L0_W2000_Calc_augm_2.5d.xlsx'))).iloc[:,:5]).dropna()
def_2000_25d_conv_cfat=((pd.read_excel(os.path.join(path_25d_cfat,'Analise_L0_W2000_Calc_augm_2.5d_pp+conv.xlsx'))).iloc[:,:5]).dropna()
def_2000_25d_fill_cfat=((pd.read_excel(os.path.join(path_25d_cfat,'Analise_L0_W2000_Calc_augm_2.5d_pp+fill2d.xlsx'))).iloc[:,:5]).dropna()

def_350_25d_cfat=((pd.read_excel(os.path.join(path_25d_cfat,'Analise_L50_W350_Calc_augm_2.5d.xlsx'))).iloc[:,:5]).dropna()

#3d
def_2000_3d_cfat=((pd.read_excel(os.path.join(path_3d_cfat,'Analise_L0_W2000_Calc_augm_3d.xlsx'))).iloc[:,:5]).dropna()
def_2000_3d_conv_cfat=((pd.read_excel(os.path.join(path_3d_cfat,'Analise_L0_W2000_Calc_augm_3d+pp+conv.xlsx'))).iloc[:,:5]).dropna()

#3d+2.5d
def_2000_325d_cfat=((pd.read_excel(os.path.join(path_325d_cfat,'Analise_L0_W2000_Calc_augm_3d2.5d.xlsx'))).iloc[:,:5]).dropna()
def_2000_325d_conv_cfat=((pd.read_excel(os.path.join(path_325d_cfat,'Analise_L0_W2000_Calc_augm_3d2.5d+pp+conv.xlsx'))).iloc[:,:5]).dropna()

#slc+2.5d

def_2000_slc25d_cfat=((pd.read_excel(os.path.join(path_slc25d_cfat,'Analise_L0_W2000_Calc_augm_slc+2.5d.xlsx'))).iloc[:,:5]).dropna()
def_2000_slc25d_conv_cfat=((pd.read_excel(os.path.join(path_slc25d_cfat,'Analise_L0_W2000_Calc_augm_slc+2.5d+conv.xlsx'))).iloc[:,:5]).dropna()


"OSIC"

def_2000_dice_osic= ((pd.read_excel(os.path.join(path_dice_osic,'Analise_L0_W2000.xlsx'))).iloc[:,:5]).dropna() # Remove rows with any NaN values
def_2000_512_osic=((pd.read_excel(os.path.join(path_dice_osic,'512','Analise_L0_W2000.xlsx'))).iloc[:,:5]).dropna() 

def_2000_bce_osic= ((pd.read_excel(os.path.join(path_bce_osic,'Analise_L0_W2000_BCE.xlsx'))).iloc[:,:5]).dropna() 
def_350_bce_osic= ((pd.read_excel(os.path.join(path_bce_osic,'Analise_L50_W350_BCE.xlsx'))).iloc[:,:5]).dropna() 


def_350_osic= ((pd.read_excel(os.path.join(path_dice_osic,'Analise_L50_W350.xlsx'))).iloc[:,:5]).dropna()
da_2000_osic= ((pd.read_excel(os.path.join(path_dice_osic,'Analise_L0_W2000_augm.xlsx'))).iloc[:,:5]).dropna()
da_350_osic= ((pd.read_excel(os.path.join(path_dice_osic,'Analise_L50_W350_augm.xlsx'))).iloc[:,:5]).dropna()
da_ca_2000_osic= ((pd.read_excel(os.path.join(path_dice_osic,'Analise_L0_W2000_calc_augm.xlsx'))).iloc[:,:5]).dropna()
da_ca_350_osic= ((pd.read_excel(os.path.join(path_dice_osic,'Analise_L50_W350_Calc_augm.xlsx'))).iloc[:,:5]).dropna()

def_2000_25d_osic=((pd.read_excel(os.path.join(path_25d_osic,'Analise_L0_W2000_Calc_augm_2.5d.xlsx'))).iloc[:,:5]).dropna()
def_2000_25d_conv_osic=((pd.read_excel(os.path.join(path_25d_osic,'Analise_L0_W2000_Calc_augm_2.5d_pp+conv.xlsx'))).iloc[:,:5]).dropna()
def_2000_25d_fill_osic=((pd.read_excel(os.path.join(path_25d_osic,'Analise_L0_W2000_Calc_augm_2.5d_pp+fill2d.xlsx'))).iloc[:,:5]).dropna()

def_350_25d_osic=((pd.read_excel(os.path.join(path_25d_osic,'Analise_L50_W350_Calc_augm_2.5d.xlsx'))).iloc[:,:5]).dropna()

#3d
def_2000_3d_osic=((pd.read_excel(os.path.join(path_3d_osic,'Analise_L0_W2000_Calc_augm_3d.xlsx'))).iloc[:,:5]).dropna()
def_2000_3d_conv_osic=((pd.read_excel(os.path.join(path_3d_osic,'Analise_L0_W2000_Calc_augm_3d+pp+conv.xlsx'))).iloc[:,:5]).dropna()


def_2000_325d_osic=((pd.read_excel(os.path.join(path_325d_osic,'Analise_L0_W2000_Calc_augm_3d2.5d.xlsx'))).iloc[:,:5]).dropna()
def_2000_325d_conv_osic=((pd.read_excel(os.path.join(path_325d_osic,'Analise_L0_W2000_Calc_augm_3d2.5d+pp+conv.xlsx'))).iloc[:,:5]).dropna()

#slc+2.5d

def_2000_slc25d_osic=((pd.read_excel(os.path.join(path_slc25d_osic,'Analise_L0_W2000_Calc_augm_slc+2.5d.xlsx'))).iloc[:,:5]).dropna()
def_2000_slc25d_conv_osic=((pd.read_excel(os.path.join(path_slc25d_osic,'Analise_L0_W2000_Calc_augm_slc+2.5d+conv.xlsx'))).iloc[:,:5]).dropna()



"Hospital"

def_2000_dice_hosp= ((pd.read_excel(os.path.join(path_dice_hosp,'Analise_L0_W2000.xlsx'))).iloc[:,:5]).dropna() # Remove rows with any NaN values
def_2000_512_hosp=((pd.read_excel(os.path.join(path_dice_hosp,'512','Analise_L0_W2000.xlsx'))).iloc[:,:5]).dropna() 

def_2000_bce_hosp= ((pd.read_excel(os.path.join(path_bce_hosp,'Analise_L0_W2000_BCE.xlsx'))).iloc[:,:5]).dropna() 
def_350_bce_hosp= ((pd.read_excel(os.path.join(path_bce_hosp,'Analise_L50_W350_BCE.xlsx'))).iloc[:,:5]).dropna() 


def_350_hosp= ((pd.read_excel(os.path.join(path_dice_hosp,'Analise_L50_W350.xlsx'))).iloc[:,:5]).dropna()
da_2000_hosp= ((pd.read_excel(os.path.join(path_dice_hosp,'Analise_L0_W2000_augm.xlsx'))).iloc[:,:5]).dropna()
da_350_hosp= ((pd.read_excel(os.path.join(path_dice_hosp,'Analise_L50_W350_augm.xlsx'))).iloc[:,:5]).dropna()
da_ca_2000_hosp= ((pd.read_excel(os.path.join(path_dice_hosp,'Analise_L0_W2000_calc_augm.xlsx'))).iloc[:,:5]).dropna()
da_ca_350_hosp= ((pd.read_excel(os.path.join(path_dice_hosp,'Analise_L50_W350_Calc_augm.xlsx'))).iloc[:,:5]).dropna()

def_2000_25d_hosp=((pd.read_excel(os.path.join(path_25d_hosp,'Analise_L0_W2000_Calc_augm_2.5d.xlsx'))).iloc[:,:5]).dropna()
def_2000_25d_conv_hosp=((pd.read_excel(os.path.join(path_25d_hosp,'Analise_L0_W2000_calc_augm_2.5d _pp+conv.xlsx'))).iloc[:,:5]).dropna()
def_2000_25d_fill_hosp=((pd.read_excel(os.path.join(path_25d_hosp,'Analise_L0_W2000_Calc_augm_2.5d _pp+fill2d.xlsx'))).iloc[:,:5]).dropna()

def_350_25d_hosp=((pd.read_excel(os.path.join(path_25d_hosp,'Analise_L50_W350_Calc_augm_2.5d.xlsx'))).iloc[:,:5]).dropna()

#3d
def_2000_3d_hosp=((pd.read_excel(os.path.join(path_3d_hosp,'Analise_L0_W2000_Calc_augm_3d.xlsx'))).iloc[:,:5]).dropna()
def_2000_3d_conv_hosp=((pd.read_excel(os.path.join(path_3d_hosp,'Analise_L0_W2000_Calc_augm_3d+pp+conv.xlsx'))).iloc[:,:5]).dropna()

def_2000_325d_hosp=((pd.read_excel(os.path.join(path_325d_hosp,'Analise_L0_W2000_Calc_augm_3d2.5d.xlsx'))).iloc[:,:5]).dropna()
def_2000_325d_conv_hosp=((pd.read_excel(os.path.join(path_325d_hosp,'Analise_L0_W2000_Calc_augm_3d2.5d+pp+conv.xlsx'))).iloc[:,:5]).dropna()

#slc+2.5d

def_2000_slc25d_hosp=((pd.read_excel(os.path.join(path_slc25d_hosp,'Analise_L0_W2000_Calc_augm_slc+2.5d.xlsx'))).iloc[:,:5]).dropna()
def_2000_slc25d_conv_hosp=((pd.read_excel(os.path.join(path_slc25d_hosp,'Analise_L0_W2000_Calc_augm_slc+2.5d+conv.xlsx'))).iloc[:,:5]).dropna()




#name_dataset=path.split('/')[-3]+'_'+path.split('/')[-2]

import matplotlib.pyplot as plt

# Create a list of the metric column names
metric_columns = list(def_2000_dice_cfat.columns[1:])
metric_columns.remove('jaccard')
# Create subplots for each metric

#Create subplots for each metric
# fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(25, 25))

# "Modelo data augmentation, etc."
# # Iterate over each metric column
# for i, metric in enumerate(metric_columns):
    
#     #fig = plt.figure(figsize =(10, 7),dpi=500)
     
#     # Creating axes instance
#     #axes = fig.add_axes([0, 0, 1, 1])
#     # Plot box plots for the metric from da_350 on the left
#     positions = [0.5, 1, 2, 2.5, 3.5, 4.0]
#     box_width=0.3
    
#     axes[i].boxplot(def_2000_512_cfat[metric], positions=[positions[0]], widths=box_width,patch_artist=True,boxprops=dict(color='black',facecolor='lightblue'))

#     # Plot box plots for the metric from def_2000 on the right
#     axes[i].boxplot(def_2000_dice_cfat[metric], positions=[positions[1]], widths=box_width,patch_artist=True,boxprops=dict(color='black',facecolor='lightblue'))
    
#     # Plot box plots for the metric from def_2000 on the right
#     axes[i].boxplot(def_2000_512_osic[metric], positions=[positions[2]], widths=box_width,patch_artist=True,boxprops=dict(color='black',facecolor='lightblue'))
    
#     # Plot box plots for the metric from def_2000 on the right
#     axes[i].boxplot(def_2000_dice_osic[metric], positions=[positions[3]], widths=box_width,patch_artist=True,boxprops=dict(color='black',facecolor='lightblue'))
    
#     # Plot box plots for the metric from def_2000 on the right
#     axes[i].boxplot(def_2000_512_hosp[metric], positions=[positions[4]], widths=box_width,patch_artist=True,boxprops=dict(color='black',facecolor='lightblue'))

#     axes[i].boxplot(def_2000_dice_hosp[metric], positions=[positions[5]], widths=box_width,patch_artist=True,boxprops=dict(color='black',facecolor='lightblue'))


#     # Add horizontal gridlines
#     axes[i].grid(axis='y', linestyle='--', linewidth=0.5, color='gray')

# # Calculate the mean for each metric
#     boxp = [[def_2000_512_cfat[metric], def_2000_dice_cfat[metric]
#               ],
#               [ def_2000_512_osic[metric], def_2000_dice_osic[metric]  
#               ] ,
#               [ def_2000_512_hosp[metric], def_2000_dice_hosp[metric]      
#               ]     
#               ]
    
#     # Plot average line

# # Plot average points
#     means=[[def_2000_512_cfat[metric].mean(), def_2000_dice_cfat[metric].mean()
#               ],
#               [ def_2000_512_osic[metric].mean(), def_2000_dice_osic[metric].mean()  
#               ] ,
#               [ def_2000_512_hosp[metric].mean(), def_2000_dice_hosp[metric].mean()     
#               ]     
#               ]
#     points=[[0.5, 1],[ 2, 2.5],[ 3.5, 4.0]]

#     axes[i].plot(points, means[:],'o' ,color='black',markersize=6)
#     # Add average values as text above each boxplot
#     if metric=='dice':
#         for j, p in zip(points[0],boxp[0]):
#             mean=p.mean()
#             axes[i].text(j, max(p)+0.05*max(p), f'{mean:.3f}', color='black', ha='center', va='center', fontsize=20)
#         for j, p in zip(points[1],boxp[1]):
#             mean=p.mean()
#             axes[i].text(j, max(p)+0.05*max(p), f'{mean:.3f}', color='black', ha='center', va='center', fontsize=20)
#         for j, p in zip(points[2],boxp[2]):
#             mean=p.mean()
#             axes[i].text(j, max(p)+0.05*max(p), f'{mean:.3f}', color='black', ha='center', va='center', fontsize=20)

#         axes[i].set_ylim(ymax=1.05)


#     elif metric=='hd':   
#         for j, p in zip(points[0],boxp[0]):
#             mean=p.mean()
#             axes[i].text(j, max(p)+0.5*max(p), f'{mean:.3f}', color='black', ha='center', va='center', fontsize=20)
#         for j, p in zip(points[1],boxp[1]):
#             mean=p.mean()
#             axes[i].text(j, max(p)+0.1*max(p), f'{mean:.3f}', color='black', ha='center', va='center', fontsize=20)
#         for j, p in zip(points[2],boxp[2]):
#             mean=p.mean()
#             axes[i].text(j, max(p)+0.5*max(p), f'{mean:.3f}', color='black', ha='center', va='center', fontsize=20)
#         axes[i].set_ylim(ymax=300)

#     else:
#         for j, p in zip(points[0],boxp[0]):
#             mean=p.mean()
#             axes[i].text(j, max(p)+0.2*max(p), f'{mean:.3f}', color='black', ha='center', va='center', fontsize=20)
#         for j, p in zip(points[1],boxp[1]):
#             mean=p.mean()
#             axes[i].text(j, max(p)+0.1*max(p), f'{mean:.3f}', color='black', ha='center', va='center', fontsize=20)
#         for j, p in zip(points[2],boxp[2]):
#             mean=p.mean()
#             axes[i].text(j, max(p)+0.2*max(p), f'{mean:.3f}', color='black', ha='center', va='center', fontsize=20)
#         axes[i].set_ylim(ymax=23)

#     # Set x-axis labels
#     axes[i].set_xticks([0.5, 1, 2, 2.5, 3.5, 4.0])
#     axes[i].set_xticklabels(['512x512', '256x256','512x512', '256x256','512x512', '256x256'], fontsize=20, fontweight='bold')
    
#     labels_dt=[0.167, 0.5, 0.832]
#     labels=['Cardiac Fat','OSIC','CHVNGE']
#     y_max=axes[i].get_ylim()[1]
#     for j, label in enumerate(labels):
#         axes[i].text(labels_dt[j], 1.03, label, color='black', ha='center', va='center',fontweight='bold', fontsize=28, transform=axes[i].transAxes)

#     # Set the x-axis tick positions and labels at the top
    
#     if metric !='dice':
#         axes[i].set_ylabel(metric.upper()+'(mm)',size=25,fontweight='bold')
#     else:
#         metric='DSC'
#         axes[i].set_ylabel(metric,size=25,fontweight='bold')
    
#     # Set x-axis labels
#     axes[i].tick_params(axis='both', labelsize=20)

"2D vs 2.5D"

#Create subplots for each metric
# fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(25, 25))

# "Modelo data augmentation, etc."
# # Iterate over each metric column
# for i, metric in enumerate(metric_columns):
    
#     #fig = plt.figure(figsize =(10, 7),dpi=500)
     
#     # Creating axes instance
#     #axes = fig.add_axes([0, 0, 1, 1])
#     # Plot box plots for the metric from da_350 on the left
#     positions = [0.5, 1, 2, 2.5, 3.5, 4.0]
#     box_width=0.3
    
#     axes[i].boxplot(da_ca_2000_cfat[metric], positions=[positions[0]], widths=box_width,patch_artist=True,boxprops=dict(color='black',facecolor='lightblue'))

#     # Plot box plots for the metric from def_2000 on the right
#     axes[i].boxplot(def_2000_25d_cfat[metric], positions=[positions[1]], widths=box_width,patch_artist=True,boxprops=dict(color='black',facecolor='lightblue'))
    
#     # Plot box plots for the metric from def_2000 on the right
#     axes[i].boxplot(da_ca_2000_osic[metric], positions=[positions[2]], widths=box_width,patch_artist=True,boxprops=dict(color='black',facecolor='lightblue'))
    
#     # Plot box plots for the metric from def_2000 on the right
#     axes[i].boxplot(def_2000_25d_osic[metric], positions=[positions[3]], widths=box_width,patch_artist=True,boxprops=dict(color='black',facecolor='lightblue'))
    
#     # Plot box plots for the metric from def_2000 on the right
#     axes[i].boxplot(da_ca_2000_hosp[metric], positions=[positions[4]], widths=box_width,patch_artist=True,boxprops=dict(color='black',facecolor='lightblue'))

#     axes[i].boxplot(def_2000_25d_hosp[metric], positions=[positions[5]], widths=box_width,patch_artist=True,boxprops=dict(color='black',facecolor='lightblue'))


#     # Add horizontal gridlines
#     axes[i].grid(axis='y', linestyle='--', linewidth=0.5, color='gray')

# # Calculate the mean for each metric
#     boxp = [[da_ca_2000_cfat[metric], def_2000_25d_cfat[metric]
#               ],
#               [ da_ca_2000_osic[metric], def_2000_25d_osic[metric]  
#               ] ,
#               [ da_ca_2000_hosp[metric], def_2000_25d_hosp[metric]      
#               ]     
#               ]
    
#     # Plot average line

# # Plot average points
#     means=[[da_ca_2000_cfat[metric].mean(), def_2000_25d_cfat[metric].mean()
#               ],
#               [ da_ca_2000_osic[metric].mean(), def_2000_25d_osic[metric].mean()  
#               ] ,
#               [ da_ca_2000_hosp[metric].mean(), def_2000_25d_hosp[metric].mean()     
#               ]     
#               ]
#     points=[[0.5, 1],[ 2, 2.5],[ 3.5, 4.0]]

#     axes[i].plot(points, means[:],'o' ,color='black',markersize=6)
#     # Add average values as text above each boxplot
#     if metric=='dice':
#         for j, p in zip(points[0],boxp[0]):
#             mean=p.mean()
#             axes[i].text(j, max(p)+0.05*max(p), f'{mean:.3f}', color='black', ha='center', va='center', fontsize=20)
#         for j, p in zip(points[1],boxp[1]):
#             mean=p.mean()
#             axes[i].text(j, max(p)+0.05*max(p), f'{mean:.3f}', color='black', ha='center', va='center', fontsize=20)
#         for j, p in zip(points[2],boxp[2]):
#             mean=p.mean()
#             axes[i].text(j, max(p)+0.05*max(p), f'{mean:.3f}', color='black', ha='center', va='center', fontsize=20)

#         axes[i].set_ylim(ymax=1.05)


#     elif metric=='hd':   
#         for j, p in zip(points[0],boxp[0]):
#             mean=p.mean()
#             axes[i].text(j, max(p)+0.5*max(p), f'{mean:.3f}', color='black', ha='center', va='center', fontsize=20)
#         for j, p in zip(points[1],boxp[1]):
#             mean=p.mean()
#             axes[i].text(j, mean+2.2*mean, f'{mean:.3f}', color='black', ha='center', va='center', fontsize=20)
#             print(j,max(p))
#         for j, p in zip(points[2],boxp[2]):
#             mean=p.mean()
#             axes[i].text(j, max(p)+0.5*max(p), f'{mean:.3f}', color='black', ha='center', va='center', fontsize=20)
#         axes[i].set_ylim(ymax=400)

#     else:
#         for j, p in zip(points[0],boxp[0]):
#             mean=p.mean()
#             axes[i].text(j, max(p)+0.2*max(p), f'{mean:.3f}', color='black', ha='center', va='center', fontsize=20)
#         for j, p in zip(points[1],boxp[1]):
#             mean=p.mean()
#             axes[i].text(j, max(p)+0.06*max(p), f'{mean:.3f}', color='black', ha='center', va='center', fontsize=20)
#         for j, p in zip(points[2],boxp[2]):
#             mean=p.mean()
#             axes[i].text(j, max(p)+0.2*max(p), f'{mean:.3f}', color='black', ha='center', va='center', fontsize=20)
#         axes[i].set_ylim(ymax=23)

#     # Set x-axis labels
#     axes[i].set_xticks([0.5, 1, 2, 2.5, 3.5, 4.0])
#     axes[i].set_xticklabels(['2D', '2.5D','2D', '2.5D','2D', '2.5D'], fontsize=20, fontweight='bold')
    
#     labels_dt=[0.167, 0.5, 0.832]
#     labels=['Cardiac Fat','OSIC','CHVNGE']
#     y_max=axes[i].get_ylim()[1]
#     for j, label in enumerate(labels):
#         axes[i].text(labels_dt[j], 1.03, label, color='black', ha='center', va='center',fontweight='bold', fontsize=28, transform=axes[i].transAxes)

#     # Set the x-axis tick positions and labels at the top
    
#     if metric !='dice':
#         axes[i].set_ylabel(metric.upper()+'(mm)',size=25,fontweight='bold')
#     else:
#         metric='DSC'
#         axes[i].set_ylabel(metric,size=25,fontweight='bold')
    
#     # Set x-axis labels
#     axes[i].tick_params(axis='both', labelsize=20)

"2d, 2.5d e 3d "

# #Create subplots for each metric
# fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(25, 25))


# # Iterate over each metric column
# for i, metric in enumerate(metric_columns):
    
#     #fig = plt.figure(figsize =(10, 7),dpi=500)
     
#     # Creating axes instance
#     #axes = fig.add_axes([0, 0, 1, 1])
#     # Plot box plots for the metric from da_350 on the left
#     positions=[0.5, 1, 1.5,2,3,3.5,4,4.5,5.5,6,6.5,7]
#     box_width=0.3
    
#     #axes[i].boxplot(da_ca_2000_cfat[metric], positions=[positions[0]],patch_artist=True,boxprops=dict(color='black',facecolor='lightblue'))

#     # Plot box plots for the metric from def_2000 on the right
#     axes[i].boxplot(def_2000_25d_cfat[metric], positions=[positions[0]],widths=box_width,patch_artist=True,boxprops=dict(color='black',facecolor='lightblue'))
    
#     # Plot box plots for the metric from def_2000 on the right
#     #axes[i].boxplot(def_2000_3d_cfat[metric], positions=[positions[2]],patch_artist=True,boxprops=dict(color='black',facecolor='lightblue'))
#     axes[i].boxplot(def_2000_25d_conv_cfat[metric], positions=[positions[1]],widths=box_width,patch_artist=True,boxprops=dict(color='black',facecolor='lightblue'))
    
#     #axes[i].boxplot(def_2000_slc25d_cfat[metric], positions=[positions[1]],widths=box_width,patch_artist=True,boxprops=dict(color='black',facecolor='lightblue'))
#     axes[i].boxplot(def_2000_325d_conv_cfat[metric], positions=[positions[3]],widths=box_width,patch_artist=True,boxprops=dict(color='black',facecolor='lightblue'))
    

#     # Plot box plots for the metric from def_2000 on the right
#     axes[i].boxplot(def_2000_325d_cfat[metric], positions=[positions[2]],widths=box_width,patch_artist=True,boxprops=dict(color='black',facecolor='lightblue'))
    
    
#     # Plot box plots for the metric from def_2000 on the right
    
#     # Plot box plots for the metric from def_2000 on the right
#     #axes[i].boxplot(def_2000_25d_fill_cfat[metric], positions=[3],patch_artist=True,boxprops=dict(color='black',facecolor='lightblue'))
#     #axes[i].boxplot(def_2000_slc25d_conv_cfat[metric], positions=[positions[6]],patch_artist=True,boxprops=dict(color='black',facecolor='lightblue'))
    
    
#     #axes[i].boxplot(def_2000_3d_conv_cfat[metric], positions=[positions[7]],patch_artist=True,boxprops=dict(color='black',facecolor='lightblue'))

    
#     # Plot box plots for the metric from def_2000 on the right
#     #axes[i].boxplot(def_2000_325d_conv_cfat[metric], positions=[positions[8]],patch_artist=True,boxprops=dict(color='black',facecolor='lightblue'))
    
    
    
    
#     #axes[i].boxplot(da_ca_2000_osic[metric], positions=[positions[9]],patch_artist=True,boxprops=dict(color='black',facecolor='lightblue'))

#     # Plot box plots for the metric from def_2000 on the right
#     axes[i].boxplot(def_2000_25d_osic[metric], positions=[positions[4]],widths=box_width,patch_artist=True,boxprops=dict(color='black',facecolor='lightblue'))
    
#     # Plot box plots for the metric from def_2000 on the right
#     axes[i].boxplot(def_2000_25d_conv_osic[metric], positions=[positions[5]],widths=box_width,patch_artist=True,boxprops=dict(color='black',facecolor='lightblue'))
    
#     axes[i].boxplot(def_2000_325d_conv_osic[metric], positions=[positions[7]],widths=box_width,patch_artist=True,boxprops=dict(color='black',facecolor='lightblue'))
   
#     #axes[i].boxplot(def_2000_3d_osic[metric], positions=[positions[11]],patch_artist=True,boxprops=dict(color='black',facecolor='lightblue'))
    
#     # Plot box plots for the metric from def_2000 on the right
#     axes[i].boxplot(def_2000_325d_osic[metric], positions=[positions[6]],widths=box_width,patch_artist=True,boxprops=dict(color='black',facecolor='lightblue'))
    
#     #axes[i].boxplot(def_2000_slc25d_osic[metric], positions=[positions[5]],widths=box_width,patch_artist=True,boxprops=dict(color='black',facecolor='lightblue'))

#     # Plot box plots for the metric from def_2000 on the right
#     #
#     # Plot box plots for the metric from def_2000 on the right
#     #axes[i].boxplot(def_2000_25d_fill_osic[metric], positions=[positions[13]],patch_artist=True,boxprops=dict(color='black',facecolor='lightblue'))
#     #axes[i].boxplot(def_2000_slc25d_conv_osic[metric], positions=[positions[15]],patch_artist=True,boxprops=dict(color='black',facecolor='lightblue'))
    
#     #axes[i].boxplot(def_2000_3d_conv_osic[metric], positions=[positions[16]],patch_artist=True,boxprops=dict(color='black',facecolor='lightblue'))

#     # Plot box plots for the metric from def_2000 on the right
#     #
    
    
    
    
#     #axes[i].boxplot(da_ca_2000_hosp[metric], positions=[positions[18]],patch_artist=True,boxprops=dict(color='black',facecolor='lightblue'))

#     # Plot box plots for the metric from def_2000 on the right
#     axes[i].boxplot(def_2000_25d_hosp[metric], positions=[positions[8]],widths=box_width,patch_artist=True,boxprops=dict(color='black',facecolor='lightblue'))
    
#     axes[i].boxplot(def_2000_25d_conv_hosp[metric], positions=[positions[9]],widths=box_width,patch_artist=True,boxprops=dict(color='black',facecolor='lightblue'))

#     axes[i].boxplot(def_2000_325d_conv_hosp[metric], positions=[positions[11]],widths=box_width,patch_artist=True,boxprops=dict(color='black',facecolor='lightblue'))
    
#     # Plot box plots for the metric from def_2000 on the right
#     #axes[i].boxplot(def_2000_3d_hosp[metric], positions=[positions[20]],patch_artist=True,boxprops=dict(color='black',facecolor='lightblue'))
    
#     # Plot box plots for the metric from def_2000 on the right
#     axes[i].boxplot(def_2000_325d_hosp[metric], positions=[positions[10]],widths=box_width,patch_artist=True,boxprops=dict(color='black',facecolor='lightblue'))
    
#     #axes[i].boxplot(def_2000_slc25d_hosp[metric], positions=[positions[8]],widths=box_width,patch_artist=True,boxprops=dict(color='black',facecolor='lightblue'))

#     # Plot box plots for the metric from def_2000 on the right
    
#     # Plot box plots for the metric from def_2000 on the right
#     #axes[i].boxplot(def_2000_25d_fill_hosp[metric], positions=[positions[21]],patch_artist=True,boxprops=dict(color='black',facecolor='lightblue'))
#     #axes[i].boxplot(def_2000_slc25d_conv_hosp[metric], positions=[positions[24]],patch_artist=True,boxprops=dict(color='black',facecolor='lightblue'))

#     #axes[i].boxplot(def_2000_3d_conv_hosp[metric], positions=[positions[25]],patch_artist=True,boxprops=dict(color='black',facecolor='lightblue'))

        
#     # Plot box plots for the metric from def_2000 on the right
    
    
    
#     # Add horizontal gridlines
#     axes[i].grid(axis='y', linestyle='--', linewidth=0.5, color='gray')

# # Calculate the mean for each metric
#     means = [[#da_ca_2000_cfat[metric].mean(), 
#               def_2000_25d_cfat[metric].mean(), 
#               #def_2000_3d_cfat[metric].mean(), 
              
#               #def_2000_slc25d_cfat[metric].mean()#,
#               def_2000_25d_conv_cfat[metric].mean(),
#               #,def_2000_slc25d_conv_cfat[metric].mean(),def_2000_3d_conv_cfat[metric].mean(),
#               def_2000_325d_cfat[metric].mean(),
#               def_2000_325d_conv_cfat[metric].mean()
#               ],
#               [ #da_ca_2000_osic[metric].mean(), 
#                 def_2000_25d_osic[metric].mean(),
#                 #def_2000_3d_osic[metric].mean(), 
#                 #def_2000_slc25d_osic[metric].mean()#,
#                 def_2000_25d_conv_osic[metric].mean(),#,def_2000_slc25d_conv_osic[metric].mean(),def_2000_3d_conv_osic[metric].mean(),
#                 def_2000_325d_osic[metric].mean(),
#                 def_2000_325d_conv_osic[metric].mean()
                            
#               ] ,
#               [ #da_ca_2000_hosp[metric].mean(), 
#                 def_2000_25d_hosp[metric].mean(), 
#                 #def_2000_3d_hosp[metric].mean(), 
#                 #def_2000_slc25d_hosp[metric].mean()
#                 def_2000_25d_conv_hosp[metric].mean(),#def_2000_slc25d_conv_hosp[metric].mean(),def_2000_3d_conv_hosp[metric].mean(),
#                 def_2000_325d_hosp[metric].mean(),
#                 def_2000_325d_conv_hosp[metric].mean()
                            
#               ]     
#               ]
    
#     # Plot average line

#     boxp = [[#da_ca_2000_cfat[metric], 
#               def_2000_25d_cfat[metric], 
#               #def_2000_3d_cfat[metric], 
#               #
#               #def_2000_slc25d_cfat[metric]#,
#               def_2000_25d_conv_cfat[metric],#,def_2000_slc25d_conv_cfat[metric],def_2000_3d_conv_cfat[metric],
#               def_2000_325d_cfat[metric],
#               def_2000_325d_conv_cfat[metric]
#               ],
#               [ #da_ca_2000_osic[metric], 
#                 def_2000_25d_osic[metric],
#                 #def_2000_3d_osic[metric], 
#                 #
#                 #def_2000_slc25d_osic[metric]#,
#                 def_2000_25d_conv_osic[metric],#,def_2000_slc25d_conv_osic[metric],def_2000_3d_conv_osic[metric],
#                 def_2000_325d_osic[metric],
#                 def_2000_325d_conv_osic[metric]
                            
#               ] ,
#               [ #da_ca_2000_hosp[metric], 
#                 def_2000_25d_hosp[metric], 
#                 #def_2000_3d_hosp[metric], 
#                 #
#                 #def_2000_slc25d_hosp[metric]
#                 def_2000_25d_conv_hosp[metric],#def_2000_slc25d_conv_hosp[metric],def_2000_3d_conv_hosp[metric],
#                 def_2000_325d_hosp[metric],
#                 def_2000_325d_conv_hosp[metric]
                            
#               ]     
#               ]
#     # Plot average line

# # Plot average points
#     points=[[0.5, 1, 1.5,2],[ 3,3.5,4,4.5],[5.5,6,6.5,7]]

#     axes[i].plot(points, means[:],'o' ,color='black',markersize=6)
#         # Add average values as text above each boxplot
#     if metric=='dice':
#         for j, p in zip(points[0],boxp[0]):
#             mean=p.mean()
#             axes[i].text(j, max(p)+0.02*max(p), f'{mean:.3f}', color='black', ha='center', va='center', fontsize=20)
#         for j, p in zip(points[1],boxp[1]):
#             mean=p.mean()
#             axes[i].text(j, max(p)+0.02*max(p), f'{mean:.3f}', color='black', ha='center', va='center', fontsize=20)
#         for j, p in zip(points[2],boxp[2]):
#             mean=p.mean()
#             axes[i].text(j, max(p)+0.02*max(p), f'{mean:.3f}', color='black', ha='center', va='center', fontsize=20)

#         axes[i].set_ylim(ymax=1.02)


#     elif metric=='hd':   
#         for j, p in zip(points[0],boxp[0]):
#             mean=p.mean()
#             axes[i].text(j, max(p)+1.5*max(p), f'{mean:.3f}', color='black', ha='center', va='center', fontsize=20)
#         for j, p in zip(points[1],boxp[1]):
#             mean=p.mean()
#             axes[i].text(j, mean+2.5*mean, f'{mean:.3f}', color='black', ha='center', va='center', fontsize=20)
#         for j, p in zip(points[2],boxp[2]):
#             mean=p.mean()
#             axes[i].text(j, max(p)+0.5*max(p), f'{mean:.3f}', color='black', ha='center', va='center', fontsize=20)
#         axes[i].set_ylim(ymax=350)

#     else:
#         for j, p in zip(points[0],boxp[0]):
#             mean=p.mean()
#             axes[i].text(j, max(p)+0.8*max(p), f'{mean:.3f}', color='black', ha='center', va='center', fontsize=20)
#         for j, p in zip(points[1],boxp[1]):
#             mean=p.mean()
#             axes[i].text(j, max(p)+0.5*max(p), f'{mean:.3f}', color='black', ha='center', va='center', fontsize=20)
#         for j, p in zip(points[2],boxp[2]):
#             mean=p.mean()
#             axes[i].text(j, max(p)+0.3*max(p), f'{mean:.3f}', color='black', ha='center', va='center', fontsize=20)
#         axes[i].set_ylim(ymax=22)



   
#     #labels_dt=[1.75, 5.25, 8.75]
#     labels_dt=[0.167, 0.5, 0.832]
#     labels=['Cardiac Fat','OSIC','CHVNGE']
#     y_max=axes[i].get_ylim()[1]
#     for j, label in enumerate(labels):
#         axes[i].text(labels_dt[j], 1.03, label, color='black', ha='center', va='center',fontweight='bold', fontsize=28, transform=axes[i].transAxes)

#     # Set the x-axis tick positions and labels at the top
    
#     if metric !='dice':
#         axes[i].set_ylabel(metric.upper()+'(mm)',size=25,fontweight='bold')
#     else:
#         metric='DSC'
#         axes[i].set_ylabel(metric,size=25,fontweight='bold')
    
#     # Set x-axis labels
#     axes[i].tick_params(axis='both', labelsize=20)
    
#     # Set x-axis labels
#     axes[i].set_xticks(positions)
#     axes[i].set_xticklabels(['2.5D','2.5D+pp','3D+2.5D','3D+2.5D+pp', '2.5D','2.5D+pp','3D+2.5D','3D+2.5D+pp', '2.5D','2.5D+pp','3D+2.5D','3D+2.5D+pp'], fontsize=14.5, fontweight='bold')
#     #axes[i].set_xticklabels(['2.5D','3D+2.5D','slc+2.5D', '2.5D','3D+2.5D','slc+2.5D', '2.5D','3D+2.5D','slc+2.5D'], fontsize=20, fontweight='bold')


"2.5d 350 e 1000 HU "

# # Create subplots for each metric
# fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(25, 25))


# # Iterate over each metric column
# for i, metric in enumerate(metric_columns):
    
#     #fig = plt.figure(figsize =(10, 7),dpi=500)
     
#     # Creating axes instance
#     #axes = fig.add_axes([0, 0, 1, 1])
#     # Plot box plots for the metric from da_350 on the left
#     axes[i].boxplot(def_2000_25d_cfat[metric], positions=[0.5],patch_artist=True,boxprops=dict(color='black',facecolor='lightblue'))

#     # Plot box plots for the metric from def_2000 on the right
#     axes[i].boxplot(def_350_25d_cfat[metric], positions=[1],patch_artist=True,boxprops=dict(color='black',facecolor='lightgreen'))
    
    
    
#     axes[i].boxplot(def_2000_25d_osic[metric], positions=[2],patch_artist=True,boxprops=dict(color='black',facecolor='lightblue'))

#     # Plot box plots for the metric from def_2000 on the right
#     axes[i].boxplot(def_350_25d_osic[metric], positions=[2.5],patch_artist=True,boxprops=dict(color='black',facecolor='lightgreen'))
    
   
    
#     axes[i].boxplot(def_2000_25d_hosp[metric], positions=[3.5],patch_artist=True,boxprops=dict(color='black',facecolor='lightblue'))

#     # Plot box plots for the metric from def_2000 on the right
#     axes[i].boxplot(def_350_25d_hosp[metric], positions=[4],patch_artist=True,boxprops=dict(color='black',facecolor='lightgreen'))
    
   

#     # Add horizontal gridlines
#     axes[i].grid(axis='y', linestyle='--', linewidth=0.5, color='gray')

# # Calculate the mean for each metric
#     means = [[def_2000_25d_cfat[metric].mean(), def_350_25d_cfat[metric].mean()
              
#               ],
#               [ def_2000_25d_osic[metric].mean(), def_350_25d_osic[metric].mean()
                       
#               ] ,
#               [ def_2000_25d_hosp[metric].mean(), def_350_25d_hosp[metric].mean()
                       
#               ]     
#               ]
    
#     # Plot average line

# # Plot average points
#     points=[[0.5, 1],[ 2, 2.5],[ 3.5, 4]]

#     axes[i].plot(points, means[:],'o' ,color='black',markersize=6)
#     # Add average values as text above each boxplot
#     if metric=='dice':
#         for j, mean in zip(points[0],means[0]):
#             axes[i].text(j, mean-0.1*mean, f'{mean:.3f}', color='black', ha='center', va='center', fontsize=15)
#         for j, mean in zip(points[1],means[1]):
#             axes[i].text(j, mean-0.15*mean, f'{mean:.3f}', color='black', ha='center', va='center', fontsize=15)
#         for j, mean in zip(points[2],means[2]):
#             axes[i].text(j, 0.97, f'{mean:.3f}', color='black', ha='center', va='center', fontsize=15)

#     elif metric=='hd':   
#         for j, mean in zip(points[0],means[0]):
#             axes[i].text(j, mean+0.7*mean, f'{mean:.3f}', color='black', ha='center', va='center', fontsize=15)
#         for j, mean in zip(points[1],means[1]):
#             axes[i].text(j, mean+2.4*mean, f'{mean:.3f}', color='black', ha='center', va='center', fontsize=15)
#         for j, mean in zip(points[2],means[2]):
#             axes[i].text(j, mean+1.8*mean, f'{mean:.3f}', color='black', ha='center', va='center', fontsize=15)

#     else:
#         for j, mean in zip(points[0],means[0]):
#             axes[i].text(j, mean+1.5*mean, f'{mean:.3f}', color='black', ha='center', va='center', fontsize=15)
#         for j, mean in zip(points[1],means[1]):
#             axes[i].text(j, mean+2.8*mean, f'{mean:.3f}', color='black', ha='center', va='center', fontsize=15)
#         for j, mean in zip(points[2],means[2]):
#             axes[i].text(j, 2, f'{mean:.3f}', color='black', ha='center', va='center', fontsize=15)



#     # Set x-axis labels
#     axes[i].set_xticks([0.5, 1, 2, 2.5, 3.5, 4])
#     axes[i].set_xticklabels(['2.5D', '2.5D','2.5D', '2.5D','2.5D', '2.5D'], fontsize=15, fontweight='bold')
#     #
#     #labels_dt=[1.75, 5.25, 8.75]
#     labels_dt=[0.167, 0.5, 0.832]
#     labels=['Cardiac Fat','OSIC','CHVNGE']
#     y_max=axes[i].get_ylim()[1]
#     for j, label in enumerate(labels):
#         axes[i].text(labels_dt[j], 1.02, label, color='black', ha='center', va='center',fontweight='bold', fontsize=22, transform=axes[i].transAxes)

#     # Set the x-axis tick positions and labels at the top
    
#     if metric !='dice':
#         axes[i].set_ylabel(metric.upper()+'(mm)',size=25,fontweight='bold')
#     else:
#         metric='DSC'
#         axes[i].set_ylabel(metric,size=25,fontweight='bold')
    
#     # Set x-axis labels
#     axes[i].tick_params(axis='both', labelsize=16)
    
"Boxplots BCE DICE"
# Create subplots for each metric
# fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(25, 25))

# "Modelo data augmentation, etc."
# # Iterate over each metric column
# for i, metric in enumerate(metric_columns):
    
#     #fig = plt.figure(figsize =(10, 7),dpi=500)
#     box_width=0.3
#     # Creating axes instance
#     #axes = fig.add_axes([0, 0, 1, 1])
#     # Plot box plots for the metric from da_350 on the left
#     axes[i].boxplot(def_2000_bce_cfat[metric], positions=[0.5],widths=box_width,patch_artist=True,boxprops=dict(color='black',facecolor='lightblue'))

#     # Plot box plots for the metric from def_2000 on the right
#     axes[i].boxplot(def_350_bce_cfat[metric], positions=[1],widths=box_width,patch_artist=True,boxprops=dict(color='black',facecolor='lightgreen'))
    
#     # Plot box plots for the metric from def_2000 on the right
#     axes[i].boxplot(def_2000_dice_cfat[metric], positions=[1.5],widths=box_width,patch_artist=True,boxprops=dict(color='black',facecolor='lightblue'))
    
#     # Plot box plots for the metric from def_2000 on the right
#     axes[i].boxplot(def_350_cfat[metric], positions=[2],widths=box_width,patch_artist=True,boxprops=dict(color='black',facecolor='lightgreen'))
    
#     # # Plot box plots for the metric from def_2000 on the right
#     # axes[i].boxplot(da_350_cfat[metric], positions=[2.5],patch_artist=True,boxprops=dict(color='black',facecolor='lightgreen'))

#     # axes[i].boxplot(da_ca_350_cfat[metric], positions=[3],patch_artist=True,boxprops=dict(color='black',facecolor='lightgreen'))


#     # Plot box plots for the metric from da_350 on the left
#     axes[i].boxplot(def_2000_bce_osic[metric], positions=[3],widths=box_width,patch_artist=True,boxprops=dict(color='black',facecolor='lightblue'))

#     # Plot box plots for the metric from def_2000 on the right
#     axes[i].boxplot(def_350_bce_osic[metric], positions=[3.5],widths=box_width,patch_artist=True,boxprops=dict(color='black',facecolor='lightgreen'))
    
#     # Plot box plots for the metric from def_2000 on the right
#     axes[i].boxplot(def_2000_dice_osic[metric], positions=[4],widths=box_width,patch_artist=True,boxprops=dict(color='black',facecolor='lightblue'))
    
#     # # Plot box plots for the metric from def_2000 on the right
#     axes[i].boxplot(def_350_osic[metric], positions=[4.5],widths=box_width,patch_artist=True,boxprops=dict(color='black',facecolor='lightgreen'))
    
#     # # Plot box plots for the metric from def_2000 on the right
#     # axes[i].boxplot(da_350_osic[metric], positions=[6],patch_artist=True,boxprops=dict(color='black',facecolor='lightgreen'))

#     # axes[i].boxplot(da_ca_350_osic[metric], positions=[6.5],patch_artist=True,boxprops=dict(color='black',facecolor='lightgreen'))

    
#     # Plot box plots for the metric from da_350 on the left
#     axes[i].boxplot(def_2000_bce_hosp[metric], positions=[5.5],widths=box_width,patch_artist=True,boxprops=dict(color='black',facecolor='lightblue'))

#     # Plot box plots for the metric from def_2000 on the right
#     axes[i].boxplot(def_350_bce_hosp[metric], positions=[6],widths=box_width,patch_artist=True,boxprops=dict(color='black',facecolor='lightgreen'))
    
#     # Plot box plots for the metric from def_2000 on the right
#     axes[i].boxplot(def_2000_dice_hosp[metric], positions=[6.5],widths=box_width,patch_artist=True,boxprops=dict(color='black',facecolor='lightblue'))
    
#     # # Plot box plots for the metric from def_2000 on the right
#     axes[i].boxplot(def_350_hosp[metric], positions=[7],widths=box_width,patch_artist=True,boxprops=dict(color='black',facecolor='lightgreen'))
    
#     # # Plot box plots for the metric from def_2000 on the right
#     # axes[i].boxplot(da_350_hosp[metric], positions=[9.5],patch_artist=True,boxprops=dict(color='black',facecolor='lightgreen'))

#     # axes[i].boxplot(da_ca_350_hosp[metric], positions=[10],patch_artist=True,boxprops=dict(color='black',facecolor='lightgreen'))

#     # Add horizontal gridlines
#     axes[i].grid(axis='y', linestyle='--', linewidth=0.5, color='gray')

# # Calculate the mean for each metric
#     means = [[def_2000_bce_cfat[metric].mean(), def_350_bce_cfat[metric].mean(), def_2000_dice_cfat[metric].mean(),
#               def_350_cfat[metric].mean()#, da_350_cfat[metric].mean(), da_ca_350_cfat[metric].mean()
#               ],
#               [ def_2000_bce_osic[metric].mean(), def_350_bce_osic[metric].mean(), def_2000_dice_osic[metric].mean(),
#               def_350_osic[metric].mean()#, da_350_osic[metric].mean(), da_ca_350_osic[metric].mean()
#               ] ,
#               [ def_2000_bce_hosp[metric].mean(), def_350_bce_hosp[metric].mean(), def_2000_dice_hosp[metric].mean(),
#               def_350_hosp[metric].mean()#, da_350_hosp[metric].mean(), da_ca_350_hosp[metric].mean()
#               ]     
#               ]
    
#     boxp = [[def_2000_bce_cfat[metric], def_350_bce_cfat[metric], def_2000_dice_cfat[metric],
#               def_350_cfat[metric]#, da_350_cfat[metric], da_ca_350_cfat[metric]
#               ],
#               [ def_2000_bce_osic[metric], def_350_bce_osic[metric], def_2000_dice_osic[metric],
#               def_350_osic[metric]#, da_350_osic[metric], da_ca_350_osic[metric]
#               ] ,
#               [ def_2000_bce_hosp[metric], def_350_bce_hosp[metric], def_2000_dice_hosp[metric],
#               def_350_hosp[metric]#, da_350_hosp[metric], da_ca_350_hosp[metric]
#               ]     
#               ]
    
#     # Plot average line

# # Plot average points
#     points=[[0.5, 1, 1.5,2],[ 3, 3.5, 4,4.5],[ 5.5, 6, 6.5,7]]

#     axes[i].plot(points, means[:],'o' ,color='black',markersize=6)
#         # Add average values as text above each boxplot
#     if metric=='dice':
#         for j, p in zip(points[0],boxp[0]):
#             mean=p.mean()
#             axes[i].text(j, max(p)+0.02*max(p), f'{mean:.3f}', color='black', ha='center', va='center', fontsize=20)
#         for j, p in zip(points[1],boxp[1]):
#             mean=p.mean()
#             axes[i].text(j, max(p)+0.02*max(p), f'{mean:.3f}', color='black', ha='center', va='center', fontsize=20)
#         for j, p in zip(points[2],boxp[2]):
#             mean=p.mean()
#             axes[i].text(j, max(p)+0.02*max(p), f'{mean:.3f}', color='black', ha='center', va='center', fontsize=20)

#         axes[i].set_ylim(ymax=1.02)


#     elif metric=='hd':   
#         for j, p in zip(points[0],boxp[0]):
#             mean=p.mean()
#             axes[i].text(j, max(p)+0.5*max(p), f'{mean:.3f}', color='black', ha='center', va='center', fontsize=20)
#         for j, p in zip(points[1],boxp[1]):
#             mean=p.mean()
#             axes[i].text(j, max(p)+0.07*max(p), f'{mean:.3f}', color='black', ha='center', va='center', fontsize=20)
#         for j, p in zip(points[2],boxp[2]):
#             mean=p.mean()
#             axes[i].text(j, max(p)+0.5*max(p), f'{mean:.3f}', color='black', ha='center', va='center', fontsize=20)
#         axes[i].set_ylim(ymax=375)

#     else:
#         for j, p in zip(points[0],boxp[0]):
#             mean=p.mean()
#             axes[i].text(j, max(p)+0.5*max(p), f'{mean:.3f}', color='black', ha='center', va='center', fontsize=20)
#         for j, p in zip(points[1],boxp[1]):
#             mean=p.mean()
#             axes[i].text(j, max(p)+0.105*max(p), f'{mean:.3f}', color='black', ha='center', va='center', fontsize=20)
#         for j, p in zip(points[2],boxp[2]):
#             mean=p.mean()
#             axes[i].text(j, max(p)+0.3*max(p), f'{mean:.3f}', color='black', ha='center', va='center', fontsize=20)
#         axes[i].set_ylim(ymax=42)



#     # Set x-axis labels
#     axes[i].set_xticks([0.5, 1, 1.5,2, 3, 3.5, 4,4.5, 5.5, 6, 6.5,7])
#     axes[i].set_xticklabels(['BCE', 'BCE','DICE', 'DICE','BCE', 'BCE','DICE', 'DICE','BCE', 'BCE','DICE', 'DICE'], fontsize=20, fontweight='bold')
#     #
#     #labels_dt=[1.75, 5.25, 8.75]
#     labels_dt=[0.167, 0.5, 0.832]
#     labels=['Cardiac Fat','OSIC','CHVNGE']
#     y_max=axes[i].get_ylim()[1]
#     for j, label in enumerate(labels):
#         axes[i].text(labels_dt[j], 1.03, label, color='black', ha='center', va='center',fontweight='bold', fontsize=28, transform=axes[i].transAxes)

#     # Set the x-axis tick positions and labels at the top
    
#     if metric !='dice':
#         axes[i].set_ylabel(metric.upper()+'(mm)',size=25,fontweight='bold')
#     else:
#         metric='DSC'
#         axes[i].set_ylabel(metric,size=25,fontweight='bold')
    
#     # Set x-axis labels
#     axes[i].tick_params(axis='both', labelsize=20)

"Boxplot of the 2D vs 2.5D"    



"Boxplots das calcifications e augmentations"
# Create subplots for each metric
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(25, 25))

"Modelo data augmentation, etc."
# Iterate over each metric column
for i, metric in enumerate(metric_columns):
    box_width=0.3
    #fig = plt.figure(figsize =(10, 7),dpi=500)
     
    # Creating axes instance
    #axes = fig.add_axes([0, 0, 1, 1])
    # Plot box plots for the metric from da_350 on the left
    axes[i].boxplot(def_2000_dice_cfat[metric], positions=[0.5],widths=box_width,patch_artist=True,boxprops=dict(color='black',facecolor='lightblue'))

    # Plot box plots for the metric from def_2000 on the right
    #axes[i].boxplot(da_2000_cfat[metric], positions=[1],widths=box_width,patch_artist=True,boxprops=dict(color='black',facecolor='lightblue'))
    
    # Plot box plots for the metric from def_2000 on the right
    axes[i].boxplot(da_ca_2000_cfat[metric], positions=[1],widths=box_width,patch_artist=True,boxprops=dict(color='black',facecolor='lightblue'))
    
    # Plot box plots for the metric from def_2000 on the right
    # axes[i].boxplot(def_350_cfat[metric], positions=[2],patch_artist=True,boxprops=dict(color='black',facecolor='lightgreen'))
    
    # # Plot box plots for the metric from def_2000 on the right
    # axes[i].boxplot(da_350_cfat[metric], positions=[2.5],patch_artist=True,boxprops=dict(color='black',facecolor='lightgreen'))

    # axes[i].boxplot(da_ca_350_cfat[metric], positions=[3],patch_artist=True,boxprops=dict(color='black',facecolor='lightgreen'))


    # Plot box plots for the metric from da_350 on the left
    axes[i].boxplot(def_2000_dice_osic[metric], positions=[2],widths=box_width,patch_artist=True,boxprops=dict(color='black',facecolor='lightblue'))

    # Plot box plots for the metric from def_2000 on the right
    #axes[i].boxplot(da_2000_osic[metric], positions=[3],widths=box_width,patch_artist=True,boxprops=dict(color='black',facecolor='lightblue'))
    
    # Plot box plots for the metric from def_2000 on the right
    axes[i].boxplot(da_ca_2000_osic[metric], positions=[2.5],widths=box_width,patch_artist=True,boxprops=dict(color='black',facecolor='lightblue'))
    
    # # Plot box plots for the metric from def_2000 on the right
    # axes[i].boxplot(def_350_osic[metric], positions=[5.5],patch_artist=True,boxprops=dict(color='black',facecolor='lightgreen'))
    
    # # Plot box plots for the metric from def_2000 on the right
    # axes[i].boxplot(da_350_osic[metric], positions=[6],patch_artist=True,boxprops=dict(color='black',facecolor='lightgreen'))

    # axes[i].boxplot(da_ca_350_osic[metric], positions=[6.5],patch_artist=True,boxprops=dict(color='black',facecolor='lightgreen'))

    
    # Plot box plots for the metric from da_350 on the left
    axes[i].boxplot(def_2000_dice_hosp[metric], positions=[3.5],widths=box_width,patch_artist=True,boxprops=dict(color='black',facecolor='lightblue'))

    # Plot box plots for the metric from def_2000 on the right
    #axes[i].boxplot(da_2000_hosp[metric], positions=[5],widths=box_width,patch_artist=True,boxprops=dict(color='black',facecolor='lightblue'))
    
    # Plot box plots for the metric from def_2000 on the right
    axes[i].boxplot(da_ca_2000_hosp[metric], positions=[4],widths=box_width,patch_artist=True,boxprops=dict(color='black',facecolor='lightblue'))
    
    # # Plot box plots for the metric from def_2000 on the right
    # axes[i].boxplot(def_350_hosp[metric], positions=[9],patch_artist=True,boxprops=dict(color='black',facecolor='lightgreen'))
    
    # # Plot box plots for the metric from def_2000 on the right
    # axes[i].boxplot(da_350_hosp[metric], positions=[9.5],patch_artist=True,boxprops=dict(color='black',facecolor='lightgreen'))

    # axes[i].boxplot(da_ca_350_hosp[metric], positions=[10],patch_artist=True,boxprops=dict(color='black',facecolor='lightgreen'))

    # Add horizontal gridlines
    axes[i].grid(axis='y', linestyle='--', linewidth=0.5, color='gray')

# Calculate the mean for each metric
    means = [[def_2000_dice_cfat[metric].mean(),  da_ca_2000_cfat[metric].mean(),
              #def_350_cfat[metric].mean(), da_350_cfat[metric].mean(), da_ca_350_cfat[metric].mean()
              ],
              [ def_2000_dice_osic[metric].mean(), da_ca_2000_osic[metric].mean(),
              #def_350_osic[metric].mean(), da_350_osic[metric].mean(), da_ca_350_osic[metric].mean()
              ] ,
              [ def_2000_dice_hosp[metric].mean(), da_ca_2000_hosp[metric].mean(),
              #def_350_hosp[metric].mean(), da_350_hosp[metric].mean(), da_ca_350_hosp[metric].mean()
              ]     
              ]
    
    boxp = [[def_2000_dice_cfat[metric], da_ca_2000_cfat[metric],
              #def_350_cfat[metric], da_350_cfat[metric], da_ca_350_cfat[metric]
              ],
              [ def_2000_dice_osic[metric], da_ca_2000_osic[metric],
              #def_350_osic[metric], da_350_osic[metric], da_ca_350_osic[metric]
              ] ,
              [ def_2000_dice_hosp[metric], da_ca_2000_hosp[metric],
              #def_350_hosp[metric], da_350_hosp[metric], da_ca_350_hosp[metric]
              ]     
              ]
    
    # Plot average line

# Plot average points
    points=[[0.5, 1],[ 2,2.5],[ 3.5, 4]]

    axes[i].plot(points, means[:],'o' ,color='black',markersize=6)
        # Add average values as text above each boxplot
    if metric=='dice':
        for j, p in zip(points[0],boxp[0]):
            mean=p.mean()
            axes[i].text(j, max(p)+0.02*max(p), f'{mean:.3f}', color='black', ha='center', va='center', fontsize=20)
        for j, p in zip(points[1],boxp[1]):
            mean=p.mean()
            axes[i].text(j, max(p)+0.02*max(p), f'{mean:.3f}', color='black', ha='center', va='center', fontsize=20)
        for j, p in zip(points[2],boxp[2]):
            mean=p.mean()
            axes[i].text(j, max(p)+0.02*max(p), f'{mean:.3f}', color='black', ha='center', va='center', fontsize=20)

        axes[i].set_ylim(ymax=1.0)


    elif metric=='hd':   
        for j, p in zip(points[0],boxp[0]):
            mean=p.mean()
            axes[i].text(j, max(p)+0.4*max(p), f'{mean:.3f}', color='black', ha='center', va='center', fontsize=20)
        for j, p in zip(points[1],boxp[1]):
            mean=p.mean()
            axes[i].text(j, max(p)+0.09*max(p), f'{mean:.3f}', color='black', ha='center', va='center', fontsize=20)
        for j, p in zip(points[2],boxp[2]):
            mean=p.mean()
            axes[i].text(j, max(p)+0.3*max(p), f'{mean:.3f}', color='black', ha='center', va='center', fontsize=20)
        axes[i].set_ylim(ymax=350)

    else:
        for j, p in zip(points[0],boxp[0]):
            mean=p.mean()
            axes[i].text(j, max(p)+0.2*max(p), f'{mean:.3f}', color='black', ha='center', va='center', fontsize=20)
        for j, p in zip(points[1],boxp[1]):
            mean=p.mean()
            axes[i].text(j, max(p)+0.12*max(p), f'{mean:.3f}', color='black', ha='center', va='center', fontsize=20)
        for j, p in zip(points[2],boxp[2]):
            mean=p.mean()
            axes[i].text(j, max(p)+0.1*max(p), f'{mean:.3f}', color='black', ha='center', va='center', fontsize=20)
        axes[i].set_ylim(ymax=23.3)



    # Set x-axis labels
    axes[i].set_xticks([0.5, 1, 2, 2.5, 3.5,4])
    axes[i].set_xticklabels(['S','DA+AC', 'S','DA+AC','S','DA+AC'], fontsize=20, fontweight='bold')
    #
    #labels_dt=[1.75, 5.25, 8.75]
    labels_dt=[0.167, 0.5, 0.832]
    labels=['Cardiac Fat','OSIC','CHVNGE']
    y_max=axes[i].get_ylim()[1]
    for j, label in enumerate(labels):
        axes[i].text(labels_dt[j], 1.03, label, color='black', ha='center', va='center',fontweight='bold', fontsize=28, transform=axes[i].transAxes)

    # Set the x-axis tick positions and labels at the top
    
    if metric !='dice':
        axes[i].set_ylabel(metric.upper()+'(mm)',size=25,fontweight='bold')
    else:
        metric='DSC'
        axes[i].set_ylabel(metric,size=25,fontweight='bold')
    
    # Set x-axis labels
    axes[i].tick_params(axis='both', labelsize=20)
    
    # #Set subplot title
    # axes[i].set_title(metric, fontsize=25, fontweight='bold')
    # #Set subplot title in uppercase
    
    
    # axes[i].text(0.5, 1.08, metric.upper(), fontsize=25, fontweight='bold', ha='center', transform=axes[i].transAxes)
    
#Iterate over each metric column
# for i, metric in enumerate(metric_columns):
    
#     #fig = plt.figure(figsize =(10, 7),dpi=500)
     
#     # Creating axes instance
#     #axes = fig.add_axes([0, 0, 1, 1])
#     # Plot box plots for the metric from da_350 on the left
#     axes[i].boxplot(def_2000_512[metric], positions=[0],patch_artist=True,boxprops=dict(color='black',facecolor='lightblue'))

#     # Plot box plots for the metric from def_2000 on the right
#     axes[i].boxplot(def_2000_dice[metric], positions=[1],patch_artist=True,boxprops=dict(color='black',facecolor='lightblue'))
        
#     # Plot box plots for the metric from def_2000 on the right
#     axes[i].boxplot(def_2000_bce[metric], positions=[2],patch_artist=True,boxprops=dict(color='black',facecolor='lightblue'))
    
#     # Plot box plots for the metric from def_2000 on the right
#     axes[i].boxplot(def_350[metric], positions=[3],patch_artist=True,boxprops=dict(color='black',facecolor='lightgreen'))
    
#     # Plot box plots for the metric from def_2000 on the right

# # Calculate the mean for each metric
#     means = [def_2000_512[metric].mean(), def_2000_dice[metric].mean(), def_2000_bce[metric].mean(),
#               def_350[metric].mean()]
    
#     # Plot average line

# # Plot average points
   
#     axes[i].plot([0, 1, 2, 3], means,'o' ,color='black',markersize=10)
    
#     # Add average values as text above each boxplot
#     for j, mean in enumerate(means):
#         axes[i].text(j-0.18, mean, f'{mean:.3f}', color='black', ha='center', va='center', fontsize=15)
 
#     # Set x-axis labels
#     axes[i].set_xticks([0,1,2,3])
#     axes[i].set_xticklabels(['DICE 512', 'DICE 256','BCE 256', 'DICE 256'], fontsize=16, fontweight='bold')

#     # Set y-axis label
#     if metric !='dice':
#         axes[i].set_ylabel('mm',size=18)
    
#     # Set x-axis labels
#     axes[i].tick_params(axis='both', labelsize=16)
    
#     # Set subplot title
#     #axes[i].set_title(metric, fontsize=25, fontweight='bold')
#     # Set subplot title in uppercase
#     axes[i].text(0.5, 1.03, metric.upper(), fontsize=25, fontweight='bold', ha='center', transform=axes[i].transAxes)



# labels_x=np.array([['512','256'],['DICE','BCE'],['DICE','DICE']])
# datasets=[[def_2000_512,def_2000_dice],[def_2000_dice,def_2000_bce],[def_2000_dice,def_350]]

# for i in range(3):
    
#     #fig = plt.figure(figsize =(10, 7),dpi=500)
     
#     # Creating axes instance
#     #axes = fig.add_axes([0, 0, 1, 1])
#     # Plot box plots for the metric from da_350 on the left
#     # Create a second y-axis
    
    
#     axes[i][0].boxplot(datasets[i][0]['dice'], positions=[0],patch_artist=True,boxprops=dict(color='black',facecolor='lightblue'))

#     # Plot box plots for the metric from def_2000 on the right
#     axes[i][0].boxplot(datasets[i][1]['dice'], positions=[1],patch_artist=True,boxprops=dict(color='black',facecolor='lightblue'))
    
#     # Plot box plots for the metric from def_2000 on the right
#     axes[i][1].boxplot(datasets[i][0]['hd'], positions=[0],patch_artist=True,boxprops=dict(color='black',facecolor='lightblue'))
    
#     # Plot box plots for the metric from def_2000 on the right
#     axes[i][1].boxplot(datasets[i][1]['hd'], positions=[1],patch_artist=True,boxprops=dict(color='black',facecolor='lightgreen'))
    
#     # Plot box plots for the metric from def_2000 on the right
#     axes[i][2].boxplot(datasets[i][0]['mad'], positions=[0],patch_artist=True,boxprops=dict(color='black',facecolor='lightblue'))
    
#     # Plot box plots for the metric from def_2000 on the right
#     axes[i][2].boxplot(datasets[i][1]['mad'], positions=[1],patch_artist=True,boxprops=dict(color='black',facecolor='lightgreen'))
    
    
# # Calculate the mean for each metric
#     means = [datasets[i][0]['dice'].mean(), datasets[i][1]['dice'].mean(), datasets[i][0]['hd'].mean(), datasets[i][1]['hd'].mean(),
#               datasets[i][0]['mad'].mean(), datasets[i][1]['mad'].mean()]
    
#     # Plot average line

# # Plot average points
   
#     axes[i][0].plot([0, 1], means[0:2],'o' ,color='black',markersize=10)
#     axes[i][1].plot([0, 1], means[2:4],'o' ,color='black',markersize=10)
#     axes[i][2].plot([0, 1], means[4:],'o' ,color='black',markersize=10)

    
#     # Add average values as text above each boxplot
#     for j in range(2):
#         axes[i][0].text(j-0.18, means[j], f'{means[j]:.3f}', color='black', ha='center', va='center', fontsize=15)
#         axes[i][1].text(j-0.18, means[j+2], f'{means[j+2]:.3f}', color='black', ha='center', va='center', fontsize=15)
#         axes[i][2].text(j-0.18, means[j+4], f'{means[j+4]:.3f}', color='black', ha='center', va='center', fontsize=15)



#     # Set x-axis labels
#     axes[i][0].set_xticks([0, 1])
#     axes[i][0].set_xticklabels([labels_x[i][0], labels_x[i][1]], fontsize=16, fontweight='bold')
    
#     axes[i][1].set_xticks([0, 1])
#     axes[i][1].set_xticklabels([labels_x[i][0], labels_x[i][1]], fontsize=16, fontweight='bold')
    
#     axes[i][2].set_xticks([0, 1])
#     axes[i][2].set_xticklabels([labels_x[i][0], labels_x[i][1]], fontsize=16, fontweight='bold')

    
#     # Set y-axis label
#     # if metric !='dice':
#     #     axes[i].set_ylabel('mm',size=18)
    
#     # Set x-axis labels
#     axes[i][0].tick_params(axis='both', labelsize=16)
#     axes[i][1].tick_params(axis='both', labelsize=16)
#     axes[i][2].tick_params(axis='both', labelsize=16)

#     # Set subplot title
#     #Set subplot title in uppercase
#     axes[0][0].text(0.5, 1.03, 'DSC', fontsize=25, fontweight='bold', ha='center', transform=axes[0][0].transAxes)
#     axes[0][1].text(0.5, 1.03, 'HD', fontsize=25, fontweight='bold', ha='center', transform=axes[0][1].transAxes)
#     axes[0][2].text(0.5, 1.03, 'MAD', fontsize=25, fontweight='bold', ha='center', transform=axes[0][2].transAxes)

   
    
# Adjust the spacing between subplots

path_to_copy=os.path.join("C:/Users/RubenSilva/Desktop/Results/Boxplots",'Overall')
isExist = os.path.exists(path_to_copy)
if not isExist:                         
  # Create a new directory because it does not exist 
  os.makedirs(path_to_copy)
  
os.chdir(path_to_copy)
 
# Create a legend
# legend_elements = [
#     plt.Line2D([0], [0], marker='s', color='w', label='[-1000,1000] HU', markerfacecolor='lightblue', markersize=12),
#     plt.Line2D([0], [0], marker='s', color='w', label='[-125,225] HU', markerfacecolor='lightgreen', markersize=12)
# ]

# legend_elements = [
#     plt.Line2D( label='cv= convex', markersize=15),
#     #plt.Line2D([0], [0], marker='s', color='w', label='[-125,225] HU', markerfacecolor='lightgreen', markersize=12)
# ]

# Add legend to the figure
#fig.legend(handles=legend_elements, loc='upper right',bbox_to_anchor=(0.94, 0.92),fontsize=17)

# Save the figure with the desired DPI
fig.savefig('dacappw_novo'+'.png',bbox_inches = 'tight')




