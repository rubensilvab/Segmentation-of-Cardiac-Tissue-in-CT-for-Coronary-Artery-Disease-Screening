# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 16:35:34 2023

@author: RubenSilva
"""

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


path_25d_hosp='C:/Users/RubenSilva/Desktop/Results/Hospital/2.5d'




"Extrair dice"
eat_dice_r1_auto=((pd.read_excel(os.path.join(path_25d_hosp,'Analise_L0_W2000_calc_augm_2.5d _EAT_pp+conv.xlsx'))).iloc[:,:5]).dropna()
eat_dice_r1_r2=((pd.read_excel(os.path.join(path_25d_hosp,'Analise_EAT_fabio_carolina.xlsx'))).iloc[:,:5]).dropna()
#eat_dice_r2_auto=((pd.read_excel(os.path.join(path_25d_hosp,'Analise_EAT_fabio_unet.xlsx'))).iloc[:,:5]).dropna()
#eat_dice_r1_r1=((pd.read_excel(os.path.join(path_25d_hosp,'Analise_EAT_fabio_carolina.xlsx'))).iloc[:,:5]).dropna()
eat_dice_intra=((pd.read_excel(os.path.join(path_25d_hosp,'Analise_EAT_carol1_carol2.xlsx'))).iloc[:,:5]).dropna()

"Extrair precision e recall"
eat_r1_r2_recprec=((pd.read_excel('C:/Users/RubenSilva/Desktop/segmentation_fabio/selection/inverted/EAT_segm/NRRD/Reports_r1_r2/Metrics_Report.xlsx')).iloc[:,:10]).dropna()
eat_r1_auto_recprec=((pd.read_excel('X:/Ruben/TESE/New_training_Unet/Models_only_pub_dset/Mix_datasets/All_data/predict/2.5D_Unet/Dice_loss/Hospital_tif/L0_W2000_calc_augm_tif/EAT_segm/NRRD/Reports/Metrics_Report.xlsx')).iloc[:,:10]).dropna()
#eat_r2_auto_recprec=((pd.read_excel('C:/Users/RubenSilva/Desktop/segmentation_fabio/selection/inverted/EAT_segm/NRRD/Reports_r2_unet/Metrics_Report.xlsx')).iloc[:,:10]).dropna()
eat_intra=((pd.read_excel('C:/Users/RubenSilva/Desktop/segmentation_inter_intra/selection/inverted/EAT_segm/NRRD/Reports_intra/Metrics_Report.xlsx')).iloc[:,:10]).dropna()

"""Restringir para os 20 pacientes"""

patients=np.unique(eat_dice_r1_r2['patient'])
"dice"
eat_dice_r1_auto_20=eat_dice_r1_auto[eat_dice_r1_auto['patient'].isin(patients)]
"recall e precision"
eat_r1_auto_recprec_20=eat_r1_auto_recprec[eat_r1_auto_recprec['Patient'].isin(patients)]

"""Adicionar dice ao csv"""

eat_r1_r2_recprec['dice']=eat_dice_r1_r2['dice']
eat_r1_auto_recprec['dice']=eat_dice_r1_auto['dice']
eat_intra['dice']=eat_dice_intra['dice']
#eat_r1_auto_recprec_20['dice']=np.array(eat_dice_r1_auto_20['dice'])
#eat_r2_auto_recprec['dice']=eat_dice_r2_auto['dice']
metric_columns=['dice','precision.1', 'recall.1']

import matplotlib.pyplot as plt


# Create subplots for each metric



"2D vs 2.5D"

# #Create subplots for each metric
# fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(25, 25))

# "Modelo data augmentation, etc."
# # Iterate over each metric column
# for i, metric in enumerate(metric_columns):
    
#     #fig = plt.figure(figsize =(10, 7),dpi=500)
     
#     # Creating axes instance
#     #axes = fig.add_axes([0, 0, 1, 1])
#     # Plot box plots for the metric from da_350 on the left
#     positions = [0.5, 1,1.5]
#     box_width=0.3
    
#     axes[i].boxplot(eat_r1_auto_recprec_20[metric], positions=[positions[0]], widths=box_width,patch_artist=True,boxprops=dict(color='black',facecolor='lightblue'))

#     # Plot box plots for the metric from def_2000 on the right
#     axes[i].boxplot(eat_r1_r2_recprec[metric], positions=[positions[1]], widths=box_width,patch_artist=True,boxprops=dict(color='black',facecolor='lightblue'))
    
#     # Plot box plots for the metric from def_2000 on the right
#     axes[i].boxplot(eat_r2_auto_recprec[metric], positions=[positions[2]], widths=box_width,patch_artist=True,boxprops=dict(color='black',facecolor='lightblue'))
    
#     # # Plot box plots for the metric from def_2000 on the right
#     # axes[i].boxplot(def_2000_25d_osic[metric], positions=[positions[3]], widths=box_width,patch_artist=True,boxprops=dict(color='black',facecolor='lightblue'))
    
#     # # Plot box plots for the metric from def_2000 on the right
#     # axes[i].boxplot(da_ca_2000_hosp[metric], positions=[positions[4]], widths=box_width,patch_artist=True,boxprops=dict(color='black',facecolor='lightblue'))

#     # axes[i].boxplot(def_2000_25d_hosp[metric], positions=[positions[5]], widths=box_width,patch_artist=True,boxprops=dict(color='black',facecolor='lightblue'))


#     # Add horizontal gridlines
#     axes[i].grid(axis='y', linestyle='--', linewidth=0.5, color='gray')

# # Calculate the mean for each metric
#     boxp = [[eat_r1_auto_recprec_20[metric], eat_r1_r2_recprec[metric],eat_r2_auto_recprec[metric]
#               ]]
    
#     # Plot average line

# # Plot average points
#     means=[[eat_r1_auto_recprec_20[metric].mean(), eat_r1_r2_recprec[metric].mean(),eat_r2_auto_recprec[metric].mean()
    
#               ]]
#     points=[[0.5, 1,1.5]]

#     axes[i].plot(points, means[:],'o' ,color='black',markersize=6)
#     # Add average values as text above each boxplot
#     if metric=='dice':
#         for j, p in zip(points[0],boxp[0]):
#             mean=p.mean()
#             axes[i].text(j, max(p)+0.01*max(p), f'{mean:.3f}', color='black', ha='center', va='center', fontsize=20)
        
#             axes[i].set_ylim(ymax=0.9)   

#     elif metric=='precision.1':   
#         for j, p in zip(points[0],boxp[0]):
#             mean=p.mean()
#             axes[i].text(j, max(p)+0.01*max(p), f'{mean:.3f}', color='black', ha='center', va='center', fontsize=20)
            
#             axes[i].set_ylim(ymax=0.95) 

#     else:
#         for j, p in zip(points[0],boxp[0]):
#             mean=p.mean()
#             axes[i].text(j, max(p)+0.01*max(p), f'{mean:.3f}', color='black', ha='center', va='center', fontsize=20)
       
#             axes[i].set_ylim(ymax=0.95) 


#     # Set x-axis labels
#     axes[i].set_xticks([0.5, 1,1.5])
#     axes[i].set_xticklabels(['Reader 1 - Auto', 'Reader 1 - Reader 2', 'Auto - Reader 2'], fontsize=20, fontweight='bold')
    
    
#     # Set the x-axis tick positions and labels at the top
    
#     if metric =='dice':
#         metric='DSC'
#         axes[i].set_ylabel(metric,size=25,fontweight='bold')
#     elif metric=='precision.1':
#         metric='Precision'
#         axes[i].set_ylabel(metric,size=25,fontweight='bold')
#     else:
        
#         metric='Recall'
#         axes[i].set_ylabel(metric,size=25,fontweight='bold')
    
#     # Set x-axis labels
#     axes[i].tick_params(axis='both', labelsize=20)


#Create subplots for each metric
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(15, 15))

"Modelo data augmentation, etc."
# Iterate over each metric column
for i, metric in enumerate(metric_columns):
    
    #fig = plt.figure(figsize =(10, 7),dpi=500)
     
    # Creating axes instance
    #axes = fig.add_axes([0, 0, 1, 1])
    # Plot box plots for the metric from da_350 on the left
    positions = [0.2,0.7,1.2]
    box_width=0.16
    
    axes[i].boxplot(eat_intra[metric], positions=[positions[0]],widths=box_width,patch_artist=True,boxprops=dict(color='black',facecolor='lightblue'))

    
    axes[i].boxplot(eat_r1_auto_recprec[metric], positions=[positions[2]],widths=box_width,patch_artist=True,boxprops=dict(color='black',facecolor='lightblue'))

    # Plot box plots for the metric from def_2000 on the right
    axes[i].boxplot(eat_r1_r2_recprec[metric], positions=[positions[1]], widths=box_width,patch_artist=True,boxprops=dict(color='black',facecolor='lightblue'))
    
    # Plot box plots for the metric from def_2000 on the right
    #axes[i].boxplot(da_ca_2000_osic[metric], positions=[positions[2]], widths=box_width,patch_artist=True,boxprops=dict(color='black',facecolor='lightblue'))
    
    # # Plot box plots for the metric from def_2000 on the right
    # axes[i].boxplot(def_2000_25d_osic[metric], positions=[positions[3]], widths=box_width,patch_artist=True,boxprops=dict(color='black',facecolor='lightblue'))
    
    # # Plot box plots for the metric from def_2000 on the right
    # axes[i].boxplot(da_ca_2000_hosp[metric], positions=[positions[4]], widths=box_width,patch_artist=True,boxprops=dict(color='black',facecolor='lightblue'))

    # axes[i].boxplot(def_2000_25d_hosp[metric], positions=[positions[5]], widths=box_width,patch_artist=True,boxprops=dict(color='black',facecolor='lightblue'))


    # Add horizontal gridlines
    axes[i].grid(axis='y', linestyle='--', linewidth=0.5, color='gray')

# Calculate the mean for each metric
    boxp = [[ eat_intra[metric], eat_r1_r2_recprec[metric],eat_r1_auto_recprec[metric]
              ]]
    
    # Plot average line

# Plot average points
    means=[[
        eat_intra[metric].mean(), eat_r1_r2_recprec[metric].mean(),eat_r1_auto_recprec[metric].mean()

              ]]
    points=[positions]

    axes[i].plot(points, means[:],'o' ,color='black',markersize=6)
    # Add average values as text above each boxplot
    if metric=='dice':
        for j, p in zip(points[0],boxp[0]):
            mean=p.mean()
            axes[i].text(j, max(p)+0.02*max(p), f'{mean:.3f}', color='black', ha='center', va='center', fontsize=20)
            axes[i].set_ylim(ymax=1)   
        

    elif metric=='precision.1':   
        for j, p in zip(points[0],boxp[0]):
            mean=p.mean()
            axes[i].text(j, max(p)+0.02*max(p), f'{mean:.3f}', color='black', ha='center', va='center', fontsize=20)
        
            axes[i].set_ylim(ymax=1)   
    else:
        for j, p in zip(points[0],boxp[0]):
            mean=p.mean()
            axes[i].text(j, max(p)+0.02*max(p), f'{mean:.3f}', color='black', ha='center', va='center', fontsize=20)
            axes[i].set_ylim(ymax=1)   
        

    # Set x-axis labels
    axes[i].set_xticks(positions)
    axes[i].set_xticklabels(['Intrareader', 'Interreader','Reader 1 - Automatic'], fontsize=20, fontweight='bold')
    
    
    # Set the x-axis tick positions and labels at the top
    
    if metric =='dice':
        metric='DSC'
        axes[i].set_ylabel(metric,size=25,fontweight='bold')
        #axes[i].set_title(metric,size=25,fontweight='bold')
    elif metric=='precision.1':
        metric='Precision'
        axes[i].set_ylabel(metric,size=25,fontweight='bold')
        #axes[i].set_title(metric,size=25,fontweight='bold')
    else:
        
        metric='Recall'
        axes[i].set_ylabel(metric,size=25,fontweight='bold')
        #axes[i].set_title(metric,size=25,fontweight='bold')
    # Set x-axis labels
    axes[i].tick_params(axis='both', labelsize=20)





# Adjust the spacing between subplots

path_to_copy=os.path.join("C:/Users/RubenSilva/Desktop/Results/Results",'intervar')
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
fig.savefig('dice_recall_precision_new'+'.png',bbox_inches = 'tight')




