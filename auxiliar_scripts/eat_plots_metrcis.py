# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 15:36:13 2023

@author: RubenSilva
"""
import os
#os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.6/bin")

import numpy as np 
from matplotlib import pyplot as plt
import matplotlib 
import numpy as np
import glob
import cv2
import time
import pandas as pd
from scipy.stats import pearsonr

os.chdir('C:/Users/RubenSilva/Desktop/baltman')  


"R1"
path_eat="X:/Ruben/TESE/New_training_Unet/Models_only_pub_dset/Mix_datasets/All_data/predict/2.5D_Unet/Dice_loss/Hospital_tif/L0_W2000_calc_augm_tif/EAT_segm_nHU/Volume Results"
eat_volume= (pd.read_excel(os.path.join(path_eat,'volumes_eat_r1auto.xlsx')))

"R2"
path_eat_r2="C:/Users/RubenSilva/Desktop/segmentation_inter_intra/selection/inverted/EAT_segm_nHU/Volume Results"
eat_volume_r2=(pd.read_excel(os.path.join(path_eat_r2,'volumes_eat_r2.xlsx')))

"R1 intra"
path_eat_r3="C:/Users/RubenSilva/Desktop/segmentation_inter_intra/selection/inverted/EAT_segm_nHU/Volume Results"
eat_volume_r3=(pd.read_excel(os.path.join(path_eat_r3,'volumes_eat_r3.xlsx')))
eat_volume_r1=(pd.read_excel(os.path.join(path_eat_r3,'volumes_eat_r1.xlsx')))


"""Restringir para os 20 pacientes"""
#patients=np.unique(eat_volume_r2['Patient'])
#eat_volume=eat_volume[eat_volume['Patient'].isin(patients)]

eat_manual,eat_pred=eat_volume_r1['Volume EAT manual (cm^3)'],eat_volume_r2['Volume EAT manual (cm^3)']
eat_manual,eat_pred=np.array(eat_manual),np.array(eat_pred)

# Calculate the Pearson Correlation Coefficient (PCC)
pcc, p_value = pearsonr(eat_manual, eat_pred)
print("Pearson Correlation Coefficient (PCC):", pcc)

# Calculate the linear regression line
slope, intercept = np.polyfit(eat_manual, eat_pred, 1)

# Plot the scatter plot
plt.scatter(eat_manual, eat_pred)
plt.plot(eat_manual, eat_manual ,linestyle='--', color='gray', label='Linear Regression Line')
plt.xlabel('Reader 1 EAT Volume ($cm^3$)', fontsize=12 )
plt.ylabel('Reader 2 EAT Volume ($cm^3$)', fontsize=12)
#plt.title('Scatter Plot of EAT Volumes')
# Add horizontal gridlines
plt.grid(axis='y', linestyle='--', linewidth=0.5, color='gray')

# Add the PCC value to the plot as text
plt.text(0.7, 0.15, f'PCC = {pcc:.3f}', transform=plt.gca().transAxes, fontsize=12)


plt.savefig('PCC_eat_volumes_inter_v2.png',dpi=500)
plt.show()

#Balt-Altman

# Calculate the difference between manual and predicted EAT volumes
diff_eat = eat_pred - eat_manual

# Calculate the mean difference and the limits of agreement
mean_diff = np.mean(diff_eat)
upper_limit = mean_diff + 1.96 * np.std(diff_eat)
lower_limit = mean_diff - 1.96 * np.std(diff_eat)

# Plot the Bland-Altman plot
plt.scatter((eat_manual), diff_eat)
plt.axhline(y=mean_diff, color='red', linestyle='--')
plt.axhline(y=upper_limit, color='black', linestyle='--')
plt.axhline(y=lower_limit, color='black', linestyle='--')

# Add the text for upper and lower limits of agreement
plt.text(190, upper_limit +0.8, f'{upper_limit:.2f}', color='black')
plt.text(190, lower_limit +0.8, f'{lower_limit:.2f}', color='black')
plt.text(190, mean_diff +0.8, f'{mean_diff:.2f}', color='red')


plt.xlabel('Reader 1 ($cm^3$)', fontsize=12)
plt.ylabel('Reader 2 - Reader 1 EAT Volume ($cm^3$)', fontsize=12)
plt.grid(axis='y', linestyle='--', linewidth=0.5, color='gray')

#plt.title('Bland-Altman Plot of EAT Volumes')
#plt.legend()

# Set a higher DPI for the saved image (e.g., 300)
plt.savefig('bland_altman_plot_inter_v2.png', dpi=500)

# Show the plot (optional)
plt.show()

