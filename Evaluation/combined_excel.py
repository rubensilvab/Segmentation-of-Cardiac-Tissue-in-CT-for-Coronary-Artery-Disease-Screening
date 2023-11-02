# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 00:29:39 2023

@author: RubenSilva
"""

import pandas as pd
import glob
import os

import os

# Set the directory path you want to search for CSV files
directory_path = "D:/RubenSilva/3d/hosp/split"

# Create an empty list to store the file paths of the CSV files
csv_files = []

# Walk through the directory and its subdirectories
for root, directories, files in os.walk(directory_path):
    # Loop through each file in the directory
    for file in files:
        # Check if the file has a .csv extension
        if file.endswith('.csv'):
            # If it does, add the file path to the csv_files list
            csv_files.append(os.path.join(root, file))


combined_data = pd.DataFrame()

for file in  csv_files:
    data = pd.read_csv(file)
    combined_data = combined_data.append(data)

combined_data.to_excel(os.path.join(directory_path,'combined_data.xlsx'), index=False)
