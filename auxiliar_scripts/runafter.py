# -*- coding: utf-8 -*-
"""
Created on Mon May  1 13:04:04 2023

@author: RubenSilva
"""


import time

print("totl")

import subprocess

only_peri=False
a=1
b=2

print("Waiting for 20 seconds...")
time.sleep(20)
print("Done!")



if only_peri:
# run the second script
   locals().clear()
   # delete all variables in the current namespace
   for var in list(locals()):
       del locals()[var]
       
   runfile('runafter.py')
# start the second script in a new process
