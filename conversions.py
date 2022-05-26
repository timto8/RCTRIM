# -*- coding: utf-8 -*-
"""
Created on Wed May 18 12:09:08 2022

@author: Tim
"""
import TRIM_conversions as tc
import DEGRAD_conversions as dc

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
save_TRIM = True

# Directory should always contain separate folders named after each element:
trim_file_dir = "D:/2022-05-16-RCTRIM/NRs/DD_CF4/"

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
save_DEGRAD = False

degrad_file_dir = "D:/2022-05-16-RCTRIM/electrons/raw/"
degrad_file_name = "DEGRAD.OUT"

electron_save_dir = "D:/2022-05-16-RCTRIM/electrons/"
electron_energy = 5.9 

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

if __name__ == "__main__":
  
  if save_TRIM:
    
    tc.convert_all_and_save(trim_file_dir)
  
  
  if save_DEGRAD:
    
    dc.convert(load_name = degrad_file_name, load_dir = degrad_file_dir,
               energy = electron_energy, save_dir = electron_save_dir)
    
