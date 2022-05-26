# -*- coding: utf-8 -*-
"""
Created on Fri May 13 13:04:53 2022

@author: Tim Marley
"""
import RCTRIM
import numpy as np

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
run_RCTRIM = True

RCTRIM_params = {
  "dir_in_str": "D:/2022-05-16-RCTRIM/NRs/DD_CF4/", # NR directory
  "proj": "F", # Fluorine as primary species
  "E_r": 400*np.random.rand(100)+50, # Primary ion starting energy/energies [keV], will infer from file if None.
  "batch_size": 10, # Number of tracks to calculate simultaneously.
  
  "migdal": False, # Don't make Migdal events.
  "e_energy": 5.9, # Electron energy [keV].
  "e_dir_in_str": "D:/2022-05-16-RCTRIM/electrons/", # Directory containing electron files.
  
  "W": 0.0345, # W-value (energy for ionisation) [keV].
  
  "random_xy_offset": False, # Whether to randomly shift the track in an 8x8 window.
  
  "dz_shift": True, # Offset the z-coordinate such that z > 0 & convert to time.
  "drift_velocity": 0.013, # Drift velocity [cm/ns] to convert dz to dt.
  
  "diff_T": 0.026, # Transverse diffusion [cm/cm**0.5].
  "diff_L": 0.016, # Longitudinal diffusion [cm/cm**0.5].
}

save_dir = "D:/2022-05-16-RCTRIM/save_dir/F/"


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
run_just_electron = False

just_electron_params = {
  "E": 5.9, # Electron energy [keV].
  "dir_in_str": "D:/2022-05-16-RCTRIM/electrons/", # Directory containing electron files.
  "save_dir": "D:/2022-05-16-RCTRIM/save_dir/e/", 
  
  "dz_shift": True, # Offset the z-coordinate such that z > 0 & convert to time.
  "drift_velocity": 0.013, # Drift velocity [cm/ns] to convert dz to dt.
  
  "diff_T": 0.026, # Transverse diffusion [cm/cm**0.5].
  "diff_L": 0.016 # Longitudinal diffusion [cm/cm**0.5].
}

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
if __name__ == "__main__":
  
  if run_RCTRIM:
    # Initialise
    RC = RCTRIM.RCTRIM(**RCTRIM_params)
    
    # Iterate and save
    RC.save_tracks(save_dir)
  
  
  if run_just_electron:
    RCTRIM.save_electron_tracks(**just_electron_params)
  
