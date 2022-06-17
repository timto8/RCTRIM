# -*- coding: utf-8 -*-
"""
Created on Fri May 13 13:04:53 2022

@author: Tim Marley
"""
import RCTRIM
import numpy as np

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
run_RCTRIM = True

CDF = np.genfromtxt("DD_C(n,n)_CDF.txt")
gen = np.random.rand(5000)
Es = CDF[np.searchsorted(CDF[:,1],gen),0]+ np.random.normal(0,1,len(gen))
# Es = Es[Es<467]
# Es = Es[Es>1]
Es = Es[Es<700]
Es = Es[Es>1]

RCTRIM_params = {
  "dir_in_str": "D:/2022-05-16-RCTRIM/NRs/DD_CF4_1/", # NR directory
  # "dir_in_str": "/run/media/timmarley/My Passport/2022-05-16-RCTRIM/NRs/DD_CF4/", # NR directory
  "proj": "C", # primary species
  "E_r": Es, # Primary ion starting energy/energies [keV], will infer from file if None.
  "batch_size": 50, # Number of tracks to calculate simultaneously.
  
  "event_viewer": False, # This is very slow and requires open3d to be installed.
  # event_viewer displays the location of the NRs. 
  # Colour is based on position/directionality, (x,y,z) = (r,g,b).
  
  "migdal": False, # Whether to make Migdal events.
  "e_energy": 5.9, # Electron energy [keV].
  "e_dir_in_str": "D:/2022-05-16-RCTRIM/electrons/", # Directory containing electron files.
  # "e_dir_in_str": "/run/media/timmarley/My Passport/2022-05-16-RCTRIM/electrons/", # Directory containing electron files.
  
  "W": 0.0345, # W-value (energy for ionisation) [keV].
  
  "random_xy_offset": True, # Whether to randomly shift the track in an 8x8 window.
  
  "dz_shift": True, # Offset the z-coordinate such that z > 0 & convert to time.
  "drift_velocity": 0.013, # Drift velocity [cm/ns] to convert dz to dt.
  
  "diff_T": 0.026, # Transverse diffusion [cm/cm**0.5].
  "diff_L": 0.016, # Longitudinal diffusion [cm/cm**0.5].
  
}

save_dir = "D:/2022-05-16-RCTRIM/save_dir/C/"
# save_dir = "/run/media/timmarley/My Passport/2022-05-16-RCTRIM/save_dir/test/"


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
    # data, list_of_all_tracks,il,cl,ccl = RC.save_tracks(save_dir)
    RC.save_tracks(save_dir)
  
  
  if run_just_electron:
    RCTRIM.save_electron_tracks(**just_electron_params)
  
