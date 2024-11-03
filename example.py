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
# gen = np.random.rand(10000)

# Es = CDF[np.searchsorted(CDF[:,1],gen),0] + (CDF[1,0]-1e-3)*np.random.rand(len(gen))
# Es = Es[(Es<=470)&(Es>=1)]
# Es = Es[(Es<=700)&(Es>=1)]
# Es = Es[(Es<=18)&(Es>=8)]
# C_lims = np.array([11.1166, 12.0438, 12.971, 14.8254, 16.6798, 18.5342, 20.3886, 22.243])
# C_nums = np.array([1646, 1387, 1603, 1380, 1403, 847, 742])

# F_lims = np.array([13.8358, 14.9733, 16.1108, 18.3858, 20.6607, 22.9357, 25.2106, 27.4856])
# F_nums = np.array([9085, 4866, 8167, 6198, 5032, 4098, 2607])

# lims = {'C':C_lims,'F':F_lims}
# nums = {'C':C_nums,'F':F_nums}

Es = 170
# carbon needs 11-25 keV
# fluorine needs 14-35.5 keV

RCTRIM_params = {
  "dir_in_str": "F:/2022-05-16-RCTRIM/NRs/DD_CF4/", # NR directory
  # "dir_in_str": "/run/media/timmarley/My Passport/2022-05-16-RCTRIM/NRs/DD_CF4/", # NR directory
  "proj": "F", # primary species
  "E_r": Es, # Primary ion starting energy/energies [keV], will infer from file if None.
  "batch_size": 1, # Number of tracks to calculate simultaneously.
  
  "event_viewer": True, # This is very slow and requires open3d to be installed.
  # event_viewer displays the location of the NRs. 
  # Colour is based on position/directionality, (x,y,z) = (r,g,b).
  
  "migdal": True, # Whether to make Migdal events.
  "e_energy": 5.9, # Electron energy [keV].
  "e_dir_in_str": "F:/2022-05-16-RCTRIM/electrons/", # Directory containing electron files.
  # "e_dir_in_str": "/run/media/timmarley/My Passport/2022-05-16-RCTRIM/electrons/", # Directory containing electron files.
  
  "W": 0.0342, # W-value (energy for ionisation) [keV].
  
  "random_xy_offset": False, # Whether to randomly shift the track in an 8x8 window.
  
  "dz_shift": True, # Offset the z-coordinate such that z > 0 & convert to time.
  "drift_velocity": 0.013, # Drift velocity [cm/ns] to convert dz to dt.
  
  "diff_T": 0.026, # Transverse diffusion [cm/cm**0.5].
  "diff_L": 0.016, # Longitudinal diffusion [cm/cm**0.5].
  
}

# save_dir = "F:/2022-05-16-RCTRIM/save_dir/F/"
# save_dir = "/run/media/timmarley/My Passport/2022-05-16-RCTRIM/save_dir/test/"


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
run_just_electron = False

# electron_Es = np.arange(2.2,12.1,0.2)

just_electron_params = {
  "E": 5.25, # Electron energy [keV].
  "dir_in_str": "F:/2022-05-16-RCTRIM/photoelectrons/npy/", # Directory containing electron files.
  # "dir_in_str": "/run/media/timmarley/My Passport/2022-05-16-RCTRIM/electrons/sec_NR_es",
  "num": 100, # Number of tracks.
  "save_dir": "F:/2022-05-16-RCTRIM/photoelectrons/drifted/", 
  # "save_dir": "/run/media/timmarley/My Passport/2022-05-16-RCTRIM/save_dir/e/",
  
  "dz_shift": True, # Offset the z-coordinate such that z > 0 & convert to time.
  "drift_velocity": 0.013, # Drift velocity [cm/ns] to convert dz to dt.
  
  "diff_T": 0.026, # Transverse diffusion [cm/cm**0.5].
  "diff_L": 0.016, # Longitudinal diffusion [cm/cm**0.5].
  "fname": 'z_photoelectron_5.25keV.npy', # specific filename
}

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
if __name__ == "__main__":
  
  if run_RCTRIM:
    # Initialise
    for elem in ['F','C']:
      RCTRIM_params["proj"] = elem
      save_dir = f"F:/2022-05-16-RCTRIM/save_dir/{elem}/"
      
      RC = RCTRIM.RCTRIM(**RCTRIM_params)
      RC.save_tracks(save_dir)
      del RC
      
      # for i in range(len(nums[elem])):
      #   Es = np.random.uniform(lims[elem][i],lims[elem][i+1],nums[elem][i])
      #   RCTRIM_params["E_r"] = Es
        
      #   RC = RCTRIM.RCTRIM(**RCTRIM_params)
      
      #   # Iterate and save
      #   # data, list_of_all_tracks,il,cl,ccl = RC.save_tracks(save_dir)
      #   RC.save_tracks(save_dir)
      #   del RC
  
  
  if run_just_electron:
    # for E in electron_Es:
    #   just_electron_params["E"] = E
      
    RCTRIM.save_electron_tracks(**just_electron_params)
  
