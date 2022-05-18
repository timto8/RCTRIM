# -*- coding: utf-8 -*-
"""
Created on Mon May 16 15:58:52 2022

@author: Tim
"""
import numpy as np

def convert(load_name,load_dir="./",energy=None,save_dir="./"):
  """
  Convert DEGRAD output file to a numpy array

  Parameters
  ----------
  load_name : str
    Name of DEGRAD file being loaded.
  load_dir : str, optional
    Directory containing DEGRAD file. The default is "./".
  energy : float, optional
    Energy of electron in file. 
    Save file must have format: "electron_{energy}keV.npy". 
    The default is None.
  save_dir : str, optional
    Directory in which to save the output file. The default is "./".

  Returns
  -------
  tracks : np.ndarray(np.float64)
    columns: track id, x, y, z, t, flag

  """
  tracks = []
  with open(f"{load_dir}{load_name}","r") as f:
    
    for i, line in enumerate(f):
      
      if i%2==0:
        num_es = int(line[21:42])
        track = np.zeros((num_es,6))
        
      else:
        index = i//2
        track[:,0] = index
        
        for e_num in range(num_es):
          j = e_num * 167
          
          track[e_num,1] = float(line[j:j+26])
          track[e_num,2] = float(line[j+26:j+52])
          track[e_num,3] = float(line[j+52:j+78])
          track[e_num,4] = float(line[j+78:j+104])
          
          # code to 0,1,2,3:
          f = int(line[j+104:j+125])
          pp = line[j+145:j+146]
          b = line[j+166:j+167]
          
          if f != 0:
            track[e_num,5] = 1.
          elif pp != "0":
            track[e_num,5] = 2.
          elif b != "0":
            track[e_num,5] = 3.
            
          # alternative is already zero
        
        track = track[np.argsort(track[:,4]),:] # sort by time
        
        if abs(track[0,1]) > 25 or abs(track[0,2]) > 25 or abs(track[0,3]) > 25:
          track[:,1:4] -= track[0:1,1:4] + np.random.normal(0,5,(1,3))
        
        tracks.append(track)
  
  tracks = np.concatenate(tracks)
  tracks[:,1:4] /= 1e4 # convert um to cm
  
  if energy != None:
    np.save(f"{save_dir}/electron_{energy}keV.npy",tracks)

  return tracks




