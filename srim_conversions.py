# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 14:38:02 2022

@author: Tim
"""
import numpy as np
import numba
import os
import time
import threading

import elements

def convert_collision_file(dir_in_str):
  with open(f"{dir_in_str}/COLLISON.txt", "r") as collision:
    with open(f"{dir_in_str}/COLLISON_REDUCED.txt", "w") as coll:
    
      for line in collision:
        try:
          num = int(line[1:6]) # Check line is the correct line:
          write_line = line.replace("³"," ")
          write_line = write_line[1:17] + write_line[-43:-30] +"\n"

          for elem in elements.elements.keys():
            
            if elem in write_line:
              write_line = write_line.replace(f"{elem} ".rjust(3),f" {elements.elements[elem]}")
              break
          
          coll.write(write_line)
          
        except:
          continue
        

def convert_colls_to_numpys(dir_in_str,elems):
  
  for elem in elems:
    convert_collision_file(f"{dir_in_str}/{elem}/")
  
  colls = [np.genfromtxt(f"{dir_in_str}/{elem}/COLLISON_REDUCED.txt",dtype=np.float64) for elem in elems]

  for i, elem in enumerate(elems):
    np.save(f"{dir_in_str}/{elem}/collision.npy",colls[i])
  

def find_indexes(data_elem,data_list,masses):
  
  required_secs = data_elem[:,7]
  targets = data_elem[:,13]
  prims = [data[np.argsort(data[:, 2]),0:3] for data in data_list]
  
  for i in range(len(required_secs)):
    E_sec = required_secs[i]
    M = targets[i]
    
    if M != M: # Don't need secondary tracks for phonons.
      continue
    
    prim_Es = prims[masses.index(M)]
    
    index = np.searchsorted(prim_Es[:,2],E_sec,side="left")
    counter_index = prim_Es[index,0]
    
    data_elem[i,14] = counter_index
  
  return None
  

def convert_to_np(dir_in_str, elem="C"):

  data = np.genfromtxt(f"{dir_in_str}/{elem}/EXYZ.txt", skip_header=15, dtype=np.float64)
  data = data[(data[:,-1] > 1) | (data[:,-1] == 0),:] # keep hits above 1 eV
  unique, counts = np.unique(data[:,0], return_counts=True)
  new_data = np.zeros((len(data),15))
  
  cumulative_counts = np.cumsum(counts)
  
  cumulative_counts[1:] = cumulative_counts[:-1]
  cumulative_counts[0] = 0
  # Convert to keV:
  data[:,5] /= 1000
  data[:,6] /= 1000
  
  new_data[:,0] = np.arange(len(data))
  # Place into bigger array
  new_data[:,1:8] = data
  
  # Calculate distances from previous
  new_data[1:,8] = np.linalg.norm(new_data[1:,3:6]-new_data[:-1,3:6],axis=1)
  new_data[cumulative_counts,8] = 0
  
  # Direction:
  new_data[1:,9:12] = (new_data[1:,3:6]-new_data[:-1,3:6])/new_data[1:,8].reshape((len(new_data)-1,1))
  new_data[cumulative_counts,9:12] = np.array([1,0,0]).reshape(1,3)
  
  # Angle to next
  new_data[:-1,12] = np.arccos(new_data[1:,9]*new_data[:-1,9] + (
                                      new_data[1:,10]*new_data[:-1,10])+ (
                                      new_data[1:,11]*new_data[:-1,11])  )
  new_data[cumulative_counts-1,12] = 0
  
  collision = np.load(f"{dir_in_str}/{elem}/collision.npy")
  
  M = np.zeros(len(data))
  j = 0
  for i in range(len(M)):
      if data[i,-1] == collision[j,-1]/1000 and j < len(collision)-1:
          M[i] = collision[j,-2]
          j += 1
      else:
          M[i] = np.nan


  new_data[:,13] = M
  
  return new_data


def convert_all_and_save(dir_in_str):

  elems = os.listdir(dir_in_str)
  
  convert_colls_to_numpys(dir_in_str, elems)
  
  data = [convert_to_np(dir_in_str, elem) for elem in elems]
  masses = [elements.elements[elem] for elem in elems]
  
  # Find the indexes of relevant secondaries: 
  for data_elem in data:
    
    find_indexes(data_elem,
                  data,
                  masses)  

  for elem,data_elem in zip(elems,data):
    np.save(f"{dir_in_str}/{elem}/EXYZ.npy",data_elem)
  