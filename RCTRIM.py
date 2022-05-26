# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 16:17:35 2021

@author: Tim
"""
import os
import time
import random
import numpy as np
import numba

import drift
import elements
import tracks
    
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


class RCTRIM:
  def load_NRs(self):
    self._elems = tuple(os.listdir(self.dir_in_str))
    self._masses = tuple(elements.elements[elem] for elem in self._elems)
    self._data = tuple(np.load(f"{self.dir_in_str}/{elem}/EXYZ.npy") for elem in self._elems)
    self._proj_index = self._elems.index(self.proj)
    self._proj_data = self._data[self._proj_index].copy()
  
  
  def load_ERs(self):
    self._e_data = np.load(f"{self.e_dir_in_str}/electron_{self.e_energy}keV.npy")[:,0:4] # i,x,y,z,t,flag
    self._e_indexes, self._e_counts, self._e_cumulative_counts = tracks.get_counts_and_cumulative(self._e_data[:,0])
  
  
  def set_ICCs(self):
    self._INDEXES = [0 for i in range(len(self._elems))]
    self._COUNTS = [0 for i in range(len(self._elems))]
    self._CUMULATIVE_COUNTS = [0 for i in range(len(self._elems))]
    
    for i in range(len(self._elems)):
      self._INDEXES[i], self._COUNTS[i], self._CUMULATIVE_COUNTS[i] = tracks.get_counts_and_cumulative(self._data[i][:,1])
    
    self._INDEXES = tuple(self._INDEXES)
    self._COUNTS = tuple(self._COUNTS)
    self._CUMULATIVE_COUNTS = tuple(self._CUMULATIVE_COUNTS)
    
    
  def split_batches(self):
    E_r_check = isinstance(self.E_r, np.ndarray) or isinstance(self.E_r, list)
    if (not self._batch_size) or E_r_check:
      self._data_split = self._proj_data
      self._counts_split = self._COUNTS[self._proj_index]
      self._cumulative_counts_split = self._CUMULATIVE_COUNTS[self._proj_index]
      
      if self._batch_size:
        split_indices = [self._batch_size*(i+1) for i in range(len(self.E_r)//self._batch_size - 1)]
        self.E_r_split = np.split(self.E_r,split_indices)  
      else:
        self.E_r_split = None
    
    else: # Only split these arrays if using a single E_r and there's a batch size
      self._data_split = np.split(self._proj_data,self._CUMULATIVE_COUNTS[self._proj_index][self._batch_size::self._batch_size],axis=0)
      icc_split = [self._batch_size*(i+1) for i in range(len(self._COUNTS[self._proj_index])//self._batch_size)]
      self._counts_split = np.split(self._COUNTS[self._proj_index],icc_split,axis=0)
      self._cumulative_counts_split = np.split(self._CUMULATIVE_COUNTS[self._proj_index],icc_split,axis=0)
      self.E_r_split = None
      
  
  def save_tracks(self, save_dir=""):
    counter = 0
    E = self.E_r
    
    for data, list_of_all_tracks, prim_Es, elec_Es in self.track_gen:
      counts_list, cumulative_counts_list = tracks.get_counts_lists(data, list_of_all_tracks)
      
      for track_no in range(len(cumulative_counts_list[0])):
    
        data_secs = [track[cumulative_counts_list[i+1][track_no]:cumulative_counts_list[i+1][track_no] + counts_list[i+1][track_no], :-2] for i, track in enumerate(list_of_all_tracks) if (track_no < len(counts_list[i+1]))]
        data_ = np.concatenate(
          [data[cumulative_counts_list[0][track_no]:cumulative_counts_list[0][track_no] + counts_list[0][track_no],:]] + data_secs,
          axis=0)
        
        
        num_electrons = int(np.nansum((data_[:-1,6]*data_[1:,8])/self.W))
        
        track_ioni = generate_ionisation(data_,num_electrons,self.W)/1e8 # convert to cm
      
        if self.migdal:
          e_index = random.randint(0,len(self._e_counts)-1)
          e_c = self._e_counts[e_index]
          e_cc = self._e_cumulative_counts[e_index]
          e_data_ = self.e_data[e_cc:e_cc+e_c,1:]
          track_ioni = np.concatenate([track_ioni,e_data_],axis=0)
          
        try:
          
          drift_length = (2*random.random()+0.5)
          xyzdxdydzs = drift.drift_tracks(track_ioni[:,0], 
                                          track_ioni[:,1], 
                                          track_ioni[:,2],
                                          drift=drift_length,
                                          diff_T = self.diff_T, diff_L = self.diff_L)
          
          if self.dz_shift:
            xyzdxdydzs[:,-1] -= np.nanmin(xyzdxdydzs[:,-1]) # change to delta_z
            xyzdxdydzs[:,-1] = xyzdxdydzs[:,-1] / self.drift_velocity
          
          if isinstance(self.E_r, np.ndarray) or isinstance(self.E_r, list):
            E = f"{self.E_r[counter]:.3f}"
          
          if self.migdal:
            file_name = f"{save_dir}/{E}keV_{self.proj}_{self.e_energy}keV_e_{drift_length:.3f}cm_{counter}.txt"
          else:
            file_name = f"{save_dir}/{E}keV_{self.proj}_{drift_length:.3f}cm_{counter}.txt"
          
          if self.random_xy_offset:
            xyzdxdydzs[:,[0,3]] += random.rand(-4,4)
            xyzdxdydzs[:,[1,4]] += random.rand(-4,4)
  
          np.savetxt(file_name,xyzdxdydzs,fmt="%.4f")
          counter += 1
          
        except ValueError: # if the array is empty, skip to the next one.
          continue
        
  
  def __init__(self, dir_in_str, proj, 
               E_r = None, batch_size = None,
               migdal = False, e_energy=5.9, e_dir_in_str="./",
               W = 0.0345, dz_shift = True, random_xy_offset = False,
               drift_velocity = 0.013, diff_T = 0.026, diff_L = 0.016):
    
    self.dir_in_str = dir_in_str
    self.proj = proj
    self.proj_m = elements.elements[self.proj]
    self.E_r = E_r
    self._batch_size = batch_size
    self.migdal = migdal
    
    self.W = W
    self.dz_shift = dz_shift
    self.random_xy_offset = random_xy_offset
    self.drift_velocity = drift_velocity
    self.diff_T = diff_T
    self.diff_L = diff_L
    
    self.load_NRs()
    
    if self.migdal:
      self._e_energy = e_energy
      self.e_dir_in_str = e_dir_in_str
      self.load_ERs()
    
    self.set_ICCs()
    
    self.split_batches()
    
    self.max_E_allowed = self._proj_data[0,2] # the first value is the largest.
    if self.E_r is None:
      self.E_r = self.max_E_allowed
    
    assert np.all(self.E_r <= self.max_E_allowed)
    
    self.tracks = tracks.tracks(self.E_r, self._data_split, self.max_E_allowed, self.proj_m, 
                                self._counts_split, self._cumulative_counts_split,
                                self._data, self._masses, self._COUNTS, self._CUMULATIVE_COUNTS,
                                E_r_split = self.E_r_split, rotate = True, E_threshold=0.01)
    
    self.track_gen = self.tracks.cascade()
    
    
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


@numba.njit
def generate_ionisation(data,num_electrons,W=0.0342,fano=0.28):
  """
  Parameters
  ----------
  data : np.ndarray 
    Array of secondary NR sites.
  num_electrons : int
    estimated number of electrons needed.
  W : float, optional
    Work function for gas. The default is 0.034.

  Returns
  -------
  np.ndarray
    array of electron positions.

  """
  
  track = np.nan*np.zeros((2*num_electrons,3))
  counter = 0
  
  ends_of_tracks = np.where(data[:,9]!=data[:,9])[0]
  starting_point = 0
  for end in ends_of_tracks:
    data_section = data[starting_point:end,:]
    
    # the starting points from where to place the electrons:
    starts = data_section[:-1,3:6] 
    # the direction vectors:
    dirs = data_section[1:,9:12] 
    # distances:
    dists = data_section[1:,8] 
    # electrons in each gap:
    electrons_each_stage = (data_section[:-1,6]+data_section[1:,6])/2 * dists / W
    
    for i in range(len(electrons_each_stage)):
      n = round(np.random.poisson(electrons_each_stage[i]/fano)*fano)
      d = dists[i]
      direction = dirs[i]
      start = starts[i]
      # ds for each electron in the line:
      ds = np.linspace(0.,d,n+2)[1:-1].reshape(n,1)
      
      if counter+n > len(track):
        print("Not enough space, had to extend array...")
        track = np.append(track, np.nan*np.zeros((2*num_electrons,3)),axis=0)
      
      track[counter:counter+n,:] = ds*direction + start
      
      counter += n
    
    starting_point = end
    
  return track[track[:,0]==track[:,0]]
    
    
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


def save_electron_tracks(E, dir_in_str, save_dir="./", dz_shift = True, 
                         drift_velocity = 0.013, diff_T = 0.026, diff_L = 0.016):
    
    data_E = np.load(f"{dir_in_str}/electron_{E}keV.npy")[:,0:4] # i,x,y,z,t,flag

    indexes, counts, cumulative_counts = tracks.get_counts_and_cumulative(data_E[:,0])
    
    drifts = 2*np.random.rand(len(counts))+0.4
    
    for track_no in range(len(counts)):
      c = counts[track_no]
      cc = cumulative_counts[track_no]
      ioni = data_E[cc:cc+c,1:]
      ioni = ioni[ioni[:,0]==ioni[:,0],:]
      
      drift_length = drifts[track_no]
      xyzdxdydzs = drift.drift_tracks(ioni[:,0], ioni[:,1], ioni[:,2],
                                      drift=drift_length,
                                      diff_T = diff_T, diff_L = diff_L)
      
      track_no_str = f"{track_no}"
      
      if dz_shift:
        try:
          xyzdxdydzs[:,-1] -= np.nanmin(xyzdxdydzs[:,-1]) # change to delta_z
          xyzdxdydzs[:,-1] = xyzdxdydzs[:,-1] / drift_velocity
          
        except:
          print(track_no, "empty")
          continue
          
      np.savetxt(f"{save_dir}/{E}keV_e_{drift_length:.3f}cm_{track_no_str}.txt",xyzdxdydzs,fmt="%.4f")
     
  

  