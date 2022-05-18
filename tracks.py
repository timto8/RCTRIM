# -*- coding: utf-8 -*-
"""
Created on Tue May 10 11:54:22 2022

@author: Tim
"""
import numpy as np
import numba

import math
import random

import scattering_kinematics as sK  


class tracks:
  def clip_E(self):
    arr_truth = self.data[:,2] <= self.E_r
    # can grab first energy above threshold:
    arr_truth2 = np.roll(arr_truth,-1)
    arr_truth = arr_truth | arr_truth2
    self.data = self.data[arr_truth]
    # get new counts and cumulative
    self.indexes, self.counts, self.cumulative_counts = get_counts_and_cumulative(self.data[:,1])
  
  def rot_track(self, acceptance=0.01, histogram=None, m = 1.0086649, E_n = 2.45e3):
    if histogram == None:
      # Elastic scattering only:
      cos_sq_alpha = self.E_r*(self.proj_m+m)*(self.proj_m+m)/(4*m*self.proj_m*E_n)
      cos_alpha = math.sqrt(cos_sq_alpha)
      sin_alpha = math.sqrt(1-cos_sq_alpha)
    
    # print(self.cumulative_counts)
    for i in range(len(self.counts)):
      count = self.counts[i]
      cumulative_count = self.cumulative_counts[i]
      
      
      data_ = self.data[cumulative_count:cumulative_count + count,:]
      # print(cumulative_count, len(self.data))
    
      # If the energy is too high, code it to -1 so it can be removed later
      if data_[0,2] >= self.E_r*(1+acceptance):
        data_[0,2] = -1
        # If the second energy is too low, we don't want the track at all
        if data_[1,2] <= self.E_r*(1-acceptance):
          data_[1:,2] = -1
          continue
        
      data_[:,3:6] = data_[:,3:6]-data_[0,3:6] # translate to 0,0,0
      
      # want the initial direction to be based on energy:
      # Two options: Elastic, or histogram
      
      if histogram == None:
        phi = 2*math.pi*random.random()
        x_i = np.array([cos_alpha,
                        sin_alpha*math.cos(phi),
                        sin_alpha*math.sin(phi)])
      else:
        pass # Code to be added
      
      data_[:,3:6] = sK.rot_secondary_to_primary(data_[1,9:12],x_i,data_[:,3:6].T).T
      
      data_[0,9:12] = x_i
      
      data_[1:,9:12] = (data_[1:,3:6]-data_[:-1,3:6])/np.expand_dims(data_[1:,8],axis=1)
      
  def prepare_secondaries(self):
    self.num_secondaries = np.sum(self.arr_truth)

    self.secondaries = np.nan*np.zeros((self.num_secondaries,self.data.shape[1]+3)) # +3 for x_iplus1
    self.secondaries[:,:-3] = self.data[self.arr_truth,:]
    self.secondaries[:,-3:] = self.data[np.roll(self.arr_truth,1),9:12]
  
  def iter_sec_tracks(self):
    all_sec_tracks = np.nan*np.zeros((50*len(self.secondaries),self.DATA[0].shape[1]+2)) # bit of a hacky *50
    counter = 0 # counter for filling in all_sec_tracks

    for i in range(len(self.secondaries)):
      index = int(self.secondaries[i,14])
      E_prim = self.secondaries[i,2]
      # E = secondaries[i,7]
      M = self.secondaries[i,13]
      if M != M:
        continue
      alpha = self.alphas[i]
      x_i = self.secondaries[i,9:12]
      x_iplus1 = self.secondaries[i,-3:]
      sec_pos = self.secondaries[i,3:6]#.T
      track_index = self.secondaries[i,1]
      
      tuple_index = self.MASSES.index(M)
      
      count_location = np.searchsorted(self.CUMULATIVE_COUNTS[tuple_index]+self.COUNTS[tuple_index],index,side="right")
      
      track_end = self.CUMULATIVE_COUNTS[tuple_index][count_location]+self.COUNTS[tuple_index][count_location]
      sec_track = self.DATA[tuple_index][index:track_end,:].copy()
      
      if len(sec_track) == 1:
        continue
      
      sec_track[:,3:6] = sec_track[:,3:6]-sec_track[0,3:6] # translate to 0,0,0
      
      sec_track[:,3:6] = sK.rot_secondary_to_primary(sec_track[1,9:12],x_i,sec_track[:,3:6].T).T
      
      sec_track[:,3:6] = sK.rot_secondary_to_recoil_pos(alpha, x_i, x_iplus1, sec_track[:,3:6].T).T

      
      sec_track[:,3:6] = sec_track[:,3:6] + sec_pos
      
      sec_track[0,9:12] = np.nan # previous direction is nan
      
      sec_track[1:,9:12] = (sec_track[1:,3:6]-sec_track[:-1,3:6])/np.expand_dims(sec_track[1:,8],axis=1)
      
      if counter+len(sec_track) > len(all_sec_tracks):
        print("Not enough space in all_sec_tracks, had to extend array...")
        all_sec_tracks = np.append(all_sec_tracks, np.nan*np.zeros((len(sec_track)*(len(self.secondaries)-i),self.DATA[0].shape[1]+2)), 0)
      all_sec_tracks[counter:counter+len(sec_track),:-2] = sec_track
      
      all_sec_tracks[counter:counter+len(sec_track),-2] = M # identify recoil type
      all_sec_tracks[counter:counter+len(sec_track),-1] = E_prim
      all_sec_tracks[counter:counter+len(sec_track),1] = track_index # index of track
      counter += len(sec_track)
      
    self.list_of_all_tracks.append(all_sec_tracks[:counter,:])
    
  def tertiary_loop(self):
    while np.sum(self.list_of_all_tracks[-1][:,7] > self.E_threshold):
      all_sec_tracks = self.list_of_all_tracks[-1]
      
      arr_truth = all_sec_tracks[:,7] > self.E_threshold
      
      nan_check = all_sec_tracks[:,9] != all_sec_tracks[:,9]
      arr_truth[nan_check] = False # make sure the first points don't have a big recoil from previous interaction
      
      num_terts = np.sum(arr_truth)
      
      terts = np.nan*np.zeros((num_terts,self.data.shape[1]+3)) # +3 for x_iplus1
      terts[:,:-3] = all_sec_tracks[arr_truth,:-2]
      terts[:,-3:] = all_sec_tracks[np.roll(arr_truth,1),9:12]
          
      self.alphas = sK.alpha(all_sec_tracks[arr_truth,-2], 
                             terts[:,13], 
                             terts[:,7], 
                             terts[:,2]+terts[:,7])
      
      self.secondaries = terts
      
      self.iter_sec_tracks()
      
      if len(self.list_of_all_tracks[-1]) == 0:
        self.list_of_all_tracks = self.list_of_all_tracks[:-1]
        break
      
      self.tert_indexes, self.tert_counts, self.tert_cumulative_counts = get_counts_and_cumulative(self.list_of_all_tracks[-1][:,1])
      
      self.elec_Es.append(calc_electron_stop(self.list_of_all_tracks[-1], self.tert_counts, self.tert_cumulative_counts, self.num_points))
      
    
  def cascade(self):
    
    for i in range(len(self.data_split)):
      
      self.data = self.data_split[i]      
      self.counts = self.counts_split[i]
      self.cumulative_counts = self.cumulative_counts_split[i]
      self.cumulative_counts = self.cumulative_counts - self.cumulative_counts[0]
      
      if self.E_r < self.max_E_allowed:
        # need to only grab relevant energies
        self.clip_E()
        
      if self.rotate:
        self.rot_track()
        
      self.data = self.data[self.data[:,2] >= 0,:]
      
      self.arr_truth = self.data[:,7] > self.E_threshold
      
      self.arr_truth[self.cumulative_counts] = False
      
      self.list_of_all_tracks = []
    
      self.prepare_secondaries()
      
      self.alphas = sK.alpha(self.proj_m, 
                             self.secondaries[:,13], 
                             self.secondaries[:,7], 
                             self.secondaries[:,2]+self.secondaries[:,7])
      
      self.iter_sec_tracks()
      
      self.sec_indexes, self.sec_counts, self.sec_cumulative_counts = get_counts_and_cumulative(self.list_of_all_tracks[-1][:,1])
      
      self.num_points = len(self.counts)
      
      self.elec_Es = []
    
      # Primaries:
      self.elec_Es.append(
        calc_electron_stop(self.data, self.counts, self.cumulative_counts, self.num_points)
      )
      # Secondaries:
      self.elec_Es.append(
        calc_electron_stop(self.list_of_all_tracks[-1], self.sec_counts, self.sec_cumulative_counts, self.num_points)
      )
      
      self.tertiary_loop()
        
      self.prim_Es = self.elec_Es[0]
      self.sum_elec_Es = np.nansum(self.elec_Es,axis=0)
      
      yield self.data, self.list_of_all_tracks, self.prim_Es, self.sum_elec_Es
  
  def __init__(self, E_r, data_split, max_E_allowed, proj_m, 
               counts_split, cumulative_counts_split,
               DATA, MASSES, COUNTS, CUMULATIVE_COUNTS,
               rotate = True, E_threshold=0.01):
    
    self.E_r = E_r
    self.data_split = data_split
    self.max_E_allowed = max_E_allowed
    self.proj_m = proj_m
    
    self.DATA = DATA
    self.MASSES = MASSES
    self.COUNTS = COUNTS
    self.CUMULATIVE_COUNTS = CUMULATIVE_COUNTS
    
    self.counts_split, self.cumulative_counts_split = counts_split, cumulative_counts_split
    self.rotate = rotate
    self.E_threshold = E_threshold
    
    # print(self.data_split)
    

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

    
@numba.njit
def calc_electron_stop(all_sec_tracks, counts, cumulative_counts, num_points):
  elec_E = np.nan*np.zeros(num_points)
  for i in range(len(counts)):
    # index = int(indexes[i])-1
    count = counts[i]
    cumulative_count = cumulative_counts[i]
    
    track = all_sec_tracks[cumulative_count:cumulative_count+count,:]

    delta_E = track[:-1,2]-track[1:,2]
    E_R = track[1:,7]
    E_e = delta_E - E_R
    elec_E[i] = 0
    for e in E_e:
      if e > 0:
        elec_E[i] = elec_E[i] + e
        
  return elec_E


@numba.njit
def get_counts_and_cumulative(data):
  """
  Useful function for extracting the indexes of each individual
  track within the data array. Helps a lot with speed.

  Parameters
  ----------
  data : np.ndarray(np.float64)
    Array of track IDs.

  Returns
  -------
  unique : np.ndarray(np.float64)
    Unique track indexes.
  counts : np.ndarray(np.int32)
    Length of each individual track within data array.
  cumulative_counts : np.ndarray(np.int32)
    Starting index of each individual track in data array.

  """
  unique = [data[0]] # bit hacky, but start with first line.
  counts = [1] # start with a count of one.
  cumulative_counts = [0,1] # First index is zero
  for i in data[1:]:
    if i == unique[-1]:
      counts[-1] = counts[-1] + 1
      cumulative_counts[-1] = cumulative_counts[-1] + 1
    else:
      unique.append(i)
      counts.append(1)
      cumulative_counts.append(cumulative_counts[-1] + 1)
  
  return np.asarray(unique), np.asarray(counts), np.asarray(cumulative_counts[:-1])    


def get_counts_lists(data, list_of_all_tracks):
  
  indexes, counts, cumulative_counts = get_counts_and_cumulative(data[:,1])
  counts_list = [counts]
  cumulative_counts_list = [cumulative_counts]
  
  for sec_track in list_of_all_tracks:
    indexes, counts, cumulative_counts = get_counts_and_cumulative(sec_track[:,1])
    counts_list.append(counts)
    cumulative_counts_list.append(cumulative_counts)

  return counts_list, cumulative_counts_list