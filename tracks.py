# -*- coding: utf-8 -*-
"""
Created on Tue May 10 11:54:22 2022
@author: Tim
"""
import numpy as np
import numba

import math

import scattering_kinematics as sK  


class tracks:
  def clip_E_multiple(self):
    """
    clip the tracks based on multiple specified energies.
    Returns
    -------
    None.
    """
    
    new_arr = np.zeros((self.cumulative_counts_split[len(self.E_r)],self.data_split.shape[-1]))
    
    new_arr_shape = new_arr.shape
    len_Es = len(self.E_r)
    
    counter = 0
    
    self.E_r[self.E_r > self.max_E_allowed] = self.max_E_allowed
    
    for i,E in enumerate(self.E_r):
    
      index = int(self.sorted_arr[np.searchsorted(self.sorted_arr[:,2],E,side="left"),0])
      # print(E, self.data_split[index,2])

      cc_ = self.cumulative_counts_split[np.searchsorted(self.cumulative_counts_split, index,side="right")]
      
      if counter+(cc_-index) > new_arr_shape[0]:
        print("couldn't fit array in remaining space, extending...")
        
        new_arr = np.append(
          new_arr, np.zeros( (len_Es*new_arr_shape[0]//i,new_arr_shape[1]) ), axis=0
        )
        new_arr_shape = new_arr.shape

      new_arr[counter:counter+(cc_-index),:] = self.data_split[index:cc_,:]
      new_arr[counter:counter+(cc_-index),1] = i +1
        
      counter += (cc_-index)
    
    self.data = new_arr[:counter,:]
    
    self.indexes, self.counts, self.cumulative_counts = get_counts_and_cumulative(self.data[:,1])
    
    # np.savetxt("data.csv",self.data,delimiter=",",fmt="%.4f")
    
  
  def clip_E(self,acceptance = 0.01):
    """
    Parameters
    ----------
    acceptance : float, optional
      Fractional acceptance of track energy 
      (e.g. if you ask to access the last 10 keV you should expect 
      some energies of 10.01 keV, 9.99 keV etc.) The default is 0.01 (1%).
    """
    arr_truth = self.data[:,2] <= self.E_r
    # can grab first energy above threshold:
    arr_truth2 = np.roll(arr_truth,-1)
    arr_truth = arr_truth | arr_truth2
    self.data = self.data[arr_truth]
    # get new counts and cumulative
    self.indexes, self.counts, self.cumulative_counts = get_counts_and_cumulative(self.data[:,1])
    
    too_high = self.data[self.cumulative_counts,2] >= self.E_r*(1+acceptance)
    too_low = self.data[self.cumulative_counts+1,2] <= self.E_r*(1-acceptance)
    clip_track = too_low & too_high
    # If the energy is too high, code it to -1 so it can be removed later
    self.data[self.cumulative_counts[too_high],2] = -1
    # If the second energy is too low, we don't want the track at all
    if len(self.cumulative_counts[clip_track]):
      for c,cc in zip(self.counts[clip_track],self.cumulative_counts[clip_track]):
        self.data[cc:cc+c,2] = -1
    
    self.data = self.data[self.data[:,2] >= 0,:]
    self.indexes, self.counts, self.cumulative_counts = get_counts_and_cumulative(self.data[:,1])
    
  
  def rot_track(self,  
                histogram = None, specify_angles = None, 
                m = 1.0086649, E_n = 2.47e3):
    """
    Method for rotating the tracks to their energy-appropriate angle based on an incident projectile.
    
    Parameters
    ----------
    
    histogram : tuple of arrays (n, bins), optional
      Tuple of the direct output of matplotlib/numpy histogram of recoil angles (n, bin edges).
      The default is None.
    specify_angles : np.ndarray(np.float64), optional
      Specific recoil angles for use with multiple recoil energies. 
      Must have same length as self.E_r. The default is None.
    m : float, optional
      Mass of incident particle (in u). The default is 1.0086649.
    E_n : float, optional
      Energy of incident particle (keV). The default is 2.45e3.
    Returns
    -------
    None.
    """
    # Check whether one E_r or multiple:
    # print(self.data_split.shape)
    # want the initial direction to be based on energy:
    # Two options: Elastic, or histogram
    # Histogram needs to generate an array of angles instead of 
    if histogram == None:
      if self.energy_check:
        # Elastic scattering only:
        # Non-relativistic:
        # cos_sq_alpha = self.E_r*(self.proj_m+m)*(self.proj_m+m)/(4*m*self.proj_m*E_n)
        # Relativistic: (needs change of units to proj_m and m)
        cos_sq_alpha = self.E_r*(E_n+self.proj_m*931493.6148+m*931493.6148)*(E_n+self.proj_m*931493.6148+m*931493.6148)/(
                        (2*m*931493.6148+self.E_r)*(E_n*E_n + 2*self.proj_m*931493.6148*E_n))
        cos_alpha = math.sqrt(cos_sq_alpha)
        sin_alpha = math.sqrt(1-cos_sq_alpha)
      else:
        if (isinstance(specify_angles, np.ndarray) or isinstance(specify_angles, list)):
          # elastic + inelastic:
          cos_alphas = np.cos(specify_angles)
          sin_alphas = np.sqrt(1- cos_alphas*cos_alphas)
        else: # only elastic scattering:
          cos_sq_alphas = self.E_r*(self.proj_m+m)*(self.proj_m+m)/(4*m*self.proj_m*E_n)
          cos_alphas = np.sqrt(cos_sq_alphas)
          sin_alphas = np.sqrt(1-cos_sq_alphas)
        
    else:
      assert self.energy_check, "If using multiple recoil energies, histogram must be None."
      assert (isinstance(histogram,tuple) or isinstance(histogram,list)), "Histogram must be of format (n, bin_edges)."
      BIN_CENTRES = (histogram[1][1:]+histogram[1][:-1])/2

      CDF = np.cumsum(histogram[0])
      CDF /= CDF[-1]

      gen = np.random.rand(len(self.counts))
      cos_alphas = np.cos(BIN_CENTRES[np.searchsorted(CDF,gen)])
      sin_alphas = np.sqrt(1- cos_alphas*cos_alphas) # quicker than np.sin
      
    # The angle perpendicular to the beam is isotropic:
    phis = 2*math.pi*np.random.rand(len(self.counts))
    
    for i in range(len(self.counts)):
      count = self.counts[i]
      cumulative_count = self.cumulative_counts[i]
      
      if (histogram is not None) or (not self.energy_check):
        cos_alpha = cos_alphas[i]
        sin_alpha = sin_alphas[i]
      
      data_ = self.data[cumulative_count:cumulative_count + count,:]
        
      data_[:,3:6] = data_[:,3:6]-data_[0,3:6] # translate to 0,0,0
      
      x_i = np.array([cos_alpha,
                      sin_alpha*math.cos(phis[i]),
                      sin_alpha*math.sin(phis[i])])

      data_[:,3:6] = sK.rot_secondary_to_primary(data_[1,9:12],x_i,data_[:,3:6].T).T
      
      data_[0,9:12] = x_i
      
      data_[1:,9:12] = (data_[1:,3:6]-data_[:-1,3:6])/np.expand_dims(data_[1:,8],axis=1)
      
      
  def prepare_secondaries(self):
    self.num_secondaries = np.sum(self.arr_truth)

    self.secondaries = np.nan*np.zeros((self.num_secondaries,self.data.shape[1]+3)) # +3 for x_iplus1
    self.secondaries[:,:-3] = self.data[self.arr_truth,:]
    if self.arr_truth[-1]:
      self.secondaries[:,-3:] = np.roll(self.data[np.roll(self.arr_truth,1),9:12],-1,axis=0)
    else:
      self.secondaries[:,-3:] = self.data[np.roll(self.arr_truth,1),9:12]
  
  
  def iter_sec_tracks(self):
    all_sec_tracks = np.nan*np.zeros((50*len(self.secondaries),self.DATA[0].shape[1]+2)) # bit of a hacky *50
    counter = 0 # counter for filling in all_sec_tracks

    for i in range(len(self.secondaries)):
      index = int(self.secondaries[i,14])
      E_prim = self.secondaries[i,2]
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
      
      if (x_iplus1 == x_iplus1).all():
        # alternatively, if the 'primary' came to a complete stop, 
        # keep x_i as direction of secondary.
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
      
      nan_check = all_sec_tracks[:,9] != all_sec_tracks[:,9] # xdir is nan
      arr_truth[nan_check] = False # make sure the first points don't have a big recoil from previous interaction
      num_terts = np.sum(arr_truth)
      
      terts = np.nan*np.zeros((num_terts,self.data.shape[1]+3)) # +3 for x_iplus1
      terts[:,:-3] = all_sec_tracks[arr_truth,:-2]
      
      if arr_truth[-1]:
        terts[:,-3:] = np.roll(all_sec_tracks[np.roll(arr_truth,1),9:12],-1,axis=0)
      else:
        terts[:,-3:] = all_sec_tracks[np.roll(arr_truth,1),9:12]
      
      if np.all(terts[:,9:12]==terts[:,-3:],axis=1).any():
        pass
        # print(sum(np.all(terts[:,9:12]==terts[:,-3:],axis=1)))
        # np.savetxt("all_sec_tracks.csv",all_sec_tracks,delimiter=",",fmt="%.4f")
        # np.savetxt("terts.csv",terts,delimiter=",",fmt="%.4f")
          
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
    
    if not self.energy_check:
      self.sorted_arr = self.data_split[np.argsort(self.data_split[:,2]),:]
    
    for i in range(self._num_splits):
      
      if not self.energy_check:
        
        if self._num_splits != 1:
          self.E_r = self.E_r_split[i]
        self.clip_E_multiple()
      
      else:
        self.data = self.data_split[i]
        self.counts = self.counts_split[i]
        self.cumulative_counts = self.cumulative_counts_split[i]
        self.cumulative_counts = self.cumulative_counts - self.cumulative_counts[0]
        if self.E_r < self.max_E_allowed:
          # need to only grab relevant energies
          self.clip_E()
      
      if self.rotate:
        self.rot_track()
        
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
               E_r_split = None, rotate = True, E_threshold=0.01):
    
    self.E_r = E_r
    self.E_r_split = E_r_split
    self.energy_check = (isinstance(self.E_r, float) or isinstance(self.E_r, int))
    self.data_split = data_split
    if E_r_split is not None:
      self._num_splits = len(self.E_r_split)
    elif isinstance(self.data_split, list):
      self._num_splits = len(self.data_split)
    else:
      self._num_splits = 1
    
    self.max_E_allowed = max_E_allowed
    self.proj_m = proj_m
    
    self.DATA = DATA
    self.MASSES = MASSES
    self.COUNTS = COUNTS
    self.CUMULATIVE_COUNTS = CUMULATIVE_COUNTS
    
    self.counts_split, self.cumulative_counts_split = counts_split, cumulative_counts_split
    self.rotate = rotate
    self.E_threshold = E_threshold
    
    

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
  indexes_list = [indexes]
  counts_list = [counts]
  cumulative_counts_list = [cumulative_counts]
  
  for sec_track in list_of_all_tracks:
    indexes, counts, cumulative_counts = get_counts_and_cumulative(sec_track[:,1])
    indexes_list.append(indexes)
    counts_list.append(counts)
    cumulative_counts_list.append(cumulative_counts)

  return indexes_list,counts_list, cumulative_counts_list
