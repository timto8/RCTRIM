# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 15:14:46 2022

@author: Tim
"""
import numpy as np
import random

def drift_tracks(x, y, z,
                 drift = None,
                 diff_T = 0.026, diff_L = 0.016,
                 catch_collision_with_GEM = True):
  """
  Parameters
  ----------
  x,y,z : np.ndarray(np.float64), shape (n,)
    x,y,z positions of ionisation in cm.
  drift : float, optional
    Distance from z=0 to 1 mm above GEM. The default is None.
  diff_T : float, optional
    Transverse diffusion in gas. The default is 0.026 cm/cm**0.5.
  diff_L : float, optional
    Longitudinal diffusion in gas. The default is 0.016 cm/cm**0.5.
  catch_collision_with_GEM : bool, optional
    Whether to remove ionisation which hits the GEM surface. The default is True.

  Returns
  -------
  xyzdxdydzs : np.ndarray(np.float64), shape (n,6)
    x,y,z positions of ionisation before and after diffusion.

  """

  if drift==None:
    drift = 0.4+random.random()*2 # (random between 0.4 & 2.4)

  if catch_collision_with_GEM:
    x, y, z = x[z>-drift], y[z>-drift], z[z>-drift]
  
  x_drift,y_drift,z_drift = x.copy(),y.copy(),z.copy()
  
  DT = abs(diff_T * np.sqrt(abs(drift + z)) )
  DL = abs(diff_L * np.sqrt(abs(drift + z)) )
  z_drift += np.random.normal(0,DL,len(x))
  x_drift += np.random.normal(0,DT,len(x))
  y_drift += np.random.normal(0,DT,len(x)) 
      
  xyzdxdydzs = np.stack((x,y,z,x_drift,y_drift,z_drift),axis=-1)
  
  return xyzdxdydzs



if __name__ == '__main__':
  # generating PSF:
    for i in range(100):
      x = (np.random.rand()-0.5)*np.ones(100)
      y = (np.random.rand()-0.5)*np.ones(100)
      z = np.zeros(100)
      for drift in np.arange(0.4,2.5,0.2):
        xyzdxdydzs = drift_tracks(x,y,z,drift = drift)
        xyzdxdydzs[:,-1] -= np.nanmin(xyzdxdydzs[:,-1]) # change to delta_z
        xyzdxdydzs[:,-1] = xyzdxdydzs[:,-1] / 0.013
        
        np.savetxt(f'D:/2022-05-16-RCTRIM/PSF/raw/{drift:.1f}cm_{i}.txt',xyzdxdydzs,fmt='%.4f')
    
