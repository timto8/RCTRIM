# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 15:42:32 2021

@author: Tim
"""
import numpy as np
# import math
import numba


def targetMass(m, E_r, E_n, theta):
  """
  Parameters
  ----------
  m : float
    Mass of projectile.
  E_r : float
    KE of secondary (target) ejectile.
  E_n : float
    KE of projectile.
  theta : float
    Angle deflection of projectile after recoil.

  Returns
  -------
  M : float
    Mass of target.

  """
  M = m*(-1 + 2* (E_n - np.cos(theta) * np.sqrt(E_n*(E_n - E_r))) / E_r )
  
  return M

@numba.njit
def alpha(m,M,E_r,E_n):
  """
  Parameters
  ----------
  m : float
    Mass of projectile.
  M : float
    Mass of target.
  E_r : float
    KE of secondary (target) ejectile.
  E_n : float
    KE of projectile.

  Returns
  -------
  float
    Scattering angle of secondary NR (elastic).

  """
  return np.arccos(np.sqrt( (m+M)*(m+M)*E_r / (4*m*M*E_n) ))


@numba.njit
def rot_matrix(alpha,x,y,z):
  """
  Parameters
  ----------
  alpha : float
    Angle of rotation.
  x,y,z : float
    normalised x,y,z component of arbitrary vector.

  Returns
  -------
  np.ndarray(np.float64)
    Rotation matrix for rotating about arbitrary vector (x,y,z).

  """
  c = np.cos(alpha)
  s = np.sin(alpha)
  
  return np.array([[c+x*x*(1-c) , x*y*(1-c)-z*s , x*z*(1-c)+y*s],
                   [x*y*(1-c)+z*s , c+y*y*(1-c) , y*z*(1-c)-x*s],
                   [x*z*(1-c)-y*s , y*z*(1-c)+x*s , c+z*z*(1-c)]])

@numba.njit
def rot_secondary_to_primary(y,x,s_track):
  """
  
  Parameters
  ----------
  y : np.ndarray(np.float64), shape (3,)
    Initial direction vector of secondary track
  x : np.ndarray(np.float64), shape (3,)
    Direction vector of primary section to rotate track to
  s_track : np.ndarray(np.float64), shape (3,n)
    x,y,z positions of secondary track

  Returns
  -------
  rotated secondary along position of primary section

  """
  angle = np.arccos(np.dot(y,x))
  u = np.cross(y,x)/np.sin(angle)
  
  MAT = rot_matrix(angle, u[0],u[1],u[2])
  
  rot = np.zeros(s_track.shape)

  rot[0,:] = MAT[0,0]*s_track[0,:] + MAT[0,1]*s_track[1,:] + MAT[0,2]*s_track[2,:]
  rot[1,:] = MAT[1,0]*s_track[0,:] + MAT[1,1]*s_track[1,:] + MAT[1,2]*s_track[2,:]
  rot[2,:] = MAT[2,0]*s_track[0,:] + MAT[2,1]*s_track[1,:] + MAT[2,2]*s_track[2,:]
                
  return rot
  

@numba.njit
def rot_secondary_to_recoil_pos(alpha, x_i, x_iplus1, s_track):
  """
  Parameters
  ----------
  alpha : float
    Scattering angle of secondary NR (elastic).
  x_i : np.ndarray(np.float64), shape (3,)
    Normalised direction vector leading to site of NR.
  x_iplus1 : np.ndarray(np.float64), shape (3,)
    Normalised direction vector leaving from site of NR.
  s_track : np.ndarray(np.float64), shape (3,n)
    x,y,z positions of secondary track.

  Returns
  -------
  rot : np.ndarray(np.float64), shape (3,n)
    x,y,z positions of secondary track now appropriately rotated.

  """
  u = np.cross(x_i, x_iplus1)
  
  u = u/np.sqrt(u[0]*u[0] + u[1]*u[1] + u[2]*u[2])
  
  MAT = rot_matrix(-alpha, u[0],u[1],u[2])
  
  rot = np.zeros(s_track.shape)
  
  rot[0,:] = MAT[0,0]*s_track[0,:] + MAT[0,1]*s_track[1,:] + MAT[0,2]*s_track[2,:]
  rot[1,:] = MAT[1,0]*s_track[0,:] + MAT[1,1]*s_track[1,:] + MAT[1,2]*s_track[2,:]
  rot[2,:] = MAT[2,0]*s_track[0,:] + MAT[2,1]*s_track[1,:] + MAT[2,2]*s_track[2,:]
            
  return rot


  
  
