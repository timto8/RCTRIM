# -*- coding: utf-8 -*-
"""
Created on Wed May 18 12:09:08 2022

@author: Tim
"""
import TRIM_conversions as tc
import DEGRAD_conversions as dc

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
save_TRIM = False

# Directory should always contain separate folders named after each element:
trim_file_dir = "D:/2022-05-16-RCTRIM/NRs/DD_CF4/"

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
save_DEGRAD = True

degrad_file_dir = "F:/2022-05-16-RCTRIM/photoelectrons/raw/"
# degrad_file_dir = "/run/media/timmarley/My Passport/2022-05-16-RCTRIM/electrons/raw_sec_NR_es/"
# degrad_file_name = "data1.OUT"#"DEGRAD.OUT"

electron_save_dir = "F:/2022-05-16-RCTRIM/photoelectrons/npy/"
# electron_save_dir = "/run/media/timmarley/My Passport/2022-05-16-RCTRIM/electrons/sec_NR_es/"
# electron_energy = 5.9

Is = list(range(1,4))
Es = [f'5.25{i}' for i in range(1,4)]


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

if __name__ == "__main__":
  
  if save_TRIM:
    
    tc.convert_all_and_save(trim_file_dir)
  
  
  if save_DEGRAD:
    for i,E in zip(Is,Es):
      degrad_file_name = f"data{i}.OUT"#"DEGRAD.OUT"
      electron_energy = E
      dc.convert(load_name = degrad_file_name, load_dir = degrad_file_dir,
                 energy = electron_energy, save_dir = electron_save_dir)
    
