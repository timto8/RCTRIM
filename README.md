# RCTRIM (Recursive Cascade TRIM)
Python add-on for TRIM.
Takes information from EXYZ.txt and COLLISION.txt files to put together full nuclear recoil cascades and generate ionisation.


RCTRIM requires modified versions of the EXYZ.txt and COLLISION.txt files, but includes the conversion tool necessary to reformat them.

# srim_conversions.py
- The EXYZ.txt and COLLISION.txt files must be placed in directories with a specific structure. 
- You must know every element present in your TRIM simulation; knowledge of the relative abundances are not necessary.

#
If you have only carbon and fluorine in your simulation (e.g. studying nuclear recoils in CF4), then you should create two directories:
- C/
- F/

Now you should: 
- Run TRIM with carbon as your primary projectile and put the corresponding EXYZ.txt & COLLISION.txt in the C/ directory.
- Do the same with fluorine in the F/ directory.

#
If you have argon, carbon, and fluorine in your simulation (e.g. studying nuclear recoils in Ar/CF4 mixture), then you should create three directories:
- C/
- F/
- Ar/

Now you should: 
- Run TRIM with carbon as your primary projectile and put the corresponding EXYZ.txt & COLLISION.txt in the C/ directory.
- Do the same with fluorine in the F/ directory.
- Do the same with argon in the Ar/ directory.

#
In the srim_conversions.py file there is a function: convert_all_and_save(dir_in_str)

The parameter dir_in_str is a string with the file path to the folder which contains the folders we created earlier.

#
The function will run and create three new files:
- COLLISON_REDUCED.txt (intermediate file, removed in pipeline)
- collision.npy
- EXYZ.npy

You can now remove the EXYZ.txt and COLLISION.txt files if you wish to save some space.
