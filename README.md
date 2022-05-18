# RCTRIM (Recursive Cascade TRIM)
Python add-on for TRIM (used in MIGDAL experiment).
Takes information from EXYZ.txt and COLLISION.txt files to put together full nuclear recoil cascades and generate ionisation.
Allows for the approximation of Migdal events by adding output from DEGRAD.

RCTRIM requires modified versions of the EXYZ.txt and COLLISION.txt files, but includes the conversion tool necessary to reformat them.

# TRIM_conversions.py
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
In the TRIM_conversions.py file there is a function: convert_all_and_save(dir_in_str)

The parameter dir_in_str is a string with the file path to the folder which contains the folders we created earlier.

#
The function will run and create three new files:
- COLLISON_REDUCED.txt (intermediate file, removed in pipeline)
- collision.npy
- EXYZ.npy

You can now remove the EXYZ.txt and COLLISION.txt files if you wish to save some space.

# DEGRAD_conversions.py

DEGRAD's IWRITE=1 output is useful as it contains the x,y,z,t information of each thermalised electron, but the formatting isn't too friendly. It's much more convenient to convert this output file to a numpy binary file for quick loading.

DEGRAD_conversions.py contains a single function: convert(load_name, load_dir, energy, save_dir)

This function takes four parameters:
- load_name : Name of DEGRAD file being loaded. 
- load_dir : Directory containing DEGRAD file.
- energy : Energy of electron in file.
- save_dir : Directory to save the output file in.

Calling this function will save the DEGRAD output in a more workable numpy array with shape (n, 6). The name of the file will be: electron_{energy}keV.npy.
The 6 columns are:
- id : The track number (separates different events).
- x,y,z : The x,y,z positions of each thermalised electron [cm].
- t : The timing information of each thermalised electron [ps].
- flag : 1, fluorescence; 2, pair production; 3, bremsstrahlung; 0, otherwise.

You can now remove the original DEGRAD output file if you wish to save some space.

# conversions.py

This is a very short script which is laid out to conveniently run the TRIM and DEGRAD conversions. Just change the variables at the top of the file to suit your situation.

# RCTRIM.py

