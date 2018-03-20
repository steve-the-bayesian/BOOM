#!/usr/local/bin/python3

import glob
import shutil

# Convert the files in the Boom R package back to a functioning C++ library, by
# moving the headers in with the source files, and movinng the Interface files
# to their own directory.

header_files = glob.iglob('Boom/inst/include/**/*.hpp', recursive = True)
for fname in header_files:
    if 'Nonparametric' in fname:
        continue
    elif 'r_interface' in fname:
        source = fname
        dest = fname.replace('Boom/inst/include/r_interface/', 'Boom/src/Interfaces/R/')
        shutil.move(source, dest)
    else:
        source = fname
        dest = fname.replace('Boom/inst/include/', 'Boom/src/')
        shutil.move(source, dest)

interface_files = glob.iglob('Boom/src/*.cpp')

# Move the interface files from the top level source directory to the
# Boom/src/Interfaces/R directory.
for fname in interface_files:
    source = fname
    dest = fname.replace('Boom/src/', 'Boom/src/Interfaces/R/')
    shutil.move(source, dest)

