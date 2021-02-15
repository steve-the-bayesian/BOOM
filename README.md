# BOOM
A C++ library for Bayesian modeling, mainly through Markov chain Monte Carlo, but with a few other methods supported.
BOOM = "Bayesian Object Oriented Modeling".  It is also the sound your computer makes when it crashes.

The BOOM project began around 2005 while I was on the faculty at USC.  It continued during my time at Google.  Google claims
copyright on modifications to BOOM files between 2009 and 2017.  Copyright for changes made either before or after that time
is claimed by Steven L. Scott.

## Installing the library
The primary build system for the BOOM C++ library is bazel.
```
bazel build boom
bazel build -c opt boom
```
If you have difficulty building the library, check local BUILD files for compiler and linker options that don't work on your system.  `-lpthread` is required on linux but prohibited on Mac.  If anyone knows a portable way around this I'm open to suggestions.  There is also a Makefile which might be a bit out of date.

Most people interested in BOOM probably care about the R packages.  To build the Boom package, you will want to run the install script
```
./install/create_boom_rpackage -i
```
from the project root directory.  This will install the Boom package on your machine and create the Boom package in the rpackage/ directory.  Once Boom is installed
```
./install/boom_spike_slab -i
./install/bsts -i
```
will install the BoomSpikeSlab and bsts packages.

These scripts involve lots of copying files around.  Some of that is handled by a python script `install_headers.py`.  Please make sure the hashbang line at the top of that file points to the version of python you want to use.

## BOOM and Python
I have played around with pybind11 bindings for BOOM with the intent of exposing the library in Python.  These should be considered experimental, but I'd love help fleshing this bit out.
```
./install/pyboom
```
is probably what you're looking for.
