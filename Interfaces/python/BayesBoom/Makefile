### NOTE: This makefile is intended for limited use during package development.
### The primary build utility for BayesBoom is the setup.py script that is part
### of setuptools.

all:	BayesBoom.cpython-37m-darwin.so

# When compiling remotely (e.g. on CRAN's winbuilder) flags to be
# passed to make can be specified here.
# -k:  keep going
# -j 8: use 8 threads
# MAKEFLAGS=" -k -j 8 "

# CFLAGS = -I. -I./Bmath -I./math/cephes -DADD_ -O3 -Wall -fPIC
CPPFLAGS = -O3 -Wall  -std=c++11 -fPIC -I'/usr/local/include/BOOM' `python3 -m pybind11 --includes` 

SRCS := module.cpp $(wildcard */*.cpp)
OBJECTS = ${SRCS:.cpp=.o}
	echo "srcs= ${SRCS}"

BayesBoom.cpython-37m-darwin.so: ${OBJECTS}
	c++ -Wall -shared -std=c++11 -fPIC -undefined dynamic_lookup $^ -oBayesBoom`python3-config --extension-suffix` -L/usr/local/lib -lboom

.PHONY: clean
clean:
	rm *.o *.so */*.o
