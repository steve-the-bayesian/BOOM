#!/usr/bin/env bash

# Install and run bsts under the docker images with sanitizers installed.

# See https://github.com/wch/r-debug

# Install all the prerequisite packages.  This takes a long time, because some
# of these packages need to compile.

# RD -e "install.packages(c('zoo', 'xts', 'MASS', 'testthat', 'mlbench', 'igraph', 'lattice'))"

cd /home/steve/code/BOOM

# The packages must be created and installed in separate commands, without using
# the -i option to the scripts because the executable is called RD, not R.

# Next time you run these, be sure to update the version numbers.

./install/create_boom_rpackage
# BOOM=`ls rpackage/Boom_*tar.gz | sort -n`

MAKEVARS="-j 16" RD CMD INSTALL rpackage/Boom_0.9.9.tar.gz

./install/boom_spike_slab
MAKEVARS="-j 16" RD CMD INSTALL rpackage/BoomSpikeSlab_1.2.5.tar.gz

./install/bsts
MAKEVARS="-j 16" RD CMD INSTALL rpackage/bsts_0.9.8.tar.gz

RD CMD check rpackage/bsts_0.9.8.tar.gz
