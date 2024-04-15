### NOTE: BOOM is primarily built using 'bazel' instead of 'make'.  This
### Makefile could build and install BOOM as of January 2018, but it may be out
### of date.

all:	libboom.a

# When compiling remotely (e.g. on CRAN's winbuilder) flags to be
# passed to make can be specified here.
# -k:  keep going
# -j 8: use 8 threads
# MAKEFLAGS=" -k -j 8 "

CFLAGS = -I. -I./Bmath -I./math/cephes -DADD_ -O3
CPPFLAGS = -I. -I./Bmath -I./math/cephes -std=c++11 -DADD_ -O3

############################################################################
# Begin the list of all the BOOM source files.

TOP_HDRS := $(wildcard *.hpp)

BART_SRCS :=
# BART_SRCS := $(wildcard Models/Bart/*.cpp) \
# 	  $(wildcard Models/Bart/PosteriorSamplers/*.cpp)
BART_HDRS :=

DISTRIBUTION_SRCS := $(wildcard distributions/*.cpp)
DISTRIBUTION_HDRS := distributions.hpp $(wildcard distributions/*.hpp)

GLM_SRCS := $(wildcard Models/Glm/*.cpp Models/Glm/PosteriorSamplers/*.cpp)
GLM_HDRS := $(wildcard Models/Glm/*.hpp Models/Glm/PosteriorSamplers/*.hpp)

HIERARCHICAL_SRCS := $(wildcard Models/Hierarchical/*.cpp) \
		  $(wildcard Models/Hierarchical/PosteriorSamplers/*.cpp)
HIERARCHICAL_HDRS := $(wildcard Models/Hierarchical/*.hpp) \
		  $(wildcard Models/Hierarchical/PosteriorSamplers/*.hpp)

HMM_SRCS := $(wildcard Models/HMM/*.cpp) \
	 $(wildcard Models/HMM/Clickstream/*.cpp) \
	 $(wildcard Models/HMM/Clickstream/PosteriorSamplers/*.cpp) \
	 $(wildcard Models/HMM/PosteriorSamplers/*.cpp)
HMM_HDRS := $(wildcard Models/HMM/*.hpp) \
	 $(wildcard Models/HMM/Clickstream/*.hpp) \
	 $(wildcard Models/HMM/Clickstream/PosteriorSamplers/*.hpp) \
	 $(wildcard Models/HMM/PosteriorSamplers/*.hpp)

R_INTERFACE_SRCS := $(wildcard *.cpp)

IRT_SRCS := $(wildcard Models/IRT/*.cpp)
IRT_HDRS := $(wildcard Models/IRT/*.hpp)

LINALG_SRCS := $(wildcard LinAlg/*.cpp)
LINALG_HDRS := $(wildcard LinAlg/*.hpp)

MATH_SRCS := $(wildcard math/cephes/*.cpp) \
	     $(wildcard math/*.cpp)
MATH_HDRS := $(wildcard math/cephes/*.hpp) \
	     $(wildcard math/*.hpp)

MIXTURE_SRCS := $(wildcard Models/Mixtures/*.cpp) \
	     $(wildcard Models/Mixtures/PosteriorSamplers/*.cpp)
MIXTURE_HDRS := $(wildcard Models/Mixtures/*.hpp) \
	     $(wildcard Models/Mixtures/PosteriorSamplers/*.hpp)

MODELS_SRCS := $(wildcard Models/*.cpp Models/Policies/*.cpp) \
	$(wildcard Models/PosteriorSamplers/*.cpp)
MODELS_HDRS := $(wildcard Models/*.hpp Models/Policies/*.hpp) \
	$(wildcard Models/PosteriorSamplers/*.hpp)

NUMOPT_SRCS := $(wildcard numopt/*.cpp)
NUMOPT_HDRS := $(wildcard numopt/*.hpp)

POINTPROCESS_SRCS := $(wildcard Models/PointProcess/*.cpp) \
		  $(wildcard Models/PointProcess/PosteriorSamplers/*.cpp)
POINTPROCESS_HDRS := $(wildcard Models/PointProcess/*.hpp) \
		  $(wildcard Models/PointProcess/PosteriorSamplers/*.hpp)

RMATH_SRCS := $(wildcard Bmath/*.cpp)
RMATH_HDRS := $(wildcard Bmath/*.hpp)

SAMPLERS_SRCS := $(wildcard Samplers/*.cpp Samplers/Gilks/*.cpp)
SAMPLERS_HDRS := $(wildcard Samplers/*.hpp Samplers/Gilks/*.hpp)

STATESPACE_SRCS := $(wildcard Models/StateSpace/*.cpp) \
	$(wildcard Models/StateSpace/Filters/*.cpp) \
	$(wildcard Models/StateSpace/PosteriorSamplers/*.cpp) \
	$(wildcard Models/StateSpace/StateModels/*.cpp) \
	$(wildcard Models/StateSpace/StateModels/PosteriorSamplers/*.cpp)
STATESPACE_HDRS := $(wildcard Models/StateSpace/*.hpp) \
	$(wildcard Models/StateSpace/Filters/*.hpp) \
	$(wildcard Models/StateSpace/PosteriorSamplers/*.hpp) \
	$(wildcard Models/StateSpace/StateModels/*.hpp) \
	$(wildcard Models/StateSpace/StateModels/PosteriorSamplers/*.hpp)

STATS_SRCS := $(wildcard stats/*.cpp)
STATS_HDRS := $(wildcard stats/*.hpp)

TARGETFUN_SRCS := $(wildcard TargetFun/*.cpp)
TARGETFUN_HDRS := $(wildcard TargetFun/*.hpp)

TIMESERIES_SRCS := $(wildcard Models/TimeSeries/*.cpp) \
		$(wildcard Models/TimeSeries/PosteriorSamplers/*.cpp)
TIMESERIES_HDRS := $(wildcard Models/TimeSeries/*.hpp) \
		$(wildcard Models/TimeSeries/PosteriorSamplers/*.hpp)

UTIL_SRCS := $(wildcard cpputil/*.cpp)
UTIL_HDRS := $(wildcard cpputil/*.hpp)

CXX_SRCS = ${BART_SRCS} \
	${DISTRIBUTION_SRCS} \
	${GLM_SRCS} \
	${HIERARCHICAL_SRCS} \
	${HMM_SRCS} \
	${IRT_SRCS} \
	${LINALG_SRCS} \
	${MATH_SRCS} \
	${MIXTURE_SRCS} \
	${MODELS_SRCS} \
	${NUMOPT_SRCS} \
	${POINTPROCESS_SRCS} \
        ${R_INTERFACE_SRCS} \
	${RMATH_SRCS} \
	${SAMPLERS_SRCS} \
	${STATS_SRCS} \
	${STATESPACE_SRCS} \
	${TARGETFUN_SRCS} \
	${TIMESERIES_SRCS} \
	${UTIL_SRCS}

CXX_HDRS = ${TOP_HDRS} \
	${BART_HDRS} \
	${DISTRIBUTION_HDRS} \
	${GLM_HDRS} \
	${HIERARCHICAL_HDRS} \
	${HMM_HDRS} \
	${IRT_HDRS} \
	${LINALG_HDRS} \
	${MATH_HDRS} \
	${MIXTURE_HDRS} \
	${MODELS_HDRS} \
	${NUMOPT_HDRS} \
	${POINTPROCESS_HDRS} \
        ${R_INTERFACE_HDRS} \
	${RMATH_HDRS} \
	${SAMPLERS_HDRS} \
	${STATS_HDRS} \
	${STATESPACE_HDRS} \
	${TARGETFUN_HDRS} \
	${TIMESERIES_HDRS} \
	${UTIL_HDRS} \

CXX_STD = CXX11

OPT_BUILD_DIR := opt
OBJECTS = ${CXX_SRCS:.cpp=.o}

HDRS = ${CXX_HDRS}

# End list of BOOM source files
############################################################################

libboom.a: ${OBJECTS}
	   ${AR} rcs $@ $^

.PHONY: install
install: libboom.a
	install libboom.a /usr/local/lib
	rm -rf /usr/local/include/BOOM
	mkdir -p /usr/local/include/BOOM
	./install/install_headers.py $(HDRS) /usr/local/include/BOOM
	./install/install_headers.py `find ./Eigen -name "*" | gsed 's|\.\/||g' | awk 'NR > 1{print}'` /usr/local/include/BOOM

.PHONY: clean
clean:
	rm ${OBJECTS} libboom.a

## Remove all build objects and uninstall the library.
.PHONY: distclean
distclean: clean
	rm -f /usr/local/lib/libboom.a
	rm -rf /usr/local/include/BOOM
