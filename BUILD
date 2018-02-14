TOP_SRCS = glob(["*.cpp"])

TOP_HDRS = glob(["*.hpp"])

BMATH_HDRS = glob(["Bmath/*.hpp"])

BMATH_SRCS = glob(["Bmath/*.cpp"])

LINALG_SRCS = glob(["LinAlg/*.cpp"])

LINALG_HDRS = glob(["LinAlg/*.hpp"])

SAMPLER_SRCS = glob([
    "Samplers/*.cpp",
    "Samplers/Gilks/arms.cpp",
])

SAMPLER_HDRS = glob([
    "Samplers/*.hpp",
    "Samplers/Gilks/arms.hpp",
])

TARGETFUN_SRCS = glob(["TargetFun/*.cpp"])

TARGETFUN_HDRS = glob(["TargetFun/*.hpp"])

### Don't forget to exclude stuff that we don't want
CPPUTIL_SRCS = glob(["cpputil/*.cpp"])

CPPUTIL_HDRS = glob(["cpputil/*.hpp"])

DISTRIBUTIONS_SRCS = glob(["distributions/*.cpp"])

DISTRIBUTIONS_HDRS = glob(["distributions/*.hpp"])

MATH_SRCS = glob(["math/*.cpp"])

MATH_HDRS = glob(["math/*.hpp"])

MODELS_SRCS = glob([
    "Models/*.cpp",
    "Models/Policies/*.cpp",
    "Models/PosteriorSamplers/*.cpp",
])

MODELS_HDRS = glob([
    "Models/*.hpp",
    "Models/Policies/*.hpp",
    "Models/PosteriorSamplers/*.hpp",
])

NUMOPT_SRCS = glob(["numopt/*.cpp"])

NUMOPT_HDRS = glob(["numopt/*.hpp"])

STATS_SRCS = glob(["stats/*.cpp"])

STATS_HDRS = glob(["stats/*.hpp"])

GLM_SRCS = glob([
    "Models/Glm/*.cpp",
    "Models/Glm/PosteriorSamplers/*.cpp",
])

GLM_HDRS = glob([
    "Models/Glm/*.hpp",
    "Models/Glm/PosteriorSamplers/*.hpp",
])

HMM_SRCS = glob([
    "Models/HMM/*.cpp",
    "Models/HMM/PosteriorSamplers/*.cpp",
])

HMM_HDRS = glob([
    "Models/HMM/*.hpp",
    "Models/HMM/PosteriorSamplers/*.hpp",
])

HIERARCHICAL_SRCS = glob([
    "Models/Hierarchical/*.cpp",
    "Models/Hierarchical/PosteriorSamplers/*.cpp",
])

HIERARCHICAL_HDRS = glob([
    "Models/Hierarchical/*.hpp",
    "Models/Hierarchical/PosteriorSamplers/*.hpp",
])

IRT_SRCS = glob([
    "Models/IRT/*.cpp",
    "Models/IRT/PosteriorSamplers/*.cpp",
])

IRT_HDRS = glob([
    "Models/IRT/*.hpp",
    "Models/IRT/PosteriorSamplers/*.hpp",
])

MIXTURE_SRCS = glob([
    "Models/Mixtures/*.cpp",
    "Models/Mixtures/PosteriorSamplers/*.cpp",
])

MIXTURE_HDRS = glob([
    "Models/Mixtures/*.hpp",
    "Models/Mixtures/PosteriorSamplers/*.hpp",
])

POINT_PROCESS_SRCS = glob([
    "Models/PointProcesses/*.cpp",
    "Models/PointProcesses/PosteriorSamplers/*.cpp",
])

POINT_PROCESS_HDRS = glob([
    "Models/PointProcesses/*.hpp",
    "Models/PointProcesses/PosteriorSamplers/*.hpp",
])

STATE_SPACE_SRCS = glob([
    "Models/StateSpace/*.cpp",
    "Models/StateSpace/Filters/*.cpp",
    "Models/StateSpace/StateModels/*.cpp",
    "Models/StateSpace/PosteriorSamplers/*.cpp",
])

STATE_SPACE_HDRS = glob([
    "Models/StateSpace/*.hpp",
    "Models/StateSpace/Filters/*.hpp",
    "Models/StateSpace/StateModels/*.hpp",
    "Models/StateSpace/PosteriorSamplers/*.hpp",
])

TIMESERIES_SRCS = glob([
    "Models/TimeSeries/*.cpp",
    "Models/TimeSeries/PosteriorSamplers/*.cpp",
])

TIMESERIES_HDRS = glob([
    "Models/TimeSeries/*.hpp",
    "Models/TimeSeries/PosteriorSamplers/*.hpp",
])

BOOM_SRCS = TOP_SRCS + \
            BMATH_SRCS + \
            LINALG_SRCS + \
            SAMPLER_SRCS + \
            TARGETFUN_SRCS + \
            CPPUTIL_SRCS + \
            DISTRIBUTIONS_SRCS + \
            MATH_SRCS + \
            MODELS_SRCS + \
            NUMOPT_SRCS + \
            STATS_SRCS + \
            GLM_SRCS + HMM_SRCS + \
            HIERARCHICAL_SRCS + \
            IRT_SRCS + \
            MIXTURE_SRCS + \
            POINT_PROCESS_SRCS + \
            STATE_SPACE_SRCS + \
            TIMESERIES_SRCS

BOOM_HDRS = TOP_HDRS + \
            BMATH_HDRS + \
            LINALG_HDRS + \
            SAMPLER_HDRS + \
            TARGETFUN_HDRS + \
            CPPUTIL_HDRS + \
            DISTRIBUTIONS_HDRS + \
            MATH_HDRS + \
            MODELS_HDRS + \
            NUMOPT_HDRS + \
            STATS_HDRS + \
            GLM_HDRS + HMM_HDRS + \
            HIERARCHICAL_HDRS + \
            IRT_HDRS + \
            MIXTURE_HDRS + \
            POINT_PROCESS_HDRS + \
            STATE_SPACE_HDRS + \
            TIMESERIES_HDRS

cc_library(
    name = "boom",
    srcs = BOOM_SRCS,
    hdrs = BOOM_HDRS,
    copts = [
        "-I.",
        "-I/usr/local/include",
        "-std=c++11",
    ],
    linkopts = [
        "-L/usr/local/lib",
        "-L/usr/lib",
        "-llapack",
        "-lblas",
        "-lm",
        "-lboost_thread-mt",
    ],
    visibility = ["//visibility:public"],
)
