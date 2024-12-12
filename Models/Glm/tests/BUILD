COPTS = [
    "-Wno-sign-compare",
]

# All the tests in this file depend on the same set of deps.
COMMON_DEPS = [
    "//:boom",
    "//:boom_test_utils",
    "@googletest//:gtest_main",
]

cc_test(
    name = "big_regression_test",
    size = "large",
    timeout = "eternal",
    srcs = ["big_regression_test.cc"],
    copts = COPTS,
    deps = COMMON_DEPS,
)

cc_test(
    name = "binomial_logit_test",
    size = "small",
    srcs = ["binomial_logit_test.cc"],
    copts = COPTS,
    deps = COMMON_DEPS,
)

cc_test(
    name = "binomial_probit_test",
    size = "small",
    srcs = ["binomial_probit_test.cc"],
    copts = COPTS,
    deps = COMMON_DEPS,
)

cc_test(
    name = "binomial_regression_data_test",
    size = "small",
    srcs = ["binomial_regression_data_test.cc"],
    copts = COPTS,
    deps = COMMON_DEPS,
)

cc_test(
    name = "choice_data_test",
    size = "small",
    srcs = ["choice_data_test.cc"],
    copts = COPTS,
    deps = COMMON_DEPS,
)

cc_test(
    name = "glm_coefs_test",
    size = "small",
    srcs = ["glm_coefs_test.cc"],
    copts = COPTS,
    deps = COMMON_DEPS,
)

cc_test(
    name = "loglinear_model_test",
    size = "small",
    srcs = ["loglinear_model_test.cc"],
    copts = COPTS,
    deps = COMMON_DEPS + [":minn38"],
)

cc_library(
    name = "minn38",
    srcs = ["minn38.cc"],
    copts = COPTS,
    visibility = ["//visibility:private"],
    deps = COMMON_DEPS,
)

filegroup(
    name = "test_data",
    srcs = [
        "Car89.csv",
        "autopref.txt",
        "minn38.csv",
    ],
)

cc_test(
    name = "mlogit_test",
    size = "large",
    srcs = ["mlogit_test.cc"],
    copts = COPTS + ["-lprofiler"],
    data = [":test_data"],
    #     linkopts = ["-lprofiler"],
    deps = COMMON_DEPS,
)

cc_test(
    name = "multivariate_regression_test",
    size = "small",
    srcs = ["multivariate_regression_test.cc"],
    copts = COPTS,
    deps = COMMON_DEPS, 
)

cc_test(
    name = "mvn_given_x_test",
    size = "small",
    srcs = ["mvn_given_x_test.cc"],
    copts = COPTS,
    deps = COMMON_DEPS, 
)

cc_test(
    name = "ordinal_logit_test",
    size = "small",
    srcs = ["ordinal_logit_test.cc"],
    copts = COPTS,
    deps = COMMON_DEPS,
)

cc_test(
    name = "regression_model_test",
    size = "small",
    srcs = ["regression_model_test.cc"],
    copts = COPTS,
    deps = COMMON_DEPS,
)

cc_test(
    name = "regression_spike_slab_test",
    size = "small",
    srcs = ["regression_spike_slab_test.cc"],
    copts = COPTS,
    deps = COMMON_DEPS, 
)

cc_test(
    name = "student_spike_slab_test",
    size = "small",
    srcs = ["student_spike_slab_test.cc"],
    copts = COPTS,
    deps = COMMON_DEPS, 
)

cc_test(
    name = "variable_selection_prior_test",
    size = "small",
    srcs = ["variable_selection_prior_test.cc"],
    copts = COPTS,
    deps = COMMON_DEPS, 
)
