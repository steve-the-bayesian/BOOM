COPTS = [
    "-Iexternal/gtest/googletest-release-1.8.0/googletest/include",
    "-Wno-sign-compare",
]

# All the tests in this file depend on the same set of deps.
COMMON_DEPS = [
        "//:boom",
        "//:boom_test_utils",
        "@gtest//:gtest_main",
]


cc_test(
    name = "binomial_logit_test",
    srcs = ["binomial_logit_test.cc"],
    copts = COPTS,
    deps = COMMON_DEPS,
)

cc_test(
    name = "binomial_probit_test",
    srcs = ["binomial_probit_test.cc"],
    copts = COPTS,
    deps = COMMON_DEPS,
)

cc_test(
    name = "binomial_regression_data_test",
    srcs = ["binomial_regression_data_test.cc"],
    copts = COPTS,
    deps = [
        "//:boom",
        "//:boom_test_utils",
        "@gtest//:gtest_main",
    ],
)

cc_test(
    name = "encoder_test",
    srcs = ["encoder_test.cc"],
    copts = COPTS,
    deps = [
        "//:boom",
        "//:boom_test_utils",
        "@gtest//:gtest_main",
    ],
)

cc_test(
    name = "glm_coefs_test",
    srcs = ["glm_coefs_test.cc"],
    copts = COPTS,
    deps = [
        "//:boom",
        "//:boom_test_utils",
        "@gtest//:gtest_main",
    ],
)

cc_test(
    name = "loglinear_model_test",
    srcs = ["loglinear_model_test.cc"],
    copts = COPTS,
    deps = COMMON_DEPS + [":minn38"],
)

cc_library(
    name = "minn38",
    srcs = ["minn38.cc"],
    copts = COPTS,
    deps = COMMON_DEPS,
    visibility=["//visibility:private"],
)

cc_test(
    name = "multivariate_regression_test",
    srcs = ["multivariate_regression_test.cc"],
    copts = COPTS,
    deps = [
        "//:boom",
        "//:boom_test_utils",
        "@gtest//:gtest_main",
    ],
)

cc_test(
    name = "mvn_given_x_test",
    srcs = ["mvn_given_x_test.cc"],
    copts = COPTS,
    deps = [
        "//:boom",
        "//:boom_test_utils",
        "@gtest//:gtest_main",
    ],
)

cc_test(
    name = "ordinal_logit_test",
    srcs = ["ordinal_logit_test.cc"],
    copts = COPTS,
    deps = [
        "//:boom",
        "//:boom_test_utils",
        "@gtest//:gtest_main",
    ],
)

cc_test(
    name = "regression_model_test",
    srcs = ["regression_model_test.cc"],
    copts = COPTS,
    deps = [
        "//:boom",
        "//:boom_test_utils",
        "@gtest//:gtest_main",
    ],
)

cc_test(
    name = "regression_spike_slab_test",
    srcs = ["regression_spike_slab_test.cc"],
    copts = COPTS,
    deps = [
        "//:boom",
        "//:boom_test_utils",
        "@gtest//:gtest_main",
    ],
)

cc_test(
    name = "student_spike_slab_test",
    srcs = ["student_spike_slab_test.cc"],
    copts = COPTS,
    deps = [
        "//:boom",
        "//:boom_test_utils",
        "@gtest//:gtest_main",
    ],
)

cc_test(
    name = "variable_selection_prior_test",
    srcs = ["variable_selection_prior_test.cc"],
    copts = COPTS,
    deps = [
        "//:boom",
        "//:boom_test_utils",
        "@gtest//:gtest_main",
    ],
)
