COPTS = [
    "-I/usr/local/include",
    "-Wno-sign-compare",
]

cc_test(
    name = "binomial_logit_test",
    srcs = ["binomial_logit_test.cc"],
    copts = COPTS,
    deps = [
        "//:boom",
        "//:boom_test_utils",
        "@gtest//:gtest_main",
    ],
)

cc_test(
    name = "binomial_probit_test",
    srcs = ["binomial_probit_test.cc"],
    copts = COPTS,
    deps = [
        "//:boom",
        "//:boom_test_utils",
        "@gtest//:gtest_main",
    ],
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
    name = "multivariate_regression_test",
    srcs = ["multivariate_regression_test.cc"],
    copts = COPTS,
    linkopts = ["-lprofiler"],
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
    linkopts = ["-lprofiler"],
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
    linkopts = ["-lprofiler"],
    deps = [
        "//:boom",
        "//:boom_test_utils",
        "@gtest//:gtest_main",
    ],
)
