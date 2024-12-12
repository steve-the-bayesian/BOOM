COPTS = ["-Wno-sign-compare"]

DEPS = [
    "//:boom",
    "//:boom_test_utils",
    "@googletest//:gtest",
    "@googletest//:gtest_main",
]

filegroup(
    name = "test_data",
    srcs = [
        "Car89.csv",
        "CarsClean.csv",
        "autopref.txt",
    ],
)

cc_test(
    name = "ascii_distribution_compare_test",
    size = "small",
    srcs = ["ascii_distribution_compare_test.cc"],
    copts = COPTS,
    deps = DEPS,
)

cc_test(
    name = "data_table_test",
    size = "small",
    srcs = ["data_table_test.cc"],
    copts = COPTS,
    data = [":test_data"],
    deps = DEPS,
)

cc_test(
    name = "ecdf_test",
    size = "small",
    srcs = ["ecdf_test.cc"],
    copts = COPTS,
    deps = DEPS,
)

cc_test(
    name = "empirical_density_test",
    size = "small",
    srcs = ["empirical_density_test.cc"],
    copts = COPTS,
    deps = DEPS,
)

cc_test(
    name = "encoder_test",
    srcs = ["encoder_test.cc"],
    copts = COPTS,
    deps = DEPS,
)

cc_test(
    name = "logit_test",
    size = "small",
    srcs = ["logit_test.cc"],
    copts = COPTS,
    deps = DEPS,
)
cc_test(
    name = "mixed_multivariate_data_test",
    size = "small",
    srcs = ["mixed_multivariate_data_test.cc"],
    copts = COPTS,
    data = [":test_data"],
    deps = DEPS,
)

cc_test(
    name = "moments_test",
    size = "small",
    srcs = ["moments_test.cc"],
    copts = COPTS,
    deps = DEPS,
)

cc_test(
    name = "optimal_arm_probabilities_test",
    size = "small",
    srcs = ["optimal_arm_probabilities_test.cc"],
    copts = COPTS,
    deps = DEPS,
)

cc_test(
    name = "quantile_test",
    size = "small",
    srcs = ["quantile_test.cc"],
    copts = COPTS,
    deps = DEPS,
)

cc_test(
    name = "resampler_test",
    size = "small",
    srcs = ["resampler_test.cc"],
    copts = COPTS,
    deps = DEPS,
)

cc_test(
    name = "spline_test",
    size = "small",
    srcs = ["spline_test.cc"],
    copts = COPTS,
    deps = DEPS,
)

cc_test(
    name = "summary_test",
    size = "small",
    srcs = ["summary_test.cc"],
    copts = COPTS,
    deps = DEPS,
)
