COPTS = ["-Wno-sign-compare"]

DEPS = [
    "//:boom",
    "//:boom_test_utils",
    "@gtest//:gtest_main",
]

cc_test(
    name = "ascii_distribution_compare_test",
    srcs = ["ascii_distribution_compare_test.cc"],
    copts = COPTS,
    deps = DEPS,
    size = "small",
)

cc_test(
    name = "data_table_test",
    srcs = ["data_table_test.cc"],
    copts = COPTS,
    deps = DEPS,
    size = "small",
)

cc_test(
    name = "ecdf_test",
    srcs = ["ecdf_test.cc"],
    copts = COPTS,
    deps = DEPS,
    size = "small",
)

cc_test(
    name = "empirical_density_test",
    srcs = ["empirical_density_test.cc"],
    copts = COPTS,
    deps = DEPS,
    size = "small",
)

cc_test(
    name = "encoder_test",
    srcs = ["encoder_test.cc"],
    copts = COPTS,
    deps = DEPS,
)

cc_test(
    name = "logit_test",
    srcs = ["logit_test.cc"],
    copts = COPTS,
    deps = DEPS,
    size = "small",
)

cc_test(
    name = "resampler_test",
    srcs = ["resampler_test.cc"],
    copts = COPTS,
    deps = DEPS,
    size = "small",
)

cc_test(
    name = "spline_test",
    srcs = ["spline_test.cc"],
    copts = COPTS,
    deps = DEPS,
    size = "small",
)

cc_test(
    name = "summary_test",
    srcs = ["summary_test.cc"],
    copts = COPTS,
    deps = DEPS,
    size = "small",
)
