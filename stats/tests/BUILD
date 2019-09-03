COPTS = [
    "-I/usr/local/include",
]

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
)

cc_test(
    name = "ecdf_test",
    srcs = ["ecdf_test.cc"],
    copts = COPTS,
    deps = DEPS,
)

cc_test(
    name = "empirical_density_test",
    srcs = ["empirical_density_test.cc"],
    copts = COPTS,
    deps = DEPS,
)

cc_test(
    name = "logit_test",
    srcs = ["logit_test.cc"],
    copts = COPTS,
    deps = DEPS,
)

cc_test(
    name = "resampler_test",
    srcs = ["resampler_test.cc"],
    copts = COPTS,
    deps = DEPS,
)

cc_test(
    name = "spline_test",
    srcs = ["spline_test.cc"],
    copts = COPTS,
    deps = DEPS,
)
