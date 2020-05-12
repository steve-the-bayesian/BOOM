COPTS = [
    "-Iexternal/gtest/googletest-release-1.8.0/googletest/include",
    "-Wno-sign-compare",
]

COMMON_DEPS = [
    "//:boom",
    "//:boom_test_utils",
    "@gtest//:gtest_main",
]


cc_test(
    name = "gaussian_test",
    srcs = ["gaussian_test.cc"],
    copts = COPTS,
    includes = ["@gtest"],
    deps = COMMON_DEPS,
)

cc_test(
    name = "exponential_increment_model_test",
    srcs = ["exponential_increment_model_test.cc"],
    copts = COPTS,
    deps = COMMON_DEPS,
)

cc_test(
    name = "independent_mvn_test",
    srcs = ["independent_mvn_test.cc"],
    copts = COPTS,
    deps = COMMON_DEPS,
)

cc_test(
    name = "matrix_normal_test",
    srcs = ["matrix_normal_test.cc"],
    copts = COPTS,
    deps = COMMON_DEPS,
)

cc_test(
    name = "multinomial_test",
    srcs = ["multinomial_test.cc"],
    copts = COPTS,
    deps = COMMON_DEPS,
)

cc_test(
    name = "spddata_test",
    srcs = ["spddata_test.cc"],
    copts = COPTS,
    deps = COMMON_DEPS,
)
