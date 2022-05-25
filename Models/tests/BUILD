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
    size = "small",
    srcs = ["gaussian_test.cc"],
    copts = COPTS,
    includes = ["@gtest"],
    deps = COMMON_DEPS,
)

cc_test(
    name = "exponential_increment_model_test",
    size = "small",
    srcs = ["exponential_increment_model_test.cc"],
    copts = COPTS,
    deps = COMMON_DEPS,
)

cc_test(
    name = "independent_mvn_test",
    size = "small",
    srcs = ["independent_mvn_test.cc"],
    copts = COPTS,
    deps = COMMON_DEPS,
)

cc_test(
    name = "markov_test",
    size = "small",
    srcs = ["markov_test.cc"],
    copts = COPTS,
    deps = COMMON_DEPS,
)

cc_test(
    name = "matrix_normal_test",
    size = "small",
    srcs = ["matrix_normal_test.cc"],
    copts = COPTS,
    deps = COMMON_DEPS,
)

cc_test(
    name = "multinomial_test",
    size = "small",
    srcs = ["multinomial_test.cc"],
    copts = COPTS,
    deps = COMMON_DEPS,
)

cc_test(
    name = "spddata_test",
    size = "small",
    srcs = ["spddata_test.cc"],
    copts = COPTS,
    deps = COMMON_DEPS,
)

cc_test(
    name = "student_test",
    size = "small",
    srcs = ["student_test.cc"],
    copts = COPTS,
    deps = COMMON_DEPS,
)
