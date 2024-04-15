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
    name = "multinomial_factor_model_test",
    size = "small",
    srcs = ["multinomial_factor_model_test.cc"],
    copts = COPTS,
    deps = COMMON_DEPS,
)

cc_test(
    name = "poisson_factor_model_test",
    size = "small",
    srcs = ["poisson_factor_model_test.cc"],
    copts = COPTS,
    deps = COMMON_DEPS,
)
