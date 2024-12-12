COPTS = [
    "-Wno-sign-compare",
]

COMMON_DEPS = [
    "//:boom",
    "//:boom_test_utils",
    "@googletest//:gtest_main",
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
