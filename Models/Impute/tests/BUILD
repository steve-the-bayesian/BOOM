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
    name = "copula_imputer_test",
    srcs = ["copula_imputer_test.cc"],
    copts = COPTS,
    includes = ["@gtest"],
    deps = COMMON_DEPS,
)
