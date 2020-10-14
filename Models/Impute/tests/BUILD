COPTS = [
    "-Iexternal/gtest/googletest-release-1.8.0/googletest/include",
    "-Wno-sign-compare",
    "-fsanitize=thread"
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
    linkopts=["-fsanitize=thread"],
    includes = ["@gtest"],
    deps = COMMON_DEPS,
)

cc_test(
    name = "mixed_data_imputer_test",
    srcs = ["mixed_data_imputer_test.cc"],
    copts = COPTS,
    linkopts = ["-fsanitize=thread"],
    includes = ["@gtest"],
    deps = COMMON_DEPS,
)

cc_test(
    name = "mixed_data_imputer_with_error_correction_test",
    srcs = ["mixed_data_imputer_with_error_correction_test.cc"],
    copts = COPTS,
    linkopts=["-fsanitize=thread"],
    includes = ["@gtest"],
    deps = COMMON_DEPS,
)
