COPTS = [
    "-Wno-sign-compare",
]

COMMON_DEPS = [
    "//:boom",
    "//:boom_test_utils",
    "@googletest//:gtest_main",
]

# The following test fails under -c opt because of threading issues.  I don't understand them.  It does not fail under -c dbg.
cc_test(
    name = "copula_imputer_test",
    size = "small",
    srcs = ["copula_imputer_test.cc"],
    copts = COPTS,
    #    linkopts=["-fsanitize=thread"],
    deps = COMMON_DEPS,
)

cc_test(
    name = "mixed_data_imputer_test",
    size = "small",
    srcs = ["mixed_data_imputer_test.cc"],
    copts = COPTS,  # + ["-fsanitize=thread"],
    #     linkopts = ["-fsanitize=thread"],
    deps = COMMON_DEPS,
)

cc_test(
    name = "mixed_data_imputer_with_error_correction_test",
    size = "small",
    srcs = ["mixed_data_imputer_with_error_correction_test.cc"],
    copts = COPTS,  # + ["-fsanitize=thread"],
    #    linkopts=["-fsanitize=thread"],
    deps = COMMON_DEPS,
)
