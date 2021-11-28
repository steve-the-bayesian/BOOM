COPTS = [
    "-Iexternal/gtest/googletest-release-1.8.0/googletest/include",
    "-Wno-sign-compare",
]

SANITIZERS = [
    #        "-fsanitize=address",
    #        "-fsanitize=undefined",
    #        "-fsanitize=vptr",
    #        "-fsanitize=thread"
]

cc_test(
    name = "multivariate_state_space_regression_test",
    size = "small",
    srcs = ["multivariate_state_space_regression_test.cc"],
    copts = COPTS,
    deps = [
        "//:boom",
        "//:boom_test_utils",
        "@gtest//:gtest_main",
    ],
)
