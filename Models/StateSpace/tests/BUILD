COPTS = [
    "-I/usr/local/include",
    "-Iexternal/gtest/googletest-release-1.8.0/googletest/include",
    "-Wno-sign-compare",
]

filegroup(
    name = "testdata",
    data = ["airpassengers.txt"],
)

cc_library(
    name = "test_utils",
    testonly = True,
    srcs = [
        "DynamicInterceptTestFramework.cpp",
        "StateSpaceTestFramework.cpp",
    ],
    hdrs = [
        "DynamicInterceptTestFramework.hpp",
        "StateSpaceTestFramework.hpp",
        "TestFrameworkBase.hpp",
        "state_space_test_utils.hpp",
    ],
    copts = ["-I/usr/local/include"],
    visibility = [
        "//Models/StateSpace/StateModels/tests:__pkg__",
        "//Models/StateSpace/tests:__pkg__",
    ],
    deps = [
        "//:boom",
        "//:boom_test_utils",
        "//Models/StateSpace/StateModels/test_utils",
        "@gtest",
    ],
)

cc_test(
    name = "dynamic_intercept_regression_test",
    srcs = ["dynamic_intercept_regression_test.cc"],
    copts = COPTS,
    data = [":testdata"],
    deps = [
        ":test_utils",
        "//:boom",
        "//:boom_test_utils",
        "@gtest//:gtest_main",
    ],
)

genrule(
    name = "dynamic_intercept_regression_test_dsym",
    testonly = True,
    srcs = [":dynamic_intercept_regression_test"],
    outs = ["dynamic_intercept_regression_test.dSYM"],
    cmd = "dsymutil $(location :dynamic_intercept_regression_test) -o $@",
    output_to_bindir = True,
)

cc_test(
    name = "multivariate_state_space_regression_test",
    srcs = ["multivariate_state_space_regression_test.cc"],
    copts = COPTS,
    deps = [
        "//:boom",
        "//:boom_test_utils",
        "@gtest//:gtest_main",
    ],
)

cc_test(
    name = "state_space_gaussian_model_test",
    srcs = ["state_space_gaussian_model_test.cc"],
    copts = COPTS,
    data = [":testdata"],
    deps = [
        "//:boom",
        "//:boom_test_utils",
        "@gtest//:gtest_main",
    ],
)

cc_test(
    name = "state_space_regression_model_test",
    srcs = ["state_space_regression_model_test.cc"],
    copts = COPTS,
    data = [":testdata"],
    deps = [
        "//:boom",
        "//:boom_test_utils",
        "@gtest//:gtest_main",
    ],
)
