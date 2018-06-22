COPTS = [
    "-I/usr/local/include",
    "-Wno-sign-compare",
]

filegroup(
    name = "testdata",
    data = ["airpassengers.txt"],
)

cc_library(
    name = "test_utils",
    #    srcs = ["test_utils.hpp"],
    hdrs = ["state_space_test_utils.hpp"],
    deps = ["//:boom"],
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
        "@gtest//:main",
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
    name = "state_space_gaussian_model_test",
    srcs = ["state_space_gaussian_model_test.cc"],
    copts = COPTS,
    data = [":testdata"],
    deps = [
        "//:boom",
        "//:boom_test_utils",
        "@gtest//:main",
    ],
)
