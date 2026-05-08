load("@rules_cc//cc:defs.bzl", "cc_test") 
load("@rules_cc//cc:defs.bzl", "cc_library") 

COPTS = [
    "-Wno-sign-compare",
]

cc_test(
    name = "param_policy2_test",
    srcs = ["param_policy2_test.cc"],
    copts = COPTS,
    deps = [
        "//:boom",
        "//:boom_test_utils",
        "@googletest//:gtest_main",
    ],
    size = "small",
)
