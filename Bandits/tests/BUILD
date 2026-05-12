load("@rules_cc//cc:defs.bzl", "cc_test") 
load("@rules_cc//cc:defs.bzl", "cc_library") 

COPTS = ["-Wno-sign-compare"]

COMMON_DEPS = [
    "//:boom",
    "//:boom_test_utils",
    "@googletest//:gtest_main",
]

cc_test(
    name = "binomial_bandit_test",
    size = "small",
    srcs = ["binomial_bandit_test.cc"],
    copts = COPTS,
    deps = COMMON_DEPS,
)

