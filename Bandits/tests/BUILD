load("@rules_cc//cc:defs.bzl", "cc_library", "cc_test")

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

cc_test(
    name = "linear_bandit_encoder_test",
    size = "small",
    srcs = ["linear_bandit_encoder_test.cc"],
    copts = COPTS,
    deps = COMMON_DEPS,
)

cc_test(
    name = "logit_bandit_test",
    size = "small",
    srcs = ["logit_bandit_test.cc"],
    copts = COPTS,
    deps = COMMON_DEPS,
)
