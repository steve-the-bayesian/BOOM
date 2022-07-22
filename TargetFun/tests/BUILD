COPTS = []

COMMON_DEPS = [
    "//:boom",
    "//:boom_test_utils",
    "@gtest//:gtest_main",
]

cc_test(
    name = "multinomial_logit_transform_test",
    size = "small",
    srcs = ["multinomial_logit_transform_test.cc"],
    copts = COPTS,
    deps = COMMON_DEPS,
)
