COPTS = []

COMMON_DEPS = [
    "//:boom",
    "//:boom_test_utils",
    "@googletest//:gtest_main",
]

cc_test(
    name = "multinomial_logit_transform_test",
    size = "small",
    srcs = ["multinomial_logit_transform_test.cc"],
    copts = COPTS,
    deps = COMMON_DEPS,
)

cc_test(
    name = "logit_transform_test",
    size = "small",
    srcs = ["logit_transform_test.cc"],
    copts = COPTS,
    deps = COMMON_DEPS,
)

cc_test(
    name = "sum_multinomial_logit_test",
    size = "small",
    srcs = ["sum_multinomial_logit_test.cc"],
    copts = COPTS,
    deps = COMMON_DEPS,
)
