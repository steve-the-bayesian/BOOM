COPTS = []

COMMON_DEPS = [
    "//:boom",
    "//:boom_test_utils",
    "@gtest//:gtest_main",
]

cc_test(
    name = "lap_test",
    size = "small",
    srcs = ["lap_test.cc"],
    copts = COPTS,
    deps = COMMON_DEPS,
)
