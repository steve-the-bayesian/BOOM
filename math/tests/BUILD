COPTS = ["-Wno-sign-compare"]

DEPS = [
    "//:boom",
    "//:boom_test_utils",
    "@gtest//:gtest_main",
]

cc_test(
    name = "fft_test",
    size = "small",
    srcs = ["fft_test.cc"],
    copts = COPTS,
    deps = DEPS,
)
