COPTS = ["-Wno-sign-compare"]

DEPS = [
    "//:boom",
    "//:boom_test_utils",
    "@googletest//:gtest_main",
]

cc_test(
    name = "fft_test",
    size = "small",
    srcs = ["fft_test.cc"],
    copts = COPTS,
    deps = DEPS,
)

cc_test(
    name = "permutation_test",
    size = "small",
    srcs = ["permutation_test.cc"],
    copts = COPTS,
    deps = DEPS,
)
