COPTS = [
    "-I/usr/local/include",
    "-Wno-sign-compare",
]

cc_test(
    name = "gaussian_test",
    srcs = ["gaussian_test.cc"],
    copts = COPTS,
    deps = [
        "//:boom",
        "//:boom_test_utils",
        "@gtest//:main",
    ],
)
