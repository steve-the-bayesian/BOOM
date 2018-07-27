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

cc_test(
    name = "independent_mvn_test",
    srcs = ["independent_mvn_test.cc"],
    copts = COPTS,
    deps = [
        "//:boom",
        "//:boom_test_utils",
        "@gtest//:main",
    ],
)
