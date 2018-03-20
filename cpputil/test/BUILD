COPTS = [
    "-I/usr/local/include",
]

cc_test(
    name = "polynomial_test",
    srcs = ["polynomial_test.cc"],
    copts = COPTS,
    deps = [
        "//:boom",
        "//:boom_test_utils",
        "@gtest//:main",
    ],
)
