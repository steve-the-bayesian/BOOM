COPTS = [
    "-I/usr/local/include",
]

cc_test(
    name = "polynomial_test",
    srcs = ["polynomial_test.cc"],
    copts = COPTS,
    deps = [
        "//:boom",
        "@gtest//:main",
    ],
)
