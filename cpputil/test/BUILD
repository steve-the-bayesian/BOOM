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

cc_test(
    name = "constants_test",
    srcs = ["constants_test.cc"],
    copts = COPTS,
    deps = [
        "//:boom",
        "//:boom_test_utils",
        "@gtest//:main",
    ],
)
