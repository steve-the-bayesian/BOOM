COPTS = [
    "-I/usr/local/include",
]

cc_test(
    name = "SpdMatrix_test",
    srcs = ["SpdMatrix_test.cc"],
    copts = COPTS,
    deps = [
        "//:boom",
        "@gtest//:main",
    ],
)

cc_test(
    name = "tail_test",
    srcs = ["tail_test.cc"],
    copts = COPTS,
    deps = [
        "//:boom",
        "@gtest//:main",
    ],
)
