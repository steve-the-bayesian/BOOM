COPTS = [
    "-I/usr/local/include",
]

cc_test(
    name = "tail_test",
    srcs = ["tail_test.cc"],
    copts = COPTS,
    deps = [
        "//:boom",
        "@gtest//:main",
    ],
)
