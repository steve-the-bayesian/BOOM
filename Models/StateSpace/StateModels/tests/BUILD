COPTS = [
    "-I/usr/local/include",
]

cc_test(
    name = "Holiday_test",
    srcs = ["Holiday_test.cc"],
    copts = COPTS,
    deps = [
        "//:boom",
        "@gtest//:main",
    ],
)
