COPTS = [
    "-I/usr/local/include",
]

cc_test(
    name = "ArmaModel_test",
    srcs = ["ArmaModel_test.cc"],
    copts = COPTS,
    deps = [
        "//:boom",
        "@gtest//:main",
    ],
)
