COPTS = [
    "-I/usr/local/include",
]

cc_test(
    name = "logit_test",
    srcs = ["logit_test.cc"],
    copts = COPTS,
    deps = [
        "//:boom",
        "@gtest//:main",
    ],
)
