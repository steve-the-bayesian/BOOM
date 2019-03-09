COPTS = [
    "-I/usr/local/include",
    "-Wno-sign-compare",
    #    "-fsanitize=memory",
]

cc_test(
    name = "ArmaModel_test",
    srcs = ["ArmaModel_test.cc"],
    copts = COPTS,
    deps = [
        "//:boom",
        "//:boom_test_utils",
        "@gtest//:gtest_main",
    ],
)
