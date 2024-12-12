COPTS = [
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
        "@googletest//:gtest_main",
    ],
    size = "small",
)
