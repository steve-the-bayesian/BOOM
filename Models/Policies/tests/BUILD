COPTS = [
    "-Wno-sign-compare",
]

cc_test(
    name = "param_policy2_test",
    srcs = ["param_policy2_test.cc"],
    copts = COPTS,
    deps = [
        "//:boom",
        "//:boom_test_utils",
        "@gtest//:gtest_main",
    ],
)
