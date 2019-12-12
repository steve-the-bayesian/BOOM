COPTS = [
    "-Wno-sign-compare",
]

cc_test(
    name = "hidden_layer_imputer_test",
    srcs = ["hidden_layer_imputer_test.cc"],
    copts = COPTS,
    deps = [
        "//:boom",
        "//:boom_test_utils",
        "@gtest//:gtest_main",
    ],
)

cc_test(
    name = "nnet_test",
    srcs = ["nnet_test.cc"],
    copts = COPTS,
    deps = [
        "//:boom",
        "//:boom_test_utils",
        "@gtest//:gtest_main",
    ],
)
