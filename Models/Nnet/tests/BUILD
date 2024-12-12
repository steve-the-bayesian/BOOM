COPTS = [
    "-Wno-sign-compare",
]
COMMON_DEPS = [
        "//:boom",
        "//:boom_test_utils",
        "@googletest//:gtest_main",
    ]

cc_test(
    name = "hidden_layer_imputer_test",
    srcs = ["hidden_layer_imputer_test.cc"],
    copts = COPTS,
    size = "small",
    deps = COMMON_DEPS,
)

cc_test(
    name = "nnet_test",
    srcs = ["nnet_test.cc"],
    copts = COPTS,
    deps = COMMON_DEPS, 
    size = "small",
)
