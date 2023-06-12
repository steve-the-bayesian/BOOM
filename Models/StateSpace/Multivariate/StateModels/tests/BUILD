COPTS = [
    "-Iexternal/gtest/googletest-release-1.8.0/googletest/include",
    "-Wno-sign-compare",
]

cc_test(
    name = "shared_local_level_test",
    size = "small",
    srcs = ["shared_local_level_test.cc"],
    copts = COPTS,
    deps = [
        "//:boom",
        "//:boom_test_utils",
        "//Models/StateSpace/tests:test_utils",
        "@gtest//:gtest_main",
    ],
)

cc_test(
    name = "scalar_state_model_adapter_test",
    size = "small",
    srcs = ["scalar_state_model_adapter_test.cc"],
    copts = COPTS,
    deps = [
        "//:boom",
        "//:boom_test_utils",
        "//Models/StateSpace/tests:test_utils",
        "@gtest//:gtest_main",
    ],
)
