COPTS = [
    "-I/usr/local/include",
    "-Wno-sign-compare",
]

cc_test(
    name = "regression_model_test",
    srcs = ["regression_model_test.cc"],
    copts = COPTS,
    deps = [
        "//:boom",
        "//:boom_test_utils",
        "@gtest//:main",
    ],
)

cc_test(
    name = "regression_spike_slab_test",
    srcs = ["regression_spike_slab_test.cc"],
    copts = COPTS,
    linkopts = ["-lprofiler"],
    deps = [
        "//:boom",
        "//:boom_test_utils",
        "@gtest//:main",
    ],
)
