COPTS = ["--std=c++17"]
COMMON_DEPS = [
        "//:boom",
        "//:boom_test_utils",
        "@gtest//:gtest_main",
    ]

cc_test(
    name = "bessel_test",
    srcs = ["bessel_test.cc"],
    copts = COPTS,
    deps = COMMON_DEPS,
    size = "small",
 )

cc_test(
    name = "chisq_test",
    srcs = ["chisq_test.cc"],
    copts = COPTS,
    deps = [
        "//:boom",
        "//:boom_test_utils",
        "@gtest//:gtest_main",
    ],
    size = "small",
)

cc_test(
    name = "extreme_value_test",
    srcs = ["extreme_value_test.cc"],
    copts = COPTS,
    deps = [
        "//:boom",
        "//:boom_test_utils",
        "@gtest//:gtest_main",
    ],
    size = "small",
)

cc_test(
    name = "gig_test",
    srcs = ["gig_test.cc"],
    copts = COPTS,
    deps = COMMON_DEPS,
    size = "small",
)

cc_test(
    name = "mvn_test",
    srcs = ["mvn_test.cc"],
    copts = COPTS,
    deps = [
        "//:boom",
        "//:boom_test_utils",
        "@gtest//:gtest_main",
    ],
    size = "small",
)

cc_test(
    name = "matrix_normal_test",
    srcs = ["matrix_normal_test.cc"],
    copts = COPTS,
    deps = [
        "//:boom",
        "//:boom_test_utils",
        "@gtest//:gtest_main",
    ],
    size = "small",
)

cc_test(
    name = "trun_gamma_test",
    srcs = ["trun_gamma_test.cc"],
    copts = COPTS,
    deps = [
        "//:boom",
        "//:boom_test_utils",
        "@gtest//:gtest_main",
    ],
    size = "small",
)
