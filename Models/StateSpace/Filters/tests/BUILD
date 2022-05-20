COPTS = [
    "-Iexternal/gtest/googletest-release-1.8.0/googletest/include",
    "-Wno-sign-compare",
]

cc_test(
    name = "kalman_filter_test",
    size = "small",
    srcs = ["kalman_filter_test.cc"],
    copts = COPTS,
    deps = [
        "//:boom",
        "//:boom_test_utils",
        "@gtest//:gtest_main",
    ],
)

cc_test(
    name = "conditionally_independent_kalman_filter_test",
    size = "small",
    srcs = ["conditionally_independent_kalman_filter_test.cc"],
    copts = COPTS,
    deps = [
        "//:boom",
        "//:boom_test_utils",
        "//Models/StateSpace/Multivariate/tests:mv_statespace_test_framework",
        "@gtest//:gtest_main",
    ],
)

cc_test(
    name = "multivariate_kalman_filter_test",
    size = "small",
    srcs = ["multivariate_kalman_filter_test.cc"],
    copts = COPTS,
    deps = [
        "//:boom",
        "//:boom_test_utils",
        "@gtest//:gtest_main",
    ],
)

cc_test(
    name = "sparse_matrix_test",
    size = "small",
    srcs = ["sparse_matrix_test.cc"],
    copts = COPTS,
    deps = [
        "//:boom",
        "//:boom_test_utils",
        "@gtest//:gtest_main",
    ],
)
