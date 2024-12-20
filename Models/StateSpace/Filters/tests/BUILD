COPTS = [
    "-Wno-sign-compare",
]

COMMON_DEPS = [
        "//:boom",
        "//:boom_test_utils",
        "@googletest//:gtest_main",
]

MVSS_COMMON_DEPS = COMMON_DEPS + [
    "//Models/StateSpace/Multivariate/tests:mv_statespace_test_framework"
]


cc_test(
    name = "kalman_filter_test",
    size = "small",
    srcs = ["kalman_filter_test.cc"],
    copts = COPTS,
    deps = COMMON_DEPS, 
)

cc_test(
    name = "conditionally_independent_kalman_filter_test",
    size = "small",
    srcs = ["conditionally_independent_kalman_filter_test.cc"],
    copts = COPTS,
    deps = MVSS_COMMON_DEPS, 
)

cc_test(
    name = "multivariate_kalman_filter_test",
    size = "small",
    srcs = ["multivariate_kalman_filter_test.cc"],
    copts = COPTS,
    deps = COMMON_DEPS, 
)

cc_test(
    name = "sparse_matrix_test",
    size = "small",
    srcs = ["sparse_matrix_test.cc"],
    copts = COPTS,
    deps = COMMON_DEPS, 
)

cc_test(
    name = "sparse_binomial_inverse_test",
    size = "small",
    srcs = ["sparse_binomial_inverse_test.cc"],
    copts = COPTS,
    deps = COMMON_DEPS, 
)
