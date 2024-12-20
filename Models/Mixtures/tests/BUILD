COPTS = [
    "-Wno-sign-compare",
]

COMMON_DEPS = [
    "//:boom",
    "//:boom_test_utils",
    "@googletest//:gtest_main",
]

cc_test(
    name = "beta_binomial_mixture_test",
    size = "small",
    srcs = ["beta_binomial_mixture_test.cc"],
    copts = COPTS,
    deps = COMMON_DEPS,
)

cc_test(
    name = "dp_mvn_collapsed_gibbs_test",
    size = "small",
    srcs = ["dp_mvn_collapsed_gibbs_test.cc"],
    copts = COPTS,
    deps = COMMON_DEPS,
)

cc_test(
    name = "identify_permutation_test",
    size = "small",
    srcs = ["identify_permutation_test.cc"],
    copts = COPTS,
    deps = COMMON_DEPS,
)
