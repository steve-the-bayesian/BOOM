COPTS = [
    "-Wno-sign-compare",
]

# All the tests in this file depend on the same set of deps.
COMMON_DEPS = [
    "//:boom",
    "//:boom_test_utils",
    "@googletest//:gtest_main",
]

cc_test(
    name = "gp_test",
    srcs = ["gp_test.cc"],
    copts = COPTS,
    deps = COMMON_DEPS,
)

cc_test(
    name = "hierarchical_gp_test",
    srcs = ["hierarchical_gp_test.cc"],
    copts = COPTS,
    deps = COMMON_DEPS,
)

cc_test(
    name = "kernel_test",
    srcs = ["kernel_test.cc"],
    copts = COPTS,
    deps = COMMON_DEPS,
)
