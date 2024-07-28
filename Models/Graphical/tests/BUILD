COPTS = [
    "-Iexternal/gtest/googletest-release-1.8.0/googletest/include",
    "-Wno-sign-compare",
]

# All the tests in this file depend on the same set of deps.
COMMON_DEPS = [
    "//:boom",
    "//:boom_test_utils",
    "@gtest//:gtest_main",
]

cc_test(
    name = "node_test",
    size = "small",
    srcs = ["node_test.cc"],
    copts = COPTS,
    deps = COMMON_DEPS,
)

cc_test(
    name = "clique_test",
    size = "small",
    srcs = ["clique_test.cc"],
    copts = COPTS,
    deps = COMMON_DEPS,
)

cc_test(
    name = "junction_tree_test",
    size = "small",
    srcs = ["junction_tree_test.cc"],
    copts = COPTS,
    deps = COMMON_DEPS,
)

cc_test(
    name = "multinomial_node_test",
    size = "small",
    srcs = ["multinomial_node_test.cc"],
    copts = COPTS,
    deps = COMMON_DEPS,
)
