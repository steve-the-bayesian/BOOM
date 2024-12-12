COPTS = []

COMMON_DEPS = [
    "//:boom",
    "//:boom_test_utils",
    "@googletest//:gtest_main",
]

cc_test(
    name = "class_assigner_test",
    size = "small",
    srcs = ["class_assigner_test.cc"],
    copts = COPTS,
    deps = COMMON_DEPS,
)

cc_test(
    name = "lap_test",
    size = "small",
    srcs = ["lap_test.cc"],
    copts = COPTS,
    deps = COMMON_DEPS,
)

cc_test(
    name = "mdp_test",
    size = "small",
    srcs = ["mdp_test.cc"],
    copts = COPTS,
    deps = COMMON_DEPS,
)

cc_test(
    name = "simulated_annealing_test",
    size = "small",
    srcs = ["simulated_annealing_test.cc"],
    copts = COPTS,
    deps = COMMON_DEPS,
)
