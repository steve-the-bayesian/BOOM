COPTS = [
    "-I/usr/local/include",
    "-Wno-sign-compare",
]

cc_test(
    name = "ar_state_model_test",
    srcs = ["ar_state_model_test.cc"],
    copts = COPTS,
    deps = [
        "//:boom",
        "//:boom_test_utils",
        "//Models/StateSpace/tests:test_utils",
        "@gtest//:main",
    ],
)

cc_test(
    name = "dynamic_regression_test",
    srcs = ["dynamic_regression_test.cc"],
    copts = COPTS,
    deps = [
        "//:boom",
        "//:boom_test_utils",
        "//Models/StateSpace/tests:test_utils",
        "@gtest//:main",
    ],
)

cc_test(
    name = "local_level_test",
    srcs = ["local_level_test.cc"],
    copts = COPTS,
    deps = [
        "//:boom",
        "//:boom_test_utils",
        "//Models/StateSpace/tests:test_utils",
        "@gtest//:main",
    ],
)

cc_test(
    name = "local_linear_trend_test",
    srcs = ["local_linear_trend_test.cc"],
    copts = COPTS,
    deps = [
        "//:boom",
        "//:boom_test_utils",
        "//Models/StateSpace/StateModels/test_utils",
        "//Models/StateSpace/tests:test_utils",
        "@gtest//:main",
    ],
)

cc_test(
    name = "HierarchicalRegressionHolidayStateModel_test",
    srcs = ["HierarchicalRegressionHolidayStateModel_test.cc"],
    copts = COPTS,
    deps = [
        "//:boom",
        "//:boom_test_utils",
        "@gtest//:main",
    ],
)

cc_test(
    name = "Holiday_test",
    srcs = ["Holiday_test.cc"],
    copts = COPTS,
    deps = [
        "//:boom",
        "@gtest//:main",
    ],
)

cc_test(
    name = "HolidayMcmc_test",
    srcs = ["HolidayMcmc_test.cc"],
    copts = COPTS,
    deps = [
        "//:boom",
        "@gtest//:main",
    ],
)

cc_test(
    name = "HolidayMcmcExample",
    srcs = ["HolidayMcmcExample.cc"],
    copts = COPTS,
    deps = [
        "//:boom",
    ],
)

cc_test(
    name = "RegressionHolidayStateModel_test",
    srcs = ["RegressionHolidayStateModel_test.cc"],
    copts = COPTS,
    deps = [
        "//:boom",
        "//:boom_test_utils",
        "@gtest//:main",
    ],
)

cc_test(
    name = "SeasonalStateModel_test",
    srcs = ["SeasonalStateModel_test.cc"],
    copts = COPTS,
    deps = [
        "//:boom",
        "//:boom_test_utils",
        "//Models/StateSpace/StateModels/test_utils",
        "//Models/StateSpace/tests:test_utils",
        "@gtest//:main",
    ],
)

cc_test(
    name = "Trig_test",
    srcs = ["Trig_test.cc"],
    copts = COPTS + ["-g"],
    linkopts = ["-lprofiler"],
    deps = [
        "//:boom",
        "//:boom_test_utils",
        "@gtest//:main",
    ],
)
