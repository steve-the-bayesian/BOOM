COPTS = [
    "-Wno-sign-compare",
]

COMMON_DEPS = [
        "//:boom",
        "//:boom_test_utils",
        "//Models/StateSpace/tests:test_utils",
        "@googletest//:gtest_main",
    ]

STATE_MODEL_COMMON_DEPS = COMMON_DEPS + [
			"//Models/StateSpace/StateModels/test_utils"
			]

cc_test(
    name = "ar_state_model_test",
    size = "small",
    srcs = ["ar_state_model_test.cc"],
    copts = COPTS,
    deps = COMMON_DEPS,
)

cc_test(
    name = "dynamic_regression_test",
    size = "small",
    srcs = ["dynamic_regression_test.cc"],
    copts = COPTS,
    deps = COMMON_DEPS,
)

cc_test(
    name = "general_seasonal_llt_test",
    size = "small",
    srcs = ["general_seasonal_llt_test.cc"],
    copts = COPTS,
    deps = COMMON_DEPS,
)

cc_test(
    name = "local_level_test",
    size = "small",
    srcs = ["local_level_test.cc"],
    copts = COPTS,
    deps = COMMON_DEPS,
)

cc_test(
    name = "local_linear_trend_test",
    size = "small",
    srcs = ["local_linear_trend_test.cc"],
    copts = COPTS,
    deps = STATE_MODEL_COMMON_DEPS,
)

cc_test(
    name = "HierarchicalRegressionHolidayStateModel_test",
    size = "small",
    srcs = ["HierarchicalRegressionHolidayStateModel_test.cc"],
    copts = COPTS,
    deps = STATE_MODEL_COMMON_DEPS,
)

cc_test(
    name = "Holiday_test",
    size = "small",
    srcs = ["Holiday_test.cc"],
    copts = COPTS,
    deps = COMMON_DEPS,
)

cc_test(
    name = "HolidayMcmc_test",
    size = "small",
    srcs = ["HolidayMcmc_test.cc"],
    copts = COPTS,
    deps = COMMON_DEPS, 
)

cc_binary(
    name = "HolidayMcmcExample",
    srcs = ["HolidayMcmcExample.cc"],
    copts = COPTS,
    deps = [
        "//:boom",
    ],
)

cc_test(
    name = "RegressionHolidayStateModel_test",
    size = "small",
    srcs = ["RegressionHolidayStateModel_test.cc"],
    copts = COPTS,
    deps = STATE_MODEL_COMMON_DEPS, 
)

cc_test(
    name = "random_walk_holiday_test",
    size = "small",
    srcs = ["random_walk_holiday_test.cc"],
    copts = COPTS,
    deps = STATE_MODEL_COMMON_DEPS, 
)

cc_test(
    name = "seasonal_state_model_test",
    size = "small",
    srcs = ["SeasonalStateModel_test.cc"],
    copts = COPTS,
    # linkopts = [
    #     "-lprofiler",  # to get the profiler on ubuntu I had to sudo apt install libgoogle-perftools-dev
    # ],
    deps = STATE_MODEL_COMMON_DEPS, 
)

cc_test(
    name = "semilocal_linear_trend_test",
    size = "small",
    srcs = ["semilocal_linear_trend_test.cc"],
    copts = COPTS,
    deps = STATE_MODEL_COMMON_DEPS, 
)

cc_test(
    name = "static_intercept_test",
    size = "small",
    srcs = ["static_intercept_test.cc"],
    copts = COPTS,
    deps = STATE_MODEL_COMMON_DEPS, 
)

cc_test(
    name = "student_local_linear_trend_test",
    size = "small",
    srcs = ["student_local_linear_trend_test.cc"],
    copts = COPTS,
    deps = STATE_MODEL_COMMON_DEPS, 
)

cc_test(
    name = "Trig_test",
    size = "small",
    srcs = ["Trig_test.cc"],
    copts = COPTS + ["-g"],
    #     linkopts = ["-lprofiler"],
    deps = STATE_MODEL_COMMON_DEPS,
)
